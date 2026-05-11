"""
2-Mile Well Uplift & Decline Analysis — Streamlit App
=====================================================
Compare 2-mile wells against 1-mile analogs to quantify uplift,
then build custom type curves that reflect expected 2-mile performance.
"""

import io
import math
import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import streamlit as st

warnings.filterwarnings("ignore")

# ──────────────────────────── page config ─────────────────────────────

st.set_page_config(
    page_title="2-Mile Well Analysis",
    page_icon="🛢️",
    layout="wide",
)

# ──────────────────────────── file paths ──────────────────────────────

XLSX_PATH = "w.xlsx"
SHP_1M_PATH = "1M.shp"
SHP_2M_PATH = "2M.shp"
RTC_XLSX_PATH = "rtc.xlsx"

# ──────────────────────────── constants ───────────────────────────────

TARGET_METRIC_CRS = "EPSG:26913"
WATERFLOOD_BUFFER_M = 200.0
CORRIDOR_HALF_WIDTH_M = 900.0

Q_LIMIT = 2.0
FLAT_MONTHS = 1.44
FLAT_DAYS = int(round(FLAT_MONTHS * 30.4375))
B_FIXED = 0.95

COLORS = {
    "1-Mile": "#1f77b4",
    "2-Mile": "#ff7f0e",
    "incremental": "#2ca02c",
}

PERF_METRICS = {
    "eur_bbl": "EUR (bbl)",
    "ip90_bpd": "IP90 (bbl/d)",
    "sixm_bpd": "6M Cal. Rate (bbl/d)",
    "twelvem_bpd": "12M Cal. Rate (bbl/d)",
}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Arial, sans-serif", size=12),
    margin=dict(l=60, r=30, t=50, b=50),
)

RTC_PALETTE = [
    "#8e44ad", "#e67e22", "#1abc9c", "#c0392b",
    "#2980b9", "#7f8c8d", "#d35400", "#27ae60",
]

# Oil & gas convention: P10 = optimistic (high value), P90 = conservative.
# When a user picks label Pxx we compute data percentile at (100 - xx).
PCT_TO_EFF = {10: 90, 25: 75, 50: 50, 75: 25, 90: 10}

# ──────────────────────────── well-file schema ────────────────────────

REQUIRED_WELL_COLUMNS = [
    "UWI", "Section Name", "Well Type", "Hz Length (m)",
    "Oil + Cond: EUR (Mbbl)", "Oil + Cond: IP 90 Cal. Rate (bbl/d)",
    "Oil + Cond: 6M Cal. Rate (bbl/d)", "Oil + Cond: 12M Cal. Rate (bbl/d)",
    "Objective", "On Prod Date", "On Inj Date", "FOOZ",
]

RENAME_WELL = {
    "UWI": "uwi",
    "Section Name": "section_name",
    "Well Type": "well_type",
    "Hz Length (m)": "hz_length_m",
    "Oil + Cond: EUR (Mbbl)": "eur_mbbl",
    "Oil + Cond: IP 90 Cal. Rate (bbl/d)": "ip90_bpd",
    "Oil + Cond: 6M Cal. Rate (bbl/d)": "sixm_bpd",
    "Oil + Cond: 12M Cal. Rate (bbl/d)": "twelvem_bpd",
    "Objective": "objective",
    "On Prod Date": "on_prod_date",
    "On Inj Date": "on_inj_date",
    "FOOZ": "fooz",
}


# ──────────────────────────── helpers ─────────────────────────────────


def _std_str(s: pd.Series) -> pd.Series:
    """Standardise a string column to upper-case, replacing blanks with NaN."""
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
        .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    )


def _require_cols(df: pd.DataFrame, required: list, label: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"{label} missing columns: {missing}")
        st.stop()


def _trapz(y, x):
    """Trapezoidal integration compatible with NumPy 1.x and 2.x."""
    try:
        return float(np.trapezoid(y, x))
    except AttributeError:
        return float(np.trapz(y, x))


# ──────────────────────────── decline-curve math ──────────────────────


def hyp_rate(t, qi, di, b):
    t = np.asarray(t, dtype=float)
    if b == 0:
        return qi * np.exp(-di * t)
    with np.errstate(divide="ignore", invalid="ignore"):
        return qi / np.power(1.0 + b * di * t, 1.0 / b)


def cum_arps(t, qi, di, b):
    t = np.asarray(t, dtype=float)
    if di <= 0:
        return qi * t
    if b == 0:
        return (qi / di) * (1.0 - np.exp(-di * t))
    if b == 1:
        return (qi / di) * np.log(1.0 + di * t)
    factor = qi / ((1.0 - b) * di)
    power = (b - 1.0) / b
    return factor * (1.0 - np.power(1.0 + b * di * t, power))


def t_to_rate(di, qi, b, q_target):
    if q_target <= 0 or di <= 0:
        return np.inf
    if b == 0:
        return max(0.0, math.log(qi / q_target) / di)
    return max(0.0, ((qi / q_target) ** b - 1.0) / (b * di))


def find_Di_for_eur_post(qi, eur_post, b, q_target, tol=1e-6):
    """Bisection solve for Di that honours a target post-flat EUR."""
    if eur_post <= 0 or qi <= q_target:
        return None

    def _residual(di):
        t_end = t_to_rate(di, qi, b, q_target)
        if np.isinf(t_end):
            return np.inf
        return float(cum_arps(t_end, qi, di, b) - eur_post)

    lo, hi = 1e-12, 1.0
    for _ in range(300):
        f_lo, f_hi = _residual(lo), _residual(hi)
        if np.isinf(f_lo) or np.isinf(f_hi) or f_lo * f_hi > 0:
            hi *= 2.0
            if hi > 1e6:
                return None
        else:
            break
    else:
        return None

    for _ in range(300):
        mid = 0.5 * (lo + hi)
        f_mid = _residual(mid)
        if abs(f_mid) < tol:
            return mid
        if _residual(lo) * f_mid <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def build_curve_from_eur(qi, eur_bbl, b, flat_days=FLAT_DAYS, q_limit=Q_LIMIT):
    """Build a piecewise flat+decline curve that honours a target EUR."""
    eur_flat = qi * flat_days
    eur_post = max(eur_bbl - eur_flat, 0.0)
    di = None
    if eur_post > 0 and qi > q_limit:
        di = find_Di_for_eur_post(qi, eur_post, b, q_limit)
    di = di if di else 0.0

    # flat period
    t_list = list(range(flat_days + 1))
    q_list = [qi] * (flat_days + 1)

    # decline period
    if di > 0:
        t_dec = 0
        while True:
            t_dec += 1
            q = hyp_rate(t_dec, qi, di, b)
            t_list.append(flat_days + t_dec)
            q_list.append(max(q, q_limit))
            if q <= q_limit or t_dec > 60_000:
                break

    t_arr = np.array(t_list, dtype=float)
    q_arr = np.array(q_list, dtype=float)

    # cumulative via trapezoidal integration
    N_arr = np.zeros_like(t_arr)
    for i in range(1, len(t_arr)):
        dt = t_arr[i] - t_arr[i - 1]
        N_arr[i] = N_arr[i - 1] + 0.5 * (q_arr[i - 1] + q_arr[i]) * dt

    return {
        "t": t_arr,
        "q": q_arr,
        "N": N_arr,
        "qi": float(qi),
        "di": float(di),
        "b": float(b),
        "eur_target_bbl": float(eur_bbl),
        "eur_actual_bbl": float(N_arr[-1]) if len(N_arr) > 0 else 0.0,
        "flat_days": int(flat_days),
    }


# ──────────────────────────── data loaders ────────────────────────────


@st.cache_data(show_spinner=False)
def load_well_table():
    xls = pd.ExcelFile(XLSX_PATH)
    raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

    # Normalise column names (strip whitespace so " Hz Length (m)" → "Hz Length (m)")
    raw.columns = [c.strip() for c in raw.columns]

    _require_cols(raw, REQUIRED_WELL_COLUMNS, f"`{XLSX_PATH}` sheet 0")
    out = raw.rename(columns=RENAME_WELL)

    for c in ("uwi", "section_name", "well_type", "objective", "fooz"):
        if c in out.columns:
            out[c] = _std_str(out[c])
    for c in ("on_prod_date", "on_inj_date"):
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    if "eur_mbbl" in out.columns:
        out["eur_bbl"] = pd.to_numeric(out["eur_mbbl"], errors="coerce") * 1000.0
    for c in ("hz_length_m", "ip90_bpd", "sixm_bpd", "twelvem_bpd", "eur_bbl"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    out["vintage_year"] = out["on_prod_date"].dt.year
    out["is_fooz"] = out["fooz"].fillna("").eq("YES")
    out["is_injector"] = out["objective"].fillna("").eq("INJ")
    return out


@st.cache_data(show_spinner=False)
def load_section_ooip():
    xls = pd.ExcelFile(XLSX_PATH)
    raw = pd.read_excel(xls, sheet_name=xls.sheet_names[1])
    _require_cols(raw, ["Section", "OOIP"], f"`{XLSX_PATH}` sheet 1")
    out = raw.copy()
    out["Section"] = _std_str(out["Section"])
    out["OOIP"] = pd.to_numeric(out["OOIP"], errors="coerce")
    return out.rename(columns={"Section": "section_name", "OOIP": "section_ooip"})


@st.cache_resource(show_spinner=False)
def load_geometries():
    g1 = gpd.read_file(SHP_1M_PATH)
    g2 = gpd.read_file(SHP_2M_PATH)
    for g, path in [(g1, SHP_1M_PATH), (g2, SHP_2M_PATH)]:
        if "UWI" not in g.columns:
            st.error(f"`{path}` missing field `UWI`.")
            st.stop()

    g1["lateral_group"] = "1-Mile"
    g2["lateral_group"] = "2-Mile"
    gdf = pd.concat([g1, g2], ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=g1.crs or g2.crs)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    try:
        gdf = gdf.to_crs(TARGET_METRIC_CRS)
    except Exception:
        union_geom = gdf.geometry.union_all()
        c = union_geom.centroid
        utm = int((c.x + 180) // 6) + 1
        epsg = 32600 + utm if c.y >= 0 else 32700 + utm
        gdf = gdf.to_crs(f"EPSG:{epsg}")

    gdf["uwi"] = _std_str(gdf["UWI"])
    gdf = (
        gdf.sort_values("lateral_group", ascending=False)
        .drop_duplicates("uwi", keep="first")
    )
    gdf["midpoint"] = gdf.geometry.apply(
        lambda g: g.interpolate(0.5, normalized=True)
        if g is not None and not g.is_empty
        else None
    )
    return gdf[["uwi", "lateral_group", "geometry", "midpoint"]].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_rtc_curves():
    if not os.path.exists(RTC_XLSX_PATH):
        return []
    try:
        rtc_df = pd.read_excel(RTC_XLSX_PATH)
    except Exception:
        return []
    required = {"Name", "Months", "Qi", "b", "EUR"}
    if not required.issubset(set(rtc_df.columns)):
        return []
    curves = []
    for _, row in rtc_df.iterrows():
        try:
            result = build_curve_from_eur(
                float(row["Qi"]),
                float(row["EUR"]),
                float(row["b"]),
                int(float(row["Months"]) * 30.0),
                Q_LIMIT,
            )
            result["name"] = str(row["Name"])
            curves.append(result)
        except Exception:
            continue
    return curves


# ──────────────────────────── spatial features ────────────────────────


@st.cache_resource(show_spinner=False)
def compute_spatial_features(_geoms_gdf, _well_meta):
    if _geoms_gdf.empty:
        return pd.DataFrame(
            columns=[
                "uwi", "nearest_producer_m", "nearest_injector_m",
                "n_within_400m", "n_within_800m", "waterflood_flag",
            ]
        )
    g = _geoms_gdf.copy().reset_index(drop=True)
    meta = _well_meta[
        ["uwi", "on_prod_date", "on_inj_date", "objective", "is_injector"]
    ].copy()
    g = g.merge(meta, on="uwi", how="left")
    g["is_injector"] = g["is_injector"].fillna(False).astype(bool)

    sindex = g.sindex
    rows = []

    for i, row in g.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            rows.append(dict(
                uwi=row["uwi"],
                nearest_producer_m=np.nan,
                nearest_injector_m=np.nan,
                n_within_400m=0,
                n_within_800m=0,
                waterflood_flag=False,
            ))
            continue

        cand = [j for j in sindex.intersection(geom.buffer(5000).bounds) if j != i]
        nearest_prod, nearest_inj = np.nan, np.nan
        n400, n800 = 0, 0

        for j in cand:
            og = g.iloc[j].geometry
            if og is None or og.is_empty:
                continue
            d = geom.distance(og)
            if d <= 400:
                n400 += 1
            if d <= 800:
                n800 += 1
            if bool(g.iloc[j].get("is_injector", False)):
                if not np.isfinite(nearest_inj) or d < nearest_inj:
                    nearest_inj = d
            else:
                if not np.isfinite(nearest_prod) or d < nearest_prod:
                    nearest_prod = d

        # Waterflood flag — only for 2-mile laterals
        wf = False
        if row.get("lateral_group", "") == "2-Mile":
            buf = geom.buffer(WATERFLOOD_BUFFER_M)
            for j in [jj for jj in sindex.intersection(buf.bounds) if jj != i]:
                o = g.iloc[j]
                if o.geometry is None or o.geometry.is_empty:
                    continue
                if not buf.intersects(o.geometry):
                    continue
                inj_d = o.get("on_inj_date", pd.NaT)
                prod_d = row.get("on_prod_date", pd.NaT)
                if pd.notna(inj_d) and pd.notna(prod_d) and inj_d <= prod_d:
                    wf = True
                    break

        rows.append(dict(
            uwi=row["uwi"],
            nearest_producer_m=float(nearest_prod) if np.isfinite(nearest_prod) else np.nan,
            nearest_injector_m=float(nearest_inj) if np.isfinite(nearest_inj) else np.nan,
            n_within_400m=int(n400),
            n_within_800m=int(n800),
            waterflood_flag=bool(wf),
        ))
    return pd.DataFrame(rows)


# ──────────────────────────── matching logic ──────────────────────────


def corridor_match(target_geom, geoms_gdf, half_width=CORRIDOR_HALF_WIDTH_M):
    if target_geom is None or target_geom.is_empty:
        return []
    corridor = target_geom.buffer(half_width)
    ones = geoms_gdf[geoms_gdf["lateral_group"] == "1-Mile"]
    if ones.empty:
        return []
    inside = ones["midpoint"].apply(
        lambda mp: mp is not None and not mp.is_empty and corridor.contains(mp)
    )
    return ones.loc[inside, "uwi"].tolist()


def range_match(target_row, df_1mile, tolerances, active_features):
    if df_1mile.empty:
        return []
    mask = pd.Series(True, index=df_1mile.index)
    for feat, tol in tolerances.items():
        if feat not in active_features:
            continue
        tv = target_row.get(feat, np.nan)
        if pd.isna(tv) or feat not in df_1mile.columns:
            continue
        mask &= df_1mile[feat].between(tv - tol, tv + tol)
    tp = target_row.get("on_prod_date", pd.NaT)
    if pd.notna(tp) and "on_prod_date" in df_1mile.columns:
        mask &= df_1mile["on_prod_date"].notna() & (df_1mile["on_prod_date"] < tp)
    return df_1mile.loc[mask, "uwi"].tolist()


def build_cohort_map(df_2mile, df_1mile, geoms, analysis_mode,
                     tolerances, active_features, corridor_width):
    mapping = {}
    is_mode_a = analysis_mode.startswith("Mode A")
    for _, row2 in df_2mile.iterrows():
        uwi2 = row2["uwi"]
        if is_mode_a:
            matched = range_match(row2, df_1mile, tolerances, active_features)
        else:
            gr = geoms[geoms["uwi"] == uwi2]
            matched = (
                corridor_match(gr.iloc[0].geometry, geoms, corridor_width)
                if not gr.empty
                else []
            )
        mapping[uwi2] = matched
    return mapping


# ──────────────────────────── stats & incremental ─────────────────────


def empirical_summary(values):
    vals = np.array([v for v in values if np.isfinite(v)], dtype=float)
    n = len(vals)
    if n == 0:
        return {k: np.nan for k in
                ("n", "mean", "std", "min", "max",
                 "p10", "p25", "p50", "p75", "p90")} | {"n": 0}
    return dict(
        n=n,
        mean=float(np.mean(vals)),
        std=float(np.std(vals, ddof=1)) if n > 1 else np.nan,
        min=float(np.min(vals)),
        max=float(np.max(vals)),
        # Oil & gas convention: P10 → data percentile 90, etc.
        p10=float(np.percentile(vals, 90)),
        p25=float(np.percentile(vals, 75)),
        p50=float(np.percentile(vals, 50)),
        p75=float(np.percentile(vals, 25)),
        p90=float(np.percentile(vals, 10)),
    )


def geometric_mean_of_series(series: pd.Series) -> float:
    s = series.dropna()
    s = s[(np.isfinite(s)) & (s > 0)]
    if s.empty:
        return np.nan
    return float(np.exp(np.log(s).mean()))


def compute_incremental(well_row, comparator_df, metric_keys):
    result = {"uwi": well_row["uwi"]}
    for mk in metric_keys:
        val = well_row.get(mk, np.nan)
        if comparator_df.empty or pd.isna(val):
            result[f"{mk}_baseline"] = np.nan
            result[f"{mk}_incremental"] = np.nan
            result[f"{mk}_pct_uplift"] = np.nan
            continue
        bl = float(comparator_df[mk].median())
        incr = float(val) - bl
        pct = (incr / bl * 100) if bl else np.nan
        result[f"{mk}_baseline"] = bl
        result[f"{mk}_incremental"] = incr
        result[f"{mk}_pct_uplift"] = pct
    result["n_comparators"] = len(comparator_df)
    return result


def compute_ratio_per_well(df_2mi, df_1mi, cohort_map):
    rows = []
    for _, row2 in df_2mi.iterrows():
        uwi2 = row2["uwi"]
        comp = df_1mi[df_1mi["uwi"].isin(cohort_map.get(uwi2, []))]
        rec = {"uwi": uwi2}
        for mk, short in [("eur_bbl", "eur"), ("ip90_bpd", "ip90"),
                           ("sixm_bpd", "sixm"), ("twelvem_bpd", "twelvem")]:
            actual = row2[mk]
            baseline = comp[mk].median() if not comp.empty else np.nan
            ratio = (
                actual / baseline
                if np.isfinite(baseline) and baseline > 0
                else np.nan
            )
            rec[f"{short}_2mi_actual"] = actual
            rec[f"{short}_1mi_baseline"] = baseline
            rec[f"{short}_ratio"] = ratio
        rows.append(rec)
    return pd.DataFrame(rows)


def build_incremental_frame(df_2mile, df_1mile, cohort_map, metric_keys):
    records = []
    for _, row2 in df_2mile.iterrows():
        uwi2 = row2["uwi"]
        comp_df = df_1mile[df_1mile["uwi"].isin(cohort_map.get(uwi2, []))]
        rec = compute_incremental(row2, comp_df, metric_keys)
        for col in (
            "section_name", "hz_length_m", "on_prod_date", "vintage_year",
            "eur_bbl", "ip90_bpd", "sixm_bpd", "twelvem_bpd",
            "waterflood_flag", "section_ooip",
        ):
            rec[col] = row2.get(col, np.nan)
        records.append(rec)
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["uwi"])


# ──────────────────────────── assembly ────────────────────────────────


@st.cache_data(show_spinner="Loading well table & geometries…")
def load_and_assemble_wells():
    well_raw = load_well_table()
    sec_ooip = load_section_ooip()
    geoms = load_geometries()

    membership = geoms[["uwi", "lateral_group"]].drop_duplicates()
    well_df = membership.merge(well_raw, on="uwi", how="left")
    well_df = well_df.merge(sec_ooip, on="section_name", how="left")
    well_df = well_df.replace([np.inf, -np.inf], np.nan)
    for c in ("is_injector", "is_fooz"):
        if c not in well_df.columns:
            well_df[c] = False
        well_df[c] = well_df[c].fillna(False).astype(bool)

    meta = well_df[
        ["uwi", "on_prod_date", "on_inj_date", "objective", "is_injector"]
    ].copy()
    spatial = compute_spatial_features(geoms, meta)
    well_df = well_df.merge(spatial, on="uwi", how="left")
    well_df = well_df[~well_df["is_fooz"]].reset_index(drop=True)
    return well_df, geoms


# ──────────────────────────── plotting helpers ────────────────────────


def _finite(data):
    return np.array([v for v in data if np.isfinite(v)], dtype=float)


def _pct_values(vals, percentiles=(10, 25, 50, 75, 90)):
    """Return {label: value} using oil & gas convention."""
    return {p: float(np.percentile(vals, PCT_TO_EFF.get(p, p)))
            for p in percentiles}


def make_density_plot(data_1m, data_2m, title, xaxis_title,
                      percentiles=(10, 25, 50, 75, 90)):
    """KDE density plot for 1M and 2M data with percentile lines."""
    fig = go.Figure()

    for data, label, color in [
        (data_1m, "1-Mile", COLORS["1-Mile"]),
        (data_2m, "2-Mile", COLORS["2-Mile"]),
    ]:
        vals = _finite(data)
        if len(vals) < 3:
            continue

        kde = gaussian_kde(vals, bw_method="scott")
        x_grid = np.linspace(vals.min() * 0.8, vals.max() * 1.2, 300)
        y_grid = kde(x_grid)
        y_grid /= y_grid.max()  # normalise for comparison

        pvs = _pct_values(vals, percentiles)
        ptxt = " | ".join(f"P{p}={v:,.0f}" for p, v in pvs.items())
        n = len(vals)

        fig.add_trace(go.Scatter(
            x=x_grid, y=y_grid, mode="lines",
            name=f"{label} (n={n})",
            line=dict(color=color, width=2.5),
            fill="tozeroy", opacity=0.25,
            hovertemplate=(
                f"<b>{label}</b><br>"
                f"{xaxis_title}: %{{x:,.0f}}<br>"
                f"Density: %{{y:.4f}}<br>"
                f"n={n} | Mean={vals.mean():,.0f}<br>"
                f"{ptxt}<extra></extra>"
            ),
        ))

        for p, pval in pvs.items():
            y_at_p = float(kde(np.array([pval]))[0])
            fig.add_trace(go.Scatter(
                x=[pval, pval], y=[0, y_at_p],
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                showlegend=False,
                hovertemplate=f"{label} P{p} = {pval:,.0f}<extra></extra>",
            ))
            fig.add_annotation(
                x=pval, y=y_at_p, text=f"P{p}", showarrow=False,
                font=dict(size=9, color=color), yshift=10,
            )

    fig.update_layout(**PLOTLY_LAYOUT, height=400, title=title,
                      xaxis_title=xaxis_title, yaxis_title="Density",
                      hovermode="closest")
    fig.update_yaxes(rangemode="tozero")
    return fig


def make_ecdf_plot(data_1m, data_2m, title, xaxis_title,
                   percentiles=(10, 25, 50, 75, 90)):
    """Step-ECDF plot for 1M and 2M with percentile markers."""
    fig = go.Figure()

    for data, label, color in [
        (data_1m, "1-Mile", COLORS["1-Mile"]),
        (data_2m, "2-Mile", COLORS["2-Mile"]),
    ]:
        vals = _finite(data)
        if vals.size == 0:
            continue
        xs = np.sort(vals)
        ys = np.arange(1, xs.size + 1, dtype=float) / xs.size

        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            name=f"{label} ECDF",
            line=dict(color=color, width=2),
        ))

        pvs = _pct_values(vals, percentiles)
        for p, pval in pvs.items():
            y_at_p = float(np.mean(vals <= pval))
            fig.add_trace(go.Scatter(
                x=[pval, pval], y=[0.0, y_at_p],
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                showlegend=False,
                hovertemplate=f"{label} P{p} = {pval:,.0f}<extra></extra>",
            ))
            fig.add_annotation(
                x=pval, y=y_at_p, text=f"P{p}", showarrow=False,
                font=dict(size=9, color=color), yshift=10,
            )

    fig.update_layout(**PLOTLY_LAYOUT, height=350, title=title,
                      xaxis_title=xaxis_title, yaxis_title="ECDF",
                      hovermode="closest")
    fig.update_yaxes(range=[0, 1])
    return fig


# ──────────────────────────── main app ────────────────────────────────


def main():
    st.title("🛢️ 2-Mile Uplift & Decline Analysis")
    st.caption(
        "Compare 2-mile wells against 1-mile analogs to quantify uplift, "
        "then build custom type curves that reflect expected 2-mile performance."
    )

    well_df, geoms = load_and_assemble_wells()
    df_1mile = well_df[well_df["lateral_group"] == "1-Mile"].reset_index(drop=True)
    df_2mile = well_df[well_df["lateral_group"] == "2-Mile"].reset_index(drop=True)
    rtc_curves = load_rtc_curves()

    # ── sidebar ───────────────────────────────────────────────────────
    st.sidebar.title("⚙️ Configuration")

    analysis_mode = st.sidebar.radio(
        "Analog Matching Mode",
        ["Mode A: Range-Based Analog Matching",
         "Mode B: Geometric Corridor"],
        key="analysis_mode_radio",
    )

    st.sidebar.divider()
    active_features, tolerances = [], {}
    corridor_width = CORRIDOR_HALF_WIDTH_M

    if analysis_mode.startswith("Mode A"):
        st.sidebar.subheader("Matching Tolerances")
        available = []
        if well_df["section_ooip"].notna().any():
            available.append("section_ooip")
        if well_df["nearest_producer_m"].notna().any():
            available.append("nearest_producer_m")
        if well_df["nearest_injector_m"].notna().any():
            available.append("nearest_injector_m")
        active_features = st.sidebar.multiselect(
            "Active features", available, default=available)
        if "section_ooip" in active_features:
            tolerances["section_ooip"] = st.sidebar.number_input(
                "± OOIP tol", 0.0, 1e6, 250_000.0, 100_000.0)
        if "nearest_producer_m" in active_features:
            tolerances["nearest_producer_m"] = st.sidebar.number_input(
                "± Nearest producer tol (m)", 0.0, 5000.0, 150.0, 50.0)
        if "nearest_injector_m" in active_features:
            tolerances["nearest_injector_m"] = st.sidebar.number_input(
                "± Nearest injector tol (m)", 0.0, 5000.0, 150.0, 50.0)
    else:
        st.sidebar.subheader("Corridor Parameters")
        corridor_width = st.sidebar.number_input(
            "Corridor half-width (m)", 100.0, 3000.0,
            CORRIDOR_HALF_WIDTH_M, 100.0)

    st.sidebar.divider()
    st.sidebar.subheader("Type Curve Settings")
    st.sidebar.markdown(f"**b (fixed)** = `{B_FIXED}`")
    st.sidebar.markdown(
        f"**flat period** = `{FLAT_DAYS} d` (≈{FLAT_MONTHS} mo)  \n"
        f"**q-limit** = `{Q_LIMIT} bbl/d`"
    )

    # ── core computations ─────────────────────────────────────────────
    cohort_map = build_cohort_map(
        df_2mile, df_1mile, geoms, analysis_mode,
        tolerances, active_features, corridor_width,
    )
    metric_keys = list(PERF_METRICS.keys())

    ratio_df = compute_ratio_per_well(df_2mile, df_1mile, cohort_map)
    gm = {
        col: geometric_mean_of_series(ratio_df[col]) if not ratio_df.empty else np.nan
        for col in ("eur_ratio", "ip90_ratio", "sixm_ratio", "twelvem_ratio")
    }

    incr_df = build_incremental_frame(df_2mile, df_1mile, cohort_map, metric_keys)

    # Distribution arrays
    eur_1m = df_1mile["eur_bbl"].dropna().values
    eur_2m = df_2mile["eur_bbl"].dropna().values
    ip90_1m = df_1mile["ip90_bpd"].dropna().values
    ip90_2m = df_2mile["ip90_bpd"].dropna().values

    default_qi = float(np.median(ip90_2m)) if len(ip90_2m) > 0 else 150.0

    # ── tabs ──────────────────────────────────────────────────────────
    tab_uplift, tab_curves = st.tabs(["📊 Uplift Analysis", "📈 Type Curves"])

    # ================================================================
    # TAB 1 — UPLIFT
    # ================================================================
    with tab_uplift:
        st.header("📊 Empirical Uplift (2-Mile vs 1-Mile)")
        mode_label = (
            "Range-Based" if analysis_mode.startswith("Mode A")
            else f"Corridor (±{int(corridor_width)} m)"
        )
        st.caption(
            f"Mode: **{mode_label}** · {len(df_2mile)} two-mile wells · "
            f"{len(df_1mile)} one-mile wells · "
            f"{sum(len(v) for v in cohort_map.values())} cohort links"
        )

        st.subheader("KPI Ratios (2 mi / 1 mi) — Geometric Mean")
        cols = st.columns(4)
        labels = [("EUR", "eur_ratio"), ("IP90", "ip90_ratio"),
                  ("6M Cal", "sixm_ratio"), ("12M Cal", "twelvem_ratio")]
        for col, (lbl, key) in zip(cols, labels):
            v = gm[key]
            col.metric(f"{lbl} Ratio GM",
                       f"{v:.3f}" if np.isfinite(v) else "—")

        with st.expander("📋 Per-Well KPI Ratios", expanded=False):
            if not ratio_df.empty:
                show = [c for c in ratio_df.columns if c != "index"]
                st.dataframe(ratio_df[show], use_container_width=True,
                             hide_index=True)
            else:
                st.info("No ratio data available.")

        for mk in metric_keys:
            col_name = f"{mk}_incremental"
            if col_name not in incr_df.columns:
                continue
            vals = incr_df[col_name].dropna().values
            if len(vals) == 0:
                continue
            with st.expander(
                f"**{PERF_METRICS[mk]}** — incremental distribution (n={len(vals)})",
                expanded=(mk == "eur_bbl"),
            ):
                sub = (incr_df.dropna(subset=[col_name])
                       .sort_values(col_name, ascending=False))
                bar_colors = [
                    COLORS["incremental"] if v >= 0 else "#d62728"
                    for v in sub[col_name]
                ]
                fig = go.Figure(go.Bar(
                    x=sub["uwi"], y=sub[col_name],
                    marker_color=bar_colors,
                    text=[f"{v:+,.0f}" for v in sub[col_name]],
                    textposition="outside",
                ))
                fig.add_hline(y=0, line=dict(color="gray", dash="dot"))
                fig.update_layout(
                    **PLOTLY_LAYOUT, height=400,
                    title=f"Incremental {PERF_METRICS[mk]} per 2-Mile Well",
                    yaxis_title=f"Δ {PERF_METRICS[mk]}",
                )
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Full Incremental Results")
        disp = [
            c for c in (
                "uwi", "section_name", "n_comparators",
                "eur_bbl", "ip90_bpd", "sixm_bpd", "twelvem_bpd",
                "waterflood_flag", "vintage_year",
            )
            if c in incr_df.columns
        ]
        st.dataframe(incr_df[disp], use_container_width=True, hide_index=True)

    # ================================================================
    # TAB 2 — TYPE CURVES
    # ================================================================
    with tab_curves:
        st.header("📈 Type Curve Builder")
        st.markdown(
            "Use the distributions below to understand the range of 1-mile "
            "and 2-mile performance, then build a custom type curve by "
            "selecting a percentile for **IP90** (sets qi) and **EUR** "
            "(sets the decline)."
        )

        # ── distributions ─────────────────────────────────────────────
        st.subheader("Performance Distributions — 1-Mile vs 2-Mile")
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(
                make_density_plot(eur_1m, eur_2m,
                                  "EUR Distribution", "EUR (bbl)"),
                use_container_width=True,
            )
        with col_b:
            st.plotly_chart(
                make_density_plot(ip90_1m, ip90_2m,
                                  "IP90 Distribution", "IP90 (bbl/d)"),
                use_container_width=True,
            )

        st.subheader("ECDF — EUR")
        st.plotly_chart(
            make_ecdf_plot(eur_1m, eur_2m, "ECDF — EUR", "EUR (bbl)"),
            use_container_width=True,
        )
        st.subheader("ECDF — IP90")
        st.plotly_chart(
            make_ecdf_plot(ip90_1m, ip90_2m, "ECDF — IP90", "IP90 (bbl/d)"),
            use_container_width=True,
        )

        with st.expander("📊 Distribution Summary Statistics", expanded=False):
            rows = []
            for lbl, vals in [
                ("1-Mile EUR (bbl)", eur_1m),
                ("2-Mile EUR (bbl)", eur_2m),
                ("1-Mile IP90 (bbl/d)", ip90_1m),
                ("2-Mile IP90 (bbl/d)", ip90_2m),
            ]:
                s = empirical_summary(vals)
                rows.append({
                    "Metric": lbl, "n": s["n"],
                    "P10": f"{s['p10']:,.0f}" if np.isfinite(s["p10"]) else "—",
                    "P25": f"{s['p25']:,.0f}" if np.isfinite(s["p25"]) else "—",
                    "P50": f"{s['p50']:,.0f}" if np.isfinite(s["p50"]) else "—",
                    "P75": f"{s['p75']:,.0f}" if np.isfinite(s["p75"]) else "—",
                    "P90": f"{s['p90']:,.0f}" if np.isfinite(s["p90"]) else "—",
                    "Mean": f"{s['mean']:,.0f}" if np.isfinite(s["mean"]) else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True)

        st.divider()

        # ── curve builder ─────────────────────────────────────────────
        st.subheader("🔧 Build Your Type Curve")
        st.markdown(
            "Select a percentile for **IP90** (determines qi) and **EUR** "
            "(determines decline). Percentiles reference the **2-mile** "
            "distributions. Oil & gas convention: P10 = optimistic."
        )

        bc = st.columns([1, 1, 2])

        with bc[0]:
            ip90_pct = st.number_input(
                "IP90 Percentile — P(X)", 1, 99, 50, 5,
                help="P10 = optimistic (high value), P90 = conservative.",
            )
            qi_val = (
                float(np.percentile(ip90_2m, 100 - ip90_pct))
                if len(ip90_2m) > 2 else default_qi
            )
            st.metric("Resulting qi (bbl/d)", f"{qi_val:,.1f}")

        with bc[1]:
            eur_pct = st.number_input(
                "EUR Percentile — P(Y)", 1, 99, 50, 5,
                help="P10 = optimistic (high value), P90 = conservative.",
            )
            eur_val = (
                float(np.percentile(eur_2m, 100 - eur_pct))
                if len(eur_2m) > 2 else 0.0
            )
            st.metric("Resulting EUR (Mbbl)", f"{eur_val / 1000:,.1f}")

        # Build curve
        user_curve = None
        if qi_val > 0 and eur_val > 0:
            user_curve = build_curve_from_eur(
                qi_val, eur_val, B_FIXED, FLAT_DAYS, Q_LIMIT)

        with bc[2]:
            if user_curve is not None:
                st.markdown("**Solved Decline Parameters**")
                di_d = user_curve["di"]
                di_y = di_d * 365.25
                eff_dec = (
                    (1.0 - 1.0 / (1.0 + B_FIXED * di_d * 365.25)
                     ** (1.0 / B_FIXED)) * 100
                    if di_d > 0 else 0.0
                )
                pc = st.columns(3)
                pc[0].metric("Di (1/day)", f"{di_d:.6f}")
                pc[1].metric("Di (1/yr)", f"{di_y:.4f}")
                pc[2].metric("b", f"{B_FIXED:.2f}")
                st.caption(f"Effective annual decline ≈ {eff_dec:.1f}%")
            else:
                st.info("Enter valid percentiles to generate a curve.")

        # ── rate plot ─────────────────────────────────────────────────
        st.subheader("Rate Type Curve — q(t)")
        fig_rate = go.Figure()

        if user_curve is not None:
            fig_rate.add_trace(go.Scatter(
                x=user_curve["t"], y=user_curve["q"], mode="lines",
                line=dict(color="#e74c3c", width=3.5),
                name=(
                    f"Your Curve — P{ip90_pct} IP90 / P{eur_pct} EUR "
                    f"(qi={qi_val:,.0f}, EUR={eur_val / 1000:,.0f} Mbbl)"
                ),
                hovertemplate=(
                    "Your Curve<br>Day %{x:,.0f}<br>"
                    "%{y:,.1f} bbl/d<extra></extra>"
                ),
            ))

        for i, rtc in enumerate(rtc_curves):
            fig_rate.add_trace(go.Scatter(
                x=rtc["t"], y=rtc["q"], mode="lines",
                line=dict(color=RTC_PALETTE[i % len(RTC_PALETTE)],
                          width=2, dash="dash"),
                name=(
                    f"RTC: {rtc['name']} "
                    f"(qi={rtc['qi']:,.0f}, "
                    f"EUR={rtc['eur_actual_bbl'] / 1000:,.0f} Mbbl)"
                ),
                hovertemplate=(
                    f"RTC: {rtc['name']}<br>"
                    "Day %{x:,.0f}<br>%{y:,.1f} bbl/d<extra></extra>"
                ),
            ))

        fig_rate.add_hline(
            y=Q_LIMIT, line_dash="dot", line_color="grey", opacity=0.5,
            annotation_text=f"{Q_LIMIT} bbl/d limit",
        )
        fig_rate.update_layout(
            **PLOTLY_LAYOUT, height=550,
            xaxis_title="Days since start of flat period",
            yaxis_title="Rate (bbl/d)",
            title="Rate — q(t)", hovermode="closest",
        )
        fig_rate.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_rate, use_container_width=True)

        # ── cumulative plot ───────────────────────────────────────────
        st.subheader("Cumulative Type Curve — N(t)")
        fig_cum = go.Figure()

        if user_curve is not None:
            fig_cum.add_trace(go.Scatter(
                x=user_curve["t"], y=user_curve["N"] / 1000, mode="lines",
                line=dict(color="#e74c3c", width=3.5),
                name=f"Your Curve — EUR={user_curve['eur_actual_bbl'] / 1000:,.0f} Mbbl",
                hovertemplate=(
                    "Your Curve<br>Day %{x:,.0f}<br>"
                    "%{y:,.1f} Mbbl<extra></extra>"
                ),
            ))

        for i, rtc in enumerate(rtc_curves):
            fig_cum.add_trace(go.Scatter(
                x=rtc["t"], y=rtc["N"] / 1000, mode="lines",
                line=dict(color=RTC_PALETTE[i % len(RTC_PALETTE)],
                          width=2, dash="dash"),
                name=f"RTC: {rtc['name']} (EUR={rtc['eur_actual_bbl'] / 1000:,.0f} Mbbl)",
                hovertemplate=(
                    f"RTC: {rtc['name']}<br>"
                    "Day %{x:,.0f}<br>%{y:,.1f} Mbbl<extra></extra>"
                ),
            ))

        fig_cum.update_layout(
            **PLOTLY_LAYOUT, height=500,
            xaxis_title="Days since start of flat period",
            yaxis_title="Cumulative (Mbbl)",
            title="Cumulative — N(t)", hovermode="closest",
        )
        fig_cum.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_cum, use_container_width=True)

        # ── RTC benchmarking ──────────────────────────────────────────
        if rtc_curves and user_curve is not None:
            st.subheader("RTC Benchmarking Summary")
            bench = []
            for rtc in rtc_curves:
                d_abs = (user_curve["eur_actual_bbl"]
                         - rtc["eur_actual_bbl"]) / 1000
                d_pct = (
                    (user_curve["eur_actual_bbl"] - rtc["eur_actual_bbl"])
                    / rtc["eur_actual_bbl"] * 100
                    if rtc["eur_actual_bbl"] else np.nan
                )
                qi_r = (user_curve["qi"] / rtc["qi"]
                        if rtc["qi"] > 0 else np.nan)
                bench.append({
                    "RTC Name": rtc["name"],
                    "RTC qi (bbl/d)": round(rtc["qi"], 1),
                    "RTC EUR (Mbbl)": round(rtc["eur_actual_bbl"] / 1000, 1),
                    "RTC b": round(rtc["b"], 2),
                    "Your qi (bbl/d)": round(user_curve["qi"], 1),
                    "Your EUR (Mbbl)": round(
                        user_curve["eur_actual_bbl"] / 1000, 1),
                    "ΔEUR (Mbbl)": f"{d_abs:+,.1f}",
                    "ΔEUR (%)": (f"{d_pct:+.1f}%"
                                 if np.isfinite(d_pct) else "—"),
                    "qi Ratio": (f"{qi_r:.2f}"
                                 if np.isfinite(qi_r) else "—"),
                })
            st.dataframe(pd.DataFrame(bench), use_container_width=True,
                         hide_index=True)

            # EUR bar comparison
            st.subheader("EUR Bar — RTC vs Your Curve")
            fig_bar = go.Figure()
            cats, bar_vals, bar_cols = [], [], []
            for i, rtc in enumerate(rtc_curves):
                cats.append(f"RTC: {rtc['name']}")
                bar_vals.append(rtc["eur_actual_bbl"] / 1000)
                bar_cols.append(RTC_PALETTE[i % len(RTC_PALETTE)])
            cats.append(f"Your Curve (P{ip90_pct}/P{eur_pct})")
            bar_vals.append(user_curve["eur_actual_bbl"] / 1000)
            bar_cols.append("#e74c3c")
            fig_bar.add_trace(go.Bar(
                x=cats, y=bar_vals, marker_color=bar_cols,
                text=[f"{v:,.0f}" for v in bar_vals],
                textposition="outside",
            ))
            fig_bar.update_layout(
                **PLOTLY_LAYOUT, height=400,
                yaxis_title="EUR (Mbbl)",
                title="EUR Comparison", hovermode="closest",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── export ────────────────────────────────────────────────────
        st.divider()
        st.subheader("📥 Export")

        curve_df = None
        if user_curve is not None:
            curve_df = pd.DataFrame({
                "day": user_curve["t"],
                "rate_bpd": user_curve["q"],
                "cum_bbl": user_curve["N"],
                "cum_Mbbl": user_curve["N"] / 1000,
            })

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            if user_curve is not None and curve_df is not None:
                pd.DataFrame([{
                    "qi_bpd": user_curve["qi"],
                    "di_per_day": user_curve["di"],
                    "di_per_year": user_curve["di"] * 365.25,
                    "b": user_curve["b"],
                    "eur_target_bbl": user_curve["eur_target_bbl"],
                    "eur_actual_bbl": user_curve["eur_actual_bbl"],
                    "eur_actual_Mbbl": user_curve["eur_actual_bbl"] / 1000,
                    "flat_days": user_curve["flat_days"],
                    "ip90_percentile": ip90_pct,
                    "eur_percentile": eur_pct,
                }]).to_excel(writer, index=False, sheet_name="Curve_Parameters")
                curve_df.to_excel(writer, index=False,
                                  sheet_name="Curve_Data")

            if not incr_df.empty:
                incr_df.to_excel(writer, index=False,
                                 sheet_name="Incremental_Results")

            if not ratio_df.empty:
                ratio_df.to_excel(writer, index=False,
                                  sheet_name="Per_Well_Ratios")

            cohort_rows = [
                {"uwi_2mile": u, "uwi_1mile_comparator": w}
                for u, lst in cohort_map.items() for w in lst
            ]
            if cohort_rows:
                pd.DataFrame(cohort_rows).to_excel(
                    writer, index=False, sheet_name="Cohort_Mappings")

            if rtc_curves:
                pd.DataFrame([{
                    "name": r["name"],
                    "qi_bpd": r["qi"],
                    "di_per_day": r["di"],
                    "b": r["b"],
                    "eur_bbl": r["eur_actual_bbl"],
                    "eur_Mbbl": r["eur_actual_bbl"] / 1000,
                } for r in rtc_curves]).to_excel(
                    writer, index=False, sheet_name="RTC_Reference")

        st.download_button(
            "📥 Download Full Results (Excel)",
            buf.getvalue(),
            file_name="2mile_type_curves_uplift.xlsx",
            mime="application/vnd.openxmlformats-officedocument"
                 ".spreadsheetml.sheet",
        )

        if curve_df is not None:
            st.download_button(
                "📥 Download Curve Data (CSV)",
                curve_df.to_csv(index=False).encode(),
                file_name="2mile_type_curve.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()