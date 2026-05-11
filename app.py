import io
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from scipy.interpolate import PchipInterpolator
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="2-Mile Well Analysis",
    page_icon="🛢️",
    layout="wide",
)

XLSX_PATH        = "w.xlsx"
SHP_1M_PATH      = "1M.shp"
SHP_2M_PATH      = "2M.shp"
PROD_DATA_FILE   = "tcgenprod.xlsx"
RTC_XLSX_PATH    = "rtc.xlsx"

TARGET_METRIC_CRS       = "EPSG:26913"
WATERFLOOD_BUFFER_M     = 200.0
MILE_TO_M               = 1609.34
CORRIDOR_HALF_WIDTH_M   = 900.0

Q_LIMIT            = 2.0
FLAT_MONTHS        = 1.44
FLAT_DAYS          = int(round(FLAT_MONTHS * 30.4375))
B_FIXED            = 0.95
DAYS_PER_MONTH_AVG = 30.4375

MIN_COMPS_FOR_UPLIFT = 3
MIN_MONTHS_FOR_FIT   = 6
OUTLIER_Z_CUTOFF     = 3.5

COLORS = {
    "1-Mile": "#1f77b4", "2-Mile": "#ff7f0e", "incremental": "#2ca02c",
    "P10": "#27ae60", "P25": "#2ecc71", "P50": "#3498db",
    "P75": "#e74c3c", "P90": "#c0392b",
}

PERF_METRICS = {
    "eur_bbl":      "EUR (bbl)",
    "ip90_bpd":     "IP90 (bbl/d)",
    "sixm_bpd":     "6M Cal. Rate (bbl/d)",
    "twelvem_bpd":  "12M Cal. Rate (bbl/d)",
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


def _std_uwi(s: pd.Series) -> pd.Series:
    return (
        s.astype(str).str.strip().str.upper()
        .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})
    )


def _std_str(s: pd.Series) -> pd.Series:
    return _std_uwi(s)


def require_cols(df: pd.DataFrame, required: list, df_name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"{df_name} missing columns: {missing}")
        st.stop()


def _safe_unary_union(gseries):
    try:
        return gseries.union_all()
    except AttributeError:
        return gseries.unary_union


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
    power  = (b - 1.0) / b
    return factor * (1.0 - np.power(1.0 + b * di * t, power))


def t_to_rate(di, qi, b, q_target):
    if q_target <= 0 or di <= 0:
        return np.inf
    if b == 0:
        return max(0.0, math.log(qi / q_target) / di)
    return max(0.0, ((qi / q_target) ** b - 1.0) / (b * di))


def find_Di_for_eur_post(qi, eur_post, b, q_target, tol=1e-6):
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


def build_piecewise_curve(qi, di, b, flat_days=FLAT_DAYS, q_limit=Q_LIMIT):
    t_list = list(range(flat_days + 1))
    q_list = [qi] * (flat_days + 1)
    if di is not None and di > 0:
        t_dec = 0
        while True:
            t_dec += 1
            q = hyp_rate(t_dec, qi, di, b)
            t_list.append(flat_days + t_dec)
            q_list.append(max(q, q_limit))
            if q <= q_limit or t_dec > 60000:
                break
    return np.array(t_list, dtype=float), np.array(q_list, dtype=float)


def build_curve_from_eur(qi, eur_bbl, b, flat_days=FLAT_DAYS, q_limit=Q_LIMIT):
    eur_flat = qi * flat_days
    eur_post = max(eur_bbl - eur_flat, 0.0)
    di = None
    if eur_post > 0 and qi > q_limit:
        di = find_Di_for_eur_post(qi, eur_post, b, q_limit)
    t_arr, q_arr = build_piecewise_curve(qi, di if di else 0.0, b, flat_days, q_limit)
    N_arr = np.zeros_like(t_arr)
    if len(t_arr) > 1:
        for i in range(1, len(t_arr)):
            dt = t_arr[i] - t_arr[i - 1]
            N_arr[i] = N_arr[i - 1] + 0.5 * (q_arr[i - 1] + q_arr[i]) * dt
    return {
        "t": t_arr, "q": q_arr, "N": N_arr,
        "qi": float(qi), "di": float(di) if di else 0.0, "b": float(b),
        "eur_target_bbl": float(eur_bbl),
        "eur_actual_bbl": float(N_arr[-1]) if len(N_arr) > 0 else 0.0,
        "flat_days": int(flat_days),
    }


def calc_eur_trap(t_arr, q_arr) -> float:
    if len(t_arr) < 2:
        return 0.0
    return float(np.trapezoid(q_arr, t_arr))


REQUIRED_WELL_COLUMNS = [
    "UWI", "Section Name", "Well Type", " Hz Length (m)",
    "Oil + Cond: EUR (Mbbl)", "Oil + Cond: IP 90 Cal. Rate (bbl/d)",
    "Oil + Cond: 6M Cal. Rate (bbl/d)", "Oil + Cond: 12M Cal. Rate (bbl/d)",
    "Objective", "On Prod Date", "On Inj Date", "FOOZ",
]

RENAME_WELL = {
    "UWI": "uwi", "Section Name": "section_name", "Well Type": "well_type",
    "Hz Length (m)": "hz_length_m",
    "Oil + Cond: EUR (Mbbl)": "eur_mbbl",
    "Oil + Cond: IP 90 Cal. Rate (bbl/d)": "ip90_bpd",
    "Oil + Cond: 6M Cal. Rate (bbl/d)": "sixm_bpd",
    "Oil + Cond: 12M Cal. Rate (bbl/d)": "twelvem_bpd",
    "Objective": "objective", "On Prod Date": "on_prod_date",
    "On Inj Date": "on_inj_date", "FOOZ": "fooz",
}


@st.cache_data(show_spinner=False)
def load_well_table():
    xls = pd.ExcelFile(XLSX_PATH)
    raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    require_cols(raw, REQUIRED_WELL_COLUMNS, f"`{XLSX_PATH}` sheet 0")
    out = raw.rename(columns=RENAME_WELL)
    for c in ["uwi", "section_name", "well_type", "objective", "fooz"]:
        if c in out.columns:
            out[c] = _std_str(out[c])
    for c in ["on_prod_date", "on_inj_date"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    if "eur_mbbl" in out.columns:
        out["eur_bbl"] = pd.to_numeric(out["eur_mbbl"], errors="coerce") * 1000.0
    for c in ["hz_length_m", "ip90_bpd", "sixm_bpd", "twelvem_bpd", "eur_bbl"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    out["vintage_year"] = out["on_prod_date"].dt.year
    out["is_fooz"]      = out["fooz"].fillna("").eq("YES")
    out["is_injector"]  = out["objective"].fillna("").eq("INJ")
    return out


@st.cache_data(show_spinner=False)
def load_section_ooip():
    xls = pd.ExcelFile(XLSX_PATH)
    raw = pd.read_excel(xls, sheet_name=xls.sheet_names[1])
    require_cols(raw, ["Section", "OOIP"], f"`{XLSX_PATH}` sheet 1")
    out = raw.copy()
    out["Section"] = _std_str(out["Section"])
    out["OOIP"]    = pd.to_numeric(out["OOIP"], errors="coerce")
    return out.rename(columns={"Section": "section_name", "OOIP": "section_ooip"})


@st.cache_resource(show_spinner=False)
def load_geometries():
    g1 = gpd.read_file(SHP_1M_PATH)
    g2 = gpd.read_file(SHP_2M_PATH)
    for g, path in [(g1, SHP_1M_PATH), (g2, SHP_2M_PATH)]:
        if "UWI" not in g.columns:
            st.error(f"`{path}` missing field `UWI`."); st.stop()
    g1["lateral_group"] = "1-Mile"
    g2["lateral_group"] = "2-Mile"
    gdf = pd.concat([g1, g2], ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=g1.crs or g2.crs)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    try:
        gdf = gdf.to_crs(TARGET_METRIC_CRS)
    except Exception:
        c = _safe_unary_union(gdf.geometry).centroid
        utm = int((c.x + 180) // 6) + 1
        epsg = 32600 + utm if c.y >= 0 else 32700 + utm
        gdf = gdf.to_crs(f"EPSG:{epsg}")
    gdf["uwi"] = _std_uwi(gdf["UWI"])
    gdf = gdf.sort_values("lateral_group", ascending=False)\
             .drop_duplicates("uwi", keep="first")
    gdf["midpoint"] = gdf.geometry.apply(
        lambda g: g.interpolate(0.5, normalized=True)
        if g is not None and not g.is_empty else None
    )
    return gdf[["uwi", "lateral_group", "geometry", "midpoint"]].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_production_data():
    if not os.path.exists(PROD_DATA_FILE):
        return None, f"`{PROD_DATA_FILE}` not found."
    df = pd.read_excel(PROD_DATA_FILE)
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if "uwi" in cl:              rename[c] = "uwi"
        elif "month" in cl or "date" in cl: rename[c] = "month"
        elif "bbl" in cl or "rate"  in cl:  rename[c] = "rate"
    df = df.rename(columns=rename)
    for req in ["uwi", "month", "rate"]:
        if req not in df.columns:
            return None, f"`{PROD_DATA_FILE}` missing column `{req}`."
    df["uwi"]  = _std_uwi(df["uwi"])
    df["date"] = pd.to_datetime(df["month"], errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna(subset=["date", "rate", "uwi"])
    df = df[df["rate"] >= 0].copy()
    df = df.sort_values(["uwi", "date"]).reset_index(drop=True)
    df["days_in_month"] = df["date"].dt.days_in_month
    df["monthly_vol"]   = df["rate"] * df["days_in_month"]
    return df, None


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
            name           = str(row["Name"])
            qi_rtc         = float(row["Qi"])
            b_rtc          = float(row["b"])
            eur_rtc        = float(row["EUR"])
            flat_days_rtc  = float(row["Months"]) * 30.0
            result = build_curve_from_eur(qi_rtc, eur_rtc, b_rtc,
                                           int(flat_days_rtc), Q_LIMIT)
            result["name"] = name
            curves.append(result)
        except Exception:
            continue
    return curves


@st.cache_resource(show_spinner=False)
def compute_spatial_features(_geoms_gdf, _well_meta):
    if _geoms_gdf.empty:
        return pd.DataFrame(columns=[
            "uwi", "nearest_producer_m", "nearest_injector_m",
            "n_within_400m", "n_within_800m", "waterflood_flag",
        ])
    g = _geoms_gdf.copy().reset_index(drop=True)
    meta = _well_meta[["uwi", "on_prod_date", "on_inj_date",
                        "objective", "is_injector"]].copy()
    g = g.merge(meta, on="uwi", how="left")
    g["is_injector"] = g["is_injector"].fillna(False).astype(bool)
    sindex = g.sindex
    rows = []
    for i, row in g.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            rows.append(dict(uwi=row["uwi"], nearest_producer_m=np.nan,
                             nearest_injector_m=np.nan,
                             n_within_400m=0, n_within_800m=0,
                             waterflood_flag=False))
            continue
        cand = [j for j in sindex.intersection(geom.buffer(5000).bounds) if j != i]
        nearest_prod, nearest_inj = np.nan, np.nan
        n400, n800 = 0, 0
        for j in cand:
            og = g.iloc[j].geometry
            if og is None or og.is_empty: continue
            d = geom.distance(og)
            if d <= 400: n400 += 1
            if d <= 800: n800 += 1
            if bool(g.iloc[j].get("is_injector", False)):
                if not np.isfinite(nearest_inj) or d < nearest_inj:
                    nearest_inj = d
            else:
                if not np.isfinite(nearest_prod) or d < nearest_prod:
                    nearest_prod = d
        wf = False
        if row.get("lateral_group") == "2-Mile":
            buf = geom.buffer(WATERFLOOD_BUFFER_M)
            for j in [jj for jj in sindex.intersection(buf.bounds) if jj != i]:
                o = g.iloc[j]
                if o.geometry is None or o.geometry.is_empty: continue
                if not buf.intersects(o.geometry): continue
                inj_d  = o.get("on_inj_date", pd.NaT)
                prod_d = row.get("on_prod_date", pd.NaT)
                if pd.notna(inj_d) and pd.notna(prod_d) and inj_d <= prod_d:
                    wf = True; break
        rows.append(dict(
            uwi=row["uwi"],
            nearest_producer_m=float(nearest_prod) if np.isfinite(nearest_prod) else np.nan,
            nearest_injector_m=float(nearest_inj) if np.isfinite(nearest_inj) else np.nan,
            n_within_400m=int(n400), n_within_800m=int(n800),
            waterflood_flag=bool(wf),
        ))
    return pd.DataFrame(rows)


def corridor_match(target_geom, geoms_gdf, half_width=CORRIDOR_HALF_WIDTH_M):
    if target_geom is None or target_geom.is_empty:
        return []
    corridor = target_geom.buffer(half_width)
    ones = geoms_gdf[geoms_gdf["lateral_group"] == "1-Mile"]
    if ones.empty: return []
    inside = ones["midpoint"].apply(
        lambda mp: mp is not None and not mp.is_empty and corridor.contains(mp)
    )
    return ones.loc[inside, "uwi"].tolist()


def range_match(target_row, df_1mile, tolerances, active_features):
    if df_1mile.empty: return []
    mask = pd.Series(True, index=df_1mile.index)
    for feat, tol in tolerances.items():
        if feat not in active_features: continue
        tv = target_row.get(feat, np.nan)
        if pd.isna(tv) or feat not in df_1mile.columns: continue
        mask &= df_1mile[feat].between(tv - tol, tv + tol)
    tp = target_row.get("on_prod_date", pd.NaT)
    if pd.notna(tp) and "on_prod_date" in df_1mile.columns:
        mask &= df_1mile["on_prod_date"].notna() & (df_1mile["on_prod_date"] < tp)
    return df_1mile.loc[mask, "uwi"].tolist()


def empirical_summary(values):
    vals = np.array([v for v in values if np.isfinite(v)], dtype=float)
    n = len(vals)
    if n == 0:
        return dict(n=0, median=np.nan, mean=np.nan, std=np.nan,
                    min=np.nan, q10=np.nan, q25=np.nan, q50=np.nan,
                    q75=np.nan, q90=np.nan, max=np.nan)
    return dict(
        n=n,
        median=float(np.median(vals)),
        mean=float(np.mean(vals)),
        std=float(np.std(vals, ddof=1)) if n > 1 else np.nan,
        min=float(np.min(vals)),
        q10=float(np.percentile(vals, 10)),
        q25=float(np.percentile(vals, 25)),
        q50=float(np.percentile(vals, 50)),
        q75=float(np.percentile(vals, 75)),
        q90=float(np.percentile(vals, 90)),
        max=float(np.max(vals)),
    )


def compute_incremental(well_row, comparator_df, metric_keys):
    result = {"uwi": well_row["uwi"]}

    for mk in metric_keys:
        val = well_row.get(mk, np.nan)
        if comparator_df.empty or pd.isna(val):
            result[f"{mk}_baseline"]    = np.nan
            result[f"{mk}_incremental"] = np.nan
            result[f"{mk}_pct_uplift"]  = np.nan
            result[f"{mk}_ratio"]       = np.nan
            continue

        bl_total = float(comparator_df[mk].median())
        incr = float(val) - bl_total
        pct  = (incr / bl_total * 100) if bl_total else np.nan
        ratio = float(val) / bl_total if bl_total else np.nan

        result[f"{mk}_baseline"]    = bl_total
        result[f"{mk}_incremental"] = incr
        result[f"{mk}_pct_uplift"]  = pct
        result[f"{mk}_ratio"]       = ratio

    result["n_comparators"] = len(comparator_df)
    return result


def compute_ratio_per_well_2mi_vs_1mi(df_2mi: pd.DataFrame,
                                      df_1mi: pd.DataFrame,
                                      cohort_map: dict) -> pd.DataFrame:
    rows = []
    for _, row2 in df_2mi.iterrows():
        uwi2 = row2["uwi"]
        matched = cohort_map.get(uwi2, [])
        comp_df = df_1mi[df_1mi["uwi"].isin(matched)]
        medians = {
            "eur_bbl":   comp_df["eur_bbl"].median(),
            "ip90_bpd":  comp_df["ip90_bpd"].median(),
            "sixm_bpd":  comp_df["sixm_bpd"].median(),
            "twelvem_bpd": comp_df["twelvem_bpd"].median(),
        }
        eur_ratio = (row2["eur_bbl"] / medians["eur_bbl"]) if (np.isfinite(medians["eur_bbl"]) and medians["eur_bbl"] > 0) else np.nan
        ip90_ratio = (row2["ip90_bpd"] / medians["ip90_bpd"]) if (np.isfinite(medians["ip90_bpd"]) and medians["ip90_bpd"] > 0) else np.nan
        sixm_ratio = (row2["sixm_bpd"] / medians["sixm_bpd"]) if (np.isfinite(medians["sixm_bpd"]) and medians["sixm_bpd"] > 0) else np.nan
        twelvem_ratio = (row2["twelvem_bpd"] / medians["twelvem_bpd"]) if (np.isfinite(medians["twelvem_bpd"]) and medians["twelvem_bpd"] > 0) else np.nan

        rows.append({
            "uwi": uwi2,
            "eur_2mi_actual": row2["eur_bbl"],
            "eur_1mi_baseline": medians["eur_bbl"],
            "eur_ratio": eur_ratio,
            "ip90_2mi_actual": row2["ip90_bpd"],
            "ip90_1mi_baseline": medians["ip90_bpd"],
            "ip90_ratio": ip90_ratio,
            "sixm_2mi_actual": row2["sixm_bpd"],
            "sixm_1mi_baseline": medians["sixm_bpd"],
            "sixm_ratio": sixm_ratio,
            "twelvem_2mi_actual": row2["twelvem_bpd"],
            "twelvem_1mi_baseline": medians["twelvem_bpd"],
            "twelvem_ratio": twelvem_ratio,
        })
    return pd.DataFrame(rows)


def geometric_mean_of_series(series: pd.Series) -> float:
    s = series.dropna()
    s = s[(np.isfinite(s)) & (s > 0)]
    if s.empty:
        return np.nan
    return float(np.exp(np.log(s).mean()))


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
            matched = corridor_match(
                gr.iloc[0].geometry, geoms, corridor_width
            ) if not gr.empty else []
        mapping[uwi2] = matched
    return mapping


def build_incremental_frame(df_2mile, df_1mile, cohort_map, metric_keys):
    records = []
    for _, row2 in df_2mile.iterrows():
        uwi2     = row2["uwi"]
        matched  = cohort_map.get(uwi2, [])
        comp_df  = df_1mile[df_1mile["uwi"].isin(matched)]
        rec      = compute_incremental(row2, comp_df, metric_keys)
        for col in ["section_name", "hz_length_m", "on_prod_date",
                     "vintage_year", "eur_bbl", "ip90_bpd", "sixm_bpd", "twelvem_bpd",
                     "waterflood_flag", "section_ooip"]:
            rec[col] = row2.get(col, np.nan)
        records.append(rec)
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["uwi"])


@st.cache_data(show_spinner="Loading well table & geometries…")
def load_and_assemble_wells():
    well_raw = load_well_table()
    sec_ooip = load_section_ooip()
    geoms    = load_geometries()

    membership = geoms[["uwi", "lateral_group"]].drop_duplicates()
    well_df = membership.merge(well_raw, on="uwi", how="left")
    well_df = well_df.merge(sec_ooip, on="section_name", how="left")
    well_df = well_df.replace([np.inf, -np.inf], np.nan)
    for c in ["is_injector", "is_fooz"]:
        if c not in well_df.columns:
            well_df[c] = False
        well_df[c] = well_df[c].fillna(False).astype(bool)

    meta = well_df[["uwi", "on_prod_date", "on_inj_date",
                     "objective", "is_injector"]].copy()
    spatial = compute_spatial_features(geoms, meta)
    well_df = well_df.merge(spatial, on="uwi", how="left")

    well_df = well_df[~well_df["is_fooz"]].reset_index(drop=True)
    return well_df, geoms


def make_density_plot_with_percentiles(
    data_1m, data_2m, title, xaxis_title,
    percentiles_to_show=(90, 75, 50, 25, 10)
):
    """Create a KDE density plot for 1M and 2M data with percentile vertical lines
    and enhanced tooltips (n, min/max/mean, percentiles)."""
    fig = go.Figure()

    datasets = [
        (data_1m, "1-Mile", COLORS["1-Mile"]),
        (data_2m, "2-Mile", COLORS["2-Mile"]),
    ]

    for data, label, color in datasets:
        vals = np.array([v for v in data if np.isfinite(v)], dtype=float)
        if len(vals) < 3:
            continue

        # KDE
        kde = gaussian_kde(vals, bw_method="scott")
        x_min, x_max = vals.min() * 0.8, vals.max() * 1.2
        x_grid = np.linspace(x_min, x_max, 300)
        y_grid = kde(x_grid)
        # Normalize density for nicer comparison
        y_grid = y_grid / y_grid.max()

        # Summary stats
        n_vals = len(vals)
        min_v  = float(vals.min())
        max_v  = float(vals.max())
        mean_v = float(vals.mean())
        pcts = {p: float(np.percentile(vals, p)) for p in percentiles_to_show}
        ptxt = " | ".join([f"P{p}={pcts[p]:,.0f}" for p in percentiles_to_show])
        # Build hover text with summary data embedded in the tooltip
        hover_text = (
            f"<b>{label}</b><br>"
            f"{xaxis_title}: %{{x:,.0f}}<br>"
            f"Density: %{{y:.6f}}<br>"
            f"n={n_vals} | Min={min_v:,.0f} | Max={max_v:,.0f} | Mean={mean_v:,.0f}<br>"
            f"{ptxt}<br>"
            f"<extra></extra>"
        )

        fig.add_trace(go.Scatter(
            x=x_grid, y=y_grid, mode="lines", name=f"{label} (n={n_vals})",
            line=dict(color=color, width=2.5),
            fill="tozeroy", opacity=0.25,
            hovertemplate=hover_text
        ))

        # Add percentile vertical lines with lightweight hover
        for p in percentiles_to_show:
            pval = pcts[p]
            y_at_p = float(kde(np.array([pval]))[0])
            fig.add_trace(go.Scatter(
                x=[pval, pval], y=[0, y_at_p],
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                showlegend=False,
                hovertemplate=f"{label} P{p} = {pval:,.0f}<extra></extra>",
            ))
            fig.add_annotation(
                x=pval, y=y_at_p,
                text=f"P{p}", showarrow=False,
                font=dict(size=9, color=color),
                yshift=10,
            )

    fig.update_layout(
        **PLOTLY_LAYOUT, height=400,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Density",
        hovermode="closest",
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def make_smoothed_ecdf_plot(data_1m, data_2m, title, xaxis_title):
    """
    Create a smoothed ECDF plot (monotone) for 1-Mile and 2-Mile data using a PCHIP interpolant
    built on strictly increasing (unique) ECDF points. Handles duplicates in the data.
    """

    def _ecdf_points(data):
        vals = np.array([v for v in data if np.isfinite(v)], dtype=float)
        if len(vals) == 0:
            return None, None, None

        vals_sorted = np.sort(vals)
        n = len(vals_sorted)

        # Build ECDF at unique x-values (collapse duplicates)
        x_unique = []
        y_ecdf = []
        i = 0
        while i < n:
            v = vals_sorted[i]
            j = i
            # move to the last index where value == v
            while j + 1 < n and vals_sorted[j + 1] == v:
                j += 1
            x_unique.append(v)
            # ECDF value at the last occurrence of v
            y_ecdf.append((j + 1) / n)
            i = j + 1

        x_unique = np.asarray(x_unique)
        y_ecdf = np.asarray(y_ecdf)

        # If we have fewer than 2 unique points, provide a safe fallback interpolator
        if x_unique.size < 2:
            min_v = x_unique[0]
            max_v = x_unique[-1]

            def interp_fallback(xgrid):
                if max_v == min_v:
                    # All data equal: jump from 0 to 1 at min_v (approximate)
                    return np.where(xgrid < min_v, 0.0, 1.0)
                # Simple linear ramp between min and max
                return np.clip((xgrid - min_v) / (max_v - min_v), 0.0, 1.0)

            return x_unique, y_ecdf, interp_fallback

        # Normal case: at least 2 unique points, use PCHIP on the unique points
        interp = PchipInterpolator(x_unique, y_ecdf)
        return x_unique, y_ecdf, interp

    def _safe_eval(interp, xgrid, fallback=None):
        if interp is None:
            if callable(fallback):
                return fallback(xgrid)
            else:
                return None
        return interp(xgrid)

    # Build data for both datasets
    x1, y1, interp1 = _ecdf_points(data_1m)
    x2, y2, interp2 = _ecdf_points(data_2m)

    if interp1 is None and interp2 is None:
        fig = go.Figure()
        fig.update_layout(**PLOTLY_LAYOUT, height=350, title=title,
                          xaxis_title=xaxis_title, yaxis_title="ECDF",
                          hovermode="closest")
        return fig

    # Determine x grid domain
    xmin = None
    xmax = None
    if x1 is not None and x2 is not None:
        xmin = min(x1[0], x2[0])
        xmax = max(x1[-1], x2[-1])
    elif x1 is not None:
        xmin, xmax = x1[0], x1[-1]
    elif x2 is not None:
        xmin, xmax = x2[0], x2[-1]

    if xmin is None or xmax is None:
        xmin, xmax = 0.0, 1.0  # fallback

    xgrid = np.linspace(xmin, xmax, 300)
    fig = go.Figure()

    if interp1 is not None:
        ygrid1 = interp1(xgrid) if callable(interp1) else interp1(xgrid)
        fig.add_trace(go.Scatter(
            x=xgrid, y=ygrid1, mode="lines",
            name="1-Mile ECDF (smoothed)",
            line=dict(color=COLORS["1-Mile"], width=2)
        ))
    if interp2 is not None:
        ygrid2 = interp2(xgrid) if callable(interp2) else interp2(xgrid)
        fig.add_trace(go.Scatter(
            x=xgrid, y=ygrid2, mode="lines",
            name="2-Mile ECDF (smoothed)",
            line=dict(color=COLORS["2-Mile"], width=2)
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT, height=350,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="ECDF",
        hovermode="closest",
    )
    fig.update_yaxes(range=[0, 1])
    return fig


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

    # --- Sidebar ---
    st.sidebar.title("⚙️ Configuration")

    analysis_mode = st.sidebar.radio(
        "Analog Matching Mode",
        ["Mode A: Range-Based Analog Matching", "Mode B: Geometric Corridor"],
        key="analysis_mode_radio",
    )

    st.sidebar.divider()
    active_features, tolerances = [], {}
    corridor_width = CORRIDOR_HALF_WIDTH_M

    if analysis_mode.startswith("Mode A"):
        st.sidebar.subheader("Matching Tolerances")
        available = []
        if well_df["section_ooip"].notna().any():         available.append("section_ooip")
        if well_df["nearest_producer_m"].notna().any():   available.append("nearest_producer_m")
        if well_df["nearest_injector_m"].notna().any():   available.append("nearest_injector_m")
        active_features = st.sidebar.multiselect("Active features", available, default=available)
        if "section_ooip" in active_features:
            tolerances["section_ooip"] = st.sidebar.number_input("± OOIP tol", 0.0, 1e6, 250000.0, 1e5)
        if "nearest_producer_m" in active_features:
            tolerances["nearest_producer_m"] = st.sidebar.number_input("± Nearest producer tol (m)", 0.0, 5000.0, 150.0, 50.0)
        if "nearest_injector_m" in active_features:
            tolerances["nearest_injector_m"] = st.sidebar.number_input("± Nearest injector tol (m)", 0.0, 5000.0, 150.0, 50.0)
    else:
        st.sidebar.subheader("Corridor Parameters")
        corridor_width = st.sidebar.number_input(
            "Corridor half-width (m)", 100.0, 3000.0, CORRIDOR_HALF_WIDTH_M, 100.0)

    st.sidebar.divider()
    st.sidebar.subheader("Type Curve Settings")
    st.sidebar.markdown(f"**b (fixed)** = `{B_FIXED}`")
    st.sidebar.markdown(
        f"**flat period** = `{FLAT_DAYS} d` (≈{FLAT_MONTHS} mo)<br>"
        f"**q-limit** = `{Q_LIMIT} bbl/d`",
        unsafe_allow_html=True,
    )

    # --- Prepare core data ---
    cohort_map = build_cohort_map(df_2mile, df_1mile, geoms, analysis_mode,
                                  tolerances, active_features, corridor_width)

    ratio_df = compute_ratio_per_well_2mi_vs_1mi(df_2mile, df_1mile, cohort_map)
    gm_eur   = geometric_mean_of_series(ratio_df["eur_ratio"]) if not ratio_df.empty else np.nan
    gm_ip90  = geometric_mean_of_series(ratio_df["ip90_ratio"]) if not ratio_df.empty else np.nan
    gm_sixm  = geometric_mean_of_series(ratio_df["sixm_ratio"]) if not ratio_df.empty else np.nan
    gm_twelvem = geometric_mean_of_series(ratio_df["twelvem_ratio"]) if not ratio_df.empty else np.nan

    metric_keys = list(PERF_METRICS.keys())
    incr_df = build_incremental_frame(df_2mile, df_1mile, cohort_map, metric_keys)

    # Default qi = median 2-mile IP90
    ip90_2m_vals = df_2mile["ip90_bpd"].dropna().values
    ip90_1m_vals = df_1mile["ip90_bpd"].dropna().values
    default_qi = float(np.median(ip90_2m_vals)) if len(ip90_2m_vals) > 0 else 150.0

    # Compute distributions for the type curve tab
    eur_1m_vals = df_1mile["eur_bbl"].dropna().values
    eur_2m_vals = df_2mile["eur_bbl"].dropna().values
    ip90_1m_vals = df_1mile["ip90_bpd"].dropna().values
    ip90_2m_vals_all = df_2mile["ip90_bpd"].dropna().values

    eur_2m_summary = empirical_summary(eur_2m_vals)
    ip90_2m_summary = empirical_summary(ip90_2m_vals_all)

    # --- Tabs ---
    tab_uplift, tab_curves = st.tabs([
        "📊 Uplift Analysis", "📈 Type Curves"
    ])

    # ==========================================================================
    # Tab 1: Uplift Analysis
    # ==========================================================================
    with tab_uplift:
        st.header("📊 Empirical Uplift (2-Mile vs 1-Mile)")
        mode_label = ("Range-Based" if analysis_mode.startswith("Mode A")
                      else f"Corridor (±{int(corridor_width)} m)")
        st.caption(
            f"Mode: **{mode_label}** · {len(df_2mile)} two-mile wells · "
            f"{len(df_1mile)} one-mile wells · "
            f"{sum(len(v) for v in cohort_map.values())} cohort links"
        )

        st.subheader("KPI Ratios (2mi / 1mi) — Geometric Mean")
        gm_cols = st.columns(4)
        gm_cols[0].metric("EUR Ratio GM", f"{gm_eur:.3f}" if np.isfinite(gm_eur) else "—")
        gm_cols[1].metric("IP90 Ratio GM", f"{gm_ip90:.3f}" if np.isfinite(gm_ip90) else "—")
        gm_cols[2].metric("6M Cal. Rate GM", f"{gm_sixm:.3f}" if np.isfinite(gm_sixm) else "—")
        gm_cols[3].metric("12M Cal. Rate GM", f"{gm_twelvem:.3f}" if np.isfinite(gm_twelvem) else "—")

        # Collapsible per-well ratios with actuals and baselines
        with st.expander("📋 Per-Well KPI Ratios (click to expand)", expanded=False):
            if not ratio_df.empty:
                display_cols = [
                    "uwi",
                    "ip90_2mi_actual", "ip90_1mi_baseline", "ip90_ratio",
                    "eur_2mi_actual", "eur_1mi_baseline", "eur_ratio",
                    "sixm_2mi_actual", "sixm_1mi_baseline", "sixm_ratio",
                    "twelvem_2mi_actual", "twelvem_1mi_baseline", "twelvem_ratio",
                ]
                display_cols = [c for c in display_cols if c in ratio_df.columns]
                st.dataframe(ratio_df[display_cols], use_container_width=True, hide_index=True)
            else:
                st.info("No ratio data available.")

        # Incremental bar charts (no P10/25/50/75/90 metrics above them)
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
                sub = incr_df.dropna(subset=[col_name])\
                             .sort_values(col_name, ascending=False)
                if not sub.empty:
                    bar_colors = [COLORS["incremental"] if v >= 0 else "#d62728"
                                  for v in sub[col_name]]
                    fig_wf = go.Figure(go.Bar(
                        x=sub["uwi"], y=sub[col_name],
                        marker_color=bar_colors,
                        text=[f"{v:+,.0f}" for v in sub[col_name]],
                        textposition="outside",
                    ))
                    fig_wf.add_hline(y=0, line=dict(color="gray", dash="dot"))
                    fig_wf.update_layout(
                        **PLOTLY_LAYOUT, height=400,
                        title=f"Incremental {PERF_METRICS[mk]} per 2-Mile Well",
                        yaxis_title=f"Δ {PERF_METRICS[mk]}",
                    )
                    st.plotly_chart(fig_wf, use_container_width=True)

        st.subheader("Full Incremental Results")
        disp_cols = ["uwi", "section_name", "n_comparators",
                     "eur_bbl", "sixm_bpd", "sixm_bpd_incremental",
                     "sixm_bpd_pct_uplift", "sixm_bpd_ratio",
                     "twelvem_bpd", "twelvem_bpd_incremental",
                     "twelvem_bpd_pct_uplift", "twelvem_bpd_ratio",
                     "waterflood_flag", "vintage_year"]
        disp_cols = [c for c in disp_cols if c in incr_df.columns]
        st.dataframe(incr_df[disp_cols], use_container_width=True, hide_index=True)

    # ==========================================================================
    # Tab 2: Type Curves
    # ==========================================================================
    with tab_curves:
        st.header("📈 Type Curve Builder")
        st.markdown(
            "Use the distributions below to understand the range of 1-mile and 2-mile "
            "performance, then build a custom type curve by selecting a percentile for "
            "**IP90** (sets qi) and **EUR** (sets the decline). The resulting curve is "
            "displayed alongside corporate reference type curves (RTCs)."
        )

        # --- Distribution Plots ---
        st.subheader("Performance Distributions — 1-Mile vs 2-Mile")

        col_a, col_b = st.columns(2)
        with col_a:
            fig_eur_dist = make_density_plot_with_percentiles(
                eur_1m_vals, eur_2m_vals,
                title="EUR Distribution",
                xaxis_title="EUR (bbl)",
            )
            st.plotly_chart(fig_eur_dist, use_container_width=True)
        with col_b:
            fig_ip90_dist = make_density_plot_with_percentiles(
                ip90_1m_vals, ip90_2m_vals_all,
                title="IP90 Distribution",
                xaxis_title="IP90 (bbl/d)",
            )
            st.plotly_chart(fig_ip90_dist, use_container_width=True)

        # Smoothed ECDFs: add right under the distribution plots
        st.subheader("Smoothed ECDF — EUR (1-Mile vs 2-Mile)")
        ecdf_eur = make_smoothed_ecdf_plot(eur_1m_vals, eur_2m_vals,
                                           title="Smoothed ECDF — EUR",
                                           xaxis_title="EUR (bbl)")
        st.plotly_chart(ecdf_eur, use_container_width=True)

        st.subheader("Smoothed ECDF — IP90 (1-Mile vs 2-Mile)")
        ecdf_ip90 = make_smoothed_ecdf_plot(ip90_1m_vals, ip90_2m_vals_all,
                                            title="Smoothed ECDF — IP90",
                                            xaxis_title="IP90 (bbl/d)")
        st.plotly_chart(ecdf_ip90, use_container_width=True)

        # Summary stats table
        with st.expander("📊 Distribution Summary Statistics", expanded=False):
            sum_rows = []
            for label, vals in [("1-Mile EUR (bbl)", eur_1m_vals),
                                ("2-Mile EUR (bbl)", eur_2m_vals),
                                ("1-Mile IP90 (bbl/d)", ip90_1m_vals),
                                ("2-Mile IP90 (bbl/d)", ip90_2m_vals_all)]:
                s = empirical_summary(vals)
                sum_rows.append({
                    "Metric": label, "n": s["n"],
                    "P90": f"{s['q90']:,.0f}", "P75": f"{s['q75']:,.0f}", "P50": f"{s['q50']:,.0f}", "P25": f"{s['q25']:,.0f}", "P10": f"{s['q10']:,.0f}",
                    "Mean": f"{s['mean']:,.0f}",
                })
            st.dataframe(pd.DataFrame(sum_rows), use_container_width=True, hide_index=True)

        st.divider()

        # --- Curve Builder ---
        st.subheader("🔧 Build Your Type Curve")
        st.markdown(
            "Select a percentile for **IP90** (determines qi) and a percentile for "
            "**EUR** (determines the decline). The percentiles reference the "
            "**2-mile** distributions shown above."
        )

        builder_cols = st.columns([1, 1, 2])
        with builder_cols[0]:
            ip90_pct = st.number_input(
                "IP90 Percentile (X) — P(X)",
                min_value=1, max_value=99, value=50, step=5,
                help="E.g. 50 means the median 2-mile IP90 will be used as qi."
            )
            qi_from_pct = float(np.percentile(ip90_2m_vals, ip90_pct)) if len(ip90_2m_vals) > 2 else default_qi
            st.metric("Resulting qi (bbl/d)", f"{qi_from_pct:,.1f}")

        with builder_cols[1]:
            eur_pct = st.number_input(
                "EUR Percentile (Y) — P(Y)",
                min_value=1, max_value=99, value=50, step=5,
                help="E.g. 50 means the median 2-mile EUR will be used as target EUR."
            )
            eur_from_pct = float(np.percentile(eur_2m_vals, eur_pct)) if len(eur_2m_vals) > 2 else 0.0
            st.metric("Resulting EUR (Mbbl)", f"{eur_from_pct/1000:,.1f}")

        # Build the user curve
        user_curve = None
        if qi_from_pct > 0 and eur_from_pct > 0:
            user_curve = build_curve_from_eur(qi_from_pct, eur_from_pct, B_FIXED, FLAT_DAYS, Q_LIMIT)

        with builder_cols[2]:
            if user_curve is not None:
                st.markdown("**Solved Decline Parameters**")
                di_daily = user_curve["di"]
                di_annual = di_daily * 365.25
                di_nominal_pct = (1.0 - (1.0 / (1.0 + B_FIXED * di_daily * 365.25) ** (1.0 / B_FIXED))) * 100 if di_daily > 0 else 0.0
                param_cols = st.columns(3)
                param_cols[0].metric("Di (1/day)", f"{di_daily:.6f}")
                param_cols[1].metric("Di (1/yr)", f"{di_annual:.4f}")
                param_cols[2].metric("b", f"{B_FIXED:.2f}")
                st.caption(f"Effective annual decline ≈ {di_nominal_pct:.1f}%")
            else:
                st.info("Enter valid percentiles to generate a curve.")

        # --- Rate-Time Plot ---
        st.subheader("Rate Type Curve — q(t)")
        fig_rate = go.Figure()

        # Plot user curve
        if user_curve is not None:
            fig_rate.add_trace(go.Scatter(
                x=user_curve["t"], y=user_curve["q"], mode="lines",
                line=dict(color="#e74c3c", width=3.5),
                name=f"Your Curve — P{ip90_pct} IP90 / P{eur_pct} EUR "
                     f"(qi={qi_from_pct:,.0f}, EUR={eur_from_pct/1000:,.0f} Mbbl)",
                hovertemplate="Your Curve<br>Day %{x:,.0f}<br>%{y:,.1f} bbl/d<extra></extra>",
            ))

        # Plot RTC curves
        if rtc_curves:
            for i, rtc in enumerate(rtc_curves):
                fig_rate.add_trace(go.Scatter(
                    x=rtc["t"], y=rtc["q"], mode="lines",
                    line=dict(color=RTC_PALETTE[i % len(RTC_PALETTE)],
                               width=2, dash="dash"),
                    name=f"RTC: {rtc['name']} (qi={rtc['qi']:,.0f}, EUR={rtc['eur_actual_bbl']/1000:,.0f} Mbbl)",
                    hovertemplate=f"RTC: {rtc['name']}<br>Day %{{x:,.0f}}<br>%{{y:,.1f}} bbl/d<extra></extra>",
                ))

        fig_rate.add_hline(y=Q_LIMIT, line_dash="dot", line_color="grey",
                           opacity=0.5,
                           annotation_text=f"{Q_LIMIT} bbl/d limit")
        fig_rate.update_layout(**PLOTLY_LAYOUT, height=550,
                               xaxis_title="Days since start of flat period",
                               yaxis_title="Rate (bbl/d)",
                               title="Rate — q(t)",
                               hovermode="closest")
        fig_rate.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_rate, use_container_width=True)

        # --- Cumulative Plot ---
        st.subheader("Cumulative Type Curve — N(t)")
        fig_cum = go.Figure()

        if user_curve is not None:
            fig_cum.add_trace(go.Scatter(
                x=user_curve["t"], y=user_curve["N"] / 1000, mode="lines",
                line=dict(color="#e74c3c", width=3.5),
                name=f"Your Curve — EUR={user_curve['eur_actual_bbl']/1000:,.0f} Mbbl",
                hovertemplate="Your Curve<br>Day %{x:,.0f}<br>%{y:,.1f} Mbbl<extra></extra>",
            ))

        if rtc_curves:
            for i, rtc in enumerate(rtc_curves):
                fig_cum.add_trace(go.Scatter(
                    x=rtc["t"], y=rtc["N"] / 1000, mode="lines",
                    line=dict(color=RTC_PALETTE[i % len(RTC_PALETTE)],
                               width=2, dash="dash"),
                    name=f"RTC: {rtc['name']} (EUR={rtc['eur_actual_bbl']/1000:,.0f} Mbbl)",
                    hovertemplate=f"RTC: {rtc['name']}<br>Day %{{x:,.0f}}<br>%{{y:,.1f}} Mbbl<extra></extra>",
                ))

        fig_cum.update_layout(**PLOTLY_LAYOUT, height=500,
                               xaxis_title="Days since start of flat period",
                               yaxis_title="Cumulative (Mbbl)",
                               title="Cumulative — N(t)",
                               hovermode="closest")
        fig_cum.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_cum, use_container_width=True)

        # --- RTC Benchmarking ---
        if rtc_curves and user_curve is not None:
            st.subheader("RTC Benchmarking Summary")
            bench_rows = []
            for rtc in rtc_curves:
                d_abs = (user_curve["eur_actual_bbl"] - rtc["eur_actual_bbl"]) / 1000
                d_pct = ((user_curve["eur_actual_bbl"] - rtc["eur_actual_bbl"])
                          / rtc["eur_actual_bbl"] * 100
                          if rtc["eur_actual_bbl"] else np.nan)
                qi_ratio = user_curve["qi"] / rtc["qi"] if rtc["qi"] > 0 else np.nan
                bench_rows.append({
                    "RTC Name":              rtc["name"],
                    "RTC qi (bbl/d)":        round(rtc["qi"], 1),
                    "RTC EUR (Mbbl)":        round(rtc["eur_actual_bbl"] / 1000, 1),
                    "RTC b":                 round(rtc["b"], 2),
                    "Your qi (bbl/d)":       round(user_curve["qi"], 1),
                    "Your EUR (Mbbl)":       round(user_curve["eur_actual_bbl"] / 1000, 1),
                    "ΔEUR (Mbbl)":           f"{d_abs:+,.1f}",
                    "ΔEUR (%)":             f"{d_pct:+.1f}%" if np.isfinite(d_pct) else "—",
                    "qi Ratio (You/RTC)":    f"{qi_ratio:.2f}" if np.isfinite(qi_ratio) else "—",
                })
            st.dataframe(pd.DataFrame(bench_rows), use_container_width=True, hide_index=True)

            # EUR bar comparison
            st.subheader("EUR Bar — RTC vs Your Curve")
            fig_eurbar = go.Figure()
            cats, vals, cols = [], [], []
            for i_rtc, rtc in enumerate(rtc_curves):
                cats.append(f"RTC: {rtc['name']}")
                vals.append(rtc["eur_actual_bbl"] / 1000)
                cols.append(RTC_PALETTE[i_rtc % len(RTC_PALETTE)])
            cats.append(f"Your Curve (P{ip90_pct}/P{eur_pct})")
            vals.append(user_curve["eur_actual_bbl"] / 1000)
            cols.append("#e74c3c")
            fig_eurbar.add_trace(go.Bar(x=cats, y=vals, marker_color=cols,
                                         text=[f"{v:,.0f}" for v in vals],
                                         textposition="outside"))
            fig_eurbar.update_layout(**PLOTLY_LAYOUT, height=400,
                                     yaxis_title="EUR (Mbbl)",
                                     title="EUR Comparison",
                                     hovermode="closest")
            st.plotly_chart(fig_eurbar, use_container_width=True)

        # --- Export ---
        st.divider()
        st.subheader("📥 Export")

        export_data = {}
        if user_curve is not None:
            cdf = pd.DataFrame({
                "day":      user_curve["t"],
                "rate_bpd": user_curve["q"],
                "cum_bbl":  user_curve["N"],
                "cum_Mbbl": user_curve["N"] / 1000,
            })
            export_data["Your_Curve"] = cdf

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            # Curve data
            if user_curve is not None:
                params_df = pd.DataFrame([{
                    "qi_bpd":            user_curve["qi"],
                    "di_per_day":        user_curve["di"],
                    "di_per_year":       user_curve["di"] * 365.25,
                    "b":                 user_curve["b"],
                    "eur_target_bbl":    user_curve["eur_target_bbl"],
                    "eur_actual_bbl":    user_curve["eur_actual_bbl"],
                    "eur_actual_Mbbl":   user_curve["eur_actual_bbl"] / 1000,
                    "flat_days":         user_curve["flat_days"],
                    "ip90_percentile":   ip90_pct,
                    "eur_percentile":    eur_pct,
                }])
                params_df.to_excel(writer, index=False, sheet_name="Curve_Parameters")
                cdf.to_excel(writer, index=False, sheet_name="Curve_Data")

            if not incr_df.empty:
                export_incr = incr_df.copy()
                for c in export_incr.columns:
                    if c.endswith("_bbl") or c == "eur_bbl":
                        mbbl_col = c.replace("_bbl", "_Mbbl")
                        export_incr[mbbl_col] = export_incr[c] / 1000
                export_incr.to_excel(writer, index=False,
                                      sheet_name="Incremental_Results")

            if not ratio_df.empty:
                ratio_df.to_excel(writer, index=False, sheet_name="Per_Well_Ratios")

            cohort_rows = [{"uwi_2mile": u, "uwi_1mile_comparator": w}
                           for u, lst in cohort_map.items() for w in lst]
            if cohort_rows:
                pd.DataFrame(cohort_rows).to_excel(writer, index=False,
                                                    sheet_name="Cohort_Mappings")

            if rtc_curves:
                rtc_rows = [{
                    "name": r["name"],
                    "qi_bpd":     r["qi"],
                    "di_per_day": r["di"],
                    "b":          r["b"],
                    "eur_bbl":    r["eur_actual_bbl"],
                    "eur_Mbbl":   r["eur_actual_bbl"] / 1000,
                } for r in rtc_curves]
                pd.DataFrame(rtc_rows).to_excel(writer, index=False,
                                                  sheet_name="RTC_Reference")

        st.download_button(
            "📥 Download Full Results (Excel)",
            buf.getvalue(),
            file_name="2mile_type_curves_uplift.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        if user_curve is not None:
            st.download_button(
                "📥 Download Curve Data (CSV)",
                cdf.to_csv(index=False).encode(),
                file_name="2mile_type_curve.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()