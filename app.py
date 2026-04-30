"""
Combined 2-Mile Production Type Curve Generator
================================================
Merges App 1 (uplift evaluator), App 2 (decline fitting), and App 3 (curve solver/RTC)
into a single Streamlit application that produces P25 / P50 / P75 type curves for
2-mile wells using 1-mile analogs.

Input files (must be in the working directory):
  - w.xlsx          : Well table + section OOIP (App 1)
  - 1M.shp / 2M.shp: Lateral geometries (App 1)
  - tcgenprod.xlsx  : Monthly production data (App 2)
  - rtc.xlsx        : Corporate reference type curves (App 3)

Run:  streamlit run combined_type_curve_app.py
"""

import io
import math
import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import streamlit as st

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="2-Mile Type Curve Generator (P25/P50/P75)",
    page_icon="🛢️",
    layout="wide",
)

# File paths
XLSX_PATH = "w.xlsx"
SHP_1M_PATH = "1M.shp"
SHP_2M_PATH = "2M.shp"
PROD_DATA_FILE = "tcgenprod.xlsx"
RTC_XLSX_PATH = "rtc.xlsx"

# Spatial / physical constants
TARGET_METRIC_CRS = "EPSG:3347"
WATERFLOOD_BUFFER_M = 200.0
MILE_TO_M = 1609.34
CORRIDOR_HALF_WIDTH_M = 900.0

# Decline constants
Q_LIMIT = 2.0           # bbl/d economic limit
FLAT_MONTHS = 1.44
FLAT_DAYS = int(round(FLAT_MONTHS * 30.4375))  # ≈44 days
B_DEFAULT = 0.95

# Display
COLORS = {
    "1-Mile": "#1f77b4", "2-Mile": "#ff7f0e", "incremental": "#2ca02c",
    "P25": "#2ecc71", "P50": "#3498db", "P75": "#e74c3c",
}

PERF_METRICS = {
    "eur_bbl": "EUR (bbl)",
    "ip30_bpd": "IP30 (bbl/d)",
    "ip90_bpd": "IP90 (bbl/d)",
    "cum6_bbl": "6-Month Cum (bbl)",
    "cum12_bbl": "12-Month Cum (bbl)",
}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Arial, sans-serif", size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    hovermode="x unified",
)

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _std_uwi(s: pd.Series) -> pd.Series:
    """Standardize UWI strings: strip whitespace, uppercase, NaN-ify blanks."""
    return (
        s.astype(str)
        .str.strip()
        .str.upper()
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


# ═══════════════════════════════════════════════════════════════════════════════
# DECLINE MATH (from Apps 2 & 3, unified)
# ═══════════════════════════════════════════════════════════════════════════════

def hyp_rate(t, qi, di, b):
    """Arps hyperbolic rate at time t (days)."""
    t = np.asarray(t, dtype=float)
    if b == 0:
        return qi * np.exp(-di * t)
    with np.errstate(divide="ignore", invalid="ignore"):
        return qi / np.power(1.0 + b * di * t, 1.0 / b)


def cum_arps(t, qi, di, b):
    """Arps analytical cumulative production."""
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
    """Time (days) for rate to decline from qi to q_target."""
    if q_target <= 0 or di <= 0:
        return np.inf
    if b == 0:
        return max(0.0, math.log(qi / q_target) / di)
    return max(0.0, ((qi / q_target) ** b - 1.0) / (b * di))


def find_Di_for_eur_post(qi, eur_post, b, q_target, tol=1e-6):
    """
    Bisection solver (App 3): find Di such that cumulative production
    from qi down to q_target equals eur_post (bbl).
    """
    if eur_post <= 0:
        return 0.0
    if qi <= q_target:
        return None

    def _residual(di):
        t_end = t_to_rate(di, qi, b, q_target)
        if np.isinf(t_end):
            return np.inf
        return float(cum_arps(t_end, qi, di, b) - eur_post)

    lo, hi = 1e-12, 1.0
    for _ in range(300):
        f_lo = _residual(lo)
        f_hi = _residual(hi)
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


def fit_hyperbolic(t_days, q_vals, b_fixed=B_DEFAULT):
    """Fit qi and di with fixed b on post-peak production data."""
    t = np.asarray(t_days, dtype=float)
    q = np.asarray(q_vals, dtype=float)
    mask = (t > 0) & (q > 0) & np.isfinite(t) & np.isfinite(q)
    t, q = t[mask], q[mask]
    if len(t) < 4:
        return None
    qi0 = float(q[0])
    try:
        popt, _ = curve_fit(
            lambda tt, qi, di: hyp_rate(tt, qi, di, b_fixed),
            t, q,
            p0=[qi0, 0.003],
            bounds=([0.1, 1e-8], [qi0 * 5, 0.1]),
            maxfev=30000,
        )
        return (popt[0], popt[1], b_fixed)
    except Exception:
        return None


def build_piecewise_curve(qi, di, b, flat_days=FLAT_DAYS, q_limit=Q_LIMIT):
    """
    Build a full piecewise type curve: flat period at qi, then Arps decline
    until q_limit. Returns (t_days_array, q_array).
    """
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
    """
    Given target EUR (bbl), qi, b → solve for Di, then build piecewise curve.
    Returns dict with t, q, N (cumulative), qi, di, b, eur.
    """
    eur_flat = qi * flat_days
    eur_post = max(eur_bbl - eur_flat, 0.0)

    di = None
    if eur_post > 0 and qi > q_limit:
        di = find_Di_for_eur_post(qi, eur_post, b, q_limit)

    t_arr, q_arr = build_piecewise_curve(qi, di if di else 0.0, b, flat_days, q_limit)

    # Cumulative via trapezoidal integration for consistency
    N_arr = np.zeros_like(t_arr)
    if len(t_arr) > 1:
        for i in range(1, len(t_arr)):
            dt = t_arr[i] - t_arr[i - 1]
            N_arr[i] = N_arr[i - 1] + 0.5 * (q_arr[i - 1] + q_arr[i]) * dt

    actual_eur = float(N_arr[-1]) if len(N_arr) > 0 else 0.0

    return {
        "t": t_arr,
        "q": q_arr,
        "N": N_arr,
        "qi": qi,
        "di": di if di else 0.0,
        "b": b,
        "eur_target_bbl": eur_bbl,
        "eur_actual_bbl": actual_eur,
    }


def calc_eur_from_arrays(t_arr, q_arr):
    if len(t_arr) < 2:
        return 0.0
    return float(np.trapezoid(q_arr, t_arr))


# ═══════════════════════════════════════════════════════════════════════════════
# DATA INGESTION — APP 1 (well table, OOIP, geometries)
# ═══════════════════════════════════════════════════════════════════════════════

REQUIRED_WELL_COLUMNS = [
    "UWI", "Section Name", "Well Type", "Hz Length (m)",
    "Oil + Cond: EUR (Mbbl)", "Oil + Cond: IP 30 Cal. Rate (bbl/d)",
    "Oil + Cond: IP 90 Cal. Rate (bbl/d)", "Oil + Cond: 6M CalTime Cum (Mbbl)",
    "Oil + Cond: 12M CalTime Cum (Mbbl)", "Objective", "On Prod Date",
    "On Inj Date", "FOOZ",
]

RENAME_WELL = {
    "UWI": "uwi",
    "Section Name": "section_name",
    "Well Type": "well_type",
    "Hz Length (m)": "hz_length_m",
    "Oil + Cond: EUR (Mbbl)": "eur_mbbl",
    "Oil + Cond: IP 30 Cal. Rate (bbl/d)": "ip30_bpd",
    "Oil + Cond: IP 90 Cal. Rate (bbl/d)": "ip90_bpd",
    "Oil + Cond: 6M CalTime Cum (Mbbl)": "cum6_mbbl",
    "Oil + Cond: 12M CalTime Cum (Mbbl)": "cum12_mbbl",
    "Objective": "objective",
    "On Prod Date": "on_prod_date",
    "On Inj Date": "on_inj_date",
    "FOOZ": "fooz",
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
    # Convert Mbbl → bbl for all cumulative/EUR columns at ingestion
    for c_mbbl, c_bbl in [
        ("eur_mbbl", "eur_bbl"),
        ("cum6_mbbl", "cum6_bbl"),
        ("cum12_mbbl", "cum12_bbl"),
    ]:
        if c_mbbl in out.columns:
            out[c_bbl] = pd.to_numeric(out[c_mbbl], errors="coerce") * 1000.0
    for c in ["hz_length_m", "ip30_bpd", "ip90_bpd", "eur_bbl", "cum6_bbl", "cum12_bbl"]:
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
    require_cols(raw, ["Section", "OOIP"], f"`{XLSX_PATH}` sheet 1")
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
    gdf = gdf.sort_values("lateral_group", ascending=False).drop_duplicates("uwi", keep="first")
    gdf["midpoint"] = gdf.geometry.apply(
        lambda g: g.interpolate(0.5, normalized=True)
        if g is not None and not g.is_empty else None
    )
    return gdf[["uwi", "lateral_group", "geometry", "midpoint"]].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA INGESTION — APP 2 (production time series)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_production_data():
    if not os.path.exists(PROD_DATA_FILE):
        return None, f"`{PROD_DATA_FILE}` not found."
    df = pd.read_excel(PROD_DATA_FILE)
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if "uwi" in cl:
            rename[c] = "uwi"
        elif "month" in cl or "date" in cl:
            rename[c] = "month"
        elif "bbl" in cl or "rate" in cl:
            rename[c] = "rate"
    df = df.rename(columns=rename)
    for req in ["uwi", "month", "rate"]:
        if req not in df.columns:
            return None, f"`{PROD_DATA_FILE}` missing column `{req}`."
    df["uwi"] = _std_uwi(df["uwi"])
    df["date"] = pd.to_datetime(df["month"], errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna(subset=["date", "rate", "uwi"])
    df = df[df["rate"] >= 0].copy()
    df = df.sort_values(["uwi", "date"]).reset_index(drop=True)
    df["days_in_month"] = df["date"].dt.days_in_month
    df["monthly_vol"] = df["rate"] * df["days_in_month"]
    return df, None


# ═══════════════════════════════════════════════════════════════════════════════
# DATA INGESTION — APP 3 (corporate RTC curves)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_rtc_curves():
    """Load corporate reference type curves from rtc.xlsx and solve each."""
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
            name = str(row["Name"])
            qi_rtc = float(row["Qi"])
            b_rtc = float(row["b"])
            eur_rtc = float(row["EUR"])  # already in bbl
            flat_days_rtc = float(row["Months"]) * 30.0

            result = build_curve_from_eur(qi_rtc, eur_rtc, b_rtc, int(flat_days_rtc), Q_LIMIT)
            result["name"] = name
            curves.append(result)
        except Exception:
            continue
    return curves


# ═══════════════════════════════════════════════════════════════════════════════
# SPATIAL COMPUTATION (from App 1)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def compute_spatial_features(_geoms_gdf, _well_meta):
    if _geoms_gdf.empty:
        return pd.DataFrame(columns=[
            "uwi", "nearest_producer_m", "nearest_injector_m",
            "n_within_400m", "n_within_800m", "waterflood_flag",
        ])
    g = _geoms_gdf.copy().reset_index(drop=True)
    meta = _well_meta[["uwi", "on_prod_date", "on_inj_date", "objective", "is_injector"]].copy()
    g = g.merge(meta, on="uwi", how="left")
    g["is_injector"] = g["is_injector"].fillna(False).astype(bool)
    sindex = g.sindex
    rows = []
    for i, row in g.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            rows.append(dict(uwi=row["uwi"], nearest_producer_m=np.nan,
                             nearest_injector_m=np.nan, n_within_400m=0,
                             n_within_800m=0, waterflood_flag=False))
            continue
        cand = [j for j in sindex.intersection(geom.buffer(5000).bounds) if j != i]
        nearest_prod, nearest_inj = np.nan, np.nan
        n400, n800 = 0, 0
        for j in cand:
            og = g.iloc[j].geometry
            if og is None or og.is_empty:
                continue
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
                if o.geometry is None or o.geometry.is_empty or not buf.intersects(o.geometry):
                    continue
                inj_d = o.get("on_inj_date", pd.NaT)
                prod_d = row.get("on_prod_date", pd.NaT)
                if pd.notna(inj_d) and pd.notna(prod_d) and inj_d <= prod_d:
                    wf = True; break
        rows.append(dict(
            uwi=row["uwi"],
            nearest_producer_m=float(nearest_prod) if np.isfinite(nearest_prod) else np.nan,
            nearest_injector_m=float(nearest_inj) if np.isfinite(nearest_inj) else np.nan,
            n_within_400m=int(n400), n_within_800m=int(n800), waterflood_flag=bool(wf),
        ))
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALOG MATCHING (from App 1)
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# INCREMENTAL UPLIFT COMPUTATION (App 1)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_incremental(well_row, comparator_df, metric_keys):
    """Compute incremental for one 2-mile well vs comparator median. All in bbl."""
    result = {"uwi": well_row["uwi"]}
    for mk in metric_keys:
        val = well_row.get(mk, np.nan)
        comp = comparator_df[mk].dropna() if mk in comparator_df.columns else pd.Series(dtype=float)
        if comp.empty or pd.isna(val):
            result[f"{mk}_baseline"] = np.nan
            result[f"{mk}_incremental"] = np.nan
            result[f"{mk}_pct_uplift"] = np.nan
        else:
            bl = float(comp.median())
            incr = float(val) - bl
            pct = (incr / bl * 100) if bl != 0 else np.nan
            result[f"{mk}_baseline"] = bl
            result[f"{mk}_incremental"] = incr
            result[f"{mk}_pct_uplift"] = pct
    result["n_comparators"] = len(comparator_df)
    return result


def empirical_summary(values):
    vals = np.array([v for v in values if np.isfinite(v)], dtype=float)
    n = len(vals)
    if n == 0:
        return dict(n=0, median=np.nan, mean=np.nan, std=np.nan,
                    min=np.nan, q25=np.nan, q50=np.nan, q75=np.nan, max=np.nan)
    return dict(
        n=n,
        median=float(np.median(vals)),
        mean=float(np.mean(vals)),
        std=float(np.std(vals, ddof=1)) if n > 1 else np.nan,
        min=float(np.min(vals)),
        q25=float(np.percentile(vals, 25)),
        q50=float(np.percentile(vals, 50)),
        q75=float(np.percentile(vals, 75)),
        max=float(np.max(vals)),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PER-WELL DECLINE ANALYSIS (App 2)
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_well_production(df_w, b_fixed=B_DEFAULT):
    """Fit decline on a single well's production time series."""
    df_w = df_w.sort_values("date").reset_index(drop=True)
    peak_idx = df_w["rate"].idxmax()
    peak_rate = df_w.loc[peak_idx, "rate"]
    peak_date = df_w.loc[peak_idx, "date"]
    df_w["t_days"] = (df_w["date"] - peak_date).dt.days.astype(float)
    df_decline = df_w[df_w["t_days"] > 0].copy()

    hyp = fit_hyperbolic(df_decline["t_days"].values, df_decline["rate"].values, b_fixed)
    if hyp:
        qi_h, di_h, b_h = hyp
        pred = hyp_rate(df_decline["t_days"].values, qi_h, di_h, b_h)
        ss_res = np.sum((df_decline["rate"].values - pred) ** 2)
        ss_tot = np.sum((df_decline["rate"].values - df_decline["rate"].mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        qi_h, di_h, b_h, r2 = peak_rate, 0.003, b_fixed, 0.0

    eur_trap = calc_eur_from_arrays(
        (df_w["date"] - df_w["date"].min()).dt.days.values.astype(float),
        df_w["rate"].values,
    )

    return dict(
        uwi=df_w["uwi"].iloc[0],
        peak_rate=peak_rate, peak_date=peak_date,
        qi=qi_h, di=di_h, b=b_h, r2=r2,
        eur_trap_bbl=eur_trap,
        n_months=len(df_w),
        df_post=df_w[df_w["t_days"] >= 0].copy(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER DATA LOADING & ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading well table & geometries…")
def load_and_assemble_wells():
    well_raw = load_well_table()
    sec_ooip = load_section_ooip()
    geoms = load_geometries()

    membership = geoms[["uwi", "lateral_group"]].drop_duplicates()
    well_df = membership.merge(well_raw, on="uwi", how="left")
    well_df = well_df.merge(sec_ooip, on="section_name", how="left")
    well_df = well_df.replace([np.inf, -np.inf], np.nan)
    for c in ["is_injector", "is_fooz"]:
        if c not in well_df.columns:
            well_df[c] = False
        well_df[c] = well_df[c].fillna(False).astype(bool)

    # Spatial features
    meta = well_df[["uwi", "on_prod_date", "on_inj_date", "objective", "is_injector"]].copy()
    spatial = compute_spatial_features(geoms, meta)
    well_df = well_df.merge(spatial, on="uwi", how="left")

    # Exclude FOOZ
    well_df = well_df[~well_df["is_fooz"]].reset_index(drop=True)

    return well_df, geoms


@st.cache_data(show_spinner="Fitting decline parameters on production data…")
def fit_all_declines(b_fixed):
    prod_df, err = load_production_data()
    if prod_df is None:
        return {}, err
    results = {}
    for uwi, grp in prod_df.groupby("uwi"):
        if len(grp) < 3:
            continue
        results[uwi] = analyse_well_production(grp, b_fixed)
    return results, None


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("🛢️ 2-Mile Production Type Curves — P25 / P50 / P75")
    st.caption(
        "Combines field data & uplift analysis (App 1), decline fitting (App 2), "
        "and the EUR-constrained curve solver (App 3) into a single reproducible workflow."
    )

    # ── Load data ──────────────────────────────────────────────────────────
    well_df, geoms = load_and_assemble_wells()
    df_1mile = well_df[well_df["lateral_group"] == "1-Mile"].reset_index(drop=True)
    df_2mile = well_df[well_df["lateral_group"] == "2-Mile"].reset_index(drop=True)
    rtc_curves = load_rtc_curves()

    # ── Sidebar ────────────────────────────────────────────────────────────
    st.sidebar.title("⚙️ Configuration")

    analysis_mode = st.sidebar.radio(
        "Analog Matching Mode",
        ["Mode A: Range-Based Analog Matching", "Mode B: Geometric Corridor"],
    )

    st.sidebar.divider()

    # Matching parameters
    active_features = []
    tolerances = {}
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
        active_features = st.sidebar.multiselect("Active features", available, default=available)
        if "section_ooip" in active_features:
            tolerances["section_ooip"] = st.sidebar.number_input(
                "± OOIP tol", 0.0, 1e6, 250000.0, 1e5)
        if "nearest_producer_m" in active_features:
            tolerances["nearest_producer_m"] = st.sidebar.number_input(
                "± Nearest prod tol (m)", 0.0, 5000.0, 150.0, 50.0)
        if "nearest_injector_m" in active_features:
            tolerances["nearest_injector_m"] = st.sidebar.number_input(
                "± Nearest inj tol (m)", 0.0, 5000.0, 150.0, 50.0)
    else:
        st.sidebar.subheader("Corridor Parameters")
        corridor_width = st.sidebar.number_input(
            "Corridor half-width (m)", 100.0, 3000.0, CORRIDOR_HALF_WIDTH_M, 100.0)

    st.sidebar.divider()
    b_fixed = st.sidebar.number_input("b (Arps exponent, fixed)", 0.01, 2.0, B_DEFAULT, 0.05)

    st.sidebar.divider()
    show_curves = st.sidebar.multiselect(
        "Display curves", ["P25", "P50", "P75"], default=["P25", "P50", "P75"])

    st.sidebar.divider()
    overlay_rtc = st.sidebar.checkbox("Overlay corporate RTC curves", value=True)

    # ── Decline fitting (App 2) ────────────────────────────────────────────
    decline_results, prod_err = fit_all_declines(b_fixed)
    if prod_err:
        st.warning(f"Production data issue: {prod_err}. Decline fitting skipped; "
                    "qi will be estimated from well-table IP30.")

    # ── Incremental uplift computation (App 1) ─────────────────────────────
    metric_keys = list(PERF_METRICS.keys())
    incr_records = []
    well_cohort_map = {}

    for _, row2 in df_2mile.iterrows():
        uwi2 = row2["uwi"]
        if analysis_mode.startswith("Mode A"):
            matched = range_match(row2, df_1mile, tolerances, active_features)
        else:
            gr = geoms[geoms["uwi"] == uwi2]
            matched = corridor_match(gr.iloc[0].geometry, geoms, corridor_width) if not gr.empty else []
        comparator_df = df_1mile[df_1mile["uwi"].isin(matched)]
        well_cohort_map[uwi2] = matched
        rec = compute_incremental(row2, comparator_df, metric_keys)
        # Carry forward useful fields
        for col in ["section_name", "hz_length_m", "on_prod_date", "vintage_year",
                     "eur_bbl", "ip30_bpd", "ip90_bpd", "cum6_bbl", "cum12_bbl",
                     "waterflood_flag", "section_ooip"]:
            rec[col] = row2.get(col, np.nan)
        incr_records.append(rec)

    incr_df = pd.DataFrame(incr_records) if incr_records else pd.DataFrame(columns=["uwi"])

    # ── Define P25 / P50 / P75 EUR targets ─────────────────────────────────
    # Strategy: for each 2-mile well that has both an incremental EUR and a
    # 1-mile baseline, the *expected* 2-mile EUR = baseline + incremental.
    # The distribution of these expected 2-mile EURs defines the percentiles.
    # This captures both the baseline variability and the uplift variability.

    # We prefer to use the actual 2-mile EUR where available
    eur_2mi_values = df_2mile["eur_bbl"].dropna().values

    # If we have actual 2-mile EURs, use those directly for the distribution
    # (this is the most honest representation of 2-mile outcomes)
    if len(eur_2mi_values) >= 3:
        eur_dist = eur_2mi_values
        eur_source = "actual 2-mile EUR distribution"
    elif "eur_bbl_incremental" in incr_df.columns:
        # Fallback: baseline + incremental
        valid = incr_df.dropna(subset=["eur_bbl_baseline", "eur_bbl_incremental"])
        eur_dist = (valid["eur_bbl_baseline"] + valid["eur_bbl_incremental"]).values
        eur_source = "baseline + incremental EUR distribution"
    else:
        eur_dist = np.array([])
        eur_source = "insufficient data"

    eur_summary = empirical_summary(eur_dist)

    # P25 = lower outcome, P75 = higher outcome (budget framing)
    eur_p25 = eur_summary["q25"]
    eur_p50 = eur_summary["q50"]
    eur_p75 = eur_summary["q75"]

    # ── Determine qi anchor ────────────────────────────────────────────────
    # Use median fitted qi from App 2's decline fits on 2-mile wells
    twomile_uwis = set(df_2mile["uwi"].dropna())
    fitted_qi_vals = []
    fitted_di_vals = []
    fitted_wells = []
    for uwi, res in decline_results.items():
        if uwi in twomile_uwis:
            fitted_qi_vals.append(res["qi"])
            fitted_di_vals.append(res["di"])
            fitted_wells.append(uwi)

    if fitted_qi_vals:
        qi_anchor = float(np.median(fitted_qi_vals))
        qi_source = f"median of {len(fitted_qi_vals)} fitted 2-mile wells"
    else:
        # Fallback: use IP30 from well table
        ip30_vals = df_2mile["ip30_bpd"].dropna().values
        qi_anchor = float(np.median(ip30_vals)) if len(ip30_vals) > 0 else 150.0
        qi_source = "median IP30 from well table (no production fits available)"

    # ── Generate final P25/P50/P75 curves using App 3 solver ──────────────
    pct_targets = {"P25": eur_p25, "P50": eur_p50, "P75": eur_p75}
    final_curves = {}

    for label, eur_target in pct_targets.items():
        if np.isfinite(eur_target) and eur_target > 0:
            final_curves[label] = build_curve_from_eur(
                qi_anchor, eur_target, b_fixed, FLAT_DAYS, Q_LIMIT
            )
        else:
            final_curves[label] = None

    # ══════════════════════════════════════════════════════════════════════
    # DISPLAY
    # ══════════════════════════════════════════════════════════════════════

    # ── Tab structure ──────────────────────────────────────────────────────
    tab_curves, tab_uplift, tab_params, tab_well, tab_export = st.tabs([
        "📈 Type Curves", "📊 Uplift Analysis", "🔧 Fitted Parameters",
        "🔍 Well-by-Well", "📥 Export",
    ])

    # ──────────────────────────────────────────────────────────────────────
    # TAB 1: TYPE CURVES
    # ──────────────────────────────────────────────────────────────────────
    with tab_curves:
        st.header("Production Type Curves — P25 / P50 / P75")

        # Summary metrics
        cols = st.columns(5)
        cols[0].metric("2-Mile Wells", len(df_2mile))
        cols[1].metric("EUR Source", eur_source.split(" ")[0])
        cols[2].metric("qi Anchor (bbl/d)", f"{qi_anchor:,.0f}")
        cols[3].metric("b (fixed)", f"{b_fixed:.2f}")
        cols[4].metric("Wells with Fits", len(fitted_wells))

        # Parameter table
        st.subheader("Curve Parameters")
        param_rows = []
        for label in ["P25", "P50", "P75"]:
            c = final_curves.get(label)
            if c is None:
                param_rows.append({"Curve": label, "qi (bbl/d)": "—", "Di (1/d)": "—",
                                   "Di (1/yr)": "—", "b": "—",
                                   "EUR Target (Mbbl)": "—", "EUR Solved (Mbbl)": "—"})
            else:
                param_rows.append({
                    "Curve": label,
                    "qi (bbl/d)": f"{c['qi']:,.1f}",
                    "Di (1/d)": f"{c['di']:.6f}",
                    "Di (1/yr)": f"{c['di'] * 365.25:.4f}",
                    "b": f"{c['b']:.2f}",
                    "EUR Target (Mbbl)": f"{c['eur_target_bbl'] / 1000:,.1f}",
                    "EUR Solved (Mbbl)": f"{c['eur_actual_bbl'] / 1000:,.1f}",
                })
        st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)

        # ── Rate plot ──────────────────────────────────────────────────────
        st.subheader("Rate Type Curves — q(t)")
        fig_rate = go.Figure()

        # Background: individual 2-mile well post-peak data
        n_bg = 0
        for uwi in fitted_wells:
            res = decline_results[uwi]
            dfp = res.get("df_post")
            if dfp is not None and not dfp.empty:
                fig_rate.add_trace(go.Scatter(
                    x=dfp["t_days"], y=dfp["rate"],
                    mode="lines", line=dict(color="grey", width=0.6),
                    opacity=0.25, showlegend=(n_bg == 0),
                    name="2-Mile wells (observed)" if n_bg == 0 else None,
                    hoverinfo="skip",
                ))
                n_bg += 1

        # P-curves
        for label in ["P25", "P50", "P75"]:
            if label not in show_curves:
                continue
            c = final_curves.get(label)
            if c is None:
                continue
            fig_rate.add_trace(go.Scatter(
                x=c["t"], y=c["q"], mode="lines",
                line=dict(color=COLORS[label], width=3),
                name=f"{label} — EUR={c['eur_actual_bbl']/1000:,.0f} Mbbl",
                hovertemplate=f"{label}<br>Day %{{x:,.0f}}<br>%{{y:,.1f}} bbl/d",
            ))

        # RTC overlay
        rtc_palette = [
            "#8e44ad", "#e67e22", "#1abc9c", "#c0392b",
            "#2980b9", "#7f8c8d", "#d35400", "#27ae60",
        ]
        if overlay_rtc and rtc_curves:
            for i, rtc in enumerate(rtc_curves):
                fig_rate.add_trace(go.Scatter(
                    x=rtc["t"], y=rtc["q"], mode="lines",
                    line=dict(color=rtc_palette[i % len(rtc_palette)], width=2, dash="dash"),
                    name=f"RTC: {rtc['name']}",
                ))

        fig_rate.add_hline(y=Q_LIMIT, line_dash="dot", line_color="grey", opacity=0.5,
                           annotation_text=f"{Q_LIMIT} bbl/d limit")
        fig_rate.update_layout(
            **PLOTLY_LAYOUT, height=550,
            xaxis_title="Days since peak / start of flat period",
            yaxis_title="Rate (bbl/d)",
            title="2-Mile Type Curves — Rate",
        )
        fig_rate.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_rate, use_container_width=True)

        # ── Cumulative plot ────────────────────────────────────────────────
        st.subheader("Cumulative Type Curves — N(t)")
        fig_cum = go.Figure()

        for label in ["P25", "P50", "P75"]:
            if label not in show_curves:
                continue
            c = final_curves.get(label)
            if c is None:
                continue
            fig_cum.add_trace(go.Scatter(
                x=c["t"], y=c["N"] / 1000, mode="lines",
                line=dict(color=COLORS[label], width=3),
                name=f"{label} — EUR={c['eur_actual_bbl']/1000:,.0f} Mbbl",
                hovertemplate=f"{label}<br>Day %{{x:,.0f}}<br>%{{y:,.1f}} Mbbl",
            ))

        if overlay_rtc and rtc_curves:
            for i, rtc in enumerate(rtc_curves):
                fig_cum.add_trace(go.Scatter(
                    x=rtc["t"], y=rtc["N"] / 1000, mode="lines",
                    line=dict(color=rtc_palette[i % len(rtc_palette)], width=2, dash="dash"),
                    name=f"RTC: {rtc['name']}",
                ))

        fig_cum.update_layout(
            **PLOTLY_LAYOUT, height=500,
            xaxis_title="Days since peak / start of flat period",
            yaxis_title="Cumulative (Mbbl)",
            title="2-Mile Type Curves — Cumulative",
        )
        fig_cum.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_cum, use_container_width=True)

        # ── EUR distribution context ───────────────────────────────────────
        st.subheader("EUR Distribution Context")
        if len(eur_dist) > 0:
            fig_eur = go.Figure()
            fig_eur.add_trace(go.Box(
                y=eur_dist / 1000, name="2-Mile EUR (Mbbl)",
                boxpoints="all", jitter=0.5, pointpos=0,
                marker=dict(color=COLORS["2-Mile"], size=9),
                line=dict(color="#e67e22"),
                fillcolor="rgba(255,127,14,0.2)",
                boxmean=True,
            ))
            for label, val in [("P25", eur_p25), ("P50", eur_p50), ("P75", eur_p75)]:
                if np.isfinite(val):
                    fig_eur.add_hline(
                        y=val / 1000, line_dash="dash",
                        line_color=COLORS[label], line_width=2,
                        annotation_text=f"{label}: {val/1000:,.0f} Mbbl",
                        annotation_position="top right",
                    )
            fig_eur.update_layout(
                **PLOTLY_LAYOUT, height=420,
                title=f"2-Mile EUR Distribution (n={len(eur_dist)}, source: {eur_source})",
                yaxis_title="EUR (Mbbl)",
            )
            st.plotly_chart(fig_eur, use_container_width=True)
        else:
            st.warning("No EUR distribution data available.")

    # ──────────────────────────────────────────────────────────────────────
    # TAB 2: UPLIFT ANALYSIS (App 1 core output)
    # ──────────────────────────────────────────────────────────────────────
    with tab_uplift:
        st.header("📊 Incremental Uplift Analysis (2-Mile vs 1-Mile)")
        mode_label = ("Range-Based Analog Matching" if analysis_mode.startswith("Mode A")
                      else f"Geometric Corridor (±{int(corridor_width)} m)")
        st.caption(f"Mode: **{mode_label}** · {len(df_2mile)} two-mile wells · {len(df_1mile)} one-mile wells")

        # Summary stats
        for mk in metric_keys:
            col_name = f"{mk}_incremental"
            if col_name not in incr_df.columns:
                continue
            vals = incr_df[col_name].dropna().values
            if len(vals) == 0:
                continue
            s = empirical_summary(vals)
            with st.expander(f"**{PERF_METRICS[mk]}** — incremental distribution (n={s['n']})", expanded=(mk == "eur_bbl")):
                mc = st.columns(5)
                mc[0].metric("Median", f"{s['median']:+,.0f}")
                mc[1].metric("Mean", f"{s['mean']:+,.0f}")
                mc[2].metric("Q25", f"{s['q25']:+,.0f}")
                mc[3].metric("Q75", f"{s['q75']:+,.0f}")
                mc[4].metric("Std", f"{s['std']:,.0f}" if np.isfinite(s["std"]) else "—")

                # Waterfall bar
                sub = incr_df.dropna(subset=[col_name]).sort_values(col_name, ascending=False)
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

        # Full incremental table
        st.subheader("Full Incremental Results")
        disp_cols = ["uwi", "section_name", "n_comparators", "eur_bbl",
                     "eur_bbl_baseline", "eur_bbl_incremental", "eur_bbl_pct_uplift",
                     "ip30_bpd", "ip30_bpd_incremental", "ip30_bpd_pct_uplift",
                     "waterflood_flag", "vintage_year"]
        disp_cols = [c for c in disp_cols if c in incr_df.columns]
        st.dataframe(incr_df[disp_cols], use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 3: FITTED PARAMETERS (App 2 core output)
    # ──────────────────────────────────────────────────────────────────────
    with tab_params:
        st.header("🔧 Decline Parameter Fits (App 2)")

        if not decline_results:
            st.warning("No production data loaded or no valid fits.")
        else:
            rows = []
            for uwi in sorted(decline_results.keys()):
                r = decline_results[uwi]
                lg = "2-Mile" if uwi in twomile_uwis else "1-Mile"
                rows.append({
                    "UWI": uwi,
                    "Lateral": lg,
                    "Peak (bbl/d)": round(r["peak_rate"], 1),
                    "qi (bbl/d)": round(r["qi"], 1),
                    "Di (1/d)": round(r["di"], 6),
                    "Di (1/yr)": round(r["di"] * 365.25, 4),
                    "b": round(r["b"], 3),
                    "R²": round(r["r2"], 3),
                    "EUR data (Mbbl)": round(r["eur_trap_bbl"] / 1000, 1),
                    "Months": r["n_months"],
                })
            fit_df = pd.DataFrame(rows)

            # Filter to 2-mile only by default
            show_all = st.checkbox("Show all wells (including 1-mile)", value=False)
            display_fit = fit_df if show_all else fit_df[fit_df["Lateral"] == "2-Mile"]

            st.dataframe(display_fit, use_container_width=True, hide_index=True)

            # Descriptive stats
            if len(display_fit) > 0:
                st.subheader("Descriptive Statistics")
                num_cols = ["Peak (bbl/d)", "qi (bbl/d)", "Di (1/d)", "R²", "EUR data (Mbbl)"]
                num_cols = [c for c in num_cols if c in display_fit.columns]
                st.dataframe(display_fit[num_cols].describe().T, use_container_width=True)

            # qi / di scatter
            if len(fitted_qi_vals) >= 2:
                st.subheader("qi vs Di — 2-Mile Wells")
                fig_qd = go.Figure()
                fig_qd.add_trace(go.Scatter(
                    x=fitted_di_vals, y=fitted_qi_vals,
                    mode="markers", marker=dict(color=COLORS["2-Mile"], size=10),
                    text=fitted_wells,
                    hovertemplate="<b>%{text}</b><br>Di=%{x:.5f}<br>qi=%{y:,.0f}<extra></extra>",
                ))
                fig_qd.add_hline(y=qi_anchor, line_dash="dash", line_color="black",
                                 annotation_text=f"Median qi={qi_anchor:,.0f}")
                fig_qd.update_layout(
                    **PLOTLY_LAYOUT, height=400,
                    xaxis_title="Di (1/day)", yaxis_title="qi (bbl/d)",
                    title="Fitted qi vs Di (2-Mile Wells)",
                )
                st.plotly_chart(fig_qd, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 4: WELL-BY-WELL DIAGNOSTIC
    # ──────────────────────────────────────────────────────────────────────
    with tab_well:
        st.header("🔍 Well-by-Well Diagnostic")

        if df_2mile.empty:
            st.warning("No 2-mile wells."); return

        def _fmt(u):
            n = len(well_cohort_map.get(u, []))
            has_fit = "✓" if u in decline_results else "✗"
            return f"{u}  (comps={n}, fit={has_fit})"

        sel_uwi = st.selectbox("Select 2-mile well", df_2mile["uwi"].tolist(), format_func=_fmt)
        sel_row = df_2mile[df_2mile["uwi"] == sel_uwi].iloc[0]

        # Context
        cc = st.columns(5)
        cc[0].metric("Section", sel_row.get("section_name", "—") or "—")
        v = sel_row.get("hz_length_m", np.nan)
        cc[1].metric("Hz Length (m)", f"{v:,.0f}" if pd.notna(v) else "—")
        v = sel_row.get("eur_bbl", np.nan)
        cc[2].metric("EUR (Mbbl)", f"{v/1000:,.1f}" if pd.notna(v) else "—")
        cc[3].metric("IP30 (bbl/d)", f"{sel_row.get('ip30_bpd', np.nan):,.0f}"
                     if pd.notna(sel_row.get("ip30_bpd")) else "—")
        cc[4].metric("# Comparators", len(well_cohort_map.get(sel_uwi, [])))

        # Incremental table
        sel_incr = incr_df[incr_df["uwi"] == sel_uwi] if "uwi" in incr_df.columns else pd.DataFrame()
        if not sel_incr.empty:
            st.subheader("Incremental vs 1-Mile Comparators")
            ir = []
            for mk in metric_keys:
                act = sel_row.get(mk, np.nan)
                bl = sel_incr.iloc[0].get(f"{mk}_baseline", np.nan)
                inc = sel_incr.iloc[0].get(f"{mk}_incremental", np.nan)
                pct = sel_incr.iloc[0].get(f"{mk}_pct_uplift", np.nan)
                ir.append({
                    "Metric": PERF_METRICS[mk],
                    "2-Mile Actual": f"{act:,.0f}" if pd.notna(act) else "—",
                    "1-Mile Baseline": f"{bl:,.0f}" if pd.notna(bl) else "—",
                    "Incremental": f"{inc:+,.0f}" if pd.notna(inc) else "—",
                    "Uplift %": f"{pct:+.1f}%" if pd.notna(pct) else "—",
                })
            st.dataframe(pd.DataFrame(ir), use_container_width=True, hide_index=True)

        # Decline fit overlay
        if sel_uwi in decline_results:
            res = decline_results[sel_uwi]
            st.subheader("Decline Fit")
            pc = st.columns(4)
            pc[0].metric("qi (bbl/d)", f"{res['qi']:,.1f}")
            pc[1].metric("Di (1/yr)", f"{res['di']*365.25:.4f}")
            pc[2].metric("R²", f"{res['r2']:.3f}")
            pc[3].metric("EUR data (Mbbl)", f"{res['eur_trap_bbl']/1000:,.1f}")

            dfp = res["df_post"]
            fig_well = go.Figure()
            fig_well.add_trace(go.Bar(
                x=dfp["t_days"], y=dfp["rate"],
                marker_color="#2c3e50", opacity=0.5, name="Observed",
            ))
            t_fit = np.linspace(0, dfp["t_days"].max(), 300)
            q_fit = hyp_rate(t_fit, res["qi"], res["di"], res["b"])
            fig_well.add_trace(go.Scatter(
                x=t_fit, y=q_fit, mode="lines",
                line=dict(color="#3498db", width=2, dash="dash"),
                name=f"Hyp fit (R²={res['r2']:.3f})",
            ))

            # Overlay P-curves
            for label in show_curves:
                c = final_curves.get(label)
                if c is None:
                    continue
                fig_well.add_trace(go.Scatter(
                    x=c["t"], y=c["q"], mode="lines",
                    line=dict(color=COLORS[label], width=2),
                    name=f"{label} type curve",
                ))

            fig_well.update_layout(
                **PLOTLY_LAYOUT, height=450,
                xaxis_title="Days from peak",
                yaxis_title="Rate (bbl/d)",
                title=f"Well {sel_uwi} vs Type Curves",
            )
            st.plotly_chart(fig_well, use_container_width=True)
        else:
            st.info(f"No production time-series available for {sel_uwi}.")

        # Comparator table
        comp_uwis = well_cohort_map.get(sel_uwi, [])
        if comp_uwis:
            with st.expander("📋 Comparator 1-mile wells", expanded=False):
                comp_df = df_1mile[df_1mile["uwi"].isin(comp_uwis)]
                disp = ["uwi", "section_name", "hz_length_m", "eur_bbl",
                        "ip30_bpd", "on_prod_date"]
                disp = [c for c in disp if c in comp_df.columns]
                st.dataframe(comp_df[disp], use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────────
    # TAB 5: EXPORT
    # ──────────────────────────────────────────────────────────────────────
    with tab_export:
        st.header("📥 Export Results")

        # Build traceability dataframe
        trace_rows = []
        for label in ["P25", "P50", "P75"]:
            c = final_curves.get(label)
            if c is None:
                continue
            trace_rows.append({
                "Curve": label,
                "qi_bpd": c["qi"],
                "di_per_day": c["di"],
                "di_per_year": c["di"] * 365.25,
                "b": c["b"],
                "eur_target_bbl": c["eur_target_bbl"],
                "eur_solved_bbl": c["eur_actual_bbl"],
                "eur_target_Mbbl": c["eur_target_bbl"] / 1000,
                "eur_solved_Mbbl": c["eur_actual_bbl"] / 1000,
                "qi_source": qi_source,
                "eur_source": eur_source,
                "n_2mi_wells_in_distribution": len(eur_dist),
                "b_fixed": b_fixed,
                "flat_days": FLAT_DAYS,
                "q_limit_bpd": Q_LIMIT,
            })
        trace_df = pd.DataFrame(trace_rows)

        # Curve data (for plotting/re-creation)
        curve_data_frames = {}
        for label in ["P25", "P50", "P75"]:
            c = final_curves.get(label)
            if c is None:
                continue
            cdf = pd.DataFrame({
                "day": c["t"],
                "rate_bpd": c["q"],
                "cum_bbl": c["N"],
                "cum_Mbbl": c["N"] / 1000,
            })
            cdf.insert(0, "curve", label)
            curve_data_frames[label] = cdf

        # Cohort mapping
        cohort_rows = []
        for u2, u1_list in well_cohort_map.items():
            for u1 in u1_list:
                cohort_rows.append({"uwi_2mile": u2, "uwi_1mile_comparator": u1})
        cohort_df = pd.DataFrame(cohort_rows) if cohort_rows else pd.DataFrame(
            columns=["uwi_2mile", "uwi_1mile_comparator"])

        # Contributing wells summary
        contrib_rows = []
        for _, row in df_2mile.iterrows():
            u = row["uwi"]
            has_fit = u in decline_results
            contrib_rows.append({
                "uwi": u,
                "eur_bbl": row.get("eur_bbl", np.nan),
                "eur_Mbbl": row.get("eur_bbl", np.nan) / 1000 if pd.notna(row.get("eur_bbl")) else np.nan,
                "ip30_bpd": row.get("ip30_bpd", np.nan),
                "n_comparators": len(well_cohort_map.get(u, [])),
                "has_decline_fit": has_fit,
                "fitted_qi": decline_results[u]["qi"] if has_fit else np.nan,
                "fitted_di": decline_results[u]["di"] if has_fit else np.nan,
                "fitted_r2": decline_results[u]["r2"] if has_fit else np.nan,
                "incremental_eur_bbl": incr_df.loc[incr_df["uwi"] == u, "eur_bbl_incremental"].values[0]
                    if (not incr_df.empty and "eur_bbl_incremental" in incr_df.columns
                        and u in incr_df["uwi"].values) else np.nan,
            })
        contrib_df = pd.DataFrame(contrib_rows)

        # Preview
        st.subheader("Traceability")
        st.dataframe(trace_df, use_container_width=True, hide_index=True)

        st.subheader("Contributing Wells")
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)

        # Excel download
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            trace_df.to_excel(writer, index=False, sheet_name="Curve_Parameters")
            for label, cdf in curve_data_frames.items():
                cdf.to_excel(writer, index=False, sheet_name=f"Curve_{label}")
            if not incr_df.empty:
                # Convert bbl cols back to Mbbl for export readability
                export_incr = incr_df.copy()
                for c in export_incr.columns:
                    if c.endswith("_bbl") or c == "eur_bbl":
                        mbbl_col = c.replace("_bbl", "_Mbbl")
                        export_incr[mbbl_col] = export_incr[c] / 1000
                export_incr.to_excel(writer, index=False, sheet_name="Incremental_Results")
            contrib_df.to_excel(writer, index=False, sheet_name="Contributing_Wells")
            cohort_df.to_excel(writer, index=False, sheet_name="Cohort_Mappings")

            # Fitted parameters
            if decline_results:
                fit_rows = []
                for uwi in sorted(decline_results.keys()):
                    r = decline_results[uwi]
                    fit_rows.append({
                        "uwi": uwi,
                        "lateral_group": "2-Mile" if uwi in twomile_uwis else "1-Mile",
                        "qi_bpd": r["qi"], "di_per_day": r["di"],
                        "di_per_year": r["di"] * 365.25,
                        "b": r["b"], "r2": r["r2"],
                        "eur_data_bbl": r["eur_trap_bbl"],
                        "eur_data_Mbbl": r["eur_trap_bbl"] / 1000,
                    })
                pd.DataFrame(fit_rows).to_excel(writer, index=False, sheet_name="Fitted_Parameters")

        st.download_button(
            "📥 Download Full Results (Excel)",
            buf.getvalue(),
            file_name="2mile_type_curves_P25_P50_P75.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # CSV for curves only
        if curve_data_frames:
            all_curves = pd.concat(curve_data_frames.values(), ignore_index=True)
            st.download_button(
                "📥 Download Curve Data (CSV)",
                all_curves.to_csv(index=False).encode(),
                file_name="2mile_type_curves.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()