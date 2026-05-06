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
import streamlit as st

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="2-Mile Well Analysis",
    page_icon="🛢️",
    layout="wide",
)

# Constants
XLSX_PATH        = "w.xlsx"
SHP_1M_PATH      = "1M.shp"
SHP_2M_PATH      = "2M.shp"
PROD_DATA_FILE   = "tcgenprod.xlsx"
RTC_XLSX_PATH    = "rtc.xlsx"

TARGET_METRIC_CRS       = "EPSG:26913"
WATERFLOOD_BUFFER_M     = 200.0
MILE_TO_M               = 1609.34
CORRIDOR_HALF_WIDTH_M   = 900.0

Q_LIMIT            = 2.0          # economic limit (bbl/d)
FLAT_MONTHS        = 1.44
FLAT_DAYS          = int(round(FLAT_MONTHS * 30.4375))
B_FIXED            = 0.95         # Arps exponent — LOCKED per spec
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
    "ip30_bpd":     "IP30 (bbl/d)",
    "ip90_bpd":     "IP90 (bbl/d)",
    "cum6_bbl":     "6-Month Cum (bbl)",
    "cum12_bbl":    "12-Month Cum (bbl)",
}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Arial, sans-serif", size=12),
    margin=dict(l=60, r=30, t=50, b=50),
    hovermode="x unified",
)

RTC_PALETTE = [
    "#8e44ad", "#e67e22", "#1abc9c", "#c0392b",
    "#2980b9", "#7f8c8d", "#d35400", "#27ae60",
]

# Helpers
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

# Removed MAD-filter uplift outliers feature
# def mad_filter(values: np.ndarray, z_cut: float = OUTLIER_CUTOFF) -> np.ndarray:
#     v = np.asarray(values, dtype=float)
#     finite = np.isfinite(v)
#     if finite.sum() < 4:
#         return finite
#     med = np.median(v[finite])
#     mad = np.median(np.abs(v[finite] - med))
#     if mad == 0:
#         return finite
#     mz = 0.6745 * (v - med) / mad
#     return finite & (np.abs(mz) <= z_cut)

# Arps math
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

def fit_hyperbolic_fixed_b(t_days, q_vals, qi_fixed=None, b_fixed=B_FIXED):
    t = np.asarray(t_days, dtype=float)
    q = np.asarray(q_vals, dtype=float)
    m = (t > 0) & (q > 0) & np.isfinite(t) & np.isfinite(q)
    t, q = t[m], q[m]
    if len(t) < 3:
        return None
    try:
        if qi_fixed is not None:
            popt, _ = curve_fit(
                lambda tt, di: hyp_rate(tt, qi_fixed, di, b_fixed),
                t, q,
                p0=[0.003], bounds=([1e-8], [0.5]), maxfev=30000,
            )
            return (qi_fixed, float(popt[0]), b_fixed)
        else:
            qi0 = float(np.nanmax(q))
            popt, _ = curve_fit(
                lambda tt, qi, di: hyp_rate(tt, qi, di, b_fixed),
                t, q,
                p0=[qi0, 0.003],
                bounds=([0.1, 1e-8], [qi0 * 5.0, 0.5]), maxfev=30000,
            )
            return (float(popt[0]), float(popt[1]), b_fixed)
    except Exception:
        return None

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

# Data loaders
REQUIRED_WELL_COLUMNS = [
    "UWI", "Section Name", "Well Type", "Hz Length (m)",
    "Oil + Cond: EUR (Mbbl)", "Oil + Cond: IP 30 Cal. Rate (bbl/d)",
    "Oil + Cond: IP 90 Cal. Rate (bbl/d)", "Oil + Cond: 6M CalTime Cum (Mbbl)",
    "Oil + Cond: 12M CalTime Cum (Mbbl)", "Objective", "On Prod Date",
    "On Inj Date", "FOOZ",
]

RENAME_WELL = {
    "UWI": "uwi", "Section Name": "section_name", "Well Type": "well_type",
    "Hz Length (m)": "hz_length_m",
    "Oil + Cond: EUR (Mbbl)": "eur_mbbl",
    "Oil + Cond: IP 30 Cal. Rate (bbl/d)": "ip30_bpd",
    "Oil + Cond: IP 90 Cal. Rate (bbl/d)": "ip90_bpd",
    "Oil + Cond: 6M CalTime Cum (Mbbl)": "cum6_mbbl",
    "Oil + Cond: 12M CalTime Cum (Mbbl)": "cum12_mbbl",
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
    for c_mbbl, c_bbl in [("eur_mbbl", "eur_bbl"),
                           ("cum6_mbbl", "cum6_bbl"),
                           ("cum12_mbbl", "cum12_bbl")]:
        if c_mbbl in out.columns:
            out[c_bbl] = pd.to_numeric(out[c_mbbl], errors="coerce") * 1000.0
    for c in ["hz_length_m", "ip30_bpd", "ip90_bpd",
              "eur_bbl", "cum6_bbl", "cum12_bbl"]:
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

# Spatial analytics
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

# Helper: compute per-well ratios (2mi vs median 1mi) for KPI uplift
def compute_ratio_per_well_2mi_vs_1mi(df_2mi: pd.DataFrame,
                                      df_1mi: pd.DataFrame,
                                      cohort_map: dict) -> pd.DataFrame:
    """
    For every 2-mile well, compute ratios:
      EUR ratio = EUR_2mi / median(EUR_1mi) across matched 1-mile wells
      IP30 ratio = IP30_2mi / median(IP30_1mi)
      IP90 ratio = IP90_2mi / median(IP90_1mi)
      1YCum ratio = Cum12_2mi / median(Cum12_1mi)
      6MCum ratio = Cum6_2mi / median(Cum6_1mi)
    Returns a DataFrame with columns: uwi, eur_ratio, ip30_ratio, ip90_ratio, cum12_ratio, cum6_ratio
    """
    rows = []
    for _, row2 in df_2mi.iterrows():
        uwi2 = row2["uwi"]
        matched = cohort_map.get(uwi2, [])
        comp_df1 = df_1mi[df_1mi["uwi"].isin(matched)]
        medians = {
            "eur_bbl":   comp_df1["eur_bbl"].median(),
            "ip30_bpd":  comp_df1["ip30_bpd"].median(),
            "ip90_bpd":  comp_df1["ip90_bpd"].median(),
            "cum12_bbl": comp_df1["cum12_bbl"].median(),
            "cum6_bbl":  comp_df1["cum6_bbl"].median(),
        }
        eur_ratio = (row2["eur_bbl"] / medians["eur_bbl"]) if (np.isfinite(medians["eur_bbl"]) and medians["eur_bbl"] > 0) else np.nan
        ip30_ratio = (row2["ip30_bpd"] / medians["ip30_bpd"]) if (np.isfinite(medians["ip30_bpd"]) and medians["ip30_bpd"] > 0) else np.nan
        ip90_ratio = (row2["ip90_bpd"] / medians["ip90_bpd"]) if (np.isfinite(medians["ip90_bpd"]) and medians["ip90_bpd"] > 0) else np.nan
        cum12_ratio = (row2["cum12_bbl"] / medians["cum12_bbl"]) if (np.isfinite(medians["cum12_bbl"]) and medians["cum12_bbl"] > 0) else np.nan
        cum6_ratio = (row2["cum6_bbl"] / medians["cum6_bbl"]) if (np.isfinite(medians["cum6_bbl"]) and medians["cum6_bbl"] > 0) else np.nan

        rows.append({
            "uwi": uwi2,
            "eur_ratio": eur_ratio,
            "ip30_ratio": ip30_ratio,
            "ip90_ratio": ip90_ratio,
            "cum12_ratio": cum12_ratio,
            "cum6_ratio": cum6_ratio
        })
    return pd.DataFrame(rows)

def geometric_mean_of_series(series: pd.Series) -> float:
    s = series.dropna()
    s = s[(np.isfinite(s)) & (s > 0)]
    if s.empty:
        return np.nan
    return float(np.exp(np.log(s).mean()))

# Production fitting (qi from peak recipe) with optional user override
def compute_qi_from_peak_window(df_w: pd.DataFrame) -> dict:
    df_w = df_w.sort_values("date").reset_index(drop=True)
    if df_w.empty:
        return dict(qi=np.nan, peak_date=pd.NaT, peak_rate=np.nan,
                    peak_idx=-1, used_months=[])
    peak_idx  = int(df_w["date"].idxmax())
    peak_rate = float(df_w.loc[peak_idx, "rate"])
    peak_date = df_w.loc[peak_idx, "date"]

    w_peak, w_prev, w_next = 0.5, 0.25, 0.25
    rate_peak = peak_rate
    rate_prev = df_w.loc[peak_idx - 1, "rate"] if peak_idx - 1 >= 0 else np.nan
    rate_next = df_w.loc[peak_idx + 1, "rate"] if peak_idx + 1 < len(df_w) else np.nan

    weights, rates, used = [], [], [f"peak({peak_date.date()})"]
    weights.append(w_peak); rates.append(rate_peak)
    if np.isfinite(rate_prev):
        weights.append(w_prev); rates.append(rate_prev); used.append("prev")
    if np.isfinite(rate_next):
        weights.append(w_next); rates.append(rate_next); used.append("next")

    weights = np.array(weights) / np.sum(weights)
    qi = float(np.dot(weights, rates))

    return dict(qi=qi, peak_date=peak_date, peak_rate=peak_rate,
                peak_idx=peak_idx, used_months=used)

def analyse_well_production(df_w: pd.DataFrame, b_fixed: float = B_FIXED, qi_override: Optional[float] = None) -> dict:
    df_w = df_w.sort_values("date").reset_index(drop=True)
    peak_info = compute_qi_from_peak_window(df_w)
    qi = peak_info["qi"]
    peak_date = peak_info["peak_date"]
    peak_idx  = peak_info["peak_idx"]

    df_w["t_days"] = (df_w["date"] - peak_date).dt.days.astype(float)
    df_decline = df_w[df_w["t_days"] > 0].copy()

    # Anchor qi from override if provided
    qi_anchor = qi_override if (qi_override is not None and np.isfinite(qi_override)) else qi

    hyp = None
    if len(df_decline) >= 3 and qi_anchor is not None and np.isfinite(qi_anchor):
        hyp = fit_hyperbolic_fixed_b(
            df_decline["t_days"].values, df_decline["rate"].values,
            qi_fixed=qi_anchor, b_fixed=b_fixed,
        )
    if hyp:
        qi_h, di_h, b_h = hyp
        pred = hyp_rate(df_decline["t_days"].values, qi_h, di_h, b_h)
        ss_res = np.sum((df_decline["rate"].values - pred) ** 2)
        ss_tot = np.sum((df_decline["rate"].values - df_decline["rate"].mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        if qi_anchor is not None and np.isfinite(qi_anchor):
            qi_h = qi_anchor
        else:
            qi_h = qi if np.isfinite(qi) else peak_info["peak_rate"]
        di_h = 0.003
        b_h = b_fixed
        r2 = 0.0

    eur_trap = calc_eur_trap(
        (df_w["date"] - df_w["date"].min()).dt.days.values.astype(float),
        df_w["rate"].values,
    )

    return dict(
        uwi=df_w["uwi"].iloc[0],
        peak_rate=peak_info["peak_rate"], peak_date= peak_date,
        qi=qi_h, di=di_h, b=b_h, r2=r2,
        qi_recipe=qi, qi_used_months=peak_info["used_months"],
        eur_trap_bbl=eur_trap, n_months=len(df_w),
        df_post=df_w[df_w["t_days"] >= 0].copy(),
    )

# Assembly
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

@st.cache_data(show_spinner="Fitting decline parameters on production data…")
def fit_all_declines(b_fixed, qi_override=None):
    prod_df, err = load_production_data()
    if prod_df is None:
        return {}, err
    results = {}
    for uwi, grp in prod_df.groupby("uwi"):
        if len(grp) < MIN_MONTHS_FOR_FIT:
            continue
        results[uwi] = analyse_well_production(grp, b_fixed, qi_override)
    return results, None

# Cohort + uplift builder
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
                     "vintage_year", "eur_bbl", "ip30_bpd", "ip90_bpd",
                     "cum6_bbl", "cum12_bbl",
                     "waterflood_flag", "section_ooip"]:
            rec[col] = row2.get(col, np.nan)
        records.append(rec)
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["uwi"])

# EUR target builder (mode-dependent)
def derive_eur_targets(df_2mile: pd.DataFrame,
                       df_1mile: pd.DataFrame,
                       incr_df: pd.DataFrame,
                       cohort_map: dict) -> dict:
    results = {"method": "mode-dependent empirical uplift",
               "per_well_equivalent_eur": [],
               "ratios": [],
               "baseline_perm": [],
               "used_2mi_actuals": False}

    if incr_df.empty or "eur_bbl_ratio" not in incr_df.columns:
        return results

    all_comp_uwis = {u for lst in cohort_map.values() for u in lst}
    pooled = df_1mile[df_1mile["uwi"].isin(all_comp_uwis)].copy()
    if not pooled.empty and "hz_length_m" in pooled.columns:
        perm = (pooled["eur_bbl"] / pooled["hz_length_m"]).replace(
            [np.inf, -np.inf], np.nan).dropna()
        if not perm.empty:
            results["pooled_baseline_perm"] = float(perm.median())

    Lh_targets = df_2mile["hz_length_m"].dropna()
    Lh_typ = float(Lh_targets.median()) if not Lh_targets.empty else MILE_TO_M * 2

    ratios, perms = [], []
    equiv_eurs   = []

    for _, r in incr_df.iterrows():
        ratio = r.get("eur_bbl_ratio", np.nan)
        bperm = r.get("eur_bbl_baseline_perm", np.nan)
        if np.isfinite(ratio) and np.isfinite(bperm) and ratio > 0 and bperm > 0:
            ratios.append(ratio)
            perms.append(bperm)
            equiv_eurs.append(ratio * bperm * Lh_typ)

    results["ratios"]        = ratios
    results["baseline_perm"] = perms
    results["per_well_equivalent_eur"] = equiv_eurs
    results["Lh_typical_m"]  = Lh_typ
    return results

# UI
def main():
    st.title("🛢️ 2-Mile Uplift & Decline Analysis")
    st.caption(
        "Raw comparison: 2-mile wells are compared against 1-mile wells using "
        "unscaled, direct raw metrics. EUR, IP30/IP90, 6-month cum and 12-month cum "
        "uplifts are shown with baselines (1-mile) and actuals (2-mile)."
    )

    well_df, geoms = load_and_assemble_wells()
    df_1mile = well_df[well_df["lateral_group"] == "1-Mile"].reset_index(drop=True)
    df_2mile = well_df[well_df["lateral_group"] == "2-Mile"].reset_index(drop=True)
    rtc_curves = load_rtc_curves()

    # Build cohort and ratio metrics (new uplift approach)
    # Sidebar: qi override
    st.sidebar.title("⚙️ Configuration")
    qi_override = st.sidebar.number_input("Override qi (bbl/d) for type curves (0 = auto)", min_value=0.0, value=0.0, step=1.0)

    # UI: analysis mode
    analysis_mode = st.sidebar.radio(
        "Analog Matching Mode",
        ["Mode A: Range-Based Analog Matching", "Mode B: Geometric Corridor"],
        key="analysis_mode_radio",
    )

    # tolerances / features
    st.sidebar.divider()
    active_features, tolerances = [], {}
    corridor_width = CORRIDOR_HALF_WIDTH_M

    if analysis_mode.startswith("Mode A"):
        st.sidebar.subheader("Matching Tolerances")
        available = []
        if well_df["section_ooip"].notna().any():         available.append("section_ooip")
        if well_df["nearest_producer_m"].notna().any():   available.append("nearest_producer_m")
        if well_df["nearest_injector_m"].notna().any():   available.append("nearest_injector_m")
        # Remove MAD-outliers toggle; keep only active features by default
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
        f"**qi recipe**: `0.5·peak + 0.25·prev + 0.25·next`<br>"
        f"**flat period** = `{FLAT_DAYS} d` (≈{FLAT_MONTHS} mo)<br>"
        f"**q-limit** = `{Q_LIMIT} bbl/d`",
        unsafe_allow_html=True,
    )
    # NEW uplift section: compute ratios (2mi vs 1mi)
    ratio_df = compute_ratio_per_well_2mi_vs_1mi(df_2mile, df_1mile, build_cohort_map(
        df_2mile, df_1mile, geoms, analysis_mode, tolerances, active_features, corridor_width
    ))
    gm_eur  = geometric_mean_of_series(ratio_df["eur_ratio"]) if not ratio_df.empty else np.nan
    gm_ip30 = geometric_mean_of_series(ratio_df["ip30_ratio"]) if not ratio_df.empty else np.nan
    gm_ip90 = geometric_mean_of_series(ratio_df["ip90_ratio"]) if not ratio_df.empty else np.nan
    gm_cum12 = geometric_mean_of_series(ratio_df["cum12_ratio"]) if not ratio_df.empty else np.nan
    gm_cum6  = geometric_mean_of_series(ratio_df["cum6_ratio"]) if not ratio_df.empty else np.nan

    # Top-level display: Geometric means
    gm_html = f"""
    <details open>
      <summary>Geometric means of KPI ratios (2mi / 1mi)</summary>
      <ul>
        <li>EUR (2mi/1mi) ratio GM: {gm_eur:.3f} </li>
        <li>IP30 (2mi/1mi) ratio GM: {gm_ip30:.3f} </li>
        <li>IP90 (2mi/1mi) ratio GM: {gm_ip90:.3f} </li>
        <li>1YCum (12M) (2mi/1mi) ratio GM: {gm_cum12:.3f} </li>
        <li>6MCum (6M) (2mi/1mi) ratio GM: {gm_cum6:.3f} </li>
      </ul>
    </details>
    """
    st.markdown(gm_html, unsafe_allow_html=True)

    # Decline fits (prod data)
    decline_results, prod_err = fit_all_declines(B_FIXED, qi_override if (qi_override and qi_override > 0) else None)
    if prod_err:
        st.warning(f"⚠️ Production data issue: {prod_err}. Decline fitting skipped — qi will fall back to IP30 from the well table.")

    metric_keys = list(PERF_METRICS.keys())

    # Build cohort mapping
    cohort_map  = build_cohort_map(df_2mile, df_1mile, geoms, analysis_mode, tolerances, active_features, corridor_width)

    # NEW: Per-well KPI ratios have been computed above as ratio_df
    # Display per-well KPI ratios (optional quick view)
    if not ratio_df.empty:
        st.subheader("Per-well KPI ratios (2mi / 1mi)")
        st.dataframe(ratio_df[["uwi", "eur_ratio", "ip30_ratio", "ip90_ratio", "cum12_ratio", "cum6_ratio"]],
                     use_container_width=True, hide_index=True)

    # Optional: EUR uplift distribution from 2mi vs 1mi using existing incremental framework
    incr_df     = build_incremental_frame(df_2mile, df_1mile, cohort_map, metric_keys)

    eur_info = derive_eur_targets(df_2mile, df_1mile, incr_df, cohort_map)
    equiv_eurs = np.array(eur_info.get("per_well_equivalent_eur", []), dtype=float)

    # Original EUR distribution logic for Type Curves
    if len(equiv_eurs) >= MIN_COMPS_FOR_UPLIFT:
        eur_dist   = equiv_eurs
        eur_source = (f"empirical uplift from {len(equiv_eurs)} 2-mi wells "
                       f"× 1-mi comparators ({analysis_mode.split(':')[0]})")
    elif not df_2mile["eur_bbl"].dropna().empty:
        eur_dist   = df_2mile["eur_bbl"].dropna().values
        eur_source = "fallback — actual 2-mile EUR distribution"
    else:
        eur_dist, eur_source = np.array([]), "insufficient data"

    eur_summary = empirical_summary(eur_dist)

    # qi anchor (allow override)
    twomile_uwis  = set(df_2mile["uwi"].dropna())
    fitted_qi_vals, fitted_di_vals, fitted_wells = [], [], []
    qi_anchor = None
    qi_source = ""

    if qi_override and qi_override > 0:
        qi_anchor = float(qi_override)
        qi_source = f"user-specified qi = {qi_anchor:,.0f}"
    else:
        for uwi, res in decline_results.items():
            if uwi in twomile_uwis:
                if np.isfinite(res["qi"]) and res["qi"] > 0:
                    fitted_qi_vals.append(res["qi"])
                    fitted_di_vals.append(res["di"])
                    fitted_wells.append(uwi)

        if fitted_qi_vals:
            qi_anchor = float(np.median(fitted_qi_vals))
            qi_source = (f"median of {len(fitted_qi_vals)} 2-mi wells — "
                          f"qi recipe: 0.5·peak + 0.25·prev + 0.25·next")
        else:
            ip30_vals = df_2mile["ip30_bpd"].dropna().values
            qi_anchor = float(np.median(ip30_vals)) if len(ip30_vals) > 0 else 150.0
            qi_source = "fallback — median IP30 from well table"

    # Final curves
    pct_targets = {
        "P10": eur_summary["q10"], "P25": eur_summary["q25"],
        "P50": eur_summary["q50"], "P75": eur_summary["q75"],
        "P90": eur_summary["q90"],
    }
    final_curves = {}
    for label, eur_target in pct_targets.items():
        if np.isfinite(eur_target) and eur_target > 0:
            final_curves[label] = build_curve_from_eur(
                qi_anchor, eur_target, B_FIXED, FLAT_DAYS, Q_LIMIT)
        else:
            final_curves[label] = None

    tab_curves, tab_uplift, tab_rtc, tab_params, tab_well, tab_export = st.tabs([
        "📈 Type Curves", "📊 Uplift Analysis", "🆚 vs RTC",
        "🔧 Fitted Parameters", "🔍 Well-by-Well",
        "📥 Export",
    ])

    # Type Curves
    with tab_curves:
        st.header("Type Curves")

        c0 = st.columns(6)
        c0[0].metric("Mode", "A (Range)" if analysis_mode.startswith("Mode A") else "B (Corridor)")
        c0[1].metric("2-Mile Wells", len(df_2mile))
        c0[2].metric("n in EUR dist", eur_summary["n"])
        c0[3].metric("qi anchor (bbl/d)", f"{qi_anchor:,.0f}")
        c0[4].metric("b (fixed)", f"{B_FIXED:.2f}")
        c0[5].metric("Prod fits", len(fitted_wells))

        # Optional explanatory note removed to keep UI concise

        st.subheader("Curve Parameters")
        param_rows = []
        for label in ["P10", "P25", "P50", "P75", "P90"]:
            c = final_curves.get(label)
            if c is None:
                param_rows.append({"Curve": label, "qi (bbl/d)": "—",
                                    "Di (1/d)": "—", "Di (1/yr)": "—",
                                    "b": "—", "EUR tgt (Mbbl)": "—",
                                    "EUR solved (Mbbl)": "—"})
            else:
                param_rows.append({
                    "Curve": label,
                    "qi (bbl/d)":         f"{c['qi']:,.1f}",
                    "Di (1/d)":           f"{c['di']:.6f}",
                    "Di (1/yr)":          f"{c['di'] * 365.25:.4f}",
                    "b":                  f"{c['b']:.2f}",
                    "EUR tgt (Mbbl)":     f"{c['eur_target_bbl'] / 1000:,.1f}",
                    "EUR solved (Mbbl)":  f"{c['eur_actual_bbl'] / 1000:,.1f}",
                })
        st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)

        # Rate plot
        st.subheader("Rate Type Curves — q(t)")
        fig_rate = go.Figure()
        n_bg = 0
        for label in ["P10","P25","P50","P75","P90"]:
            c = final_curves.get(label)
            if c is None: continue
            fig_rate.add_trace(go.Scatter(
                x=c["t"], y=c["q"], mode="lines",
                line=dict(color=COLORS[label], width=3),
                name=f"{label} — EUR={c['eur_actual_bbl']/1000:,.0f} Mbbl",
                hovertemplate=f"{label}<br>Day %{{x:,.0f}}<br>%{{y:,.1f}} bbl/d",
            ))
        if overlay_rtc := True:  # keep RTC overlay option (always on for simplicity)
            if rtc_curves:
                for i, rtc in enumerate(rtc_curves):
                    fig_rate.add_trace(go.Scatter(
                        x=rtc["t"], y=rtc["q"], mode="lines",
                        line=dict(color=RTC_PALETTE[i % len(RTC_PALETTE)],
                                   width=2, dash="dash"),
                        name=f"RTC: {rtc['name']}",
                    ))
        fig_rate.add_hline(y=Q_LIMIT, line_dash="dot", line_color="grey",
                           opacity=0.5,
                           annotation_text=f"{Q_LIMIT} bbl/d limit")
        fig_rate.update_layout(**PLOTLY_LAYOUT, height=550,
                               xaxis_title="Days since peak / start of flat period",
                               yaxis_title="Rate (bbl/d)",
                               title="Rate — q(t)")
        fig_rate.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_rate, use_container_width=True)

        st.subheader("Cumulative Type Curves — N(t)")
        fig_cum = go.Figure()
        for label in ["P10","P25","P50","P75","P90"]:
            c = final_curves.get(label)
            if c is None: continue
            fig_cum.add_trace(go.Scatter(
                x=c["t"], y=c["N"] / 1000, mode="lines",
                line=dict(color=COLORS[label], width=3),
                name=f"{label} — EUR={c['eur_target_bbl']/1000:,.0f} Mbbl",
                hovertemplate=f"{label}<br>Day %{{x:,.0f}}<br>%{{y:,.1f}} Mbbl",
            ))
        if rtc_curves:
            for i, rtc in enumerate(rtc_curves):
                fig_cum.add_trace(go.Scatter(
                    x=rtc["t"], y=rtc["N"] / 1000, mode="lines",
                    line=dict(color=RTC_PALETTE[i % len(RTC_PALETTE)],
                               width=2, dash="dash"),
                    name=f"RTC: {rtc['name']}",
                ))
        fig_cum.update_layout(**PLOTLY_LAYOUT, height=500,
                               xaxis_title="Days since peak / start of flat period",
                               yaxis_title="Cumulative (Mbbl)",
                               title="Cumulative — N(t)")
        fig_cum.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig_cum, use_container_width=True)

        st.subheader("EUR Distribution Context")
        if len(eur_dist) > 0:
            fig_eur = go.Figure()
            fig_eur.add_trace(go.Box(
                y=eur_dist / 1000, name="Derived EUR (Mbbl)",
                boxpoints="all", jitter=0.5, pointpos=0,
                marker=dict(color=COLORS["2-Mile"], size=9),
                line=dict(color="#e67e22"),
                fillcolor="rgba(255,127,14,0.2)", boxmean=True,
            ))
            for label in ["P10", "P25", "P50", "P75", "P90"]:
                val = pct_targets.get(label)
                if np.isfinite(val):
                    fig_eur.add_hline(
                        y=val / 1000, line_dash="dash",
                        line_color=COLORS[label], line_width=2,
                        annotation_text=f"{label}: {val/1000:,.0f} Mbbl",
                        annotation_position="top right")
            fig_eur.update_layout(
                **PLOTLY_LAYOUT, height=420,
                title=f"EUR Distribution (n={len(eur_dist)}, source: {eur_source})",
                yaxis_title="EUR (Mbbl)",
            )
            st.plotly_chart(fig_eur, use_container_width=True)
        else:
            st.warning("No EUR distribution data available.")

    # Uplift analysis
    with tab_uplift:
        st.header("📊 Empirical Uplift (2-Mile vs 1-Mile)")
        mode_label = ("Range-Based" if analysis_mode.startswith("Mode A")
                      else f"Corridor (±{int(corridor_width)} m)")
        st.caption(
            f"Mode: **{mode_label}** · {len(df_2mile)} two-mile wells · "
            f"{len(df_1mile)} one-mile wells · "
            f"{sum(len(v) for v in cohort_map.values())} cohort links"
        )

        # Per-well KPI ratios shown above (new uplift metric)
        if not ratio_df.empty:
            st.subheader("Per-well KPI ratios (2mi / 1mi) — detailed view above")

        # Incremental uplift distributions (existing logic)
        for mk in metric_keys:
            col_name = f"{mk}_incremental"
            if col_name not in incr_df.columns: continue
            vals = incr_df[col_name].dropna().values
            if len(vals) == 0: continue
            # Percentile-based distribution (kept for backward compatibility)
            if len(vals) >= 5:
                p10, p25, p50, p75, p90 = np.percentile(vals, [10, 25, 50, 75, 90])
            else:
                p10 = p25 = p50 = p75 = p90 = np.nan

            with st.expander(
                f"**{PERF_METRICS[mk]}** — incremental distribution (n={len(vals)})",
                expanded=(mk == "eur_bbl"),
            ):
                mc = st.columns(5)
                mc[0].metric("P10", f"{p10:+,.0f}" if np.isfinite(p10) else "—")
                mc[1].metric("P25", f"{p25:+,.0f}" if np.isfinite(p25) else "—")
                mc[2].metric("P50", f"{p50:+,.0f}" if np.isfinite(p50) else "—")
                mc[3].metric("P75", f"{p75:+,.0f}" if np.isfinite(p75) else "—")
                mc[4].metric("P90", f"{p90:+,.0f}" if np.isfinite(p90) else "—")

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
                     "eur_bbl", "eur_bbl_baseline", "eur_bbl_incremental",
                     "eur_bbl_pct_uplift", "eur_bbl_ratio",
                     "ip30_bpd", "ip30_bpd_incremental",
                     "ip30_bpd_pct_uplift", "ip30_bpd_ratio",
                     "waterflood_flag", "vintage_year"]
        disp_cols = [c for c in disp_cols if c in incr_df.columns]
        st.dataframe(incr_df[disp_cols], use_container_width=True, hide_index=True)

    # vs RTC
    with tab_rtc:
        st.header("🆚 Benchmarking vs Corporate RTC Curves (`rtc.xlsx`)")
        if not rtc_curves:
            st.info("No RTC curves loaded. Verify `rtc.xlsx` presence and "
                    "required columns: Name, Months, Qi, b, EUR.")
        else:
            st.subheader("Summary Comparison")
            rows = []
            for rtc in rtc_curves:
                row = {
                    "RTC Name":        rtc["name"],
                    "RTC qi (bbl/d)":  round(rtc["qi"], 1),
                    "RTC Di (1/yr)":   round(rtc["di"] * 365.25, 4),
                    "RTC b":           round(rtc["b"], 2),
                    "RTC EUR (Mbbl)":  round(rtc["eur_actual_bbl"] / 1000, 1),
                }
                for label in ["P10", "P25", "P50", "P75", "P90"]:
                    c = final_curves.get(label)
                    if c is None:
                        row[f"Δ{label} EUR (Mbbl)"] = "—"
                        row[f"Δ{label} %"]          = "—"
                        continue
                    d_abs = (c["eur_actual_bbl"] - rtc["eur_actual_bbl"]) / 1000
                    d_pct = ((c["eur_actual_bbl"] - rtc["eur_actual_bbl"])
                              / rtc["eur_actual_bbl"] * 100
                              if rtc["eur_actual_bbl"] else np.nan)
                    row[f"Δ{label} EUR (Mbbl)"] = f"{d_abs:+,.1f}"
                    row[f"Δ{label} %"]          = f"{d_pct:+.1f}%"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.subheader("Rate Overlay — RTC vs Derived")
            fig_bench = go.Figure()
            for label in ["P10","P25","P50","P75","P90"]:
                c = final_curves.get(label)
                if c is None: continue
                fig_bench.add_trace(go.Scatter(
                    x=c["t"], y=c["q"], mode="lines",
                    line=dict(color=COLORS[label], width=3),
                    name=f"Derived {label}",
                ))
            for i, rtc in enumerate(rtc_curves):
                fig_bench.add_trace(go.Scatter(
                    x=rtc["t"], y=rtc["q"], mode="lines",
                    line=dict(color=RTC_PALETTE[i % len(RTC_PALETTE)],
                               width=2, dash="dash"),
                    name=f"RTC: {rtc['name']}",
                ))
            fig_bench.update_layout(**PLOTLY_LAYOUT, height=550,
                                    xaxis_title="Days",
                                    yaxis_title="Rate (bbl/d)",
                                    title="Derived vs RTC — q(t)")
            st.plotly_chart(fig_bench, use_container_width=True)

            st.subheader("EUR Bar — RTC vs Derived")
            fig_eurbar = go.Figure()
            cats, vals, cols = [], [], []
            for rtc in rtc_curves:
                cats.append(f"RTC: {rtc['name']}")
                vals.append(rtc["eur_actual_bbl"] / 1000)
                cols.append("#7f8c8d")
            for label in ["P10","P25","P50","P75","P90"]:
                c = final_curves.get(label)
                if c is None: continue
                cats.append(f"Derived {label}")
                vals.append(c["eur_actual_bbl"] / 1000)
                cols.append(COLORS[label])
            fig_eurbar.add_trace(go.Bar(x=cats, y=vals, marker_color=cols,
                                         text=[f"{v:,.0f}" for v in vals],
                                         textposition="outside"))
            fig_eurbar.update_layout(**PLOTLY_LAYOUT, height=400,
                                     yaxis_title="EUR (Mbbl)",
                                     title="EUR Comparison")
            st.plotly_chart(fig_eurbar, use_container_width=True)

    # Fitted Parameters
    with tab_params:
        st.header("🔧 Decline Parameter Fits (b fixed = 0.95)")

        if not decline_results:
            st.warning("No production data loaded or no valid fits.")
        else:
            rows = []
            for uwi in sorted(decline_results.keys()):
                r = decline_results[uwi]
                lg = "2-Mile" if uwi in twomile_uwis else "1-Mile"
                rows.append({
                    "UWI": uwi, "Lateral": lg,
                    "Peak (bbl/d)":     round(r["peak_rate"], 1),
                    "qi recipe":        round(r["qi_recipe"], 1)
                                         if np.isfinite(r["qi_recipe"]) else "—",
                    "qi fitted":        round(r["qi"], 1),
                    "Di (1/d)":         round(r["di"], 6),
                    "Di (1/yr)":        round(r["di"] * 365.25, 4),
                    "b":                round(r["b"], 3),
                    "R²":               round(r["r2"], 3),
                    "EUR data (Mbbl)":  round(r["eur_trap_bbl"] / 1000, 1),
                    "Months":           r["n_months"],
                })
            fit_df = pd.DataFrame(rows)

            show_all = st.checkbox("Show all wells (incl. 1-mile)", value=False)
            display_fit = fit_df if show_all else \
                          fit_df[fit_df["Lateral"] == "2-Mile"]

            st.dataframe(display_fit, use_container_width=True, hide_index=True)

            if len(display_fit) > 0:
                st.subheader("Descriptive Statistics")
                num_cols = ["Peak (bbl/d)", "qi fitted", "Di (1/d)",
                             "R²", "EUR data (Mbbl)"]
                num_cols = [c for c in num_cols if c in display_fit.columns]
                st.dataframe(display_fit[num_cols].describe().T,
                              use_container_width=True)

            if len(fitted_qi_vals) >= 2:
                st.subheader("qi vs Di — 2-Mile Wells")
                fig_qd = go.Figure()
                fig_qd.add_trace(go.Scatter(
                    x=fitted_di_vals, y=fitted_qi_vals,
                    mode="markers",
                    marker=dict(color=COLORS["2-Mile"], size=10),
                    text=fitted_wells,
                    hovertemplate="<b>%{text}</b><br>Di=%{x:.5f}<br>"
                                    "qi=%{y:,.0f}<extra></extra>",
                ))
                fig_qd.add_hline(y=qi_anchor, line_dash="dash",
                                  line_color="black",
                                  annotation_text=f"Median qi={qi_anchor:,.0f}")
                fig_qd.update_layout(**PLOTLY_LAYOUT, height=400,
                                     xaxis_title="Di (1/day)",
                                     yaxis_title="qi (bbl/d)",
                                     title="Fitted qi vs Di (2-Mile Wells)")
                st.plotly_chart(fig_qd, use_container_width=True)

    # Well-by-Well
    with tab_well:
        st.header("🔍 Well-by-Well Diagnostic")
        if df_2mile.empty:
            st.warning("No 2-mile wells."); return

        def _fmt(u):
            n = len(cohort_map.get(u, []))
            has_fit = "✓" if u in decline_results else "✗"
            return f"{u}  (comps={n}, fit={has_fit})"

        sel_uwi = st.selectbox("Select 2-mile well",
                                df_2mile["uwi"].tolist(), format_func=_fmt)
        sel_row = df_2mile[df_2mile["uwi"] == sel_uwi].iloc[0]

        cc = st.columns(5)
        cc[0].metric("Section", sel_row.get("section_name", "—") or "—")
        v = sel_row.get("hz_length_m", np.nan)
        cc[1].metric("Hz Length (m)", f"{v:,.0f}" if pd.notna(v) else "—")
        v = sel_row.get("eur_bbl", np.nan)
        cc[2].metric("EUR (Mbbl)", f"{v/1000:,.1f}" if pd.notna(v) else "—")
        cc[3].metric("IP30 (bbl/d)",
                     f"{sel_row.get('ip30_bpd', np.nan):,.0f}"
                      if pd.notna(sel_row.get("ip30_bpd")) else "—")
        cc[4].metric("# Comparators", len(cohort_map.get(sel_uwi, [])))

        sel_incr = incr_df[incr_df["uwi"] == sel_uwi] \
                     if "uwi" in incr_df.columns else pd.DataFrame()
        if not sel_incr.empty:
            st.subheader("Incremental vs 1-Mile Comparators")
            ir = []
            for mk in metric_keys:
                act   = sel_row.get(mk, np.nan)
                bl    = sel_incr.iloc[0].get(f"{mk}_baseline", np.nan)
                inc   = sel_incr.iloc[0].get(f"{mk}_incremental", np.nan)
                pct   = sel_incr.iloc[0].get(f"{mk}_pct_uplift", np.nan)
                rat   = sel_incr.iloc[0].get(f"{mk}_ratio", np.nan)
                ir.append({
                    "Metric":         PERF_METRICS[mk],
                    "2-Mile Actual":  f"{act:,.0f}" if pd.notna(act) else "—",
                    "1-Mile Baseline (Raw)": f"{bl:,.0f}"  if pd.notna(bl) else "—",
                    "Incremental":    f"{inc:+,.0f}" if pd.notna(inc) else "—",
                    "Uplift %":       f"{pct:+.1f}%" if pd.notna(pct) else "—",
                    "Uplift Ratio": f"{rat:.3f}"   if pd.notna(rat) else "—",
                })
            st.dataframe(pd.DataFrame(ir), use_container_width=True, hide_index=True)

        if sel_uwi in decline_results:
            res = decline_results[sel_uwi]
            st.subheader("Decline Fit (b fixed = 0.95)")
            pc = st.columns(4)
            pc[0].metric("qi recipe (bbl/d)",
                         f"{res['qi_recipe']:,.1f}" if np.isfinite(res['qi_recipe']) else "—")
            pc[1].metric("Di (1/yr)", f"{res['di']*365.25:.4f}")
            pc[2].metric("R²", f"{res['r2']:.3f}")
            pc[3].metric("EUR data (Mbbl)", f"{res['eur_trap_bbl']/1000:,.1f}")

            st.caption(f"qi window: {', '.join(res['qi_used_months'])}")

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
            for label in []:  # keep structure for future: show_curves
                pass
            fig_well.update_layout(**PLOTLY_LAYOUT, height=450,
                                   xaxis_title="Days from peak",
                                   yaxis_title="Rate (bbl/d)",
                                   title=f"Well {sel_uwi} vs Type Curves")
            st.plotly_chart(fig_well, use_container_width=True)
        else:
            st.info(f"No production time-series available for {sel_uwi}.")

        comp_uwis = cohort_map.get(sel_uwi, [])
        if comp_uwis:
            with st.expander("📋 Comparator 1-mile wells", expanded=False):
                comp_df = data = df_1mile[df_1mile["uwi"].isin(comp_uwis)]
                disp = ["uwi", "section_name", "hz_length_m",
                        "eur_bbl", "ip30_bpd", "on_prod_date"]
                disp = [c for c in disp if c in comp_df.columns]
                st.dataframe(comp_df[disp], use_container_width=True, hide_index=True)

    # Export
    with tab_export:
        st.header("📥 Export Results")

        trace_rows = []
        for label in ["P10", "P25", "P50", "P75", "P90"]:
            c = final_curves.get(label)
            if c is None: continue
            trace_rows.append({
                "Curve":               label,
                "qi_bpd":              c["qi"],
                "di_per_day":          c["di"],
                "di_per_year":         c["di"] * 365.25,
                "b":                   c["b"],
                "eur_target_bbl":      c["eur_target_bbl"],
                "eur_solved_bbl":      c["eur_actual_bbl"],
                "eur_target_Mbbl":     c["eur_target_bbl"] / 1000,
                "eur_solved_Mbbl":     c["eur_actual_bbl"] / 1000,
                "qi_source":           qi_source,
                "eur_source":          eur_source,
                "analysis_mode":       analysis_mode,
                "n_in_eur_distribution": eur_summary["n"],
                "b_fixed":             B_FIXED,
                "flat_days":           FLAT_DAYS,
                "q_limit_bpd":         Q_LIMIT,
            })
        trace_df = pd.DataFrame(trace_rows)

        curve_data_frames = {}
        for label in ["P10", "P25", "P50", "P75", "P90"]:
            c = final_curves.get(label)
            if c is None: continue
            cdf = pd.DataFrame({
                "day":        c["t"],
                "rate_bpd":   c["q"],
                "cum_bbl":    c["N"],
                "cum_Mbbl":   c["N"] / 1000,
            })
            cdf.insert(0, "curve", label)
            curve_data_frames[label] = cdf

        cohort_rows = []
        for u2, u1_list in cohort_map.items():
            for u1 in u1_list:
                cohort_rows.append({"uwi_2mile": u2, "uwi_1mile_comparator": u1})
        cohort_df = pd.DataFrame(cohort_rows) if cohort_rows else pd.DataFrame(
            columns=["uwi_2mile", "uwi_1mile_comparator"])

        contrib_rows = []
        for _, row in df_2mile.iterrows():
            u = row["uwi"]
            has_fit = u in twomile_uwis
            contrib_rows.append({
                "uwi":                  u,
                "eur_bbl":              row.get("eur_bbl", np.nan),
                "eur_Mbbl":             row.get("eur_bbl", np.nan) / 1000
                                         if pd.notna(row.get("eur_bbl")) else np.nan,
                "ip30_bpd":             row.get("ip30_bpd", np.nan),
                "hz_length_m":          row.get("hz_length_m", np.nan),
                "n_comparators":        len(cohort_map.get(u, [])),
                "has_decline_fit":      has_fit,
                "fitted_qi":            decline_results[u]["qi"] if has_fit else np.nan,
                "fitted_di":            decline_results[u]["di"] if has_fit else np.nan,
                "fitted_r2":            decline_results[u]["r2"] if has_fit else np.nan,
                "incremental_eur_bbl":  incr_df.loc[incr_df["uwi"] == u,
                                                    "eur_bbl_incremental"].values[0]
                    if (not incr_df.empty and "eur_bbl_incremental" in incr_df.columns
                        and u in incr_df["uwi"].values) else np.nan,
                "eur_ratio_per_m":      incr_df.loc[incr_df["uwi"] == u,
                                                    "eur_bbl_ratio"].values[0]
                    if (not incr_df.empty and "eur_bbl_ratio" in incr_df.columns
                        and u in incr_df["uwi"].values) else np.nan,
            })
        contrib_df = pd.DataFrame(contrib_rows)

        st.subheader("Traceability")
        st.dataframe(trace_df, use_container_width=True, hide_index=True)
        st.subheader("Contributing Wells")
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            trace_df.to_excel(writer, index=False,
                               sheet_name="Curve_Parameters")
            for label, cdf in curve_data_frames.items():
                cdf.to_excel(writer, index=False,
                              sheet_name=f"Curve_{label}")
            if not incr_df.empty:
                export_incr = incr_df.copy()
                for c in export_incr.columns:
                    if c.endswith("_bbl") or c == "eur_bbl":
                        mbbl_col = c.replace("_bbl", "_Mbbl")
                        export_incr[mbbl_col] = export_incr[c] / 1000
                export_incr.to_excel(writer, index=False,
                                      sheet_name="Incremental_Results")
            contrib_df.to_excel(writer, index=False,
                                 sheet_name="Contributing_Wells")
            cohort_df.to_excel(writer, index=False,
                                sheet_name="Cohort_Mappings")
            if decline_results:
                fit_rows = []
                for uwi in sorted(decline_results.keys()):
                    r = decline_results[uwi]
                    fit_rows.append({
                        "uwi":           uwi,
                        "lateral_group": "2-Mile" if uwi in twomile_uwis else "1-Mile",
                        "qi_recipe":     r["qi_recipe"],
                        "qi_bpd":        r["qi"],
                        "di_per_day":    r["di"],
                        "di_per_year":   r["di"] * 365.25,
                        "b":             r["b"],
                        "r2":            r["r2"],
                        "eur_data_bbl":  r["eur_trap_bbl"],
                        "eur_data_Mbbl": r["eur_trap_bbl"] / 1000,
                    })
                pd.DataFrame(fit_rows).to_excel(writer, index=False,
                                                  sheet_name="Fitted_Parameters")
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
            file_name="2mile_type_curves_raw_uplift.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        if curve_data_frames:
            all_curves = pd.concat(curve_data_frames.values(), ignore_index=True)
            st.download_button(
                "📥 Download Curve Data (CSV)",
                all_curves.to_csv(index=False).encode(),
                file_name="2mile_type_curves_raw.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()