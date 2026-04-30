import io
import math
import os
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import streamlit as st

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="2-Mile Type Curve Generator", page_icon="🛢️", layout="wide")

XLSX_PATH      = "w.xlsx"
SHP_1M_PATH    = "1M.shp"
SHP_2M_PATH    = "2M.shp"
PROD_DATA_FILE = "tcgenprod.xlsx"
RTC_XLSX_PATH  = "rtc.xlsx"

TARGET_METRIC_CRS     = "EPSG:3347"
MILE_TO_M             = 1609.34
CORRIDOR_HALF_WIDTH_M = 900.0

B_FIXED    = 0.95
Q_LIMIT    = 2.0
FLAT_DAYS  = int(round(1.44 * 30.4375))   # ≈ 44 days plateau at qi

COLORS = {"P25": "#e74c3c", "P50": "#3498db", "P75": "#2ecc71",
          "2-Mile": "#ff7f0e", "1-Mile": "#1f77b4"}

PLOTLY_LAYOUT = dict(template="plotly_white",
                     font=dict(family="Inter, Arial, sans-serif", size=12),
                     margin=dict(l=60, r=30, t=50, b=50),
                     hovermode="x unified")


# ═══════════════════════════════════════════════════════════════════════════════
# ARPS DECLINE MATH
# ═══════════════════════════════════════════════════════════════════════════════

def hyp_rate(t, qi, di, b=B_FIXED):
    # Arps hyperbolic rate at t (days)
    t = np.asarray(t, dtype=float)
    if b == 0:
        return qi * np.exp(-di * t)
    return qi / np.power(1.0 + b * di * t, 1.0 / b)


def cum_arps(t, qi, di, b=B_FIXED):
    # Analytical Arps cumulative (bbl) given t in days, qi in bbl/d
    t = np.asarray(t, dtype=float)
    if di <= 0:
        return qi * t
    if b == 0:
        return (qi / di) * (1.0 - np.exp(-di * t))
    if b == 1:
        return (qi / di) * np.log(1.0 + di * t)
    return (qi / ((1.0 - b) * di)) * (1.0 - np.power(1.0 + b * di * t, (b - 1.0) / b))


def t_to_rate(di, qi, b, q_target):
    # Time (days) for rate to fall from qi to q_target
    if q_target <= 0 or di <= 0 or qi <= q_target:
        return np.inf
    if b == 0:
        return math.log(qi / q_target) / di
    return ((qi / q_target) ** b - 1.0) / (b * di)


def find_di_for_eur(qi, eur_post_flat, b, q_target=Q_LIMIT):
    # Bisection: find Di such that post-plateau Arps cum from qi → q_target equals eur_post_flat
    if eur_post_flat <= 0 or qi <= q_target:
        return 0.0

    def _resid(di):
        t_end = t_to_rate(di, qi, b, q_target)
        return np.inf if np.isinf(t_end) else cum_arps(t_end, qi, di, b) - eur_post_flat

    lo, hi = 1e-10, 1.0
    for _ in range(200):
        if np.sign(_resid(lo)) * np.sign(_resid(hi)) < 0:
            break
        hi *= 2
        if hi > 1e4:
            return 0.0

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = _resid(mid)
        if abs(f_mid) < 1e-4:
            return mid
        if _resid(lo) * f_mid <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def build_curve(qi, eur_bbl, b=B_FIXED, flat_days=FLAT_DAYS, q_limit=Q_LIMIT):
    # Build piecewise curve: flat plateau at qi for flat_days, then Arps to q_limit targeting eur_bbl total
    eur_post = max(eur_bbl - qi * flat_days, 0.0)
    di = find_di_for_eur(qi, eur_post, b, q_limit) if eur_post > 0 else 0.0

    t_list = list(range(flat_days + 1))
    q_list = [qi] * (flat_days + 1)

    if di > 0:
        for k in range(1, 60001):
            q = hyp_rate(k, qi, di, b)
            t_list.append(flat_days + k)
            q_list.append(max(q, q_limit))
            if q <= q_limit:
                break

    t_arr = np.array(t_list, dtype=float)
    q_arr = np.array(q_list, dtype=float)

    # Cumulative via trapezoidal
    N_arr = np.concatenate([[0.0], np.cumsum(0.5 * (q_arr[1:] + q_arr[:-1]) * np.diff(t_arr))])

    return {"t": t_arr, "q": q_arr, "N": N_arr, "qi": qi, "di": di, "b": b,
            "eur_target_bbl": eur_bbl, "eur_actual_bbl": float(N_arr[-1])}


def fit_di_only(t_days, q_vals, qi, b=B_FIXED):
    # With qi pinned, fit only Di on post-peak production data
    t = np.asarray(t_days, dtype=float)
    q = np.asarray(q_vals, dtype=float)
    mask = (t > 0) & (q > 0) & np.isfinite(t) & np.isfinite(q)
    t, q = t[mask], q[mask]
    if len(t) < 4:
        return None, 0.0
    try:
        popt, _ = curve_fit(lambda tt, di: hyp_rate(tt, qi, di, b),
                            t, q, p0=[0.003], bounds=([1e-8], [0.1]), maxfev=20000)
        di = float(popt[0])
        pred = hyp_rate(t, qi, di, b)
        ss_res = np.sum((q - pred) ** 2)
        ss_tot = np.sum((q - q.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return di, r2
    except Exception:
        return None, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# DATA INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

def _std(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.upper()
            .replace({"NAN": np.nan, "NONE": np.nan, "": np.nan}))


@st.cache_data(show_spinner=False)
def load_well_table():
    xls = pd.ExcelFile(XLSX_PATH)
    raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0])
    rename = {"UWI": "uwi", "Section Name": "section_name", "Hz Length (m)": "hz_length_m",
              "Oil + Cond: EUR (Mbbl)": "eur_mbbl",
              "Oil + Cond: IP 30 Cal. Rate (bbl/d)": "ip30_bpd",
              "On Prod Date": "on_prod_date", "FOOZ": "fooz", "Objective": "objective"}
    out = raw.rename(columns=rename)
    for c in ["uwi", "section_name", "fooz", "objective"]:
        if c in out.columns:
            out[c] = _std(out[c])
    out["on_prod_date"] = pd.to_datetime(out.get("on_prod_date"), errors="coerce")
    out["eur_bbl"]      = pd.to_numeric(out.get("eur_mbbl"),  errors="coerce") * 1000.0
    out["hz_length_m"]  = pd.to_numeric(out.get("hz_length_m"), errors="coerce")
    out["ip30_bpd"]     = pd.to_numeric(out.get("ip30_bpd"),    errors="coerce")
    out["is_fooz"]      = out.get("fooz", pd.Series([""]*len(out))).fillna("").eq("YES")
    return out.replace([np.inf, -np.inf], np.nan)


@st.cache_resource(show_spinner=False)
def load_geometries():
    g1 = gpd.read_file(SHP_1M_PATH); g1["lateral_group"] = "1-Mile"
    g2 = gpd.read_file(SHP_2M_PATH); g2["lateral_group"] = "2-Mile"
    gdf = gpd.GeoDataFrame(pd.concat([g1, g2], ignore_index=True),
                           geometry="geometry", crs=g1.crs or g2.crs)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    try:
        gdf = gdf.to_crs(TARGET_METRIC_CRS)
    except Exception:
        c = gdf.geometry.union_all().centroid
        utm = int((c.x + 180) // 6) + 1
        epsg = 32600 + utm if c.y >= 0 else 32700 + utm
        gdf = gdf.to_crs(f"EPSG:{epsg}")
    gdf["uwi"] = _std(gdf["UWI"])
    gdf = (gdf.sort_values("lateral_group", ascending=False)
              .drop_duplicates("uwi", keep="first"))
    gdf["midpoint"] = gdf.geometry.apply(
        lambda g: g.interpolate(0.5, normalized=True) if g is not None and not g.is_empty else None)
    return gdf[["uwi", "lateral_group", "geometry", "midpoint"]].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_production_data():
    if not os.path.exists(PROD_DATA_FILE):
        return None
    df = pd.read_excel(PROD_DATA_FILE)
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {}
    for c in df.columns:
        if "uwi" in c: rename[c] = "uwi"
        elif "month" in c or "date" in c: rename[c] = "month"
        elif "bbl" in c or "rate" in c:   rename[c] = "rate"
    df = df.rename(columns=rename)
    df["uwi"]  = _std(df["uwi"])
    df["date"] = pd.to_datetime(df["month"], errors="coerce")
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df = df.dropna(subset=["uwi", "date", "rate"]).query("rate >= 0")
    df = df.sort_values(["uwi", "date"]).reset_index(drop=True)
    df["days_in_month"] = df["date"].dt.days_in_month
    return df


@st.cache_data(show_spinner=False)
def load_rtc_curves():
    if not os.path.exists(RTC_XLSX_PATH):
        return []
    try:
        rtc = pd.read_excel(RTC_XLSX_PATH)
    except Exception:
        return []
    if not {"Name", "Months", "Qi", "b", "EUR"}.issubset(rtc.columns):
        return []
    out = []
    for _, r in rtc.iterrows():
        try:
            c = build_curve(float(r["Qi"]), float(r["EUR"]),
                            float(r["b"]), int(float(r["Months"]) * 30))
            c["name"] = str(r["Name"])
            out.append(c)
        except Exception:
            continue
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# PEAK-MONTH qi CALCULATION  ⭐ qi = 0.5*peak + 0.25*(before) + 0.25*(after)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_peak_qi(df_well):
    # Weighted peak-month qi: handles edge cases where peak is at start/end of history
    df_well = df_well.sort_values("date").reset_index(drop=True)
    if df_well.empty:
        return np.nan, None, None
    peak_idx  = int(df_well["rate"].idxmax())
    peak_rate = float(df_well.loc[peak_idx, "rate"])
    peak_date = df_well.loc[peak_idx, "date"]

    before = float(df_well.loc[peak_idx - 1, "rate"]) if peak_idx - 1 >= 0 else peak_rate
    after  = float(df_well.loc[peak_idx + 1, "rate"]) if peak_idx + 1 < len(df_well) else peak_rate

    qi = 0.5 * peak_rate + 0.25 * before + 0.25 * after
    return qi, peak_idx, peak_date


def observed_eur(df_well):
    # Trapezoidal cum of monthly volumes (bbl)
    if df_well.empty:
        return 0.0
    vols = df_well["rate"].values * df_well["days_in_month"].values
    return float(vols.sum())


# ═══════════════════════════════════════════════════════════════════════════════
# ANALOG MATCHING (corridor-based, 1-mile wells for each 2-mile target)
# ═══════════════════════════════════════════════════════════════════════════════

def corridor_match(target_geom, geoms_gdf, half_width):
    if target_geom is None or target_geom.is_empty:
        return []
    corridor = target_geom.buffer(half_width)
    ones = geoms_gdf[geoms_gdf["lateral_group"] == "1-Mile"]
    if ones.empty:
        return []
    inside = ones["midpoint"].apply(
        lambda mp: mp is not None and not mp.is_empty and corridor.contains(mp))
    return ones.loc[inside, "uwi"].tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# PER-WELL EUR ESTIMATION — the core statistical engine
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_well_eur(uwi, prod_df, well_meta, analog_uwis, df_1mile,
                      uplift_factor, min_months_for_fit=6):
    """
    Return three EUR estimates + blended EUR for a single 2-mile well:
      - eur_obs:    trapezoidal integration of actual production to date
      - eur_fit:    Arps forecast from (qi_peak, Di_fitted, b=0.95) to q_limit
      - eur_uplift: median 1-mile analog EUR × uplift_factor

    Weights are maturity-aware:
      - Young well (<6 months): fit + uplift only (obs is too truncated to trust)
      - Mid (6-18 months): blend all three equally-ish
      - Mature (>18 months): obs + fit dominate

    Returns dict with all components and blended EUR.
    """
    out = {"uwi": uwi, "eur_obs": np.nan, "eur_fit": np.nan, "eur_uplift": np.nan,
           "eur_blended": np.nan, "qi": np.nan, "di": np.nan, "r2": np.nan,
           "n_months": 0, "n_analogs": len(analog_uwis)}

    df_w = prod_df[prod_df["uwi"] == uwi] if prod_df is not None else pd.DataFrame()
    n_months = len(df_w)
    out["n_months"] = n_months

    # ── qi from peak-month formula ─────────────────────────────────────────
    if n_months >= 3:
        qi, peak_idx, peak_date = compute_peak_qi(df_w)
        out["qi"] = qi

        # Observed EUR (trapezoidal)
        out["eur_obs"] = observed_eur(df_w)

        # Fit Di only (qi pinned to peak-month value)
        if n_months >= min_months_for_fit:
            df_w = df_w.sort_values("date").reset_index(drop=True)
            df_w["t_days"] = (df_w["date"] - peak_date).dt.days.astype(float)
            post = df_w[df_w["t_days"] > 0]
            if len(post) >= 4:
                di, r2 = fit_di_only(post["t_days"].values, post["rate"].values, qi, B_FIXED)
                if di is not None:
                    out["di"], out["r2"] = di, r2
                    # Forecast EUR = observed to peak + plateau + Arps forecast
                    t_end = t_to_rate(di, qi, B_FIXED, Q_LIMIT)
                    if np.isfinite(t_end):
                        out["eur_fit"] = observed_eur(df_w.iloc[:peak_idx + 1]) + \
                                         qi * FLAT_DAYS + cum_arps(t_end, qi, di, B_FIXED)

    # ── Uplift-anchored EUR from 1-mile analogs ────────────────────────────
    if analog_uwis and not df_1mile.empty:
        analog_eurs = df_1mile.loc[df_1mile["uwi"].isin(analog_uwis), "eur_bbl"].dropna()
        if len(analog_eurs) > 0:
            out["eur_uplift"] = float(analog_eurs.median()) * uplift_factor

    # ── Fall back to well-table EUR if production missing ──────────────────
    table_eur = well_meta.get("eur_bbl", np.nan)
    if pd.isna(out["eur_obs"]) and pd.notna(table_eur):
        out["eur_obs"] = float(table_eur)

    # ── Blend with maturity-aware weights ──────────────────────────────────
    comps = []
    if n_months >= 18:
        # Mature: obs + fit dominate
        weights = {"eur_obs": 0.40, "eur_fit": 0.45, "eur_uplift": 0.15}
    elif n_months >= 6:
        weights = {"eur_obs": 0.25, "eur_fit": 0.45, "eur_uplift": 0.30}
    elif n_months >= 1:
        # Young: lean on fit + uplift
        weights = {"eur_obs": 0.10, "eur_fit": 0.35, "eur_uplift": 0.55}
    else:
        # No production: uplift anchor + table EUR only
        weights = {"eur_obs": 0.40, "eur_fit": 0.0, "eur_uplift": 0.60}

    total_w, total_v = 0.0, 0.0
    for key, w in weights.items():
        v = out.get(key, np.nan)
        if pd.notna(v) and v > 0:
            total_w += w
            total_v += w * v
    out["eur_blended"] = total_v / total_w if total_w > 0 else np.nan
    out["weights_used"] = weights
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.title("🛢️ 2-Mile Type Curve Generator — P25 / P50 / P75")
    st.caption(f"Peak-month `qi` weighting · `b` fixed at {B_FIXED} · "
               "EUR blended from observed + fit + uplift anchors")

    # ── Load data ──────────────────────────────────────────────────────────
    well_df  = load_well_table()
    geoms    = load_geometries()
    prod_df  = load_production_data()
    rtc      = load_rtc_curves()

    # Join lateral group onto well table
    membership = geoms[["uwi", "lateral_group"]].drop_duplicates()
    well_df = membership.merge(well_df, on="uwi", how="left")
    well_df = well_df[~well_df["is_fooz"].fillna(False)].reset_index(drop=True)

    df_1mile = well_df[well_df["lateral_group"] == "1-Mile"].reset_index(drop=True)
    df_2mile = well_df[well_df["lateral_group"] == "2-Mile"].reset_index(drop=True)

    # ── Sidebar ────────────────────────────────────────────────────────────
    st.sidebar.title("⚙️ Configuration")
    corridor_width = st.sidebar.number_input("Corridor half-width (m)",
                                              100.0, 3000.0, CORRIDOR_HALF_WIDTH_M, 100.0)
    uplift_factor = st.sidebar.number_input("2-mile uplift factor (×1-mile median EUR)",
                                             1.0, 3.0, 1.85, 0.05,
                                             help="Typical 2-mile laterals recover ~1.7–2.0× a 1-mile analog.")
    min_months_fit = st.sidebar.number_input("Min months for Arps fit", 3, 24, 6, 1)
    st.sidebar.divider()
    show_curves = st.sidebar.multiselect("Display curves",
                                          ["P25", "P50", "P75"],
                                          default=["P25", "P50", "P75"])
    overlay_rtc = st.sidebar.checkbox("Overlay corporate RTC curves", value=True)
    show_priors = st.sidebar.checkbox("Show priors (75/125/175 Mbbl)", value=True)

    # ── Analog matching ────────────────────────────────────────────────────
    analog_map = {}
    for _, r2 in df_2mile.iterrows():
        gr = geoms[geoms["uwi"] == r2["uwi"]]
        analog_map[r2["uwi"]] = (corridor_match(gr.iloc[0].geometry, geoms, corridor_width)
                                  if not gr.empty else [])

    # ── Per-well EUR estimates ─────────────────────────────────────────────
    estimates = []
    for _, r2 in df_2mile.iterrows():
        meta = r2.to_dict()
        est = estimate_well_eur(r2["uwi"], prod_df, meta,
                                analog_map[r2["uwi"]], df_1mile,
                                uplift_factor, min_months_fit)
        estimates.append(est)
    est_df = pd.DataFrame(estimates)

    # ── Percentile EURs (P25 = low case, P75 = high case; reservoir convention) ──
    blended = est_df["eur_blended"].dropna().values
    if len(blended) < 3:
        st.error("Not enough 2-mile wells with estimable EURs (need ≥3). Check data files.")
        return

    eur_p25 = float(np.percentile(blended, 25))
    eur_p50 = float(np.percentile(blended, 50))
    eur_p75 = float(np.percentile(blended, 75))

    # ── qi anchor: median of fitted qi across 2-mile wells with fits ───────
    fitted_qis = est_df.loc[est_df["qi"].notna(), "qi"].values
    qi_anchor = float(np.median(fitted_qis)) if len(fitted_qis) else 150.0

    # ── Build final curves ─────────────────────────────────────────────────
    curves = {lbl: build_curve(qi_anchor, eur, B_FIXED)
              for lbl, eur in [("P25", eur_p25), ("P50", eur_p50), ("P75", eur_p75)]}

    # ══════════════════════════════════════════════════════════════════════
    # DISPLAY
    # ══════════════════════════════════════════════════════════════════════
    tab_curves, tab_stats, tab_wells, tab_export = st.tabs(
        ["📈 Type Curves", "📊 EUR Statistics", "🔍 Per-Well Detail", "📥 Export"])

    # ── TAB 1: Curves ──────────────────────────────────────────────────────
    with tab_curves:
        c = st.columns(5)
        c[0].metric("2-Mile Wells", len(df_2mile))
        c[1].metric("With Blended EUR", len(blended))
        c[2].metric("qi Anchor (bbl/d)", f"{qi_anchor:,.0f}")
        c[3].metric("b (fixed)", f"{B_FIXED}")
        c[4].metric("Uplift Factor", f"{uplift_factor:.2f}×")

        # Parameter table
        rows = []
        for lbl in ["P25", "P50", "P75"]:
            cv = curves[lbl]
            rows.append({"Curve": lbl, "qi (bbl/d)": f"{cv['qi']:,.1f}",
                         "Di (1/d)": f"{cv['di']:.6f}", "Di (1/yr)": f"{cv['di']*365.25:.3f}",
                         "b": f"{cv['b']:.2f}",
                         "EUR Target (Mbbl)": f"{cv['eur_target_bbl']/1000:,.1f}",
                         "EUR Solved (Mbbl)": f"{cv['eur_actual_bbl']/1000:,.1f}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Prior comparison
        if show_priors:
            st.markdown("**Priors vs Computed (Mbbl):**")
            pri = pd.DataFrame({
                "Percentile": ["P25", "P50", "P75"],
                "Prior": [75, 125, 175],
                "Computed": [eur_p25/1000, eur_p50/1000, eur_p75/1000],
                "Δ vs Prior": [eur_p25/1000 - 75, eur_p50/1000 - 125, eur_p75/1000 - 175],
            })
            st.dataframe(pri.style.format({"Prior": "{:,.0f}", "Computed": "{:,.1f}",
                                            "Δ vs Prior": "{:+,.1f}"}),
                         use_container_width=True, hide_index=True)

        # Rate plot
        st.subheader("Rate — q(t)")
        fig = go.Figure()

        # Observed 2-mile wells in background
        if prod_df is not None:
            shown = False
            for uwi in df_2mile["uwi"]:
                dfw = prod_df[prod_df["uwi"] == uwi].sort_values("date")
                if len(dfw) < 2:
                    continue
                qi_peak, _, pdate = compute_peak_qi(dfw)
                t_days = (dfw["date"] - pdate).dt.days.values.astype(float) + FLAT_DAYS
                fig.add_trace(go.Scatter(
                    x=t_days, y=dfw["rate"], mode="lines",
                    line=dict(color="grey", width=0.6), opacity=0.25,
                    name="2-Mile wells (obs)" if not shown else None,
                    showlegend=not shown, hoverinfo="skip"))
                shown = True

        for lbl in show_curves:
            cv = curves[lbl]
            fig.add_trace(go.Scatter(
                x=cv["t"], y=cv["q"], mode="lines",
                line=dict(color=COLORS[lbl], width=3),
                name=f"{lbl} — {cv['eur_actual_bbl']/1000:,.0f} Mbbl",
                hovertemplate=f"{lbl}<br>Day %{{x:,.0f}}<br>%{{y:,.1f}} bbl/d<extra></extra>"))

        if overlay_rtc and rtc:
            palette = ["#8e44ad", "#e67e22", "#1abc9c", "#c0392b", "#7f8c8d"]
            for i, rc in enumerate(rtc):
                fig.add_trace(go.Scatter(
                    x=rc["t"], y=rc["q"], mode="lines",
                    line=dict(color=palette[i % len(palette)], width=2, dash="dash"),
                    name=f"RTC: {rc['name']}"))

        fig.add_hline(y=Q_LIMIT, line_dash="dot", line_color="grey", opacity=0.5,
                      annotation_text=f"{Q_LIMIT} bbl/d limit")
        fig.update_layout(**PLOTLY_LAYOUT, height=550,
                          xaxis_title="Days since peak",
                          yaxis_title="Rate (bbl/d)",
                          title="2-Mile Type Curves — Rate")
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, use_container_width=True)

        # Cumulative plot
        st.subheader("Cumulative — N(t)")
        figc = go.Figure()
        for lbl in show_curves:
            cv = curves[lbl]
            figc.add_trace(go.Scatter(
                x=cv["t"], y=cv["N"]/1000, mode="lines",
                line=dict(color=COLORS[lbl], width=3),
                name=f"{lbl} — {cv['eur_actual_bbl']/1000:,.0f} Mbbl"))
        if overlay_rtc and rtc:
            palette = ["#8e44ad", "#e67e22", "#1abc9c", "#c0392b", "#7f8c8d"]
            for i, rc in enumerate(rtc):
                figc.add_trace(go.Scatter(
                    x=rc["t"], y=rc["N"]/1000, mode="lines",
                    line=dict(color=palette[i % len(palette)], width=2, dash="dash"),
                    name=f"RTC: {rc['name']}"))
        figc.update_layout(**PLOTLY_LAYOUT, height=480,
                           xaxis_title="Days since peak",
                           yaxis_title="Cumulative (Mbbl)",
                           title="2-Mile Type Curves — Cumulative")
        figc.update_yaxes(rangemode="tozero")
        st.plotly_chart(figc, use_container_width=True)

    # ── TAB 2: Stats ───────────────────────────────────────────────────────
    with tab_stats:
        st.header("EUR Distribution Across 2-Mile Wells")

        # Distribution plot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=blended/1000, name="Blended EUR",
            boxpoints="all", jitter=0.5, pointpos=0,
            marker=dict(color=COLORS["2-Mile"], size=9),
            line=dict(color="#e67e22"),
            fillcolor="rgba(255,127,14,0.2)", boxmean=True))
        for lbl, v in [("P25", eur_p25), ("P50", eur_p50), ("P75", eur_p75)]:
            fig_box.add_hline(y=v/1000, line_dash="dash", line_color=COLORS[lbl],
                              line_width=2,
                              annotation_text=f"{lbl}: {v/1000:,.0f} Mbbl",
                              annotation_position="top right")
        fig_box.update_layout(**PLOTLY_LAYOUT, height=450,
                              title=f"Blended EUR distribution (n={len(blended)})",
                              yaxis_title="EUR (Mbbl)")
        st.plotly_chart(fig_box, use_container_width=True)

        # Component comparison — each well's 3 estimates
        st.subheader("EUR Component Breakdown")
        melt = est_df[["uwi", "eur_obs", "eur_fit", "eur_uplift", "eur_blended"]].copy()
        for c in ["eur_obs", "eur_fit", "eur_uplift", "eur_blended"]:
            melt[c] = melt[c] / 1000
        comp = melt.melt(id_vars="uwi", var_name="Component", value_name="EUR (Mbbl)")
        comp["Component"] = comp["Component"].map({
            "eur_obs": "Observed", "eur_fit": "Arps Fit",
            "eur_uplift": "Analog × Uplift", "eur_blended": "Blended"})
        fig_comp = go.Figure()
        for comp_name, color in [("Observed", "#7f8c8d"), ("Arps Fit", "#3498db"),
                                  ("Analog × Uplift", "#e67e22"), ("Blended", "#2c3e50")]:
            sub = comp[comp["Component"] == comp_name].dropna()
            fig_comp.add_trace(go.Box(y=sub["EUR (Mbbl)"], name=comp_name,
                                       marker_color=color, boxpoints="all",
                                       jitter=0.3, pointpos=0))
        fig_comp.update_layout(**PLOTLY_LAYOUT, height=450,
                                title="Distribution of each EUR estimator across wells",
                                yaxis_title="EUR (Mbbl)")
        st.plotly_chart(fig_comp, use_container_width=True)

        # Summary table
        st.subheader("Summary Statistics (Mbbl)")
        summ = pd.DataFrame({
            "Component": ["Observed", "Arps Fit", "Analog × Uplift", "Blended"],
            "n":     [est_df[c].notna().sum() for c in ["eur_obs","eur_fit","eur_uplift","eur_blended"]],
            "P25":   [np.nanpercentile(est_df[c]/1000, 25) for c in ["eur_obs","eur_fit","eur_uplift","eur_blended"]],
            "P50":   [np.nanpercentile(est_df[c]/1000, 50) for c in ["eur_obs","eur_fit","eur_uplift","eur_blended"]],
            "P75":   [np.nanpercentile(est_df[c]/1000, 75) for c in ["eur_obs","eur_fit","eur_uplift","eur_blended"]],
            "Mean":  [np.nanmean(est_df[c]/1000) for c in ["eur_obs","eur_fit","eur_uplift","eur_blended"]],
            "Std":   [np.nanstd(est_df[c]/1000, ddof=1) for c in ["eur_obs","eur_fit","eur_uplift","eur_blended"]],
        })
        st.dataframe(summ.style.format({"P25":"{:,.1f}","P50":"{:,.1f}","P75":"{:,.1f}",
                                         "Mean":"{:,.1f}","Std":"{:,.1f}"}),
                     use_container_width=True, hide_index=True)

    # ── TAB 3: Per-well ────────────────────────────────────────────────────
    with tab_wells:
        st.header("Per-Well Detail")

        display = est_df.copy()
        for c in ["eur_obs", "eur_fit", "eur_uplift", "eur_blended"]:
            display[c + "_Mbbl"] = display[c] / 1000
        show_cols = ["uwi", "n_months", "n_analogs", "qi", "di", "r2",
                     "eur_obs_Mbbl", "eur_fit_Mbbl", "eur_uplift_Mbbl", "eur_blended_Mbbl"]
        st.dataframe(
            display[show_cols].style.format({
                "qi": "{:,.1f}", "di": "{:.5f}", "r2": "{:.2f}",
                "eur_obs_Mbbl": "{:,.1f}", "eur_fit_Mbbl": "{:,.1f}",
                "eur_uplift_Mbbl": "{:,.1f}", "eur_blended_Mbbl": "{:,.1f}"
            }), use_container_width=True, hide_index=True)

        st.divider()
        sel = st.selectbox("Inspect a well",
                            df_2mile["uwi"].tolist(),
                            format_func=lambda u: f"{u} (n_months={est_df.loc[est_df['uwi']==u,'n_months'].values[0]})")
        rec = est_df[est_df["uwi"] == sel].iloc[0]

        c = st.columns(4)
        c[0].metric("qi (peak-wtd)", f"{rec['qi']:,.1f}" if pd.notna(rec['qi']) else "—")
        c[1].metric("Di (1/yr)", f"{rec['di']*365.25:.3f}" if pd.notna(rec['di']) else "—")
        c[2].metric("R²", f"{rec['r2']:.3f}" if pd.notna(rec['r2']) else "—")
        c[3].metric("Months", f"{rec['n_months']}")

        c2 = st.columns(4)
        c2[0].metric("EUR obs (Mbbl)",      f"{rec['eur_obs']/1000:,.1f}"    if pd.notna(rec['eur_obs'])    else "—")
        c2[1].metric("EUR fit (Mbbl)",      f"{rec['eur_fit']/1000:,.1f}"    if pd.notna(rec['eur_fit'])    else "—")
        c2[2].metric("EUR uplift (Mbbl)",   f"{rec['eur_uplift']/1000:,.1f}" if pd.notna(rec['eur_uplift']) else "—")
        c2[3].metric("EUR blended (Mbbl)",  f"{rec['eur_blended']/1000:,.1f}" if pd.notna(rec['eur_blended']) else "—")

        if prod_df is not None and pd.notna(rec["qi"]):
            dfw = prod_df[prod_df["uwi"] == sel].sort_values("date")
            if not dfw.empty:
                _, _, pdate = compute_peak_qi(dfw)
                t_obs = (dfw["date"] - pdate).dt.days.values.astype(float)

                fig_w = go.Figure()
                fig_w.add_trace(go.Bar(x=t_obs, y=dfw["rate"],
                                        marker_color="#2c3e50", opacity=0.5,
                                        name="Observed"))
                if pd.notna(rec["di"]):
                    t_fit = np.linspace(1, max(t_obs.max(), 365*5), 400)
                    q_fit = hyp_rate(t_fit, rec["qi"], rec["di"], B_FIXED)
                    fig_w.add_trace(go.Scatter(x=t_fit, y=q_fit, mode="lines",
                                                line=dict(color="#3498db", width=2, dash="dash"),
                                                name=f"Fit (R²={rec['r2']:.2f})"))
                for lbl in show_curves:
                    cv = curves[lbl]
                    fig_w.add_trace(go.Scatter(x=cv["t"] - FLAT_DAYS, y=cv["q"], mode="lines",
                                                line=dict(color=COLORS[lbl], width=2),
                                                name=f"{lbl} type curve"))
                fig_w.update_layout(**PLOTLY_LAYOUT, height=480,
                                     xaxis_title="Days from peak",
                                     yaxis_title="Rate (bbl/d)",
                                     title=f"{sel} — observed vs fit vs type curves")
                fig_w.update_yaxes(rangemode="tozero")
                st.plotly_chart(fig_w, use_container_width=True)

    # ── TAB 4: Export ──────────────────────────────────────────────────────
    with tab_export:
        st.header("Export")

        trace = pd.DataFrame([{
            "Curve": lbl, "qi_bpd": curves[lbl]["qi"],
            "di_per_day": curves[lbl]["di"], "di_per_year": curves[lbl]["di"]*365.25,
            "b": curves[lbl]["b"],
            "eur_target_Mbbl": curves[lbl]["eur_target_bbl"]/1000,
            "eur_solved_Mbbl": curves[lbl]["eur_actual_bbl"]/1000,
            "flat_days": FLAT_DAYS, "q_limit_bpd": Q_LIMIT,
            "uplift_factor": uplift_factor, "corridor_halfwidth_m": corridor_width,
            "n_wells_in_distribution": len(blended),
        } for lbl in ["P25", "P50", "P75"]])

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
            trace.to_excel(w, sheet_name="Curve_Parameters", index=False)
            for lbl in ["P25", "P50", "P75"]:
                cv = curves[lbl]
                pd.DataFrame({"day": cv["t"], "rate_bpd": cv["q"],
                               "cum_Mbbl": cv["N"]/1000}).to_excel(
                    w, sheet_name=f"Curve_{lbl}", index=False)
            exp = est_df.copy()
            for c in ["eur_obs", "eur_fit", "eur_uplift", "eur_blended"]:
                exp[c.replace("eur_", "eur_") + "_Mbbl"] = exp[c] / 1000
            exp.to_excel(w, sheet_name="Per_Well_EUR", index=False)

            cohort = pd.DataFrame([{"uwi_2mile": u2, "uwi_1mile": u1}
                                    for u2, lst in analog_map.items() for u1 in lst])
            if not cohort.empty:
                cohort.to_excel(w, sheet_name="Analog_Cohorts", index=False)

        st.dataframe(trace, use_container_width=True, hide_index=True)
        st.download_button("📥 Download Excel Report", buf.getvalue(),
                            file_name="2mile_type_curves.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    main()