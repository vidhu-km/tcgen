# ============================================================
# 2-Mile Well Uplift & Type Curve Analysis
# Streamlined / Refactored Version
# ============================================================

import io
import math
import os
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="2-Mile Well Analysis",
    page_icon="🛢️",
    layout="wide",
)

# ============================================================
# FILES
# ============================================================

XLSX_PATH = "w.xlsx"
SHP_1M_PATH = "1M.shp"
SHP_2M_PATH = "2M.shp"
RTC_XLSX_PATH = "rtc.xlsx"

# ============================================================
# CONSTANTS
# ============================================================

TARGET_METRIC_CRS = "EPSG:26913"

Q_LIMIT = 2.0
FLAT_MONTHS = 1.44
FLAT_DAYS = int(round(FLAT_MONTHS * 30.4375))
B_FIXED = 0.95

CORRIDOR_HALF_WIDTH_M = 900.0
WATERFLOOD_BUFFER_M = 200.0

COLORS = {
    "1M": "#1f77b4",
    "2M": "#ff7f0e",
    "USER": "#e74c3c",
}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(size=12),
    margin=dict(l=50, r=20, t=50, b=50),
)

METRIC_LABELS = {
    "eur_bbl": "EUR (bbl)",
    "ip90_bpd": "IP90 (bbl/d)",
    "sixm_bpd": "6M Rate (bbl/d)",
    "twelvem_bpd": "12M Rate (bbl/d)",
}

PCT_MAP = {
    10: 90,
    25: 75,
    50: 50,
    75: 25,
    90: 10,
}

# ============================================================
# HELPERS
# ============================================================


def std_text(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"": np.nan, "NONE": np.nan, "NAN": np.nan})
    )


def require_cols(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"{name} missing columns: {missing}")
        st.stop()


def finite(values):
    return np.array([v for v in values if np.isfinite(v)], dtype=float)


def geometric_mean(series):
    s = pd.Series(series).dropna()
    s = s[(s > 0) & np.isfinite(s)]

    if s.empty:
        return np.nan

    return float(np.exp(np.log(s).mean()))


# ============================================================
# DECLINE CURVES
# ============================================================


def hyp_rate(t, qi, di, b):
    t = np.asarray(t, dtype=float)

    if b == 0:
        return qi * np.exp(-di * t)

    return qi / np.power(1 + b * di * t, 1 / b)


def cumulative_arps(t, qi, di, b):
    t = np.asarray(t, dtype=float)

    if di <= 0:
        return qi * t

    if b == 0:
        return (qi / di) * (1 - np.exp(-di * t))

    if b == 1:
        return (qi / di) * np.log(1 + di * t)

    factor = qi / ((1 - b) * di)
    power = (b - 1) / b

    return factor * (1 - np.power(1 + b * di * t, power))


def time_to_limit(qi, di, b, q_limit):
    if q_limit <= 0 or di <= 0:
        return np.inf

    if b == 0:
        return math.log(qi / q_limit) / di

    return ((qi / q_limit) ** b - 1) / (b * di)


def solve_di(qi, eur_post, b, q_limit):
    if eur_post <= 0:
        return None

    def residual(di):
        t_end = time_to_limit(qi, di, b, q_limit)

        if np.isinf(t_end):
            return np.inf

        return cumulative_arps(t_end, qi, di, b) - eur_post

    lo, hi = 1e-10, 1.0

    for _ in range(100):
        f_lo = residual(lo)
        f_hi = residual(hi)

        if f_lo * f_hi <= 0:
            break

        hi *= 2

    for _ in range(100):
        mid = (lo + hi) / 2
        f_mid = residual(mid)

        if abs(f_mid) < 1e-5:
            return mid

        if residual(lo) * f_mid <= 0:
            hi = mid
        else:
            lo = mid

    return mid


def build_curve(qi, eur_bbl, b=B_FIXED):
    eur_flat = qi * FLAT_DAYS
    eur_post = max(eur_bbl - eur_flat, 0)

    di = solve_di(qi, eur_post, b, Q_LIMIT)

    if di is None:
        di = 0

    t = list(range(FLAT_DAYS + 1))
    q = [qi] * (FLAT_DAYS + 1)

    td = 0

    while True:
        td += 1

        rate = hyp_rate(td, qi, di, b)

        t.append(FLAT_DAYS + td)
        q.append(max(rate, Q_LIMIT))

        if rate <= Q_LIMIT or td > 50000:
            break

    t = np.array(t, dtype=float)
    q = np.array(q, dtype=float)

    cum = np.zeros_like(t)

    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        cum[i] = cum[i - 1] + 0.5 * (q[i] + q[i - 1]) * dt

    return {
        "t": t,
        "q": q,
        "N": cum,
        "qi": qi,
        "di": di,
        "eur": cum[-1],
        "b": b,
    }


# ============================================================
# DATA LOADERS
# ============================================================


@st.cache_data
def load_wells():

    required = [
        "UWI",
        "Section Name",
        "Well Type",
        "Oil + Cond: EUR (Mbbl)",
        "Oil + Cond: IP 90 Cal. Rate (bbl/d)",
        "Oil + Cond: 6M Cal. Rate (bbl/d)",
        "Oil + Cond: 12M Cal. Rate (bbl/d)",
        "On Prod Date",
    ]

    df = pd.read_excel(XLSX_PATH, sheet_name=0)

    require_cols(df, required, XLSX_PATH)

    rename = {
        "UWI": "uwi",
        "Section Name": "section_name",
        "Oil + Cond: EUR (Mbbl)": "eur_mbbl",
        "Oil + Cond: IP 90 Cal. Rate (bbl/d)": "ip90_bpd",
        "Oil + Cond: 6M Cal. Rate (bbl/d)": "sixm_bpd",
        "Oil + Cond: 12M Cal. Rate (bbl/d)": "twelvem_bpd",
        "On Prod Date": "on_prod_date",
    }

    df = df.rename(columns=rename)

    df["uwi"] = std_text(df["uwi"])

    df["eur_bbl"] = pd.to_numeric(df["eur_mbbl"], errors="coerce") * 1000

    numeric_cols = [
        "ip90_bpd",
        "sixm_bpd",
        "twelvem_bpd",
        "eur_bbl",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["on_prod_date"] = pd.to_datetime(
        df["on_prod_date"],
        errors="coerce",
    )

    df["vintage_year"] = df["on_prod_date"].dt.year

    return df


@st.cache_resource
def load_geometries():

    g1 = gpd.read_file(SHP_1M_PATH)
    g2 = gpd.read_file(SHP_2M_PATH)

    g1["lateral_group"] = "1-Mile"
    g2["lateral_group"] = "2-Mile"

    gdf = pd.concat([g1, g2], ignore_index=True)

    gdf = gpd.GeoDataFrame(
        gdf,
        geometry="geometry",
        crs=g1.crs,
    )

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs(TARGET_METRIC_CRS)

    gdf["uwi"] = std_text(gdf["UWI"])

    gdf["midpoint"] = gdf.geometry.interpolate(
        0.5,
        normalized=True,
    )

    return gdf[
        ["uwi", "geometry", "midpoint", "lateral_group"]
    ]


@st.cache_data
def load_rtc():

    if not os.path.exists(RTC_XLSX_PATH):
        return []

    rtc = pd.read_excel(RTC_XLSX_PATH)

    required = {"Name", "Months", "Qi", "b", "EUR"}

    if not required.issubset(rtc.columns):
        return []

    curves = []

    for _, row in rtc.iterrows():

        try:
            curve = build_curve(
                row["Qi"],
                row["EUR"],
                row["b"],
            )

            curve["name"] = row["Name"]

            curves.append(curve)

        except Exception:
            continue

    return curves


# ============================================================
# MATCHING
# ============================================================


def corridor_match(target_geom, geoms, width):

    corridor = target_geom.buffer(width)

    ones = geoms[geoms["lateral_group"] == "1-Mile"]

    inside = ones["midpoint"].apply(
        lambda x: corridor.contains(x)
    )

    return ones.loc[inside, "uwi"].tolist()


def build_cohort_map(df2, df1, geoms, width):

    mapping = {}

    for _, row in df2.iterrows():

        uwi = row["uwi"]

        g = geoms[geoms["uwi"] == uwi]

        if g.empty:
            mapping[uwi] = []
            continue

        matches = corridor_match(
            g.iloc[0].geometry,
            geoms,
            width,
        )

        mapping[uwi] = matches

    return mapping


# ============================================================
# ANALYTICS
# ============================================================


def ratio_frame(df2, df1, cohort_map):

    rows = []

    for _, row in df2.iterrows():

        uwi = row["uwi"]

        matches = cohort_map.get(uwi, [])

        comp = df1[df1["uwi"].isin(matches)]

        med_eur = comp["eur_bbl"].median()
        med_ip = comp["ip90_bpd"].median()

        rows.append({
            "uwi": uwi,
            "eur_ratio": row["eur_bbl"] / med_eur if med_eur > 0 else np.nan,
            "ip90_ratio": row["ip90_bpd"] / med_ip if med_ip > 0 else np.nan,
            "n_comparators": len(comp),
        })

    return pd.DataFrame(rows)


def summary_stats(values):

    vals = finite(values)

    if len(vals) == 0:
        return {}

    return {
        "n": len(vals),
        "P90": np.percentile(vals, 90),
        "P75": np.percentile(vals, 75),
        "P50": np.percentile(vals, 50),
        "P25": np.percentile(vals, 25),
        "P10": np.percentile(vals, 10),
        "mean": np.mean(vals),
    }


# ============================================================
# PLOTS
# ============================================================


def density_plot(data1, data2, title, xlabel):

    fig = go.Figure()

    datasets = [
        (data1, "1-Mile", COLORS["1M"]),
        (data2, "2-Mile", COLORS["2M"]),
    ]

    for data, label, color in datasets:

        vals = finite(data)

        if len(vals) < 3:
            continue

        kde = gaussian_kde(vals)

        x = np.linspace(vals.min() * 0.8, vals.max() * 1.2, 300)

        y = kde(x)
        y = y / y.max()

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            fill="tozeroy",
            name=f"{label} (n={len(vals)})",
            line=dict(color=color, width=2.5),
        ))

        for p in [10, 25, 50, 75, 90]:

            pv = np.percentile(vals, PCT_MAP[p])

            fig.add_vline(
                x=pv,
                line_dash="dot",
                line_color=color,
                opacity=0.6,
            )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        xaxis_title=xlabel,
        yaxis_title="Density",
        height=400,
    )

    return fig


def ecdf_plot(data1, data2, title, xlabel):

    fig = go.Figure()

    for vals, label, color in [
        (finite(data1), "1-Mile", COLORS["1M"]),
        (finite(data2), "2-Mile", COLORS["2M"]),
    ]:

        if len(vals) == 0:
            continue

        x = np.sort(vals)
        y = np.arange(1, len(x) + 1) / len(x)

        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=label,
            line=dict(color=color, width=2),
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=title,
        xaxis_title=xlabel,
        yaxis_title="ECDF",
        height=350,
    )

    fig.update_yaxes(range=[0, 1])

    return fig


# ============================================================
# MAIN
# ============================================================


def main():

    st.title("🛢️ 2-Mile Well Uplift & Type Curve Analysis")

    wells = load_wells()
    geoms = load_geometries()
    rtc_curves = load_rtc()

    wells = wells.merge(
        geoms[["uwi", "lateral_group"]],
        on="uwi",
        how="inner",
    )

    df1 = wells[wells["lateral_group"] == "1-Mile"]
    df2 = wells[wells["lateral_group"] == "2-Mile"]

    # ========================================================
    # SIDEBAR
    # ========================================================

    st.sidebar.header("Settings")

    corridor_width = st.sidebar.slider(
        "Corridor Half Width (m)",
        100,
        3000,
        900,
        100,
    )

    st.sidebar.markdown("---")

    st.sidebar.markdown(f"**b-factor:** {B_FIXED}")
    st.sidebar.markdown(f"**Flat Days:** {FLAT_DAYS}")
    st.sidebar.markdown(f"**Economic Limit:** {Q_LIMIT} bbl/d")

    # ========================================================
    # MATCHING
    # ========================================================

    cohort_map = build_cohort_map(
        df2,
        df1,
        geoms,
        corridor_width,
    )

    ratios = ratio_frame(df2, df1, cohort_map)

    gm_eur = geometric_mean(ratios["eur_ratio"])
    gm_ip = geometric_mean(ratios["ip90_ratio"])

    # ========================================================
    # TABS
    # ========================================================

    tab1, tab2 = st.tabs([
        "📊 Uplift Analysis",
        "📈 Type Curves",
    ])

    # ========================================================
    # TAB 1
    # ========================================================

    with tab1:

        st.subheader("Geometric Mean Uplift")

        c1, c2 = st.columns(2)

        c1.metric(
            "EUR Ratio GM",
            f"{gm_eur:.3f}" if np.isfinite(gm_eur) else "—",
        )

        c2.metric(
            "IP90 Ratio GM",
            f"{gm_ip:.3f}" if np.isfinite(gm_ip) else "—",
        )

        st.markdown("---")

        st.subheader("Per-Well Ratios")

        st.dataframe(
            ratios,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # TAB 2
    # ========================================================

    with tab2:

        st.subheader("Distribution Analysis")

        eur1 = finite(df1["eur_bbl"])
        eur2 = finite(df2["eur_bbl"])

        ip1 = finite(df1["ip90_bpd"])
        ip2 = finite(df2["ip90_bpd"])

        c1, c2 = st.columns(2)

        with c1:
            st.plotly_chart(
                density_plot(
                    eur1,
                    eur2,
                    "EUR Distribution",
                    "EUR (bbl)",
                ),
                use_container_width=True,
            )

        with c2:
            st.plotly_chart(
                density_plot(
                    ip1,
                    ip2,
                    "IP90 Distribution",
                    "IP90 (bbl/d)",
                ),
                use_container_width=True,
            )

        st.plotly_chart(
            ecdf_plot(
                eur1,
                eur2,
                "EUR ECDF",
                "EUR (bbl)",
            ),
            use_container_width=True,
        )

        st.markdown("---")

        st.subheader("Type Curve Builder")

        c1, c2 = st.columns(2)

        with c1:

            ip_pct = st.slider(
                "IP90 Percentile",
                1,
                99,
                50,
            )

            ip_eff = 100 - ip_pct

            qi = np.percentile(ip2, ip_eff)

            st.metric("qi (bbl/d)", f"{qi:,.1f}")

        with c2:

            eur_pct = st.slider(
                "EUR Percentile",
                1,
                99,
                50,
            )

            eur_eff = 100 - eur_pct

            eur = np.percentile(eur2, eur_eff)

            st.metric("EUR (Mbbl)", f"{eur/1000:,.1f}")

        user_curve = build_curve(qi, eur)

        # ====================================================
        # RATE PLOT
        # ====================================================

        st.subheader("Rate Curve")

        fig_rate = go.Figure()

        fig_rate.add_trace(go.Scatter(
            x=user_curve["t"],
            y=user_curve["q"],
            mode="lines",
            name="User Curve",
            line=dict(
                color=COLORS["USER"],
                width=4,
            ),
        ))

        for rtc in rtc_curves:

            fig_rate.add_trace(go.Scatter(
                x=rtc["t"],
                y=rtc["q"],
                mode="lines",
                name=rtc["name"],
                line=dict(dash="dash"),
            ))

        fig_rate.add_hline(
            y=Q_LIMIT,
            line_dash="dot",
        )

        fig_rate.update_layout(
            **PLOTLY_LAYOUT,
            height=550,
            title="Rate vs Time",
            xaxis_title="Days",
            yaxis_title="Rate (bbl/d)",
        )

        st.plotly_chart(
            fig_rate,
            use_container_width=True,
        )

        # ====================================================
        # CUMULATIVE
        # ====================================================

        st.subheader("Cumulative Curve")

        fig_cum = go.Figure()

        fig_cum.add_trace(go.Scatter(
            x=user_curve["t"],
            y=user_curve["N"] / 1000,
            mode="lines",
            name="User Curve",
            line=dict(
                color=COLORS["USER"],
                width=4,
            ),
        ))

        for rtc in rtc_curves:

            fig_cum.add_trace(go.Scatter(
                x=rtc["t"],
                y=rtc["N"] / 1000,
                mode="lines",
                name=rtc["name"],
                line=dict(dash="dash"),
            ))

        fig_cum.update_layout(
            **PLOTLY_LAYOUT,
            height=500,
            title="Cumulative vs Time",
            xaxis_title="Days",
            yaxis_title="Cumulative (Mbbl)",
        )

        st.plotly_chart(
            fig_cum,
            use_container_width=True,
        )

        # ====================================================
        # EXPORT
        # ====================================================

        st.markdown("---")

        st.subheader("Export")

        curve_df = pd.DataFrame({
            "day": user_curve["t"],
            "rate_bpd": user_curve["q"],
            "cum_bbl": user_curve["N"],
            "cum_mbbl": user_curve["N"] / 1000,
        })

        output = io.BytesIO()

        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:

            curve_df.to_excel(
                writer,
                index=False,
                sheet_name="Curve",
            )

            ratios.to_excel(
                writer,
                index=False,
                sheet_name="Ratios",
            )

        st.download_button(
            "📥 Download Excel",
            output.getvalue(),
            file_name="2mile_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()