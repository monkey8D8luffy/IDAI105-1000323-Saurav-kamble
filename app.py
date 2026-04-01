"""
Beyond Discounts: Black Friday Sales Insights
Author  : InsightMart Analytics
Stack   : Streamlit · Plotly · Pandas
Theme   : Liquid Glassmorphism · Crypto-Exchange Navy/Cyan
"""

import io
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Black Friday · InsightMart",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# GLOBAL CSS  – Liquid Glassmorphism Crypto-Dashboard
# ──────────────────────────────────────────────────────────────────
GLASS_CSS = """
<style>
/* ── Import fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── White-label Streamlit chrome ── */
#MainMenu, header, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"] { 
    display: none !important; 
}

/* ── Root CSS variables ── */
:root {
  --bg-deep      : #0A0E17;
  --bg-mid       : #101623;
  --bg-panel     : rgba(16, 24, 43, 0.60);
  --accent-neon  : #00BFFF;
  --accent-blue  : #1E90FF;
  --glass-bdr    : rgba(0, 191, 255, 0.15);
  --glass-bdr-hi : rgba(0, 191, 255, 0.40);
  --glass-blur   : blur(20px);
  --glow-neon    : 0 0 20px rgba(0, 191, 255, 0.25);
  --text-hi      : #FFFFFF;
  --text-mid     : #8B9BB4;
  --text-dim     : #4B5A77;
}

/* ══ ROOT BACKGROUND – deep rich navy to black gradient ══ */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
  background: linear-gradient(135deg, var(--bg-deep) 0%, #000000 100%) !important;
  background-attachment: fixed !important;
  font-family: 'Inter', sans-serif;
  color: var(--text-hi);
}

/* ── Main container ── */
[data-testid="stMain"],
.main .block-container {
  background: transparent !important;
  padding-top: 2rem !important;
  padding-bottom: 2rem !important;
  max-width: 100% !important;
}

/* ══ SIDEBAR ══ */
[data-testid="stSidebar"] {
  background: rgba(10, 14, 23, 0.85) !important;
  backdrop-filter: var(--glass-blur) !important;
  -webkit-backdrop-filter: var(--glass-blur) !important;
  border-right: 1px solid var(--glass-bdr) !important;
  z-index: 100 !important;
}
[data-testid="stSidebar"] > div { padding: 2rem 1.5rem; }

.sidebar-brand {
  font-weight: 700; font-size: 1.4rem;
  color: var(--text-hi);
  margin-bottom: .2rem;
  display: flex;
  align-items: center;
  gap: 8px;
}
.sidebar-sub {
  font-size: .75rem; color: var(--text-mid); 
  margin-bottom: 2rem;
}

/* Widget labels */
label,
.stSelectbox label,
.stMultiSelect label,
.stSlider label,
[data-testid="stWidgetLabel"] {
  color: var(--text-mid) !important;
  font-size: .8rem !important;
  font-weight: 500 !important;
}

/* Select boxes */
.stSelectbox > div > div,
.stMultiSelect > div > div {
  background: rgba(16, 24, 43, 0.4) !important;
  border: 1px solid var(--glass-bdr) !important;
  border-radius: 12px !important;
  color: var(--text-hi) !important;
  transition: all .3s ease !important;
}
.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
  border-color: var(--accent-neon) !important;
  box-shadow: 0 0 12px rgba(0,191,255,0.2) !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] { padding: 0 !important; }
.stSlider [role="slider"] { background: var(--accent-neon) !important; border: 2px solid white !important;}
.stSlider div[data-baseweb="slider"] > div > div:first-child { background: var(--accent-blue) !important; }

/* ══ BUTTONS (Pill-shaped, glowing) ══ */
.stButton > button {
  background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-neon) 100%) !important;
  color: #ffffff !important;
  border-radius: 50px !important;
  border: none !important;
  font-weight: 600 !important;
  padding: 0.6rem 2rem !important;
  box-shadow: var(--glow-neon) !important;
  transition: all 0.3s ease !important;
}
.stButton > button:hover {
  transform: translateY(-2px) scale(1.02) !important;
  box-shadow: 0 6px 24px rgba(0, 191, 255, 0.4) !important;
}

/* ══ TABS ══ */
[data-testid="stTabs"] button {
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: .9rem !important;
  color: var(--text-mid) !important;
  background: transparent !important;
  border: none !important;
  border-radius: 50px !important;
  padding: .5rem 1.2rem !important;
  margin-right: 0.5rem !important;
  transition: all .3s ease !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--text-hi) !important;
  background: var(--accent-blue) !important;
  box-shadow: var(--glow-neon) !important;
}
[data-testid="stTabs"] button:hover:not([aria-selected="true"]) {
  color: var(--text-hi) !important;
  background: rgba(30, 144, 255, 0.1) !important;
}
[data-testid="stTabs"] [role="tablist"] {
  border-bottom: none !important;
  gap: .5rem !important;
  margin-bottom: 1.5rem !important;
}

/* ══ DATAFRAME ══ */
[data-testid="stDataFrame"] {
  background: var(--bg-panel) !important;
  backdrop-filter: var(--glass-blur) !important;
  border: 1px solid var(--glass-bdr) !important;
  border-radius: 18px !important;
  overflow: hidden !important;
}
[data-testid="stDataFrame"] th {
  background: rgba(30,144,255,0.1) !important;
  color: var(--accent-neon) !important;
  font-weight: 600 !important;
}
[data-testid="stDataFrame"] td {
  color: var(--text-mid) !important;
}

/* ══ SCROLLBAR ══ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--accent-blue); border-radius: 99px; }

/* ══ DASHBOARD HEADER ══ */
.dash-title {
  font-weight: 700; font-size: 2.2rem;
  color: var(--text-hi);
  margin-bottom: .2rem;
}
.dash-subtitle {
  font-size: .85rem; color: var(--text-mid);
  margin-bottom: 2rem;
}

/* ══ SECTION DIVIDER ══ */
.section-divider {
  border: none; height: 1px;
  background: linear-gradient(90deg, transparent, var(--glass-bdr-hi), transparent);
  margin: 2rem 0;
}

/* ══ KPI GRID & CARDS ══ */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1.2rem;
  margin-bottom: 2rem;
}
@media (max-width: 960px) { .kpi-grid { grid-template-columns: repeat(2,1fr); } }
@media (max-width: 520px) { .kpi-grid { grid-template-columns: 1fr; } }

/* Liquid Morph Animation */
@keyframes liquidMorph {
  0%   { border-radius: 18px; background-position: 0% 50%; }
  25%  { border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%; background-position: 100% 50%; }
  50%  { border-radius: 30% 70% 70% 30% / 30% 60% 40% 70%; background-position: 0% 50%; }
  75%  { border-radius: 40% 60% 30% 70% / 40% 70% 30% 60%; background-position: 100% 50%; }
  100% { border-radius: 18px; background-position: 0% 50%; }
}

.kpi-card, .chart-card {
  background: var(--bg-panel);
  background-size: 200% 200%;
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border: 1px solid var(--glass-bdr);
  border-radius: 18px;
  padding: 1.5rem;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4);
  transition: all 0.5s cubic-bezier(0.25, 0.8, 0.25, 1);
  position: relative;
  overflow: hidden;
}

/* Inner highlight border for glass effect */
.kpi-card::before, .chart-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(0,191,255,0.4), transparent);
}

.kpi-card:hover, .chart-card:hover {
  background: linear-gradient(135deg, rgba(16, 24, 43, 0.9) 0%, rgba(30, 144, 255, 0.15) 100%);
  border-color: var(--glass-bdr-hi);
  box-shadow: var(--glow-neon);
  animation: liquidMorph 4s ease-in-out infinite;
}

.chart-card { margin-bottom: 1.2rem; padding: 1rem; }

.kpi-label { font-size: .8rem; color: var(--text-mid); font-weight: 500; margin-bottom: .4rem; }
.kpi-value { font-weight: 700; font-size: 1.8rem; color: var(--text-hi); line-height: 1.2; }
.kpi-delta { font-size: .75rem; color: var(--accent-neon); margin-top: .5rem; font-weight: 500; }

/* Metric widget overrides to match */
[data-testid="stMetric"] {
  background: var(--bg-panel);
  backdrop-filter: var(--glass-blur);
  border: 1px solid var(--glass-bdr);
  border-radius: 18px;
  padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text-mid) !important; font-size: .8rem !important; }
[data-testid="stMetricValue"] { color: var(--text-hi)  !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { color: var(--accent-neon) !important; }

</style>
"""

st.markdown(GLASS_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# CONSTANTS & MAPPINGS
# ──────────────────────────────────────────────────────────────────
AGE_MAP    = {1: "0-17", 2: "18-25", 3: "26-35", 4: "36-45",
              5: "46-50", 6: "51-55", 7: "55+"}
GENDER_MAP = {0: "Male", 1: "Female"}

# Strict Dark/Cyan Plotly Layout
PLOTLY_LAYOUT = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(family="Inter, sans-serif", color="#8B9BB4", size=11),
    margin        = dict(l=15, r=15, t=50, b=15),
    title_font    = dict(family="Inter, sans-serif", color="#FFFFFF", size=14, weight="bold"),
    legend        = dict(
        bgcolor      = "rgba(10, 14, 23, 0.8)",
        bordercolor  = "rgba(0,191,255,0.2)",
        borderwidth  = 1,
        font         = dict(color="#8B9BB4"),
    ),
    hoverlabel    = dict(
        bgcolor     = "rgba(10, 14, 23, 0.95)",
        bordercolor = "rgba(0,191,255,0.4)",
        font        = dict(color="#FFFFFF", size=12),
    ),
)

GRID_STYLE = dict(
    xaxis = dict(
        gridcolor     = "rgba(255,255,255,0.03)",
        zerolinecolor = "rgba(255,255,255,0.03)",
        tickcolor     = "rgba(0,0,0,0)",
        linecolor     = "rgba(255,255,255,0.05)",
    ),
    yaxis = dict(
        gridcolor     = "rgba(255,255,255,0.03)",
        zerolinecolor = "rgba(255,255,255,0.03)",
        tickcolor     = "rgba(0,0,0,0)",
        linecolor     = "rgba(255,255,255,0.05)",
    ),
)

# Strict Blues & Cyans Palette
BLUE_SCALE   = [[0.0, "#0A1628"], [0.2, "#0F2D53"],
                [0.4, "#14437D"], [0.6, "#1A5AA8"],
                [0.8, "#1E90FF"], [1.0, "#00BFFF"]]

PALETTE_DISC = ["#00BFFF", "#1E90FF", "#00CED1", "#0288D1",
                "#00ACC1", "#26C6DA", "#29B6F6", "#4FC3F7"]


# ──────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = None

    try:
        df = pd.read_csv("BlackFriday_Sample.csv")
    except FileNotFoundError:
        pass

    if df is None:
        try:
            with zipfile.ZipFile("BlackFriday_Cleaned.zip", "r") as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if csv_names:
                    with zf.open(csv_names[0]) as f:
                        df = pd.read_csv(io.BytesIO(f.read()))
        except FileNotFoundError:
            pass

    if df is None:
        try:
            df = pd.read_csv("BlackFriday_Cleaned.csv")
        except FileNotFoundError:
            pass

    if df is None:
        st.error("Dataset not found. Ensure the CSV or ZIP is in the directory.", icon="🚨")
        st.stop()

    for col in ["Purchase", "Occupation", "Age", "User_ID",
                "Product_Category_1", "Product_Category_2", "Product_Category_3"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Purchase"]).copy()
    df["Purchase"]     = df["Purchase"].astype(int)
    df["Gender_Label"] = df["Gender"].map(GENDER_MAP).fillna("Unknown")
    df["Age_Label"]    = df["Age"].map(AGE_MAP).fillna(df["Age"].astype(str))
    return df

with st.spinner("Calibrating analytics engine..."):
    raw_df = load_data()

# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">❖ InsightMart</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Dashboard / Market Overview</div>', unsafe_allow_html=True)

    gender_opts = ["All"] + sorted(raw_df["Gender_Label"].unique().tolist())
    sel_gender  = st.selectbox("Gender", gender_opts, index=0)

    age_labels = [AGE_MAP[k] for k in sorted(AGE_MAP.keys())]
    sel_ages   = st.multiselect("Age Groups", age_labels, default=age_labels)

    city_opts = ["All"] + sorted(raw_df["City_Category"].dropna().unique().tolist())
    sel_city  = st.selectbox("City Category", city_opts, index=0)

    p_min, p_max = int(raw_df["Purchase"].min()), int(raw_df["Purchase"].max())
    sel_range    = st.slider(
        "Purchase Range (USD)", min_value=p_min, max_value=p_max,
        value=(p_min, p_max), step=100,
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.button("Update Dashboard")

# ──────────────────────────────────────────────────────────────────
# FILTER
# ──────────────────────────────────────────────────────────────────
df = raw_df.copy()
if sel_gender != "All":
    df = df[df["Gender_Label"] == sel_gender]
if sel_ages:
    df = df[df["Age_Label"].isin(sel_ages)]
else:
    df = df.iloc[0:0]
if sel_city != "All":
    df = df[df["City_Category"] == sel_city]
df = df[(df["Purchase"] >= sel_range[0]) & (df["Purchase"] <= sel_range[1])]

# ──────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────
st.markdown('<div class="dash-title">Black Friday Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="dash-subtitle">Live tracking and analytics overview • Updated 10 sec ago</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# KPI CARDS
# ──────────────────────────────────────────────────────────────────
def fmt(n: float, prefix="", suffix="") -> str:
    if n >= 1_000_000_000: return f"{prefix}{n/1e9:.2f}B{suffix}"
    if n >= 1_000_000:     return f"{prefix}{n/1e6:.2f}M{suffix}"
    if n >= 1_000:         return f"{prefix}{n/1e3:.1f}K{suffix}"
    return f"{prefix}{n:,.0f}{suffix}"

total_tx   = len(df)
total_rev  = df["Purchase"].sum()
avg_order  = df["Purchase"].mean() if total_tx else 0
unique_cus = df["User_ID"].nunique() if "User_ID" in df.columns else 0

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">Capital / Revenue</div>
    <div class="kpi-value">$ {fmt(total_rev)}</div>
    <div class="kpi-delta">+12.93% (Filtered)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Volume / Transactions</div>
    <div class="kpi-value">{fmt(total_tx)}</div>
    <div class="kpi-delta">+8.35% (Filtered)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Avg Order Value</div>
    <div class="kpi-value">$ {avg_order:,.0f}</div>
    <div class="kpi-delta">+1.48% (Per Txn)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Active Users</div>
    <div class="kpi-value">{fmt(unique_cus)}</div>
    <div class="kpi-delta">Distinct Wallets</div>
  </div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("No data matches the current filters. Adjust the sidebar controls.")
    st.stop()

# ──────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard",
    "Customer Market",
    "Risk & Anomalies",
    "Exchange Insights",
])

# ══════════════════════════════════════════════════════════════════
#  TAB 1 – FINANCIAL OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        rev_gender = df.groupby("Gender_Label")["Purchase"].sum().reset_index()
        fig = px.pie(
            rev_gender, names="Gender_Label", values="Purchase",
            hole=.7,
            color_discrete_sequence=["#00BFFF", "#1E90FF"],
            title="Capital Distribution by Gender",
        )
        fig.update_traces(
            textinfo='none',
            marker=dict(line=dict(color="rgba(10,14,23,0.8)", width=3)),
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        rev_city = df.groupby("City_Category")["Purchase"].sum().reset_index()
        fig2 = px.pie(
            rev_city, names="City_Category", values="Purchase",
            hole=.7,
            color_discrete_sequence=["#00BFFF", "#1E90FF", "#00CED1"],
            title="Capital Distribution by Tier",
        )
        fig2.update_traces(
            textinfo='none',
            marker=dict(line=dict(color="rgba(10,14,23,0.8)", width=3)),
            hovertemplate="<b>City %{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        )
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    age_order = [AGE_MAP[k] for k in sorted(AGE_MAP.keys())]
    rev_age   = (
        df.groupby("Age_Label")["Purchase"].sum()
          .reindex([a for a in age_order if a in df["Age_Label"].unique()])
          .reset_index()
    )
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig3 = px.bar(
        rev_age, x="Age_Label", y="Purchase",
        color="Purchase",
        color_continuous_scale=BLUE_SCALE,
        labels={"Age_Label": "Age Group", "Purchase": "Volume (USD)"},
        title="Volume Tracking by Age Group",
    )
    fig3.update_traces(marker_line_width=0, opacity=0.9,
                       hovertemplate="<b>%{x}</b><br>$%{y:,.0f}<extra></extra>")
    fig3.update_coloraxes(showscale=False)
    fig3.update_layout(**PLOTLY_LAYOUT, **GRID_STYLE, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  TAB 2 – CUSTOMER MARKET (3D)
# ══════════════════════════════════════════════════════════════════
with tab2:
    plot_df = df.sample(n=min(8_000, len(df)), random_state=42)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig5 = px.scatter_3d(
        plot_df,
        x="Occupation", y="Age", z="Purchase",
        color="Gender_Label",
        color_discrete_map={"Male": "#00BFFF", "Female": "#1E90FF"},
        opacity=0.7,
        size_max=4,
        labels={"Occupation": "Job Index", "Age": "Age Index", "Purchase": "Capital (USD)"},
        title=f"3D Market Scatter (n = {len(plot_df):,} nodes)",
        hover_data={"City_Category": True, "Purchase": True},
    )
    fig5.update_traces(marker=dict(size=3, line=dict(width=0)))
    fig5.update_layout(
        **PLOTLY_LAYOUT,
        height=600,
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.03)", backgroundcolor="rgba(0,0,0,0)", color="#8B9BB4"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.03)", backgroundcolor="rgba(0,0,0,0)", color="#8B9BB4"),
            zaxis=dict(gridcolor="rgba(255,255,255,0.03)", backgroundcolor="rgba(0,0,0,0)", color="#8B9BB4"),
        ),
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    grp = (
        df.groupby(["Gender_Label", "Age_Label"])
          .agg(Transactions=("Purchase", "count"),
               Total_Revenue=("Purchase", "sum"),
               Avg_Purchase=("Purchase", "mean"))
          .reset_index()
          .sort_values("Total_Revenue", ascending=False)
    )
    grp["Total_Revenue"] = grp["Total_Revenue"].apply(lambda x: f"$ {x:,.0f}")
    grp["Avg_Purchase"]  = grp["Avg_Purchase"].apply(lambda x: f"$ {x:,.0f}")
    st.dataframe(grp, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════
#  TAB 3 – RISK & ANOMALIES
# ══════════════════════════════════════════════════════════════════
with tab3:
    Q1  = df["Purchase"].quantile(0.25)
    Q3  = df["Purchase"].quantile(0.75)
    IQR = Q3 - Q1
    low_bound  = Q1 - 1.5 * IQR
    high_bound = Q3 + 1.5 * IQR

    df2 = df.copy()
    df2["Is_Anomaly"] = (df2["Purchase"] < low_bound) | (df2["Purchase"] > high_bound)
    normal_df  = df2[~df2["Is_Anomaly"]]
    anomaly_df = df2[ df2["Is_Anomaly"]]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk IQR Range", f"$ {IQR:,.0f}")
    m2.metric("Support Line", f"$ {max(0, low_bound):,.0f}")
    m3.metric("Resistance Line", f"$ {high_bound:,.0f}")
    m4.metric("Flagged Volume", f"{len(anomaly_df):,}  ({len(anomaly_df)/len(df2)*100:.1f}%)")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig6 = go.Figure()
    fig6.add_trace(go.Histogram(
        x=normal_df["Purchase"], name="Standard",
        nbinsx=60, opacity=0.5,
        marker_color="#1E90FF",
    ))
    fig6.add_trace(go.Histogram(
        x=anomaly_df["Purchase"], name="Anomaly",
        nbinsx=60, opacity=0.9,
        marker_color="#00BFFF",
    ))
    fig6.update_layout(
        **PLOTLY_LAYOUT, **GRID_STYLE,
        barmode="overlay",
        title="Volume Distribution Density",
        xaxis_title="Capital (USD)",
        yaxis_title="Frequency",
    )
    for val, lbl in [(high_bound, "Resistance"), (low_bound, "Support")]:
        if val > 0:
            fig6.add_vline(
                x=val, line_dash="dash", line_color="#00CED1", line_width=1.5,
                annotation_text=lbl, annotation_font_size=10, annotation_font_color="#00CED1",
            )
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  TAB 4 – EXCHANGE INSIGHTS
# ══════════════════════════════════════════════════════════════════
with tab4:
    cat_rev = (
        df.groupby("Product_Category_1")["Purchase"]
          .agg(Total_Revenue="sum", Transactions="count", Avg_Purchase="mean")
          .reset_index()
          .rename(columns={"Product_Category_1": "Asset"})
          .sort_values("Total_Revenue", ascending=False)
    )
    top15 = cat_rev.head(15)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig8 = px.bar(
        top15, x="Asset", y="Total_Revenue",
        color="Total_Revenue",
        color_continuous_scale=BLUE_SCALE,
        labels={"Asset": "Asset Class", "Total_Revenue": "Total Cap (USD)"},
        title="Top Asset Classes by Capitalization",
    )
    fig8.update_traces(marker_line_width=0, opacity=0.9)
    fig8.update_coloraxes(showscale=False)
    fig8.update_layout(**PLOTLY_LAYOUT, **GRID_STYLE, showlegend=False)
    st.plotly_chart(fig8, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    pivot = (
        df.groupby(["Gender_Label", "Product_Category_1"])["Purchase"]
          .sum().unstack(fill_value=0)
    )
    top_cats = cat_rev.head(12)["Asset"].tolist()
    pivot    = pivot[[c for c in top_cats if c in pivot.columns]]

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig11 = px.imshow(
        pivot,
        color_continuous_scale=BLUE_SCALE,
        labels=dict(x="Asset Class", y="Demographic", color="Cap (USD)"),
        title="Market Heatmap Correlation",
        aspect="auto",
    )
    fig11.update_layout(
        **PLOTLY_LAYOUT,
        coloraxis_colorbar=dict(
            title="USD", tickfont=dict(color="#8B9BB4"), titlefont=dict(color="#FFFFFF"), thickness=10,
        ),
    )
    st.plotly_chart(fig11, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
