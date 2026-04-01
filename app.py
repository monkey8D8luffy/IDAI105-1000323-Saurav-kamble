"""
Beyond Discounts: Black Friday Sales Insights
Author  : InsightMart Analytics
Stack   : Streamlit · Plotly · Pandas · scikit-learn
Theme   : Liquid Glassmorphism · Deep-Space Dark Mode
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
# GLOBAL CSS  – Liquid Glassmorphism
# ──────────────────────────────────────────────────────────────────
GLASS_CSS = """
<style>
/* ── Import fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── White-label Streamlit chrome ── */
#MainMenu, header, footer,
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"]          { display: none !important; }

/* ── Root variables ── */
:root {
  --bg-start   : #05070f;
  --bg-end     : #16103a;
  --accent-1   : #7b5ea7;
  --accent-2   : #3ecfcf;
  --accent-3   : #e05c8a;
  --glass-bg   : rgba(255,255,255,0.04);
  --glass-bdr  : rgba(255,255,255,0.09);
  --glass-blur : blur(22px);
  --text-hi    : #f0eeff;
  --text-mid   : #9c97c4;
  --radius     : 18px;
}

/* ── Full-page deep-space background ── */
html, body, [data-testid="stApp"],
[data-testid="stAppViewContainer"] {
  background: radial-gradient(ellipse at 20% 30%, #1a1140 0%, var(--bg-start) 60%),
              radial-gradient(ellipse at 80% 70%, #0d1f40 0%, var(--bg-start) 60%) !important;
  background-color: var(--bg-start) !important;
  font-family: 'DM Sans', sans-serif;
  color: var(--text-hi);
}

/* ── Animated grain overlay ── */
[data-testid="stApp"]::before {
  content: '';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  opacity: .035;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23noise)' opacity='1'/%3E%3C/svg%3E");
  animation: grain 8s steps(10) infinite;
}
@keyframes grain {
  0%,100%{transform:translate(0,0)}  10%{transform:translate(-2%,-3%)}
  20%{transform:translate(3%,2%)}    30%{transform:translate(-1%,4%)}
  40%{transform:translate(4%,-1%)}   50%{transform:translate(-3%,3%)}
  60%{transform:translate(2%,-4%)}   70%{transform:translate(-4%,1%)}
  80%{transform:translate(1%,3%)}    90%{transform:translate(3%,-2%)}
}

/* ── Main content area ── */
[data-testid="stMain"], .main .block-container {
  background: transparent !important;
  padding-top: 1.5rem !important;
  padding-bottom: 2rem !important;
}

/* ── Sidebar glass ── */
[data-testid="stSidebar"] {
  background: rgba(14, 10, 40, 0.72) !important;
  backdrop-filter: var(--glass-blur) !important;
  -webkit-backdrop-filter: var(--glass-blur) !important;
  border-right: 1px solid var(--glass-bdr) !important;
}
[data-testid="stSidebar"] > div { padding: 1.5rem 1.2rem; }

/* ── Sidebar headings / labels ── */
.sidebar-brand { font-family:'Syne',sans-serif; font-weight:800; font-size:1.3rem;
  background: linear-gradient(135deg,var(--accent-2),var(--accent-1));
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  margin-bottom:.2rem; }
.sidebar-sub { font-size:.72rem; color:var(--text-mid); letter-spacing:.08em;
  text-transform:uppercase; margin-bottom:1.6rem; }

/* ── ALL Streamlit widget labels ── */
label, .stSelectbox label, .stMultiSelect label,
.stSlider label, [data-testid="stWidgetLabel"] {
  color: var(--text-mid) !important;
  font-size: .78rem !important;
  letter-spacing: .06em !important;
  text-transform: uppercase !important;
  font-family: 'DM Sans', sans-serif !important;
}

/* ── Select boxes ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid var(--glass-bdr) !important;
  border-radius: 10px !important;
  color: var(--text-hi) !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { padding: 0 !important; }
.stSlider [role="slider"] { background: var(--accent-2) !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  font-size: .82rem !important;
  color: var(--text-mid) !important;
  letter-spacing: .05em !important;
  border-radius: 10px 10px 0 0 !important;
  transition: color .25s, background .25s !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--text-hi) !important;
  background: rgba(255,255,255,0.07) !important;
  border-bottom: 2px solid var(--accent-2) !important;
}
[data-testid="stTabs"] [role="tablist"] {
  border-bottom: 1px solid var(--glass-bdr) !important;
  gap: .25rem !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
  background: var(--glass-bg) !important;
  border: 1px solid var(--glass-bdr) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--accent-1); border-radius:99px; }

/* ── Dashboard title ── */
.dash-title {
  font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.1rem;
  background: linear-gradient(120deg, #ffffff 0%, var(--accent-2) 50%, var(--accent-3) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  line-height: 1.15; margin-bottom: .2rem;
}
.dash-subtitle {
  font-size: .82rem; color: var(--text-mid); letter-spacing: .1em;
  text-transform: uppercase; margin-bottom: 1.6rem;
}

/* ── Section divider ── */
.section-divider {
  border: none; height: 1px;
  background: linear-gradient(90deg, transparent, var(--glass-bdr), transparent);
  margin: 1.4rem 0;
}

/* ── KPI card grid ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  margin-bottom: 1.6rem;
}
@media (max-width: 900px) { .kpi-grid { grid-template-columns: repeat(2,1fr); } }
@media (max-width: 500px) { .kpi-grid { grid-template-columns: 1fr; } }

.kpi-card {
  background: var(--glass-bg);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border: 1px solid var(--glass-bdr);
  border-radius: var(--radius);
  padding: 1.25rem 1.4rem 1.1rem;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.07),
              0 8px 32px rgba(0,0,0,0.35);
  transition: transform .28s cubic-bezier(.34,1.56,.64,1),
              box-shadow .28s ease;
  cursor: default;
}
.kpi-card:hover {
  transform: scale(1.035) translateY(-2px);
}
.kpi-card:nth-child(1):hover { box-shadow: 0 0 28px rgba(62,207,207,.28), 0 8px 32px rgba(0,0,0,.35); }
.kpi-card:nth-child(2):hover { box-shadow: 0 0 28px rgba(123, 94,167,.35), 0 8px 32px rgba(0,0,0,.35); }
.kpi-card:nth-child(3):hover { box-shadow: 0 0 28px rgba(224, 92,138,.28), 0 8px 32px rgba(0,0,0,.35); }
.kpi-card:nth-child(4):hover { box-shadow: 0 0 28px rgba(255,185, 80,.28), 0 8px 32px rgba(0,0,0,.35); }

.kpi-icon  { font-size: 1.5rem; margin-bottom: .5rem; }
.kpi-label { font-size: .7rem; color: var(--text-mid); text-transform: uppercase;
  letter-spacing: .1em; margin-bottom: .25rem; }
.kpi-value { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.7rem;
  color: var(--text-hi); line-height: 1; }
.kpi-delta { font-size: .72rem; color: var(--accent-2); margin-top: .3rem; }

/* ── Section label ── */
.section-label {
  font-family: 'Syne', sans-serif; font-weight: 700;
  font-size: .9rem; color: var(--text-hi);
  text-transform: uppercase; letter-spacing: .1em;
  margin: 1.2rem 0 .6rem;
  display: flex; align-items: center; gap: .5rem;
}
.section-label::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--glass-bdr), transparent);
}

/* ── Chart wrapper glass card ── */
.chart-card {
  background: var(--glass-bg);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border: 1px solid var(--glass-bdr);
  border-radius: var(--radius);
  padding: .8rem .8rem .3rem;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.06), 0 8px 32px rgba(0,0,0,.3);
  margin-bottom: .8rem;
}

/* ── Anomaly badge ── */
.anomaly-badge {
  display:inline-block; padding:.2rem .7rem; border-radius:99px;
  font-size:.7rem; font-weight:600; letter-spacing:.06em;
  background: rgba(224,92,138,.18); color: var(--accent-3);
  border: 1px solid rgba(224,92,138,.3);
}
.normal-badge {
  display:inline-block; padding:.2rem .7rem; border-radius:99px;
  font-size:.7rem; font-weight:600; letter-spacing:.06em;
  background: rgba(62,207,207,.12); color: var(--accent-2);
  border: 1px solid rgba(62,207,207,.25);
}
</style>
"""

st.markdown(GLASS_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# CONSTANTS & MAPPINGS
# ──────────────────────────────────────────────────────────────────
AGE_MAP = {1: "0–17", 2: "18–25", 3: "26–35", 4: "36–45",
           5: "46–50", 6: "51–55", 7: "55+"}
GENDER_MAP = {0: "Male", 1: "Female"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#c9c3f0"),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(
        bgcolor="rgba(255,255,255,0.04)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
        font=dict(color="#c9c3f0"),
    ),
)

GRID_STYLE = dict(
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.06)"),
)

PALETTE_SEQ  = px.colors.sequential.Teal
PALETTE_DISC = ["#3ecfcf", "#7b5ea7", "#e05c8a", "#f5a623",
                "#4fc3f7", "#aed581", "#ff8a65", "#ce93d8"]


# ──────────────────────────────────────────────────────────────────
# DATA LOADING  – priority: Sample CSV → ZIP → Raw CSV
# ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = None

    # Priority 1 – lightweight sample (useful for fast dev)
    try:
        df = pd.read_csv("BlackFriday_Sample.csv")
    except FileNotFoundError:
        pass

    # Priority 2 – zipped CSV  (production dataset)
    if df is None:
        try:
            with zipfile.ZipFile("BlackFriday_Cleaned.zip", "r") as zf:
                csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                if csv_names:
                    with zf.open(csv_names[0]) as f:
                        df = pd.read_csv(io.BytesIO(f.read()))
        except FileNotFoundError:
            pass

    # Priority 3 – plain CSV
    if df is None:
        try:
            df = pd.read_csv("BlackFriday_Cleaned.csv")
        except FileNotFoundError:
            pass

    if df is None:
        st.error(
            "❌  Dataset not found.  "
            "Please place **BlackFriday_Cleaned.zip**, **BlackFriday_Cleaned.csv**, "
            "or **BlackFriday_Sample.csv** in the same directory as app.py.",
            icon="🚨",
        )
        st.stop()

    # ── Coerce dtypes ──
    for col in ["Purchase", "Occupation", "Age", "User_ID",
                "Product_Category_1", "Product_Category_2", "Product_Category_3"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Purchase"]).copy()
    df["Purchase"] = df["Purchase"].astype(int)

    # ── Human-readable labels ──
    df["Gender_Label"] = df["Gender"].map(GENDER_MAP).fillna("Unknown")
    df["Age_Label"]    = df["Age"].map(AGE_MAP).fillna(df["Age"].astype(str))

    return df


# ──────────────────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────────────────
with st.spinner("🔭  Initialising deep-space analytics…"):
    raw_df = load_data()

# ──────────────────────────────────────────────────────────────────
# SIDEBAR  – filters
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🛍 InsightMart</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Black Friday Analytics</div>', unsafe_allow_html=True)

    # Gender
    gender_opts = ["All"] + sorted(raw_df["Gender_Label"].unique().tolist())
    sel_gender  = st.selectbox("Gender", gender_opts, index=0)

    # Age
    age_labels  = [AGE_MAP[k] for k in sorted(AGE_MAP.keys())]
    sel_ages    = st.multiselect("Age Groups", age_labels, default=age_labels)

    # City
    city_opts = ["All"] + sorted(raw_df["City_Category"].dropna().unique().tolist())
    sel_city  = st.selectbox("City Category", city_opts, index=0)

    # Purchase range
    p_min, p_max = int(raw_df["Purchase"].min()), int(raw_df["Purchase"].max())
    sel_range = st.slider(
        "Purchase Range (₹)", min_value=p_min, max_value=p_max,
        value=(p_min, p_max), step=100,
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:.68rem;color:var(--text-mid);text-align:center;">'
        'Data Mining · Summative Assessment<br>'
        'InsightMart Analytics © 2025</p>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────
# FILTER DATA
# ──────────────────────────────────────────────────────────────────
df = raw_df.copy()

if sel_gender != "All":
    df = df[df["Gender_Label"] == sel_gender]

if sel_ages:
    df = df[df["Age_Label"].isin(sel_ages)]
else:
    df = df.iloc[0:0]   # empty but keeps schema

if sel_city != "All":
    df = df[df["City_Category"] == sel_city]

df = df[(df["Purchase"] >= sel_range[0]) & (df["Purchase"] <= sel_range[1])]

# ──────────────────────────────────────────────────────────────────
# DASHBOARD HEADER
# ──────────────────────────────────────────────────────────────────
st.markdown('<div class="dash-title">Beyond Discounts</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="dash-subtitle">Black Friday Sales Intelligence · InsightMart Analytics</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────
# KPI CARDS
# ──────────────────────────────────────────────────────────────────
def fmt(n: float, prefix="", suffix="") -> str:
    if n >= 1_000_000_000:
        return f"{prefix}{n/1e9:.2f}B{suffix}"
    if n >= 1_000_000:
        return f"{prefix}{n/1e6:.2f}M{suffix}"
    if n >= 1_000:
        return f"{prefix}{n/1e3:.1f}K{suffix}"
    return f"{prefix}{n:,.0f}{suffix}"


total_tx   = len(df)
total_rev  = df["Purchase"].sum()
avg_order  = df["Purchase"].mean() if total_tx else 0
unique_cus = df["User_ID"].nunique() if "User_ID" in df.columns else 0

kpi_html = f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-icon">🧾</div>
    <div class="kpi-label">Total Transactions</div>
    <div class="kpi-value">{fmt(total_tx)}</div>
    <div class="kpi-delta">↑ filtered dataset</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-icon">💰</div>
    <div class="kpi-label">Total Revenue</div>
    <div class="kpi-value">₹{fmt(total_rev)}</div>
    <div class="kpi-delta">gross purchase value</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-icon">🛒</div>
    <div class="kpi-label">Avg Order Value</div>
    <div class="kpi-value">₹{avg_order:,.0f}</div>
    <div class="kpi-delta">per transaction</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-icon">👤</div>
    <div class="kpi-label">Unique Customers</div>
    <div class="kpi-value">{fmt(unique_cus)}</div>
    <div class="kpi-delta">distinct User_ID</div>
  </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# GUARD – empty state
# ──────────────────────────────────────────────────────────────────
if df.empty:
    st.warning("⚠️  No data matches the current filters. Adjust the sidebar controls.", icon="🔭")
    st.stop()

# ──────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💹  Financial Overview",
    "🌐  3D Customer Mapping",
    "🔍  Anomaly Detection",
    "📦  Product Insights",
])


# ══════════════════════════════════════════════════════════════════
#  TAB 1 – FINANCIAL OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">Revenue Distribution</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    # ── Donut: Revenue by Gender ──
    with c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        rev_gender = df.groupby("Gender_Label")["Purchase"].sum().reset_index()
        fig = px.pie(
            rev_gender, names="Gender_Label", values="Purchase",
            hole=.62, color_discrete_sequence=["#3ecfcf", "#e05c8a"],
            title="Revenue by Gender",
        )
        fig.update_traces(
            textfont_size=11,
            marker=dict(line=dict(color="rgba(0,0,0,0.4)", width=2)),
            hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>",
        )
        fig.update_layout(**PLOTLY_LAYOUT, title_font_size=13)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Donut: City Tier ──
    with c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        rev_city = df.groupby("City_Category")["Purchase"].sum().reset_index()
        fig2 = px.pie(
            rev_city, names="City_Category", values="Purchase",
            hole=.62,
            color_discrete_sequence=["#7b5ea7", "#3ecfcf", "#f5a623"],
            title="Revenue by City Tier",
        )
        fig2.update_traces(
            textfont_size=11,
            marker=dict(line=dict(color="rgba(0,0,0,0.4)", width=2)),
            hovertemplate="<b>City %{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>",
        )
        fig2.update_layout(**PLOTLY_LAYOUT, title_font_size=13)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Spend by Demographics</div>',
                unsafe_allow_html=True)

    # ── Bar: Revenue by Age Group ──
    age_order = [AGE_MAP[k] for k in sorted(AGE_MAP.keys())]
    rev_age   = (
        df.groupby("Age_Label")["Purchase"].sum()
          .reindex([a for a in age_order if a in df["Age_Label"].unique()])
          .reset_index()
    )
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig3 = px.bar(
        rev_age, x="Age_Label", y="Purchase",
        color="Purchase", color_continuous_scale="Teal",
        labels={"Age_Label": "Age Group", "Purchase": "Total Revenue (₹)"},
        title="Revenue by Age Group",
        text_auto=".2s",
    )
    fig3.update_traces(
        marker_line_width=0,
        textfont_size=10,
        hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>",
    )
    fig3.update_coloraxes(showscale=False)
    fig3.update_layout(**PLOTLY_LAYOUT, **GRID_STYLE, title_font_size=13,
                       showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Bar: Revenue by Occupation ──
    rev_occ = (
        df.groupby("Occupation")["Purchase"].sum()
          .sort_values(ascending=True).reset_index()
    )
    rev_occ["Occupation"] = "Occ " + rev_occ["Occupation"].astype(str)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig4 = px.bar(
        rev_occ, x="Purchase", y="Occupation", orientation="h",
        color="Purchase", color_continuous_scale="Purpor",
        labels={"Occupation": "Occupation Code", "Purchase": "Total Revenue (₹)"},
        title="Revenue by Occupation",
        text_auto=".2s",
    )
    fig4.update_traces(marker_line_width=0, textfont_size=9)
    fig4.update_coloraxes(showscale=False)
    fig4.update_layout(**PLOTLY_LAYOUT, **GRID_STYLE, title_font_size=13,
                       showlegend=False, height=520)
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 2 – 3D CUSTOMER MAPPING
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">3D Behavioural Scatter</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:.78rem;color:var(--text-mid);margin-bottom:.8rem;">'
        'Each point is a transaction. Axes represent Occupation → Age → Purchase value. '
        'Color encodes Gender. Rotate / zoom freely.</p>',
        unsafe_allow_html=True,
    )

    # Sample for smooth rendering
    plot_df = df.sample(n=min(8_000, len(df)), random_state=42)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig5 = px.scatter_3d(
        plot_df,
        x="Occupation", y="Age", z="Purchase",
        color="Gender_Label",
        color_discrete_map={"Male": "#3ecfcf", "Female": "#e05c8a"},
        opacity=0.70,
        size_max=5,
        labels={"Occupation": "Occupation", "Age": "Age Code", "Purchase": "Purchase (₹)"},
        title=f"3D Purchase Landscape  (n = {len(plot_df):,} sampled transactions)",
        hover_data={"City_Category": True, "Purchase": True},
    )
    fig5.update_traces(
        marker=dict(size=3, line=dict(width=0)),
    )
    fig5.update_layout(
        **PLOTLY_LAYOUT,
        height=600,
        title_font_size=13,
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#9c97c4"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#9c97c4"),
            zaxis=dict(gridcolor="rgba(255,255,255,0.08)", color="#9c97c4"),
        ),
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Summary table ──
    st.markdown('<div class="section-label">Cluster-Level Aggregation</div>',
                unsafe_allow_html=True)
    grp = (
        df.groupby(["Gender_Label", "Age_Label"])
          .agg(Transactions=("Purchase", "count"),
               Total_Revenue=("Purchase", "sum"),
               Avg_Purchase=("Purchase", "mean"),
               Unique_Users=("User_ID", "nunique"))
          .reset_index()
          .sort_values("Total_Revenue", ascending=False)
    )
    grp["Total_Revenue"] = grp["Total_Revenue"].apply(lambda x: f"₹{x:,.0f}")
    grp["Avg_Purchase"]  = grp["Avg_Purchase"].apply(lambda x: f"₹{x:,.0f}")
    st.dataframe(grp, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 3 – ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">IQR-Based Anomaly Detection</div>',
                unsafe_allow_html=True)

    # IQR calculation
    Q1  = df["Purchase"].quantile(0.25)
    Q3  = df["Purchase"].quantile(0.75)
    IQR = Q3 - Q1
    low_bound  = Q1 - 1.5 * IQR
    high_bound = Q3 + 1.5 * IQR

    df["Is_Anomaly"] = (df["Purchase"] < low_bound) | (df["Purchase"] > high_bound)
    normal_df    = df[~df["Is_Anomaly"]]
    anomaly_df   = df[ df["Is_Anomaly"]]

    # ── Metric strip ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("IQR", f"₹{IQR:,.0f}")
    m2.metric("Lower Fence", f"₹{max(0, low_bound):,.0f}")
    m3.metric("Upper Fence", f"₹{high_bound:,.0f}")
    m4.metric("Anomalies Found", f"{len(anomaly_df):,}  ({len(anomaly_df)/len(df)*100:.1f}%)")

    st.markdown('<div class="section-label">Purchase Distribution: Normal vs Anomaly</div>',
                unsafe_allow_html=True)

    # ── Overlapping Histogram ──
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig6 = go.Figure()
    fig6.add_trace(go.Histogram(
        x=normal_df["Purchase"], name="Normal",
        nbinsx=60, opacity=0.72,
        marker_color="#3ecfcf",
        hovertemplate="Purchase: %{x}<br>Count: %{y}<extra>Normal</extra>",
    ))
    fig6.add_trace(go.Histogram(
        x=anomaly_df["Purchase"], name="Anomaly",
        nbinsx=60, opacity=0.82,
        marker_color="#e05c8a",
        hovertemplate="Purchase: %{x}<br>Count: %{y}<extra>Anomaly</extra>",
    ))
    fig6.update_layout(
        **PLOTLY_LAYOUT, **GRID_STYLE,
        barmode="overlay",
        title="Purchase Amount Distribution  (Normal vs Anomaly)",
        title_font_size=13,
        xaxis_title="Purchase (₹)",
        yaxis_title="Transaction Count",
    )
    # Fence lines
    for val, lbl, col in [
        (high_bound, "Upper Fence", "#f5a623"),
        (low_bound,  "Lower Fence", "#f5a623"),
    ]:
        if val > 0:
            fig6.add_vline(
                x=val, line_dash="dash", line_color=col, line_width=1.5,
                annotation_text=lbl, annotation_font_size=10,
                annotation_font_color=col,
            )
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Violin: density by anomaly status ──
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    df_vio = df.copy()
    df_vio["Status"] = df_vio["Is_Anomaly"].map({True: "Anomaly", False: "Normal"})
    fig7 = px.violin(
        df_vio, x="Gender_Label", y="Purchase", color="Status",
        color_discrete_map={"Normal": "#3ecfcf", "Anomaly": "#e05c8a"},
        box=True, points=False,
        labels={"Purchase": "Purchase (₹)", "Gender_Label": "Gender"},
        title="Purchase Density Violin  (by Gender & Anomaly Status)",
    )
    fig7.update_layout(**PLOTLY_LAYOUT, **GRID_STYLE, title_font_size=13)
    st.plotly_chart(fig7, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Top-20 anomalous spenders ──
    st.markdown('<div class="section-label">Top 20 Anomalous Transactions</div>',
                unsafe_allow_html=True)
    top_anom = (
        anomaly_df.sort_values("Purchase", ascending=False)
        .head(20)[["User_ID", "Product_ID", "Gender_Label", "Age_Label",
                   "City_Category", "Occupation", "Purchase"]]
        .copy()
    )
    top_anom["Purchase"] = top_anom["Purchase"].apply(lambda x: f"₹{x:,}")
    top_anom = top_anom.rename(columns={
        "Gender_Label": "Gender", "Age_Label": "Age",
        "City_Category": "City",
    })
    st.dataframe(top_anom, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 4 – PRODUCT INSIGHTS
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-label">Category Performance</div>',
                unsafe_allow_html=True)

    cat_rev = (
        df.groupby("Product_Category_1")["Purchase"]
          .agg(Total_Revenue="sum", Transactions="count", Avg_Purchase="mean")
          .reset_index()
          .rename(columns={"Product_Category_1": "Category"})
          .sort_values("Total_Revenue", ascending=False)
    )
    top15 = cat_rev.head(15)

    # ── Bar: Top-15 by Revenue ──
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig8 = px.bar(
        top15, x="Category", y="Total_Revenue",
        color="Total_Revenue", color_continuous_scale="Teal",
        text_auto=".2s",
        labels={"Category": "Product Category", "Total_Revenue": "Total Revenue (₹)"},
        title="Top 15 Product Categories by Revenue",
    )
    fig8.update_traces(marker_line_width=0, textfont_size=10)
    fig8.update_coloraxes(showscale=False)
    fig8.update_layout(**PLOTLY_LAYOUT, **GRID_STYLE, title_font_size=13, showlegend=False)
    st.plotly_chart(fig8, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)

    # ── Donut: Transaction volume by category ──
    with c3:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig9 = px.pie(
            top15.head(8), names="Category", values="Transactions",
            hole=.60,
            color_discrete_sequence=PALETTE_DISC,
            title="Transaction Volume (Top 8 Categories)",
        )
        fig9.update_traces(
            textfont_size=10,
            marker=dict(line=dict(color="rgba(0,0,0,0.4)", width=2)),
            hovertemplate="<b>Cat %{label}</b><br>%{value:,} txns<br>%{percent}<extra></extra>",
        )
        fig9.update_layout(**PLOTLY_LAYOUT, title_font_size=13)
        st.plotly_chart(fig9, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Bubble: Avg Purchase vs Total Revenue (sized by volume) ──
    with c4:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        fig10 = px.scatter(
            top15, x="Avg_Purchase", y="Total_Revenue",
            size="Transactions", color="Total_Revenue",
            color_continuous_scale="Purpor",
            hover_name="Category",
            labels={"Avg_Purchase": "Avg Purchase (₹)", "Total_Revenue": "Total Revenue (₹)",
                    "Transactions": "Vol"},
            title="Avg Purchase vs Revenue  (bubble = volume)",
            text="Category",
        )
        fig10.update_traces(
            textfont_size=8, textposition="top center",
            marker=dict(line=dict(width=0)),
        )
        fig10.update_coloraxes(showscale=False)
        fig10.update_layout(**PLOTLY_LAYOUT, **GRID_STYLE, title_font_size=13, showlegend=False)
        st.plotly_chart(fig10, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Heatmap: Gender vs Product Category (px.imshow) ──
    st.markdown('<div class="section-label">Gender × Category Heatmap</div>',
                unsafe_allow_html=True)
    pivot = (
        df.groupby(["Gender_Label", "Product_Category_1"])["Purchase"]
          .sum().unstack(fill_value=0)
    )
    # keep top-12 categories for readability
    top_cats = cat_rev.head(12)["Category"].tolist()
    pivot = pivot[[c for c in top_cats if c in pivot.columns]]

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    fig11 = px.imshow(
        pivot,
        color_continuous_scale="Teal",
        labels=dict(x="Product Category", y="Gender", color="Revenue (₹)"),
        title="Revenue Heatmap  — Gender × Product Category (Top 12)",
        text_auto=".2s",
        aspect="auto",
    )
    fig11.update_layout(**PLOTLY_LAYOUT, title_font_size=13,
                        coloraxis_colorbar=dict(
                            title="₹",
                            tickfont=dict(color="#9c97c4"),
                            titlefont=dict(color="#9c97c4"),
                        ))
    fig11.update_xaxes(side="bottom")
    st.plotly_chart(fig11, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Category insight table ──
    st.markdown('<div class="section-label">Full Category Summary</div>',
                unsafe_allow_html=True)
    disp = cat_rev.copy()
    disp["Total_Revenue"]  = disp["Total_Revenue"].apply(lambda x: f"₹{x:,.0f}")
    disp["Avg_Purchase"]   = disp["Avg_Purchase"].apply(lambda x: f"₹{x:,.0f}")
    disp["Transactions"]   = disp["Transactions"].apply(lambda x: f"{x:,}")
    disp = disp.rename(columns={
        "Category": "Product Category",
        "Total_Revenue": "Total Revenue",
        "Avg_Purchase": "Avg Purchase",
    })
    st.dataframe(disp, use_container_width=True, hide_index=True)
