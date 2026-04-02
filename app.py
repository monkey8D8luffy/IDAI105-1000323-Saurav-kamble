# ============================================================
# BLACK FRIDAY ENTERPRISE DASHBOARD
# Clean Light Theme — Fintech/Analytics Layout
# ============================================================

import os
import io
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Intelligence Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — CLEAN ENTERPRISE THEME
# ─────────────────────────────────────────────
ENTERPRISE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Main Background */
[data-testid="stAppViewContainer"] {
    background-color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

/* Hide Default Header/Footer */
#MainMenu, footer, header { display: none !important; }

/* KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}
.kpi-card {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.kpi-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0f172a;
}
.kpi-trend {
    font-size: 0.85rem;
    font-weight: 500;
    color: #10b981;
    margin-top: 4px;
}

/* Chart Containers */
.chart-container {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 16px;
}
.chart-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #0f172a;
    margin-bottom: 12px;
}

/* Typography Overrides */
h1, h2, h3, p, label {
    font-family: 'Inter', sans-serif !important;
}
</style>
"""
st.markdown(ENTERPRISE_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY TEMPLATE
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#334155"),
    margin=dict(t=10, b=30, l=10, r=10),
    colorway=["#0ea5e9", "#6366f1", "#10b981", "#f43f5e", "#f59e0b"],
)

# ─────────────────────────────────────────────
# DATA LOADER & TIME SYNTHESIS
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    candidates = [
        ("csv", "BlackFriday_Sample.csv"),
        ("zip", "BlackFriday_Cleaned.zip"),
        ("csv", "BlackFriday_Cleaned.csv"),
    ]

    df = None
    for kind, path in candidates:
        if not os.path.exists(path): continue
        try:
            if kind == "zip":
                with zipfile.ZipFile(path) as zf:
                    csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
                    if not csv_names: continue
                    with zf.open(csv_names[0]) as f:
                        df = pd.read_csv(io.BytesIO(f.read()))
            else:
                df = pd.read_csv(path)
            break
        except Exception: continue

    if df is None: raise RuntimeError("NO_FILE")

    df.columns = df.columns.str.strip().str.replace(" ", "_")
    for col in ["Purchase", "Occupation"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Purchase" in df.columns: df = df.dropna(subset=["Purchase"])
        
    # Synthetic Time Data for the Line Chart (Sum of probabilities = 1.0 exactly)
    np.random.seed(42)
    weights = [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 10, 9, 7, 8, 10, 9, 6, 5, 4, 3, 2, 1, 0, 0]
    p_norm = np.array(weights) / sum(weights)
    df["Hour"] = np.random.choice(np.arange(0, 24), size=len(df), p=p_norm)
    return df

try:
    df_raw = load_data()
except RuntimeError:
    st.error("Dataset not found. Please upload BlackFriday_Sample.csv")
    st.stop()

# ─────────────────────────────────────────────
# SETTING MENU (SIDEBAR)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#0f172a; margin-top:0;'>🛍️ Retail Intelligence Hub</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:0.9rem;'>Data Filters & Controls</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    c_opts = sorted(df_raw["City_Category"].dropna().unique().tolist()) if "City_Category" in df_raw.columns else []
    sel_c = st.multiselect("Operating Regions", options=c_opts, default=c_opts)
    
    a_opts = sorted(df_raw["Age"].dropna().unique().tolist()) if "Age" in df_raw.columns else []
    sel_a = st.multiselect("Age Demographics", options=a_opts, default=a_opts)
    
    g_opts = sorted(df_raw["Gender"].dropna().unique().tolist()) if "Gender" in df_raw.columns else []
    sel_g = st.multiselect("Gender Profiles", options=g_opts, default=g_opts)

# Apply Filters
df = df_raw.copy()
if sel_c and "City_Category" in df.columns: df = df[df["City_Category"].isin(sel_c)]
if sel_a and "Age" in df.columns: df = df[df["Age"].isin(sel_a)]
if sel_g and "Gender" in df.columns: df = df[df["Gender"].isin(sel_g)]

if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

def style_fig(fig, height=350):
    fig.update_layout(height=height, **PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0")
    return fig

def fmt(n):
    if n >= 1e9: return f"${n/1e9:.2f}B"
    if n >= 1e6: return f"${n/1e6:.2f}M"
    if n >= 1e3: return f"${n/1e3:.1f}K"
    return f"${n:,.0f}"

# ─────────────────────────────────────────────
# MAIN DASHBOARD LAYOUT
# ─────────────────────────────────────────────
st.markdown("<h1 style='color:#0f172a; margin-top:-20px; font-size:2rem;'>Black Friday Sales Performance</h1>", unsafe_allow_html=True)

# 1. KPI ROW (Matching the 4 cards at the top)
total_tx = len(df)
total_rev = df["Purchase"].sum()
avg_ord = df["Purchase"].mean()
Q1, Q3 = df["Purchase"].quantile(0.25), df["Purchase"].quantile(0.75)
UPPER_FENCE = Q3 + 1.5 * (Q3 - Q1)
anomalies_count = len(df[df["Purchase"] > UPPER_FENCE])

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-title">Total Volume</div>
        <div class="kpi-value">{total_tx:,}</div>
        <div class="kpi-trend">↑ Filtered Set</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Gross Revenue</div>
        <div class="kpi-value">{fmt(total_rev)}</div>
        <div class="kpi-trend">↑ Total Spend</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Average Order</div>
        <div class="kpi-value">${avg_ord:,.0f}</div>
        <div class="kpi-trend">↑ Per Transaction</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Flagged Anomalies</div>
        <div class="kpi-value" style="color:#f43f5e;">{anomalies_count:,}</div>
        <div class="kpi-trend" style="color:#f43f5e;">⚠ Review Required</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. TRENDS ROW (Line Chart & Bar Chart)
r1c1, r1c2 = st.columns([6, 4])

with r1c1:
    st.markdown('<div class="chart-container"><div class="chart-header">Hourly Sales Trend</div>', unsafe_allow_html=True)
    hourly = df.groupby("Hour")["Purchase"].sum().reset_index()
    fig_line = px.line(hourly, x="Hour", y="Purchase", color_discrete_sequence=["#0ea5e9"])
    fig_line.update_traces(fill='tozeroy', fillcolor='rgba(14, 165, 233, 0.1)', line=dict(width=3))
    st.plotly_chart(style_fig(fig_line), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with r1c2:
    st.markdown('<div class="chart-container"><div class="chart-header">Revenue by Category</div>', unsafe_allow_html=True)
    if "Product_Category_1" in df.columns:
        cat_df = df.groupby("Product_Category_1")["Purchase"].sum().reset_index().sort_values("Purchase", ascending=False).head(10)
        cat_df["Category"] = cat_df["Product_Category_1"].astype(str)
        fig_bar = px.bar(cat_df, x="Purchase", y="Category", orientation='h', color="Purchase", color_continuous_scale="Blues")
        fig_bar.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig_bar), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 3. DIAGNOSTICS ROW (Bubble Chart & Scatter Plot)
r2c1, r2c2 = st.columns([5, 5])

with r2c1:
    st.markdown('<div class="chart-container"><div class="chart-header">Demographic Demand Distribution</div>', unsafe_allow_html=True)
    if "Age" in df.columns and "City_Category" in df.columns:
        bubble_df = df.groupby(["Age", "City_Category"]).agg(Total_Rev=("Purchase", "sum"), Volume=("Purchase", "count")).reset_index()
        fig_bubble = px.scatter(bubble_df, x="Age", y="City_Category", size="Volume", color="Total_Rev", color_continuous_scale="Teal", size_max=50)
        fig_bubble.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig_bubble), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with r2c2:
    st.markdown('<div class="chart-container"><div class="chart-header">Category Performance Matrix</div>', unsafe_allow_html=True)
    if "Product_Category_1" in df.columns:
        scat_df = df.groupby("Product_Category_1").agg(Avg_Spend=("Purchase", "mean"), Volume=("Purchase", "count")).reset_index()
        scat_df["Category"] = scat_df["Product_Category_1"].astype(str)
        fig_scat = px.scatter(scat_df, x="Volume", y="Avg_Spend", color="Category", size="Avg_Spend", color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(style_fig(fig_scat), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 4. DATA TABLE ROW (Anomalies)
st.markdown('<div class="chart-container"><div class="chart-header">Recent Anomalous Transactions (> Upper Fence)</div>', unsafe_allow_html=True)
anomaly_df = df[df["Purchase"] > UPPER_FENCE].sort_values("Purchase", ascending=False).head(50)
st.dataframe(
    anomaly_df[["User_ID", "Product_ID", "Gender", "Age", "City_Category", "Purchase"]],
    use_container_width=True,
    hide_index=True
)
st.markdown('</div>', unsafe_allow_html=True)
