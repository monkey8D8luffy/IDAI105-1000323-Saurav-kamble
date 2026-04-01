# ============================================================
# BLACK FRIDAY SALES INTELLIGENCE DASHBOARD
# Light Liquid Glass Edition — Enterprise UI + Animated 3D
# ============================================================

import os
import zipfile
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Black Friday Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — LIGHT LIQUID GLASS THEME
# ─────────────────────────────────────────────
LIQUID_GLASS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

:root {
    --bg-color: #f4f7f9;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --glass-bg: rgba(255, 255, 255, 0.75);
    --glass-border: rgba(255, 255, 255, 0.6);
    --glass-shadow: 0 8px 32px rgba(148, 163, 184, 0.15);
    --accent-1: #0ea5e9;
    --accent-2: #8b5cf6;
    --accent-3: #f43f5e;
}

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    background-attachment: fixed;
    font-family: 'Inter', sans-serif;
    color: var(--text-main);
}

#MainMenu, footer, header { display: none !important; }

/* ── Top Navigation / Control Tier ── */
.top-nav-container {
    background: var(--glass-bg);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--glass-border);
    border-radius: 24px;
    padding: 24px;
    margin-bottom: 30px;
    box-shadow: var(--glass-shadow);
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.kpi-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.9), rgba(255,255,255,0.5));
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    cursor: default;
}

.kpi-card:hover {
    transform: translateY(-5px) scale(1.03);
    box-shadow: 0 20px 40px rgba(14, 165, 233, 0.15);
    border-color: var(--accent-1);
}

.kpi-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Tabs Styling (Liquid Slider) ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(16px);
    border-radius: 50px;
    padding: 8px;
    border: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    gap: 10px;
    margin-bottom: 25px;
}

[data-testid="stTabs"] [role="tab"] {
    border-radius: 50px !important;
    padding: 10px 24px !important;
    font-weight: 600;
    color: var(--text-muted) !important;
    border: none !important;
    transition: all 0.3s ease !important;
    background: transparent !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #ffffff !important;
    color: var(--accent-1) !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08) !important;
}

[data-testid="stTabs"] [role="tabpanel"] { border: none !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Dropdowns / Selectboxes ── */
.stMultiSelect > div > div {
    background: rgba(255,255,255,0.8) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px);
    color: var(--text-main) !important;
}
</style>
"""
st.markdown(LIQUID_GLASS_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY LIGHT TEMPLATE
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#0f172a"),
    margin=dict(t=40, b=30, l=30, r=20),
    colorway=["#0ea5e9", "#8b5cf6", "#f43f5e", "#10b981", "#f59e0b", "#06b6d4"],
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
        if not os.path.exists(path):
            continue
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
        except Exception:
            continue

    if df is None:
        raise RuntimeError("NO_FILE")

    df.columns = df.columns.str.strip().str.replace(" ", "_")
    for col in ["Purchase", "Occupation", "Age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Purchase" in df.columns:
        df = df.dropna(subset=["Purchase"])
        
    # --- GENERATING SYNTHETIC TIME DATA FOR MINUTE TRACKING ---
    # Since the original dataset lacks timestamps, we generate a normal distribution 
    # of purchase times simulating a typical Black Friday 24-hour cycle.
    np.random.seed(42)
    # Simulate peak hours around 14:00 (2 PM) and 20:00 (8 PM)
    hours = np.random.choice(
        np.arange(0, 24), 
        size=len(df), 
        p=[0.01,0.01,0.01,0.01,0.01,0.01,0.02,0.04,0.06,0.08,0.10,0.09,0.07,0.08,0.10,0.09,0.06,0.05,0.04,0.03,0.02,0.01,0.01,0.00]
    )
    minutes = np.random.randint(0, 60, size=len(df))
    
    df["Hour"] = hours
    df["Minute"] = minutes
    df["Time_String"] = df.apply(lambda row: f"{int(row['Hour']):02d}:{int(row['Minute']):02d}", axis=1)
    
    return df

try:
    df_raw = load_data()
except RuntimeError:
    st.error("Dataset not found. Please upload BlackFriday_Sample.csv or BlackFriday_Cleaned.zip.")
    st.stop()

# ─────────────────────────────────────────────
# TOP NAVIGATION TIER (LIQUID GLASS FILTERS)
# ─────────────────────────────────────────────
st.markdown('<div class="top-nav-container">', unsafe_allow_html=True)
st.markdown("<h2 style='margin-top:0; margin-bottom: 15px; font-weight:800; background: linear-gradient(135deg, #0ea5e9, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Enterprise Data Miner</h2>", unsafe_allow_html=True)

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    gender_opts = sorted(df_raw["Gender"].dropna().unique().tolist()) if "Gender" in df_raw.columns else []
    selected_genders = st.multiselect("Select Gender Constraints", options=gender_opts, default=gender_opts)

with col_f2:
    if "Age" in df_raw.columns:
        age_opts = sorted(df_raw["Age"].dropna().unique().tolist())
    else:
        age_opts = []
    selected_ages = st.multiselect("Select Age Demographics", options=age_opts, default=age_opts)

with col_f3:
    if "City_Category" in df_raw.columns:
        city_opts = sorted(df_raw["City_Category"].dropna().unique().tolist())
        selected_cities = st.multiselect("Select Operating Regions", options=city_opts, default=city_opts)
    else:
        selected_cities = []

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────
df = df_raw.copy()

if selected_genders and "Gender" in df.columns:
    df = df[df["Gender"].isin(selected_genders)]
if selected_ages and "Age" in df.columns:
    df = df[df["Age"].isin(selected_ages)]
if selected_cities and "City_Category" in df.columns:
    df = df[df["City_Category"].isin(selected_cities)]

if len(df) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()

def style_fig(fig, height=400):
    fig.update_layout(height=height, **PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.04)", zerolinecolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.04)", zerolinecolor="rgba(0,0,0,0.1)")
    return fig

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
total_txn = len(df)
total_rev = df["Purchase"].sum()
avg_order = df["Purchase"].mean()

def fmt(n):
    if n >= 1e9: return f"${n/1e9:.2f}B"
    if n >= 1e6: return f"${n/1e6:.2f}M"
    if n >= 1e3: return f"${n/1e3:.1f}K"
    return f"${n:.0f}"

kpi_html = f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-title">Total Volume</div>
        <div class="kpi-value">{total_txn:,}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Gross Revenue</div>
        <div class="kpi-value">{fmt(total_rev)}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title">Average Order</div>
        <div class="kpi-value">{fmt(avg_order)}</div>
    </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⏱️ Minute/Time Tracking",
    "📊 Financial Overview",
    "🧬 Animated 3D Network",
    "🚨 Anomaly Detection"
])

# ══════════════════════════════════════════════
# TAB 1 — MINUTE / TIME TRACKING
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### ⏱️ High-Resolution Temporal Analytics")
    
    col_t1, col_t2 = st.columns([7, 3])
    
    with col_t1:
        # Minute-by-Minute variations
        time_series = df.groupby(["Hour", "Minute"])["Purchase"].sum().reset_index()
        time_series["Time"] = pd.to_datetime(time_series["Hour"].astype(str) + ":" + time_series["Minute"].astype(str), format="%H:%M")
        time_series = time_series.sort_values("Time")
        
        fig_time = px.line(time_series, x="Time", y="Purchase", 
                           title="Minute-by-Minute Revenue Flow",
                           color_discrete_sequence=["#0ea5e9"])
        fig_time.update_traces(line=dict(width=2), fill='tozeroy', fillcolor='rgba(14, 165, 233, 0.1)')
        fig_time.update_xaxes(tickformat="%H:%M")
        st.plotly_chart(style_fig(fig_time, 450), use_container_width=True)

    with col_t2:
        # Hour Heatmap
        hourly_rev = df.groupby("Hour")["Purchase"].sum().reset_index()
        fig_hour = px.bar(hourly_rev, x="Purchase", y="Hour", orientation='h',
                          title="Peak Volume by Hour",
                          color="Purchase", color_continuous_scale="PuBu")
        fig_hour.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig_hour, 450), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2 — FINANCIAL OVERVIEW
# ══════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        if "Gender" in df.columns:
            gen_rev = df.groupby("Gender")["Purchase"].sum().reset_index()
            fig_donut = px.pie(gen_rev, values='Purchase', names='Gender', hole=0.6, 
                               color_discrete_sequence=["#0ea5e9", "#8b5cf6"], 
                               title="Revenue Distribution by Gender")
            st.plotly_chart(style_fig(fig_donut), use_container_width=True)

    with col_b:
        if "Product_Category_1" in df.columns:
            cat_rev = df.groupby("Product_Category_1")["Purchase"].sum().reset_index()
            cat_rev["Cat_Str"] = cat_rev["Product_Category_1"].astype(str)
            fig_cat = px.bar(cat_rev, x="Cat_Str", y="Purchase", color="Purchase", 
                             color_continuous_scale="Sunset", title="Variation Across Product Categories")
            fig_cat.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig_cat), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — ANIMATED 3D NETWORK
# ══════════════════════════════════════════════
with tab3:
    st.markdown("### 🧬 Time-Animated Cluster Network (K-Means)")
    st.write("Press play to animate how customer spending clusters evolve dynamically throughout the 24-hour cycle.")
    
    if {"Purchase", "Occupation"}.issubset(df.columns):
        # We sample tightly to ensure high-fps animation in browser
        cluster_data = df.dropna(subset=["Purchase", "Occupation"]).sample(min(1500, len(df)), random_state=42)
        
        # Prepare data for K-Means Clustering
        X = cluster_data[["Purchase", "Occupation"]].copy()
        if "Age" in df.columns and pd.api.types.is_numeric_dtype(df["Age"]):
            X["Age"] = cluster_data["Age"]
        else:
            X["Age"] = np.random.rand(len(X))
            
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        cluster_data["Cluster"] = kmeans.fit_predict(X).astype(str)
        
        # Sort by Hour to make the animation frame linear
        cluster_data = cluster_data.sort_values("Hour")

        fig_network = px.scatter_3d(
            cluster_data, 
            x="Occupation", 
            y="Age", 
            z="Purchase",
            color="Cluster",
            animation_frame="Hour",
            animation_group="User_ID",
            color_discrete_sequence=["#0ea5e9", "#f43f5e", "#8b5cf6", "#10b981"],
            title="Interactive 3D Evolution of Sales Data Clusters",
            range_x=[cluster_data["Occupation"].min(), cluster_data["Occupation"].max()],
            range_y=[cluster_data["Age"].min(), cluster_data["Age"].max()],
            range_z=[cluster_data["Purchase"].min(), cluster_data["Purchase"].max()]
        )
        
        fig_network.update_traces(marker=dict(size=8, line=dict(width=1, color='White')))
        fig_network.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", backgroundcolor="rgba(0,0,0,0)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", backgroundcolor="rgba(0,0,0,0)"),
                zaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)", backgroundcolor="rgba(0,0,0,0)")
            )
        )
        st.plotly_chart(fig_network, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — ANOMALY DETECTION
# ══════════════════════════════════════════════
with tab4:
    Q1 = df["Purchase"].quantile(0.25)
    Q3 = df["Purchase"].quantile(0.75)
    IQR = Q3 - Q1
    UPPER_FENCE = Q3 + 1.5 * IQR
    
    anomalies = df[df["Purchase"] > UPPER_FENCE]
    normal = df[df["Purchase"] <= UPPER_FENCE]
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=normal["Purchase"], name="Standard Patterns", marker_color="#0ea5e9"))
    fig_hist.add_trace(go.Histogram(x=anomalies["Purchase"], name="Anomalous Spikes", marker_color="#f43f5e"))
    fig_hist.update_layout(barmode="overlay", title="Algorithmic Anomaly Separation (IQR Filter)")
    st.plotly_chart(style_fig(fig_hist), use_container_width=True)
