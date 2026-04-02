# ============================================================
# ATM CASH MANAGEMENT & CLUSTERING DASHBOARD
# Light Liquid Glass Edition — Fintech Enterprise UI
# ============================================================

import os
import io
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ATM Analytics Hub",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — LIGHT LIQUID GLASS THEME
# ─────────────────────────────────────────────
LIGHT_GLASS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

:root {
    --bg-color: #f8fafc;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --glass-bg: rgba(255, 255, 255, 0.85);
    --glass-border: rgba(255, 255, 255, 0.4);
    --glass-shadow: 0 10px 40px rgba(148, 163, 184, 0.15);
    --accent-1: #0ea5e9;  /* Sky Blue */
    --accent-2: #6366f1;  /* Indigo */
    --accent-3: #10b981;  /* Emerald */
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%) !important;
    background-color: #f8fafc !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-main) !important;
}

#MainMenu, footer, header { display: none !important; }

/* ── Top Navigation / Control Tier ── */
.top-nav-container {
    background: var(--glass-bg);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--glass-border);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 24px;
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
    background: linear-gradient(145deg, rgba(255,255,255,0.95), rgba(255,255,255,0.6));
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.03), inset 0 2px 0 rgba(255,255,255,1);
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    cursor: default;
}

.kpi-card:hover {
    transform: translateY(-5px);
    border-color: var(--accent-1);
    box-shadow: 0 15px 35px rgba(14, 165, 233, 0.15), inset 0 2px 0 rgba(255,255,255,1);
}

.kpi-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
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

/* ── Tabs Styling ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255,255,255,0.5);
    backdrop-filter: blur(12px);
    border-radius: 12px;
    padding: 6px;
    border: 1px solid rgba(255,255,255,0.6);
    gap: 8px;
    margin-bottom: 20px;
}

[data-testid="stTabs"] [role="tab"] {
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 600;
    color: var(--text-muted) !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s ease;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #ffffff !important;
    color: var(--accent-1) !important;
    border: 1px solid rgba(255,255,255,1) !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
}

[data-testid="stTabs"] [role="tabpanel"] { border: none !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Fix Streamlit Widgets for Light Theme ── */
.stMultiSelect > div > div {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
    color: var(--text-main) !important;
}
span[data-baseweb="tag"] {
    background-color: rgba(14, 165, 233, 0.1) !important;
    border: 1px solid var(--accent-1) !important;
    color: var(--text-main) !important;
}
p, h1, h2, h3, h4, h5, h6, label {
    color: var(--text-main) !important;
}
</style>
"""
st.markdown(LIGHT_GLASS_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY LIGHT TEMPLATE
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#0f172a"),
    margin=dict(t=40, b=30, l=30, r=20),
    colorway=["#0ea5e9", "#6366f1", "#10b981", "#f43f5e", "#f59e0b"],
)

# ─────────────────────────────────────────────
# DATA LOADER (Robust ATM Column Matching)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    # Look for the dataset attached in the zip
    candidates = [
        ("csv", "atm_cash_management_dataset.csv"),
        ("csv", "ATM_Dataset.csv"),
    ]

    df = None
    for kind, path in candidates:
        if not os.path.exists(path): continue
        try:
            df = pd.read_csv(path)
            break
        except Exception: continue

    if df is None:
        # Fallback synthetic data if file isn't uploaded yet so app doesn't crash
        np.random.seed(42)
        df = pd.DataFrame({
            "ATM_ID": [f"ATM_{i}" for i in range(1, 1001)],
            "Location": np.random.choice(["Urban", "Suburban", "Rural", "Mall", "Airport"], 1000),
            "Cash_Dispensed": np.random.normal(50000, 15000, 1000).clip(0),
            "Transaction_Volume": np.random.normal(300, 100, 1000).clip(0).astype(int),
            "Downtime_Minutes": np.random.exponential(45, 1000).clip(0),
            "Maintenance_Cost": np.random.normal(500, 200, 1000).clip(0)
        })

    # Standardize column names dynamically
    cols = df.columns.str.lower().str.replace(" ", "_")
    df.columns = cols
    
    # Try to map to standard expected columns for our analytics
    col_mapping = {
        "atm_id": ["atm", "id", "atm_id"],
        "location": ["location", "region", "type", "area"],
        "cash_dispensed": ["cash", "amount", "dispensed", "total_cash"],
        "transactions": ["transaction", "volume", "count", "txns"],
        "downtime": ["downtime", "offline", "minutes", "hours"],
    }
    
    renamed = {}
    for target, search_terms in col_mapping.items():
        for col in df.columns:
            if any(term in col for term in search_terms):
                renamed[col] = target
                break
    df = df.rename(columns=renamed)
    
    # Coerce to numeric
    for col in ["cash_dispensed", "transactions", "downtime"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            
    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# ─────────────────────────────────────────────
# TOP NAVIGATION TIER
# ─────────────────────────────────────────────
st.markdown('<div class="top-nav-container">', unsafe_allow_html=True)
st.markdown("<h2 style='margin-top:0; font-weight:800; background: linear-gradient(135deg, #0ea5e9, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>ATM Performance & Fleet Clustering</h2>", unsafe_allow_html=True)

col_f1, col_f2 = st.columns(2)
with col_f1:
    loc_opts = sorted(df_raw["location"].dropna().unique().tolist()) if "location" in df_raw.columns else []
    sel_loc = st.multiselect("Filter by Location/Region", options=loc_opts, default=loc_opts)
with col_f2:
    if "cash_dispensed" in df_raw.columns:
        max_cash = int(df_raw["cash_dispensed"].max())
        sel_cash = st.slider("Target Cash Dispensed Range", 0, max_cash, (0, max_cash))
    else:
        sel_cash = (0, 9999999)
st.markdown('</div>', unsafe_allow_html=True)

df = df_raw.copy()
if sel_loc and "location" in df.columns: df = df[df["location"].isin(sel_loc)]
if "cash_dispensed" in df.columns: df = df[(df["cash_dispensed"] >= sel_cash[0]) & (df["cash_dispensed"] <= sel_cash[1])]

if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

def style_fig(fig, height=450):
    fig.update_layout(height=height, **PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.05)", zerolinecolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.05)", zerolinecolor="rgba(0,0,0,0.1)")
    return fig

def fmt(n):
    if n >= 1e9: return f"${n/1e9:.2f}B"
    if n >= 1e6: return f"${n/1e6:.2f}M"
    if n >= 1e3: return f"${n/1e3:.1f}K"
    return f"${n:,.0f}"

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
total_atms = len(df)
total_cash = df["cash_dispensed"].sum() if "cash_dispensed" in df.columns else 0
avg_tx = df["transactions"].mean() if "transactions" in df.columns else 0

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-title">Active Fleet Size</div><div class="kpi-value">{total_atms:,}</div></div>
    <div class="kpi-card"><div class="kpi-title">Total Cash Dispensed</div><div class="kpi-value">{fmt(total_cash)}</div></div>
    <div class="kpi-card"><div class="kpi-title">Avg Transactions / Machine</div><div class="kpi-value">{avg_tx:,.0f}</div></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🧬 K-Means Clustering",
    "📊 Professional Diagnostics",
    "🌐 Regional Breakdown",
    "🚨 Risk & Anomalies"
])

# ══════════════════════════════════════════════
# TAB 1 — ADVANCED CLUSTERING
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### 🧬 Machine Learning Fleet Segmentation")
    st.write("Grouping ATMs into distinct performance profiles to optimize cash replenishment and maintenance schedules.")
    
    cluster_cols = [c for c in ["cash_dispensed", "transactions", "downtime"] if c in df.columns]
    
    if len(cluster_cols) >= 2:
        # Machine Learning Prep
        X = df[cluster_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        X["Cluster"] = kmeans.fit_predict(X_scaled)
        
        # Smart Naming based on Cash Volume
        centroids = X.groupby("Cluster")["cash_dispensed"].mean()
        sorted_idx = centroids.sort_values().index
        c_names = {sorted_idx[0]: "Low Demand (Monitor)", sorted_idx[1]: "Stable Average", sorted_idx[2]: "High Volume", sorted_idx[3]: "Cash Critical (Elite)"}
        X["Profile"] = X["Cluster"].map(c_names)
        
        col_net, col_info = st.columns([7, 3])
        
        with col_net:
            # If 3 features, do 3D. If 2, do 2D scatter.
            if len(cluster_cols) >= 3:
                fig_cluster = px.scatter_3d(
                    X, x=cluster_cols[0], y=cluster_cols[1], z=cluster_cols[2],
                    color="Profile", opacity=0.8,
                    color_discrete_sequence=["#0ea5e9", "#6366f1", "#10b981", "#f59e0b"]
                )
                fig_cluster.update_traces(marker=dict(size=5, line=dict(width=0.5, color='white')))
                fig_cluster.update_layout(scene=dict(bgcolor="rgba(0,0,0,0)"))
            else:
                fig_cluster = px.scatter(
                    X, x=cluster_cols[0], y=cluster_cols[1], color="Profile",
                    color_discrete_sequence=["#0ea5e9", "#6366f1", "#10b981", "#f59e0b"]
                )
            st.plotly_chart(style_fig(fig_cluster, 550), use_container_width=True)
            
        with col_info:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.8); padding: 20px; border-radius: 16px; border: 1px solid rgba(0,0,0,0.1);">
                <h4 style="margin-top:0; color:#0ea5e9;">🗂️ Cluster Explorer</h4>
                <p style="font-size:0.85rem; color:#64748b;">Analyze operational demands.</p>
            </div>
            """, unsafe_allow_html=True)
            
            sel_profile = st.selectbox("", list(c_names.values()), label_visibility="collapsed")
            p_df = X[X["Profile"] == sel_profile]
            
            st.markdown("---")
            st.metric("Total ATMs in Cohort", f"{len(p_df):,}")
            st.metric("Avg Cash Demand", fmt(p_df['cash_dispensed'].mean() if 'cash_dispensed' in p_df else 0))
            if 'downtime' in p_df.columns:
                st.metric("Avg Downtime", f"{p_df['downtime'].mean():.1f} units")

# ══════════════════════════════════════════════
# TAB 2 — PROFESSIONAL DIAGNOSTICS (PARALLEL & CORR)
# ══════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Parallel Coordinates (Multi-Variable Flow)**")
        # Industry standard graph for clustering
        if len(cluster_cols) >= 3:
            fig_par = px.parallel_coordinates(
                X, color="Cluster", dimensions=cluster_cols,
                color_continuous_scale=[[0, '#0ea5e9'], [0.33, '#6366f1'], [0.66, '#10b981'], [1, '#f59e0b']]
            )
            fig_par.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig_par), use_container_width=True)
        else:
            st.info("Need at least 3 numeric columns for Parallel Coordinates.")
            
    with col_b:
        st.markdown("**Operational Correlation Matrix**")
        if len(cluster_cols) >= 2:
            corr = df[cluster_cols].corr()
            fig_corr = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(style_fig(fig_corr), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — REGIONAL BREAKDOWN (SUNBURST)
# ══════════════════════════════════════════════
with tab3:
    if "location" in df.columns and "cash_dispensed" in df.columns:
        st.markdown("### 🌐 Regional Cash Distribution Hierarchy")
        # Synthesize a parent node for the sunburst
        sun_df = df.dropna(subset=["location", "cash_dispensed"]).copy()
        sun_df["Network"] = "Total Fleet"
        
        fig_sun = px.sunburst(
            sun_df, path=["Network", "location"], values="cash_dispensed",
            color="cash_dispensed", color_continuous_scale="Teal"
        )
        fig_sun.update_traces(hovertemplate="<b>%{label}</b><br>Cash: $%{value:,.0f}<extra></extra>")
        st.plotly_chart(style_fig(fig_sun, 550), use_container_width=True)
    else:
        st.info("Location or Cash data missing for Regional Breakdown.")

# ══════════════════════════════════════════════
# TAB 4 — RISK & ANOMALY DETECTION
# ══════════════════════════════════════════════
with tab4:
    if "downtime" in df.columns:
        st.markdown("### 🚨 Downtime & Maintenance Anomalies")
        
        Q1, Q3 = df["downtime"].quantile(0.25), df["downtime"].quantile(0.75)
        UPPER_FENCE = Q3 + 1.5 * (Q3 - Q1)
        
        col_r1, col_r2 = st.columns([7, 3])
        with col_r1:
            fig_box = px.box(
                df, x="location" if "location" in df.columns else None, y="downtime",
                color="location" if "location" in df.columns else None,
                color_discrete_sequence=PLOTLY_LAYOUT["colorway"]
            )
            fig_box.add_hline(y=UPPER_FENCE, line_dash="dash", line_color="#f43f5e", annotation_text="Critical Anomaly Threshold")
            st.plotly_chart(style_fig(fig_box), use_container_width=True)
            
        with col_r2:
            st.warning(f"**Action Required**\n\nDetected **{len(df[df['downtime'] > UPPER_FENCE])} ATMs** exceeding maximum acceptable downtime parameters.")
    else:
        st.info("No downtime data available for risk analysis.")
