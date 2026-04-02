# ============================================================
# BLACK FRIDAY SALES INTELLIGENCE DASHBOARD
# Light Liquid Glass Edition — Enterprise Analytics UI
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
    page_title="Black Friday Analytics Hub",
    page_icon="🛍️",
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
# DATA LOADER
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
    
    # Safely coerce Purchase and Occupation
    for col in ["Purchase", "Occupation"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Map Age to numeric codes for clustering/parallel coordinates
    if "Age" in df.columns:
        if df["Age"].dtype == object:
            age_mapping = {'0-17': 1, '18-25': 2, '26-35': 3, '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7}
            df["Age_Code"] = df["Age"].map(age_mapping).fillna(0)
        else:
            df["Age_Code"] = df["Age"]

    if "Purchase" in df.columns: df = df.dropna(subset=["Purchase"])
        
    # Synthetic Time Data for correlations and tracking
    np.random.seed(42)
    weights = [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 10, 9, 7, 8, 10, 9, 6, 5, 4, 3, 2, 1, 0, 0]
    hours = np.random.choice(np.arange(0, 24), size=len(df), p=np.array(weights)/sum(weights))
    df["Hour"] = hours
    return df

try:
    df_raw = load_data()
except RuntimeError:
    st.error("Dataset not found. Please upload your Black Friday CSV.")
    st.stop()

# ─────────────────────────────────────────────
# TOP NAVIGATION TIER
# ─────────────────────────────────────────────
st.markdown('<div class="top-nav-container">', unsafe_allow_html=True)
st.markdown("<h2 style='margin-top:0; font-weight:800; background: linear-gradient(135deg, #0ea5e9, #6366f1); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Enterprise Retail Miner</h2>", unsafe_allow_html=True)

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    g_opts = sorted(df_raw["Gender"].dropna().unique().tolist()) if "Gender" in df_raw.columns else []
    sel_g = st.multiselect("Gender Constraints", options=g_opts, default=g_opts)
with col_f2:
    a_opts = sorted(df_raw["Age"].dropna().unique().tolist()) if "Age" in df_raw.columns else []
    sel_a = st.multiselect("Age Demographics", options=a_opts, default=a_opts)
with col_f3:
    c_opts = sorted(df_raw["City_Category"].dropna().unique().tolist()) if "City_Category" in df_raw.columns else []
    sel_c = st.multiselect("Operating Regions", options=c_opts, default=c_opts)
st.markdown('</div>', unsafe_allow_html=True)

# Apply Filters
df = df_raw.copy()
if sel_g and "Gender" in df.columns: df = df[df["Gender"].isin(sel_g)]
if sel_a and "Age" in df.columns: df = df[df["Age"].isin(sel_a)]
if sel_c and "City_Category" in df.columns: df = df[df["City_Category"].isin(sel_c)]

if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

def style_fig(fig, height=450):
    fig.update_layout(height=height, **PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.05)", zerolinecolor="rgba(0,0,0,0.1)")
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.05)", zerolinecolor="rgba(0,0,0,0.1)")
    return fig

def fmt(n):
    if n >= 1e9: return f"₹{n/1e9:.2f}B"
    if n >= 1e6: return f"₹{n/1e6:.2f}M"
    if n >= 1e3: return f"₹{n/1e3:.1f}K"
    return f"₹{n:,.0f}"

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-title">Total Volume</div><div class="kpi-value">{len(df):,}</div></div>
    <div class="kpi-card"><div class="kpi-title">Gross Revenue</div><div class="kpi-value">{fmt(df["Purchase"].sum())}</div></div>
    <div class="kpi-card"><div class="kpi-title">Average Order Value</div><div class="kpi-value">{fmt(df["Purchase"].mean())}</div></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🧬 Persona Clustering",
    "📊 Professional Diagnostics",
    "🌐 Demographic Hierarchy",
    "🚨 Spend Anomalies"
])

# ══════════════════════════════════════════════
# TAB 1 — K-MEANS CLUSTERING (ATM STYLE)
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### 🧬 Machine Learning Customer Segmentation")
    st.write("Grouping customers into distinct performance profiles to optimize retail targeting.")
    
    cluster_cols = ["Purchase", "Occupation", "Age_Code"]
    if all(c in df.columns for c in cluster_cols):
        # Machine Learning Prep
        cluster_data = df.dropna(subset=cluster_cols).sample(min(3000, len(df)), random_state=42)
        X = cluster_data[cluster_cols].copy()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        X["Cluster"] = kmeans.fit_predict(X_scaled)
        
        # Smart Naming based on Purchase Volume
        centroids = X.groupby("Cluster")["Purchase"].mean()
        sorted_idx = centroids.sort_values().index
        c_names = {sorted_idx[0]: "Value Hunters (Low)", sorted_idx[1]: "Mainstream (Avg)", sorted_idx[2]: "Premium Buyers (High)", sorted_idx[3]: "Elite Whales (Critical)"}
        X["Profile"] = X["Cluster"].map(c_names)
        
        # Merge back readable Age strings if possible
        if "Age" in cluster_data.columns:
            X["Age_Str"] = cluster_data["Age"]
        else:
            X["Age_Str"] = X["Age_Code"]
        
        col_net, col_info = st.columns([7, 3])
        
        with col_net:
            fig_cluster = px.scatter_3d(
                X, x="Occupation", y="Age_Code", z="Purchase",
                color="Profile", opacity=0.8,
                hover_data={"Age_Code":False, "Age_Str":True, "Purchase":True, "Occupation":True},
                color_discrete_sequence=["#0ea5e9", "#6366f1", "#10b981", "#f59e0b"]
            )
            fig_cluster.update_traces(marker=dict(size=4, line=dict(width=0.5, color='white')))
            fig_cluster.update_layout(
                scene=dict(bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
                           yaxis=dict(gridcolor="rgba(0,0,0,0.05)", title="Age Code"),
                           zaxis=dict(gridcolor="rgba(0,0,0,0.05)")),
                margin=dict(t=0, b=0, l=0, r=0), height=550
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
            
        with col_info:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.8); padding: 20px; border-radius: 16px; border: 1px solid rgba(0,0,0,0.1);">
                <h4 style="margin-top:0; color:#0ea5e9;">🗂️ Persona Explorer</h4>
                <p style="font-size:0.85rem; color:#64748b;">Analyze operational demands.</p>
            </div>
            """, unsafe_allow_html=True)
            
            sel_profile = st.selectbox("", list(c_names.values()), label_visibility="collapsed")
            p_df = X[X["Profile"] == sel_profile]
            
            st.markdown("---")
            st.metric("Total Customers in Cohort", f"{len(p_df):,}")
            st.metric("Avg Spend Demand", fmt(p_df['Purchase'].mean()))
            if not p_df['Occupation'].mode().empty:
                st.metric("Primary Occupation", f"Code {int(p_df['Occupation'].mode()[0])}")

# ══════════════════════════════════════════════
# TAB 2 — PROFESSIONAL DIAGNOSTICS
# ══════════════════════════════════════════════
with tab2:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**Parallel Coordinates (Multi-Variable Flow)**")
        # Visualizing the flow between Age, Occupation, and Spend
        pc_cols = ["Purchase", "Occupation", "Age_Code"]
        if all(c in df.columns for c in pc_cols):
            # Sample for performance in parallel coordinates
            pc_df = df.dropna(subset=pc_cols).sample(min(1500, len(df)), random_state=42)
            fig_par = px.parallel_coordinates(
                pc_df, color="Purchase", dimensions=["Age_Code", "Occupation", "Purchase"],
                color_continuous_scale=[[0, '#0ea5e9'], [0.33, '#6366f1'], [0.66, '#10b981'], [1, '#f59e0b']]
            )
            fig_par.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig_par), use_container_width=True)
            
    with col_b:
        st.markdown("**Operational Correlation Matrix**")
        corr_cols = ["Purchase", "Occupation", "Age_Code", "Hour"]
        valid_corr = [c for c in corr_cols if c in df.columns]
        if len(valid_corr) >= 2:
            corr = df[valid_corr].corr()
            fig_corr = px.imshow(
                corr, text_auto=".2f", aspect="auto",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(style_fig(fig_corr), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — REGIONAL & DEMOGRAPHIC BREAKDOWN
# ══════════════════════════════════════════════
with tab3:
    if all(c in df.columns for c in ["City_Category", "Gender", "Purchase"]):
        st.markdown("### 🌐 Demographic Cash Distribution Hierarchy")
        # Sunburst hierarchy mapping City -> Gender -> Revenue
        sun_df = df.dropna(subset=["City_Category", "Gender", "Purchase"]).copy()
        sun_df["Network"] = "Total Market"
        
        fig_sun = px.sunburst(
            sun_df, path=["Network", "City_Category", "Gender"], values="Purchase",
            color="Purchase", color_continuous_scale="Teal"
        )
        fig_sun.update_traces(hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<extra></extra>")
        st.plotly_chart(style_fig(fig_sun, 550), use_container_width=True)
    else:
        st.info("Missing categorical data for Hierarchical Breakdown.")

# ══════════════════════════════════════════════
# TAB 4 — RISK & ANOMALIES
# ══════════════════════════════════════════════
with tab4:
    if "Purchase" in df.columns:
        st.markdown("### 🚨 High-Value Spend Anomalies")
        
        Q1, Q3 = df["Purchase"].quantile(0.25), df["Purchase"].quantile(0.75)
        UPPER_FENCE = Q3 + 1.5 * (Q3 - Q1)
        
        col_r1, col_r2 = st.columns([7, 3])
        with col_r1:
            # Box plot by City to locate anomalous clusters
            fig_box = px.box(
                df, x="City_Category" if "City_Category" in df.columns else None, y="Purchase",
                color="City_Category" if "City_Category" in df.columns else None,
                color_discrete_sequence=PLOTLY_LAYOUT["colorway"]
            )
            fig_box.add_hline(y=UPPER_FENCE, line_dash="dash", line_color="#f43f5e", annotation_text="Critical Anomaly Threshold")
            st.plotly_chart(style_fig(fig_box), use_container_width=True)
            
        with col_r2:
            anomaly_count = len(df[df['Purchase'] > UPPER_FENCE])
            st.error(f"**Action Required**\n\nDetected **{anomaly_count:,} transactions** exceeding expected parameters. Review these extreme high-rollers for fraud or VIP status.")
