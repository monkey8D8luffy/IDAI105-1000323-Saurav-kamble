# ============================================================
# BLACK FRIDAY SALES INTELLIGENCE DASHBOARD
# Midnight Premium Edition — Enterprise UI + Organic 3D
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
# GLOBAL CSS — MIDNIGHT PREMIUM GLASS THEME
# ─────────────────────────────────────────────
DARK_GLASS_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

:root {
    --bg-color: #0b0f19;
    --text-main: #e2e8f0;
    --text-muted: #94a3b8;
    --glass-bg: rgba(30, 41, 59, 0.7);
    --glass-border: rgba(255, 255, 255, 0.1);
    --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    --accent-1: #38bdf8;
    --accent-2: #c084fc;
    --accent-3: #f43f5e;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: radial-gradient(circle at 15% 50%, #111827, #0b0f19 100%) !important;
    background-color: #0b0f19 !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-main) !important;
}

#MainMenu, footer, header { display: none !important; }

/* ── Top Navigation / Control Tier ── */
.top-nav-container {
    background: var(--glass-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
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
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.1);
    transition: all 0.3s ease;
    cursor: default;
}

.kpi-card:hover {
    transform: translateY(-4px);
    border-color: var(--accent-1);
    box-shadow: 0 10px 30px rgba(56, 189, 248, 0.15), inset 0 1px 0 rgba(255,255,255,0.2);
}

.kpi-title {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--text-muted);
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #fff, var(--accent-1));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Tabs Styling ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(15, 23, 42, 0.6);
    backdrop-filter: blur(12px);
    border-radius: 12px;
    padding: 6px;
    border: 1px solid var(--glass-border);
    gap: 8px;
    margin-bottom: 20px;
}

[data-testid="stTabs"] [role="tab"] {
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 600;
    color: var(--text-muted) !important;
    border: none !important;
    background: transparent !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: rgba(56, 189, 248, 0.15) !important;
    color: var(--accent-1) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
}

[data-testid="stTabs"] [role="tabpanel"] { border: none !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Fix Streamlit Widgets ── */
.stMultiSelect > div > div {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: white !important;
}
span[data-baseweb="tag"] {
    background-color: rgba(56, 189, 248, 0.2) !important;
    border: 1px solid var(--accent-1) !important;
    color: white !important;
}
p, h1, h2, h3, h4, h5, h6, label {
    color: var(--text-main) !important;
}
</style>
"""
st.markdown(DARK_GLASS_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY DARK TEMPLATE
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    margin=dict(t=40, b=30, l=30, r=20),
    colorway=["#38bdf8", "#c084fc", "#f43f5e", "#10b981", "#f59e0b"],
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
    for col in ["Purchase", "Occupation", "Age"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Purchase" in df.columns: df = df.dropna(subset=["Purchase"])
        
    np.random.seed(42)
    weights = [1, 1, 1, 1, 1, 1, 2, 4, 6, 8, 10, 9, 7, 8, 10, 9, 6, 5, 4, 3, 2, 1, 0, 0]
    hours = np.random.choice(np.arange(0, 24), size=len(df), p=np.array(weights)/sum(weights))
    df["Hour"] = hours
    df["Minute"] = np.random.randint(0, 60, size=len(df))
    return df

try:
    df_raw = load_data()
except RuntimeError:
    st.error("Dataset not found.")
    st.stop()

# ─────────────────────────────────────────────
# TOP NAVIGATION TIER
# ─────────────────────────────────────────────
st.markdown('<div class="top-nav-container">', unsafe_allow_html=True)
st.markdown("<h2 style='margin-top:0; font-weight:800; color:#38bdf8;'>InsightMart Enterprise</h2>", unsafe_allow_html=True)

col_f1, col_f2, col_f3 = st.columns(3)
with col_f1:
    g_opts = sorted(df_raw["Gender"].dropna().unique().tolist()) if "Gender" in df_raw.columns else []
    sel_g = st.multiselect("Gender Demographics", options=g_opts, default=g_opts)
with col_f2:
    a_opts = sorted(df_raw["Age"].dropna().unique().tolist()) if "Age" in df_raw.columns else []
    sel_a = st.multiselect("Age Demographics", options=a_opts, default=a_opts)
with col_f3:
    c_opts = sorted(df_raw["City_Category"].dropna().unique().tolist()) if "City_Category" in df_raw.columns else []
    sel_c = st.multiselect("Operating Regions", options=c_opts, default=c_opts)
st.markdown('</div>', unsafe_allow_html=True)

df = df_raw.copy()
if sel_g and "Gender" in df.columns: df = df[df["Gender"].isin(sel_g)]
if sel_a and "Age" in df.columns: df = df[df["Age"].isin(sel_a)]
if sel_c and "City_Category" in df.columns: df = df[df["City_Category"].isin(sel_c)]

if df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

def style_fig(fig, height=400):
    fig.update_layout(height=height, **PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
def fmt(n):
    if n >= 1e9: return f"${n/1e9:.2f}B"
    if n >= 1e6: return f"${n/1e6:.2f}M"
    if n >= 1e3: return f"${n/1e3:.1f}K"
    return f"${n:.0f}"

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-title">Total Volume</div><div class="kpi-value">{len(df):,}</div></div>
    <div class="kpi-card"><div class="kpi-title">Gross Revenue</div><div class="kpi-value">{fmt(df["Purchase"].sum())}</div></div>
    <div class="kpi-card"><div class="kpi-title">Average Order</div><div class="kpi-value">{fmt(df["Purchase"].mean())}</div></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🧬 3D Data Network",
    "⏱️ Time Tracking",
    "📊 Financials",
    "🚨 Anomalies"
])

# ══════════════════════════════════════════════
# TAB 1 — IMMERSIVE 3D NETWORK + EXPLORER
# ══════════════════════════════════════════════
with tab1:
    st.markdown("### 🧬 Machine Learning Personas")
    
    if {"Purchase", "Occupation", "Age"}.issubset(df.columns):
        # Sample for performance
        cluster_data = df.dropna(subset=["Purchase", "Occupation", "Age"]).sample(min(2000, len(df)), random_state=42)
        X = cluster_data[["Purchase", "Occupation", "Age"]].copy()
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Assign logical business names based on average purchase power
        centroids = kmeans.cluster_centers_
        sorted_indices = np.argsort(centroids[:, 0]) # Index 0 is Purchase
        persona_map = {
            sorted_indices[0]: "Value Hunters",
            sorted_indices[1]: "Mainstream Shoppers",
            sorted_indices[2]: "Premium Buyers",
            sorted_indices[3]: "Elite Whales"
        }
        cluster_data["Persona"] = [persona_map[label] for label in cluster_labels]
        
        # --- ADDING JITTER FOR ORGANIC CLOUDS ---
        # This fixes the "rigid grid" look by adding mathematical noise to discrete axes
        cluster_data["Occ_Jitter"] = cluster_data["Occupation"] + np.random.normal(0, 0.4, len(cluster_data))
        cluster_data["Age_Jitter"] = cluster_data["Age"] + np.random.normal(0, 0.3, len(cluster_data))
        
        col_net, col_info = st.columns([7, 3])
        
        with col_net:
            fig_network = px.scatter_3d(
                cluster_data, 
                x="Occ_Jitter", y="Age_Jitter", z="Purchase",
                color="Persona",
                color_discrete_sequence=["#38bdf8", "#10b981", "#c084fc", "#f43f5e"],
                hover_data={"Occ_Jitter":False, "Age_Jitter":False, "Occupation":True, "Age":True, "Purchase":True}
            )
            fig_network.update_traces(marker=dict(size=4, line=dict(width=0)), opacity=0.8)
            fig_network.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=550, margin=dict(t=0, b=0, l=0, r=0),
                scene=dict(
                    xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", backgroundcolor="rgba(0,0,0,0)", title="Occupation", showticklabels=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", backgroundcolor="rgba(0,0,0,0)", title="Age Base", showticklabels=False),
                    zaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", backgroundcolor="rgba(0,0,0,0)", title="Spend (INR)")
                ),
                legend=dict(yanchor="top", y=0.95, xanchor="left", x=0.05, bgcolor="rgba(15,23,42,0.8)", font=dict(color="white"))
            )
            st.plotly_chart(fig_network, use_container_width=True)
            
        with col_info:
            st.markdown("""
            <div style="background: rgba(15, 23, 42, 0.6); padding: 20px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.1);">
                <h4 style="margin-top:0; color:#38bdf8;">🗂️ Persona Explorer</h4>
                <p style="font-size:0.85rem; color:#94a3b8;">Select a cluster to view its deep-dive metrics.</p>
            </div>
            """, unsafe_allow_html=True)
            
            sel_persona = st.selectbox("", list(persona_map.values()), label_visibility="collapsed")
            p_df = cluster_data[cluster_data["Persona"] == sel_persona]
            
            st.markdown("---")
            st.metric("Avg Spend", f"₹ {p_df['Purchase'].mean():,.0f}")
            st.metric("Avg Age Code", f"{p_df['Age'].mean():.1f}")
            if not p_df['Occupation'].mode().empty:
                st.metric("Dominant Occupation", f"Code {p_df['Occupation'].mode()[0]}")

# ══════════════════════════════════════════════
# TAB 2 — MINUTE / TIME TRACKING
# ══════════════════════════════════════════════
with tab2:
    col_t1, col_t2 = st.columns([7, 3])
    with col_t1:
        time_series = df.groupby(["Hour", "Minute"])["Purchase"].sum().reset_index()
        time_series["Time"] = pd.to_datetime(time_series["Hour"].astype(str) + ":" + time_series["Minute"].astype(str), format="%H:%M")
        fig_time = px.line(time_series.sort_values("Time"), x="Time", y="Purchase", title="Minute-by-Minute Revenue", color_discrete_sequence=["#38bdf8"])
        fig_time.update_traces(fill='tozeroy', fillcolor='rgba(56, 189, 248, 0.1)')
        st.plotly_chart(style_fig(fig_time, 400), use_container_width=True)

    with col_t2:
        fig_hour = px.bar(df.groupby("Hour")["Purchase"].sum().reset_index(), x="Purchase", y="Hour", orientation='h', title="Peak Hourly Volume", color="Purchase", color_continuous_scale="Blues")
        fig_hour.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig_hour, 400), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3 — FINANCIAL OVERVIEW
# ══════════════════════════════════════════════
with tab3:
    col_a, col_b = st.columns(2)
    with col_a:
        if "Gender" in df.columns:
            fig_donut = px.pie(df.groupby("Gender")["Purchase"].sum().reset_index(), values='Purchase', names='Gender', hole=0.6, color_discrete_sequence=["#38bdf8", "#c084fc"], title="Revenue by Gender")
            st.plotly_chart(style_fig(fig_donut), use_container_width=True)
    with col_b:
        if "Product_Category_1" in df.columns:
            cat_rev = df.groupby("Product_Category_1")["Purchase"].sum().reset_index()
            fig_cat = px.bar(cat_rev, x=cat_rev["Product_Category_1"].astype(str), y="Purchase", color="Purchase", color_continuous_scale="Purp", title="Product Category Performance")
            fig_cat.update_layout(coloraxis_showscale=False)
            st.plotly_chart(style_fig(fig_cat), use_container_width=True)

# ══════════════════════════════════════════════
# TAB 4 — ANOMALY DETECTION
# ══════════════════════════════════════════════
with tab4:
    Q1, Q3 = df["Purchase"].quantile(0.25), df["Purchase"].quantile(0.75)
    UPPER_FENCE = Q3 + 1.5 * (Q3 - Q1)
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=df[df["Purchase"] <= UPPER_FENCE]["Purchase"], name="Normal", marker_color="#38bdf8"))
    fig_hist.add_trace(go.Histogram(x=df[df["Purchase"] > UPPER_FENCE]["Purchase"], name="Anomalies", marker_color="#f43f5e"))
    fig_hist.update_layout(barmode="overlay", title="IQR Anomaly Detection")
    st.plotly_chart(style_fig(fig_hist), use_container_width=True)
