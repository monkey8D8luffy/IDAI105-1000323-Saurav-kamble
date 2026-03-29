# ============================================================
# BLACK FRIDAY SALES INTELLIGENCE DASHBOARD
# Glass OS Edition — Glassmorphism + Plotly + Streamlit
# Author : Expert Python Developer & UI/UX Designer
# Deploy : Streamlit Cloud ready
# ============================================================

import os
import zipfile
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Black Friday Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS — GLASS OS THEME
# ─────────────────────────────────────────────

GLASS_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── CSS Variables ── */
:root {
    --bg-deep:      #05070f;
    --bg-mid:       #0b0f1e;
    --glass-bg:     rgba(255,255,255,0.04);
    --glass-border: rgba(255,255,255,0.10);
    --glass-glow:   rgba(110,86,255,0.18);
    --accent-1:     #6e56ff;
    --accent-2:     #ff5694;
    --accent-3:     #00e5c3;
    --text-primary: #eaf0ff;
    --text-muted:   rgba(200,210,240,0.55);
    --radius:       18px;
    --radius-sm:    10px;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 20% 0%, #16103a 0%, var(--bg-deep) 55%),
                radial-gradient(ellipse at 80% 100%, #1a0a2e 0%, transparent 60%);
    background-attachment: fixed;
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

/* animated grain overlay */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.35;
    animation: grain 0.5s steps(1) infinite;
}
@keyframes grain {
    0%,100%{transform:translate(0,0)}
    10%{transform:translate(-1%,-1%)}
    20%{transform:translate(1%,0)}
    30%{transform:translate(0,1%)}
    40%{transform:translate(-1%,1%)}
    50%{transform:translate(1%,-1%)}
    60%{transform:translate(0,0)}
    70%{transform:translate(1%,1%)}
    80%{transform:translate(-1%,0)}
    90%{transform:translate(0,-1%)}
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, [data-testid="stToolbar"] { display: none !important; }
[data-testid="collapsedControl"] { color: var(--text-primary) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 35, 0.70) !important;
    backdrop-filter: blur(28px) saturate(160%) !important;
    -webkit-backdrop-filter: blur(28px) saturate(160%) !important;
    border-right: 1px solid var(--glass-border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebarNav"] { display: none; }

/* ── Block container ── */
.block-container {
    padding: 1.8rem 2.2rem 2rem !important;
    max-width: 1600px;
}

/* ── Glass card base ── */
.glass-card {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    backdrop-filter: blur(20px) saturate(140%);
    -webkit-backdrop-filter: blur(20px) saturate(140%);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.06);
    padding: 1.4rem 1.6rem;
    transition: transform 0.28s cubic-bezier(.34,1.56,.64,1),
                box-shadow 0.28s ease,
                border-color 0.28s ease;
}
.glass-card:hover {
    transform: translateY(-4px) scale(1.018);
    box-shadow: 0 18px 48px rgba(0,0,0,0.5), 0 0 0 1px var(--accent-1),
                inset 0 1px 0 rgba(255,255,255,0.10);
    border-color: rgba(110,86,255,0.45);
}

/* ── KPI row ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.1rem;
    margin-bottom: 1.6rem;
}
@media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(2,1fr); } }
@media (max-width:  600px) { .kpi-grid { grid-template-columns: 1fr; } }

.kpi-card {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    backdrop-filter: blur(22px) saturate(160%);
    -webkit-backdrop-filter: blur(22px) saturate(160%);
    box-shadow: 0 6px 24px rgba(0,0,0,0.30),
                inset 0 1px 0 rgba(255,255,255,0.07);
    padding: 1.3rem 1.5rem;
    position: relative;
    overflow: hidden;
    cursor: default;
    transition: transform 0.30s cubic-bezier(.34,1.56,.64,1),
                box-shadow 0.30s ease,
                border-color 0.30s ease;
    animation: cardEntry 0.55s cubic-bezier(.22,1,.36,1) both;
}
.kpi-card::after {
    content:'';
    position:absolute;
    inset:0;
    background: radial-gradient(circle at 80% 20%, rgba(110,86,255,0.12), transparent 65%);
    pointer-events:none;
}
.kpi-card:hover {
    transform: translateY(-5px) scale(1.03);
    box-shadow: 0 20px 50px rgba(0,0,0,0.50),
                0 0 0 1.5px var(--accent-1),
                inset 0 1px 0 rgba(255,255,255,0.12);
    border-color: rgba(110,86,255,0.55);
}
.kpi-card:hover .kpi-value { text-shadow: 0 0 22px var(--accent-1); }

@keyframes cardEntry {
    from { opacity:0; transform: translateY(14px); }
    to   { opacity:1; transform: translateY(0); }
}
.kpi-label {
    font-family: 'DM Sans', sans-serif;
    font-size: .72rem;
    font-weight: 500;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: .45rem;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(135deg, #c4b8ff 0%, var(--accent-1) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    transition: text-shadow 0.3s ease;
    line-height: 1.1;
}
.kpi-delta {
    font-size: .78rem;
    color: var(--accent-3);
    margin-top: .35rem;
    font-weight: 500;
}
.kpi-icon {
    position: absolute;
    right: 1.2rem;
    top: 1.1rem;
    font-size: 1.55rem;
    opacity: 0.18;
    filter: blur(0.3px);
}

/* ── Section title ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.18rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 .8rem;
    letter-spacing: -.01em;
}
.section-sub {
    font-size: .82rem;
    color: var(--text-muted);
    margin-top: -.4rem;
    margin-bottom: 1rem;
}

/* ── Dashboard header ── */
.dash-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.6rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--glass-border);
    animation: fadeDown 0.5s ease both;
}
@keyframes fadeDown {
    from { opacity:0; transform:translateY(-12px); }
    to   { opacity:1; transform:translateY(0); }
}
.dash-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.85rem;
    font-weight: 800;
    background: linear-gradient(100deg,#fff 0%,#c4b8ff 50%,var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
}
.dash-subtitle {
    font-size: .85rem;
    color: var(--text-muted);
    margin-top: .2rem;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    padding: .28rem .75rem;
    background: rgba(110,86,255,0.15);
    border: 1px solid rgba(110,86,255,0.35);
    border-radius: 50px;
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #c4b8ff;
    margin-top: .4rem;
}

/* ── Tab styling ── */
[data-testid="stTabs"] [role="tablist"] {
    gap: .4rem;
    background: rgba(255,255,255,0.03);
    border-radius: 50px;
    padding: .35rem .4rem;
    border: 1px solid var(--glass-border);
    width: fit-content;
    margin-bottom: 1.4rem;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 50px !important;
    padding: .45rem 1.3rem !important;
    font-family: 'DM Sans', sans-serif;
    font-size: .82rem;
    font-weight: 500;
    color: var(--text-muted) !important;
    transition: all .22s ease !important;
    border: none !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-1), #9b7aff) !important;
    color: #fff !important;
    box-shadow: 0 4px 16px rgba(110,86,255,0.45) !important;
}
[data-testid="stTabs"] [role="tab"]:hover:not([aria-selected="true"]) {
    color: var(--text-primary) !important;
    background: rgba(255,255,255,0.05) !important;
}
/* Remove default tab underline indicator */
[data-testid="stTabs"] [role="tabpanel"] { border: none !important; }
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"]    { display: none !important; }

/* ── Sidebar widgets ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: rgba(255,255,255,0.06) !important;
    border-color: var(--glass-border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}
.stSlider > div > div > div > div { background: var(--accent-1) !important; }

/* ── Sidebar header ── */
.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    background: linear-gradient(135deg, #c4b8ff, var(--accent-2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: .2rem;
}
.sidebar-desc {
    font-size: .73rem;
    color: var(--text-muted);
    line-height: 1.5;
    margin-bottom: 1.2rem;
}
.sidebar-divider {
    height: 1px;
    background: var(--glass-border);
    margin: 1rem 0;
}
.filter-label {
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: .1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: .3rem;
}

/* ── Anomaly pill ── */
.anomaly-pill {
    display: inline-block;
    padding: .18rem .65rem;
    background: rgba(255,86,148,0.15);
    border: 1px solid rgba(255,86,148,0.4);
    border-radius: 50px;
    font-size: .72rem;
    color: var(--accent-2);
    font-weight: 600;
}
.normal-pill {
    display: inline-block;
    padding: .18rem .65rem;
    background: rgba(0,229,195,0.12);
    border: 1px solid rgba(0,229,195,0.35);
    border-radius: 50px;
    font-size: .72rem;
    color: var(--accent-3);
    font-weight: 600;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(110,86,255,0.35); border-radius: 4px; }

/* ── Plotly chart containers ── */
.js-plotly-plot { border-radius: var(--radius) !important; }
</style>
"""

st.markdown(GLASS_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY DARK TEMPLATE
# ─────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#c8d2f0"),
    margin=dict(t=44, b=36, l=36, r=24),
    legend=dict(
        bgcolor="rgba(255,255,255,0.04)",
        bordercolor="rgba(255,255,255,0.10)",
        borderwidth=1,
        font=dict(size=11),
    ),
    colorway=["#6e56ff","#ff5694","#00e5c3","#ffb547","#7dd3fc","#f472b6"],
)

ACCENT = ["#6e56ff","#ff5694","#00e5c3","#ffb547","#7dd3fc","#a78bfa"]

# ─────────────────────────────────────────────
# DATA LOADER  — bulletproof with caching
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    Tries three file candidates in order:
    1. BlackFriday_Sample.csv
    2. BlackFriday_Cleaned.zip  (first CSV inside the archive)
    3. BlackFriday_Cleaned.csv
    Returns a cleaned, feature-engineered DataFrame or raises RuntimeError.
    """
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
                    if not csv_names:
                        continue
                    with zf.open(csv_names[0]) as f:
                        df = pd.read_csv(io.BytesIO(f.read()))
            else:
                df = pd.read_csv(path)
            break  # success — stop trying
        except Exception:
            continue

    if df is None:
        raise RuntimeError("NO_FILE")

    # ── Normalise column names ──
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # ── Coerce numeric columns ──
    for col in ["Purchase", "Occupation", "Age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Drop rows without Purchase ──
    if "Purchase" in df.columns:
        df = df.dropna(subset=["Purchase"])

    # ── Age label cleanup (some datasets use '55+') ──
    if "Age" in df.columns and df["Age"].dtype == object:
        pass  # keep as string category

    # ── Derived columns ──
    if "Purchase" in df.columns:
        df["Purchase_K"] = df["Purchase"] / 1000          # for readability

    return df


# ── Try loading; show friendly error if missing ──
try:
    df_raw = load_data()
except RuntimeError:
    st.error(
        "**Dataset not found.** \n"
        "Please upload one of the following files to the app directory:  \n"
        "`BlackFriday_Sample.csv` · `BlackFriday_Cleaned.csv` · `BlackFriday_Cleaned.zip`"
    )
    st.stop()

# ─────────────────────────────────────────────
# SIDEBAR — FILTER CONTROL PANEL
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">🛍️ Black Friday<br>Intelligence</div>'
        '<div class="sidebar-desc">Interactive analytics dashboard powered by Glass OS.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Gender filter ──
    st.markdown('<div class="filter-label">👤 Gender</div>', unsafe_allow_html=True)
    gender_opts = ["All"] + sorted(df_raw["Gender"].dropna().unique().tolist()) if "Gender" in df_raw.columns else ["All"]
    gender_sel = st.selectbox("", gender_opts, key="gender_sel", label_visibility="collapsed")

    st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)

    # ── Age Group filter ──
    st.markdown('<div class="filter-label">🎂 Age Group</div>', unsafe_allow_html=True)
    if "Age" in df_raw.columns and df_raw["Age"].dtype == object:
        age_opts = ["All"] + sorted(df_raw["Age"].dropna().unique().tolist())
    elif "Age" in df_raw.columns:
        age_opts = ["All"] + sorted(df_raw["Age"].dropna().unique().tolist())
    else:
        age_opts = ["All"]
    age_sel = st.selectbox("", age_opts, key="age_sel", label_visibility="collapsed")

    st.markdown('<div style="height:.7rem"></div>', unsafe_allow_html=True)

    # ── City Category filter ──
    if "City_Category" in df_raw.columns:
        st.markdown('<div class="filter-label">🏙️ City Category</div>', unsafe_allow_html=True)
        city_opts = ["All"] + sorted(df_raw["City_Category"].dropna().unique().tolist())
        city_sel = st.selectbox("", city_opts, key="city_sel", label_visibility="collapsed")
    else:
        city_sel = "All"

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    # ── Purchase range slider ──
    if "Purchase" in df_raw.columns:
        st.markdown('<div class="filter-label">💳 Purchase Range (₹)</div>', unsafe_allow_html=True)
        p_min = int(df_raw["Purchase"].min())
        p_max = int(df_raw["Purchase"].max())
        p_range = st.slider(
            "",
            min_value=p_min,
            max_value=p_max,
            value=(p_min, p_max),
            step=500,
            label_visibility="collapsed",
        )
    else:
        p_range = (0, 99999999)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="filter-label">📋 Total records: {len(df_raw):,}</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────

df = df_raw.copy()

if gender_sel != "All" and "Gender" in df.columns:
    df = df[df["Gender"] == gender_sel]

if age_sel != "All" and "Age" in df.columns:
    df = df[df["Age"] == age_sel]

if city_sel != "All" and "City_Category" in df.columns:
    df = df[df["City_Category"] == city_sel]

if "Purchase" in df.columns:
    df = df[(df["Purchase"] >= p_range[0]) & (df["Purchase"] <= p_range[1])]

# ─────────────────────────────────────────────
# HELPER — apply global Plotly layout
# ─────────────────────────────────────────────

def style_fig(fig, height=380):
    fig.update_layout(height=height, **PLOTLY_LAYOUT)
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(size=11),
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.05)",
        zerolinecolor="rgba(255,255,255,0.08)",
        tickfont=dict(size=11),
    )
    return fig

# ─────────────────────────────────────────────
# DASHBOARD HEADER
# ─────────────────────────────────────────────

st.markdown(
    f"""
    <div class="dash-header">
        <div>
            <div class="dash-title">Black Friday Sales Intelligence</div>
            <div class="dash-subtitle">Real-time analytics · Glass OS Edition</div>
            <div class="badge">⬤  Live Dashboard  ·  {len(df):,} records matched</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────

def fmt_currency(n):
    if n >= 1e9:
        return f"₹{n/1e9:.2f}B"
    if n >= 1e6:
        return f"₹{n/1e6:.2f}M"
    if n >= 1e3:
        return f"₹{n/1e3:.1f}K"
    return f"₹{n:.0f}"

if "Purchase" in df.columns and len(df) > 0:
    total_txn     = len(df)
    total_rev     = df["Purchase"].sum()
    avg_order     = df["Purchase"].mean()
    unique_users  = df["User_ID"].nunique() if "User_ID" in df.columns else "—"
    top_cat_rev   = (
        df.groupby("Product_Category_1")["Purchase"].sum().idxmax()
        if "Product_Category_1" in df.columns else "—"
    )

    kpis = [
        ("Total Transactions", f"{total_txn:,}",          "↑ Full Dataset",   "🔢"),
        ("Total Revenue",      fmt_currency(total_rev),   "Filtered Segment", "💰"),
        ("Avg Order Value",    fmt_currency(avg_order),   "Per Transaction",  "🛒"),
        ("Unique Customers",   f"{unique_users:,}" if isinstance(unique_users, int) else unique_users,
                                "Distinct User IDs",     "👥"),
    ]

    kpi_html = '<div class="kpi-grid">'
    for i, (label, value, delta, icon) in enumerate(kpis):
        kpi_html += f"""
        <div class="kpi-card" style="animation-delay:{i*0.08}s">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-delta">{delta}</div>
        </div>"""
    kpi_html += "</div>"
    st.markdown(kpi_html, unsafe_allow_html=True)

else:
    st.warning("No data matches the current filters. Adjust the sidebar controls.")
    st.stop()

# ─────────────────────────────────────────────
# ANOMALY DETECTION — IQR method (precompute)
# ─────────────────────────────────────────────

Q1 = df["Purchase"].quantile(0.25)
Q3 = df["Purchase"].quantile(0.75)
IQR = Q3 - Q1
UPPER_FENCE = Q3 + 1.5 * IQR
LOWER_FENCE = Q1 - 1.5 * IQR

df["anomaly_flag"] = df["Purchase"].apply(
    lambda x: "🔴 Anomaly" if (x > UPPER_FENCE or x < LOWER_FENCE) else "🟢 Normal"
)
n_anomalies = (df["anomaly_flag"] == "🔴 Anomaly").sum()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Financial Overview",
    "🌐 3D Customer Mapping",
    "🚨 Anomaly Detection",
    "🧩 Product Insights",
])

# ══════════════════════════════════════════════
# TAB 1 — FINANCIAL OVERVIEW
# ══════════════════════════════════════════════

with tab1:
    col_a, col_b = st.columns(2, gap="medium")

    # ── Gender revenue donut ──
    with col_a:
        st.markdown('<div class="section-title">Revenue by Gender</div>', unsafe_allow_html=True)
        if "Gender" in df.columns:
            gen_rev = df.groupby("Gender")["Purchase"].sum().reset_index()
            fig_donut = go.Figure(go.Pie(
                labels=gen_rev["Gender"],
                values=gen_rev["Purchase"],
                hole=0.62,
                marker=dict(
                    colors=["#6e56ff", "#ff5694"],
                    line=dict(color="rgba(0,0,0,0)", width=0),
                ),
                textfont=dict(family="DM Sans", size=12),
                hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>",
            ))
            fig_donut.update_layout(
                annotations=[dict(
                    text=f"<b>{fmt_currency(total_rev)}</b><br><span style='font-size:10px'>Total</span>",
                    x=0.5, y=0.5, font_size=15, showarrow=False, font_color="#eaf0ff",
                )],
                showlegend=True,
            )
            fig_donut = style_fig(fig_donut, 360)
            st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

    # ── Age group revenue bar ──
    with col_b:
        st.markdown('<div class="section-title">Revenue by Age Group</div>', unsafe_allow_html=True)
        if "Age" in df.columns:
            age_rev = df.groupby("Age")["Purchase"].sum().reset_index().sort_values("Purchase", ascending=True)
            # Replaced go.Bar with px.bar to fix colorscale bug
            fig_age = px.bar(
                age_rev,
                x="Purchase",
                y="Age",
                orientation="h",
                color="Purchase",
                color_continuous_scale=["#3b28cc", "#6e56ff", "#ff5694"]
            )
            fig_age.update_traces(hovertemplate="<b>Age %{y}</b><br>Revenue: ₹%{x:,.0f}<extra></extra>")
            fig_age = style_fig(fig_age, 360)
            fig_age.update_layout(xaxis_title="Total Revenue (₹)", yaxis_title="", coloraxis_showscale=False)
            st.plotly_chart(fig_age, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div style="height:.5rem"></div>', unsafe_allow_html=True)

    col_c, col_d = st.columns(2, gap="medium")

    # ── City category donut ──
    with col_c:
        if "City_Category" in df.columns:
            st.markdown('<div class="section-title">Revenue by City Tier</div>', unsafe_allow_html=True)
            city_rev = df.groupby("City_Category")["Purchase"].sum().reset_index()
            fig_city = go.Figure(go.Pie(
                labels=city_rev["City_Category"],
                values=city_rev["Purchase"],
                hole=0.58,
                marker=dict(colors=["#00e5c3","#6e56ff","#ffb547"], line=dict(color="rgba(0,0,0,0)", width=0)),
                hovertemplate="<b>City %{label}</b><br>Revenue: ₹%{value:,.0f}<extra></extra>",
            ))
            fig_city = style_fig(fig_city, 340)
            st.plotly_chart(fig_city, use_container_width=True, config={"displayModeBar": False})

    # ── Marital status revenue ──
    with col_d:
        if "Marital_Status" in df.columns:
            st.markdown('<div class="section-title">Revenue by Marital Status</div>', unsafe_allow_html=True)
            ms_map = {0: "Single", 1: "Married"}
            df["Marital_Label"] = df["Marital_Status"].map(ms_map).fillna("Unknown")
            ms_rev = df.groupby("Marital_Label")["Purchase"].sum().reset_index()
            fig_ms = go.Figure(go.Bar(
                x=ms_rev["Marital_Label"],
                y=ms_rev["Purchase"],
                marker=dict(
                    color=["#6e56ff","#ff5694"],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>",
            ))
            fig_ms = style_fig(fig_ms, 340)
            fig_ms.update_layout(xaxis_title="", yaxis_title="Total Revenue (₹)", bargap=0.4)
            st.plotly_chart(fig_ms, use_container_width=True, config={"displayModeBar": False})

    # ── Occupation revenue bar (full width) ──
    if "Occupation" in df.columns:
        st.markdown('<div class="section-title">Revenue by Occupation</div>', unsafe_allow_html=True)
        occ_rev = (
            df.groupby("Occupation")["Purchase"]
            .sum()
            .reset_index()
            .sort_values("Purchase", ascending=False)
        )
        occ_rev["Occupation_Str"] = occ_rev["Occupation"].astype(str)
        # Replaced go.Bar with px.bar to fix colorscale bug
        fig_occ = px.bar(
            occ_rev,
            x="Occupation_Str",
            y="Purchase",
            color="Purchase",
            color_continuous_scale=["#1a0a2e", "#6e56ff", "#00e5c3"]
        )
        fig_occ.update_traces(hovertemplate="<b>Occupation %{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>")
        fig_occ = style_fig(fig_occ, 320)
        fig_occ.update_layout(xaxis_title="Occupation Code", yaxis_title="Total Revenue (₹)", bargap=0.25, coloraxis_showscale=False)
        st.plotly_chart(fig_occ, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════
# TAB 2 — 3D CUSTOMER MAPPING
# ══════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-title">3D Customer Segmentation Map</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Each point is a customer — plotted by Age, Occupation, and Purchase value. '
        'Rotate and zoom freely.</div>',
        unsafe_allow_html=True,
    )

    required_3d = {"Age", "Occupation", "Purchase"}
    if required_3d.issubset(df.columns):
        # sample for performance (max 8000 points)
        sample_df = df.sample(min(8000, len(df)), random_state=42)

        color_col = "Gender" if "Gender" in sample_df.columns else "Purchase"
        symbol_col = "City_Category" if "City_Category" in sample_df.columns else None

        fig_3d = px.scatter_3d(
            sample_df,
            x="Occupation",
            y="Age" if sample_df["Age"].dtype != object else sample_df["Age"].astype("category").cat.codes,
            z="Purchase",
            color=color_col,
            symbol=symbol_col,
            color_discrete_sequence=ACCENT,
            opacity=0.78,
            labels={"x": "Occupation", "y": "Age", "z": "Purchase (₹)"},
            hover_data={
                col: True
                for col in ["Gender", "Age", "City_Category", "Purchase"]
                if col in sample_df.columns
            },
        )
        fig_3d.update_traces(
            marker=dict(size=3.5, line=dict(width=0)),
        )
        fig_3d.update_layout(
            height=600,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="DM Sans", color="#c8d2f0"),
            margin=dict(t=20, b=20, l=0, r=0),
            scene=dict(
                xaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.06)",
                    showbackground=True,
                    zerolinecolor="rgba(255,255,255,0.10)",
                    title="Occupation",
                ),
                yaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.06)",
                    showbackground=True,
                    zerolinecolor="rgba(255,255,255,0.10)",
                    title="Age",
                ),
                zaxis=dict(
                    backgroundcolor="rgba(0,0,0,0)",
                    gridcolor="rgba(255,255,255,0.06)",
                    showbackground=True,
                    zerolinecolor="rgba(255,255,255,0.10)",
                    title="Purchase (₹)",
                ),
                bgcolor="rgba(0,0,0,0)",
            ),
            legend=PLOTLY_LAYOUT["legend"],
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        missing = required_3d - set(df.columns)
        st.info(f"3D chart requires columns: {missing}. Not found in dataset.")

    # ── 2D Purchase distribution by Age ──
    if "Age" in df.columns and "Purchase" in df.columns:
        st.markdown('<div class="section-title" style="margin-top:1rem">Purchase Distribution by Age Band</div>', unsafe_allow_html=True)
        fig_box = px.box(
            df,
            x="Age",
            y="Purchase",
            color="Age" if "Age" in df.columns else None,
            color_discrete_sequence=ACCENT,
            points=False,
            notched=True,
        )
        fig_box = style_fig(fig_box, 360)
        fig_box.update_layout(showlegend=False, xaxis_title="Age Band", yaxis_title="Purchase (₹)")
        fig_box.update_traces(
            marker=dict(opacity=0.6),
            line=dict(width=1.5),
        )
        st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

# ══════════════════════════════════════════════
# TAB 3 — ANOMALY DETECTION
# ══════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-title">Anomaly Detection — IQR Method</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-sub">Upper fence: <b>₹{UPPER_FENCE:,.0f}</b>  |  '
        f'Lower fence: <b>₹{LOWER_FENCE:,.0f}</b>  |  '
        f'Detected <span class="anomaly-pill">{n_anomalies:,} anomalies</span>   '
        f'<span class="normal-pill">{len(df)-n_anomalies:,} normal</span></div>',
        unsafe_allow_html=True,
    )

    # ── Histogram with anomaly overlay ──
    st.markdown('<div class="section-title" style="margin-top:.8rem">Purchase Distribution & Anomaly Overlay</div>', unsafe_allow_html=True)

    normal_df  = df[df["anomaly_flag"] == "🟢 Normal"]
    anomaly_df = df[df["anomaly_flag"] == "🔴 Anomaly"]

    fig_hist = go.Figure()

    # Normal distribution
    fig_hist.add_trace(go.Histogram(
        x=normal_df["Purchase"],
        name="Normal",
        nbinsx=60,
        marker_color="rgba(110,86,255,0.65)",
        marker_line=dict(width=0),
        opacity=0.85,
        hovertemplate="Range: %{x}<br>Count: %{y}<extra>Normal</extra>",
    ))

    # Anomaly overlay
    fig_hist.add_trace(go.Histogram(
        x=anomaly_df["Purchase"],
        name="Anomaly",
        nbinsx=60,
        marker_color="rgba(255,86,148,0.80)",
        marker_line=dict(width=0),
        opacity=0.90,
        hovertemplate="Range: %{x}<br>Count: %{y}<extra>Anomaly</extra>",
    ))

    # Fence lines
    for fence, label, clr in [
        (UPPER_FENCE, "Upper IQR Fence", "#00e5c3"),
        (LOWER_FENCE, "Lower IQR Fence", "#ffb547"),
        (Q1,          "Q1",              "rgba(255,255,255,0.3)"),
        (Q3,          "Q3",              "rgba(255,255,255,0.3)"),
    ]:
        fig_hist.add_vline(
            x=fence,
            line_dash="dash",
            line_color=clr,
            line_width=1.8,
            annotation_text=label,
            annotation_font=dict(color=clr, size=10),
            annotation_position="top right",
        )

    fig_hist.update_layout(barmode="overlay")
    fig_hist = style_fig(fig_hist, 400)
    fig_hist.update_layout(
        xaxis_title="Purchase Amount (₹)",
        yaxis_title="Transaction Count",
        bargap=0.01,
    )
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    # ── KDE-style density curve (using violin) ──
    col_v1, col_v2 = st.columns(2, gap="medium")
    with col_v1:
        st.markdown('<div class="section-title">Density — Normal vs Anomaly</div>', unsafe_allow_html=True)
        fig_vio = go.Figure()
        for grp, clr in [("🟢 Normal","#6e56ff"),("🔴 Anomaly","#ff5694")]:
            sub = df[df["anomaly_flag"]==grp]["Purchase"]
            fig_vio.add_trace(go.Violin(
                y=sub, name=grp,
                box_visible=True,
                meanline_visible=True,
                fillcolor=clr.replace("#","rgba(").rstrip(")") + ",0.3)" if "#" in clr else clr,
                line_color=clr,
                opacity=0.85,
            ))
        fig_vio = style_fig(fig_vio, 380)
        fig_vio.update_layout(violinmode="overlay", yaxis_title="Purchase (₹)", xaxis_title="")
        st.plotly_chart(fig_vio, use_container_width=True, config={"displayModeBar": False})

    with col_v2:
        st.markdown('<div class="section-title">Anomaly Share by Age Group</div>', unsafe_allow_html=True)
        if "Age" in df.columns:
            anom_age = (
                df.groupby(["Age","anomaly_flag"])
                .size()
                .reset_index(name="Count")
            )
            fig_aa = px.bar(
                anom_age,
                x="Age",
                y="Count",
                color="anomaly_flag",
                color_discrete_map={"🟢 Normal":"#6e56ff","🔴 Anomaly":"#ff5694"},
                barmode="stack",
                hovertemplate="<b>Age %{x}</b><br>Count: %{y}<extra>%{fullData.name}</extra>",
            )
            fig_aa = style_fig(fig_aa, 380)
            fig_aa.update_layout(xaxis_title="Age Band", yaxis_title="Transactions", bargap=0.25)
            st.plotly_chart(fig_aa, use_container_width=True, config={"displayModeBar": False})

    # ── Anomaly table ──
    st.markdown('<div class="section-title" style="margin-top:.8rem">🔴 Top High-Value Anomaly Records</div>', unsafe_allow_html=True)
    cols_show = [c for c in ["User_ID","Gender","Age","Occupation","City_Category","Purchase","anomaly_flag"] if c in df.columns]
    anom_table = df[df["anomaly_flag"]=="🔴 Anomaly"][cols_show].sort_values("Purchase", ascending=False).head(20)
    st.dataframe(
        anom_table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Purchase": st.column_config.NumberColumn("Purchase (₹)", format="₹%d"),
        },
    )

# ══════════════════════════════════════════════
# TAB 4 — PRODUCT INSIGHTS
# ══════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-title">Product Category Revenue Analysis</div>', unsafe_allow_html=True)

    cat_cols = [c for c in ["Product_Category_1","Product_Category_2","Product_Category_3"] if c in df.columns]

    if cat_cols:
        col_p1, col_p2 = st.columns(2, gap="medium")

        with col_p1:
            # Top-15 categories by revenue
            st.markdown('<div class="section-title">Top Product Categories (Revenue)</div>', unsafe_allow_html=True)
            cat1_rev = (
                df.groupby("Product_Category_1")["Purchase"]
                .sum()
                .reset_index()
                .sort_values("Purchase", ascending=False)
                .head(15)
            )
            cat1_rev["Category_Str"] = cat1_rev["Product_Category_1"].astype(str)
            # Replaced go.Bar with px.bar to fix colorscale bug
            fig_cat = px.bar(
                cat1_rev,
                x="Purchase",
                y="Category_Str",
                orientation="h",
                color="Purchase",
                color_continuous_scale=["#1a0a2e", "#6e56ff", "#00e5c3"]
            )
            fig_cat.update_traces(hovertemplate="<b>Category %{y}</b><br>Revenue: ₹%{x:,.0f}<extra></extra>")
            fig_cat = style_fig(fig_cat, 440)
            fig_cat.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                xaxis_title="Total Revenue (₹)",
                yaxis_title="",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_cat, use_container_width=True, config={"displayModeBar": False})

        with col_p2:
            # Category transaction volume donut
            st.markdown('<div class="section-title">Transaction Volume by Category</div>', unsafe_allow_html=True)
            cat1_cnt = (
                df.groupby("Product_Category_1")
                .size()
                .reset_index(name="Count")
                .sort_values("Count", ascending=False)
                .head(10)
            )
            fig_vol = go.Figure(go.Pie(
                labels=cat1_cnt["Product_Category_1"].astype(str),
                values=cat1_cnt["Count"],
                hole=0.55,
                marker=dict(colors=ACCENT * 3, line=dict(color="rgba(0,0,0,0)", width=0)),
                hovertemplate="<b>Category %{label}</b><br>Transactions: %{value:,}<extra></extra>",
            ))
            fig_vol = style_fig(fig_vol, 440)
            st.plotly_chart(fig_vol, use_container_width=True, config={"displayModeBar": False})

        # ── Gender × Category heatmap ──
        if "Gender" in df.columns:
            st.markdown('<div class="section-title">Gender × Product Category Heatmap</div>', unsafe_allow_html=True)
            heat_df = (
                df.groupby(["Gender","Product_Category_1"])["Purchase"]
                .sum()
                .reset_index()
            )
            pivot = heat_df.pivot(index="Gender", columns="Product_Category_1", values="Purchase").fillna(0)
            # Replaced go.Heatmap with px.imshow to fix colorscale bug
            fig_heat = px.imshow(
                pivot.values,
                x=[str(c) for c in pivot.columns],
                y=pivot.index.tolist(),
                color_continuous_scale=["#05070f", "#3b28cc", "#6e56ff", "#ff5694"],
                aspect="auto"
            )
            fig_heat.update_traces(hovertemplate="Category: %{x}<br>Gender: %{y}<br>Revenue: ₹%{z:,.0f}<extra></extra>")
            fig_heat = style_fig(fig_heat, 260)
            fig_heat.update_layout(xaxis_title="Product Category", yaxis_title="")
            st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

        # ── Avg purchase per category — scatter with size ──
        if "Product_Category_1" in df.columns:
            st.markdown('<div class="section-title">Avg vs Total Revenue per Category (Bubble)</div>', unsafe_allow_html=True)
            bubble_df = df.groupby("Product_Category_1").agg(
                Avg_Purchase=("Purchase","mean"),
                Total_Revenue=("Purchase","sum"),
                Volume=("Purchase","count"),
            ).reset_index()
            fig_bub = px.scatter(
                bubble_df,
                x="Avg_Purchase",
                y="Total_Revenue",
                size="Volume",
                color="Total_Revenue",
                color_continuous_scale=[[0,"#3b28cc"],[0.5,"#6e56ff"],[1,"#ff5694"]],
                hover_name="Product_Category_1",
                labels={
                    "Avg_Purchase":"Avg Purchase (₹)",
                    "Total_Revenue":"Total Revenue (₹)",
                    "Volume":"Transactions",
                },
                size_max=60,
            )
            fig_bub = style_fig(fig_bub, 400)
            fig_bub.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_bub, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No Product_Category columns detected in the dataset.")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────

st.markdown(
    """
    <div style="
        text-align:center;
        padding: 1.2rem 0 .4rem;
        font-size:.72rem;
        color:rgba(200,210,240,0.25);
        letter-spacing:.05em;
        border-top:1px solid rgba(255,255,255,0.06);
        margin-top:2rem;
    ">
        Black Friday Intelligence  ·  Glass OS Edition  ·  
        Built with Streamlit & Plotly  ·  © 2025
    </div>
    """,
    unsafe_allow_html=True,
)
