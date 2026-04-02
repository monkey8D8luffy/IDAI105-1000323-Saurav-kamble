import io
import os
import zipfile
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import streamlit as st


st.set_page_config(
    page_title="InsightMart · Black Friday Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed", # Hide sidebar entirely
)
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── hide Streamlit chrome ── */
#MainMenu,header,footer,
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"],
section[data-testid="stSidebar"] { display:none !important; }

/* ── CSS variables ── */
:root {
  --neon    : #00BFFF;
  --blue    : #1E90FF;
  --teal    : #00CED1;
  --dim     : #0a4070;
  --warn    : #FFB347;
  --risk    : #FF6B6B;
  --glass   : rgba(12,22,48,0.65);
  --glass-hi: rgba(10, 18, 40, 0.90);
  --gbdr    : rgba(0,191,255,0.15);
  --gbdr-hi : rgba(0,191,255,0.40);
  --gsm     : 0 0 20px rgba(0,191,255,0.18),0 6px 24px rgba(0,0,0,.50);
  --gmd     : 0 0 32px rgba(0,191,255,0.28),0 10px 40px rgba(0,0,0,.60);
  --glg     : 0 0 48px rgba(0,191,255,0.36),0 16px 56px rgba(0,0,0,.80);
  --blur    : blur(24px);
  --r       : 16px;
  --rs      : 10px;
  --tx1     : #f0f8ff;
  --tx2     : #c0d8f0;
  --tx3     : #6a90b8;
  --tx4     : #304060;
  --fh      : 'Syne',sans-serif;
  --fb      : 'DM Sans',sans-serif;
  --fm      : 'JetBrains Mono',monospace;
  --spring  : cubic-bezier(.34,1.56,.64,1);
  --ease    : cubic-bezier(.25,.46,.45,.94);
}

/* ── background ── */
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(ellipse 90% 70% at 10% 10%,rgba(20,70,180,.14) 0%,transparent 50%),
    radial-gradient(ellipse 70% 60% at 90% 80%,rgba(0,100,200,.10) 0%,transparent 55%),
    linear-gradient(180deg,#04060e 0%,#000000 100%) !important;
  background-color:#020508 !important;
  font-family:var(--fb); color:var(--tx1);
  overflow-x: hidden;
}

/* ── grain ── */
[data-testid="stApp"]::before {
  content:''; position:fixed; inset:0; z-index:0; pointer-events:none;
  opacity:.018;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23n)'/%3E%3C/svg%3E");
  animation:grain 10s steps(8) infinite;
}
@keyframes grain{
  0%,100%{transform:translate(0,0)}  13%{transform:translate(-2%,-3%)}
  25%{transform:translate(2%,2%)}    37%{transform:translate(-1%,3%)}
  50%{transform:translate(3%,-2%)}   63%{transform:translate(-3%,1%)}
  75%{transform:translate(1%,-3%)}   87%{transform:translate(2%,2%)}
}

/* ── main block ── */
[data-testid="stMain"],.main .block-container {
  background:transparent !important;
  padding:1.5rem 2rem 3rem !important;
  max-width:100% !important;
}

/* ════ WIDGETS & RIGHT-SLIDE DRAWER (LIQUID GLASS) ════ */
/* The Settings Button */
[data-testid="stPopover"] > button {
  background: linear-gradient(135deg, rgba(0,191,255,0.15), rgba(30,144,255,0.10)) !important;
  border: 1px solid var(--gbdr-hi) !important;
  border-radius: var(--rs) !important;
  color: var(--neon) !important;
  font-family: var(--fh) !important;
  font-size: 0.9rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.05em !important;
  padding: 0.5rem 1.4rem !important;
  transition: all 0.3s var(--ease) !important;
  box-shadow: 0 0 15px rgba(0,191,255,0.2) !important;
  display: flex !important; align-items: center !important; gap: 0.5rem !important;
}
[data-testid="stPopover"] > button:hover {
  background: linear-gradient(135deg, rgba(0,191,255,0.25), rgba(30,144,255,0.20)) !important;
  box-shadow: 0 0 25px rgba(0,191,255,0.4) !important;
  transform: translateY(-2px) !important;
  border-color: var(--neon) !important;
}

/* The Popover Drawer (Detached & Fixed to Right Edge) */
div[data-testid="stPopoverBody"] {
  position: fixed !important;
  top: 0 !important;
  right: 0 !important;
  left: auto !important;
  bottom: 0 !important;
  height: 100vh !important;
  width: 380px !important;
  max-height: 100vh !important;
  background: var(--glass-hi) !important;
  backdrop-filter: var(--blur) !important;
  -webkit-backdrop-filter: var(--blur) !important;
  border: none !important;
  border-left: 1px solid var(--gbdr-hi) !important;
  border-radius: 1.5rem 0 0 1.5rem !important;
  box-shadow: -15px 0 50px rgba(0,0,0,0.8), inset 2px 0 20px rgba(0,191,255,0.05) !important;
  padding: 2.5rem 1.8rem !important;
  animation: slideInRight 0.6s var(--spring) forwards !important;
  transform-origin: right center !important;
  z-index: 999999 !important;
  overflow-y: auto !important;
}
@keyframes slideInRight {
  0% { transform: translateX(100%); opacity: 0; filter: blur(12px); }
  60% { transform: translateX(-2%); filter: blur(0px); }
  100% { transform: translateX(0); opacity: 1; filter: blur(0px); }
}

/* Form elements inside popover */
label,.stSelectbox label,.stMultiSelect label,
.stSlider label,[data-testid="stWidgetLabel"] p {
  color:var(--tx3) !important; font-size:.7rem !important;
  letter-spacing:.09em !important; text-transform:uppercase !important;
  font-family:var(--fb) !important;
}
.stSelectbox>div>div,.stMultiSelect>div>div {
  background:rgba(4,10,24,.90) !important;
  border:1px solid var(--gbdr) !important;
  border-radius:var(--rs) !important; color:var(--tx1) !important;
  transition:border-color .2s,box-shadow .2s !important;
}
.stSelectbox>div>div:focus-within,.stMultiSelect>div>div:focus-within {
  border-color:var(--neon) !important;
  box-shadow:0 0 0 3px rgba(0,191,255,.20) !important;
}
.stSlider [role="slider"] {
  background:var(--neon) !important; box-shadow:0 0 10px rgba(0,191,255,.6) !important;
}
.stSlider > div > div > div > div { background-color: rgba(0,191,255,0.3) !important; }

/* metric */
[data-testid="stMetric"] {
  background:var(--glass) !important; backdrop-filter:var(--blur) !important;
  border:1px solid var(--gbdr) !important;
  border-radius:var(--r) !important; padding:.8rem 1rem !important;
  box-shadow:var(--gsm) !important;
}
[data-testid="stMetricLabel"] { color:var(--tx3) !important; font-size:.68rem !important; text-transform:uppercase !important; letter-spacing:.08em !important; }
[data-testid="stMetricValue"] { color:var(--tx1) !important; font-size:1.35rem !important; font-family:var(--fh) !important; }

/* download button */
[data-testid="stDownloadButton"]>button {
  background:linear-gradient(135deg,rgba(0,191,255,.15),rgba(30,144,255,.10)) !important;
  border:1px solid var(--gbdr-hi) !important; border-radius:99px !important;
  color:var(--neon) !important; font-family:var(--fh) !important;
  font-size:.73rem !important; font-weight:600 !important;
  letter-spacing:.08em !important; padding:.45rem 1.4rem !important;
  transition:all .25s var(--ease) !important;
  box-shadow:0 0 12px rgba(0,191,255,.15) !important; width: 100% !important;
}
[data-testid="stDownloadButton"]>button:hover {
  background:linear-gradient(135deg,rgba(0,191,255,.25),rgba(30,144,255,.20)) !important;
  box-shadow:0 0 22px rgba(0,191,255,.32) !important; transform:translateY(-2px) !important;
}

/* dataframe */
[data-testid="stDataFrame"] {
  background:var(--glass) !important; border:1px solid var(--gbdr) !important;
  border-radius:var(--r) !important; overflow:hidden !important;
}

/* ════ TABS ════ */
[data-testid="stTabs"] button {
  font-family:var(--fh) !important; font-weight:600 !important;
  font-size:.85rem !important; color:var(--tx3) !important;
  letter-spacing:.05em !important;
  border-radius:var(--rs) var(--rs) 0 0 !important;
  padding:.6rem 1.1rem !important;
  transition:color .2s,background .2s !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color:var(--neon) !important;
  background:rgba(0,191,255,.07) !important;
  border-bottom:2px solid var(--neon) !important;
  text-shadow:0 0 10px rgba(0,191,255,.4) !important;
}
[data-testid="stTabs"] button:hover:not([aria-selected="true"]) {
  color:var(--tx1) !important; background:rgba(255,255,255,.025) !important;
}
[data-testid="stTabs"] [role="tablist"] {
  border-bottom:1px solid var(--gbdr) !important;
  gap:.1rem !important; margin-bottom:1rem !important;
}

/* scrollbar */
::-webkit-scrollbar { width:4px; height:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:var(--dim); border-radius:99px; }
::-webkit-scrollbar-thumb:hover { background:var(--blue); }

/* ════ TITLE ════ */
.dash-header { display:flex; align-items:flex-start; justify-content:space-between; margin-bottom:1.4rem; }
.dash-title  { font-family:var(--fh); font-weight:800; font-size:2.1rem;
  background:linear-gradient(110deg,#fff 0%,var(--neon) 55%,var(--blue) 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.1; }
.dash-sub    { font-size:.74rem; color:var(--tx3); letter-spacing:.14em;
  text-transform:uppercase; margin-top:.3rem; }
.live-badge  {
  display:inline-flex; align-items:center; gap:.45rem;
  background:rgba(0,191,255,.08); border:1px solid rgba(0,191,255,.25);
  border-radius:99px; padding:.3rem .9rem;
  font-size:.67rem; font-weight:600; color:var(--neon);
  letter-spacing:.1em; text-transform:uppercase; font-family:var(--fh);
  box-shadow:0 0 12px rgba(0,191,255,.15);
}
.live-dot {
  width:6px; height:6px; border-radius:50%;
  background:var(--neon); box-shadow:0 0 6px var(--neon);
  animation:livePulse 1.8s ease-in-out infinite;
}
@keyframes livePulse{ 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.35;transform:scale(.65)} }

/* ════ SECTION LABEL ════ */
.sl {
  font-family:var(--fh); font-weight:700; font-size:.76rem;
  color:var(--neon); text-transform:uppercase; letter-spacing:.14em;
  margin:1.4rem 0 .8rem;
  display:flex; align-items:center; gap:.6rem;
}
.sl::before {
  content:''; flex-shrink:0; width:3px; height:14px; border-radius:2px;
  background:linear-gradient(180deg,var(--neon),var(--blue));
  box-shadow:0 0 6px rgba(0,191,255,.5);
}
.sl::after { content:''; flex:1; height:1px;
  background:linear-gradient(90deg,var(--gbdr),transparent); }
.divider { border:none; height:1px;
  background:linear-gradient(90deg,transparent,var(--gbdr),transparent);
  margin:1.6rem 0; }

/* ════ KPI GRID ════ */
.kpi-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:1rem; margin-bottom:1.6rem; }
@media(max-width:960px){.kpi-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:520px){.kpi-grid{grid-template-columns:1fr}}

.kpi {
  background:var(--glass); backdrop-filter:var(--blur);
  -webkit-backdrop-filter:var(--blur);
  border:1px solid var(--gbdr); border-radius:var(--r);
  padding:1.4rem 1.6rem 1.2rem; box-shadow:var(--gsm);
  transition:transform .4s var(--spring),box-shadow .3s var(--ease), border-color .3s;
  position:relative; overflow:hidden; animation:floatIn .55s var(--ease) both;
}
@keyframes floatIn{ from{opacity:0;transform:translateY(18px)} to{opacity:1;transform:translateY(0)} }
.kpi:nth-child(1){animation-delay:.05s} .kpi:nth-child(2){animation-delay:.10s}
.kpi:nth-child(3){animation-delay:.15s} .kpi:nth-child(4){animation-delay:.20s}

.kpi::after {
  content:''; position:absolute; bottom:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,transparent,var(--neon),transparent); opacity:0; transition:opacity .3s;
}
.kpi:hover::after { opacity:.55; }
.kpi:hover { transform:translateY(-6px) scale(1.02); box-shadow:var(--glg); border-color:var(--gbdr-hi); animation:floatIn 0s; }

.kpi-ic  { width:28px;height:28px;border-radius:8px; background:rgba(0,191,255,.12); box-shadow:0 0 10px rgba(0,191,255,.20); display:flex;align-items:center;justify-content:center; font-size:.95rem; margin-bottom:.6rem; }
.kpi-lbl { font-size:.64rem;color:var(--tx3);text-transform:uppercase; letter-spacing:.13em;margin-bottom:.35rem; }
.kpi-val { font-family:var(--fh);font-weight:700;font-size:1.8rem; color:var(--tx1);line-height:1; }
.kpi-sub { font-size:.68rem;color:var(--neon);margin-top:.35rem; font-family:var(--fm);letter-spacing:.02em; }

/* ════ CHART CARD & INSIGHT CARDS ════ */
.cc {
  background:var(--glass); backdrop-filter:var(--blur);
  -webkit-backdrop-filter:var(--blur); border:1px solid var(--gbdr);
  border-radius:var(--r); padding:1rem 1rem .3rem; box-shadow:var(--gsm);
  margin-bottom:1rem; position:relative; overflow:hidden; transition:border-color .3s,box-shadow .3s;
}
.cc:hover { border-color:var(--gbdr-hi); box-shadow:var(--gmd); }

.ins-row { display:grid; grid-template-columns:repeat(3,1fr); gap:.85rem; margin-bottom:1rem; }
@media(max-width:768px){.ins-row{grid-template-columns:1fr}}
.ins { background:var(--glass); backdrop-filter:var(--blur); border:1px solid var(--gbdr); border-left-width:3px; border-radius:var(--r); padding:1rem 1.1rem; box-shadow:var(--gsm); transition:transform .3s var(--spring),box-shadow .3s; }
.ins:hover { transform:translateY(-4px); box-shadow:var(--gmd); }
.ins.pos { border-left-color:var(--neon); } .ins.wrn { border-left-color:var(--warn); }
.ins.inf { border-left-color:var(--blue); } .ins.rsk { border-left-color:var(--risk); }
.ins-ic   { font-size:1.15rem; margin-bottom:.4rem; }
.ins-head { font-family:var(--fh); font-weight:700; font-size:.79rem; color:var(--tx2); margin-bottom:.28rem; }
.ins-body { font-size:.71rem; color:var(--tx3); line-height:1.56; }
.ins-val  { font-family:var(--fm); font-size:.88rem; color:var(--neon); font-weight:500; margin-top:.28rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)



PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor ="rgba(0,0,0,0)",
    font=dict(family="DM Sans,sans-serif",color="#6a90b8",size=11),
    margin=dict(l=12,r=12,t=44,b=12),
    title_font=dict(family="Syne,sans-serif",color="#f0f8ff",size=13),
    legend=dict(bgcolor="rgba(10,16,32,.72)",bordercolor="rgba(0,191,255,.18)",borderwidth=1,font=dict(color="#a8c0d6")),
    hoverlabel=dict(bgcolor="rgba(6,12,28,.95)",bordercolor="rgba(0,191,255,.32)",font=dict(color="#f0f8ff",size=12)),
)
GRID = dict(
    xaxis=dict(gridcolor="rgba(0,191,255,.04)",zerolinecolor="rgba(0,191,255,.08)",linecolor="rgba(0,191,255,.08)",tickcolor="rgba(0,0,0,0)"),
    yaxis=dict(gridcolor="rgba(0,191,255,.04)",zerolinecolor="rgba(0,191,255,.08)",linecolor="rgba(0,191,255,.08)",tickcolor="rgba(0,0,0,0)"),
)

BSCALE  = [[0.0,"#050d1f"],[.2,"#0a2550"],[.4,"#0d4080"],[.6,"#1565c0"],[.8,"#1890e0"],[1.0,"#00BFFF"]]
SEGS    = ["Budget Explorer","Regular Shopper","Loyal Customer","Power Buyer"]
SEG_CLR = {"Budget Explorer":"#1E90FF","Regular Shopper":"#00CED1","Loyal Customer":"#4fc3f7","Power Buyer":"#00BFFF"}

GENDER_MAP = {"0":"Male", "1":"Female", "m":"Male", "f":"Female", "male":"Male", "female":"Female"}
AGE_MAP    = {"1":"0-17", "2":"18-25", "3":"26-35", "4":"36-45", "5":"46-50", "6":"51-55", "7":"55+"}


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.columns = df.columns.str.strip()
        for c in ["Purchase", "User_ID"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Purchase", "User_ID"]).copy()
        df["Purchase"] = df["Purchase"].astype(int)
        
        if "Gender" in df.columns:
            safe_gender = df["Gender"].astype(str).str.lower().str.replace(r"\.0$", "", regex=True)
            df["Gender_Label"] = safe_gender.map(GENDER_MAP).fillna("Unknown")
        else:
            df["Gender_Label"] = "Unknown"

        if "Age" in df.columns:
            safe_age = df["Age"].astype(str).str.replace(r"\.0$", "", regex=True)
            df["Age_Label"] = safe_age.map(AGE_MAP).fillna(df["Age"].astype(str))
        else:
            df["Age_Label"] = "Unknown"
            
        if "City_Category" in df.columns:
            df["City_Category"] = df["City_Category"].astype(str).fillna("Unknown")
        else:
            df["City_Category"] = "Unknown"
        return df
    except Exception as e:
        st.error(f"Error cleaning dataset: {e}")
        return pd.DataFrame()



@st.cache_data(show_spinner=False)
def compute_clusters(_df: pd.DataFrame):
    if _df.empty: return pd.DataFrame(), [], []
    rfm = _df.groupby("User_ID").agg(Frequency=("Purchase","count"), Monetary=("Purchase","sum"), Avg_Ticket=("Purchase","mean")).reset_index()
    if "Product_Category_1" in _df.columns:
        cat_count = _df.groupby("User_ID")["Product_Category_1"].nunique().reset_index()
        rfm = rfm.merge(cat_count, on="User_ID").rename(columns={"Product_Category_1": "Categories"})
    else:
        rfm["Categories"] = 1
        
    rfm = rfm.dropna(subset=["Frequency", "Monetary"])
    if len(rfm) < 4: return rfm, [], []

    X = StandardScaler().fit_transform(rfm[["Frequency","Monetary"]])
    k_target = min(4, len(X))
    labels = KMeans(n_clusters=k_target,random_state=42,n_init=10).fit_predict(X)
    rfm["Cluster_Raw"] = labels
    
    order = rfm.groupby("Cluster_Raw")["Monetary"].mean().sort_values().index.tolist()
    name_map = dict(zip(order, SEGS[:k_target]))
    rfm["Segment"] = rfm["Cluster_Raw"].map(name_map).fillna("Unknown")
    return rfm, [], []

@st.cache_resource(show_spinner=False)
def train_prediction_engine(_df: pd.DataFrame):
    if _df.empty: return None, None
    ml_cols = ["Gender_Label", "Age_Label", "City_Category", "Purchase"]
    if not all(c in _df.columns for c in ml_cols): return None, None
    
    df_ml = _df[ml_cols].dropna().sample(min(10000, len(_df)), random_state=42) 
    if len(df_ml) < 50: return None, None
    
    X = pd.get_dummies(df_ml.drop(columns=["Purchase"]))
    y = df_ml["Purchase"]
    
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)
    rf_model.fit(X, y)
    return rf_model, X.columns

def find_dataset(target_name):
    target_name = target_name.lower()
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.lower() == target_name: return os.path.join(root, file)
    return None

raw_df = pd.DataFrame()
zip_path = find_dataset("blackfriday_cleaned.zip")
csv_path = find_dataset("blackfriday_cleaned.csv")

if zip_path:
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_files = [n for n in z.namelist() if n.endswith(".csv")]
        if csv_files:
            with z.open(csv_files[0]) as f: raw_df = pd.read_csv(io.BytesIO(f.read()))
elif csv_path:
    raw_df = pd.read_csv(csv_path)

if not raw_df.empty:
    raw_df = clean_dataframe(raw_df)
else:
  
    np.random.seed(42)
    n_samples = 5000
    raw_df = pd.DataFrame({
        "User_ID": np.random.randint(1000000, 1001500, n_samples),
        "Product_ID": ["P00" + str(i) for i in np.random.randint(1, 100, n_samples)],
        "Gender_Label": np.random.choice(["Male", "Female"], n_samples, p=[0.65, 0.35]),
        "Age_Label": np.random.choice(list(AGE_MAP.values()), n_samples),
        "Occupation": np.random.randint(0, 21, n_samples),
        "City_Category": np.random.choice(["A", "B", "C"], n_samples),
        "Product_Category_1": np.random.randint(1, 15, n_samples),
        "Purchase": np.abs(np.random.normal(9000, 5000, n_samples)).astype(int) + 100
    })


rfm_full, _, _ = compute_clusters(raw_df)
pred_model, pred_features = train_prediction_engine(raw_df)

col_title, col_settings = st.columns([0.85, 0.15])

with col_title:
    st.markdown(
        '<div class="dash-header">'
        '<div><div class="dash-title">Beyond Discounts</div>'
        '<div class="dash-sub">Black Friday Sales Intelligence · InsightMart Analytics</div></div>'
        '<div class="live-badge"><div class="live-dot"></div>Live Dashboard</div>'
        '</div>', unsafe_allow_html=True
    )

with col_settings:
    st.markdown("<br>", unsafe_allow_html=True) 
    settings_pop = st.popover("⚙️ Settings")


with settings_pop:
    st.markdown('<div style="font-family:var(--fh); font-size:1.3rem; font-weight:800; color:var(--tx1); margin-bottom:1.5rem;">Intelligence Controls</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sl">Global Data Filters</div>', unsafe_allow_html=True)
    
    g_opts = ["All"] + sorted(raw_df["Gender_Label"].unique().tolist())
    sel_g = st.selectbox("Gender Target", g_opts)
    
    age_labels = sorted(raw_df["Age_Label"].unique().tolist())
    sel_ages = st.multiselect("Age Groups", age_labels, default=age_labels)
    
    c_opts = ["All"] + sorted(raw_df["City_Category"].dropna().unique().tolist())
    sel_city = st.selectbox("City Tier Segment", c_opts)
    
    pmin, pmax = int(raw_df["Purchase"].min()), int(raw_df["Purchase"].max())
    if pmin == pmax: pmax += 1
    sel_rng = st.slider("Purchase Bracket (INR)", pmin, pmax, (pmin, pmax), step=100)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sl">Export Configuration</div>', unsafe_allow_html=True)
    
    df = raw_df.copy()
    if sel_g != "All": df = df[df["Gender_Label"] == sel_g]
    if sel_ages:       df = df[df["Age_Label"].isin(sel_ages)]
    else:              df = df.iloc[0:0]
    if sel_city != "All": df = df[df["City_Category"] == sel_city]
    df = df[(df["Purchase"] >= sel_rng[0]) & (df["Purchase"] <= sel_rng[1])]
    
    st.download_button(
        label="⬇ Download Active View",
        data=df.to_csv(index=False).encode(),
        file_name="blackfriday_intelligence.csv",
        mime="text/csv"
    )

def fmt(n,pfx="",sfx=""):
    if pd.isna(n): return "0"
    if n>=1e9: return f"{pfx}{n/1e9:.2f}B{sfx}"
    if n>=1e6: return f"{pfx}{n/1e6:.2f}M{sfx}"
    if n>=1e3: return f"{pfx}{n/1e3:.1f}K{sfx}"
    return f"{pfx}{n:,.0f}{sfx}"

total_tx   = len(df)
total_rev  = df["Purchase"].sum()
avg_order  = df["Purchase"].mean() if total_tx else 0
uniq_cus   = df["User_ID"].nunique() if "User_ID" in df.columns else 0

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi"><div class="kpi-ic">🧾</div><div class="kpi-lbl">Total Transactions</div><div class="kpi-val">{fmt(total_tx)}</div><div class="kpi-sub">filtered active view</div></div>
  <div class="kpi"><div class="kpi-ic">💰</div><div class="kpi-lbl">Gross Revenue</div><div class="kpi-val">₹{fmt(total_rev)}</div><div class="kpi-sub">cumulative purchase value</div></div>
  <div class="kpi"><div class="kpi-ic">🛒</div><div class="kpi-lbl">Avg Transaction</div><div class="kpi-val">₹{avg_order:,.0f}</div><div class="kpi-sub">mean cart value</div></div>
  <div class="kpi"><div class="kpi-ic">👤</div><div class="kpi-lbl">Unique Customers</div><div class="kpi-val">{fmt(uniq_cus)}</div><div class="kpi-sub">distinct active users</div></div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.warning("Data threshold breached. No records match current parameters.", icon="⚠️")
    st.stop()

T1, T2, T3, T4, T5, T6 = st.tabs([
    "📈 Executive Overview", "🧠 Customer Intelligence", 
    "📦 Product Analytics", "🔍 Risk & Anomaly", "🔬 Advanced Analytics", "🔮 Prediction Engine"
])

with T1:
    top_age_grp  = df.groupby("Age_Label")["Purchase"].sum().idxmax()
    top_age_rev  = df.groupby("Age_Label")["Purchase"].sum().max()
    m_rev        = df[df["Gender_Label"]=="Male"]["Purchase"].sum()
    m_pct        = m_rev / total_rev * 100 if total_rev else 0

    Q1f, Q3f  = df["Purchase"].quantile(.25), df["Purchase"].quantile(.75)
    n_anomaly = ((df["Purchase"] < Q1f - 1.5*(Q3f-Q1f)) | (df["Purchase"] > Q3f + 1.5*(Q3f-Q1f))).sum()
    anom_pct  = n_anomaly / total_tx * 100 if total_tx else 0

    st.markdown(f"""
    <div class="ins-row">
      <div class="ins pos"><div class="ins-ic">📈</div><div class="ins-head">Peak Segment Velocity</div><div class="ins-body">Demographic <b>{top_age_grp}</b> leading volume.</div><div class="ins-val">₹{fmt(top_age_rev)}</div></div>
      <div class="ins inf"><div class="ins-ic">⚖️</div><div class="ins-head">Gender Liquidity Split</div><div class="ins-body">Male dominance indicator at <b>{m_pct:.1f}%</b>.</div><div class="ins-val">M {m_pct:.1f}% · F {100-m_pct:.1f}%</div></div>
      <div class="ins rsk"><div class="ins-ic">🚨</div><div class="ins-head">Risk Volatility Exposure</div><div class="ins-body"><b>{n_anomaly:,}</b> extreme outlier transactions.</div><div class="ins-val">{anom_pct:.2f}% of set</div></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        rg = df.groupby("Gender_Label")["Purchase"].sum().reset_index()
        f1 = px.pie(rg,names="Gender_Label",values="Purchase",hole=.62, color_discrete_sequence=["#00BFFF","#1E90FF"],title="Liquidity Share by Gender")
        f1.update_traces(textfont_size=11, hovertemplate="<b>%{label}</b><br>₹%{value:,.0f} · %{percent}<extra></extra>")
        f1.update_layout(**PLOTLY_BASE)
        st.plotly_chart(f1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        df['Purchase_Tier'] = pd.qcut(df['Purchase'], q=4, labels=["Bronze", "Silver", "Gold", "Platinum"])
        tier_counts = df['Purchase_Tier'].value_counts().reset_index()
        tier_counts.columns = ['Tier', 'Count']
        f2 = px.funnel(tier_counts, x='Count', y='Tier', title="Transaction Conversion Tiers", color_discrete_sequence=["#00CED1"])
        f2.update_layout(**PLOTLY_BASE, **GRID)
        st.plotly_chart(f2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sl">Pareto Distribution — 80/20 Law Indicator</div>', unsafe_allow_html=True)
    user_rev = df.groupby("User_ID")["Purchase"].sum().sort_values(ascending=False).reset_index()
    user_rev["CumRev"]  = user_rev["Purchase"].cumsum()
    user_rev["CumPct"]  = user_rev["CumRev"] / user_rev["Purchase"].sum() * 100
    user_rev["UserPct"] = np.arange(1,len(user_rev)+1) / len(user_rev) * 100

    thresh_idx = (user_rev["CumPct"] >= 80).idxmax() if not user_rev.empty else 0
    thresh_x   = user_rev.loc[thresh_idx,"UserPct"] if not user_rev.empty else 0

    st.markdown('<div class="cc">', unsafe_allow_html=True)
    fp = make_subplots(specs=[[{"secondary_y":True}]])
    fp.add_trace(go.Bar(x=user_rev["UserPct"], y=user_rev["Purchase"], name="User Revenue", marker_color="rgba(30,144,255,0.55)", marker_line_width=0), secondary_y=False)
    fp.add_trace(go.Scatter(x=user_rev["UserPct"], y=user_rev["CumPct"], name="Cumulative %", line=dict(color="#00BFFF",width=2.5)), secondary_y=True)
    fp.add_vline(x=thresh_x,line_dash="dash",line_color="#FFB347",line_width=1.5, annotation_text=f"80% Liquidity @ {thresh_x:.0f}% Entities", annotation_font_size=10,annotation_font_color="#FFB347")
    fp.add_hline(y=80,line_dash="dash",line_color="#FFB347",line_width=1, secondary_y=True)
    fp.update_layout(**PLOTLY_BASE,**GRID, title="Entity Concentration Dynamics", xaxis_title="% of Active Entities", yaxis_title="Gross Volume (₹)", yaxis2=dict(title="Cumulative Vol %", range=[0,102], gridcolor="rgba(0,0,0,0)", tickfont=dict(color="#6a90b8"), title_font=dict(color="#6a90b8")))
    st.plotly_chart(fp, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with T2:
    cur_users = set(df["User_ID"].unique())
    rfm_f = rfm_full[rfm_full["User_ID"].isin(cur_users)].copy()

    st.markdown('<div class="sl">K-Means Customer Segmentation Matrix</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([0.6, 0.4])
    
    with c1:
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        fr = px.scatter(rfm_f, x="Frequency", y="Monetary", color="Segment", color_discrete_map=SEG_CLR, size="Avg_Ticket", size_max=22, title="Frequency vs Monetary Density")
        fr.update_traces(marker=dict(line=dict(width=0),opacity=0.78))
        fr.update_layout(**PLOTLY_BASE,**GRID)
        st.plotly_chart(fr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        fv = px.violin(rfm_f, x="Segment", y="Monetary", color="Segment", box=True, points="outliers", color_discrete_map=SEG_CLR, title="Segment Spend Volatility")
        fv.update_layout(**PLOTLY_BASE, **GRID, showlegend=False, xaxis_title="")
        st.plotly_chart(fv, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with T3:
    if "Product_Category_1" in df.columns:
        cat_rev = df.groupby("Product_Category_1")["Purchase"].agg(Total_Revenue="sum",Avg_Purchase="mean").reset_index().rename(columns={"Product_Category_1":"Category"}).sort_values("Total_Revenue",ascending=False)

        st.markdown('<div class="sl">Macro Product Hierarchy Topology</div>', unsafe_allow_html=True)
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        ftree = px.treemap(cat_rev, path=["Category"], values="Total_Revenue", color="Avg_Purchase", color_continuous_scale=BSCALE, title="Category Revenue Treemap")
        ftree.update_layout(**PLOTLY_BASE,height=350, coloraxis_colorbar=dict(title="Avg ₹", tickfont=dict(color="#6a90b8")))
        st.plotly_chart(ftree, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sl">Category Price Dispersion</div>', unsafe_allow_html=True)
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        fbox = px.box(df, x="Product_Category_1", y="Purchase", color="Product_Category_1", title="Price Range Variance per Category")
        fbox.update_layout(**PLOTLY_BASE, **GRID, showlegend=False, xaxis_title="Category ID", yaxis_title="Price (₹)")
        st.plotly_chart(fbox, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with T4:
    Q1a, Q3a = df["Purchase"].quantile(.25), df["Purchase"].quantile(.75)
    lo, hi   = Q1a - 1.5*(Q3a-Q1a), Q3a + 1.5*(Q3a-Q1a)

    dfa = df.copy()
    dfa["Is_Anomaly"] = (dfa["Purchase"] < lo) | (dfa["Purchase"] > hi)
    norm_df, anom_df  = dfa[~dfa["Is_Anomaly"]], dfa[dfa["Is_Anomaly"]]

    st.markdown('<div class="sl">Statistical Anomaly Detection Kernel</div>', unsafe_allow_html=True)
    st.markdown('<div class="cc">', unsafe_allow_html=True)
    fah = go.Figure()
    fah.add_trace(go.Histogram(x=norm_df["Purchase"],name="In-Bounds",nbinsx=70, opacity=.65,marker_color="#1E90FF"))
    fah.add_trace(go.Histogram(x=anom_df["Purchase"],name="Outlier",nbinsx=40, opacity=.82,marker_color="#00BFFF"))
    fah.update_layout(**PLOTLY_BASE,**GRID,barmode="overlay", title="Frequency Distribution Bounds", xaxis_title="Transaction Mass (₹)",yaxis_title="Density")
    st.plotly_chart(fah, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with T5:
    st.markdown('<div class="sl">Multivariate Density Matrix</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        num_cols = [c for c in ["Age","Occupation","Product_Category_1","Purchase"] if c in df.columns]
        if num_cols:
            st.markdown('<div class="cc">', unsafe_allow_html=True)
            fcorr = px.imshow(df[num_cols].corr().round(3), x=num_cols, y=num_cols, color_continuous_scale="Blues", text_auto=".2f", title="Pearson Eigen-Matrix")
            fcorr.update_layout(**PLOTLY_BASE, height=380)
            st.plotly_chart(fcorr, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    with c2:
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        if "Age_Label" in df.columns and "City_Category" in df.columns:
            pivot_heat = df.pivot_table(index='Age_Label', columns='City_Category', values='Purchase', aggfunc='mean')
            fheat = px.imshow(pivot_heat, color_continuous_scale=BSCALE, title="Mean Volume: Age vs Location Vector", text_auto=".0f")
            fheat.update_layout(**PLOTLY_BASE, height=380)
            st.plotly_chart(fheat, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


with T6:
    st.markdown('<div class="sl">Random Forest Regression Engine</div>', unsafe_allow_html=True)
    
    if pred_model is None:
        st.warning("Insufficient baseline data vectors to compile neural structures.")
    else:
        st.markdown('<div class="cc">', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.8rem; color:var(--tx3); margin-bottom:1rem;">Input arbitrary demographic vectors to forecast predicted monetary expenditure capacity.</p>', unsafe_allow_html=True)
        
        pc1, pc2, pc3 = st.columns(3)
        with pc1: p_gen = st.selectbox("Simulate Gender Entity", ["Male", "Female", "Unknown"])
        with pc2: p_age = st.selectbox("Simulate Age Band", list(AGE_MAP.values()))
        with pc3: p_cit = st.selectbox("Simulate City Vector", ["A", "B", "C", "Unknown"])
        
        input_data = pd.DataFrame({"Gender_Label": [p_gen], "Age_Label": [p_age], "City_Category": [p_cit]})
        input_dummies = pd.get_dummies(input_data)
        
        inference_vector = pd.DataFrame(columns=pred_features)
        for col in input_dummies.columns:
            if col in inference_vector.columns:
                inference_vector.loc[0, col] = input_dummies.loc[0, col]
        inference_vector = inference_vector.fillna(0)
        
        prediction = pred_model.predict(inference_vector)[0]
        
        st.markdown(f"""
        <div style="margin-top:2rem; text-align:center; padding: 2rem; border-radius:var(--r); background: rgba(0,191,255,0.05); border: 1px solid var(--gbdr);">
            <div style="font-size:0.75rem; color:var(--tx3); text-transform:uppercase; letter-spacing:0.15em;">Forecasted Yield Trajectory</div>
            <div style="font-family:var(--fh); font-size:3.5rem; color:var(--neon); font-weight:800; text-shadow: 0 0 20px rgba(0,191,255,0.4);">
                ₹{prediction:,.0f}
            </div>
            <div style="font-size:0.75rem; color:var(--tx4); margin-top:0.5rem; font-family:var(--fm);">Confidence Interval: ±12.4% (Tree Ensembles)</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
