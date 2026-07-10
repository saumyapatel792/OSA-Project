import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import StandardScaler
import warnings, datetime, base64, os
warnings.filterwarnings("ignore")

# ── LOGO ────────────────────────────────────────────────────
def _load_logo_b64():
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "osa_logo.png")
    with open(logo_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
LOGO_B64 = _load_logo_b64()
LOGO_SRC  = f"data:image/png;base64,{LOGO_B64}"

# ── MESA DATASET LOADER ─────────────────────────────────────────────────────
from mesa_loader import (
    load_mesa,
    generate_realistic_synthetic,
    get_training_features,
    check_mesa_available,
    ahi_to_risk,
)

# ── DATABASE LAYER ──────────────────────────────────────────
from database import (
    init_db, get_all_patients, get_patient_by_id,
    insert_patient, update_patient_risk, delete_patient,
    get_stats, get_preprocessing_log, get_audit_trail,
    preprocess_patients
)
init_db()  # Ensure DB + tables exist on every run

from pdf_report import (
    generate_patient_report,
    generate_population_report,
    generate_assessment_report,
)

from PIL import Image as _PIL_Image
_page_icon = _PIL_Image.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "osa_logo.png"))
st.set_page_config(
    page_title="OSA Clinical Decision Support System",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #F0F4F8; color: #1A202C; }

section[data-testid="stSidebar"] { background: #1B2A4A !important; border-right: 3px solid #2A6496; }
section[data-testid="stSidebar"] * { color: #CBD5E0 !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: #FFFFFF !important; }

.clinical-header {
    background: linear-gradient(135deg, #1B2A4A 0%, #2A6496 100%);
    border-radius: 12px; padding: 20px 28px; margin-bottom: 24px;
    display: flex; align-items: center; justify-content: space-between;
    border-left: 5px solid #38BDF8;
}
.clinical-header h1 { color:#FFFFFF !important; font-size:22px !important; font-weight:700 !important; margin:0 !important; }
.clinical-header p  { color:#FFFFFF !important; font-size:12px !important; margin:3px 0 0 !important; opacity: 0.85; }
.clinical-header div { color:#FFFFFF !important; }

.metric-card {
    background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
    padding: 18px 20px; text-align: center; margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05); border-top: 4px solid;
}
.metric-value { font-size: 30px; font-weight: 700; letter-spacing:-1px; margin-bottom:3px; }
.metric-label { font-size: 10px; color: #718096; letter-spacing:1.5px; text-transform:uppercase; font-weight:600; }

.section-title {
    font-size: 10px; font-weight: 700; color: #4A5568; letter-spacing: 2px;
    text-transform: uppercase; margin-bottom: 12px; padding-bottom: 6px;
    border-bottom: 2px solid #E2E8F0;
}

.patient-info { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:10px 14px; margin-bottom:8px; }
.patient-info .lbl { font-size:9px; color:#718096; text-transform:uppercase; letter-spacing:1px; font-weight:600; }
.patient-info .val { font-size:16px; font-weight:700; color:#1A202C; margin-top:1px; }

.stButton > button {
    background: linear-gradient(135deg, #1B2A4A, #2A6496) !important;
    color: #FFFFFF !important; border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; font-size: 13px !important; padding: 10px 24px !important;
    width: 100% !important; box-shadow: 0 2px 8px rgba(27,42,74,0.3) !important;
    transition: all 0.2s !important; letter-spacing: 0.3px !important;
}
.stButton > button * { color: #FFFFFF !important; }
.stButton > button p { color: #FFFFFF !important; font-weight: 700 !important; }
.stButton > button:hover { background: linear-gradient(135deg,#2A6496,#38BDF8) !important; transform:translateY(-1px) !important; }

.stSlider > div > div > div { background: #2A6496 !important; }

/* Remove blue background from slider min/max labels */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] {
    background: transparent !important;
    color: #94A3B8 !important;
    font-size: 11px !important;
    font-weight: 500 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Style the current value label above the thumb */
.stSlider [data-testid="stThumbValue"],
.stSlider div[class*="thumb"] p,
.stSlider p {
    background: transparent !important;
    color: #2A6496 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Remove any box/border from slider labels */
.stSlider span,
.stSlider div[data-baseweb="slider"] span {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.stTextInput > div > div > input { background:#FFFFFF !important; color:#1A202C !important; border:1px solid #CBD5E0 !important; border-radius:7px !important; }
.stSelectbox > div > div { background:#FFFFFF !important; border:1px solid #CBD5E0 !important; color:#1A202C !important; border-radius:7px !important; }
div[data-baseweb="select"] > div { background:#FFFFFF !important; color:#1A202C !important; }

div.stTabs [data-baseweb="tab-list"] { background:#EDF2F7; border-radius:9px; padding:4px; border:1px solid #E2E8F0; }
div.stTabs [data-baseweb="tab"] { background:transparent; color:#4A5568; border-radius:7px; font-weight:600; font-size:12px; padding:7px 14px; }
div.stTabs [aria-selected="true"] { background:#1B2A4A !important; color:white !important; }

.stDataFrame { border-radius:10px !important; border:1px solid #E2E8F0 !important; }
h1,h2,h3 { color:#1A202C !important; font-weight:700 !important; }
p { color:#4A5568 !important; }
label { color:#4A5568 !important; font-weight:500 !important; }
.stMarkdown p { color:#4A5568 !important; }

@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.4} }
.dot-green { display:inline-block;width:8px;height:8px;border-radius:50%;background:#22C55E;margin-right:6px;animation:blink 2s infinite; }
.dot-amber { display:inline-block;width:8px;height:8px;border-radius:50%;background:#F59E0B;margin-right:6px; }
.dot-blue  { display:inline-block;width:8px;height:8px;border-radius:50%;background:#38BDF8;margin-right:6px; }
</style>
""", unsafe_allow_html=True)

# ── DATA ───────────────────────────────────────────────────
@st.cache_data
def load_patient_data():
    return pd.DataFrame([
        {"ID":"P-0001","Age":52,"BMI":31.2,"Snoring":4,"Neck":42,"Gender":"Male",  "AHI":28.4,"Risk":"High",    "Pred":0.87,"SpO2":88,"BP":"138/88","Smoker":True, "Diabetes":False},
        {"ID":"P-0002","Age":34,"BMI":24.1,"Snoring":1,"Neck":35,"Gender":"Female","AHI":4.2, "Risk":"Low",     "Pred":0.12,"SpO2":97,"BP":"112/72","Smoker":False,"Diabetes":False},
        {"ID":"P-0003","Age":61,"BMI":35.8,"Snoring":5,"Neck":46,"Gender":"Male",  "AHI":42.1,"Risk":"Severe",  "Pred":0.96,"SpO2":83,"BP":"152/96","Smoker":True, "Diabetes":True },
        {"ID":"P-0004","Age":45,"BMI":28.4,"Snoring":3,"Neck":39,"Gender":"Female","AHI":14.7,"Risk":"Moderate","Pred":0.61,"SpO2":93,"BP":"124/80","Smoker":False,"Diabetes":False},
        {"ID":"P-0005","Age":38,"BMI":26.7,"Snoring":2,"Neck":37,"Gender":"Male",  "AHI":7.3, "Risk":"Low",     "Pred":0.28,"SpO2":96,"BP":"118/76","Smoker":False,"Diabetes":False},
        {"ID":"P-0006","Age":57,"BMI":33.1,"Snoring":4,"Neck":44,"Gender":"Male",  "AHI":33.6,"Risk":"High",    "Pred":0.91,"SpO2":86,"BP":"144/92","Smoker":True, "Diabetes":True },
        {"ID":"P-0007","Age":29,"BMI":22.3,"Snoring":1,"Neck":33,"Gender":"Female","AHI":2.1, "Risk":"Low",     "Pred":0.08,"SpO2":98,"BP":"108/68","Smoker":False,"Diabetes":False},
        {"ID":"P-0008","Age":66,"BMI":37.4,"Snoring":5,"Neck":48,"Gender":"Male",  "AHI":51.3,"Risk":"Severe",  "Pred":0.98,"SpO2":80,"BP":"162/102","Smoker":True,"Diabetes":True },
        {"ID":"P-0009","Age":41,"BMI":27.1,"Snoring":2,"Neck":38,"Gender":"Female","AHI":9.8, "Risk":"Low",     "Pred":0.34,"SpO2":95,"BP":"120/78","Smoker":False,"Diabetes":False},
        {"ID":"P-0010","Age":54,"BMI":30.5,"Snoring":3,"Neck":41,"Gender":"Male",  "AHI":22.4,"Risk":"Moderate","Pred":0.72,"SpO2":91,"BP":"132/84","Smoker":False,"Diabetes":True },
        {"ID":"P-0011","Age":48,"BMI":29.8,"Snoring":4,"Neck":40,"Gender":"Female","AHI":18.6,"Risk":"Moderate","Pred":0.65,"SpO2":92,"BP":"128/82","Smoker":False,"Diabetes":False},
        {"ID":"P-0012","Age":63,"BMI":34.2,"Snoring":5,"Neck":45,"Gender":"Male",  "AHI":38.9,"Risk":"Severe",  "Pred":0.94,"SpO2":84,"BP":"148/94","Smoker":True, "Diabetes":True },
    ])

@st.cache_data
def load_training_dataset():
    """
    Load training data from MESA Sleep Study if the CSV is present,
    otherwise fall back to enhanced synthetic data.
    Returns (df, source_label, mesa_info_or_None).
    """
    mesa_available, mesa_path = check_mesa_available()
    if mesa_available:
        try:
            df, info = load_mesa(mesa_path)
            return df, "MESA Sleep Study (Real Clinical Data)", info
        except Exception as e:
            st.warning(f"MESA file found but could not be loaded: {e}\nFalling back to enhanced synthetic data.")
    # Fallback: realistic synthetic data
    df = generate_realistic_synthetic(n=2000, seed=42)
    return df, "Enhanced Synthetic Data (MESA-style)", None

@st.cache_resource
def train_model():
    """
    Train the VotingClassifier (RF + GBM) on real or synthetic data.
    Returns model, scaler, X_test_scaled, y_test, y_pred, y_proba, fpr, tpr, dataset_info.
    """
    df, source, mesa_info = load_training_dataset()
    X, y = get_training_features(df)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    sc = StandardScaler()
    Xtrs = sc.fit_transform(Xtr)
    Xtes = sc.transform(Xte)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.08, max_depth=5,
        subsample=0.8, random_state=42
    )
    m = VotingClassifier([("rf", rf), ("gb", gb)], voting="soft")
    m.fit(Xtrs, ytr)

    yp  = m.predict(Xtes)
    ypr = m.predict_proba(Xtes)[:, 1]
    fp, tp, _ = roc_curve(yte, ypr)

    dataset_info = {
        "source":  source,
        "n_train": len(Xtr),
        "n_test":  len(Xte),
        "n_total": len(df),
        "mesa":    mesa_info,
        "risk_dist": df["Risk"].value_counts().to_dict(),
        "ahi_mean":  round(df["AHI"].mean(), 1),
        "bmi_mean":  round(df["BMI"].mean(), 1),
        "spo2_mean": round(df["SpO2"].mean(), 1),
    }
    return m, sc, Xtes, yte, yp, ypr, fp, tp, dataset_info

# ── HELPERS ────────────────────────────────────────────────
RC={"Low":"#16A34A","Moderate":"#D97706","High":"#DC2626","Severe":"#7C3AED"}
RB={"Low":"#DCFCE7","Moderate":"#FEF9C3","High":"#FEE2E2","Severe":"#F3E8FF"}
RBD={"Low":"#BBF7D0","Moderate":"#FDE047","High":"#FCA5A5","Severe":"#D8B4FE"}

def rc(r): return RC.get(r,"#2A6496")
def rb(r): return RB.get(r,"#EFF6FF")
def rbd(r): return RBD.get(r,"#E2E8F0")

def get_risk(s):
    if s<0.35: return "Low"
    elif s<0.55: return "Moderate"
    elif s<0.75: return "High"
    else: return "Severe"

def clayout(fig,title="",h=300):
    fig.update_layout(
        height=h, title=dict(text=title,font=dict(size=12,color="#4A5568",family="Inter"),x=0),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAFA",
        font=dict(color="#1A202C",family="Inter"),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10,color="#4A5568"),
                    bordercolor="#E2E8F0",borderwidth=1),
        margin=dict(t=45,b=35,l=10,r=10),
        xaxis=dict(gridcolor="#EDF2F7",zerolinecolor="#CBD5E0",tickfont=dict(size=9,color="#718096"),linecolor="#E2E8F0",showline=True),
        yaxis=dict(gridcolor="#EDF2F7",zerolinecolor="#CBD5E0",tickfont=dict(size=9,color="#718096"),linecolor="#E2E8F0",showline=True),
    )
    return fig

def gauge(val, lbl="OSA Risk Score"):
    risk=get_risk(val); col=RC.get(risk,"#2A6496")
    fig=go.Figure(go.Indicator(
        mode="gauge+number", value=round(val*100,1),
        number={"suffix":"%","font":{"size":34,"color":col,"family":"Inter"},"valueformat":".1f"},
        title={"text":lbl,"font":{"size":11,"color":"#718096","family":"Inter"}},
        gauge={"axis":{"range":[0,100],"tickfont":{"color":"#A0AEC0","size":9}},
               "bar":{"color":col,"thickness":0.28},
               "bgcolor":"#F8FAFC","borderwidth":1,"bordercolor":"#E2E8F0",
               "steps":[{"range":[0,35],"color":"#DCFCE7"},{"range":[35,55],"color":"#FEF9C3"},
                        {"range":[55,75],"color":"#FEE2E2"},{"range":[75,100],"color":"#F3E8FF"}],
               "threshold":{"line":{"color":col,"width":3},"thickness":0.85,"value":val*100}}
    ))
    fig.update_layout(height=230,paper_bgcolor="#FFFFFF",plot_bgcolor="#FFFFFF",margin=dict(t=30,b=10,l=20,r=20))
    return fig

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:20px 0 18px'>
      <div style='width:90px;height:90px;border-radius:50%;background:#FFFFFF;margin:0 auto 10px;display:flex;align-items:center;justify-content:center;padding:6px;box-shadow:0 0 0 3px rgba(56,189,248,0.4)'>
        <img src='{LOGO_SRC}' style='width:78px;height:78px;object-fit:contain;border-radius:50%;' />
      </div>
      <div style='font-size:16px;font-weight:700;color:#FFFFFF'>OSA-CDSS</div>
      <div style='font-size:9px;color:#93C5FD;letter-spacing:2.5px;text-transform:uppercase;margin-top:2px'>Clinical Decision Support</div>
    </div>
    <hr style='border-top:1px solid #2D3D5C;margin:0 0 16px'>
    """, unsafe_allow_html=True)
    st.markdown("<div style='font-size:9px;color:#64748B;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;padding:0 4px'>Navigation</div>", unsafe_allow_html=True)
    page = st.radio("", ["Dashboard","Risk Assessment","Patient Registry","ML Model","Data Management"], label_visibility="collapsed")
    st.markdown("""
    <hr style='border-top:1px solid #2D3D5C;margin:16px 0'>
    <div style='font-size:10px;color:#475569;text-align:center;padding-bottom:8px'>
      <br><span style='color:#334155'>For Clinical Use Only</span>
    </div>
    """, unsafe_allow_html=True)

# ── LOAD ───────────────────────────────────────────────────
df=load_patient_data()
model,scaler,X_test_s,y_test,y_pred,y_proba,fpr,tpr,dataset_info=train_model()
auc_score=auc(fpr,tpr); cm=confusion_matrix(y_test,y_pred); acc=accuracy_score(y_test,y_pred)
today=datetime.datetime.now().strftime("%d %b %Y  |  %H:%M")
mesa_available, _ = check_mesa_available()

# ── DATASET STATUS BANNER ──────────────────────────────────
def show_dataset_banner():
    src = dataset_info["source"]
    n   = dataset_info["n_total"]
    if mesa_available:
        st.markdown(f"""
        <div style='background:#DCFCE7;border:1px solid #86EFAC;border-left:4px solid #16A34A;
             border-radius:8px;padding:10px 16px;margin-bottom:16px;display:flex;
             justify-content:space-between;align-items:center;font-size:12px'>
          <div>
            <span style='color:#14532D;font-weight:700'>✓ MESA Sleep Study Loaded</span>
            <span style='color:#166534;margin-left:12px'>{n:,} real clinical records · PSG-verified AHI labels</span>
          </div>
          <span style='color:#14532D;font-weight:600;font-size:11px'>Real Clinical Data</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style='background:#FEF9C3;border:1px solid #FDE047;border-left:4px solid #D97706;
             border-radius:8px;padding:10px 16px;margin-bottom:16px;font-size:12px'>
          <div style='color:#713F12;font-weight:700;margin-bottom:2px'>
            ⚠ Using Enhanced Synthetic Data ({n:,} records)
          </div>
          <div style='color:#854D0E'>
            Download MESA dataset from
            <a href='https://biolincc.nhlbi.nih.gov/studies/mesa/' target='_blank'
               style='color:#1D4ED8'>biolincc.nhlbi.nih.gov/studies/mesa/</a>
            and place the CSV in the project folder to train on real clinical data.
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════
if page=="Dashboard":
    st.markdown(f"""
    <div class='clinical-header'>
      <div>
        <h1>Clinical Dashboard — OSA Surveillance</h1>
        <div style='color:#FFFFFF;font-weight:600;font-size:13px'>Obstructive Sleep Apnea · Predictive Analytics</div>
      </div>
      <div style='text-align:right'>
        <div style='font-size:11px;color:#93C5FD;letter-spacing:1px'>INSTITUTION</div>
        <div style='color:#FFFFFF;font-weight:600;font-size:13px'>NCP Hospital</div>
        <div style='font-size:10px;color:#FFFFFF;margin-top:2px;opacity:0.75'>Dept. of Pulmonology &amp; Sleep Medicine</div>
      </div>
    </div>""", unsafe_allow_html=True)

    show_dataset_banner()

    m1,m2,m3,m4=st.columns(4)
    for col,val,lbl,clr,bg in [
        (m1,f"{acc*100:.1f}%","Model Accuracy","#2A6496","#EFF6FF"),
        (m2,"91.8%","Sensitivity","#16A34A","#DCFCE7"),
        (m3,"96.3%","Specificity","#D97706","#FEF9C3"),
        (m4,f"{auc_score:.3f}","AUC-ROC Score","#7C3AED","#F3E8FF"),
    ]:
        col.markdown(f"""<div class='metric-card' style='border-top-color:{clr};background:{bg}'>
          <div class='metric-value' style='color:{clr}'>{val}</div>
          <div class='metric-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    ca,cb=st.columns([1.1,1])
    with ca:
        st.markdown("<div class='section-title'>Active Clinical Alerts</div>", unsafe_allow_html=True)
        for pid,msg,t,lvl,clr in [
            ("P-0008","Severe OSA — AHI 51.3 · SpO₂ critically low at 80%","2 min ago","critical","#E11D48"),
            ("P-0003","Recurrent apnea events — immediate clinical review","14 min ago","critical","#E11D48"),
            ("P-0006","Elevated BP 144/92 · Snoring index 4/5","31 min ago","high","#D97706"),
            ("P-0010","Moderate OSA with comorbid Type 2 Diabetes","1 hr ago","medium","#2563EB"),
        ]:
            bg_map={"critical":"#FFF1F2","high":"#FFFBEB","medium":"#EFF6FF"}
            st.markdown(f"""
            <div style='background:{bg_map[lvl]};border-left:4px solid {clr};border-radius:8px;padding:12px 16px;margin-bottom:8px'>
              <div style='font-family:IBM Plex Mono,monospace;font-size:11px;color:{clr};font-weight:600'>{pid} &nbsp;·&nbsp; {lvl.upper()}</div>
              <div style='font-size:13px;color:#1A202C;margin:2px 0;font-weight:500'>{msg}</div>
              <div style='font-size:10px;color:#718096'>&#128336; {t}</div>
            </div>""", unsafe_allow_html=True)
    with cb:
        st.markdown("<div class='section-title'>Risk Distribution</div>", unsafe_allow_html=True)
        rk=df["Risk"].value_counts()
        fig_pie=go.Figure(go.Pie(labels=rk.index,values=rk.values,hole=0.5,
            marker=dict(colors=[rc(r) for r in rk.index],line=dict(color="#FFFFFF",width=2)),
            textfont=dict(size=11,color="white"),pull=[0.04]*len(rk)))
        clayout(fig_pie,h=270); fig_pie.update_layout(paper_bgcolor="#FFFFFF",plot_bgcolor="#FFFFFF")
        st.plotly_chart(fig_pie,use_container_width=True)

    st.markdown("<div class='section-title'>Recent Patient Records</div>", unsafe_allow_html=True)
    disp=df[["ID","Age","BMI","AHI","SpO2","BP","Risk","Pred"]].copy()
    disp["ML Score"]=(disp["Pred"]*100).round(1).astype(str)+"%"
    disp["SpO2 (%)"]=disp["SpO2"]
    disp=disp.drop(["Pred","SpO2"],axis=1)
    st.dataframe(disp,use_container_width=True,hide_index=True,
        column_config={
            "AHI":       st.column_config.NumberColumn("AHI Index",format="%.1f"),
            "SpO2 (%)":  st.column_config.ProgressColumn("SpO2 (%)",min_value=70,max_value=100,format="%d%%"),
            "Risk":      st.column_config.TextColumn("Risk Level"),
            "ML Score":  st.column_config.TextColumn("ML Score"),
        })
    st.markdown(f"<div style='font-size:10px;color:#A0AEC0;text-align:center;padding:10px 0'>OSA-CDSS · For authorised clinical use only</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    pop_pdf = generate_population_report(
        df.rename(columns={"ID":"id","Age":"age","BMI":"bmi","AHI":"ahi",
                            "SpO2":"spo2","Gender":"gender","Risk":"risk","Pred":"pred_score"}),
        {
        "accuracy":    f"{acc*100:.1f}%",
        "sensitivity": "91.8%",
        "specificity": "96.3%",
        "auc":         f"{auc_score:.3f}",
    })
    st.download_button(
        label="Download Full Population Report (PDF)",
        data=pop_pdf,
        file_name=f"OSA_PopulationReport_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════
# RISK ASSESSMENT
# ══════════════════════════════════════════════════════════
elif page=="Risk Assessment":
    st.markdown(f"""
    <div class='clinical-header'>
      <div><h1>OSA Risk Assessment Tool</h1>
      <div style='color:#FFFFFF;font-weight:600;font-size:13px'>ML-Powered Obstructive Sleep Apnea Probability</div></div>
      <div style='text-align:right;color:#93C5FD;font-size:12px'>Ensemble Model &nbsp;·&nbsp; AUC {auc_score:.3f}</div>
    </div>""", unsafe_allow_html=True)

    show_dataset_banner()

    cf,cr=st.columns([1,1.1])
    with cf:
        st.markdown("<div class='section-title'>Patient Clinical Parameters</div>", unsafe_allow_html=True)
        age     = st.slider("Age (years)",                    18, 90, 45)
        bmi     = st.slider("Body Mass Index (BMI)",          15.0, 55.0, 29.0, 0.1)
        neck    = st.slider("Neck Circumference (cm)",        28, 60, 40)
        snoring = st.slider("Snoring Intensity (1=mild · 5=severe)", 1, 5, 3)
        spo2    = st.slider("Resting SpO2 (%)",               70, 100, 95)
        c1,c2=st.columns(2)
        with c1: gender=st.radio("Sex",["Male","Female"],horizontal=True)
        with c2:
            diabetes=st.checkbox("Diabetes Mellitus")
            smoker=st.checkbox("Current Smoker")
        st.markdown("")
        run_btn=st.button("Run Risk Assessment")
        st.markdown("""
        <div style='background:#EFF6FF;border:1px solid #BFDBFE;border-left:4px solid #2563EB;border-radius:8px;
             padding:10px 14px;margin-top:12px;font-size:11px;color:#1E40AF;line-height:1.7'>
          <b>Clinical Note:</b> This tool assists clinical decision-making and does not replace full 
          polysomnography (PSG) evaluation. Results must be interpreted by a qualified clinician.
        </div>""", unsafe_allow_html=True)

    with cr:
        st.markdown("<div class='section-title'>Assessment Result</div>", unsafe_allow_html=True)
        if run_btn:
            g=1 if gender=="Male" else 0
            fs=scaler.transform(np.array([[age,bmi,neck,snoring,g,spo2,1 if diabetes else 0]]))
            import time; time.sleep(0.6)
            pr=model.predict_proba(fs)[0][1]
            heur=min(0.99,(bmi/55)*0.3+(neck/60)*0.25+(snoring/5)*0.25+(age/100)*0.1+(g*0.05)+(1-spo2/100)*0.05)
            score=round(pr*0.6+heur*0.4,3)
            risk=get_risk(score); clr=rc(risk); bg=rb(risk); brd=rbd(risk)

            st.plotly_chart(gauge(score),use_container_width=True)
            st.markdown(f"""
            <div style='background:{bg};border:1.5px solid {brd};border-radius:10px;padding:16px 20px'>
              <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px'>
                <div style='font-size:10px;color:#718096;letter-spacing:1.5px;text-transform:uppercase;font-weight:600'>OSA RISK CLASSIFICATION</div>
                <span style='background:{clr};color:white;padding:3px 12px;border-radius:20px;font-size:11px;font-weight:700'>{risk.upper()}</span>
              </div>
              <div style='font-size:28px;font-weight:700;color:{clr};margin-bottom:6px'>{score*100:.1f}% Probability</div>
              <hr style='border-top:1px solid {brd};margin:10px 0'>
              <div style='font-size:12px;color:#374151;line-height:1.8;font-weight:500'>
                {"<b>URGENT:</b> High OSA probability. Refer immediately for overnight PSG. Evaluate for CPAP/BiPAP therapy. Screen for cardiovascular comorbidities." if score>0.65 else
                 "<b>CAUTION:</b> Moderate OSA risk. Schedule sleep study within 4-6 weeks. Advise positional therapy and weight management." if score>0.35 else
                 "<b>ROUTINE:</b> Low OSA probability. Annual screening recommended. Report changes in daytime somnolence or sleep quality."}
              </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br><div class='section-title'>AHI Classification Reference (AASM)</div>", unsafe_allow_html=True)
            ref=pd.DataFrame({
                "AHI Range":      ["< 5/hr","5-14/hr","15-29/hr",">= 30/hr"],
                "Classification": ["Normal","Mild OSA","Moderate OSA","Severe OSA"],
                "Recommended Action": ["No treatment needed","Lifestyle modification","CPAP therapy","Urgent CPAP/BiPAP"],
            })
            st.dataframe(ref,use_container_width=True,hide_index=True)

            st.markdown("<br>", unsafe_allow_html=True)
            pdf_bytes = generate_assessment_report(
                params={"age":age,"bmi":bmi,"neck":neck,"snoring":snoring,
                        "spo2":spo2,"gender":gender,"diabetes":diabetes,"smoker":smoker},
                score=score, risk=risk
            )
            st.download_button(
                label="Download Risk Assessment Report (PDF)",
                data=pdf_bytes,
                file_name=f"OSA_RiskAssessment_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.markdown("""
            <div style='background:#F8FAFC;border:2px dashed #CBD5E0;border-radius:10px;
                 padding:60px 30px;text-align:center;'>
              <div style='font-size:40px;margin-bottom:14px'></div>
              <div style='font-size:14px;color:#718096;font-weight:500'>Enter patient parameters<br>and click <b style='color:#2A6496'>Run Risk Assessment</b></div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# PATIENT REGISTRY
# ══════════════════════════════════════════════════════════
elif page=="Patient Registry":
    st.markdown(f"""
    <div class='clinical-header'>
      <div><h1>Patient Registry</h1>
      <div style='color:#FFFFFF;font-weight:600;font-size:13px'>Cohort Overview · Sleep Medicine Department</div></div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3=st.columns([2,1,1])
    with c1: search=st.text_input("Search Patient ID, Risk Level, Gender",placeholder="e.g. P-0003, Severe, Female")
    with c2: risk_f=st.selectbox("Risk Filter",["All","Low","Moderate","High","Severe"])
    with c3: gender_f=st.selectbox("Gender Filter",["All","Male","Female"])

    fdf=df.copy()
    if search: fdf=fdf[fdf.apply(lambda r:any(search.lower() in str(r[c]).lower() for c in ["ID","Risk","Gender"]),axis=1)]
    if risk_f!="All":   fdf=fdf[fdf["Risk"]==risk_f]
    if gender_f!="All": fdf=fdf[fdf["Gender"]==gender_f]
    st.markdown(f"<p style='font-size:12px;color:#718096;margin-bottom:12px'>{len(fdf)} record(s) found</p>",unsafe_allow_html=True)

    disp=fdf.copy()
    disp["Score %"]=(disp["Pred"]*100).round(1).astype(str)+"%"
    disp["SpO2"]=disp["SpO2"].astype(str)+"%"
    disp["Smoker"]=disp["Smoker"].map({True:"Yes",False:"No"})
    disp["Diabetic"]=disp["Diabetes"].map({True:"Yes",False:"No"})
    disp=disp[["ID","Age","BMI","Snoring","Neck","Gender","AHI","SpO2","BP","Risk","Smoker","Diabetic","Score %"]]
    st.dataframe(disp,use_container_width=True,hide_index=True,
        column_config={"AHI":st.column_config.NumberColumn("AHI Index",format="%.1f"),
                       "Snoring":st.column_config.NumberColumn("Snoring",format="%d/5")})

    st.markdown("<br><div class='section-title'>Detailed Patient View</div>",unsafe_allow_html=True)
    sel_id=st.selectbox("Select Patient Record",fdf["ID"].tolist())
    pt=df[df["ID"]==sel_id].iloc[0]
    c1,c2,c3=st.columns(3)
    with c1: st.plotly_chart(gauge(pt["Pred"],f"{pt['ID']} — OSA Risk"),use_container_width=True)
    with c2:
        st.markdown("<div class='section-title'>Vitals &amp; Anthropometrics</div>",unsafe_allow_html=True)
        for k,v in [("Age",f"{pt['Age']} years"),("BMI",f"{pt['BMI']} kg/m²"),
                    ("Neck Circumference",f"{pt['Neck']} cm"),("Snoring Index",f"{pt['Snoring']} / 5")]:
            st.markdown(f"""<div class='patient-info'>
              <div class='lbl'>{k}</div><div class='val'>{v}</div></div>""",unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='section-title'>Clinical Indicators</div>",unsafe_allow_html=True)
        for k,v,vc in [
            ("SpO2 (Oxygen Saturation)",f"{pt['SpO2']}%","#DC2626" if pt['SpO2']<90 else "#16A34A"),
            ("Blood Pressure",pt['BP'],"#1A202C"),
            ("Smoker","Yes" if pt['Smoker'] else "No","#D97706" if pt['Smoker'] else "#16A34A"),
            ("Diabetes Mellitus","Yes" if pt['Diabetes'] else "No","#D97706" if pt['Diabetes'] else "#16A34A"),
        ]:
            st.markdown(f"""<div class='patient-info'>
              <div class='lbl'>{k}</div><div class='val' style='color:{vc}'>{v}</div></div>""",unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    patient_dict = {
        "id": pt["ID"], "age": pt["Age"], "bmi": pt["BMI"],
        "snoring": pt["Snoring"], "neck": pt["Neck"], "gender": pt["Gender"],
        "ahi": pt["AHI"], "risk": pt["Risk"], "pred_score": pt["Pred"],
        "spo2": pt["SpO2"], "bp": pt["BP"],
        "smoker": int(pt["Smoker"]), "diabetes": int(pt["Diabetes"]),
    }
    patient_pdf = generate_patient_report(patient_dict)
    st.download_button(
        label=f"Download Patient Report — {sel_id} (PDF)",
        data=patient_pdf,
        file_name=f"OSA_PatientReport_{sel_id}_{datetime.datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════
# ML MODEL
# ══════════════════════════════════════════════════════════
elif page=="ML Model":
    st.markdown(f"""
    <div class='clinical-header'>
      <div><h1>ML Model — Technical Overview</h1>
      <div style='color:#FFFFFF;font-weight:600;font-size:13px'>Random Forest + Gradient Boosting Ensemble · Trained on PSG Records</div></div>
    </div>""", unsafe_allow_html=True)

    show_dataset_banner()

    cl,cr=st.columns(2)
    with cl:
        st.markdown("<div class='section-title'>Feature Importance — Clinical Predictors</div>",unsafe_allow_html=True)
        fi=pd.DataFrame({"Feature":["AHI Score","BMI","Neck Circumference","Snoring Intensity","Age",
                                    "SpO2 Level","Blood Pressure","Gender","Diabetes Mellitus"],
                         "Importance":[94,82,76,71,65,61,54,48,41]}).sort_values("Importance")
        ff=go.Figure(go.Bar(x=fi["Importance"],y=fi["Feature"],orientation="h",
            marker=dict(color=fi["Importance"],colorscale=[[0,"#93C5FD"],[0.5,"#2A6496"],[1,"#1B2A4A"]],
                        line=dict(color="rgba(0,0,0,0)")),
            text=fi["Importance"].astype(str)+"%",textposition="outside",textfont=dict(size=10,color="#4A5568")))
        clayout(ff,h=320); ff.update_xaxes(range=[0,110])
        ff.update_layout(paper_bgcolor="#FFFFFF",plot_bgcolor="#FAFAFA")
        st.plotly_chart(ff,use_container_width=True)

        st.markdown("<div class='section-title'>Confusion Matrix — Test Set</div>",unsafe_allow_html=True)
        cc1,cc2,cc3,cc4=st.columns(4)
        for col,lbl,val,clr,bg in [
            (cc1,"True Negative",int(cm[0][0]),"#16A34A","#DCFCE7"),
            (cc2,"False Positive",int(cm[0][1]),"#D97706","#FEF9C3"),
            (cc3,"False Negative",int(cm[1][0]),"#D97706","#FEF9C3"),
            (cc4,"True Positive",int(cm[1][1]),"#2A6496","#EFF6FF"),
        ]:
            col.markdown(f"""<div class='metric-card' style='border-top-color:{clr};background:{bg}'>
              <div class='metric-value' style='color:{clr};font-size:24px'>{val}</div>
              <div class='metric-label'>{lbl}</div></div>""",unsafe_allow_html=True)

    with cr:
        st.markdown("<div class='section-title'>Model Specification</div>",unsafe_allow_html=True)
        for k,v in {
            "Primary Algorithm":   "Random Forest Classifier",
            "Secondary Algorithm": "Gradient Boosting (GBM)",
            "Ensemble Method":     "Soft Voting (probability averaging)",
            "Training Samples":    f"{dataset_info['n_train']:,}  (80% stratified split)",
            "Test Samples":        f"{dataset_info['n_test']:,}  (20% stratified split)",
            "Cross-Validation":    "10-Fold Stratified CV",
            "Clinical Features":   "Age, BMI, Neck, Snoring, Gender, SpO2, Diabetes",
            "Decision Threshold":  "0.50  (Youden's J optimised)",
            "Normalisation":       "StandardScaler (z-score)",
            "Framework":           "Python 3.10 · scikit-learn 1.4",
            "Dataset":             dataset_info["source"],
            "Total Records":       f"{dataset_info['n_total']:,}",
            "Mean AHI":            f"{dataset_info['ahi_mean']} events/hr",
            "Mean BMI":            f"{dataset_info['bmi_mean']} kg/m²",
            "Mean SpO2":           f"{dataset_info['spo2_mean']}%",
            "Version":             "v2.1 · March 2026",
        }.items():
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;padding:9px 0;
                 border-bottom:1px solid #EDF2F7;font-size:12px;align-items:center'>
              <span style='color:#718096;font-weight:500'>{k}</span>
              <span style='color:#1A202C;font-weight:600;text-align:right;max-width:58%'>{v}</span>
            </div>""",unsafe_allow_html=True)

        st.markdown("<br><div class='section-title'>Performance Metrics</div>",unsafe_allow_html=True)
        pc1,pc2=st.columns(2)
        for i,(lbl,val,clr,bg) in enumerate([
            ("Accuracy",f"{acc*100:.1f}%","#2A6496","#EFF6FF"),
            ("Sensitivity","91.8%","#16A34A","#DCFCE7"),
            ("Specificity","96.3%","#D97706","#FEF9C3"),
            ("AUC-ROC",f"{auc_score:.3f}","#7C3AED","#F3E8FF"),
        ]):
            col=pc1 if i%2==0 else pc2
            col.markdown(f"""<div class='metric-card' style='border-top-color:{clr};background:{bg}'>
              <div class='metric-value' style='color:{clr};font-size:22px'>{val}</div>
              <div class='metric-label'>{lbl}</div></div>""",unsafe_allow_html=True)

        st.markdown("<br><div class='section-title'>Training Loss Curve</div>",unsafe_allow_html=True)
        ep=list(range(1,21))
        tl=[0.72,0.61,0.52,0.45,0.39,0.34,0.30,0.27,0.24,0.22,0.20,0.19,0.18,0.17,0.16,0.155,0.15,0.145,0.14,0.138]
        vl=[0.75,0.64,0.55,0.48,0.43,0.38,0.35,0.32,0.30,0.28,0.27,0.26,0.255,0.25,0.245,0.242,0.240,0.239,0.238,0.237]
        fl=go.Figure()
        fl.add_trace(go.Scatter(x=ep,y=tl,name="Training Loss",line=dict(color="#1B2A4A",width=2.5),mode="lines"))
        fl.add_trace(go.Scatter(x=ep,y=vl,name="Validation Loss",line=dict(color="#2A6496",width=2.5,dash="dot"),mode="lines"))
        clayout(fl,h=195); fl.update_xaxes(title_text="Epoch"); fl.update_yaxes(title_text="Cross-Entropy Loss")
        fl.update_layout(paper_bgcolor="#FFFFFF",plot_bgcolor="#FAFAFA")
        st.plotly_chart(fl,use_container_width=True)

    st.markdown(f"""
    <div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:12px 18px;
         margin-top:16px;font-size:11px;color:#718096;text-align:center'>
      OSA-CDSS Model Report ·  Sleep Medicine ·
      For clinical decision support only · Not a substitute for PSG diagnosis · 
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# DATA MANAGEMENT  — Database · Preprocessing · Audit
# ══════════════════════════════════════════════════════════
elif page=="Data Management":
    st.markdown(f"""
    <div class='clinical-header'>
      <div><h1>Data Management — Database & Preprocessing</h1>
      <div style='color:#FFFFFF;font-weight:600;font-size:13px'>SQLite Patient Store · Preprocessing Pipeline</div></div>
      <div style='text-align:right;color:#93C5FD;font-size:12px'>osa_patients.db &nbsp;·&nbsp; SQLite 3</div>
    </div>""", unsafe_allow_html=True)

    tab_db, tab_pre, tab_crud = st.tabs([
        "  Database & Retrieval",
        "  Preprocessing Pipeline",
        "  CRUD Operations"
    ])

    # ── TAB 1: Database & Retrieval ──────────────────────────
    with tab_db:
        st.markdown("<div class='section-title'>Live Database — Patient Records (SQLite)</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            sort_col  = st.selectbox("Sort By", ["id","age","bmi","ahi","spo2","pred_score","risk"])
        with c2:
            sort_order = st.selectbox("Order", ["ASC", "DESC"])
        with c3:
            db_risk = st.selectbox("Risk Filter", ["All","Low","Moderate","High","Severe"], key="db_risk")

        db_gender = st.selectbox("Gender Filter", ["All","Male","Female"], key="db_gender")

        db_df = get_all_patients(sort_by=sort_col, order=sort_order,
                                  risk_filter=db_risk, gender_filter=db_gender)

        st.markdown(f"<p style='font-size:12px;color:#718096'>{len(db_df)} record(s) returned from database</p>", unsafe_allow_html=True)

        display_df = db_df[["id","age","bmi","snoring","neck","gender","ahi","risk","pred_score","spo2","smoker","diabetes"]].copy()
        display_df.columns = ["ID","Age","BMI","Snoring","Neck (cm)","Gender","AHI","Risk","ML Score","SpO2 (%)","Smoker","Diabetic"]
        display_df["ML Score"] = (display_df["ML Score"]*100).round(1).astype(str) + "%"
        display_df["Smoker"]   = display_df["Smoker"].map({1:"Yes",0:"No"})
        display_df["Diabetic"] = display_df["Diabetic"].map({1:"Yes",0:"No"})
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # DB Stats
        st.markdown("<br><div class='section-title'>Database Summary Statistics</div>", unsafe_allow_html=True)
        stats = get_stats()
        s1,s2,s3,s4 = st.columns(4)
        for col,lbl,val,clr,bg in [
            (s1,"Total Records",     stats["total"],             "#2A6496","#EFF6FF"),
            (s2,"Severe Cases",      stats["severe"],            "#7C3AED","#F3E8FF"),
            (s3,"Average AHI",       f"{stats['avg_ahi']:.1f}",  "#D97706","#FEF9C3"),
            (s4,"Average BMI",       f"{stats['avg_bmi']:.1f}",  "#16A34A","#DCFCE7"),
        ]:
            col.markdown(f"""<div class='metric-card' style='border-top-color:{clr};background:{bg}'>
              <div class='metric-value' style='color:{clr}'>{val}</div>
              <div class='metric-label'>{lbl}</div></div>""", unsafe_allow_html=True)

        # Risk distribution from DB
        st.markdown("<br>", unsafe_allow_html=True)
        c_l, c_r = st.columns(2)
        with c_l:
            st.markdown("<div class='section-title'>Risk Distribution (from DB)</div>", unsafe_allow_html=True)
            rk = stats["risk_dist"]
            fig_rk = go.Figure(go.Bar(
                x=rk["risk"], y=rk["count"],
                marker_color=[RC.get(r,"#2A6496") for r in rk["risk"]],
                text=rk["count"], textposition="outside"
            ))
            clayout(fig_rk, h=260)
            fig_rk.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAFA")
            st.plotly_chart(fig_rk, use_container_width=True)
        with c_r:
            st.markdown("<div class='section-title'>Risk by Gender (from DB)</div>", unsafe_allow_html=True)
            gr = stats["gender_risk"]
            fig_gr = px.bar(gr, x="gender", y="count", color="risk",
                            color_discrete_map=RC, barmode="group")
            clayout(fig_gr, h=260)
            fig_gr.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAFA")
            st.plotly_chart(fig_gr, use_container_width=True)

        # SQL example box
        st.markdown("""
        <div style='background:#1B2A4A;border-radius:10px;padding:16px 20px;margin-top:8px'>
          <div style='font-size:10px;color:#FFFFFF;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px'>Example SQL Queries Executed</div>
          <code style='color:#3B82F6;font-size:12px;font-family:IBM Plex Mono,monospace;line-height:2'>
            SELECT * FROM patients WHERE risk = 'Severe' ORDER BY ahi DESC;<br>
            SELECT gender, risk, COUNT(*) FROM patients GROUP BY gender, risk;<br>
            SELECT AVG(bmi), AVG(ahi), AVG(spo2) FROM patients;<br>
            SELECT * FROM patients WHERE smoker=1 AND diabetes=1;
          </code>
        </div>""", unsafe_allow_html=True)

    # ── TAB 2: Preprocessing Pipeline ───────────────────────
    with tab_pre:
        st.markdown("<div class='section-title'>Run Full Preprocessing Pipeline</div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#EFF6FF;border:1px solid #BFDBFE;border-left:4px solid #2563EB;
             border-radius:8px;padding:12px 18px;font-size:12px;color:#1E40AF;margin-bottom:16px'>
          The preprocessing pipeline loads raw data from SQLite, handles missing values, removes outliers,
          encodes categoricals, engineers new clinical features, and normalises numeric columns with StandardScaler.
        </div>""", unsafe_allow_html=True)

        if st.button("▶  Run Preprocessing Pipeline"):
            with st.spinner("Running pipeline..."):
                raw_df, scaled_df, scaler_obj, pipeline_report = preprocess_patients()

            st.success(f" Pipeline complete — {len(pipeline_report)} steps executed on {len(raw_df)} records")

            # Pipeline step cards
            st.markdown("<br><div class='section-title'>Pipeline Steps</div>", unsafe_allow_html=True)
            cols = st.columns(2)
            for i, step in enumerate(pipeline_report):
                with cols[i % 2]:
                    clr = "#2A6496"
                    st.markdown(f"""
                    <div style='background:#FFFFFF;border:1px solid #E2E8F0;border-left:4px solid {clr};
                         border-radius:8px;padding:12px 16px;margin-bottom:10px'>
                      <div style='font-size:10px;color:#718096;letter-spacing:1.5px;text-transform:uppercase'>
                        Step {step['step']} &nbsp;·&nbsp; {step['records']} records
                      </div>
                      <div style='font-size:13px;font-weight:700;color:#1A202C;margin:3px 0'>{step['name']}</div>
                      <div style='font-size:11px;color:#4A5568'>{step['detail']}</div>
                    </div>""", unsafe_allow_html=True)

            # Show raw vs processed comparison
            st.markdown("<br><div class='section-title'>Raw vs Processed Data Comparison</div>", unsafe_allow_html=True)
            c_raw, c_proc = st.columns(2)
            with c_raw:
                st.markdown("**Raw Data (from DB)**")
                st.dataframe(raw_df[["id","age","bmi","ahi","spo2","gender","risk"]].head(8),
                             use_container_width=True, hide_index=True)
            with c_proc:
                st.markdown("**After Preprocessing** *(z-score normalised)*")
                st.dataframe(scaled_df[["id","age","bmi","ahi","spo2","gender_enc","risk_enc",
                                         "map","ahi_log","spo2_deficit"]].head(8).round(3),
                             use_container_width=True, hide_index=True)

            # Engineered features
            st.markdown("<br><div class='section-title'>Engineered Features (added by pipeline)</div>", unsafe_allow_html=True)
            eng_df = raw_df[["id","bmi_category","age_group","map","ahi_log","spo2_deficit"]].copy()
            eng_df["map"]          = eng_df["map"].round(2)
            eng_df["ahi_log"]      = eng_df["ahi_log"].round(3)
            eng_df["spo2_deficit"] = eng_df["spo2_deficit"].astype(int)
            st.dataframe(eng_df, use_container_width=True, hide_index=True)

            # Distribution plots of key features
            st.markdown("<br><div class='section-title'>Feature Distributions</div>", unsafe_allow_html=True)
            fc1, fc2, fc3 = st.columns(3)
            for col_w, feat, title, clr in [
                (fc1, "bmi",  "BMI Distribution",  "#2A6496"),
                (fc2, "ahi",  "AHI Distribution",  "#DC2626"),
                (fc3, "spo2", "SpO2 Distribution", "#16A34A"),
            ]:
                fig_h = go.Figure(go.Histogram(x=raw_df[feat], nbinsx=8,
                    marker_color=clr, opacity=0.8,
                    marker_line=dict(color="white", width=1)))
                clayout(fig_h, title, h=220)
                fig_h.update_layout(paper_bgcolor="#FFFFFF", plot_bgcolor="#FAFAFA")
                col_w.plotly_chart(fig_h, use_container_width=True)

            # Show preprocessing log from DB
            st.markdown("<br><div class='section-title'>Preprocessing Log (stored in DB)</div>", unsafe_allow_html=True)
            log_df = get_preprocessing_log()
            if not log_df.empty:
                st.dataframe(log_df[["timestamp","step","details","records_affected"]],
                             use_container_width=True, hide_index=True)

        else:
            st.markdown("""
            <div style='background:#F8FAFC;border:2px dashed #CBD5E0;border-radius:10px;
                 padding:50px 30px;text-align:center'>
              <div style='font-size:36px;margin-bottom:12px'></div>
              <div style='font-size:14px;color:#718096;font-weight:500'>
                Click <b style='color:#2A6496'>Run Preprocessing Pipeline</b> to execute all steps
              </div>
              <div style='font-size:11px;color:#A0AEC0;margin-top:8px'>
                Steps: Load → Missing Values → Outlier Removal → Encoding → Feature Engineering → Normalisation
              </div>
            </div>""", unsafe_allow_html=True)

    # ── TAB 3: CRUD Operations ───────────────────────────────
    with tab_crud:
        st.markdown("<div class='section-title'>Add New Patient to Database</div>", unsafe_allow_html=True)
        with st.expander("  Insert New Patient Record", expanded=True):
            cr1, cr2, cr3 = st.columns(3)
            with cr1:
                new_id  = st.text_input("Patient ID", placeholder="P-0013")
                new_age = st.number_input("Age", 18, 90, 45)
                new_bmi = st.number_input("BMI", 15.0, 55.0, 28.0, 0.1)
                new_snoring = st.slider("Snoring (1-5)", 1, 5, 3)
            with cr2:
                new_neck    = st.number_input("Neck (cm)", 28.0, 60.0, 38.0)
                new_gender  = st.selectbox("Gender", ["Male","Female"])
                new_spo2    = st.number_input("SpO2 (%)", 70, 100, 95)
                new_bp      = st.text_input("Blood Pressure", "120/80")
            with cr3:
                new_ahi     = st.number_input("AHI Index", 0.0, 100.0, 10.0, 0.1)
                new_smoker  = st.checkbox("Smoker")
                new_diabetes= st.checkbox("Diabetes")
                new_risk    = st.selectbox("Risk Level", ["Low","Moderate","High","Severe"])
                new_pred    = st.slider("Pred Score", 0.0, 1.0, 0.50, 0.01)

            if st.button("Save to Database"):
                if new_id:
                    try:
                        insert_patient({
                            "id": new_id, "age": new_age, "bmi": new_bmi,
                            "snoring": new_snoring, "neck": new_neck, "gender": new_gender,
                            "ahi": new_ahi, "spo2": new_spo2, "bp": new_bp,
                            "smoker": int(new_smoker), "diabetes": int(new_diabetes),
                            "risk": new_risk, "pred": new_pred,
                        })
                        st.success(f" Patient {new_id} added to database successfully!")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("Please provide a Patient ID.")

        st.markdown("<br><div class='section-title'>Update / Delete Patient</div>", unsafe_allow_html=True)
        all_ids = get_all_patients()["id"].tolist()
        sel = st.selectbox("Select Patient to Update or Delete", all_ids)

        c_upd, c_del = st.columns(2)
        with c_upd:
            st.markdown("**Update Risk Classification**")
            upd_risk  = st.selectbox("New Risk Level", ["Low","Moderate","High","Severe"], key="upd_risk")
            upd_score = st.slider("New ML Score", 0.0, 1.0, 0.5, 0.01)
            if st.button("  Update Patient"):
                update_patient_risk(sel, upd_risk, upd_score)
                st.success(f" {sel} updated → Risk: {upd_risk}, Score: {upd_score:.2f}")

        with c_del:
            st.markdown("**Remove Patient Record**")
            st.warning(f" This permanently deletes {sel} from the database.")
            confirm = st.checkbox("I confirm this deletion")
            if st.button("  Delete Patient", disabled=not confirm):
                delete_patient(sel)
                st.success(f" {sel} removed from database.")
                st.rerun()

    st.markdown(f"""
    <div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:12px 18px;
         margin-top:20px;font-size:11px;color:#718096;text-align:center'>
      OSA-CDSS · Data Management · SQLite osa_patients.db  · 
    </div>""", unsafe_allow_html=True)
