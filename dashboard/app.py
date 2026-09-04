from pathlib import Path
import sys
import itertools
from contextlib import contextmanager
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FlowCast",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1] if len(Path(__file__).resolve().parents) > 1 else Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

MODEL_DIR      = BASE_DIR / "models" / "classical"
DEEP_MODEL_DIR = BASE_DIR / "models" / "deep_learning"
REPORT_DIR     = BASE_DIR / "reports"


# ============================================================
# 3. COLOR SYSTEM
# ============================================================

C_BLUE   = "#3B82D0"
C_PURPLE = "#8B5CF6"
C_GREEN  = "#36B89C"
C_AMBER  = "#FFB000"
C_ORANGE = "#F97316"
C_RED    = "#F05454"

CONGESTION_COLORS = {
    "Free-flow": C_GREEN,
    "Moderate":  C_AMBER,
    "Heavy":     C_ORANGE,
    "Severe":    C_RED,
}
SEVERITY_MAP = {"Free-flow": 0, "Moderate": 1, "Heavy": 2, "Severe": 3}


# ============================================================
# 4. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ── GLOBAL ─────────────────────────────────────────── */
    .stApp { background: #111214; color: #E5E7EB; font-family: Inter, Arial, sans-serif; }
    [data-testid="stAppViewContainer"] { background: #111214; }
    [data-testid="stHeader"]           { background: #111214; }
    [data-testid="stToolbar"]          { right: 20px; }
    /* Added top padding so page header clears Streamlit's toolbar */
    .block-container { padding-top: 56px; padding-bottom: 30px; max-width: 1600px; }

    /* ============================================================
       FLOWCAST SIDEBAR
       ============================================================ */

    [data-testid="stSidebar"] {
        background: #0D0E10 !important;
        border-right: 1px solid #25282D !important;
    }

    [data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] { padding-top: 0 !important; }

    [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-left: 12px !important;
        padding-right: 12px !important;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0 !important; }

    /* BRAND */
    .sidebar-brand {
        padding: 14px 10px 12px 10px;
        border-bottom: 1px solid #25282D;
        margin-bottom: 8px;
    }
    .brand-row { display: flex; align-items: center; gap: 10px; }
    .brand-icon { font-size: 25px; line-height: 1; flex: 0 0 auto; }
    .brand-title { font-size: 21px; font-weight: 700; color: #F3F4F6; margin: 0; line-height: 1.15; }
    .brand-subtitle { font-size: 12.5px; color: #9CA3AF; margin-top: 3px; line-height: 1.2; white-space: nowrap; }
    .brand-status {
        display: flex; align-items: center; gap: 6px; margin-top: 9px;
        font-size: 10px; font-weight: 700; letter-spacing: 1.25px; color: #36B89C; text-transform: uppercase;
    }
    .brand-status-dot {
        width: 7px; height: 7px; min-width: 7px; border-radius: 50%;
        background: #36B89C; animation: pulse 2s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }

    /* SECTION HEADINGS */
    .section-title {
        font-size: 9px !important; font-weight: 700 !important; letter-spacing: 1.5px !important;
        color: #7F8794 !important; margin: 10px 8px 4px 8px !important;
        padding: 0 !important; line-height: 1.1 !important;
    }

    /* NAV BUTTONS */
    .sidebar-nav-wrap { width: 100%; margin: 0 !important; padding: 0 !important; }

    [data-testid="stSidebar"] .stButton { width: 100% !important; margin: 0 0 2px 0 !important; padding: 0 !important; }
    [data-testid="stSidebar"] .stButton > div { width: 100% !important; }
    [data-testid="stSidebar"] .stButton button {
        width: 100% !important; min-height: 34px !important; height: 34px !important;
        padding: 0 10px !important; display: flex !important; align-items: center !important;
        justify-content: flex-start !important; background: transparent !important;
        color: #C7CBD1 !important; border: 1px solid transparent !important;
        border-left: 3px solid transparent !important; border-radius: 7px !important;
        font-size: 12.5px !important; font-weight: 500 !important; line-height: 1 !important;
        margin: 0 !important; box-shadow: none !important;
        transition: background .12s ease, color .12s ease, border-color .12s ease;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: #171A1F !important; color: #FFFFFF !important; border-left-color: #30353D !important;
    }
    [data-testid="stSidebar"] .stButton button:focus,
    [data-testid="stSidebar"] .stButton button:focus-visible { outline: none !important; box-shadow: none !important; }

    [data-testid="stSidebar"] div[class*="st-key-nav-active-"] .stButton button {
        background: #15355D !important; color: #9EC5FF !important;
        border-left-color: #3B82D0 !important; font-weight: 600 !important;
    }
    [data-testid="stSidebar"] div[class*="st-key-nav-active-"] .stButton button:hover {
        background: #19416F !important;
    }

    /* SIDEBAR FOOTER */
    .sb-footer { padding: 10px 10px 4px 10px; border-top: 1px solid #25282D; margin-top: 7px; }
    .sb-status-block { margin-bottom: 8px; }
    .sb-status-label { font-size: 9px; font-weight: 700; color: #6B7280; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }
    .sb-status-row { display: flex; align-items: center; gap: 6px; font-size: 11px; color: #C7CBD1; }
    .sb-dot-ok, .sb-dot-err { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
    .sb-dot-ok { background: #36B89C; }
    .sb-dot-err { background: #F05454; }
    .sb-count { font-size: 10px; color: #6B7280; margin-top: 2px; padding-left: 13px; }

    /* ── MAIN HEADER ────────────────────────────────────── */
    .breadcrumb    { font-size: 15px; font-weight: 700; letter-spacing: 1.5px; color: #8FAFD6; margin-bottom: 7px; }
    .header-row    { display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 12px; }
    .page-title    { font-size: 29px; font-weight: 700; color: #F1F3F5; margin: 0; line-height: 1.2; }
    .page-subtitle { font-size: 17px; font-weight: 500; color: #A1A6AE; margin-top: 5px; margin-bottom: 4px; }
    .header-line   { height: 1px; width: 100%; background: #2A2D31; margin: 14px 0 18px 0; }
    .live-badge    { display: inline-flex; align-items: center; gap: 6px; background: rgba(54,184,156,0.12); border: 1px solid rgba(54,184,156,0.35); border-radius: 20px; padding: 4px 12px; font-size: 11px; font-weight: 700; color: #36B89C; letter-spacing: 0.06em; text-transform: uppercase; }
    .live-dot      { width: 6px; height: 6px; border-radius: 50%; background: #36B89C; animation: pulse 2s infinite; }
    .fc-timestamp  { font-size: 11px; color: #6B7280; margin-top: 6px; text-align: right; }

    /* ── KPI / METRIC CARDS ─────────────────────────────── */
    .metric-card { background: #1A1B1D; border: 1px solid #30343A; border-radius: 16px; padding: 22px 20px; min-height: 140px; margin-bottom: 18px; transition: 0.2s ease; }
    .metric-card:hover { border-color: #3F6FA6; transform: translateY(-2px); }
    .metric-label    { font-size: 13px; font-weight: 700; color: #9298A1; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 7px; }
    .metric-value    { font-size: 29px; font-weight: 700; color: #E8EAED; line-height: 1.1; }
    .metric-subtitle { font-size: 14px; color: #8F949C; margin-top: 7px; }

    /* ── STAT BAR CARDS (used in Historical Trends and overview) ── */
    .stat-kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 18px;
    }
    @media (max-width: 900px) { .stat-kpi-grid { grid-template-columns: repeat(2,1fr); } }

    .stat-kpi-card {
        background: #1A1B1D;
        border: 1px solid #30343A;
        border-radius: 14px;
        padding: 18px 18px 14px 18px;
        transition: border-color 0.2s, transform 0.2s;
        position: relative;
        overflow: hidden;
    }
    .stat-kpi-card:hover { border-color: #3F6FA6; transform: translateY(-2px); }
    .stat-kpi-card-accent {
        position: absolute; top: 0; left: 0; right: 0; height: 3px; border-radius: 14px 14px 0 0;
    }
    .stat-kpi-label { font-size: 11px; font-weight: 700; color: #6B7280; letter-spacing: 1.1px; text-transform: uppercase; margin-bottom: 8px; }
    .stat-kpi-value { font-size: 26px; font-weight: 700; color: #E8EAED; line-height: 1.1; }
    .stat-kpi-sub   { font-size: 13px; color: #8F949C; margin-top: 5px; }

    /* ── SECTION SPACER (generic vertical gap helper) ───── */
    .fc-spacer { width: 100%; }

    /* ── CHART CARDS ────────────────────────────────────── */
    .chart-title    { font-size: 18px; font-weight: 700; color: #E2E5E9; margin: 0 0 2px 0; }
    .chart-subtitle { font-size: 14.5px; color: #9298A1; margin: 0; }
    .chart-header   { padding: 14px 18px 12px 18px; border-bottom: 1px solid #24262A; }

    div[class*="st-key-chartcard_"] {
        background: #1A1B1D !important;
        border: 1px solid #30343A !important;
        border-radius: 16px !important;
        overflow: hidden;
        margin-bottom: 14px;
    }
    div[class*="st-key-chartcard_"] [data-testid="stVerticalBlock"]              { gap: 0 !important; }
    div[class*="st-key-chartcard_"] [data-testid="stVerticalBlockBorderWrapper"] { border: none !important; background: transparent !important; }
    div[class*="st-key-chartcard_"] [data-testid="stVerticalBlock"] > div:not(:first-child) [data-testid="element-container"],
    div[class*="st-key-chartcard_"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:not(:first-child) { padding: 0 14px; }
    div[class*="st-key-chartcard_"] [data-testid="stVerticalBlock"] > div:last-child,
    div[class*="st-key-chartcard_"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:last-child { padding-bottom: 12px; }

    /* Compact card variant */
    div[class*="st-key-chartcard_compact_"] {
        background: #1A1B1D !important;
        border: 1px solid #30343A !important;
        border-radius: 14px !important;
        overflow: hidden;
        margin-bottom: 14px;
    }
    div[class*="st-key-chartcard_compact_"] [data-testid="stVerticalBlock"]              { gap: 0 !important; }
    div[class*="st-key-chartcard_compact_"] [data-testid="stVerticalBlockBorderWrapper"] { border: none !important; background: transparent !important; }

    .data-card { background: #1A1B1D; border: 1px solid #30343A; border-radius: 16px; padding: 20px; margin-top: 10px; margin-bottom: 14px; }

    /* ── INLINE STAT CARDS (Congestion Map / Road Comparison) ── */
    .inline-stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 16px;
    }
    .inline-stat-card {
        background: #1A1B1D;
        border: 1px solid #30343A;
        border-radius: 14px;
        padding: 18px 20px 16px 20px;
        transition: border-color 0.2s, transform 0.2s;
        min-height: 128px;
    }
    .inline-stat-card:hover { border-color: #3F6FA6; transform: translateY(-2px); }
    .inline-stat-card-label { font-size: 11px; font-weight: 700; color: #6B7280; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px; }
    .inline-stat-card-sub   { font-size: 12px; color: #9298A1; margin-bottom: 10px; }
    .inline-stat-card-value { font-size: 28px; font-weight: 700; line-height: 1.1; }
    .inline-stat-card-detail{ font-size: 12px; color: #8F949C; margin-top: 6px; }

    /* ── LEGEND ─────────────────────────────────────────── */
    .legend-container { display: flex; flex-wrap: wrap; gap: 18px; margin-top: 10px; margin-bottom: 5px; }
    .legend-item { display: flex; align-items: center; gap: 7px; color: #B4B7BC; font-size: 14px; font-weight: 600; }
    .legend-dot  { width: 14px; height: 14px; border-radius: 3px; display: inline-block; }

    /* ── EMPTY STATES ───────────────────────────────────── */
    .empty-state  { text-align: center; padding: 56px 24px; background: #1A1B1D; border: 1px dashed #30343A; border-radius: 16px; }
    .empty-icon   { font-size: 2.8rem; margin-bottom: 16px; }
    .empty-title  { font-size: 1.1rem; font-weight: 700; color: #E2E5E9; margin-bottom: 8px; }
    .empty-text   { font-size: 13.5px; color: #8F949C; line-height: 1.6; max-width: 420px; margin: 0 auto; }
    .empty-code   { display: inline-block; margin-top: 14px; background: #111214; border: 1px solid #30343A; border-radius: 6px; padding: 6px 14px; font-family: monospace; font-size: 12.5px; color: #3B82D0; }

    /* ── INSIGHT CARDS ──────────────────────────────────── */
    .insight-card  { background: #16171A; border: 1px solid rgba(59,130,208,0.25); border-radius: 12px; padding: 16px 18px; margin-bottom: 12px; }
    .insight-title { font-size: 12px; font-weight: 700; color: #3B82D0; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 10px; }
    .insight-item  { font-size: 14px; color: #B4B7BC; padding: 5px 0; border-bottom: 1px solid #24262A; line-height: 1.5; }
    .insight-item:last-child { border-bottom: none; }

    /* ── PREDICTION HERO ────────────────────────────────── */
    .pred-hero   { background: linear-gradient(135deg,rgba(59,130,208,0.08) 0%,rgba(139,92,246,0.05) 100%); border: 1px solid rgba(59,130,208,0.3); border-radius: 16px; padding: 30px 32px; text-align: center; margin: 0 0 12px; }
    .pred-label  { font-size: 12px; font-weight: 700; color: #3B82D0; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px; }
    .pred-value  { font-size: 3.4rem; font-weight: 700; color: #F1F3F5; line-height: 1; }
    .pred-unit   { font-size: 14px; color: #8F949C; margin-top: 8px; }

    /* ── STATUS CHIPS ───────────────────────────────────── */
    .status-chip  { display: inline-flex; align-items: center; gap: 5px; padding: 5px 12px; border-radius: 20px; font-size: 13px; font-weight: 700; letter-spacing: 0.02em; }
    .chip-green   { background: rgba(54,184,156,0.14);  color: #36B89C; border: 1px solid rgba(54,184,156,0.3); }
    .chip-amber   { background: rgba(255,176,0,0.14);   color: #FFB000; border: 1px solid rgba(255,176,0,0.3); }
    .chip-orange  { background: rgba(249,115,22,0.14);  color: #F97316; border: 1px solid rgba(249,115,22,0.3); }
    .chip-red     { background: rgba(240,84,84,0.14);   color: #F05454; border: 1px solid rgba(240,84,84,0.3); }
    .chip-blue    { background: rgba(59,130,208,0.14);  color: #3B82D0; border: 1px solid rgba(59,130,208,0.3); }

    /* ── LEADERBOARD ────────────────────────────────────── */
    .leaderboard-row { display: flex; align-items: center; gap: 14px; padding: 14px 18px; background: #16171A; border: 1px solid #24262A; border-radius: 12px; margin-bottom: 10px; transition: border-color 0.2s; }
    .leaderboard-row:hover { border-color: rgba(59,130,208,0.35); }
    .rank-badge   { font-size: 1.1rem; font-weight: 700; min-width: 34px; text-align: center; }
    .ldr-name     { flex: 1; font-size: 14.5px; font-weight: 600; color: #E2E5E9; }
    .ldr-metrics  { display: flex; gap: 24px; flex-wrap: wrap; }
    .ldr-metric   { text-align: right; }
    .ldr-metric-label { font-size: 10.5px; color: #6B7280; letter-spacing: 0.04em; margin-bottom: 2px; }
    .ldr-metric-value { font-family: monospace; font-size: 13.5px; color: #E8EAED; font-weight: 600; }

    /* ── STAT BAR (legacy, still used in some charts) ─── */
    .stat-bar { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 0; margin-bottom: 14px; }
    @media (max-width: 900px) { .stat-bar { grid-template-columns: repeat(2, 1fr); } }
    .stat-item       { background: #16171A; border: 1px solid #24262A; border-radius: 10px; padding: 12px 14px; text-align: center; }
    .stat-item-label { font-size: 10.5px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 5px; }
    .stat-item-value { font-size: 1.2rem; font-weight: 700; color: #E8EAED; }

    /* ── CONFIDENCE GAUGE ───────────────────────────────── */
    .conf-gauge-wrap  { text-align: center; padding: 18px 0; }
    .conf-gauge-value { font-size: 2.8rem; font-weight: 700; color: #3B82D0; line-height: 1; }
    .conf-gauge-label { font-size: 12.5px; color: #8F949C; margin-top: 8px; }

    /* ── UPLOAD VALIDATION ──────────────────────────────── */
    .validation-item { display: flex; align-items: center; gap: 10px; padding: 10px 14px; background: #16171A; border-radius: 8px; margin-bottom: 8px; font-size: 13.5px; color: #B4B7BC; border: 1px solid #24262A; }
    .v-ok  { color: #36B89C; font-size: 15px; }
    .v-err { color: #F05454; font-size: 15px; }

    /* ── DATAFRAME ──────────────────────────────────────── */
    [data-testid="stDataFrame"] { border: 1px solid #30343A; border-radius: 10px; overflow: hidden; }

    /* ── REPORT CONTENT (Reports page markdown) ─────────── */
    .report-content-wrap {
        padding: 22px 26px;
        margin: 4px 0 18px 0;
    }
    .report-content-wrap h1,
    .report-content-wrap h2,
    .report-content-wrap h3 {
        margin-top: 20px;
        margin-bottom: 10px;
        color: #E2E5E9;
    }
    .report-content-wrap h1:first-child,
    .report-content-wrap h2:first-child,
    .report-content-wrap h3:first-child { margin-top: 0; }
    .report-content-wrap p { margin: 8px 0; line-height: 1.7; color: #B4B7BC; }
    .report-content-wrap ul,
    .report-content-wrap ol { margin: 8px 0 14px 22px; line-height: 1.7; color: #B4B7BC; }
    .report-content-wrap table { margin: 14px 0; border-collapse: collapse; width: 100%; }
    .report-content-wrap th, .report-content-wrap td { padding: 8px 12px; border: 1px solid #30343A; }
    .report-content-wrap code {
        background: #111214; border: 1px solid #30343A; border-radius: 4px; padding: 1px 6px;
        font-size: 12.5px; color: #3B82D0;
    }

    /* ── RESPONSIVE ─────────────────────────────────────── */
    @media (max-width: 900px) {
        .page-title  { font-size: 24px; }
        .metric-value{ font-size: 24px; }
        .pred-value  { font-size: 2.4rem; }
        .inline-stat-grid { grid-template-columns: 1fr; }
        .stat-kpi-grid    { grid-template-columns: repeat(2,1fr); }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. DATASET DISCOVERY
# ============================================================

def find_dataset():
    possible_files = [
        BASE_DIR / "data" / "processed" / "flowcast_features.csv",
        BASE_DIR / "data" / "processed" / "traffic_features.csv",
        BASE_DIR / "data" / "flowcast_features.csv",
        BASE_DIR / "data" / "traffic_data.csv",
        BASE_DIR / "dataset" / "flowcast_features.csv",
        BASE_DIR / "flowcast_features.csv",
    ]
    for p in possible_files:
        if p.exists():
            return p
    for name in ["flowcast_features.csv","traffic_features.csv","traffic_data.csv","processed_data.csv"]:
        found = list(BASE_DIR.rglob(name))
        if found:
            return found[0]
    return None


@st.cache_data
def load_data():
    path = find_dataset()
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()


# ============================================================
# 6. MODEL LOADING
# ============================================================

@st.cache_resource
def load_volume_model():
    path = MODEL_DIR / "xgboost_volume.joblib"
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


# ============================================================
# 7. COLUMN DETECTION
# ============================================================

def find_column(dataframe, possible_names):
    if dataframe.empty:
        return None
    col_map = {c.lower().strip(): c for c in dataframe.columns}
    for name in possible_names:
        if name.lower().strip() in col_map:
            return col_map[name.lower().strip()]
    return None

def has_columns(df, cols):
    return all(c in df.columns for c in cols)

def safe_mean(df, col):
    if col not in df.columns:
        return 0.0
    val = df[col].mean()
    return float(val) if pd.notna(val) else 0.0

def normalize_congestion(raw):
    s = str(raw).lower()
    if any(k in s for k in ["free","low","light"]):  return "Free-flow"
    if any(k in s for k in ["moderate","medium"]):   return "Moderate"
    if "heavy" in s:                                  return "Heavy"
    if any(k in s for k in ["severe","high"]):        return "Severe"
    return str(raw)


# ============================================================
# 8. LOAD & PREPROCESS DATA
# ============================================================

raw_df = load_data()

datetime_col    = find_column(raw_df, ["datetime","timestamp","date_time","date","time"])
road_col        = find_column(raw_df, ["road_id","road_name","road","road_segment","segment","location"])
volume_col      = find_column(raw_df, ["traffic_volume","vehicle_count","volume","vehicles","traffic_count","vehicle_volume"])
speed_col       = find_column(raw_df, ["avg_speed","average_speed","speed","vehicle_speed"])
congestion_col  = find_column(raw_df, ["congestion_level","congestion","traffic_level","congestion_status"])
rainfall_col    = find_column(raw_df, ["rainfall","rain","precipitation"])
visibility_col  = find_column(raw_df, ["visibility"])
temperature_col = find_column(raw_df, ["temperature","temp"])
travel_col      = find_column(raw_df, ["travel_time","traveltime"])
occupancy_col   = find_column(raw_df, ["occupancy"])
hour_col        = find_column(raw_df, ["hour"])

df = raw_df.copy()
rename_map = {}
if datetime_col:    rename_map[datetime_col]    = "timestamp"
if road_col:        rename_map[road_col]        = "road_id"
if volume_col:      rename_map[volume_col]      = "traffic_volume"
if speed_col:       rename_map[speed_col]       = "avg_speed"
if congestion_col:  rename_map[congestion_col]  = "congestion_level"
if rainfall_col:    rename_map[rainfall_col]    = "rainfall"
if visibility_col:  rename_map[visibility_col]  = "visibility"
if temperature_col: rename_map[temperature_col] = "temperature"
if travel_col:      rename_map[travel_col]      = "travel_time"
if occupancy_col:   rename_map[occupancy_col]   = "occupancy"
if hour_col:        rename_map[hour_col]        = "hour"

if not df.empty:
    df = df.rename(columns=rename_map)
    if "timestamp"      in df.columns: df["timestamp"]      = pd.to_datetime(df["timestamp"], errors="coerce")
    if "traffic_volume" in df.columns: df["traffic_volume"] = pd.to_numeric(df["traffic_volume"], errors="coerce")
    if "avg_speed"      in df.columns: df["avg_speed"]      = pd.to_numeric(df["avg_speed"], errors="coerce")
    if "hour" not in df.columns and "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
    if "congestion_level" in df.columns:
        df["congestion_level"] = df["congestion_level"].astype(str).str.strip().apply(normalize_congestion)

model_pkg   = load_volume_model()
model_ready = model_pkg is not None

# Resolve model name for sidebar footer
model_name = "XGBoost" if model_ready else "None"
if model_ready:
    try:
        m = model_pkg.get("model")
        if hasattr(m, "named_steps"):
            inner = m.named_steps.get("model", m)
            model_name = type(inner).__name__
        else:
            model_name = type(m).__name__
    except Exception:
        model_name = "XGBoost"


# ============================================================
# 9. PREDICTION HELPER
# ============================================================

def model_prediction_frame(data):
    if model_pkg is None or data.empty:
        return None
    try:
        model    = model_pkg["model"]
        features = model_pkg["features"]
        X = data.reindex(columns=features, fill_value=0)
        return model.predict(X)
    except Exception as e:
        st.warning(f"Prediction error: {e}")
        return None

def congestion_chip(level):
    cls = {"Free-flow":"chip-green","Moderate":"chip-amber","Heavy":"chip-orange","Severe":"chip-red"}.get(level,"chip-blue")
    return f'<span class="status-chip {cls}">{level}</span>'


# ============================================================
# 10. CHART THEME
# ============================================================

def apply_flowcast_theme(fig, height=340, title=""):
    fig.update_layout(
        paper_bgcolor="#1A1B1D", plot_bgcolor="#1A1B1D",
        font=dict(family="Inter, Arial, sans-serif", color="#9BA1A9", size=12),
        title=dict(text=title, font=dict(color="#E2E5E9", size=15)) if title else None,
        margin=dict(l=45, r=20, t=36 if title else 16, b=35),
        height=height,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#30343A", borderwidth=1, font=dict(color="#B4B7BC", size=12)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#1A1B1D", bordercolor="#3B82D0", font_color="#F1F3F5", font_size=12, namelength=-1),
        xaxis=dict(showgrid=True, gridcolor="#30343A", linecolor="#30343A", zeroline=False, tickfont=dict(color="#9BA1A9", size=11)),
        yaxis=dict(showgrid=True, gridcolor="#30343A", linecolor="#30343A", zeroline=False, tickfont=dict(color="#9BA1A9", size=11), tickformat=","),
        colorway=[C_BLUE, C_PURPLE, C_GREEN, C_AMBER, C_ORANGE, C_RED],
    )
    # Guard against any trace missing an explicit name — an unnamed trace
    # renders as the literal text "undefined" in the unified hover box.
    for trace in fig.data:
        if not getattr(trace, "name", None):
            trace.name = ""
            trace.showlegend = False
    return fig


# ============================================================
# 11. UI COMPONENTS
# ============================================================

def spacer(height_px=18):
    """Renders a plain vertical gap. Use between stacked sections/cards."""
    st.markdown(f'<div class="fc-spacer" style="height:{height_px}px"></div>', unsafe_allow_html=True)


def kpi_cards(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        color = item.get("color", C_BLUE)
        icon  = item.get("icon", "")
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-top:3px solid {color}">
                <div class="metric-label">{icon}&nbsp; {item.get('label','')}</div>
                <div class="metric-value">{item.get('value','—')}</div>
                <div class="metric-subtitle">{item.get('delta','')}</div>
            </div>""", unsafe_allow_html=True)


def stat_kpi_cards(items):
    """
    Renders a row of stat KPI cards identical in style to the metric cards
    used on other pages — with a coloured top accent, label, value, and
    optional sub-text. Used on Historical Trends and anywhere we want the
    min / max / avg / latest pattern in a proper card instead of the old
    stat-bar strip.
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        color = item.get("color", C_BLUE)
        with col:
            st.markdown(f"""
            <div class="stat-kpi-card" style="border-top:3px solid {color}">
                <div class="stat-kpi-card-accent" style="background:{color}"></div>
                <div class="stat-kpi-label">{item.get('label','')}</div>
                <div class="stat-kpi-value" style="color:{color}">{item.get('value','—')}</div>
                <div class="stat-kpi-sub">{item.get('sub','')}</div>
            </div>""", unsafe_allow_html=True)


def inline_stat_cards(items):
    """
    Renders a 3-up grid of inline stat cards (used on Congestion Map and
    Road Comparison). Rendered one column per card via st.columns + a single
    st.markdown call per card (same pattern as kpi_cards / stat_kpi_cards)
    instead of concatenating one big multi-line HTML blob — building one
    large indented f-string and handing it to a single st.markdown() call
    caused Streamlit's markdown parser to treat the indented lines as a
    literal code block, which is why raw HTML tags were showing up on
    screen instead of styled cards.
    """
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        color  = item.get("color", C_BLUE)
        detail = item.get("detail", "")
        detail_html = f'<div class="inline-stat-card-detail">{detail}</div>' if detail else ""
        with col:
            st.markdown(
                f'<div class="inline-stat-card" style="border-top:3px solid {color}">'
                f'<div class="inline-stat-card-label">{item.get("label","")}</div>'
                f'<div class="inline-stat-card-sub">{item.get("sub","")}</div>'
                f'<div class="inline-stat-card-value" style="color:{color}">{item.get("value","—")}</div>'
                f'{detail_html}'
                f'</div>',
                unsafe_allow_html=True,
            )


def empty_state(icon, title, text, code=""):
    code_html = f'<div class="empty-code">{code}</div>' if code else ""
    st.markdown(f"""
    <div class="empty-state">
        <div class="empty-icon">{icon}</div>
        <div class="empty-title">{title}</div>
        <div class="empty-text">{text}</div>
        {code_html}
    </div>""", unsafe_allow_html=True)

def insight_box(title, items):
    items_html = "".join(f'<div class="insight-item">• {i}</div>' for i in items)
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">{title}</div>
        {items_html}
    </div>""", unsafe_allow_html=True)

def stat_summary_bar(series, unit=""):
    """Legacy stat bar — kept for pages that still use it."""
    series = series.dropna()
    if series.empty:
        return
    mn, mx, avg, lat = series.min(), series.max(), series.mean(), series.iloc[-1]
    st.markdown(f"""
    <div class="stat-bar">
        <div class="stat-item"><div class="stat-item-label">Minimum</div><div class="stat-item-value">{mn:,.1f} {unit}</div></div>
        <div class="stat-item"><div class="stat-item-label">Maximum</div><div class="stat-item-value">{mx:,.1f} {unit}</div></div>
        <div class="stat-item"><div class="stat-item-label">Average</div><div class="stat-item-value">{avg:,.1f} {unit}</div></div>
        <div class="stat-item"><div class="stat-item-label">Latest</div><div class="stat-item-value">{lat:,.1f} {unit}</div></div>
    </div>""", unsafe_allow_html=True)

def page_header(icon, breadcrumb, title, subtitle):
    now = datetime.now().strftime("%I:%M %p")
    left, right = st.columns([3, 1])
    with left:
        st.markdown(f"""
        <div class="breadcrumb">{icon}&nbsp; FLOWCAST / {breadcrumb.upper()}</div>
        <div class="page-title">{title}</div>
        <div class="page-subtitle">{subtitle}</div>
        """, unsafe_allow_html=True)
    with right:
        st.markdown(f"""
        <div style="text-align:right;padding-top:6px">
            <span class="live-badge"><span class="live-dot"></span>Live System</span>
            <div class="fc-timestamp">Updated: {now}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)


_chart_card_counter = itertools.count()

@contextmanager
def chart_card(title, subtitle=""):
    key = f"chartcard_{next(_chart_card_counter)}"
    with st.container(key=key):
        st.markdown(
            f'<div class="chart-header"><div class="chart-title">{title}</div>'
            f'<div class="chart-subtitle">{subtitle}</div></div>',
            unsafe_allow_html=True,
        )
        yield


# ============================================================
# 12. SIDEBAR NAVIGATION  (Feature Importance & Data Upload removed)
# ============================================================

if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Overview"

active_page = st.session_state["active_page"]

NAV_ITEMS = [
    ("Analytics",   "Overview",          "▦", "overview"),
    ("Analytics",   "Historical Trends", "⌁", "historical"),
    ("Analytics",   "Congestion Map",    "⁙", "heatmap"),
    ("Analytics",   "Road Comparison",   "⌃", "comparison"),
    ("Analytics",   "Weather vs Traffic","☁", "weather"),
    ("Predictions", "Live Prediction",   "▷", "live_pred"),
    ("Predictions", "Forecast View",     "⌁", "forecast"),
    ("Predictions", "Confidence Band",   "⬡", "confidence"),
    ("Model & Data","Model Performance", "◈", "model_perf"),
    ("Model & Data","Reports",           "▤", "reports"),
]

PAGE_META = {
    "Overview":          ("▦", "Analytics",   "Overview",          "Network-wide traffic intelligence at a glance."),
    "Historical Trends": ("⌁", "Analytics",   "Historical Trends", "Explore traffic patterns across time and road segments."),
    "Congestion Map":    ("⁙", "Analytics",   "Congestion Map",    "Inspect congestion severity across the road network."),
    "Road Comparison":   ("⌃", "Analytics",   "Road Comparison",   "Compare traffic conditions across road segments."),
    "Weather vs Traffic":("☁", "Analytics",   "Weather vs Traffic","Understand how weather conditions relate to traffic volume."),
    "Live Prediction":   ("▷", "Predictions", "Live Prediction",   "Generate an AI-powered next-window traffic prediction."),
    "Forecast View":     ("⌁", "Predictions", "Forecast View",     "Compare actual traffic with model forecasts."),
    "Confidence Band":   ("⬡", "Predictions", "Confidence Band",   "Visualise prediction uncertainty and empirical intervals."),
    "Model Performance": ("◈", "Model & Data","Model Performance",  "Review trained model evaluation results."),
    "Reports":           ("▤", "Model & Data","Reports",            "Review generated data-quality reports and figures."),
}

VALID_PAGES = {item[1] for item in NAV_ITEMS}
if active_page not in VALID_PAGES:
    active_page = "Overview"
    st.session_state["active_page"] = active_page

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-row">
                <div class="brand-icon">🚦</div>
                <div>
                    <div class="brand-title">FlowCast</div>
                    <div class="brand-subtitle">AI-Powered Traffic Intelligence</div>
                </div>
            </div>
            <div class="brand-status">
                <span class="brand-status-dot"></span>
                SYSTEM ONLINE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    current_section = None

    for section, label, icon, route_key in NAV_ITEMS:
        if section != current_section:
            current_section = section
            st.markdown(
                f'<div class="section-title">{section.upper()}</div>',
                unsafe_allow_html=True,
            )

        is_active   = active_page == label
        wrapper_key = f"nav-active-{route_key}" if is_active else f"nav-{route_key}"

        with st.container(key=wrapper_key):
            if st.button(
                f"{icon}  {label}",
                key=f"sidebar_{route_key}",
                use_container_width=True,
            ):
                st.session_state["active_page"] = label
                st.rerun()

    # ---- Sidebar footer ----
    data_ready  = not df.empty
    rec_count   = f"{len(df):,} records loaded" if data_ready else "No records loaded"
    data_label  = "Ready" if data_ready else "Unavailable"
    data_class  = "sb-dot-ok" if data_ready else "sb-dot-err"
    model_label = f"Ready · {model_name}" if model_ready else "Unavailable"
    model_class = "sb-dot-ok" if model_ready else "sb-dot-err"

    st.markdown(
        f"""
        <div class="sb-footer">
            <div class="sb-status-block">
                <div class="sb-status-label">Data Status</div>
                <div class="sb-status-row">
                    <span class="{data_class}"></span>{data_label}
                </div>
                <div class="sb-count">{rec_count}</div>
            </div>
            <div class="sb-status-block">
                <div class="sb-status-label">Model Status</div>
                <div class="sb-status-row">
                    <span class="{model_class}"></span>{model_label}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Single page header (rendered once) ──────────────────────
icon, breadcrumb, title, subtitle = PAGE_META[active_page]
page_header(icon, breadcrumb, title, subtitle)


# ============================================================
# 13. NO-DATA GUARD
# ============================================================

if df.empty:
    empty_state(
        "📂", "No Data Available",
        "The processed traffic dataset could not be found. Make sure your dataset is located at "
        "data/processed/flowcast_features.csv, then run the pipeline to generate it.",
        "python run_pipeline.py",
    )
    st.stop()


# ============================================================
# 14. PAGE RENDERERS
# ============================================================

# ── Overview ─────────────────────────────────────────────────
def render_overview(data):
    records    = len(data)
    roads      = data["road_id"].nunique() if "road_id" in data.columns else 0
    avg_volume = safe_mean(data, "traffic_volume")
    avg_speed  = safe_mean(data, "avg_speed")

    kpi_cards([
        {"label":"Total Records",      "value":f"{records:,}",      "delta":"across all segments",  "icon":"▦",  "color":C_BLUE},
        {"label":"Road Segments",      "value":f"{roads:,}",         "delta":"active sensors",        "icon":"╱╲","color":C_PURPLE},
        {"label":"Avg Traffic Volume", "value":f"{avg_volume:,.0f}", "delta":"vehicles / window",     "icon":"⌁",  "color":C_AMBER},
        {"label":"Avg Speed",          "value":f"{avg_speed:.1f}",   "delta":"km/h network-wide",     "icon":"⚡", "color":C_GREEN},
    ])

    col1, col2 = st.columns([1, 1])

    with col1:
        with chart_card("Hourly Volume Pattern", "Network-wide average traffic volume by hour of day"):
            if has_columns(data, ["hour", "traffic_volume"]):
                hourly = data.groupby("hour")["traffic_volume"].mean().reset_index()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hourly["hour"], y=hourly["traffic_volume"], mode="lines",
                    name="Avg Volume",
                    line=dict(color=C_BLUE, width=3, shape="spline", smoothing=1.1),
                    fill="tozeroy", fillcolor="rgba(59,130,208,0.08)",
                    hovertemplate="Hour %{x}:00<br>Volume: %{y:,.0f}<extra></extra>",
                ))
                apply_flowcast_theme(fig, height=320)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Hour or traffic volume column not available.")

    with col2:
        with chart_card("Congestion Split", "Share of time in each congestion state"):
            if "congestion_level" in data.columns:
                cong = data["congestion_level"].value_counts().reset_index()
                cong.columns = ["Congestion", "Count"]
                fig = px.pie(cong, names="Congestion", values="Count", hole=0.64,
                             color="Congestion", color_discrete_map=CONGESTION_COLORS)
                apply_flowcast_theme(fig, height=320)
                fig.update_traces(textinfo="percent", hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>")
                fig.update_layout(showlegend=True, legend=dict(orientation="v", x=1, y=0.5))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Congestion data not available.")

    with chart_card("Speed Distribution", "Frequency of observed speeds across all segments"):
        if "avg_speed" in data.columns:
            fig = px.histogram(data, x="avg_speed", nbins=42, labels={"avg_speed":"Speed (km/h)"})
            fig.update_traces(marker_color=C_PURPLE, marker_opacity=0.75, name="Speed", showlegend=False)
            apply_flowcast_theme(fig, height=260)
            fig.update_layout(showlegend=False, bargap=0.04)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("Average speed column not available.")

    insights = []
    if has_columns(data, ["hour","traffic_volume"]):
        peak = data.groupby("hour")["traffic_volume"].mean().idxmax()
        insights.append(f"Peak traffic occurs around <b>{int(peak):02d}:00</b>")
    if "avg_speed" in data.columns:
        insights.append(f"Average network speed is <b>{avg_speed:.1f} km/h</b>")
    if "congestion_level" in data.columns:
        top_cong = data["congestion_level"].value_counts().idxmax()
        pct = data["congestion_level"].value_counts(normalize=True).max() * 100
        insights.append(f"<b>{top_cong}</b> is the most common state ({pct:.0f}% of observations)")
    if has_columns(data, ["road_id","traffic_volume"]):
        busiest = data.groupby("road_id")["traffic_volume"].mean().idxmax()
        insights.append(f"Highest average volume is on road <b>{busiest}</b>")
    if insights:
        insight_box("Network Insights", insights)


# ── Historical Trends ─────────────────────────────────────────
def render_historical_trends(data):
    roads = ["All"] + sorted(data["road_id"].dropna().unique().tolist()) if "road_id" in data.columns else ["All"]

    c1, c2, c3 = st.columns(3)
    with c1: selected_road = st.selectbox("Road", roads)
    with c2: metric_label  = st.selectbox("Metric", ["Traffic Volume","Average Speed","Travel Time","Occupancy"])
    with c3: agg           = st.selectbox("Aggregation", ["Raw","Hourly","Daily"])

    metric_col = {
        "Traffic Volume": "traffic_volume",
        "Average Speed":  "avg_speed",
        "Travel Time":    "travel_time",
        "Occupancy":      "occupancy",
    }.get(metric_label)

    if metric_col not in data.columns:
        empty_state("📊","Column Not Available", f"The '{metric_label}' column was not found in the dataset.")
        return
    if "timestamp" not in data.columns:
        empty_state("📊","Timestamp Missing","A date/time column is required to plot trends over time.")
        return

    subset = data.copy() if selected_road == "All" else data[data["road_id"] == selected_road].copy()
    subset = subset.dropna(subset=["timestamp"]).sort_values("timestamp")

    if agg == "Hourly":
        subset["_ts"] = subset["timestamp"].dt.floor("h")
        plot_data = subset.groupby("_ts")[metric_col].mean().reset_index().rename(columns={"_ts":"timestamp"})
    elif agg == "Daily":
        subset["_ts"] = subset["timestamp"].dt.date
        plot_data = subset.groupby("_ts")[metric_col].mean().reset_index().rename(columns={"_ts":"timestamp"})
    else:
        plot_data = subset[["timestamp", metric_col]].dropna()

    # ── Stat KPI cards (same style as other pages) ──────────
    if len(plot_data) > 1:
        s = plot_data[metric_col].dropna()
        if not s.empty:
            stat_kpi_cards([
                {"label": "Minimum", "value": f"{s.min():,.1f}",  "sub": metric_label, "color": C_BLUE},
                {"label": "Maximum", "value": f"{s.max():,.1f}",  "sub": metric_label, "color": C_RED},
                {"label": "Average", "value": f"{s.mean():,.1f}", "sub": metric_label, "color": C_AMBER},
                {"label": "Latest",  "value": f"{s.iloc[-1]:,.1f}","sub": metric_label,"color": C_GREEN},
            ])

    # ── Gap between the stat cards and the chart card below ──
    spacer(22)

    # ── Chart (full width) ───────────────────────────────────
    with chart_card(metric_label, f"{agg} aggregation · {selected_road}"):
        if plot_data.empty:
            st.info("No data available for the current selection.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_data["timestamp"], y=plot_data[metric_col], mode="lines",
                name=metric_label,
                line=dict(color=C_BLUE, width=2), fill="tozeroy", fillcolor="rgba(59,130,208,0.07)",
                hovertemplate="%{x}<br>" + metric_label + ": %{y:,.1f}<extra></extra>",
            ))
            apply_flowcast_theme(fig, height=380)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Period insight below the chart ───────────────────────
    if len(plot_data) > 1:
        first_half  = plot_data[metric_col].iloc[:len(plot_data)//2].mean()
        second_half = plot_data[metric_col].iloc[len(plot_data)//2:].mean()
        if pd.notna(first_half) and first_half:
            change    = (second_half - first_half) / first_half * 100
            direction = "increased" if change > 0 else "decreased"
            insight_box("Period Insight", [
                f"{metric_label} <b>{direction} by {abs(change):.1f}%</b> in the second half of the selected period."
            ])


# ── Congestion Heatmap ────────────────────────────────────────
def render_congestion_heatmap(data):
    if not has_columns(data, ["congestion_level","road_id"]):
        empty_state("🌡️","Data Unavailable","Congestion level or road identifier columns are missing from the dataset.")
        return

    all_roads      = sorted(data["road_id"].dropna().unique().tolist())
    selected_roads = st.multiselect("Filter Roads", all_roads, default=all_roads[:min(15, len(all_roads))])
    subset = data[data["road_id"].isin(selected_roads)] if selected_roads else data

    if "hour" not in subset.columns and "timestamp" in subset.columns:
        subset = subset.copy()
        subset["hour"] = subset["timestamp"].dt.hour

    if "hour" not in subset.columns:
        empty_state("🌡️","Hour Data Unavailable","No hour or timestamp column found to build the heatmap.")
        return

    subset = subset.copy()
    subset["severity"] = subset["congestion_level"].map(SEVERITY_MAP)

    heatmap_df = (
        subset.dropna(subset=["severity"])
        .groupby(["road_id","hour"])["severity"].mean()
        .reset_index().pivot(index="road_id", columns="hour", values="severity")
    )

    # ── Summary cards — proper inline-stat-cards (not chart_card shells) ──
    if not heatmap_df.empty:
        peak_hour  = int(heatmap_df.mean(axis=0).idxmax())
        worst_road = heatmap_df.mean(axis=1).idxmax()
        avg_sev    = float(np.nanmean(heatmap_df.values))
        sev_label  = ["Free-flow","Moderate","Heavy","Severe"][min(3, int(round(avg_sev)))]
        sev_color  = [C_GREEN, C_AMBER, C_ORANGE, C_RED][min(3, int(round(avg_sev)))]

        inline_stat_cards([
            {
                "label":  "Peak Hour",
                "sub":    "Highest average congestion",
                "value":  f"{peak_hour:02d}:00",
                "color":  C_AMBER,
            },
            {
                "label":  "Most Congested Road",
                "sub":    "Highest average severity",
                "value":  str(worst_road),
                "color":  C_RED,
            },
            {
                "label":  "Network Average",
                "sub":    "Mean severity across all roads",
                "value":  sev_label,
                "detail": f"Score: {avg_sev:.2f} / 3.0",
                "color":  sev_color,
            },
        ])

    spacer(24)
    
    # ── Heatmap ──────────────────────────────────────────────
    with chart_card("Congestion Severity Grid","Average severity score: 0 = Free-flow → 3 = Severe"):
        if heatmap_df.empty:
            st.info("No congestion data available for the current selection.")
        else:
            fig = px.imshow(
                heatmap_df, aspect="auto",
                color_continuous_scale=[[0,C_GREEN],[0.33,C_AMBER],[0.67,C_ORANGE],[1,C_RED]],
                zmin=0, zmax=3,
                labels={"x":"Hour of Day","y":"Road","color":"Severity"},
            )
            fig.update_traces(name="Severity", hovertemplate="Road %{y}<br>Hour %{x}:00<br>Severity: %{z:.2f}<extra></extra>")
            apply_flowcast_theme(fig, height=max(400, len(heatmap_df)*30))
            fig.update_layout(coloraxis_colorbar=dict(
                tickvals=[0,1,2,3], ticktext=["Free-flow","Moderate","Heavy","Severe"],
                thickness=12, len=0.55,
            ))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Road Comparison ───────────────────────────────────────────
def render_road_comparison(data):
    if not has_columns(data, ["road_id","traffic_volume"]):
        empty_state("🛣️","Data Unavailable","Road identifier or traffic volume columns are missing.")
        return

    agg_dict = {"traffic_volume":"mean"}
    for col in ["avg_speed","travel_time","occupancy"]:
        if col in data.columns:
            agg_dict[col] = "mean"

    summary = data.groupby("road_id").agg(agg_dict).reset_index()

    # Rank-by control
    ctrl_col, _ = st.columns([1, 2])
    with ctrl_col:
        metric_opts = [c for c in ["traffic_volume","avg_speed","travel_time","occupancy"] if c in summary.columns]
        metric = st.selectbox("Rank by", metric_opts, format_func=lambda c: c.replace("_"," ").title())

    sorted_sum = summary.sort_values(metric, ascending=True).reset_index(drop=True)
    norm = (sorted_sum[metric] - sorted_sum[metric].min()) / (sorted_sum[metric].max() - sorted_sum[metric].min() + 1e-9)

    # Bar chart
    with chart_card(f"Roads Ranked by {metric.replace('_',' ').title()}", "Higher values → right"):
        fig = go.Figure(go.Bar(
            x=sorted_sum[metric], y=sorted_sum["road_id"].astype(str), orientation="h",
            name=metric.replace("_"," ").title(),
            marker_color=[f"rgba(59,130,208,{0.3 + 0.7*v:.2f})" for v in norm],
            marker_line_width=0,
            hovertemplate="%{y}<br>" + metric.replace("_"," ").title() + ": %{x:,.2f}<extra></extra>",
        ))
        chart_h = min(480, max(300, len(sorted_sum) * 30))
        apply_flowcast_theme(fig, height=chart_h)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Three inline stat summary cards (properly sized)
    best_vol  = summary.loc[summary["traffic_volume"].idxmax(), "road_id"]
    least_vol = summary.loc[summary["traffic_volume"].idxmin(), "road_id"]
    fastest   = summary.loc[summary["avg_speed"].idxmax(), "road_id"] if "avg_speed" in summary.columns else "N/A"

    inline_stat_cards([
        {"label": "🚦 Busiest Road",    "sub": "Highest avg traffic volume", "value": str(best_vol),  "color": C_AMBER},
        {"label": "✅ Least Congested", "sub": "Lowest avg traffic volume",  "value": str(least_vol), "color": C_GREEN},
        {"label": "⚡ Fastest Road",    "sub": "Highest average speed",      "value": str(fastest),   "color": C_BLUE},
    ])

    spacer(24)
    
    # Full ranking table
    with chart_card("Segment Rankings","Ranked by the selected metric"):
        ranked = sorted_sum.sort_values(metric, ascending=False).reset_index(drop=True)
        ranked.insert(0, "Rank", [f"#{i+1}" for i in range(len(ranked))])
        ranked.columns = [c.replace("_"," ").title() for c in ranked.columns]
        st.dataframe(ranked, use_container_width=True, hide_index=True)


# ── Weather vs Traffic ────────────────────────────────────────
def render_weather_traffic(data):
    if "traffic_volume" not in data.columns:
        empty_state("🌦️","Data Unavailable","Traffic volume column not found in the dataset.")
        return

    sample = data.sample(min(5000, len(data)), random_state=42)
    weather_pairs = [
        ("rainfall",    "Rainfall (mm)",  "Rainfall vs Volume"),
        ("visibility",  "Visibility (m)", "Visibility vs Volume"),
        ("temperature", "Temp (°C)",      "Temperature vs Volume"),
    ]
    available = [(x,xl,t) for x,xl,t in weather_pairs if x in sample.columns]
    if not available:
        empty_state("🌦️","Weather Data Unavailable","Rainfall, visibility, or temperature columns were not found.")
        return

    color_kw = dict(color="congestion_level", color_discrete_map=CONGESTION_COLORS) if "congestion_level" in sample.columns else {}

    rows = [available[i:i+2] for i in range(0, len(available), 2)]
    for row in rows:
        cols = st.columns(len(row))
        for col, (x_col, x_label, title) in zip(cols, row):
            with col:
                with chart_card(title, "Each point is one observation window"):
                    fig = px.scatter(sample, x=x_col, y="traffic_volume", opacity=0.5,
                                     labels={x_col:x_label,"traffic_volume":"Traffic Volume"}, **color_kw)
                    apply_flowcast_theme(fig, height=320)
                    fig.update_traces(marker_size=4)
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    insights = []
    for x_col, _, label in available:
        corr_data = data[[x_col,"traffic_volume"]].dropna()
        if len(corr_data) > 1:
            corr      = corr_data.corr().iloc[0,1]
            strength  = "strong" if abs(corr)>0.6 else "moderate" if abs(corr)>0.3 else "weak"
            direction = "positive" if corr > 0 else "negative"
            insights.append(f"<b>{label.split()[0]}</b> shows a {strength} {direction} correlation with traffic volume (r = {corr:.2f})")
    if insights:
        insight_box("Correlation Analysis", insights)

    spacer(24)
    
    if "congestion_level" in data.columns and available:
        x_col, x_label, _ = available[0]
        with chart_card(f"{x_label} by Congestion Level","Box distribution per state"):
            fig = px.box(data, x="congestion_level", y=x_col, color="congestion_level",
                         color_discrete_map=CONGESTION_COLORS,
                         labels={"congestion_level":"Congestion", x_col:x_label})
            apply_flowcast_theme(fig, height=300)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Live Prediction ───────────────────────────────────────────
def render_live_prediction(data):
    roads = sorted(data["road_id"].dropna().unique()) if "road_id" in data.columns else []

    # ── ROW 1: Inputs + Prediction score + detail cards ──────
    col_ctrl, col_result = st.columns([1, 2], gap="large")

    with col_ctrl:
        with chart_card("Prediction Inputs", "Configure and run the forecast"):
            selected_road = st.selectbox("Road Segment", roads) if roads else None
            horizon       = st.slider("Forecast Horizon (windows)", 1, 12, 1)
            run_btn       = st.button("⚡  Run AI Prediction", use_container_width=True)

    if selected_road is not None:
        road_df = (
            data[data["road_id"] == selected_road].sort_values("timestamp")
            if "timestamp" in data.columns
            else data[data["road_id"] == selected_road]
        )
    else:
        road_df = data
    latest = road_df.tail(1)

    with col_result:
        if model_pkg is None:
            empty_state("🧠","Model Not Found","Train the FlowCast XGBoost model to enable live predictions.","python run_pipeline.py")
        else:
            with st.spinner("Running XGBoost prediction…"):
                pred = model_prediction_frame(latest)
            if pred is None:
                empty_state("🧠","Prediction Unavailable","The model could not generate a prediction for this selection.")
            else:
                pv = pred[0]
                st.markdown(f"""
                <div class="pred-hero">
                    <div class="pred-label">Predicted Next-Window Volume</div>
                    <div class="pred-value">{pv:,.0f}</div>
                    <div class="pred-unit">vehicles</div>
                </div>""", unsafe_allow_html=True)

                if len(latest) > 0:
                    row = latest.iloc[0]
                    level = str(row.get("congestion_level", "Unknown"))
                    stat_kpi_cards([
                        {
                            "label": "Current Volume",
                            "value": f"{row.get('traffic_volume', 0):,.0f}",
                            "sub":   "vehicles",
                            "color": C_BLUE,
                        },
                        {
                            "label": "Average Speed",
                            "value": f"{row.get('avg_speed', 0):.1f}",
                            "sub":   "km/h",
                            "color": C_PURPLE,
                        },
                        {
                            "label": "Congestion Level",
                            "value": level,
                            "sub":   "current status",
                            "color": CONGESTION_COLORS.get(level, C_BLUE),
                        },
                    ])

    # ── ROW 2: Explanation note ───────────────────────────────
    st.markdown(
        '<div class="insight-card" style="margin-top:4px;margin-bottom:4px">'
        '<div class="insight-title">Model Note</div>'
        '<div class="insight-item" style="border:none">'
        'The XGBoost model produces single-window forecasts. '
        'Multi-window horizons chain predictions iteratively — accuracy may decrease over longer horizons.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # ── Gap between Row 2 (Model Note) and Row 3 (Recent Trend) ──
    spacer(24)

    # ── ROW 3: Recent traffic trend ───────────────────────────
    if roads and "traffic_volume" in data.columns:
        road_for_trend = selected_road if selected_road else roads[0]
        trend_df = (
            data[data["road_id"] == road_for_trend].sort_values("timestamp").tail(100)
            if "timestamp" in data.columns
            else data.tail(100)
        )
        with chart_card("Recent Traffic Trend", f"Last {len(trend_df)} observations on {road_for_trend}"):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=trend_df["timestamp"] if "timestamp" in trend_df.columns else trend_df.index,
                y=trend_df["traffic_volume"], mode="lines",
                name="Traffic Volume",
                line=dict(color=C_BLUE, width=2), fill="tozeroy", fillcolor="rgba(59,130,208,0.07)",
                hovertemplate="%{y:,.0f} vehicles<extra></extra>",
            ))
            apply_flowcast_theme(fig, height=260)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Forecast Visualisation ────────────────────────────────────
def render_forecast_visualisation(data):
    if model_pkg is None:
        empty_state("🧠","No Trained Model","Run the training pipeline to enable forecast visualisations.","python run_pipeline.py")
        return

    roads = sorted(data["road_id"].dropna().unique()) if "road_id" in data.columns else []

    # ── Controls: road + data points in one row ───────────────
    c1, c2 = st.columns([1, 1])
    with c1:
        selected_road = st.selectbox("Road Segment", roads) if roads else None
    with c2:
        n_pts = st.slider("Data Points", 50, 500, 200)

    sub     = data[data["road_id"] == selected_road] if selected_road is not None else data
    ordered = sub.sort_values("timestamp") if "timestamp" in sub.columns else sub
    working = ordered.tail(n_pts).copy()
    preds   = model_prediction_frame(working)

    if preds is not None and "traffic_volume" in working.columns:
        working["predicted_volume"] = preds
        actual    = working["traffic_volume"]
        predicted = working["predicted_volume"]
        mae  = (actual - predicted).abs().mean()
        mape = ((actual - predicted).abs() / actual.replace(0, np.nan)).mean() * 100
        kpi_cards([
            {"label":"Mean Absolute Error","value":f"{mae:,.1f}",      "delta":"vehicles",       "icon":"📉","color":C_BLUE},
            {"label":"MAPE",               "value":f"{mape:.2f}%",     "delta":"avg % error",    "icon":"🎯","color":C_GREEN},
            {"label":"Data Points",        "value":str(len(working)),  "delta":str(selected_road),"icon":"📊","color":C_AMBER},
            {"label":"Model",              "value":"XGBoost",           "delta":"classical ML",   "icon":"🧠","color":C_PURPLE},
        ])

    with chart_card("Actual vs Predicted", f"Road {selected_road} · last {len(working)} points"):
        fig  = go.Figure()
        x_ax = working["timestamp"] if "timestamp" in working.columns else working.index
        if "traffic_volume" in working.columns:
            fig.add_trace(go.Scatter(x=x_ax, y=working["traffic_volume"], name="Actual",    line=dict(color=C_BLUE,   width=2)))
        if preds is not None:
            fig.add_trace(go.Scatter(x=x_ax, y=working["predicted_volume"], name="Predicted", line=dict(color=C_PURPLE, width=2, dash="dash")))
        apply_flowcast_theme(fig, height=380)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if preds is not None and "traffic_volume" in working.columns:
        residuals = actual - predicted
        with chart_card("Residual Distribution","Distribution of prediction errors on the selected segment"):
            fig2 = px.histogram(x=residuals, nbins=32, labels={"x":"Residual (Actual − Predicted)"})
            fig2.update_traces(marker_color=C_PURPLE, marker_opacity=0.75, name="Residual", showlegend=False)
            apply_flowcast_theme(fig2, height=240)
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        insight_box("Forecast Insights", [
            f"Prediction accuracy (1 − MAPE): <b>{max(0,100-mape):.1f}%</b>",
            f"Largest single-point error: <b>{residuals.abs().max():,.0f} vehicles</b>",
            f"Latest prediction error: <b>{residuals.iloc[-1]:+,.0f} vehicles</b>",
        ])


# ── Prediction Confidence ─────────────────────────────────────
def render_prediction_confidence(data):
    rpath = MODEL_DIR / "xgboost_volume_predictions.csv"
    if not rpath.exists():
        empty_state("📊","Residual Data Not Found","Train the model and run evaluation to generate prediction residual data.")
        return

    res_df = pd.read_csv(rpath)
    if "residual" not in res_df.columns:
        empty_state("📊","Column Missing","The residual column was not found in the predictions file.")
        return

    residuals    = res_df["residual"].dropna()
    lower, upper = residuals.quantile(0.05), residuals.quantile(0.95)
    width        = upper - lower

    kpi_cards([
        {"label":"Lower Bound (5th pct)", "value":f"{lower:+,.0f}", "delta":"residual shift",    "icon":"⬇️","color":C_PURPLE},
        {"label":"Upper Bound (95th pct)","value":f"{upper:+,.0f}", "delta":"residual shift",    "icon":"⬆️","color":C_BLUE},
        {"label":"Interval Width",         "value":f"{width:,.0f}",  "delta":"vehicles",          "icon":"↔️","color":C_AMBER},
        {"label":"Confidence Level",       "value":"90%",            "delta":"empirical interval","icon":"✅","color":C_GREEN},
    ])

    # ── Residual distribution: full-width row ─────────────────
    with chart_card("Residual Distribution","Distribution of prediction errors on the held-out test set"):
        fig = px.histogram(x=residuals, nbins=40, labels={"x":"Prediction Residual"})
        fig.update_traces(marker_color=C_BLUE, marker_opacity=0.7, name="Residual", showlegend=False)
        fig.add_vline(x=lower, line_color=C_AMBER, line_dash="dash", line_width=1.5)
        fig.add_vline(x=upper, line_color=C_RED,   line_dash="dash", line_width=1.5)
        apply_flowcast_theme(fig, height=300)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if "road_id" in data.columns:
        roads         = sorted(data["road_id"].dropna().unique())
        selected_road = st.selectbox("Visualise confidence band on road", roads)
        sub           = data[data["road_id"] == selected_road]
        ordered       = sub.sort_values("timestamp") if "timestamp" in sub.columns else sub
        working       = ordered.tail(100).copy()
        preds         = model_prediction_frame(working)
        if preds is not None:
            working["prediction"] = preds
            working["lower_b"]    = working["prediction"] + lower
            working["upper_b"]    = working["prediction"] + upper
            with chart_card(f"Confidence Band — {selected_road}","Shaded region shows the empirical 90% prediction interval"):
                x_ax = working["timestamp"] if "timestamp" in working.columns else working.index
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=x_ax, y=working["upper_b"], name="Upper Bound", line=dict(width=0), showlegend=False))
                fig2.add_trace(go.Scatter(x=x_ax, y=working["lower_b"], fill="tonexty",
                                          fillcolor="rgba(59,130,208,0.10)", line=dict(width=0), name="90% Interval"))
                fig2.add_trace(go.Scatter(x=x_ax, y=working["prediction"], name="Prediction", line=dict(color=C_BLUE, width=2)))
                apply_flowcast_theme(fig2, height=320)
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    insight_box("Confidence Interpretation", [
        "90% of predictions on held-out data fell within this residual interval.",
        f"The interval spans <b>{width:,.0f} vehicles</b> — wider intervals indicate higher forecast uncertainty.",
        "Use these bounds to build conservative and optimistic traffic plans.",
    ])


# ── Model Performance ─────────────────────────────────────────
def render_model_performance():
    scoreboards = [
        (MODEL_DIR / "regression_scoreboard.csv",                  "Regression Models"),
        (MODEL_DIR / "congestion_classification_scoreboard.csv",    "Congestion Classification"),
        (MODEL_DIR / "accident_risk_classification_scoreboard.csv", "Accident Risk Classification"),
        (DEEP_MODEL_DIR / "lstm_scoreboard.csv",                   "LSTM Deep Learning"),
    ]
    found = [(p,t) for p,t in scoreboards if p.exists()]
    if not found:
        empty_state("🧠","No Scoreboards Found","Run the training and evaluation pipeline to generate model performance files.","python run_pipeline.py")
        return

    tabs   = st.tabs([t for _,t in found])
    medals = ["🥇","🥈","🥉"]

    for tab, (path, title) in zip(tabs, found):
        with tab:
            tbl       = pd.read_csv(path)
            rank_col  = next((c for c in ["Model","model","name"] if c in tbl.columns), tbl.columns[0])
            metric_cols = [c for c in tbl.columns if c != rank_col]

            for i, (_, row) in enumerate(tbl.iterrows()):
                medal = medals[i] if i < 3 else f"#{i+1}"
                color = [C_AMBER,"#9CA3AF","#B08D57"][i] if i < 3 else "#6B7280"
                metrics_html = ""
                for mc in metric_cols[:4]:
                    val = row[mc]
                    try:
                        fmt = f"{float(val):.4f}" if "." in str(val) else str(val)
                    except (ValueError,TypeError):
                        fmt = str(val)
                    metrics_html += f'<div class="ldr-metric"><div class="ldr-metric-label">{mc}</div><div class="ldr-metric-value">{fmt}</div></div>'
                st.markdown(f"""
                <div class="leaderboard-row">
                    <span class="rank-badge" style="color:{color}">{medal}</span>
                    <span class="ldr-name">{row.get(rank_col,"—")}</span>
                    <div class="ldr-metrics">{metrics_html}</div>
                </div>""", unsafe_allow_html=True)

            if len(tbl) > 1 and metric_cols:
                bar_metric = metric_cols[0]
                with chart_card(f"{bar_metric} Comparison","Lower is better for error metrics; higher for accuracy/R²"):
                    fig = px.bar(tbl.sort_values(bar_metric, ascending=True), x=bar_metric, y=rank_col, orientation="h")
                    fig.update_traces(marker_color=C_BLUE, marker_opacity=0.85, marker_line_width=0, name=bar_metric, showlegend=False)
                    apply_flowcast_theme(fig, height=max(220, len(tbl)*44))
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with chart_card("Full Scoreboard"):
                st.dataframe(tbl, use_container_width=True, hide_index=True)


# ── Reports ───────────────────────────────────────────────────
def render_reports():
    report_path = REPORT_DIR / "data_quality.md"
    if report_path.exists():
        with chart_card("Data Quality Report"):
            report_text = report_path.read_text(encoding="utf-8")
            st.markdown(
                f'<div class="report-content-wrap">{report_text}</div>',
                unsafe_allow_html=True,
            )
    else:
        empty_state("📄","No Report Found","Run the pipeline to generate a data quality report.","python run_pipeline.py")

    fig_dir = REPORT_DIR / "figures"
    if fig_dir.exists():
        images = sorted(fig_dir.glob("*.png"))
        if images:
            spacer(10)
            st.markdown('<div class="chart-title" style="margin:10px 0">Generated Figures</div>', unsafe_allow_html=True)
            cols = st.columns(2)
            for i, img in enumerate(images):
                with cols[i % 2]:
                    with chart_card(img.stem.replace("_"," ").title()):
                        st.image(str(img), use_container_width=True)


# ============================================================
# 15. ROUTER
# ============================================================

NAV_ROUTE_BY_PAGE = {label: route_key for _, label, _, route_key in NAV_ITEMS}

RENDERERS = {
    "overview":   lambda: render_overview(df),
    "historical": lambda: render_historical_trends(df),
    "heatmap":    lambda: render_congestion_heatmap(df),
    "comparison": lambda: render_road_comparison(df),
    "weather":    lambda: render_weather_traffic(df),
    "live_pred":  lambda: render_live_prediction(df),
    "forecast":   lambda: render_forecast_visualisation(df),
    "confidence": lambda: render_prediction_confidence(df),
    "model_perf": render_model_performance,
    "reports":    render_reports,
}

RENDERERS.get(NAV_ROUTE_BY_PAGE.get(active_page, "overview"), lambda: None)()


# ============================================================
# 16. FOOTER
# ============================================================

st.markdown("""
<div style="
    border-top: 1px solid #2A2D31;
    margin-top: 35px; padding-top: 18px; padding-bottom: 10px;
    text-align: center; color: #737982; font-size: 13px;
">
    FlowCast Traffic Intelligence System &nbsp;•&nbsp; Data-driven traffic analytics
</div>
""", unsafe_allow_html=True)