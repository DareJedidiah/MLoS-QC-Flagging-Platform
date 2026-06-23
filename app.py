"""
MLoS QA/QC Platform — Single-file Streamlit Application
Version 1.1 | June 2026
"""

import io
import os
import warnings
import zipfile
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from rapidfuzz import fuzz

# reportlab — PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

warnings.filterwarnings("ignore")

# ── Session state key ──────────────────────────────────────────────────────────
_SS = "mlos_qaqc_results"  # single dict key holding all post-run state

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MLoS QA/QC Platform",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLUE_DARK   = "#0F2D5C"
BLUE_MID    = "#1D4ED8"
BLUE_LIGHT  = "#3B82F6"
BLUE_PALE   = "#DBEAFE"
BLUE_FAINT  = "#EFF6FF"
WHITE       = "#FFFFFF"
GREY_LIGHT  = "#F1F5F9"
RED_FLAG    = "#DC2626"
AMBER_FLAG  = "#D97706"
GREEN_OK    = "#16A34A"
PURE_BLACK  = "#000000"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Open+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Open Sans', sans-serif;
  }}

  /* Main content area: dark text on light backgrounds */
  .main .block-container p,
  .main .block-container div,
  .main .block-container span,
  .main .block-container label {{
    font-family: 'Open Sans', sans-serif;
    color: {PURE_BLACK};
  }}

  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background: {BLUE_DARK} !important;
  }}
  section[data-testid="stSidebar"] * {{
    color: {WHITE} !important;
  }}
  section[data-testid="stSidebar"] .stFileUploader label,
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stMultiselect label {{
    color: {BLUE_PALE} !important;
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  section[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.15);
  }}

  /* Top header bar */
  .app-header {{
    background: linear-gradient(135deg, {BLUE_DARK} 0%, {BLUE_MID} 100%);
    padding: 1.5rem 2rem 1.2rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }}
  .app-header h1 {{
    color: {WHITE} !important;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
  }}
  .app-header p {{
    color: {BLUE_PALE} !important;
    margin: 0.2rem 0 0 0;
    font-size: 0.88rem;
    font-weight: 400;
  }}
  .header-icon {{
    font-size: 2.4rem;
  }}

  /* Metric cards */
  .metric-card {{
    background: {WHITE};
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .metric-card .label {{
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {PURE_BLACK} !important;
    margin-bottom: 0.3rem;
  }}
  .metric-card .value {{
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1.1;
  }}
  .metric-card .sublabel {{
    font-size: 0.72rem;
    color: {PURE_BLACK} !important;
    margin-top: 0.2rem;
  }}
  .val-blue   {{ color: {BLUE_MID} !important; }}
  .val-red    {{ color: {RED_FLAG} !important; }}
  .val-amber  {{ color: {AMBER_FLAG} !important; }}
  .val-green  {{ color: {GREEN_OK} !important; }}

  /* Section headers */
  .section-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: {BLUE_DARK} !important;
    border-left: 4px solid {BLUE_MID};
    padding-left: 0.75rem;
    margin: 1.5rem 0 0.8rem 0;
    background: {BLUE_FAINT};
    border-radius: 0 6px 6px 0;
    padding: 0.45rem 0.75rem;
  }}

  /* Info box */
  .info-box {{
    background: {BLUE_FAINT};
    border: 1px solid {BLUE_PALE};
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.84rem;
    color: {PURE_BLACK} !important;
    margin-bottom: 1rem;
  }}
  .info-box *, .info-box p, .info-box span, .info-box div, .info-box strong {{
    color: {PURE_BLACK} !important;
  }}

  /* ── Light-background containers: always dark text ───────────────────── */
  /* Tabs content area */
  .stTabs [data-baseweb="tab-panel"] p,
  .stTabs [data-baseweb="tab-panel"] div,
  .stTabs [data-baseweb="tab-panel"] span,
  .stTabs [data-baseweb="tab-panel"] li,
  .stTabs [data-baseweb="tab-panel"] td,
  .stTabs [data-baseweb="tab-panel"] th,
  .stTabs [data-baseweb="tab-panel"] label {{
    color: {PURE_BLACK} !important;
    font-family: 'Open Sans', sans-serif;
  }}
  /* Metric cards */
  .metric-card,
  .metric-card .label,
  .metric-card .sublabel {{
    color: {PURE_BLACK} !important;
  }}
  /* Dropped box */
  .dropped-box *,
  .dropped-box p,
  .dropped-box span,
  .dropped-box div {{
    color: #7F1D1D !important;
  }}
  /* Dataframe / table cells */
  [data-testid="stDataFrame"] td,
  [data-testid="stDataFrame"] th,
  .stDataFrame td,
  .stDataFrame th {{
    color: {PURE_BLACK} !important;
  }}
  /* st.success / st.error / st.warning native elements */
  [data-testid="stAlert"] p,
  [data-testid="stAlert"] div,
  [data-testid="stAlert"] span {{
    color: {PURE_BLACK} !important;
  }}

  /* Dropped states panel */
  .dropped-box {{
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.84rem;
    color: #7F1D1D !important;
  }}

  /* Download button */
  .stDownloadButton > button {{
    background: {BLUE_MID} !important;
    color: {WHITE} !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
  }}
  .stDownloadButton > button:hover {{
    background: {BLUE_DARK} !important;
  }}

  /* Dataframe modifications to support high visibility black font */
  .stDataFrame [data-testid="stTable"] {{
    color: {PURE_BLACK} !important;
  }}

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 2px solid {BLUE_PALE}; }}
  .stTabs [data-baseweb="tab"] {{
    background: rgba(219,234,254,0.3);
    border-radius: 6px 6px 0 0;
    font-weight: 500;
    color: {BLUE_DARK} !important;
    padding: 0.5rem 1rem;
    font-size: 0.85rem;
  }}
  .stTabs [data-baseweb="tab"] p,
  .stTabs [data-baseweb="tab"] span {{
    color: {BLUE_DARK} !important;
  }}
  .stTabs [aria-selected="true"] {{
    background: {BLUE_MID} !important;
    color: {WHITE} !important;
    font-weight: 700 !important;
  }}
  .stTabs [aria-selected="true"] p,
  .stTabs [aria-selected="true"] span {{
    color: {WHITE} !important;
  }}
  /* Tab panel: white background, dark text */
  .stTabs [data-baseweb="tab-panel"] {{
    background: {WHITE};
    border-radius: 0 0 8px 8px;
    padding: 1rem 0.5rem;
  }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_COLUMNS = [
    "sn","unique_code","state_name","lga_name","ward_name","take_off_point",
    "settlement_name","primary_settlement_name","alternate_name","latitude",
    "longitude","security_compromised","accessibility_status",
    "reasons_for_inaccessibility","habitational_status","set_population",
    "set_target","number_of_household","noncompliant_household","team_code",
    "day_of_activity","urban","rural","scattered","highrisk","slums",
    "densely_populated","hard2reach","border","nomadic","riverine","fulani",
    "source","comments","globalid","validation_status","gis_feedback",
    "master.id","mlos_id","eha_guid"
]

TRIM_COLS = [
    "state_name","lga_name","ward_name","settlement_name",
    "primary_settlement_name","alternate_name","security_compromised",
    "accessibility_status","reasons_for_inaccessibility","habitational_status",
    "urban","rural","scattered","highrisk","slums","densely_populated",
    "hard2reach","border","nomadic","riverine","fulani","source","comments",
    "validation_status"
]

VALID_DAY_CODES = {
    "1","2","3","4",
    "1_2","1_3","1_4","2_3","2_4","3_4",
    "1_2_3","1_2_4","1_3_4","2_3_4","1_2_3_4","NA"
}

OUTPUT_FLAG_COLUMNS = [
    "Critical Error Flag","Spatial Issues","Attribute Issues","No Issues",
    "Confirm with Programs team", # Added row length verification check flag
    "attribute duplicate","Stacked points","Wrong Entry Source",
    "Population Conflict","Duplicate eha_guid","Settlement Type Conflict",
    "Non-Numerical Number of Household","Wrong Entry Day of Activity Entry","Security and Accessibility conflict",
    "Wrong Entry SecurityComp","Wrong Entry Accesibility_status",
    "Wrong Entry reasons_for_inaccessibility","Wrong Entry Habitational_status",
    "Wrong Entry Urban","Wrong Entry Rural","Wrong Entry Scattered",
    "Wrong Entry Highrisk","Wrong Entry Border","Wrong Entry Densely_populated",
    "Wrong Entry Hard2reach","Wrong Entry Nomadic","Wrong Entry Riverine",
    "Wrong Entry Fulani","Wrong Entry Validation status","No Statename",
    "No LGAname","Wrong Entry Wardname","No Settlement name",
    "Settlement name with 1Chars","Settlement name with 2Chars",
    "Settlement name with more than 50Chars","Wrong Entry Population",
    "Wrong Entry TargetPop","Non-Numerical Latitude","Non-Numerical Longitude",
    "Non-Numerical Target Population","Non-Numerical Sett Population",
    "Non-Numerical NonCompliant Household","Non-Numerical Team Code",
    "No Geocoordinates","Not Intersecting Settlement Extent",
    "10m proximity","20m proximity","30m proximity",
    "Outside State Boundary","Outside LGA","Outside Ward"
]

CRITICAL_ERROR_FLAGS = [
    "attribute duplicate","Stacked points","Wrong Entry Source",
    "Population Conflict","Duplicate eha_guid","Settlement Type Conflict",
    "Non-Numerical Number of Household","Wrong Entry Day of Activity Entry",
    "Wrong Entry Urban","Wrong Entry Rural","Wrong Entry Scattered",
    "Security and Accessibility conflict"
]

SPATIAL_FLAGS = [
    "No Geocoordinates","Stacked points",
    "10m proximity","20m proximity","30m proximity",
    "Outside State Boundary","Outside LGA","Outside Ward"
]

ATTRIBUTE_FLAGS = [
    "attribute duplicate","Wrong Entry SecurityComp","Wrong Entry Accesibility_status",
    "Wrong Entry reasons_for_inaccessibility","Wrong Entry Habitational_status",
    "Wrong Entry Urban","Wrong Entry Rural","Wrong Entry Scattered",
    "Settlement Type Conflict","Wrong Entry Source","Wrong Entry Highrisk",
    "Wrong Entry Border","Wrong Entry Densely_populated","Wrong Entry Hard2reach",
    "Wrong Entry Nomadic","Wrong Entry Riverine","Wrong Entry Fulani",
    "Duplicate eha_guid","Wrong Entry Validation status","No Statename",
    "No LGAname","Wrong Entry Wardname","No Settlement name",
    "Settlement name with 1Chars","Wrong Entry Population",
    "Wrong Entry TargetPop","Population Conflict","Non-Numerical Latitude",
    "Non-Numerical Longitude","Non-Numerical Target Population",
    "Non-Numerical Sett Population","Non-Numerical Number of Household",
    "Non-Numerical NonCompliant Household",
    "Wrong Entry Day of Activity Entry","Security and Accessibility conflict"
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def is_numeric_series(s):
    def _chk(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return False
        try:
            float(str(v).strip())
            return True
        except (ValueError, TypeError):
            return False
    return s.apply(_chk)

def is_blank(v):
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return str(v).strip() == ""

def blank_series(s):
    return s.apply(is_blank)

def flag_col(df, col_name, mask):
    df[col_name] = np.where(mask, "Y", "")
    return df

def safe_str(v):
    if is_blank(v):
        return ""
    return str(v).strip()

def name_similarity(a, b, threshold=95):
    if is_blank(a) or is_blank(b):
        return False
    sa, sb = safe_str(a).upper(), safe_str(b).upper()
    if sa == sb:
        return True
    score = fuzz.token_set_ratio(sa, sb)
    return score >= threshold

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_schema(df, filename):
    cols = [c.strip() for c in df.columns.tolist()]
    if len(cols) < len(EXPECTED_COLUMNS):
        return False, f"Too few columns ({len(cols)} vs {len(EXPECTED_COLUMNS)} expected)"
    actual = cols[:len(EXPECTED_COLUMNS)]
    mismatches = []
    for i, (exp, got) in enumerate(zip(EXPECTED_COLUMNS, actual)):
        if exp.strip().lower() != got.strip().lower():
            mismatches.append(f"Col {i+1}: expected '{exp}', got '{got}'")
    if mismatches:
        return False, "Column mismatch: " + "; ".join(mismatches[:3]) + ("..." if len(mismatches) > 3 else "")
    return True, "OK"

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(df):
    df.columns = [c.strip() for c in df.columns]

    if "unique_code" in df.columns:
        df = df.drop(columns=["unique_code"])

    for col in TRIM_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda v: "" if is_blank(v) else str(v).strip())

    # Proper Case + Trim on name columns before rebuilding unique_code
    PROPER_CASE_COLS = ["state_name", "lga_name", "ward_name", "settlement_name"]
    for col in PROPER_CASE_COLS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: "" if is_blank(v) else str(v).strip().title()
            )

    def _mk_code(row):
        parts = [
            safe_str(row.get("state_name","")),
            safe_str(row.get("lga_name","")),
            safe_str(row.get("ward_name","")),
            safe_str(row.get("settlement_name",""))
        ]
        return "_".join(parts)
    df.insert(1, "unique_code", df.apply(_mk_code, axis=1))

    for coord in ["latitude","longitude"]:
        if coord in df.columns:
            def _conv(v):
                if is_blank(v):
                    return np.nan
                try:
                    fv = float(str(v).strip())
                    return np.nan if fv == 0 else fv
                except (ValueError, TypeError):
                    return v
            df[coord] = df[coord].apply(_conv)

    return df

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — ATTRIBUTE QA/QC
# ─────────────────────────────────────────────────────────────────────────────

def run_attribute_checks(df, progress_cb=None):
    n = df.shape[0]

    def _progress(msg, pct):
        if progress_cb:
            progress_cb(msg, pct)

    # ── Pre-processing: UPPER on Y/N flag columns ─────────────────────────────
    UPPER_COLS = [
        "security_compromised","urban","rural","scattered",
        "highrisk","slums","densely_populated","hard2reach",
        "border","nomadic","riverine","fulani"
    ]
    for col in UPPER_COLS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: "" if is_blank(v) else str(v).strip().upper()
            )

    # ── Pre-processing: TRIM + PROPER on status columns ───────────────────────
    PROPER_STATUS_COLS = ["validation_status", "accessibility_status", "habitational_status"]
    for col in PROPER_STATUS_COLS:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: "" if is_blank(v) else str(v).strip().title()
            )
    # ─────────────────────────────────────────────────────────────────────────

    _progress("Checking duplicates…", 2)

    dup_key = ["state_name","lga_name","ward_name","settlement_name"]
    mask_dup = df.duplicated(subset=dup_key, keep=False)
    df = flag_col(df, "attribute duplicate", mask_dup)

    _progress("Checking security_compromised…", 5)

    # Pre-compute Validated Unknown mask for use in exemption logic (Rule 1 & 2)
    if "validation_status" in df.columns:
        is_validated_unknown = df["validation_status"].apply(
            lambda v: safe_str(v) == "Validated Unknown"
        )
    else:
        is_validated_unknown = pd.Series([False] * n, index=df.index)

    valid_sc = {"Y","N","NA"}
    sc_fixed, sc_wrong = zip(*df.apply(lambda r: _fix_sc(r.get("security_compromised",""), r.get("accessibility_status","")), axis=1))
    df["security_compromised"] = list(sc_fixed)
    # Rule 1: suppress null/blank flag on security_compromised for Validated Unknown
    sc_wrong_series = pd.Series(sc_wrong, index=df.index)
    # _fix_sc flags wrong entries (non-blank invalid values) — those still flag;
    # but blank/null entries that resolve to a default should not flag for Validated Unknown.
    # _fix_sc already returns False for blank inputs, so no additional suppression needed here.
    df = flag_col(df, "Wrong Entry SecurityComp", sc_wrong_series)

    _progress("Checking accessibility_status…", 8)

    valid_acc = {"Fully Accessible","Inaccessible","Partially Accessible"}
    # Rule 1: For Validated Unknown, do not flag blank/null accessibility_status
    mask_acc = df["accessibility_status"].apply(lambda v: safe_str(v) not in valid_acc)
    mask_acc = mask_acc & ~(is_validated_unknown & df["accessibility_status"].apply(is_blank))
    df = flag_col(df, "Wrong Entry Accesibility_status", mask_acc)

    _progress("Checking reasons_for_inaccessibility…", 11)

    # Rule 1: For Validated Unknown, do not flag blank/null reasons_for_inaccessibility
    mask_rfi = df.apply(
        lambda r: safe_str(r.get("accessibility_status","")) != "Fully Accessible" and is_blank(r.get("reasons_for_inaccessibility","")),
        axis=1
    )
    mask_rfi = mask_rfi & ~(is_validated_unknown & df["reasons_for_inaccessibility"].apply(is_blank))
    df = flag_col(df, "Wrong Entry reasons_for_inaccessibility", mask_rfi)

    _progress("Checking security & accessibility conflict…", 12)

    # Flag: security_compromised = 'Y' AND accessibility_status = 'Fully Accessible'
    # AND reasons_for_inaccessibility is blank/null/empty
    mask_sac = df.apply(
        lambda r: (
            safe_str(r.get("security_compromised","")) == "Y"
            and safe_str(r.get("accessibility_status","")) == "Fully Accessible"
            and is_blank(r.get("reasons_for_inaccessibility",""))
        ),
        axis=1
    )
    df = flag_col(df, "Security and Accessibility conflict", mask_sac)

    _progress("Checking habitational_status…", 13)

    valid_hab = {"Inhabited","Partially Inhabited","Abandoned","Migrated"}
    # Rule 1: For Validated Unknown, do not flag blank/null habitational_status
    mask_hab = df["habitational_status"].apply(lambda v: safe_str(v) not in valid_hab)
    mask_hab = mask_hab & ~(is_validated_unknown & df["habitational_status"].apply(is_blank))
    df = flag_col(df, "Wrong Entry Habitational_status", mask_hab)

    _progress("Checking urban/rural/scattered…", 15)

    for col_raw, flag_name in [("urban","Wrong Entry Urban"),("rural","Wrong Entry Rural"),("scattered","Wrong Entry Scattered")]:
        col = col_raw if col_raw in df.columns else None
        if col:
            mask = df[col].apply(lambda v: safe_str(v) not in {"Y","N"})
            df = flag_col(df, flag_name, mask)
        else:
            df[flag_name] = ""

    def _stc(row):
        u = safe_str(row.get("urban",""))
        r = safe_str(row.get("rural",""))
        s = safe_str(row.get("scattered",""))
        if u == r == s == "Y":
            return True
        if u == r == s == "N":
            return True
        return False
    df = flag_col(df, "Settlement Type Conflict", df.apply(_stc, axis=1))

    _progress("Checking source…", 18)

    mask_src = df["source"].apply(lambda v: is_blank(v)) if "source" in df.columns else pd.Series([False]*n)
    df = flag_col(df, "Wrong Entry Source", mask_src)

    _progress("Checking Y/N/NA columns…", 22)

    yna_cols = [
        ("highrisk","Wrong Entry Highrisk"),
        ("slums","Wrong Entry Border"),
        ("densely_populated","Wrong Entry Densely_populated"),
        ("hard2reach","Wrong Entry Hard2reach"),
        ("border","Wrong Entry Border"),
        ("nomadic","Wrong Entry Nomadic"),
        ("riverine","Wrong Entry Riverine"),
        ("fulani","Wrong Entry Fulani"),
    ]
    valid_yna = {"Y","N","NA"}
    for src_col, flag_name in yna_cols:
        if src_col in df.columns:
            def _fix_yna(v, _src=src_col):
                sv = safe_str(v)
                if sv in valid_yna:
                    return sv, False
                if is_blank(v):
                    return "NA", False
                return sv, True
            fixed, wrong = zip(*df[src_col].apply(_fix_yna))
            df[src_col] = list(fixed)
            if flag_name not in df.columns:
                df = flag_col(df, flag_name, pd.Series(wrong))
            else:
                df[flag_name] = np.where(pd.Series(wrong) | (df[flag_name]=="Y"), "Y", df[flag_name])

    _progress("Checking eha_guid duplicates…", 26)

    if "eha_guid" in df.columns:
        non_blank = df["eha_guid"].apply(lambda v: not is_blank(v))
        dup_guid = df[non_blank].duplicated(subset=["eha_guid"], keep=False)
        full_mask = pd.Series([False]*n, index=df.index)
        full_mask[dup_guid[dup_guid].index] = True
        df = flag_col(df, "Duplicate eha_guid", full_mask)
    else:
        df["Duplicate eha_guid"] = ""

    _progress("Checking validation_status…", 29)

    valid_vs = {"Validated","Validation Ongoing","Not Validated","Validated Unknown"}
    mask_vs = df["validation_status"].apply(lambda v: safe_str(v) not in valid_vs) if "validation_status" in df.columns else pd.Series([False]*n)
    df = flag_col(df, "Wrong Entry Validation status", mask_vs)

    _progress("Checking name fields…", 32)

    for col_name, flag_name in [
        ("state_name","No Statename"),
        ("lga_name","No LGAname"),
        ("ward_name","Wrong Entry Wardname"),
        ("settlement_name","No Settlement name"),
    ]:
        mask = blank_series(df[col_name]) if col_name in df.columns else pd.Series([True]*n)
        df = flag_col(df, flag_name, mask)

    _progress("Checking settlement name lengths…", 35)

    if "settlement_name" in df.columns:
        slen = df["settlement_name"].apply(lambda v: len(safe_str(v)))
        df = flag_col(df, "Settlement name with 1Chars", slen == 1)
        df = flag_col(df, "Settlement name with 2Chars", slen == 2)
        df = flag_col(df, "Settlement name with more than 50Chars", slen > 50)
    else:
        for f in ["Settlement name with 1Chars","Settlement name with 2Chars","Settlement name with more than 50Chars"]:
            df[f] = ""

    # Requirement 1: Create Program team verification column based on length criteria
    prog_mask = (df["Settlement name with 1Chars"] == "Y") | \
                (df["Settlement name with 2Chars"] == "Y") | \
                (df["Settlement name with more than 50Chars"] == "Y")
    df = flag_col(df, "Confirm with Programs team", prog_mask)

    _progress("Checking population fields…", 38)

    # Rule 2: Compute exemption mask — Validated Unknown OR habitational_status is Abandoned/Migrated
    if "habitational_status" in df.columns:
        is_abandoned_or_migrated = df["habitational_status"].apply(
            lambda v: safe_str(v) in {"Abandoned", "Migrated"}
        )
    else:
        is_abandoned_or_migrated = pd.Series([False] * n, index=df.index)

    exempt_pop_activity = is_validated_unknown | is_abandoned_or_migrated

    def _bad_pop(v):
        if is_blank(v):
            return True
        try:
            fv = float(str(v).strip())
            return fv <= 0
        except (ValueError, TypeError):
            return True

    pop_mask = df["set_population"].apply(_bad_pop) if "set_population" in df.columns else pd.Series([False]*n)
    tgt_mask = df["set_target"].apply(_bad_pop) if "set_target" in df.columns else pd.Series([False]*n)
    # Rule 2: suppress for exempt rows
    df = flag_col(df, "Wrong Entry Population", pop_mask & ~exempt_pop_activity)
    df = flag_col(df, "Wrong Entry TargetPop",  tgt_mask & ~exempt_pop_activity)

    def _pop_conflict(row):
        sp = row.get("set_population","")
        st = row.get("set_target","")
        if is_blank(sp) or is_blank(st):
            return False
        try:
            spf, stf = float(str(sp).strip()), float(str(st).strip())
            if spf == 0 or stf == 0:
                return False
            return stf >= spf
        except (ValueError, TypeError):
            return False
    df = flag_col(df, "Population Conflict", df.apply(_pop_conflict, axis=1))

    _progress("Checking numerical types…", 43)

    # Checks NOT subject to Rule 2 (lat/lon/set_target/set_population handled above)
    non_exempt_numeric_checks = [
        ("latitude","Non-Numerical Latitude"),
        ("longitude","Non-Numerical Longitude"),
        ("set_target","Non-Numerical Target Population"),
        ("set_population","Non-Numerical Sett Population"),
    ]
    for src_col, flag_name in non_exempt_numeric_checks:
        if src_col in df.columns:
            mask = df[src_col].apply(lambda v: not is_blank(v) and not _is_numeric_val(v))
            df = flag_col(df, flag_name, mask)
        else:
            df[flag_name] = ""

    # Rule 2 applies: number_of_household — blank/null flagged unless exempt;
    # zero is a valid entry and must NOT be flagged.
    if "number_of_household" in df.columns:
        # Flag non-numeric text entries (never exempt — wrong type is always wrong)
        hh_wrong_type = df["number_of_household"].apply(
            lambda v: not is_blank(v) and not _is_numeric_val(v)
        )
        # Flag blank/null entries only (zero is valid — excluded from flagging)
        hh_blank_only = df["number_of_household"].apply(is_blank)
        df = flag_col(df, "Non-Numerical Number of Household",
                      hh_wrong_type | (hh_blank_only & ~exempt_pop_activity))
    else:
        df["Non-Numerical Number of Household"] = ""

    # noncompliant_household and team_code: zero is a valid entry — do NOT flag zero;
    # only flag blank/null (subject to Rule 2 exemption) or non-numeric text (always).
    for src_col, flag_name in [
        ("noncompliant_household", "Non-Numerical NonCompliant Household"),
        ("team_code",              "Non-Numerical Team Code"),
    ]:
        if src_col in df.columns:
            wrong_type_mask = df[src_col].apply(
                lambda v: not is_blank(v) and not _is_numeric_val(v)
            )
            blank_only_mask = df[src_col].apply(is_blank)
            # zero is valid — excluded from flagging entirely
            combined_mask = wrong_type_mask | (blank_only_mask & ~exempt_pop_activity)
            df = flag_col(df, flag_name, combined_mask)
        else:
            df[flag_name] = ""

    _progress("Checking day_of_activity…", 46)

    if "day_of_activity" in df.columns:
        def _fix_day(v):
            sv = safe_str(v)
            if sv in VALID_DAY_CODES:
                return sv, False
            if is_blank(v):
                return "NA", False
            return sv, True
        day_fixed, day_wrong = zip(*df["day_of_activity"].apply(_fix_day))
        df["day_of_activity"] = list(day_fixed)
        # Rule 2: suppress day_of_activity flag for Validated Unknown or Abandoned/Migrated
        day_wrong_series = pd.Series(day_wrong, index=df.index) & ~exempt_pop_activity
        df = flag_col(df, "Wrong Entry Day of Activity Entry", day_wrong_series)
    else:
        df["Wrong Entry Day of Activity Entry"] = ""

    # ── Validated Unknown override (Urban/Rural/Scattered/SettlementType) ─────
    # Rule 1 & 2 exemptions are applied inline above where each flag is computed.
    # This block clears the remaining flags that are not applicable for
    # Validated Unknown (Urban, Rural, Scattered, Settlement Type Conflict).
    VU_EXEMPT_FLAGS = [
        "Settlement Type Conflict",
        "Wrong Entry Urban",
        "Wrong Entry Rural",
        "Wrong Entry Scattered",
    ]
    for flag_name in VU_EXEMPT_FLAGS:
        if flag_name in df.columns:
            df.loc[is_validated_unknown, flag_name] = ""
    # ──────────────────────────────────────────────────────────────────────────

    return df

def _fix_sc(row_sc, row_ac):
    valid_sc = {"Y","N","NA"}
    v = safe_str(row_sc)
    if v in valid_sc:
        return v, False
    if is_blank(row_sc):
        if safe_str(row_ac) == "Fully Accessible":
            return "N", False
        else:
            return "NA", False
    return v, True

def _is_numeric_val(v):
    try:
        float(str(v).strip())
        return True
    except (ValueError, TypeError):
        return False

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — SPATIAL QA/QC
# ─────────────────────────────────────────────────────────────────────────────

def run_spatial_checks(df, ward_gdf=None, settlement_gdf=None, progress_cb=None):
    def _progress(msg, pct):
        if progress_cb:
            progress_cb(msg, pct)

    _progress("Checking geocoordinates…", 50)

    def _no_geo(row):
        lat, lon = row.get("latitude", np.nan), row.get("longitude", np.nan)
        return is_blank(lat) or is_blank(lon) or (isinstance(lat, float) and np.isnan(lat)) or (isinstance(lon, float) and np.isnan(lon))
    df = flag_col(df, "No Geocoordinates", df.apply(_no_geo, axis=1))

    # Set latitude and longitude to 0 for records with no geocoordinates
    no_geo_mask = df["No Geocoordinates"] == "Y"
    if no_geo_mask.any():
        df.loc[no_geo_mask, "latitude"]  = 0
        df.loc[no_geo_mask, "longitude"] = 0

    _progress("Checking stacked points…", 53)

    has_geo = df["No Geocoordinates"] != "Y"
    latlon_concat = df["latitude"].astype(str).str.strip() + "_" + df["longitude"].astype(str).str.strip()
    stacked_mask = has_geo & latlon_concat.duplicated(keep=False)
    df = flag_col(df, "Stacked points", stacked_mask)

    spatial_cols_default = [
        "Not Intersecting Settlement Extent",
        "10m proximity","20m proximity","30m proximity",
        "Outside State Boundary","Outside LGA","Outside Ward"
    ]
    for c in spatial_cols_default:
        if c not in df.columns:
            df[c] = ""

    try:
        import geopandas as gpd
        from shapely.geometry import Point

        geo_df = df[has_geo].copy()
        if len(geo_df) == 0:
            return df

        def _safe_float(v):
            try:
                return float(str(v).strip())
            except (ValueError, TypeError):
                return np.nan

        geo_df["_lat"] = geo_df["latitude"].apply(_safe_float)
        geo_df["_lon"] = geo_df["longitude"].apply(_safe_float)
        valid_coords = geo_df["_lat"].notna() & geo_df["_lon"].notna()
        geo_df = geo_df[valid_coords]

        if len(geo_df) == 0:
            return df

        geo_df["geometry"] = geo_df.apply(lambda r: Point(r["_lon"], r["_lat"]), axis=1)
        gdf = gpd.GeoDataFrame(geo_df, geometry="geometry", crs="EPSG:4326")

        _progress("Checking settlement extent intersection…", 57)

        if settlement_gdf is not None:
            settlement_gdf = settlement_gdf.to_crs("EPSG:4326")
            joined = gpd.sjoin(gdf, settlement_gdf, how="left", predicate="within")
            not_in_sett = joined.groupby(joined.index).first()["index_right"].isna()
            df.loc[not_in_sett[not_in_sett].index, "Not Intersecting Settlement Extent"] = "Y"

        _progress("Checking proximity (10m/20m/30m)…", 62)

        for meters, col_name in [(10,"10m proximity"),(20,"20m proximity"),(30,"30m proximity")]:
            buf_deg = meters / 111320.0
            buffers = gdf.geometry.buffer(buf_deg)
            buf_gdf = gpd.GeoDataFrame(geometry=buffers, index=gdf.index, crs="EPSG:4326")
            joined_prox = gpd.sjoin(buf_gdf, buf_gdf, how="left", predicate="intersects")
            overlapping = joined_prox[joined_prox.index != joined_prox["index_right"]].index.unique()
            df.loc[overlapping, col_name] = "Y"

        _progress("Checking ward boundary intersection…", 70)

        if ward_gdf is not None:
            ward_gdf = ward_gdf.to_crs("EPSG:4326")
            wdf_cols = {c.lower().strip(): c for c in ward_gdf.columns}

            def _find_col(candidates, col_map):
                for c in candidates:
                    if c in col_map:
                        return col_map[c]
                return None

            state_col_w = _find_col(["statename","state_name","state","adm1name"], wdf_cols)
            lga_col_w   = _find_col(["lganame","lga_name","lga","adm2name","lga_name_x","lganame_1"], wdf_cols)
            ward_col_w  = _find_col(["wardname","ward_name","ward","adm3name"], wdf_cols)

            joined_w = gpd.sjoin(gdf, ward_gdf, how="left", predicate="within")

            if state_col_w:
                def _outside_state(row):
                    mlos_state = safe_str(row.get("state_name","")).upper()
                    boundary_state = safe_str(row.get(state_col_w,"")).upper() if state_col_w in row else ""
                    if pd.isna(row.get("index_right")):
                        return True
                    return not name_similarity(mlos_state, boundary_state)
                outside_state_mask = joined_w.apply(_outside_state, axis=1)
                df.loc[outside_state_mask[outside_state_mask].index, "Outside State Boundary"] = "Y"

            _progress("Checking LGA/ward name match…", 78)

            def _ward_lga_check(row):
                if pd.isna(row.get("index_right")):
                    return False, False
                outside_ward = False
                outside_lga  = False
                if ward_col_w:
                    mlos_ward = safe_str(row.get("ward_name",""))
                    bound_ward = safe_str(row.get(ward_col_w,""))
                    if mlos_ward and bound_ward:
                        outside_ward = not name_similarity(mlos_ward, bound_ward)
                if lga_col_w:
                    mlos_lga = safe_str(row.get("lga_name",""))
                    bound_lga = safe_str(row.get(lga_col_w,""))
                    if mlos_lga and bound_lga:
                        outside_lga = not name_similarity(mlos_lga, bound_lga)
                return outside_ward, outside_lga

            results = joined_w.apply(_ward_lga_check, axis=1)
            ow_mask = results.apply(lambda x: x[0])
            ol_mask = results.apply(lambda x: x[1])
            df.loc[ow_mask[ow_mask].index, "Outside Ward"] = "Y"
            df.loc[ol_mask[ol_mask].index, "Outside LGA"] = "Y"

    except ImportError:
        st.warning("⚠️ GeoPandas not available. Spatial intersection/proximity checks skipped.")
    except Exception as e:
        st.warning(f"⚠️ Spatial check error: {e}. Some spatial flags may be incomplete.")

    return df

# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — SUMMARY FLAGS
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary_flags(df):
    def _any_y(df_local, cols):
        present = [c for c in cols if c in df_local.columns]
        if not present:
            return pd.Series([False]*len(df_local), index=df_local.index)
        return (df_local[present] == "Y").any(axis=1)

    critical_mask   = _any_y(df, CRITICAL_ERROR_FLAGS)
    spatial_mask    = _any_y(df, SPATIAL_FLAGS)
    attribute_mask  = _any_y(df, ATTRIBUTE_FLAGS)

    df = flag_col(df, "Critical Error Flag", critical_mask)
    df = flag_col(df, "Spatial Issues",      spatial_mask)
    df = flag_col(df, "Attribute Issues",    attribute_mask)
    df = flag_col(df, "No Issues", ~attribute_mask & ~spatial_mask)

    return df

# ─────────────────────────────────────────────────────────────────────────────
# COLUMN ORDERING & EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def order_output_columns(df):
    base = [c for c in EXPECTED_COLUMNS if c in df.columns]
    flags = [c for c in OUTPUT_FLAG_COLUMNS if c in df.columns]
    other = [c for c in df.columns if c not in base and c not in flags and not c.startswith("_")]
    return df[base + flags]

def to_csv_bytes(df):
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

# ─────────────────────────────────────────────────────────────────────────────
# CHARTING — PURE BLACK TEXT CONFIGURATION (#000000)
# ─────────────────────────────────────────────────────────────────────────────

CHART_COLORS = {
    "critical":  "#DC2626",
    "spatial":   "#2563EB",
    "attribute": "#D97706",
    "clean":     "#16A34A",
    "blue1":     "#1D4ED8",
    "blue4":     "#DBEAFE",
}

def chart_critical_proportion(df):
    total = len(df)
    critical = (df.get("Critical Error Flag","") == "Y").sum()
    ok = total - critical
    fig = go.Figure(go.Pie(
        labels=["Critical Errors", "No Critical Errors"],
        values=[critical, ok],
        hole=0.55,
        marker_colors=[CHART_COLORS["critical"], CHART_COLORS["blue4"]],
        textinfo="label+percent",
        textfont_size=12,
        textfont_color=PURE_BLACK
    ))
    fig.update_layout(
        title=dict(text="Critical Errors — Overall Proportion", font=dict(size=14, color=PURE_BLACK)),
        showlegend=True,
        legend=dict(font=dict(color=PURE_BLACK)),
        height=320,
        margin=dict(l=10,r=10,t=50,b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        annotations=[dict(text=f"<b>{critical:,}</b><br>critical", x=0.5, y=0.5,
                          font_size=13, showarrow=False, font_color=PURE_BLACK)]
    )
    return fig

def chart_critical_by_state(df):
    if "state_name" not in df.columns:
        return None
    by_state = df.groupby("state_name").agg(
        total=("sn","count") if "sn" in df.columns else ("state_name","count"),
        critical=("Critical Error Flag", lambda x: (x=="Y").sum())
    ).reset_index().sort_values("critical", ascending=True)
    fig = px.bar(
        by_state, y="state_name", x="critical",
        orientation="h",
        color="critical",
        color_continuous_scale=[[0, CHART_COLORS["blue4"]], [1, CHART_COLORS["critical"]]],
        labels={"state_name":"State","critical":"Critical Errors"},
        title="Critical Errors by State",
        text="critical",
    )
    fig.update_traces(textposition="outside", textfont_color=PURE_BLACK)
    fig.update_layout(
        height=max(350, len(by_state)*22 + 80),
        margin=dict(l=10,r=30,t=50,b=10),
        coloraxis_showscale=False,
        paper_bgcolor="white",
        plot_bgcolor=GREY_LIGHT,
        title_font=dict(size=14, color=PURE_BLACK),
        xaxis=dict(tickfont=dict(color=PURE_BLACK), title_font=dict(color=PURE_BLACK)),
        yaxis=dict(tickfont=dict(size=10, color=PURE_BLACK), title_font=dict(color=PURE_BLACK)),
    )
    return fig

def chart_issue_distribution(df, title_suffix=""):
    spatial_n   = (df.get("Spatial Issues","") == "Y").sum()
    attribute_n = (df.get("Attribute Issues","") == "Y").sum()
    no_issue_n  = (df.get("No Issues","") == "Y").sum()
    fig = go.Figure(go.Pie(
        labels=["Spatial Issues", "Attribute Issues", "No Issues"],
        values=[spatial_n, attribute_n, no_issue_n],
        hole=0.5,
        marker_colors=[CHART_COLORS["blue1"], CHART_COLORS["attribute"], CHART_COLORS["clean"]],
        textinfo="label+percent",
        textfont_size=11,
        textfont_color=PURE_BLACK
    ))
    fig.update_layout(
        title=dict(text=f"Issue Distribution{title_suffix}", font=dict(size=14, color=PURE_BLACK)),
        legend=dict(font=dict(color=PURE_BLACK)),
        height=320,
        margin=dict(l=10,r=10,t=50,b=10),
        paper_bgcolor="white",
    )
    return fig

def chart_issues_by_state(df):
    if "state_name" not in df.columns:
        return None
    agg = df.groupby("state_name").agg(
        spatial   =("Spatial Issues",   lambda x: (x=="Y").sum()),
        attribute =("Attribute Issues", lambda x: (x=="Y").sum()),
        no_issues =("No Issues",        lambda x: (x=="Y").sum()),
    ).reset_index().sort_values("spatial", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Spatial Issues",   x=agg["state_name"], y=agg["spatial"],   marker_color=CHART_COLORS["blue1"]))
    fig.add_trace(go.Bar(name="Attribute Issues", x=agg["state_name"], y=agg["attribute"], marker_color=CHART_COLORS["attribute"]))
    fig.add_trace(go.Bar(name="No Issues",        x=agg["state_name"], y=agg["no_issues"], marker_color=CHART_COLORS["clean"]))
    fig.update_layout(
        barmode="group",
        title=dict(text="Issue Types by State", font=dict(size=14, color=PURE_BLACK)),
        height=380,
        margin=dict(l=10,r=10,t=50,b=100),
        paper_bgcolor="white",
        plot_bgcolor=GREY_LIGHT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=PURE_BLACK)),
        xaxis=dict(tickangle=-40, tickfont=dict(size=10, color=PURE_BLACK)),
        yaxis=dict(tickfont=dict(color=PURE_BLACK)),
    )
    return fig

def build_state_summary(df):
    if "state_name" not in df.columns:
        return pd.DataFrame()
    count_col = "sn" if "sn" in df.columns else df.columns[0]
    agg = df.groupby("state_name").agg(
        Total=(count_col, "count"),
        Critical=("Critical Error Flag", lambda x: (x=="Y").sum()),
        Spatial=("Spatial Issues", lambda x: (x=="Y").sum()),
        Attribute=("Attribute Issues", lambda x: (x=="Y").sum()),
        Clean=("No Issues", lambda x: (x=="Y").sum()),
    ).reset_index()
    agg.columns = ["State","Total Records","Critical Errors","Spatial Issues","Attribute Issues","Clean Records"]
    agg["Critical %"] = (agg["Critical Errors"] / agg["Total Records"] * 100).round(1).astype(str) + "%"
    agg["Clean %"]    = (agg["Clean Records"]   / agg["Total Records"] * 100).round(1).astype(str) + "%"
    return agg.sort_values("Critical Errors", ascending=False)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='padding: 1rem 0 0.5rem 0;'>
          <div style='font-size:1.5rem;'>🗺️</div>
          <div style='font-size:1rem; font-weight:700; margin-top:0.2rem; color:#FFFFFF !important;'>MLoS QA/QC</div>
          <div style='font-size:0.72rem; color: rgba(255,255,255,0.6) !important; margin-top:0.1rem;'>Data Quality Platform</div>
        </div>
        <hr/>
        """, unsafe_allow_html=True)

        st.markdown("<b style='color:#FFFFFF !important;'>STEP 1 · MLoS DATA FILES</b>", unsafe_allow_html=True)
        st.caption("Upload one or more state CSV/Excel files")
        mlos_files = st.file_uploader(
            "MLoS Files (CSV or Excel)",
            type=["csv","xlsx","xls"],
            accept_multiple_files=True,
            key="mlos_upload",
            label_visibility="collapsed",
        )

        st.markdown("<br><b style='color:#FFFFFF !important;'>STEP 2 · WARD BOUNDARY</b>", unsafe_allow_html=True)
        st.caption("Shapefile (.zip with .shp) or GeoJSON")
        st.markdown(
            "<a href='https://drive.google.com/drive/folders/1Mtp1gZ7hd7ENk9u99f5NVlxsgSl1Dba1?usp=sharing' "
            "target='_blank' style='color:#93C5FD !important; font-size:0.75rem;'>🔗 eHA Ward Boundary Download</a>",
            unsafe_allow_html=True
        )
        ward_file = st.file_uploader(
            "Ward Boundary",
            type=["zip","geojson","json"],
            key="ward_upload",
            label_visibility="collapsed",
        )

        st.markdown("<br><b style='color:#FFFFFF !important;'>STEP 3 · SETTLEMENT EXTENT</b>", unsafe_allow_html=True)
        st.caption("Grid3 Settlement Extent v3.1 (.zip or GeoJSON)")
        st.markdown(
            "<a href='https://drive.google.com/drive/folders/1jVtU3J9MsZnkoMk6l8zl4oDmIehmPimx?usp=sharing' "
            "target='_blank' style='color:#93C5FD !important; font-size:0.75rem;'>🔗 Grid3 Extents Download</a>",
            unsafe_allow_html=True
        )
        sett_file = st.file_uploader(
            "Grid3 Settlement Extent",
            type=["zip","geojson","json"],
            key="sett_upload",
            label_visibility="collapsed",
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        run_btn = st.button("▶  Run QA/QC", use_container_width=True, type="primary")

        # Clear button — only shown when results exist
        clear_btn = False
        if _SS in st.session_state and st.session_state[_SS] is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            clear_btn = st.button("🔄  Clear & Reset", use_container_width=True,
                                  help="Clears all results and returns to the welcome page.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.7rem; color:rgba(255,255,255,0.45) !important; line-height:1.6;'>
        CRS: EPSG:4326 (WGS 84)<br>
        Spatial engine: GeoPandas<br>
        String match: RapidFuzz ≥95%<br>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(
            "<br><a href='https://docs.google.com/document/d/1mZsIgAwfh4CzzIyXXqRHu9OC_3-0zSOjMI3OowvVt2Q/edit?usp=sharing' "
            "target='_blank' style='color:#93C5FD !important; font-size:0.75rem;'>📄 Platform Key Notes</a>",
            unsafe_allow_html=True
        )

    return mlos_files, ward_file, sett_file, run_btn, clear_btn

# ─────────────────────────────────────────────────────────────────────────────
# FILE LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_mlos_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file, dtype=str, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, dtype=str, encoding="latin-1")
    else:
        df = pd.read_excel(uploaded_file, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df

def load_spatial_file(uploaded_file):
    try:
        import geopandas as gpd
        name = uploaded_file.name.lower()
        if name.endswith(".zip"):
            with tempfile.TemporaryDirectory() as tmp:
                zip_path = os.path.join(tmp, "data.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded_file.read())
                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(tmp)
                for root, _, files in os.walk(tmp):
                    for fn in files:
                        if fn.endswith(".shp") or fn.endswith(".geojson") or fn.endswith(".json"):
                            return gpd.read_file(os.path.join(root, fn))
        else:
            return gpd.read_file(uploaded_file)
    except Exception as e:
        st.warning(f"Could not load spatial file {uploaded_file.name}: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown("""
    <div class='app-header'>
      <div class='header-icon'>🗺️</div>
      <div>
        <h1>MLoS Quality Assurance & Quality Control Platform</h1>
        <p>Master List of Settlements · Spatial & Attribute  QA/QC · Up to 37 States · EPSG:4326</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialise session state slot
    if _SS not in st.session_state:
        st.session_state[_SS] = None

    mlos_files, ward_file, sett_file, run_btn, clear_btn = render_sidebar()

    # ── Clear button handler ───────────────────────────────────────────────────
    if clear_btn:
        st.session_state[_SS] = None
        st.rerun()

    # ── Welcome page (no results yet) ─────────────────────────────────────────
    if st.session_state[_SS] is None and not run_btn:
        if not mlos_files:
            # ── Getting Started Banner ──────────────────────────────────────
            st.markdown(f"""
            <div class='info-box' style='background:linear-gradient(135deg,{BLUE_FAINT} 0%,#E0EAFF 100%); border:1px solid {BLUE_PALE}; border-radius:10px; padding:1rem 1.2rem; margin-bottom:1.2rem;'>
            <span style='font-size:1.1rem; font-weight:700; color:{BLUE_DARK} !important;'>👈 How to get started</span><br>
            <span style='color:{PURE_BLACK} !important; font-size:0.88rem;'>
            Upload your MLoS state file(s), Ward Boundary, and Grid3 Settlement Extent in the sidebar, then click <strong>▶ Run QA/QC</strong>.<br>
            <strong>Supported formats:</strong> MLoS → CSV or Excel &nbsp;|&nbsp; Spatial → Shapefile (.zip) or GeoJSON
            </span>
            </div>
            """, unsafe_allow_html=True)

            # ── Quick Stats ─────────────────────────────────────────────────
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='label'>Attribute Checks</div>
                  <div class='value val-blue'>39+</div>
                  <div class='sublabel'>column-level validations</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='label'>Spatial Checks</div>
                  <div class='value val-blue'>9</div>
                  <div class='sublabel'>geocoord & boundary tests</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='label'>Output Flag Columns</div>
                  <div class='value val-blue'>51</div>
                  <div class='sublabel'>added to merged CSV</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Key Notes Tabs ───────────────────────────────────────────────
            st.markdown(f"<div class='section-title'>📖 Platform Key Notes</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style='font-size:0.82rem; color:{PURE_BLACK} !important; margin-bottom:0.8rem;'>
            Read these notes carefully before using the platform. A full version of this document is available here:
            <a href='https://docs.google.com/document/d/1mZsIgAwfh4CzzIyXXqRHu9OC_3-0zSOjMI3OowvVt2Q/edit?usp=sharing'
               target='_blank' style='color:{BLUE_MID} !important; font-weight:600;'>📄 MLoS Platform Key Notes (Google Doc)</a>
            </div>
            """, unsafe_allow_html=True)

            kn_tab1, kn_tab2, kn_tab3 = st.tabs([
                "  📂 1. Data Sources  ",
                "  📋 2. MLoS Schema  ",
                "  🚨 3. Critical Flags  ",
            ])

            # Tab 1 — Data Sources
            with kn_tab1:
                st.markdown(f"""
                <p style='color:{PURE_BLACK} !important; font-size:0.88rem; margin-bottom:0.8rem;'>
                Three data sources are required to run the QA/QC. Upload them in the sidebar as described below.
                </p>
                <table style='width:100%; border-collapse:collapse; font-family:"Open Sans",sans-serif; font-size:0.83rem;'>
                  <thead>
                    <tr style='background:{BLUE_DARK};'>
                      <th style='padding:0.5rem 0.8rem; color:#FFFFFF; text-align:left; min-width:160px;'>Input</th>
                      <th style='padding:0.5rem 0.8rem; color:#FFFFFF; text-align:left; min-width:90px;'>Format</th>
                      <th style='padding:0.5rem 0.8rem; color:#FFFFFF; text-align:left;'>Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style='background:#FFFFFF;'>
                      <td style='padding:0.45rem 0.8rem; border-bottom:1px solid #E2E8F0; font-weight:600; color:{BLUE_MID} !important;'>MLoS State Files</td>
                      <td style='padding:0.45rem 0.8rem; border-bottom:1px solid #E2E8F0; color:{PURE_BLACK} !important;'>CSV / Excel</td>
                      <td style='padding:0.45rem 0.8rem; border-bottom:1px solid #E2E8F0; color:{PURE_BLACK} !important;'>One or more state files. Must match the 40-column MLoS schema exactly.</td>
                    </tr>
                    <tr style='background:{BLUE_FAINT};'>
                      <td style='padding:0.45rem 0.8rem; border-bottom:1px solid #E2E8F0; font-weight:600; color:{BLUE_MID} !important;'>Ward Boundary</td>
                      <td style='padding:0.45rem 0.8rem; border-bottom:1px solid #E2E8F0; color:{PURE_BLACK} !important;'>Shapefile (.zip) / GeoJSON</td>
                      <td style='padding:0.45rem 0.8rem; border-bottom:1px solid #E2E8F0; color:{PURE_BLACK} !important;'>eHA ward boundary file used for Outside State/LGA/Ward checks. Optional but recommended.</td>
                    </tr>
                    <tr style='background:#FFFFFF;'>
                      <td style='padding:0.45rem 0.8rem; font-weight:600; color:{BLUE_MID} !important;'>Grid3 Settlement Extent</td>
                      <td style='padding:0.45rem 0.8rem; color:{PURE_BLACK} !important;'>Shapefile (.zip) / GeoJSON</td>
                      <td style='padding:0.45rem 0.8rem; color:{PURE_BLACK} !important;'>Grid3 Nigeria Settlement Extents v3.1. Used for "Not Intersecting Settlement Extent" check. Optional.</td>
                    </tr>
                  </tbody>
                </table>
                """, unsafe_allow_html=True)

            # Tab 2 — MLoS Schema
            with kn_tab2:
                schema_rows = [
                    ("1","sn","int","Serial number / row index"),
                    ("2","unique_code","varchar","Auto-generated: State_LGA_Ward_Settlement"),
                    ("3","state_name","varchar","State name"),
                    ("4","lga_name","varchar","LGA name"),
                    ("5","ward_name","varchar","Ward name"),
                    ("6","take_off_point","varchar","Take-off point of vaccination teams"),
                    ("7","settlement_name","varchar","Settlement name"),
                    ("8","primary_settlement_name","varchar","Name of primary settlement"),
                    ("9","alternate_name","varchar","Alternative or other name of the settlement"),
                    ("10","latitude","decimal","Geocoordinate, Y value"),
                    ("11","longitude","decimal","Geocoordinate, X value"),
                    ("12","security_compromised","char(1)","Y or N or NA"),
                    ("13","accessibility_status","varchar","Fully Accessible / Inaccessible / Partially Accessible"),
                    ("14","reasons_for_inaccessibility","varchar","Reason for inaccessibility"),
                    ("15","habitational_status","varchar","Inhabited / Partially Inhabited / Abandoned / Migrated"),
                    ("16","set_population","int","Settlement population"),
                    ("17","set_target","int","Settlement target population"),
                    ("18","number_of_household","int","Number of households"),
                    ("19","noncompliant_household","int","Non-compliant households"),
                    ("20","team_code","int","Team code"),
                    ("21","day_of_activity","varchar","1, 2, 3, 4 or combinations (underscore-separated) or NA"),
                    ("22","urban","char(1)","Y or N"),
                    ("23","rural","char(1)","Y or N"),
                    ("24","scattered","char(1)","Y or N"),
                    ("25","highrisk","char(1)","Y or N or NA"),
                    ("26","slums","char(1)","Y or N or NA"),
                    ("27","densely_populated","char(1)","Y or N or NA"),
                    ("28","hard2reach","char(1)","Y or N or NA"),
                    ("29","border","char(1)","Y or N or NA"),
                    ("30","nomadic","char(1)","Y or N or NA"),
                    ("31","riverine","char(1)","Y or N or NA"),
                    ("32","fulani","char(1)","Y or N or NA"),
                    ("33","source","varchar","Source of settlement data — must not be empty"),
                    ("34","comments","text","Comments about the settlement"),
                    ("35","globalid","uuid","Settlement unique ID"),
                    ("36","validation_status","varchar","Validated / Not Validated / Validation Ongoing / Validated Unknown"),
                    ("37","gis_feedback","text","Feedback comments from state data analysts"),
                    ("38","master.id","serial","Unique ID from national MLoS harmonisation"),
                    ("39","mlos_id","varchar","Unique ID from national MLoS harmonisation"),
                    ("40","eha_guid","uuid","Unique ID from eHA database"),
                ]

                schema_rows_html = "".join(
                    f"<tr style='background:{'#F8FAFF' if int(pos) % 2 == 0 else '#FFFFFF'};'>"
                    f"<td style='padding:0.4rem 0.7rem; border-bottom:1px solid #EEF2FF; color:{PURE_BLACK} !important; text-align:center;'>{pos}</td>"
                    f"<td style='padding:0.4rem 0.7rem; border-bottom:1px solid #EEF2FF; font-family:monospace; font-weight:600; color:{BLUE_MID} !important;'>{name}</td>"
                    f"<td style='padding:0.4rem 0.7rem; border-bottom:1px solid #EEF2FF; color:#6B7280 !important;'>{dtype}</td>"
                    f"<td style='padding:0.4rem 0.7rem; border-bottom:1px solid #EEF2FF; color:{PURE_BLACK} !important;'>{desc}</td>"
                    f"</tr>"
                    for pos, name, dtype, desc in schema_rows
                )

                st.markdown(f"""
                <div style='overflow-x:auto; border-radius:8px; border:1px solid #E2E8F0; max-height:420px; overflow-y:auto;'>
                <table style='width:100%; border-collapse:collapse; font-family:"Open Sans",sans-serif; font-size:0.82rem;'>
                  <thead style='position:sticky; top:0; z-index:1;'>
                    <tr style='background:{BLUE_DARK};'>
                      <th style='padding:0.6rem 0.7rem; color:#FFFFFF; text-align:center; min-width:40px;'>#</th>
                      <th style='padding:0.6rem 0.7rem; color:#FFFFFF; text-align:left; min-width:180px;'>Field Name</th>
                      <th style='padding:0.6rem 0.7rem; color:#FFFFFF; text-align:left; min-width:80px;'>Type</th>
                      <th style='padding:0.6rem 0.7rem; color:#FFFFFF; text-align:left;'>Description</th>
                    </tr>
                  </thead>
                  <tbody>{schema_rows_html}</tbody>
                </table>
                </div>
                """, unsafe_allow_html=True)

            # Tab 3 — Critical Flags
            with kn_tab3:
                st.markdown(f"""
                <div style='background:#FEF2F2; border:1px solid #FECACA; border-radius:8px; padding:0.8rem 1rem; margin-bottom:0.8rem; font-size:0.87rem; color:#7F1D1D !important;'>
                ⛔ <strong>12 Critical Flags</strong> must be resolved before the MLoS can be accepted. Records with any critical flag active will be returned to the analyst.
                </div>
                """, unsafe_allow_html=True)

                critical_rows = [
                    ("1","attribute duplicate = 'Y'","Remove records with the same state, LGA, ward and settlement name."),
                    ("2","Stacked points = 'Y'","Remove records with identical latitude/longitude coordinates."),
                    ("3","Duplicate eha_guid = 'Y'","Remove duplicate GUID codes not present in the previous version."),
                    ("4","Wrong Entry Source","Input a valid source for the record."),
                    ("5","Population Conflict = 'Y'","Target population cannot be equal to or less than settlement population."),
                    ("6","Wrong Entry Day of Activity Entry = 'Y'",
                     "Day of activity accepts: 1, 2, 3, 4, 1_2, 1_3, 1_4, 2_3, 2_4, 3_4, 1_2_3, 1_2_4, 1_3_4, 2_3_4, 1_2_3_4, NA. "
                     "⚠️ Ignored if validation_status = 'Validated Unknown'."),
                    ("7","Non-Numerical Number of Household = 'Y'",
                     "Remove non-numerical entries. ⚠️ Ignored if validation_status = 'Validated Unknown'."),
                    ("8","Wrong Entry Urban = 'Y'",
                     "Must be 'Y' or 'N'. ⚠️ Ignored if validation_status = 'Validated Unknown'."),
                    ("9","Wrong Entry Rural = 'Y'",
                     "Must be 'Y' or 'N'. ⚠️ Ignored if validation_status = 'Validated Unknown'."),
                    ("10","Wrong Entry Scattered = 'Y'",
                     "Must be 'Y' or 'N'. ⚠️ Ignored if validation_status = 'Validated Unknown'."),
                    ("11","Settlement Type Conflict = 'Y'",
                     "'Urban', 'rural', 'scattered' cannot all be the same entry. ⚠️ Ignored if validation_status = 'Validated Unknown'."),
                    ("12","Security and Accessibility conflict = 'Y'",
                     "Flagged when security_compromised = 'Y' AND accessibility_status = 'Fully Accessible' AND "
                     "reasons_for_inaccessibility is blank/null/empty. A security-compromised settlement cannot be "
                     "fully accessible without a documented reason for inaccessibility. Update security_compromised, "
                     "accessibility_status, or provide a reason for inaccessibility to resolve."),
                ]

                crit_rows_html = "".join(
                    f"<tr style='background:{'#FFF7F7' if int(sn) % 2 == 0 else '#FFFFFF'};'>"
                    f"<td style='padding:0.45rem 0.7rem; border-bottom:1px solid #FEE2E2; color:{PURE_BLACK} !important; text-align:center; font-weight:700;'>{sn}</td>"
                    f"<td style='padding:0.45rem 0.7rem; border-bottom:1px solid #FEE2E2; font-weight:600; color:#DC2626 !important; font-size:0.82rem;'>{flag}</td>"
                    f"<td style='padding:0.45rem 0.7rem; border-bottom:1px solid #FEE2E2; color:{PURE_BLACK} !important; font-size:0.82rem;'>{resolution}</td>"
                    f"</tr>"
                    for sn, flag, resolution in critical_rows
                )

                st.markdown(f"""
                <div style='overflow-x:auto; border-radius:8px; border:1px solid #FEE2E2;'>
                <table style='width:100%; border-collapse:collapse; font-family:"Open Sans",sans-serif;'>
                  <thead>
                    <tr style='background:#991B1B;'>
                      <th style='padding:0.6rem 0.7rem; color:#FFFFFF; text-align:center; width:40px;'>#</th>
                      <th style='padding:0.6rem 0.7rem; color:#FFFFFF; text-align:left; min-width:220px;'>Critical Flag</th>
                      <th style='padding:0.6rem 0.7rem; color:#FFFFFF; text-align:left;'>Resolution / Notes</th>
                    </tr>
                  </thead>
                  <tbody>{crit_rows_html}</tbody>
                </table>
                </div>
                <p style='font-size:0.78rem; color:#6B7280 !important; margin-top:0.5rem;'>
                ⚠️ Flags marked as "Ignored if validation_status = 'Validated Unknown'" will be automatically blanked out by the platform for those records.
                </p>
                """, unsafe_allow_html=True)

        return

    # ── RUN QA/QC — only execute processing when Run button is pressed ─────────
    if run_btn:
        if not mlos_files:
            st.error("Please upload at least one MLoS file before running QA/QC.")
            return

        prog_text = st.empty()
        prog_bar  = st.progress(0)

        def update_progress(msg, pct):
            prog_text.markdown(f"<div class='info-box'>⚙️ {msg}</div>", unsafe_allow_html=True)
            prog_bar.progress(min(int(pct), 99))

        update_progress("Loading spatial reference files…", 1)

        ward_gdf       = load_spatial_file(ward_file)  if ward_file  else None
        settlement_gdf = load_spatial_file(sett_file)  if sett_file  else None

        update_progress("Validating schemas…", 3)

        valid_frames   = []
        dropped_states = []
        schema_report  = []

        for uf in mlos_files:
            try:
                df_raw = load_mlos_file(uf)
            except Exception as e:
                dropped_states.append((uf.name, f"Load error: {e}"))
                schema_report.append({"File": uf.name, "Status": "❌ Load Error", "Detail": str(e)})
                continue

            ok, msg = validate_schema(df_raw, uf.name)
            if ok:
                valid_frames.append(df_raw)
                schema_report.append({"File": uf.name, "Status": "✅ Valid", "Detail": msg})
            else:
                dropped_states.append((uf.name, msg))
                schema_report.append({"File": uf.name, "Status": "❌ Dropped", "Detail": msg})

        if not valid_frames:
            prog_bar.progress(0)
            prog_text.empty()
            st.error("No valid MLoS files found. Check column names and order then re-upload.")
            return

        update_progress("Merging state files…", 4)
        df_all = pd.concat(valid_frames, ignore_index=True)
        df_all.columns = [c.strip() for c in df_all.columns]
        total_raw = len(df_all)

        update_progress("Preparing data (TRIM, unique_code, coordinate cleaning)…", 6)
        df_all = prepare_data(df_all)

        df_all = run_attribute_checks(df_all, progress_cb=update_progress)

        update_progress("Running spatial QA/QC…", 48)
        df_all = run_spatial_checks(df_all, ward_gdf=ward_gdf, settlement_gdf=settlement_gdf, progress_cb=update_progress)

        update_progress("Computing summary flags…", 85)
        df_all = compute_summary_flags(df_all)

        update_progress("Ordering output columns…", 90)
        df_out = order_output_columns(df_all)

        prog_bar.progress(100)
        prog_text.markdown("<div class='info-box'>✅ QA/QC complete!</div>", unsafe_allow_html=True)

        # ── Persist results to session state ──────────────────────────────────
        states_available = sorted(df_out["state_name"].dropna().unique().tolist()) \
            if "state_name" in df_out.columns else []
        summary_tbl = build_state_summary(df_out)

        st.session_state[_SS] = {
            "df_out":          df_out,
            "schema_report":   schema_report,
            "schema_report_df": pd.DataFrame(schema_report),
            "dropped_states":  dropped_states,
            "total_raw":       total_raw,
            "n_valid_files":   len(valid_frames),
            "n_dropped_files": len(dropped_states),
            "states_available":states_available,
            "summary_tbl":     summary_tbl,
        }
        # Clear progress widgets now results are saved
        prog_text.empty()
        prog_bar.empty()

    # ── RESULTS RENDERING — drawn from session state on every rerun ────────────
    res = st.session_state[_SS]
    if res is None:
        return

    df_out          = res["df_out"]
    schema_report   = res["schema_report"]
    dropped_states  = res["dropped_states"]
    total_raw       = res["total_raw"]
    states_available= res["states_available"]
    summary_tbl     = res["summary_tbl"]

    # Schema validation table
    st.markdown("<div class='section-title'>📋 Schema Validation</div>", unsafe_allow_html=True)
    schema_df = pd.DataFrame(schema_report)
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

    if dropped_states:
        dropped_html = "<br>".join([f"<b>{n}</b>: {r}" for n, r in dropped_states])
        st.markdown(f"<div class='dropped-box'>⛔ <b>Dropped files (schema mismatch):</b><br>{dropped_html}</div>", unsafe_allow_html=True)

    # ── State filter — stored in session state so it survives reruns ───────────
    st.markdown("<div class='section-title'>🔎 Filter by State</div>", unsafe_allow_html=True)
    selected_states = st.multiselect(
        "Select states to drill down (leave empty for all)",
        options=states_available,
        default=[],
        key="state_filter",
    )
    df_view = df_out[df_out["state_name"].isin(selected_states)] if selected_states else df_out

    # ── Summary metrics ────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📊 Summary Metrics</div>", unsafe_allow_html=True)
    total   = len(df_view)
    n_crit  = (df_view.get("Critical Error Flag","") == "Y").sum() if "Critical Error Flag" in df_view.columns else 0
    n_spat  = (df_view.get("Spatial Issues","")      == "Y").sum() if "Spatial Issues"      in df_view.columns else 0
    n_attr  = (df_view.get("Attribute Issues","")    == "Y").sum() if "Attribute Issues"    in df_view.columns else 0
    n_clean = (df_view.get("No Issues","")           == "Y").sum() if "No Issues"           in df_view.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"<div class='metric-card'><div class='label'>Total Records</div><div class='value val-blue'>{total:,}</div><div class='sublabel'>{len(selected_states) or len(states_available)} states</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div class='label'>Critical Errors</div><div class='value val-red'>{n_crit:,}</div><div class='sublabel'>{n_crit/total*100:.1f}% of records</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card'><div class='label'>Spatial Issues</div><div class='value val-blue'>{n_spat:,}</div><div class='sublabel'>{n_spat/total*100:.1f}% of records</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'><div class='label'>Attribute Issues</div><div class='value val-amber'>{n_attr:,}</div><div class='sublabel'>{n_attr/total*100:.1f}% of records</div></div>", unsafe_allow_html=True)
    with c5:
        st.markdown(f"<div class='metric-card'><div class='label'>Clean Records</div><div class='value val-green'>{n_clean:,}</div><div class='sublabel'>{n_clean/total*100:.1f}% of records</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Critical errors panel ──────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🚨 Critical Errors Review Panel</div>", unsafe_allow_html=True)
    df_critical = df_view[df_view["Critical Error Flag"] == "Y"] \
        if "Critical Error Flag" in df_view.columns else pd.DataFrame()

    if not df_critical.empty:
        st.dataframe(df_critical, use_container_width=True, height=280)
        crit_bytes = to_csv_bytes(df_critical)
        st.download_button(
            label=f"⬇️ Download Critical Errors Table ({len(df_critical):,} records)",
            data=crit_bytes,
            file_name="MLoS_Critical_Errors_Report.csv",
            mime="text/csv",
            key="download_crit_table"
        )
    else:
        st.success("No records with active critical error flags discovered within current state profile constraints.")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📈 Quality Overview Charts</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "  Critical Errors — Overall  ",
        "  Critical Errors — By State  ",
        "  Issue Types — Overall  ",
        "  Issue Types — By State  ",
    ])

    with tab1:
        fig1 = chart_critical_proportion(df_view)
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2 = chart_critical_by_state(df_view)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("State name column not found for state-level breakdown.")

    with tab3:
        suffix = f" — {', '.join(selected_states)}" if selected_states else " — All States"
        fig3 = chart_issue_distribution(df_view, suffix)
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        fig4 = chart_issues_by_state(df_view)
        if fig4:
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("State name column not found.")

    # ── State-level summary table ──────────────────────────────────────────────
    st.markdown("<div class='section-title'>📋 State-level Summary Table</div>", unsafe_allow_html=True)
    if not summary_tbl.empty:
        def _amber_critical(val):
            try:
                if int(val) > 0:
                    return "background-color: #D97706; color: #FFFFFF; font-weight: 700;"
            except (ValueError, TypeError):
                pass
            return ""
        _style_method = getattr(summary_tbl.style, "map", None) or getattr(summary_tbl.style, "applymap")
        styled_summary = _style_method(_amber_critical, subset=["Critical Errors"])
        st.dataframe(styled_summary, use_container_width=True, hide_index=True)

    # ── Data preview ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🗃️ Data Preview (first 500 rows)</div>", unsafe_allow_html=True)
    preview_cols = list(df_out.columns[:15]) + ["Critical Error Flag","Spatial Issues","Attribute Issues","No Issues","Confirm with Programs team"]
    preview_cols = [c for c in preview_cols if c in df_view.columns]
    st.dataframe(df_view[preview_cols].head(500), use_container_width=True, height=320)

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>⬇️ Download Output</div>", unsafe_allow_html=True)

    dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 1])

    with dl_col1:
        csv_bytes = to_csv_bytes(df_out)
        st.download_button(
            label=f"⬇️  Full QA/QC Output CSV  ({len(df_out):,} records)",
            data=csv_bytes,
            file_name="MLoS_QAQC_Output.csv",
            mime="text/csv",
            key="download_full_csv",
        )

    with dl_col2:
        if selected_states:
            csv_filtered = to_csv_bytes(df_view)
            st.download_button(
                label=f"⬇️  Filtered CSV — {', '.join(selected_states)}  ({len(df_view):,} records)",
                data=csv_filtered,
                file_name=f"MLoS_QAQC_{'_'.join(selected_states)}.csv",
                mime="text/csv",
                key="download_filtered_csv",
            )

    with dl_col3:
        # PDF report — generated lazily and cached in session state
        if "pdf_bytes" not in res or res["pdf_bytes"] is None:
            with st.spinner("Generating PDF report…"):
                try:
                    pdf_bytes = generate_pdf_report(res)
                    st.session_state[_SS]["pdf_bytes"] = pdf_bytes
                except Exception as e:
                    st.session_state[_SS]["pdf_bytes"] = None
                    st.warning(f"PDF generation failed: {e}")

        pdf_data = st.session_state[_SS].get("pdf_bytes")
        if pdf_data:
            run_label = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                label="📄  Download PDF Report",
                data=pdf_data,
                file_name=f"MLoS_QAQC_Report_{run_label}.pdf",
                mime="application/pdf",
                key="download_pdf_report",
            )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align:center; font-size:0.72rem; color:#000000; padding:1rem;'>
    MLoS QA/QC Platform · EPSG:4326 · {total_raw:,} records ingested from {res['n_valid_files']} state file(s)
    {f' · {len(dropped_states)} file(s) dropped' if dropped_states else ''}
    </div>
    """, unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_pdf_report(res: dict) -> bytes:
    """Build a QA/QC summary PDF and return it as bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="MLoS QA/QC Report",
    )

    PAGE_W = A4[0] - 4*cm  # usable width

    styles = getSampleStyleSheet()
    H1 = ParagraphStyle("h1", parent=styles["Heading1"],
                        fontSize=16, textColor=colors.HexColor(BLUE_DARK),
                        spaceAfter=6)
    H2 = ParagraphStyle("h2", parent=styles["Heading2"],
                        fontSize=12, textColor=colors.HexColor(BLUE_MID),
                        spaceBefore=14, spaceAfter=4)
    BODY = ParagraphStyle("body", parent=styles["Normal"],
                          fontSize=9, leading=13,
                          textColor=colors.black)
    SMALL = ParagraphStyle("small", parent=styles["Normal"],
                           fontSize=8, leading=11,
                           textColor=colors.HexColor("#555555"))
    CELL  = ParagraphStyle("cell", parent=styles["Normal"],
                           fontSize=8, leading=10,
                           textColor=colors.black)

    story = []

    # ── Cover / header ─────────────────────────────────────────────────────────
    story.append(Paragraph("MLoS QA/QC Platform", H1))
    story.append(Paragraph("Master List of Settlements · Quality Assurance &amp; Quality Control Report", BODY))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor(BLUE_PALE),
                            spaceAfter=8))

    run_ts = datetime.now().strftime("%d %B %Y, %H:%M")
    meta_data = [
        ["Run date/time:", run_ts],
        ["Total records ingested:", f"{res['total_raw']:,}"],
        ["Valid state files processed:", str(res['n_valid_files'])],
        ["Dropped files:", str(res['n_dropped_files'])],
        ["States in output:", str(len(res['states_available']))],
    ]
    meta_tbl = Table(meta_data, colWidths=[5*cm, PAGE_W - 5*cm])
    meta_tbl.setStyle(TableStyle([
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 9),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.black),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 12))

    # ── Schema validation results ───────────────────────────────────────────────
    story.append(Paragraph("1. Schema Validation", H2))
    schema_df = res["schema_report_df"]
    tbl_data = [list(schema_df.columns)] + schema_df.values.tolist()
    col_ws = [PAGE_W * 0.35, PAGE_W * 0.18, PAGE_W * 0.47]
    schema_tbl = Table(
        [[Paragraph(str(c), CELL) for c in row] for row in tbl_data],
        colWidths=col_ws,
        repeatRows=1,
    )
    schema_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor(BLUE_DARK)),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EFF6FF")]),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#CBD5E1")),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ]))
    story.append(schema_tbl)
    story.append(Spacer(1, 10))

    # ── Summary metrics (all states) ───────────────────────────────────────────
    story.append(Paragraph("2. Summary Metrics (All States)", H2))
    df_out = res["df_out"]
    total   = len(df_out)
    n_crit  = (df_out.get("Critical Error Flag","") == "Y").sum() if "Critical Error Flag" in df_out.columns else 0
    n_spat  = (df_out.get("Spatial Issues","")      == "Y").sum() if "Spatial Issues"      in df_out.columns else 0
    n_attr  = (df_out.get("Attribute Issues","")    == "Y").sum() if "Attribute Issues"    in df_out.columns else 0
    n_clean = (df_out.get("No Issues","")           == "Y").sum() if "No Issues"           in df_out.columns else 0

    def pct(n, t): return f"{n/t*100:.1f}%" if t else "N/A"

    metrics_data = [
        ["Metric", "Count", "% of Records"],
        ["Total Records",    f"{total:,}",   "100%"],
        ["Critical Errors",  f"{n_crit:,}",  pct(n_crit, total)],
        ["Spatial Issues",   f"{n_spat:,}",  pct(n_spat, total)],
        ["Attribute Issues", f"{n_attr:,}",  pct(n_attr, total)],
        ["Clean Records",    f"{n_clean:,}", pct(n_clean, total)],
    ]
    col_ws_m = [PAGE_W*0.50, PAGE_W*0.25, PAGE_W*0.25]
    metrics_tbl = Table(metrics_data, colWidths=col_ws_m)
    metrics_tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor(BLUE_DARK)),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("ALIGN",       (1,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EFF6FF")]),
        # Highlight critical row
        ("BACKGROUND",  (0,2), (-1,2), colors.HexColor("#FEF2F2")),
        ("TEXTCOLOR",   (0,2), (-1,2), colors.HexColor(RED_FLAG)),
        ("FONTNAME",    (0,2), (-1,2), "Helvetica-Bold"),
        ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#CBD5E1")),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
    ]))
    story.append(metrics_tbl)
    story.append(Spacer(1, 10))

    # ── State-level summary table ───────────────────────────────────────────────
    story.append(Paragraph("3. State-Level Summary", H2))
    summary_tbl_df = res["summary_tbl"]
    if not summary_tbl_df.empty:
        st_data = [list(summary_tbl_df.columns)] + summary_tbl_df.values.tolist()
        n_cols = len(summary_tbl_df.columns)
        cw = PAGE_W / n_cols
        col_ws_s = [PAGE_W*0.22] + [cw * 0.95] * (n_cols - 1)
        state_tbl = Table(
            [[Paragraph(str(c), CELL) for c in row] for row in st_data],
            colWidths=col_ws_s,
            repeatRows=1,
        )
        state_tbl.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor(BLUE_DARK)),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 8),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EFF6FF")]),
            ("GRID",        (0,0), (-1,-1), 0.3, colors.HexColor("#CBD5E1")),
            ("TOPPADDING",  (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        story.append(state_tbl)
    else:
        story.append(Paragraph("State breakdown not available (state_name column missing).", SMALL))
    story.append(Spacer(1, 10))

    # ── Critical errors detail (up to 100 rows) ────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("4. Critical Errors Detail (up to 100 records)", H2))
    df_crit = df_out[df_out.get("Critical Error Flag", pd.Series(dtype=str)) == "Y"] \
        if "Critical Error Flag" in df_out.columns else pd.DataFrame()

    if not df_crit.empty:
        story.append(Paragraph(
            f"Total records with active critical error flags: <b>{len(df_crit):,}</b>. "
            f"Showing first {min(100, len(df_crit))} rows below.",
            BODY))
        story.append(Spacer(1, 6))

        crit_cols = ["state_name","lga_name","ward_name","settlement_name",
                     "Critical Error Flag"] + \
                    [c for c in CRITICAL_ERROR_FLAGS if c in df_crit.columns]
        crit_cols = [c for c in crit_cols if c in df_crit.columns]
        sample = df_crit[crit_cols].head(100)

        crit_tbl_data = [crit_cols] + sample.fillna("").values.tolist()
        crit_col_w = PAGE_W / len(crit_cols)
        crit_table = Table(
            [[Paragraph(str(v)[:60], CELL) for v in row] for row in crit_tbl_data],
            colWidths=[crit_col_w] * len(crit_cols),
            repeatRows=1,
        )
        crit_table.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#991B1B")),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 7),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#FFF7F7")]),
            ("GRID",        (0,0), (-1,-1), 0.25, colors.HexColor("#FEE2E2")),
            ("TOPPADDING",  (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0),(-1,-1), 3),
        ]))
        story.append(crit_table)
    else:
        story.append(Paragraph(
            "No records with active critical error flags were found in this dataset.", BODY))

    story.append(Spacer(1, 12))

    # ── Flag coverage summary ───────────────────────────────────────────────────
    story.append(Paragraph("5. Flag Coverage Summary", H2))
    story.append(Paragraph(
        "Number of records flagged (Y) for each QA/QC check across all states:", BODY))
    story.append(Spacer(1, 4))

    flag_rows = [["Flag Name", "Count", "% of Records", "Category"]]
    def _cat(f):
        if f in CRITICAL_ERROR_FLAGS: return "Critical"
        if f in SPATIAL_FLAGS: return "Spatial"
        if f in ATTRIBUTE_FLAGS: return "Attribute"
        return "Summary"

    for f in OUTPUT_FLAG_COLUMNS:
        if f in df_out.columns:
            cnt = (df_out[f] == "Y").sum()
            if cnt > 0:
                flag_rows.append([f, f"{cnt:,}", pct(cnt, total), _cat(f)])

    flag_col_ws = [PAGE_W*0.46, PAGE_W*0.12, PAGE_W*0.14, PAGE_W*0.18]
    flag_table = Table(
        [[Paragraph(str(v), CELL) for v in row] for row in flag_rows],
        colWidths=flag_col_ws,
        repeatRows=1,
    )

    def _row_bg(i, cat):
        if cat == "Critical": return colors.HexColor("#FEF2F2")
        if cat == "Spatial":  return colors.HexColor("#EFF6FF")
        if cat == "Attribute":return colors.HexColor("#FFFBEB")
        return colors.white

    flag_tbl_styles = [
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor(BLUE_DARK)),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 8),
        ("GRID",        (0,0), (-1,-1), 0.25, colors.HexColor("#CBD5E1")),
        ("TOPPADDING",  (0,0), (-1,-1), 3),
        ("BOTTOMPADDING",(0,0),(-1,-1), 3),
    ]
    for i, row in enumerate(flag_rows[1:], start=1):
        flag_tbl_styles.append(("BACKGROUND", (0,i), (-1,i), _row_bg(i, row[3])))

    flag_table.setStyle(TableStyle(flag_tbl_styles))
    story.append(flag_table)

    # ── Footer ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"MLoS QA/QC Platform · EPSG:4326 · Generated {run_ts} · "
        f"{res['total_raw']:,} records from {res['n_valid_files']} state file(s)",
        SMALL))

    doc.build(story)
    buf.seek(0)
    return buf.read()


if __name__ == '__main__':
    main()
