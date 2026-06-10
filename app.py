"""
MLoS QA/QC Platform — Single-file Streamlit Application
Version 1.1 | June 2026
"""

import io
import os
import warnings
import zipfile
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from rapidfuzz import fuzz

warnings.filterwarnings("ignore")

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
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght=300;400;500;600;700&family=JetBrains+Mono:wght=400;500&display=swap');

  html, body, [class*="css"], .stMarkdown, p, div, label, span {{
    font-family: 'Inter', sans-serif;
    color: {WHITE} !important;
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
    color: {WHITE} !important;
    border-left: 4px solid {BLUE_MID};
    padding-left: 0.75rem;
    margin: 1.5rem 0 0.8rem 0;
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
    background: transparent;
    border-radius: 6px 6px 0 0;
    font-weight: 500;
    color: {PURE_BLACK} !important;
    padding: 0.5rem 1rem;
  }}
  .stTabs [aria-selected="true"] {{
    background: {BLUE_PALE} !important;
    color: {PURE_BLACK} !important;
    font-weight: 700 !important;
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
    "Non-Numerical Number of Household","Wrong Entry Day of Activity Entry",
    "Wrong Entry SecurityComp","Wrong Entry Accesibility_status",
    "Wrong Entry reasons_for_inaccessibility","Wrong Entry Habitational_status",
    "Wrong Entry Urban","Wrong Entry Rural","Wrong Entry Scattered",
    "Wrong Entry Highrisk","Wrong Entry Border","Wrong Entry Densely_populated",
    "Wrong Entry Hard2reach","Wrong Entry Nomadic","Wrong Entry Riverine",
    "Wrong Entry Fulani","Wrong Entry Validation status","No Statename",
    "No LGAname","Wrong Entry Wardname","No Settlement name",
    "Settlement name with 1Chars","Settlement name with 2Chars",
    "Settlement name with more than 20Chars","Wrong Entry Population",
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
    "Wrong Entry Urban","Wrong Entry Rural","Wrong Entry Scattered"
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
    "Non-Numerical NonCompliant Household","Non-Numerical Team Code",
    "Wrong Entry Day of Activity Entry"
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

    _progress("Checking duplicates…", 2)

    dup_key = ["state_name","lga_name","ward_name","settlement_name"]
    mask_dup = df.duplicated(subset=dup_key, keep=False)
    df = flag_col(df, "attribute duplicate", mask_dup)

    _progress("Checking security_compromised…", 5)

    valid_sc = {"Y","N","NA"}
    sc_fixed, sc_wrong = zip(*df.apply(lambda r: _fix_sc(r.get("security_compromised",""), r.get("accessibility_status","")), axis=1))
    df["security_compromised"] = list(sc_fixed)
    df = flag_col(df, "Wrong Entry SecurityComp", pd.Series(sc_wrong))

    _progress("Checking accessibility_status…", 8)

    valid_acc = {"Fully Accessible","Inaccessible","Partially Accessible"}
    mask_acc = df["accessibility_status"].apply(lambda v: safe_str(v) not in valid_acc)
    df = flag_col(df, "Wrong Entry Accesibility_status", mask_acc)

    _progress("Checking reasons_for_inaccessibility…", 11)

    mask_rfi = df.apply(
        lambda r: safe_str(r.get("accessibility_status","")) != "Fully Accessible" and is_blank(r.get("reasons_for_inaccessibility","")),
        axis=1
    )
    df = flag_col(df, "Wrong Entry reasons_for_inaccessibility", mask_rfi)

    _progress("Checking habitational_status…", 13)

    valid_hab = {"Inhabited","Partially Inhabited","Abandoned","Migrated"}
    mask_hab = df["habitational_status"].apply(lambda v: safe_str(v) not in valid_hab)
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
        df = flag_col(df, "Settlement name with more than 20Chars", slen > 20)
    else:
        for f in ["Settlement name with 1Chars","Settlement name with 2Chars","Settlement name with more than 20Chars"]:
            df[f] = ""

    # Requirement 1: Create Program team verification column based on length criteria
    prog_mask = (df["Settlement name with 1Chars"] == "Y") | \
                (df["Settlement name with 2Chars"] == "Y") | \
                (df["Settlement name with more than 20Chars"] == "Y")
    df = flag_col(df, "Confirm with Programs team", prog_mask)

    _progress("Checking population fields…", 38)

    def _bad_pop(v):
        if is_blank(v):
            return True
        try:
            fv = float(str(v).strip())
            return fv <= 0
        except (ValueError, TypeError):
            return True

    df = flag_col(df, "Wrong Entry Population", df["set_population"].apply(_bad_pop) if "set_population" in df.columns else pd.Series([False]*n))
    df = flag_col(df, "Wrong Entry TargetPop", df["set_target"].apply(_bad_pop) if "set_target" in df.columns else pd.Series([False]*n))

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

    numeric_checks = [
        ("latitude","Non-Numerical Latitude"),
        ("longitude","Non-Numerical Longitude"),
        ("set_target","Non-Numerical Target Population"),
        ("set_population","Non-Numerical Sett Population"),
        ("number_of_household","Non-Numerical Number of Household"),
        ("noncompliant_household","Non-Numerical NonCompliant Household"),
        ("team_code","Non-Numerical Team Code"),
    ]
    for src_col, flag_name in numeric_checks:
        if src_col in df.columns:
            mask = df[src_col].apply(lambda v: not is_blank(v) and not _is_numeric_val(v))
            df = flag_col(df, flag_name, mask)
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
        df = flag_col(df, "Wrong Entry Day of Activity Entry", pd.Series(day_wrong))
    else:
        df["Wrong Entry Day of Activity Entry"] = ""

    return df

def _fix_sc(row_sc, row_ac):
    valid_sc = {"Y","N","NA"}
    v = safe_str(row_sc)
    if v in valid_sc:
        return v, False
    if is_blank(row_sc):
        if safe_str(row_ac) == "Fully Accessible":
            return "Y", False
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
        ward_file = st.file_uploader(
            "Ward Boundary",
            type=["zip","geojson","json"],
            key="ward_upload",
            label_visibility="collapsed",
        )

        st.markdown("<br><b style='color:#FFFFFF !important;'>STEP 3 · SETTLEMENT EXTENT</b>", unsafe_allow_html=True)
        st.caption("Grid3 Settlement Extent v3.1 (.zip or GeoJSON)")
        sett_file = st.file_uploader(
            "Grid3 Settlement Extent",
            type=["zip","geojson","json"],
            key="sett_upload",
            label_visibility="collapsed",
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        run_btn = st.button("▶  Run QA/QC", use_container_width=True, type="primary")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.7rem; color:rgba(255,255,255,0.45) !important; line-height:1.6;'>
        CRS: EPSG:4326 (WGS 84)<br>
        Spatial engine: GeoPandas<br>
        String match: RapidFuzz ≥95%<br>
        </div>
        """, unsafe_allow_html=True)

    return mlos_files, ward_file, sett_file, run_btn

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

    mlos_files, ward_file, sett_file, run_btn = render_sidebar()

    if not run_btn:
        if not mlos_files:
            st.markdown(f"""
            <div class='info-box'>
            👈  Upload your MLoS state files, ward boundary, and Grid3 settlement extent in the sidebar, then click <strong>Run QA/QC</strong>.
            <br><br>
            <strong>Supported formats:</strong> MLoS → CSV or Excel &nbsp;|&nbsp; Spatial → Shapefile (.zip) or GeoJSON
            </div>
            """, unsafe_allow_html=True)

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
        return

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

    st.markdown("<div class='section-title'>📋 Schema Validation</div>", unsafe_allow_html=True)
    schema_df = pd.DataFrame(schema_report)
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

    if dropped_states:
        dropped_html = "<br>".join([f"<b>{n}</b>: {r}" for n, r in dropped_states])
        st.markdown(f"<div class='dropped-box'>⛔ <b>Dropped files (schema mismatch):</b><br>{dropped_html}</div>", unsafe_allow_html=True)

    if not valid_frames:
        st.error("No valid MLoS files found. Check column names and order then re-upload.")
        prog_bar.progress(0)
        prog_text.empty()
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

    states_available = sorted(df_out["state_name"].dropna().unique().tolist()) if "state_name" in df_out.columns else []
    st.markdown("<div class='section-title'>🔎 Filter by State</div>", unsafe_allow_html=True)
    selected_states = st.multiselect(
        "Select states to drill down (leave empty for all)",
        options=states_available,
        default=[],
        key="state_filter",
    )
    df_view = df_out[df_out["state_name"].isin(selected_states)] if selected_states else df_out

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

    # Requirement 2: Dedicated isolation engine & download structure for Critical Flag items
    st.markdown("<div class='section-title'>🚨 Critical Errors Review Panel</div>", unsafe_allow_html=True)
    df_critical = df_view[df_view["Critical Error Flag"] == "Y"]
    
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

    st.markdown("<div class='section-title'>📋 State-level Summary Table</div>", unsafe_allow_html=True)
    summary_tbl = build_state_summary(df_view)
    if not summary_tbl.empty:
        st.dataframe(summary_tbl, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>🔬 Error Flag Breakdown</div>", unsafe_allow_html=True)
    flag_counts = {}
    for fc in OUTPUT_FLAG_COLUMNS:
        if fc in df_view.columns:
            cnt = (df_view[fc] == "Y").sum()
            if cnt > 0:
                flag_counts[fc] = cnt

    if flag_counts:
        fc_df = pd.DataFrame(list(flag_counts.items()), columns=["Flag","Count"]).sort_values("Count", ascending=False)
        fig_flags = px.bar(
            fc_df.head(25), x="Count", y="Flag", orientation="h",
            color="Count",
            color_continuous_scale=[[0, BLUE_PALE],[0.5, BLUE_LIGHT],[1, BLUE_DARK]],
            title="Top 25 Most Frequent Error Flags",
            labels={"Flag":"","Count":"Records Flagged"},
        )
        fig_flags.update_layout(
            height=520, coloraxis_showscale=False,
            margin=dict(l=10,r=20,t=50,b=10),
            paper_bgcolor="white", plot_bgcolor=GREY_LIGHT,
            title_font=dict(size=14, color=PURE_BLACK),
            xaxis=dict(tickfont=dict(color=PURE_BLACK)),
            yaxis=dict(tickfont=dict(size=10, color=PURE_BLACK)),
        )
        st.plotly_chart(fig_flags, use_container_width=True)

    st.markdown("<div class='section-title'>🗃️ Data Preview (first 500 rows)</div>", unsafe_allow_html=True)
    preview_cols = list(df_out.columns[:15]) + ["Critical Error Flag","Spatial Issues","Attribute Issues","No Issues","Confirm with Programs team"]
    preview_cols = [c for c in preview_cols if c in df_view.columns]
    st.dataframe(df_view[preview_cols].head(500), use_container_width=True, height=320)

    st.markdown("<div class='section-title'>⬇️ Download Output</div>", unsafe_allow_html=True)
    csv_bytes = to_csv_bytes(df_out)
    st.download_button(
        label=f"⬇️  Download Full QA/QC Output CSV  ({len(df_out):,} records)",
        data=csv_bytes,
        file_name="MLoS_QAQC_Output.csv",
        mime="text/csv",
    )
    if selected_states:
        csv_filtered = to_csv_bytes(df_view)
        st.download_button(
            label=f"⬇️  Download Filtered ({', '.join(selected_states)}) CSV  ({len(df_view):,} records)",
            data=csv_filtered,
            file_name=f"MLoS_QAQC_{'_'.join(selected_states)}.csv",
            mime="text/csv",
        )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align:center; font-size:0.72rem; color:#000000; padding:1rem;'>
    MLoS QA/QC Platform · EPSG:4326 · {total_raw:,} records ingested from {len(valid_frames)} state file(s)
    {f' · {len(dropped_states)} file(s) dropped' if dropped_states else ''}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
