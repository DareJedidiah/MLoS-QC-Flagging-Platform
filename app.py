import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from fuzzywuzzy import fuzz
import io
import warnings
import re
warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MLoS QA/QC Platform",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --accent:    #f0a500;
    --accent2:   #3fb950;
    --danger:    #f85149;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --card:      #1c2128;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] label { color: var(--text) !important; }

/* Headers */
h1, h2, h3, h4 { font-family: 'Space Mono', monospace !important; color: var(--text) !important; }

/* Metric cards */
[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Space Mono', monospace !important; }
[data-testid="stMetricDelta"] { color: var(--danger) !important; }

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #ffc400 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(240,165,0,0.3) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 8px !important; }

/* Progress bar */
.stProgress > div > div { background: var(--accent) !important; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Tabs */
[data-baseweb="tab-list"] { background: var(--surface) !important; border-bottom: 1px solid var(--border) !important; }
[data-baseweb="tab"] { color: var(--muted) !important; font-family: 'Space Mono', monospace !important; font-size: 0.8rem !important; }
[aria-selected="true"][data-baseweb="tab"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }

/* Info/warning/error boxes */
[data-testid="stInfo"] { background: rgba(63,185,80,0.1) !important; border-left: 3px solid var(--accent2) !important; color: var(--text) !important; }
[data-testid="stWarning"] { background: rgba(240,165,0,0.1) !important; border-left: 3px solid var(--accent) !important; color: var(--text) !important; }
[data-testid="stError"] { background: rgba(248,81,73,0.1) !important; border-left: 3px solid var(--danger) !important; color: var(--text) !important; }
[data-testid="stSuccess"] { background: rgba(63,185,80,0.1) !important; border-left: 3px solid var(--accent2) !important; color: var(--text) !important; }

/* Select/input */
[data-baseweb="select"], [data-baseweb="input"] {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Custom badge */
.badge-error { display:inline-block; background:rgba(248,81,73,0.15); color:#f85149;
    border:1px solid rgba(248,81,73,0.3); border-radius:4px; padding:2px 8px;
    font-size:0.72rem; font-family:'Space Mono',monospace; }
.badge-ok { display:inline-block; background:rgba(63,185,80,0.15); color:#3fb950;
    border:1px solid rgba(63,185,80,0.3); border-radius:4px; padding:2px 8px;
    font-size:0.72rem; font-family:'Space Mono',monospace; }
.badge-warn { display:inline-block; background:rgba(240,165,0,0.15); color:#f0a500;
    border:1px solid rgba(240,165,0,0.3); border-radius:4px; padding:2px 8px;
    font-size:0.72rem; font-family:'Space Mono',monospace; }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Session state init ────────────────────────────────────────────────────────
for key in ["mlos_raw", "mlos_processed", "qaqc_done", "step"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "step" not in st.session_state or st.session_state["step"] is None:
    st.session_state["step"] = 0


# ─── Helper functions ─────────────────────────────────────────────────────────

def normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_")

def is_blank(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and np.isnan(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False

def yn_check(series: pd.Series) -> pd.Series:
    """Return True where value is NOT 'Y' or 'N' (case-insensitive) and not blank."""
    def bad(v):
        if is_blank(v):
            return False  # blanks handled separately
        return str(v).strip().upper() not in ("Y", "N")
    return series.apply(bad)

def load_mlos(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file, dtype=str, keep_default_na=False)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file, dtype=str, keep_default_na=False)
    else:
        raise ValueError("Unsupported MLoS format. Use CSV or Excel.")
    # Normalise column names
    df.columns = [c.strip() for c in df.columns]
    return df

def load_spatial(file):
    name = file.name.lower()
    suffix = "." + name.split(".")[-1]
    import tempfile, os, zipfile
    with tempfile.TemporaryDirectory() as tmp:
        if name.endswith(".zip"):
            with zipfile.ZipFile(file) as z:
                z.extractall(tmp)
            shps = [os.path.join(tmp, f) for f in os.listdir(tmp) if f.endswith(".shp")]
            if not shps:
                raise ValueError("No .shp found in zip.")
            return gpd.read_file(shps[0])
        else:
            path = os.path.join(tmp, file.name)
            with open(path, "wb") as f:
                f.write(file.getvalue())
            return gpd.read_file(path)


# ─── STEP 1: Ingestion & Cleaning ─────────────────────────────────────────────

TRIM_COLS = [
    "State_name","Lga_name","Ward_name","Settlement_name","Primary_settlement_name",
    "Alternate_name","Security_compromised","Accessibility_status",
    "Reasons_for_inaccessibility","Habitational_status","Urban","Rural","Scattered",
    "Highrisk","Slums","Densely_populated","Hard2reach","Border","Nomadic","Riverine",
    "Fulani","Source","Comments","Validation_status"
]

def step_ingest_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Case-insensitive column lookup helper
    col_map = {c.lower().replace(" ","_"): c for c in df.columns}

    def get_col(name):
        key = name.lower().replace(" ","_")
        return col_map.get(key)

    # 1. Delete unique_code if present
    uc = get_col("unique_code")
    if uc and uc in df.columns:
        df = df.drop(columns=[uc])

    # 2. TRIM columns
    for tc in TRIM_COLS:
        actual = get_col(tc)
        if actual and actual in df.columns:
            df[actual] = df[actual].astype(str).str.strip()
            df[actual] = df[actual].replace({"nan": "", "None": "", "none": ""})

    # 3. Recalculate unique_code
    def safe(col):
        c = get_col(col)
        return df[c].fillna("") if c else pd.Series([""] * len(df))

    df["unique_code"] = (
        safe("State_name").str.strip() + " " +
        safe("Lga_name").str.strip() + " " +
        safe("Ward_name").str.strip() + " " +
        safe("Settlement_name").str.strip()
    ).str.strip()

    # 4. Convert zero/missing lat-lon to NaN
    for coord in ["latitude", "longitude"]:
        c = get_col(coord)
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] == 0, c] = np.nan

    return df


# ─── STEP 2: Attribute QA/QC checks ──────────────────────────────────────────

def step_qaqc_attributes(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {c.lower().replace(" ","_"): c for c in df.columns}

    def get(name):
        return col_map.get(name.lower().replace(" ","_"))

    def col(name):
        c = get(name)
        return df[c] if c else pd.Series([""] * len(df), index=df.index)

    # 5. Attribute duplicates
    keys = [get("State_name"), get("Lga_name"), get("Ward_name"), get("Settlement_name")]
    keys = [k for k in keys if k]
    if keys:
        dup_mask = df.duplicated(subset=keys, keep=False)
        df["attribute_duplicate"] = np.where(dup_mask, "Y", "")

    # 6-21. Y/N field checks
    yn_fields = {
        "security_compromised": "Wrong Entry SecurityComp",
        "urban":               "Wrong Entry Urban",
        "rural":               "Wrong Entry Rural",
        "scattered":           "Wrong Entry Scattered",
        "highrisk":            "Wrong Entry Highrisk",
        "slums":               "Wrong Entry Slums",
        "densely_populated":   "Wrong Entry Densely_populated",
        "hard2reach":          "Wrong Entry Hard2reach",
        "border":              "Wrong Entry Border",
        "nomadic":             "Wrong Entry Nomadic",
        "riverine":            "Wrong Entry Riverine",
        "fulani":              "Wrong Entry Fulani",
    }
    for field, label in yn_fields.items():
        c = get(field)
        if c:
            bad = df[c].apply(lambda v: str(v).strip().upper() not in ("Y","N") and not is_blank(v))
            df[label] = np.where(bad, "Y", "")

    # Accessibility_status check
    c = get("accessibility_status")
    if c:
        valid = {"Fully Accessible","Inaccessible","Partially Accessible"}
        bad = df[c].apply(lambda v: str(v).strip() not in valid and not is_blank(v))
        df["Wrong Entry Accesibility_status"] = np.where(bad, "Y", "")

    # reasons_for_inaccessibility: flagged when accesibility_status != 'Fully Accessible' AND reasons is blank
    ac = get("accessibility_status")
    rc = get("reasons_for_inaccessibility")
    if ac and rc:
        mask = (df[ac].str.strip() != "Fully Accessible") & (df[rc].apply(is_blank))
        df["Wrong Entry reasons_for_inaccessibility"] = np.where(mask, "Y", "")

    # habitational_status
    c = get("habitational_status")
    if c:
        valid = {"Inhabited","Partially Inhabited","Abandoned","Migrated"}
        bad = df[c].apply(lambda v: str(v).strip() not in valid and not is_blank(v))
        df["Wrong Entry Habitational_status"] = np.where(bad, "Y", "")

    # source – flag if NOT empty/null (i.e. has unexpected content — per spec: flag non-empty)
    c = get("source")
    if c:
        bad = df[c].apply(lambda v: not is_blank(v) and str(v).strip().upper() not in ("MLOS","IE",""))
        df["Wrong Entry Source"] = np.where(bad, "Y", "")

    # validation_status
    c = get("validation_status")
    if c:
        valid = {"Validated","Validation Ongoing","Not Validated","Validated Unknown"}
        bad = df[c].apply(lambda v: str(v).strip() not in valid and not is_blank(v))
        df["Wrong Entry Validation status"] = np.where(bad, "Y", "")

    # population / target numeric & >0
    for field, label in [("set_population","Wrong Entry Population"),("set_target","Wrong Entry TargetPop")]:
        c = get(field)
        if c:
            def pop_bad(v):
                if is_blank(v): return False
                try:
                    n = float(v)
                    return not (n > 0)
                except:
                    return True
            df[label] = np.where(df[c].apply(pop_bad), "Y", "")

    # population conflict: set_target >= set_population  (exclude nulls/zeros)
    pc = get("set_population")
    tc = get("set_target")
    if pc and tc:
        def pop_conflict(row):
            pv = row[pc]; tv = row[tc]
            if is_blank(pv) or is_blank(tv): return ""
            try:
                p = float(pv); t = float(tv)
                if p == 0 or t == 0: return ""
                return "Y" if t >= p else ""
            except:
                return ""
        df["Population Conflict"] = df.apply(pop_conflict, axis=1)

    # name presence checks
    name_checks = {
        "state_name":      "No Statename",
        "lga_name":        "No LGAname",
        "ward_name":       "Wrong Entry Wardname",
        "settlement_name": "No Settlement name",
    }
    for field, label in name_checks.items():
        c = get(field)
        if c:
            df[label] = np.where(df[c].apply(is_blank), "Y", "")

    # settlement name length checks
    sc = get("settlement_name")
    if sc:
        lens = df[sc].str.len().fillna(0)
        df["Settlement name with 2Chars"]        = np.where(lens <= 2, "Y", "")
        df["Settlement name with 3Chars"]        = np.where(lens <= 3, "Y", "")
        df["Settlement name more than 10Chars"]  = np.where(lens > 10, "Y", "")

    # non-numeric checks
    numeric_fields = ["latitude","longitude","set_target","set_population",
                      "number_of_household","noncompliant_household","team_code"]
    for field in numeric_fields:
        c = get(field)
        if c:
            def non_num(v):
                if is_blank(v): return False
                try: float(v); return False
                except: return True
            label = f"Non-numeric {field}"
            df[label] = np.where(df[c].apply(non_num), "Y", "")

    # No geocoordinates
    latc = get("latitude"); lonc = get("longitude")
    if latc and lonc:
        no_geo = df[latc].isna() | df[lonc].isna()
        df["No Geocoordinates"] = np.where(no_geo, "Y", "N")

    return df


# ─── STEP 3: Spatial checks ───────────────────────────────────────────────────

def step_spatial_checks(df: pd.DataFrame, ward_gdf=None, grid3_gdf=None) -> pd.DataFrame:
    col_map = {c.lower().replace(" ","_"): c for c in df.columns}
    latc = col_map.get("latitude"); lonc = col_map.get("longitude")
    if not latc or not lonc:
        st.warning("Latitude/Longitude columns not found – skipping spatial checks.")
        return df

    has_coords = df["No Geocoordinates"] == "N"
    df_geo = df[has_coords].copy()

    if df_geo.empty:
        st.warning("No records with valid geocoordinates for spatial checks.")
        return df

    # Build GeoDataFrame
    geometry = [Point(float(row[lonc]), float(row[latc]))
                for _, row in df_geo.iterrows()]
    gdf = gpd.GeoDataFrame(df_geo, geometry=geometry, crs="EPSG:4326")

    # ── Grid3 settlement extent intersection ──
    if grid3_gdf is not None:
        try:
            grid3_proj = grid3_gdf.to_crs("EPSG:4326")
            joined = gpd.sjoin(gdf, grid3_proj[["geometry"]], how="left", predicate="within")
            intersects = ~joined["index_right"].isna()
            df.loc[has_coords, "Intersect Settlement Extent"] = np.where(intersects.values, "Y", "N")
            df.loc[~has_coords, "Intersect Settlement Extent"] = ""
        except Exception as e:
            st.warning(f"Grid3 intersection error: {e}")

    # ── Proximity checks (10m, 20m, 30m) ──
    try:
        gdf_metric = gdf.to_crs("EPSG:32632")  # UTM zone 32N (Nigeria)
        for radius in [10, 20, 30]:
            buffers = gdf_metric.copy()
            buffers["geometry"] = buffers.geometry.buffer(radius)
            # Spatial self-join to find overlaps
            joined_prox = gpd.sjoin(buffers[["geometry"]], buffers[["geometry"]],
                                    how="left", predicate="intersects")
            # Exclude self-intersections
            overlaps = joined_prox[joined_prox.index != joined_prox["index_right"]]
            overlap_idx = set(overlaps.index.tolist())
            col_label = f"{radius}m proximity"
            prox_vals = np.where(
                np.array([i in overlap_idx for i in range(len(gdf_metric))]),
                "Y", "N"
            )
            df.loc[has_coords, col_label] = prox_vals
            df.loc[~has_coords, col_label] = ""
    except Exception as e:
        st.warning(f"Proximity check error: {e}")

    # ── Ward boundary intersection ──
    if ward_gdf is not None:
        try:
            ward_proj = ward_gdf.to_crs("EPSG:4326")
            joined_ward = gpd.sjoin(gdf, ward_proj, how="left", predicate="within")
            inside = ~joined_ward["index_right"].isna()
            df.loc[has_coords, "Outside State Boundary"] = np.where(inside.values, "N", "Y")
            df.loc[~has_coords, "Outside State Boundary"] = ""

            # Ward name / LGA name match for intersecting records
            inside_idx = gdf.index[inside.values]
            joined_inside = joined_ward[inside].copy()

            col_map2 = {c.lower().replace(" ","_"): c for c in ward_gdf.columns}

            # Try to detect ward/lga columns in ward boundary file
            ward_col_candidates = [c for c in ward_gdf.columns if "ward" in c.lower()]
            lga_col_candidates  = [c for c in ward_gdf.columns if "lga" in c.lower()]

            mlos_ward = col_map.get("ward_name")
            mlos_lga  = col_map.get("lga_name")

            if ward_col_candidates and mlos_ward:
                wbc = ward_col_candidates[0]
                def ward_mismatch(row):
                    m = str(row.get(mlos_ward, "")).strip().upper()
                    b = str(row.get(wbc, "")).strip().upper()
                    if not m or not b: return ""
                    sim = fuzz.token_sort_ratio(m, b)
                    return "Y" if sim < 95 else "N"
                df.loc[has_coords, "Outside Ward"] = ""
                outside_ward = joined_ward.apply(ward_mismatch, axis=1)
                df.loc[has_coords, "Outside Ward"] = outside_ward.values

            if lga_col_candidates and mlos_lga:
                lbc = lga_col_candidates[0]
                def lga_mismatch(row):
                    m = str(row.get(mlos_lga, "")).strip().upper()
                    b = str(row.get(lbc, "")).strip().upper()
                    if not m or not b: return ""
                    sim = fuzz.token_sort_ratio(m, b)
                    return "Y" if sim < 95 else "N"
                outside_lga = joined_ward.apply(lga_mismatch, axis=1)
                df.loc[has_coords, "Outside LGA"] = outside_lga.values

        except Exception as e:
            st.warning(f"Ward boundary check error: {e}")

    return df


# ─── Summary builder ──────────────────────────────────────────────────────────

ERROR_COLS = [
    "attribute_duplicate","Wrong Entry SecurityComp","Wrong Entry Accesibility_status",
    "Wrong Entry reasons_for_inaccessibility","Wrong Entry Habitational_status",
    "Wrong Entry Urban","Wrong Entry Rural","Wrong Entry Scattered","Wrong Entry Source",
    "Wrong Entry Highrisk","Wrong Entry Slums","Wrong Entry Densely_populated",
    "Wrong Entry Hard2reach","Wrong Entry Border","Wrong Entry Nomadic",
    "Wrong Entry Riverine","Wrong Entry Fulani","Wrong Entry Population",
    "Wrong Entry TargetPop","Population Conflict","Wrong Entry Validation status",
    "No Statename","No LGAname","Wrong Entry Wardname","No Settlement name",
    "Settlement name with 2Chars","Settlement name with 3Chars",
    "Settlement name more than 10Chars","No Geocoordinates",
    "Intersect Settlement Extent","10m proximity","20m proximity","30m proximity",
    "Outside State Boundary","Outside Ward","Outside LGA",
]

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ec in ERROR_COLS:
        if ec in df.columns:
            count = (df[ec] == "Y").sum()
            rows.append({"Check": ec, "Flagged Records": int(count),
                         "% of Total": f"{100*count/max(len(df),1):.1f}%"})
    return pd.DataFrame(rows)


def to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="QA_QC_Output")
        summary = build_summary(df)
        summary.to_excel(writer, index=False, sheet_name="Summary")
    return buf.getvalue()


# ─── UI ───────────────────────────────────────────────────────────────────────

# SIDEBAR
with st.sidebar:
    st.markdown("## 🗺️ MLoS QA/QC")
    st.markdown('<p class="section-header">Data Inputs</p>', unsafe_allow_html=True)

    mlos_file   = st.file_uploader("Standardised MLoS Data", type=["csv","xlsx","xls"], key="mlos_up")
    ward_file   = st.file_uploader("Ward Boundary (SHP zip / GeoJSON)", type=["zip","geojson","json","shp"], key="ward_up")
    grid3_file  = st.file_uploader("Grid3 Settlement Extent (SHP zip / GeoJSON)", type=["zip","geojson","json","shp"], key="grid3_up")

    st.markdown("---")
    run_btn = st.button("▶  Run QA/QC Pipeline", use_container_width=True)
    st.markdown("---")
    st.markdown('<p class="section-header">Pipeline Steps</p>', unsafe_allow_html=True)
    steps = ["① Ingest & Clean", "② Attribute Checks", "③ Spatial Checks", "④ Results & Export"]
    step_now = st.session_state.get("step", 0)
    for i, s in enumerate(steps):
        color = "#f0a500" if i < step_now else ("#3fb950" if i == step_now-1 and st.session_state.get("qaqc_done") else "#8b949e")
        st.markdown(f'<span style="color:{color};font-family:Space Mono,monospace;font-size:0.8rem">{s}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("MLoS QA/QC Platform v1.0")
    st.caption("Supports CSV / XLSX input")

# MAIN AREA
st.markdown("# MLoS QA/QC Platform")
st.markdown("**Automated Quality Assurance & Quality Control for Master List of Settlements**")
st.markdown("---")

if not run_btn and st.session_state["qaqc_done"] is None:
    # Welcome screen
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Step 1** — Upload your standardised MLoS CSV or Excel file via the sidebar.")
    with c2:
        st.info("**Step 2** — Optionally upload Ward Boundary and Grid3 Settlement Extent for spatial checks.")
    with c3:
        st.info("**Step 3** — Click **Run QA/QC Pipeline** to execute all checks and download the flagged output.")
    st.markdown("""
    #### Checks performed
    | Category | Checks |
    |---|---|
    | Data Cleaning | TRIM, unique_code recalculation, null lat/lon |
    | Duplicates | Attribute duplicates (State/LGA/Ward/Settlement) |
    | Attribute Validation | 20+ field-level Y/N and categorical checks |
    | Name Checks | Missing names, character length flags |
    | Population | Numeric validity, population conflict |
    | Spatial | No-geocoordinates, Grid3 intersection, proximity (10/20/30m), ward boundary |
    """)
    st.stop()

if run_btn:
    if not mlos_file:
        st.error("Please upload an MLoS data file to proceed.")
        st.stop()

    progress = st.progress(0, text="Loading data…")
    status   = st.empty()

    # Load files
    try:
        df = load_mlos(mlos_file)
        status.success(f"✔ MLoS loaded — {len(df):,} records, {len(df.columns)} columns")
    except Exception as e:
        st.error(f"Failed to load MLoS: {e}")
        st.stop()

    ward_gdf  = None
    grid3_gdf = None

    if ward_file:
        try:
            ward_gdf = load_spatial(ward_file)
            status.success(f"✔ Ward boundary loaded — {len(ward_gdf)} features")
        except Exception as e:
            st.warning(f"Ward boundary load failed: {e}")

    if grid3_file:
        try:
            grid3_gdf = load_spatial(grid3_file)
            status.success(f"✔ Grid3 settlement extent loaded — {len(grid3_gdf)} features")
        except Exception as e:
            st.warning(f"Grid3 load failed: {e}")

    # Step 1: Ingest & Clean
    progress.progress(10, text="Step 1: Ingesting & cleaning data…")
    st.session_state["step"] = 1
    df = step_ingest_clean(df)
    progress.progress(35, text="Step 2: Running attribute QA/QC checks…")

    # Step 2: Attribute checks
    st.session_state["step"] = 2
    df = step_qaqc_attributes(df)
    progress.progress(65, text="Step 3: Running spatial checks…")

    # Step 3: Spatial checks
    st.session_state["step"] = 3
    if ward_gdf is not None or grid3_gdf is not None:
        df = step_spatial_checks(df, ward_gdf=ward_gdf, grid3_gdf=grid3_gdf)
    else:
        # Still do the no-geocoordinates flag
        df = step_spatial_checks(df, ward_gdf=None, grid3_gdf=None)

    progress.progress(90, text="Finalising output…")
    st.session_state["step"] = 4
    st.session_state["mlos_processed"] = df
    st.session_state["qaqc_done"] = True
    progress.progress(100, text="Complete!")
    status.success("✅ QA/QC pipeline complete!")

# ─── Results ──────────────────────────────────────────────────────────────────

if st.session_state.get("qaqc_done") and st.session_state["mlos_processed"] is not None:
    df = st.session_state["mlos_processed"]
    summary = build_summary(df)

    st.markdown("## Results")

    # Top metrics
    total      = len(df)
    total_errs = int(summary["Flagged Records"].sum())
    clean      = total - int((summary["Flagged Records"] > 0).sum())
    no_geo     = int((df.get("No Geocoordinates", pd.Series([])) == "Y").sum()) if "No Geocoordinates" in df.columns else 0
    dups       = int((df.get("attribute_duplicate", pd.Series([])) == "Y").sum()) if "attribute_duplicate" in df.columns else 0

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Total Records",    f"{total:,}")
    m2.metric("Total Flags",      f"{total_errs:,}")
    m3.metric("Duplicate Records", f"{dups:,}")
    m4.metric("No Geocoordinates", f"{no_geo:,}")
    m5.metric("Checks Run",       f"{len(summary):,}")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊  QA/QC Summary", "📋  Flagged Data", "💾  Export"])

    with tab1:
        st.markdown("### Check-by-Check Summary")
        if not summary.empty:
            # Colour-code
            def highlight(row):
                fr = row["Flagged Records"]
                if fr == 0:   return ["", "background-color:#0d2e18;color:#3fb950", ""]
                elif fr < 50: return ["", "background-color:#2e2200;color:#f0a500", ""]
                else:         return ["", "background-color:#2e0d0d;color:#f85149", ""]
            st.dataframe(
                summary.style.apply(highlight, axis=1),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("No error columns found in output.")

    with tab2:
        st.markdown("### Filtered View")
        filter_col = st.selectbox(
            "Filter by error flag",
            ["All Records"] + [c for c in ERROR_COLS if c in df.columns]
        )
        if filter_col == "All Records":
            view_df = df
        else:
            view_df = df[df[filter_col] == "Y"]

        st.markdown(f"**{len(view_df):,} records shown**")

        # Column selector - unique lists fix applied here
        raw_default_cols = [c for c in [
            "unique_code","State_name","Lga_name","Ward_name","Settlement_name",
            "latitude","longitude","No Geocoordinates","attribute_duplicate"
        ] + [c for c in ERROR_COLS if c in df.columns] if c in df.columns]
        
        unique_default_cols = list(dict.fromkeys(raw_default_cols))

        show_cols = st.multiselect(
            "Columns to display",
            options=list(df.columns),
            default=unique_default_cols[:20]
        )
        if show_cols:
            st.dataframe(view_df[show_cols], use_container_width=True, height=500)
        else:
            st.dataframe(view_df, use_container_width=True, height=500)

    with tab3:
        st.markdown("### Download Results")
        st.markdown("The export includes the full flagged dataset plus a summary sheet.")

        col_a, col_b = st.columns(2)
        with col_a:
            excel_bytes = to_excel(df)
            st.download_button(
                label="⬇  Download Excel (Full + Summary)",
                data=excel_bytes,
                file_name="MLoS_QAQC_Output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        with col_b:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇  Download CSV (Full Dataset)",
                data=csv_bytes,
                file_name="MLoS_QAQC_Output.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Download errors-only
        err_present = [c for c in ERROR_COLS if c in df.columns]
        if err_present:
            any_flag = df[err_present].apply(lambda r: (r == "Y").any(), axis=1)
            errors_df = df[any_flag]
            st.markdown(f"**Errors-only export:** {len(errors_df):,} records with at least one flag.")
            err_bytes = errors_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇  Download CSV (Flagged Records Only)",
                data=err_bytes,
                file_name="MLoS_QAQC_Errors_Only.csv",
                mime="text/csv",
                use_container_width=True
            )