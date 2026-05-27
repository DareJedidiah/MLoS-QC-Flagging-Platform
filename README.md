# MLoS QA/QC Streamlit Platform — PRD & Technical Reference

## Overview
A Streamlit-based web application that ingests Master List of Settlements (MLoS) data and performs
automated Quality Assurance / Quality Control (QA/QC) checks across attribute validation, data
integrity, and spatial analysis.

---

## Architecture

```
mlos_qaqc/
├── app.py            ← Single-file Streamlit application
├── requirements.txt  ← Python dependencies
└── README.md         ← This document
```

---

## Pipeline Steps

### Step 0 — File Upload (Sidebar)
| Input | Format |
|---|---|
| Standardised MLoS | CSV or Excel (.csv, .xlsx, .xls) |
| Ward Boundary | Shapefile (zip), GeoJSON |
| Grid3 Settlement Extent | Shapefile (zip), GeoJSON |

Only the MLoS file is mandatory. Spatial checks are skipped gracefully if spatial files are absent.

---

### Step 1 — Ingest & Clean
1. Parse MLoS (CSV or Excel) with all columns read as strings.
2. **Delete** the `unique_code` column.
3. **TRIM** whitespace from 24 text columns.
4. **Recalculate** `unique_code` = `state_name + lga_name + ward_name + settlement_name`.
5. **Null-coerce** zero and missing latitude/longitude values to `NaN`.

---

### Step 2 — Attribute QA/QC Checks

| Flag Column | Logic |
|---|---|
| `attribute_duplicate` | Same state/lga/ward/settlement combination → Y |
| `Wrong Entry SecurityComp` | `security_compromised` ∉ {Y, N} |
| `Wrong Entry Accesibility_status` | Not in {Fully Accessible, Inaccessible, Partially Accessible} |
| `Wrong Entry reasons_for_inaccessibility` | accessibility ≠ Fully Accessible AND reasons is blank |
| `Wrong Entry Habitational_status` | Not in {Inhabited, Partially Inhabited, Abandoned, Migrated} |
| `Wrong Entry Urban/Rural/Scattered/Highrisk/Slums/…` | Not in {Y, N} (12 fields) |
| `Wrong Entry Source` | Source is not a known value (MLoS, IE, blank) |
| `Wrong Entry Validation status` | Not in {Validated, Validation Ongoing, Not Validated, Validated Unknown} |
| `Wrong Entry Population` | `set_population` is non-numeric or ≤ 0 |
| `Wrong Entry TargetPop` | `set_target` is non-numeric or ≤ 0 |
| `Population Conflict` | `set_target` ≥ `set_population` (non-zero, non-null) |
| `No Statename / No LGAname / Wrong Entry Wardname / No Settlement name` | Blank name fields |
| `Settlement name with 2Chars` | len(settlement_name) ≤ 2 |
| `Settlement name with 3Chars` | len(settlement_name) ≤ 3 |
| `Settlement name more than 10Chars` | len(settlement_name) > 10 |
| `Non-numeric {field}` | 7 fields checked for non-numeric content |
| `No Geocoordinates` | latitude or longitude is null |

---

### Step 3 — Spatial Checks

| Flag Column | Logic |
|---|---|
| `Intersect Settlement Extent` | Point within Grid3 settlement polygon → Y |
| `10m / 20m / 30m proximity` | Overlapping buffers at respective radius → Y |
| `Outside State Boundary` | Point does not intersect any ward boundary feature → Y |
| `Outside Ward` | MLoS ward name vs ward boundary ward name similarity < 95% (fuzzy) |
| `Outside LGA` | MLoS LGA name vs boundary LGA name similarity < 95% (fuzzy) |

Spatial projection: UTM Zone 32N (EPSG:32632) for metric operations (Nigeria coverage).

---

### Step 4 — Results & Export
- Summary table: every check, count of flagged records, % of total.
- Interactive filtered view: filter data by any error flag.
- Column selector for custom display.
- Export options:
  - Excel (full dataset + summary sheet)
  - CSV (full dataset)
  - CSV (flagged records only)

---

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)
1. Push repository to GitHub.
2. Go to share.streamlit.io → New app → select repo/branch/app.py.
3. No additional secrets required.

---

## Column Name Tolerance
All column lookups are case-insensitive and underscore/space-normalised.
The app will find `Accessibility_Status`, `accessibility_status`, `ACCESSIBILITY STATUS`, etc.

---

## Notes
- Fuzzy ward/LGA matching uses `fuzzywuzzy.token_sort_ratio` at 95% threshold.
- Proximity buffers use UTM Zone 32N for metre-accurate buffering across Nigeria.
- The app handles missing spatial files gracefully — attribute checks still run.
