"""Central configuration for NeuroSafe.

Everything that a student may want to change (drug list, date window, caps,
paths) lives here or in a local ``.env`` file (see ``.env.example``).
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# --------------------------------------------------------------------------- #
# Paths (NEUROSAFE_DATA_DIR / NEUROSAFE_MODELS_DIR allow redirecting storage,
# which is also how the smoke tests simulate an "empty project").
# --------------------------------------------------------------------------- #
DATA_DIR = Path(os.getenv("NEUROSAFE_DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.getenv("NEUROSAFE_MODELS_DIR", ROOT / "models"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = ROOT / "data" / "sample"

RAW_FILE = RAW_DIR / "faers_raw.jsonl"
RAW_META_FILE = RAW_DIR / "download_metadata.json"
REPORTS_FILE = PROCESSED_DIR / "reports.csv"
PREPROCESS_REPORT_FILE = PROCESSED_DIR / "preprocess_report.json"
MODEL_FILE = MODELS_DIR / "seriousness_model.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"
SAMPLE_FILE = SAMPLE_DIR / "sample_faers_synthetic.jsonl"

# --------------------------------------------------------------------------- #
# Data source
# --------------------------------------------------------------------------- #
OPENFDA_ENDPOINT = "https://api.fda.gov/drug/event.json"
OPENFDA_DOCS_URL = "https://open.fda.gov/apis/drug/event/"
FAERS_URL = (
    "https://www.fda.gov/drugs/drug-approvals-and-databases/"
    "fda-adverse-event-reporting-system-faers-database"
)
OPENFDA_LICENSE_URL = "https://open.fda.gov/license/"
OPENFDA_TERMS_URL = "https://open.fda.gov/terms/"
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "").strip() or None  # optional

START_DATE = os.getenv("NEUROSAFE_START_DATE", "2024-01-01")
END_DATE = os.getenv("NEUROSAFE_END_DATE", date.today().isoformat())
# Max reports fetched per (drug, calendar quarter) window. openFDA allows up to
# 1000 per request. The cap keeps the download small, but it also means raw
# counts per drug are NOT population volumes -- see README "Sampling".
MAX_REPORTS_PER_DRUG_WINDOW = int(os.getenv("NEUROSAFE_MAX_PER_WINDOW", "300"))

# Editable, NOT medically exhaustive.
SEDATIVE_DRUGS = [
    "DIAZEPAM",
    "LORAZEPAM",
    "ALPRAZOLAM",
    "CLONAZEPAM",
    "MIDAZOLAM",
    "TEMAZEPAM",
    "ZOLPIDEM",
    "DEXMEDETOMIDINE",
]

# --------------------------------------------------------------------------- #
# Cleaning / modelling constants
# --------------------------------------------------------------------------- #
MAX_VALID_AGE_YEARS = 120.0
AGE_BINS = [0, 18, 35, 50, 65, 80, 121]
AGE_LABELS = ["0-17", "18-34", "35-49", "50-64", "65-79", "80+"]
AGE_GROUP_ORDER = AGE_LABELS + ["Unknown"]

# Reaction terms that essentially restate the outcome that defines "Serious"
# (death / hospitalisation / disability / congenital anomaly). They are removed
# from the model's inputs to reduce target leakage. Editable.
LEAKY_REACTION_TERMS = {
    "DEATH",
    "SUDDEN DEATH",
    "HOSPITALISATION",
    "PROLONGED HOSPITALISATION",
    "DISABILITY",
    "CONGENITAL ANOMALY",
    "FOETAL DEATH",
    "STILLBIRTH",
}

RANDOM_STATE = 42
TEST_FRACTION = 0.20
VAL_FRACTION = 0.20
TOP_K_REACTIONS = 30
MIN_ROWS_FOR_ML = 300
MIN_MINORITY_CLASS = 30

# --------------------------------------------------------------------------- #
# Texts used across the app and the docs
# --------------------------------------------------------------------------- #
DISCLAIMER = (
    "This project is for educational and exploratory pharmacovigilance purposes only. "
    "It does not establish causality and must not be used for clinical decision-making."
)
DISCLAIMER_AR = (
    "هذا المشروع تعليمي واستكشافي فقط، ولا يثبت علاقة سببية، ولا يجوز استخدامه "
    "في اتخاذ أي قرار طبي أو علاجي."
)
CITATION = (
    "U.S. Food and Drug Administration. FDA Adverse Event Reporting System (FAERS), "
    "accessed through the openFDA Drug Adverse Event API (https://open.fda.gov/apis/drug/event/)."
)

# Data dictionary for the processed table (data/processed/reports.csv)
DATA_DICTIONARY = [
    ("report_id", "text", "openFDA `safetyreportid` - identifier of one safety report version."),
    ("case_id", "text", "Manufacturer case number (`companynumb`) when present, else `id:<report_id>`. Used to collapse follow-ups."),
    ("report_version", "int", "`safetyreportversion`; higher = later follow-up of the same report."),
    ("receive_date", "date", "Date FDA received the report (`receivedate`); parsed from YYYYMMDD."),
    ("report_year", "int", "Year of `receive_date`."),
    ("report_quarter", "text", "Calendar quarter label, e.g. `2025Q2`."),
    ("report_quarter_num", "int", "Quarter of the year (1-4); used as a seasonal feature."),
    ("report_month", "date", "First day of the report month (for monthly trends)."),
    ("serious_label", "text", "`Serious` / `Non-serious` / `Unknown` from the `serious` field (1 = serious, 2 = not serious)."),
    ("flag_death", "0/1", "`seriousnessdeath` flag. Outcome flag - NEVER used as a model feature."),
    ("flag_hospitalization", "0/1", "`seriousnesshospitalization`. Outcome flag - never a feature."),
    ("flag_lifethreatening", "0/1", "`seriousnesslifethreatening`. Outcome flag - never a feature."),
    ("flag_disabling", "0/1", "`seriousnessdisabling`. Outcome flag - never a feature."),
    ("flag_congenital", "0/1", "`seriousnesscongenitalanomali`. Outcome flag - never a feature."),
    ("flag_other", "0/1", "`seriousnessother`. Outcome flag - never a feature."),
    ("age_years", "float", "Patient age converted to years using the reported unit; NaN if missing or outside 0-120."),
    ("age_group", "text", "Binned age (0-17, 18-34, 35-49, 50-64, 65-79, 80+, Unknown)."),
    ("sex", "text", "Male / Female / Unknown (from `patientsex` 1 / 2 / other)."),
    ("country", "text", "`occurcountry` (ISO code where the event occurred); empty if missing."),
    ("selected_drugs", "text", "Pipe-separated list of study drugs (from `SEDATIVE_DRUGS`) found in the report's drug list."),
    ("selected_drug", "text", "Primary study drug: first one flagged 'suspect' by the reporter, else the first match."),
    ("selected_is_suspect", "0/1", "1 if any study drug was characterised as suspect by the reporter (unvalidated)."),
    ("n_selected_drugs", "int", "How many study drugs appear in the report."),
    ("n_drugs_total", "int", "Number of drug entries in the report (a polypharmacy proxy)."),
    ("reactions", "text", "Pipe-separated MedDRA preferred terms (upper-case, de-duplicated within the report)."),
    ("n_reactions", "int", "Number of distinct reported reaction terms."),
]
