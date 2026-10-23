"""Cleaning and flattening of openFDA drug-event records.

Every function here is pure (no Streamlit, no network) so it can be unit tested.
"""
from __future__ import annotations

import re
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from . import config

# openFDA `patientonsetageunit` codes -> years per unit
AGE_UNIT_TO_YEARS = {
    "800": 10.0,            # decade
    "801": 1.0,             # year
    "802": 1.0 / 12.0,      # month
    "803": 7.0 / 365.25,    # week
    "804": 1.0 / 365.25,    # day
    "805": 1.0 / 8766.0,    # hour
}

FLAG_FIELDS = {
    "flag_death": "seriousnessdeath",
    "flag_hospitalization": "seriousnesshospitalization",
    "flag_lifethreatening": "seriousnesslifethreatening",
    "flag_disabling": "seriousnessdisabling",
    "flag_congenital": "seriousnesscongenitalanomali",
    "flag_other": "seriousnessother",
}

REPORT_COLUMNS = [
    "report_id", "case_id", "report_version", "receive_date", "serious_label",
    *FLAG_FIELDS.keys(),
    "age_years", "sex", "country", "selected_drugs", "selected_drug",
    "selected_is_suspect", "n_selected_drugs", "n_drugs_total", "reactions", "n_reactions",
]


# --------------------------------------------------------------------------- #
# Scalar helpers
# --------------------------------------------------------------------------- #
def convert_age_to_years(value, unit, max_age: float = config.MAX_VALID_AGE_YEARS) -> float:
    """Convert a reported age + openFDA unit code to years.

    Returns NaN when the value/unit is missing, non-numeric, negative or the
    converted age is outside ``[0, max_age]``. Nothing is guessed: a missing
    unit is treated as unreliable and yields NaN.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(v) or v < 0:
        return np.nan
    if unit is None:
        return np.nan
    unit_key = str(unit).strip().split(".")[0]
    factor = AGE_UNIT_TO_YEARS.get(unit_key)
    if factor is None:
        return np.nan
    years = v * factor
    return years if 0 <= years <= max_age else np.nan


def age_to_group(age_years) -> str:
    """Bin an age in years into the groups defined in ``config.AGE_LABELS``."""
    if age_years is None or (isinstance(age_years, float) and np.isnan(age_years)):
        return "Unknown"
    try:
        a = float(age_years)
    except (TypeError, ValueError):
        return "Unknown"
    if not np.isfinite(a) or a < 0:
        return "Unknown"
    for lo, hi, label in zip(config.AGE_BINS[:-1], config.AGE_BINS[1:], config.AGE_LABELS):
        if lo <= a < hi:
            return label
    return "Unknown"


def normalize_sex(code) -> str:
    """openFDA `patientsex`: 1 = male, 2 = female, anything else = Unknown."""
    key = str(code).strip().split(".")[0] if code is not None else ""
    return {"1": "Male", "2": "Female"}.get(key, "Unknown")


def parse_receive_date(value) -> pd.Timestamp:
    """Parse openFDA `YYYYMMDD` strings; invalid -> NaT."""
    if value is None:
        return pd.NaT
    return pd.to_datetime(str(value).strip(), format="%Y%m%d", errors="coerce")


def normalize_drug_name(name) -> str:
    """Upper-case, collapse whitespace, strip trailing punctuation."""
    if name is None:
        return ""
    text = re.sub(r"\s+", " ", str(name)).strip().upper()
    return text.strip(" .,;:-")


def normalize_reaction(term) -> str:
    return re.sub(r"\s+", " ", str(term)).strip().upper().replace("|", "/")


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _drug_text(entry: dict) -> str:
    """Searchable text of one drug entry: product name + generic + substance."""
    ofda = entry.get("openfda") or {}
    parts = [entry.get("medicinalproduct") or ""]
    for key in ("generic_name", "substance_name"):
        parts.extend(str(x) for x in _as_list(ofda.get(key)))
    return normalize_drug_name(" | ".join(parts))


def match_selected_drugs(drugs: Iterable[dict], selected: Iterable[str]) -> tuple[list[str], str | None, bool]:
    """Find study drugs in a report's drug list.

    Returns ``(matched_names, primary_drug, any_suspect)``. Matching is by whole
    word on the normalised name, so "ZOLPIDEM TARTRATE" matches "ZOLPIDEM".
    The primary drug is the first match the reporter flagged as suspect
    (`drugcharacterization == 1`), otherwise the first match. Suspect status is
    a reporter's unvalidated opinion.
    """
    patterns = {name: re.compile(rf"\b{re.escape(normalize_drug_name(name))}\b") for name in selected}
    matched: list[str] = []
    suspect_matched: list[str] = []
    for entry in drugs or []:
        text = _drug_text(entry)
        is_suspect = str(entry.get("drugcharacterization", "")).strip() == "1"
        for name, pattern in patterns.items():
            if pattern.search(text):
                if name not in matched:
                    matched.append(name)
                if is_suspect and name not in suspect_matched:
                    suspect_matched.append(name)
    if not matched:
        return [], None, False
    primary = suspect_matched[0] if suspect_matched else matched[0]
    return matched, primary, bool(suspect_matched)


# --------------------------------------------------------------------------- #
# Record -> row
# --------------------------------------------------------------------------- #
def flatten_report(record: dict, selected: Iterable[str] = config.SEDATIVE_DRUGS) -> dict | None:
    """Flatten one openFDA record into one row; ``None`` if no study drug is present."""
    patient = record.get("patient") or {}
    drugs = patient.get("drug") or []
    matched, primary, any_suspect = match_selected_drugs(drugs, selected)
    if not matched:
        return None

    report_id = str(record.get("safetyreportid", "")).strip()
    company_no = str(record.get("companynumb") or "").strip()
    reactions: list[str] = []
    for r in patient.get("reaction") or []:
        term = normalize_reaction(r.get("reactionmeddrapt", ""))
        if term and term not in reactions:
            reactions.append(term)

    serious_raw = str(record.get("serious", "")).strip().split(".")[0]
    serious_label = {"1": "Serious", "2": "Non-serious"}.get(serious_raw, "Unknown")

    try:
        version = int(float(record.get("safetyreportversion", 1)))
    except (TypeError, ValueError):
        version = 1

    row = {
        "report_id": report_id,
        "case_id": company_no if company_no else f"id:{report_id}",
        "report_version": version,
        "receive_date": parse_receive_date(record.get("receivedate")),
        "serious_label": serious_label,
        "age_years": convert_age_to_years(patient.get("patientonsetage"), patient.get("patientonsetageunit")),
        "sex": normalize_sex(patient.get("patientsex")),
        "country": str(record.get("occurcountry") or "").strip().upper(),
        "selected_drugs": "|".join(matched),
        "selected_drug": primary,
        "selected_is_suspect": int(any_suspect),
        "n_selected_drugs": len(matched),
        "n_drugs_total": len(drugs),
        "reactions": "|".join(reactions),
        "n_reactions": len(reactions),
    }
    for col, field in FLAG_FIELDS.items():
        row[col] = 1 if str(record.get(field, "")).strip().split(".")[0] == "1" else 0
    return row


def iter_rows(records: Iterable[dict], selected: Iterable[str] = config.SEDATIVE_DRUGS) -> Iterator[dict]:
    for rec in records:
        row = flatten_report(rec, selected)
        if row is not None:
            yield row


# --------------------------------------------------------------------------- #
# De-duplication
# --------------------------------------------------------------------------- #
def deduplicate_reports(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Collapse duplicate/follow-up reports.

    Strategy (documented in the README):
      1. Same ``report_id`` -> keep the highest ``report_version``.
      2. Same ``case_id`` (manufacturer case number) -> keep the most recently
         received version.
    Reports sharing no identifier cannot be detected as duplicates; residual
    duplication is a known limitation.
    """
    stats = {"rows_before": int(len(df))}
    if df.empty:
        stats.update(removed_same_report_id=0, removed_same_case_id=0, rows_after=0)
        return df, stats

    d = df.sort_values(["report_id", "report_version", "receive_date"], na_position="first")
    d1 = d.drop_duplicates(subset="report_id", keep="last")
    stats["removed_same_report_id"] = int(len(d) - len(d1))

    d2 = d1.sort_values(["case_id", "receive_date", "report_version"], na_position="first")
    d2 = d2.drop_duplicates(subset="case_id", keep="last")
    stats["removed_same_case_id"] = int(len(d1) - len(d2))

    d2 = d2.sort_values(["receive_date", "report_id"], na_position="last").reset_index(drop=True)
    stats["rows_after"] = int(len(d2))
    return d2, stats


# --------------------------------------------------------------------------- #
# Table builder
# --------------------------------------------------------------------------- #
def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["receive_date"] = pd.to_datetime(d["receive_date"], errors="coerce")
    d["report_year"] = d["receive_date"].dt.year.astype("Int64")
    d["report_quarter_num"] = d["receive_date"].dt.quarter.astype("Int64")
    d["report_quarter"] = d["receive_date"].dt.to_period("Q").astype(str).where(d["receive_date"].notna(), "")
    d["report_month"] = d["receive_date"].dt.to_period("M").dt.to_timestamp()
    d["age_group"] = d["age_years"].map(age_to_group)
    return d


def build_reports_table(records: Iterable[dict], selected: Iterable[str] = config.SEDATIVE_DRUGS) -> tuple[pd.DataFrame, dict]:
    """Records -> clean one-row-per-report DataFrame plus an audit trail."""
    selected = list(selected)
    n_raw = 0
    rows = []
    for rec in records:
        n_raw += 1
        row = flatten_report(rec, selected)
        if row is not None:
            rows.append(row)
    stats: dict = {"n_raw_records": n_raw, "n_no_study_drug_match": n_raw - len(rows)}
    if not rows:
        empty = pd.DataFrame(columns=REPORT_COLUMNS)
        stats.update(rows_before=0, rows_after=0, removed_same_report_id=0, removed_same_case_id=0)
        return add_derived_columns(empty.assign(age_years=np.nan)), stats

    df = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    df, dedup_stats = deduplicate_reports(df)
    stats.update(dedup_stats)
    df = add_derived_columns(df)
    stats["n_missing_receive_date"] = int(df["receive_date"].isna().sum())
    stats["n_age_invalid_or_missing"] = int(df["age_years"].isna().sum())
    return df, stats


# --------------------------------------------------------------------------- #
# Missing-value summary
# --------------------------------------------------------------------------- #
IMPORTANT_COLUMNS = [
    "receive_date", "serious_label", "age_years", "sex", "country",
    "selected_drug", "reactions",
]


def _missing_mask(series: pd.Series) -> pd.Series:
    m = series.isna()
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        s = series.astype("string").str.strip()
        m = m | s.isin(["", "Unknown"]).fillna(False)
    return m


def missing_value_summary(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """% missing per column; empty strings and the literal 'Unknown' count as missing."""
    cols = [c for c in (columns or IMPORTANT_COLUMNS) if c in df.columns]
    n = max(len(df), 1)
    out = pd.DataFrame(
        {"column": cols, "missing_n": [int(_missing_mask(df[c]).sum()) for c in cols]}
    )
    out["missing_pct"] = (out["missing_n"] / n * 100).round(2)
    return out.sort_values("missing_pct", ascending=False).reset_index(drop=True)
