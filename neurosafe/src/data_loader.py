"""Data access: openFDA download helpers + readers for the local files.

No Streamlit imports here. Network functions take an explicit ``session`` so
they can be mocked in tests.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd
import requests

from . import config


class DownloadError(RuntimeError):
    """Raised when openFDA cannot be reached after retries."""


# --------------------------------------------------------------------------- #
# openFDA query construction
# --------------------------------------------------------------------------- #
def quarter_windows(start: str | date, end: str | date) -> list[tuple[date, date]]:
    """Split [start, end] into calendar-quarter windows (first/last clipped)."""
    s = date.fromisoformat(str(start)) if not isinstance(start, date) else start
    e = date.fromisoformat(str(end)) if not isinstance(end, date) else end
    if s > e:
        raise ValueError("start date is after end date")
    windows: list[tuple[date, date]] = []
    cur = s
    while cur <= e:
        q_end_month = ((cur.month - 1) // 3 + 1) * 3
        next_q_start = date(cur.year + (q_end_month == 12), (q_end_month % 12) + 1, 1)
        q_end = next_q_start - timedelta(days=1)
        windows.append((cur, min(q_end, e)))
        cur = next_q_start
    return windows


def build_search_query(drug: str, start: date, end: date) -> str:
    """openFDA `search` expression for one drug and one receive-date window."""
    name = drug.strip().upper().replace('"', "")
    return (
        f'(patient.drug.openfda.generic_name:"{name}" OR patient.drug.medicinalproduct:"{name}")'
        f" AND receivedate:[{start:%Y%m%d} TO {end:%Y%m%d}]"
    )


_KEEP_TOP = ("safetyreportid", "safetyreportversion", "companynumb", "receivedate",
             "serious", "seriousnessdeath", "seriousnesshospitalization",
             "seriousnesslifethreatening", "seriousnessdisabling",
             "seriousnesscongenitalanomali", "seriousnessother", "occurcountry")


def trim_record(rec: dict) -> dict:
    """Keep only the fields NeuroSafe uses (the stored 'raw' file is a trimmed copy)."""
    out = {k: rec.get(k) for k in _KEEP_TOP if k in rec}
    p = rec.get("patient") or {}
    out["patient"] = {
        "patientonsetage": p.get("patientonsetage"),
        "patientonsetageunit": p.get("patientonsetageunit"),
        "patientsex": p.get("patientsex"),
        "reaction": [{"reactionmeddrapt": r.get("reactionmeddrapt")} for r in p.get("reaction") or []],
        "drug": [
            {
                "medicinalproduct": d.get("medicinalproduct"),
                "drugcharacterization": d.get("drugcharacterization"),
                "openfda": {
                    "generic_name": (d.get("openfda") or {}).get("generic_name"),
                    "substance_name": (d.get("openfda") or {}).get("substance_name"),
                },
            }
            for d in p.get("drug") or []
        ],
    }
    return out


def fetch_window(
    session: requests.Session,
    drug: str,
    start: date,
    end: date,
    limit: int = config.MAX_REPORTS_PER_DRUG_WINDOW,
    api_key: str | None = config.OPENFDA_API_KEY,
    max_retries: int = 5,
    endpoint: str = config.OPENFDA_ENDPOINT,
    pause: float = 0.3,
) -> tuple[list[dict], int, str | None]:
    """Fetch up to ``limit`` reports for one drug/window.

    Returns ``(records, total_matches_reported_by_api, api_last_updated)``.
    HTTP 404 means "no match" for openFDA and is returned as an empty result.
    """
    params = {"search": build_search_query(drug, start, end), "limit": min(int(limit), 1000)}
    if api_key:
        params["api_key"] = api_key

    last_err = ""
    for attempt in range(max_retries):
        try:
            resp = session.get(endpoint, params=params, timeout=60)
        except requests.RequestException as exc:  # network error
            last_err = str(exc)
            time.sleep(min(2 ** attempt, 30))
            continue
        if resp.status_code == 404:
            return [], 0, None
        if resp.status_code == 429 or resp.status_code >= 500:
            last_err = f"HTTP {resp.status_code}"
            time.sleep(min(2 ** attempt * 2, 60))
            continue
        if resp.status_code != 200:
            raise DownloadError(f"HTTP {resp.status_code} for {drug} {start}..{end}: {resp.text[:200]}")
        payload = resp.json()
        meta = (payload.get("meta") or {})
        total = int(((meta.get("results") or {}).get("total")) or 0)
        time.sleep(pause)
        return payload.get("results") or [], total, meta.get("last_updated")
    raise DownloadError(f"Failed after {max_retries} attempts for {drug} {start}..{end}: {last_err}")


# --------------------------------------------------------------------------- #
# Local readers
# --------------------------------------------------------------------------- #
def read_json(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_json(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def iter_raw_records(path: Path = config.RAW_FILE) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_reports(path: Path = config.REPORTS_FILE) -> pd.DataFrame | None:
    """Load the processed table; ``None`` if it has not been built yet."""
    path = Path(path)
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["receive_date", "report_month"], keep_default_na=False,
                     na_values=[""], dtype={"report_id": str, "case_id": str})
    for col in ("sex", "country", "age_group", "serious_label", "selected_drug",
                "selected_drugs", "reactions", "report_quarter"):
        if col in df.columns:
            df[col] = df[col].fillna("")
    df["country"] = df["country"].replace("", "Unknown")
    return df


def data_source_info() -> dict:
    """Metadata of the last download (empty dict if none)."""
    return read_json(config.RAW_META_FILE) or {}


def is_sample_data() -> bool:
    return bool(data_source_info().get("is_sample", False))


def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
