import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data_loader, preprocessing as pp  # noqa: E402
from src.features import assert_no_leakage, prepare_model_frame, strip_leaky_terms, MODEL_FEATURES  # noqa: E402


# ---------------- age ----------------
@pytest.mark.parametrize("value,unit,expected", [
    ("45", "801", 45.0),
    ("7", "800", 70.0),                       # decades
    ("6", "802", 0.5),                        # months
    ("365.25", "804", 1.0),                   # days
])
def test_age_conversion_valid(value, unit, expected):
    assert pp.convert_age_to_years(value, unit) == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("value,unit", [
    ("150", "801"), ("-3", "801"), ("abc", "801"), (None, "801"), ("30", None), ("30", "999"), ("nan", "801"),
])
def test_age_conversion_invalid_is_nan(value, unit):
    assert np.isnan(pp.convert_age_to_years(value, unit))


def test_age_groups():
    assert pp.age_to_group(0) == "0-17"
    assert pp.age_to_group(17.9) == "0-17"
    assert pp.age_to_group(18) == "18-34"
    assert pp.age_to_group(80) == "80+"
    assert pp.age_to_group(np.nan) == "Unknown"
    assert pp.age_to_group(None) == "Unknown"


# ---------------- scalars ----------------
def test_sex_and_date():
    assert pp.normalize_sex("1") == "Male"
    assert pp.normalize_sex(2) == "Female"
    assert pp.normalize_sex(None) == "Unknown"
    assert pp.parse_receive_date("20250131") == pd.Timestamp("2025-01-31")
    assert pd.isna(pp.parse_receive_date("2025-99-99"))


# ---------------- drug matching ----------------
def _drug(name, char="2", generic=None):
    return {"medicinalproduct": name, "drugcharacterization": char,
            "openfda": {"generic_name": [generic or name]}}


def test_match_whole_word_and_primary_suspect():
    drugs = [_drug("IBUPROFEN"), _drug("LORAZEPAM", "2"), _drug("ZOLPIDEM TARTRATE", "1")]
    matched, primary, suspect = pp.match_selected_drugs(drugs, config.SEDATIVE_DRUGS)
    assert matched == ["LORAZEPAM", "ZOLPIDEM"]
    assert primary == "ZOLPIDEM" and suspect is True


def test_match_no_substring_false_positive():
    matched, primary, _ = pp.match_selected_drugs([_drug("MEDETOMIDINE")], ["DEXMEDETOMIDINE"])
    assert matched == [] and primary is None


# ---------------- flatten + dedup ----------------
def _record(rid, version="1", company="C1", date="20250110", serious="1", reactions=("NAUSEA",), drug="DIAZEPAM"):
    return {"safetyreportid": rid, "safetyreportversion": version, "companynumb": company,
            "receivedate": date, "serious": serious, "occurcountry": "us",
            "patient": {"patientonsetage": "40", "patientonsetageunit": "801", "patientsex": "2",
                        "reaction": [{"reactionmeddrapt": r} for r in reactions],
                        "drug": [_drug(drug, "1")]}}


def test_flatten_report_fields():
    row = pp.flatten_report(_record("R1", reactions=("Nausea", "nausea", "Fall")))
    assert row["reactions"] == "NAUSEA|FALL" and row["n_reactions"] == 2
    assert row["serious_label"] == "Serious" and row["country"] == "US"
    assert row["selected_drug"] == "DIAZEPAM" and row["age_years"] == 40.0


def test_flatten_returns_none_without_study_drug():
    assert pp.flatten_report(_record("R2", drug="ASPIRIN")) is None


def test_deduplication_keeps_latest_version_and_case():
    recs = [
        _record("R1", "1", "CASE1", "20250101"),
        _record("R1", "2", "CASE1", "20250201"),      # same report id, later version
        _record("R2", "1", "CASE1", "20250301"),      # same case, different report id, newer
        _record("R3", "1", "CASE2", "20250115"),
    ]
    df, stats = pp.build_reports_table(recs)
    assert len(df) == 2
    assert stats["removed_same_report_id"] == 1 and stats["removed_same_case_id"] == 1
    assert set(df["report_id"]) == {"R2", "R3"}
    assert df["report_id"].is_unique


def test_missing_summary_counts_unknown_and_empty():
    df = pd.DataFrame({"sex": ["Male", "Unknown", "Unknown", "Female"], "country": ["US", "", "", "DE"]})
    s = pp.missing_value_summary(df, ["sex", "country"]).set_index("column")
    assert s.loc["sex", "missing_pct"] == 50.0 and s.loc["country", "missing_n"] == 2


# ---------------- leakage guards ----------------
def test_leaky_terms_removed_and_features_clean():
    assert strip_leaky_terms("NAUSEA|DEATH|FALL|HOSPITALISATION") == "NAUSEA|FALL"
    assert_no_leakage(MODEL_FEATURES)
    with pytest.raises(ValueError):
        assert_no_leakage(MODEL_FEATURES + ["flag_death"])


def test_model_frame_drops_unknown_label():
    df, _ = pp.build_reports_table([_record("R1"), _record("R2", "1", "C2", serious="")])
    mf = prepare_model_frame(df)
    assert len(mf) == 1 and set(mf["target"]) == {1}


# ---------------- query building / download (mocked, no network) ----------------
def test_quarter_windows_cover_range():
    w = data_loader.quarter_windows("2024-02-15", "2024-11-10")
    assert w[0][0].isoformat() == "2024-02-15" and w[0][1].isoformat() == "2024-03-31"
    assert w[-1][1].isoformat() == "2024-11-10" and len(w) == 4


def test_build_search_query():
    from datetime import date
    q = data_loader.build_search_query("diazepam", date(2024, 1, 1), date(2024, 3, 31))
    assert 'generic_name:"DIAZEPAM"' in q and "receivedate:[20240101 TO 20240331]" in q


class _Resp:
    def __init__(self, code, payload=None):
        self.status_code, self._p, self.text = code, payload or {}, "x"

    def json(self):
        return self._p


class _Session:
    def __init__(self, resp):
        self.resp, self.calls = resp, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        return self.resp


def test_fetch_window_parses_total_and_404(monkeypatch):
    monkeypatch.setattr(data_loader.time, "sleep", lambda s: None)
    from datetime import date
    ok = _Session(_Resp(200, {"meta": {"last_updated": "2026-01-01", "results": {"total": 5000}},
                              "results": [{"safetyreportid": "1"}]}))
    recs, total, updated = data_loader.fetch_window(ok, "DIAZEPAM", date(2024, 1, 1), date(2024, 3, 31), limit=5000)
    assert total == 5000 and len(recs) == 1 and updated == "2026-01-01"
    assert ok.calls[0]["limit"] == 1000  # clamped to the API maximum
    none = _Session(_Resp(404))
    assert data_loader.fetch_window(none, "X", date(2024, 1, 1), date(2024, 3, 31))[:2] == ([], 0)
    bad = _Session(_Resp(500))
    with pytest.raises(data_loader.DownloadError):
        data_loader.fetch_window(bad, "X", date(2024, 1, 1), date(2024, 3, 31), max_retries=2)


def test_trim_record_keeps_only_needed_fields():
    rec = {"safetyreportid": "1", "sender": {"big": "x"}, "patient": {"patientsex": "1", "reaction": [{"reactionmeddrapt": "A", "reactionoutcome": "1"}],
                                                                       "drug": [{"medicinalproduct": "D", "drugindication": "secret-ish", "openfda": {"generic_name": ["D"], "brand_name": ["B"]}}]}}
    t = data_loader.trim_record(rec)
    assert "sender" not in t and "drugindication" not in json.dumps(t) and "brand_name" not in json.dumps(t)
