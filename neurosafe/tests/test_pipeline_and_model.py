"""End-to-end check on the SYNTHETIC sample: pipeline runs, no leakage, honest fallbacks."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from make_sample_data import generate  # noqa: E402
from src import data_loader, modeling, preprocessing as pp  # noqa: E402


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    p = tmp_path_factory.mktemp("s") / "sample.jsonl"
    generate(p, n=900, seed=3)
    df, stats = pp.build_reports_table(data_loader.iter_raw_records(p))
    assert stats["removed_same_report_id"] > 0 and stats["removed_same_case_id"] > 0
    return df


def test_age_cleaning_removes_impossible_ages(reports):
    assert reports["age_years"].dropna().between(0, 120).all()


def test_split_has_no_case_overlap_and_is_chronological(reports):
    mf = __import__("src.features", fromlist=["x"]).prepare_model_frame(reports)
    train, val, test, train_full, strategy = modeling.make_splits(mf)
    assert "temporal" in strategy
    assert train_full["receive_date"].max() <= test["receive_date"].min()
    assert not set(train_full["case_id"]) & set(test["case_id"])


def test_training_reports_test_metrics_only(reports):
    metrics, model = modeling.train_and_evaluate(reports, {"is_sample": True})
    assert metrics["status"] == "ok" and model is not None
    assert metrics["data_is_sample"] is True
    assert set(metrics["test"]) == {"dummy_prior", "logistic_regression", "random_forest"}
    assert metrics["test"]["dummy_prior"]["n"] == metrics["n_test"]
    assert metrics["selected_model"] in ("logistic_regression", "random_forest")
    assert not {"flag_death", "serious_label", "target"} & set(metrics["features"])


def test_insufficient_data_is_reported_not_forced(reports):
    metrics, model = modeling.train_and_evaluate(reports.head(40))
    assert metrics["status"] == "insufficient_data" and model is None and "reason" in metrics
