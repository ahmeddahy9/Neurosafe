"""Educational ML prototype: classify a report as Serious vs Non-serious.

Design rules (see README, "Machine-learning model"):
  * outcome flags are never features (``features.FORBIDDEN_FEATURES``);
  * time-ordered split (older -> train, newest -> test), group-aware fallback;
  * model *selection* uses a validation block carved from the training period;
    the untouched test block is used once, for reporting;
  * preprocessing lives inside the sklearn Pipeline (fit on training data only).
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .features import (CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, TEXT_FEATURE,
                       assert_no_leakage, prepare_model_frame)

POSITIVE_LABEL = "Serious"


def split_reactions(text: str) -> list[str]:
    """Tokenizer for the pipe-separated reaction column (module level so it can be pickled)."""
    return [t for t in str(text).split("|") if t]


def build_preprocessor(top_k: int = config.TOP_K_REACTIONS) -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)
    reactions = CountVectorizer(tokenizer=split_reactions, token_pattern=None, lowercase=False,
                                binary=True, max_features=top_k)
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
        ("rx", reactions, TEXT_FEATURE),
    ])


def build_models() -> dict[str, Pipeline]:
    """Baseline dummy, Logistic Regression baseline, Random Forest comparison model."""
    def pipe(clf):
        return Pipeline([("prep", build_preprocessor()), ("clf", clf)])
    return {
        "dummy_prior": pipe(DummyClassifier(strategy="prior")),
        "logistic_regression": pipe(LogisticRegression(class_weight="balanced", max_iter=2000)),
        "random_forest": pipe(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=config.RANDOM_STATE)),
    }


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def temporal_split(df: pd.DataFrame, test_frac: float, date_col: str = "receive_date"):
    """Older rows -> train, newest ``test_frac`` -> test. ``None`` if dates are unusable."""
    d = df.dropna(subset=[date_col]).sort_values(date_col)
    if len(d) < 50 or d[date_col].nunique() < 10:
        return None
    cut = d[date_col].quantile(1 - test_frac)
    train, test = d[d[date_col] < cut], d[d[date_col] >= cut]
    if train.empty or test.empty:
        return None
    return train, test


def group_split(df: pd.DataFrame, test_frac: float, group_col: str = "case_id"):
    """Fallback: random split that keeps every case in exactly one side."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=config.RANDOM_STATE)
    tr, te = next(gss.split(df, groups=df[group_col]))
    return df.iloc[tr], df.iloc[te]


def make_splits(df: pd.DataFrame):
    """Return (train, val, test, strategy)."""
    outer = temporal_split(df, config.TEST_FRACTION)
    strategy = "temporal (oldest -> train, newest -> test)"
    if outer is None:
        outer = group_split(df, config.TEST_FRACTION)
        strategy = "group-aware random split by case_id (dates unusable)"
    train_full, test = outer
    inner = temporal_split(train_full, config.VAL_FRACTION) if "temporal" in strategy else None
    if inner is None:
        inner = group_split(train_full, config.VAL_FRACTION)
    train, val = inner
    assert not set(train_full["case_id"]) & set(test["case_id"]), "case leakage between train and test"
    return train, val, test, train_full, strategy


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _downsample(arr: np.ndarray, n: int = 120) -> list[float]:
    if len(arr) <= n:
        return [float(x) for x in arr]
    idx = np.linspace(0, len(arr) - 1, n).astype(int)
    return [float(arr[i]) for i in idx]


def evaluate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, with_curves: bool = False) -> dict:
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if y.nunique() > 1 else None,
        "pr_auc": float(average_precision_score(y, proba)) if y.nunique() > 1 else None,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "positive_rate": float(y.mean()),
    }
    if with_curves and y.nunique() > 1:
        fpr, tpr, _ = roc_curve(y, proba)
        prec, rec, _ = precision_recall_curve(y, proba)
        out["roc_curve"] = {"fpr": _downsample(fpr), "tpr": _downsample(tpr)}
        out["pr_curve"] = {"precision": _downsample(prec), "recall": _downsample(rec)}
    return out


def feature_importance(pipe: Pipeline, top: int = 15) -> list[dict]:
    """Coefficients (LR) or impurity importances (RF). Associations, not causes."""
    names = [n.split("__", 1)[1] if "__" in n else n for n in pipe.named_steps["prep"].get_feature_names_out()]
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        vals = clf.coef_[0]
        kind = "coefficient (log-odds, standardised inputs)"
    elif hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
        kind = "impurity importance"
    else:
        return []
    order = np.argsort(-np.abs(vals))[:top]
    return [{"feature": names[i], "value": float(vals[i]), "kind": kind} for i in order]


def insufficient_reason(model_df: pd.DataFrame) -> str | None:
    if len(model_df) < config.MIN_ROWS_FOR_ML:
        return f"only {len(model_df)} labelled reports (need >= {config.MIN_ROWS_FOR_ML})"
    counts = model_df["target"].value_counts()
    if counts.size < 2 or counts.min() < config.MIN_MINORITY_CLASS:
        return f"minority class has {int(counts.min()) if counts.size else 0} reports (need >= {config.MIN_MINORITY_CLASS})"
    return None


# --------------------------------------------------------------------------- #
# Training driver
# --------------------------------------------------------------------------- #
def train_and_evaluate(reports: pd.DataFrame, data_source: dict | None = None) -> tuple[dict, Pipeline | None]:
    """Train, select on validation, report on test. Returns ``(metrics_dict, fitted_pipeline)``."""
    assert_no_leakage(MODEL_FEATURES)
    data_source = data_source or {}
    model_df = prepare_model_frame(reports)
    base_info = {
        "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "data_is_sample": bool(data_source.get("is_sample", False)),
        "n_labelled_reports": int(len(model_df)),
        "features": MODEL_FEATURES,
        "excluded_leaky_terms": sorted(config.LEAKY_REACTION_TERMS),
        "label": "Serious (1) vs Non-serious (0), from the `serious` field",
    }
    reason = insufficient_reason(model_df)
    if reason:
        return {**base_info, "status": "insufficient_data", "reason": reason}, None

    train, val, test, train_full, strategy = make_splits(model_df)
    if train["target"].nunique() < 2 or val["target"].nunique() < 2 or test["target"].nunique() < 2:
        return {**base_info, "status": "insufficient_data",
                "reason": "a split contains a single class; collect more data"}, None

    X = lambda d: d[MODEL_FEATURES]  # noqa: E731
    models = build_models()

    # 1) selection on validation (fit on train only)
    val_scores = {}
    for name, pipe in models.items():
        pipe.fit(X(train), train["target"])
        val_scores[name] = evaluate(pipe, X(val), val["target"])
    candidates = [n for n in models if n != "dummy_prior"]
    selected = max(candidates, key=lambda n: val_scores[n]["pr_auc"] or 0.0)

    # 2) refit everything on the full training period, evaluate once on test
    test_scores, final_pipes = {}, build_models()
    for name, pipe in final_pipes.items():
        pipe.fit(X(train_full), train_full["target"])
        test_scores[name] = evaluate(pipe, X(test), test["target"], with_curves=(name == selected))
    final = final_pipes[selected]
    beats = (test_scores[selected]["pr_auc"] or 0) > (test_scores["dummy_prior"]["pr_auc"] or 0) + 0.02

    def rng(d):
        return [str(d["receive_date"].min().date()), str(d["receive_date"].max().date())] if d["receive_date"].notna().any() else None

    top_rx = (model_df[TEXT_FEATURE].str.split("|").explode().replace("", np.nan).dropna()
              .value_counts().head(config.TOP_K_REACTIONS).index.tolist())
    metrics = {
        **base_info,
        "status": "ok",
        "split_strategy": strategy,
        "n_train": int(len(train_full)), "n_train_inner": int(len(train)),
        "n_val": int(len(val)), "n_test": int(len(test)),
        "train_date_range": rng(train_full), "test_date_range": rng(test),
        "class_balance_train": float(train_full["target"].mean()),
        "class_balance_test": float(test["target"].mean()),
        "selection_metric": "PR-AUC on the validation block (carved from the training period)",
        "validation": val_scores,
        "test": test_scores,
        "selected_model": selected,
        "beats_baseline": bool(beats),
        "feature_importance": feature_importance(final),
        "form_options": {
            "drugs": sorted(model_df["selected_drug"].dropna().unique().tolist()),
            "sexes": sorted(model_df["sex"].unique().tolist()),
            "countries": model_df["country"].value_counts().head(25).index.tolist(),
            "quarters": ["Q1", "Q2", "Q3", "Q4"],
            "top_reactions": top_rx,
            "age_median": float(model_df["age_years"].median()) if model_df["age_years"].notna().any() else None,
        },
    }
    return metrics, final
