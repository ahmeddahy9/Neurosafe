"""Feature engineering helpers shared by the dashboard and the model."""
from __future__ import annotations

import pandas as pd

from . import config

NUMERIC_FEATURES = ["age_years", "n_reactions", "n_drugs_total", "n_selected_drugs"]
CATEGORICAL_FEATURES = ["sex", "country", "selected_drug", "quarter_of_year"]
TEXT_FEATURE = "reactions_model"
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]

# Columns that reveal the label and must never reach the model.
FORBIDDEN_FEATURES = [
    "serious_label", "target",
    "flag_death", "flag_hospitalization", "flag_lifethreatening",
    "flag_disabling", "flag_congenital", "flag_other",
]


def explode_drugs(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (report, study drug). A report with two study drugs appears twice."""
    if df.empty:
        return df.assign(drug=pd.Series(dtype=str))
    out = df.assign(drug=df["selected_drugs"].str.split("|")).explode("drug")
    return out[out["drug"].notna() & (out["drug"] != "")].reset_index(drop=True)


def explode_reactions(df: pd.DataFrame, col: str = "reactions") -> pd.DataFrame:
    """One row per (row, reaction term)."""
    if df.empty:
        return df.assign(reaction=pd.Series(dtype=str))
    out = df.assign(reaction=df[col].str.split("|")).explode("reaction")
    return out[out["reaction"].notna() & (out["reaction"] != "")].reset_index(drop=True)


def strip_leaky_terms(reactions: str, leaky: set[str] = config.LEAKY_REACTION_TERMS) -> str:
    """Remove outcome-like reaction terms from a pipe-separated string."""
    if not isinstance(reactions, str) or not reactions:
        return ""
    return "|".join(t for t in reactions.split("|") if t and t not in leaky)


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known label + model-ready columns.

    * keeps only `Serious` / `Non-serious` rows (Unknown label rows are not guessed);
    * removes outcome-like reaction terms and recomputes `n_reactions`;
    * uses the quarter *of the year* (seasonality) rather than the calendar year, because a
      temporal split would make the year an extrapolation the model never saw.
    """
    d = df[df["serious_label"].isin(["Serious", "Non-serious"])].copy()
    d["target"] = (d["serious_label"] == "Serious").astype(int)
    d[TEXT_FEATURE] = d["reactions"].fillna("").map(strip_leaky_terms)
    d["n_reactions"] = d[TEXT_FEATURE].map(lambda s: len([t for t in s.split("|") if t]))
    d["quarter_of_year"] = d["report_quarter_num"].map(lambda q: f"Q{int(q)}" if pd.notna(q) else "Unknown")
    d["country"] = d["country"].replace("", "Unknown").fillna("Unknown")
    d["sex"] = d["sex"].replace("", "Unknown").fillna("Unknown")
    return d.reset_index(drop=True)


def assert_no_leakage(feature_columns: list[str]) -> None:
    bad = sorted(set(feature_columns) & set(FORBIDDEN_FEATURES))
    if bad:
        raise ValueError(f"Target-leaking columns used as features: {bad}")
