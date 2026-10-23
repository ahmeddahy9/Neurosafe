import pandas as pd
import streamlit as st

from src import config
from src import visualizations as viz
from src.features import MODEL_FEATURES, strip_leaky_terms
from src.utils import data_banner, disclaimer_box, get_metrics, get_model, setup_page

setup_page("ML Risk Classification", "🤖")
disclaimer_box()
st.error("**Educational Prototype - Not for clinical use.** The model predicts whether a *report* is likely to be "
         "flagged Serious. It does not predict risk to any patient and must never inform treatment.")
data_banner()

metrics = get_metrics()
if metrics is None:
    st.warning("No model results yet. Run `python scripts/train_model.py` (after downloading and preprocessing).")
    st.stop()

if metrics.get("data_is_sample"):
    st.warning("The numbers below were produced from SYNTHETIC data and carry no meaning.")

if metrics["status"] != "ok":
    st.warning(f"A model was **not** trained: {metrics['reason']}. Forcing a model on this data would give "
               "misleading metrics, so the project stays an exploratory dashboard. Download more data "
               "(`--cap`, wider date window, more drugs) and re-run training.")
    st.stop()

# ---- objective & features -----------------------------------------------------
st.markdown("### Objective")
st.markdown(
    f"""
Binary classification of each de-duplicated report: **{metrics['label']}**.
Data are split **{metrics['split_strategy']}**: train {metrics['n_train']:,} reports
({metrics['train_date_range'][0]} → {metrics['train_date_range'][1]}), test {metrics['n_test']:,}
({metrics['test_date_range'][0]} → {metrics['test_date_range'][1]}). All metrics below are from the
**test block**; the model was *selected* on a validation block taken from the training period
({metrics['selection_metric']}).
"""
)
with st.expander("Features used and leakage safeguards"):
    st.markdown(
        "- **Used:** " + ", ".join(f"`{f}`" for f in metrics["features"]) + "\n"
        "- **Never used:** the outcome flags (death, hospitalisation, life-threatening, disabling, congenital, other) "
        "and the label itself.\n"
        "- **Reaction terms** are limited to the 30 most frequent in the data; outcome-like terms "
        "(" + ", ".join(metrics["excluded_leaky_terms"]) + ") are removed before training.\n"
        "- **Residual leakage risk:** how severe a listed reaction is correlates with the Serious flag, so "
        "reaction terms are partly *post-event* information. Performance is therefore optimistic for any real forward-looking use.\n"
        "- Preprocessing (imputation, scaling, encoding, vocabulary) is fitted inside a scikit-learn Pipeline on training data only."
    )

# ---- metrics ------------------------------------------------------------------
st.markdown("### Test-set metrics")
names = {"dummy_prior": "Baseline: Dummy (prior)", "logistic_regression": "Logistic Regression",
         "random_forest": "Random Forest"}
rows = []
for k, m in metrics["test"].items():
    rows.append({"model": names.get(k, k) + ("  ✅ selected" if k == metrics["selected_model"] else ""),
                 "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
                 "F1": m["f1"], "ROC-AUC": m["roc_auc"], "PR-AUC": m["pr_auc"]})
tbl = pd.DataFrame(rows)
st.dataframe(tbl.style.format({c: "{:.3f}" for c in tbl.columns if c != "model"}, na_rep="n/a"),
             hide_index=True)
base_rate = metrics["class_balance_test"]
st.caption(f"Positive class = Serious ({base_rate:.1%} of test reports). Accuracy alone is misleading with imbalanced classes: "
           "a model that always says 'Serious' gets that share. Compare ROC-AUC (0.5 = chance) and PR-AUC "
           f"(chance ≈ {base_rate:.2f}) against the baseline.")

if metrics["beats_baseline"]:
    st.success("The selected model beats the dummy baseline on test PR-AUC by a margin above 0.02. "
               "This is a modest, educational result on a capped sample, not evidence of clinical usefulness.")
else:
    st.warning("The selected model does **not** clearly beat the dummy baseline. Reported data may simply not contain "
               "enough signal for this task; treat the predictor below as a demonstration of the workflow only.")

sel = metrics["test"][metrics["selected_model"]]
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(viz.confusion_matrix_fig(sel["confusion_matrix"]))
    st.caption("Rows = true class, columns = predicted class, on the held-out newest reports.")
with c2:
    st.plotly_chart(viz.roc_pr_fig(sel.get("roc_curve"), sel.get("pr_curve"), base_rate))
    st.caption("Curves from test-set probabilities of the selected model.")

st.markdown("### What the model associates with the Serious flag")
st.plotly_chart(viz.importance_bar(metrics["feature_importance"]))
st.caption("These are **associations** learned from reporting data, not causes. Reaction names appear because they are "
           "inputs; they do not mean the drug produces them.")

# ---- educational prediction form ----------------------------------------------
st.markdown("### Try it (educational)")
model_bundle = get_model()
if model_bundle is None:
    st.info("Model file not found; run `python scripts/train_model.py`.")
    st.stop()
opts = metrics["form_options"]
with st.form("predict"):
    a, b, c = st.columns(3)
    drug = a.selectbox("Study drug", opts["drugs"])
    sex = b.selectbox("Sex", opts["sexes"])
    country = c.selectbox("Country", ["Unknown"] + [x for x in opts["countries"] if x != "Unknown"])
    d, e, f = st.columns(3)
    known_age = d.checkbox("Age is known", value=True)
    age = d.number_input("Age (years)", 0, 120, int(opts.get("age_median") or 50))
    quarter = e.selectbox("Quarter of the year the report was received", opts["quarters"])
    n_drugs = f.number_input("Total drugs listed in the report", 1, 30, 2)
    rx_sel = st.multiselect("Reaction terms listed (common terms only)", opts["top_reactions"])
    go_btn = st.form_submit_button("Estimate")

if go_btn:
    rx_text = strip_leaky_terms("|".join(rx_sel))
    row = pd.DataFrame([{
        "age_years": float(age) if known_age else float("nan"),
        "n_reactions": len([t for t in rx_text.split("|") if t]),
        "n_drugs_total": int(n_drugs), "n_selected_drugs": 1,
        "sex": sex, "country": country, "selected_drug": drug, "quarter_of_year": quarter,
        "reactions_model": rx_text,
    }])[MODEL_FEATURES]
    p = float(model_bundle["pipeline"].predict_proba(row)[0, 1])
    st.metric("Model score: probability this *report* is flagged Serious", f"{p:.0%}")
    st.progress(min(max(p, 0.0), 1.0))
    st.warning("Not for clinical use. This number is an uncalibrated model score on reporting data; it says nothing "
               "about any patient, any drug's safety, or what should be done clinically.")
