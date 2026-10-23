import streamlit as st

from src import config, data_loader
from src.utils import data_banner, disclaimer_box, setup_page

setup_page("Methodology & Limitations", "📐")
disclaimer_box()
data_banner()

meta = data_loader.data_source_info()
audit = data_loader.read_json(config.PREPROCESS_REPORT_FILE) or {}

st.markdown("### 1. Data collection")
st.markdown(
    f"""
- **Source:** FDA FAERS through the openFDA Drug Adverse Event API ([docs]({config.OPENFDA_DOCS_URL})).
- **Query:** for each study drug and each calendar quarter, reports whose `patient.drug.openfda.generic_name`
  or `patient.drug.medicinalproduct` equals the drug name and whose `receivedate` falls in the quarter; up to
  **{meta.get('cap_per_drug_window', config.MAX_REPORTS_PER_DRUG_WINDOW)}** reports per drug-quarter.
- **Sampling caveat:** the cap makes this a *non-random, capped* sample. Per-drug counts are therefore not real
  volumes; the API's own match totals are shown on *Data Overview*.
- **Licence / reuse:** openFDA states that its data are public domain under a CC0 1.0 dedication
  ([license]({config.OPENFDA_LICENSE_URL}), [terms]({config.OPENFDA_TERMS_URL})). The openFDA service terms still apply
  (e.g., rate limits). The raw file is a *trimmed* copy of the fields used here.
"""
)

st.markdown("### 2. Cleaning")
st.markdown(
    f"""
- **Study-drug matching:** whole-word match on product / generic / substance names (e.g., "ZOLPIDEM TARTRATE" → ZOLPIDEM);
  reports with no match are dropped ({audit.get('n_no_study_drug_match', 'n/a')} in this run).
- **De-duplication:** keep the highest version per `report_id`, then the latest report per manufacturer case number
  (removed {audit.get('removed_same_report_id', 'n/a')} + {audit.get('removed_same_case_id', 'n/a')} rows). Duplicates with different identifiers remain undetected.
- **Age:** converted to years using the reported unit; values outside 0-120 years, negative, or without a unit become missing.
- **Sex / country / dates:** sex 1/2 → Male/Female, others Unknown; dates parsed from `YYYYMMDD`.
- **Reactions:** MedDRA preferred terms upper-cased and de-duplicated within a report.
- **Missing values** are shown, not invented. The model imputes the training median age *inside its pipeline*.
"""
)

st.markdown("### 3. Biases and data limitations")
st.markdown(
    """
- **Voluntary reporting:** under-reporting, stimulated reporting (publicity, litigation) and reporting bias.
- **No denominator:** there is no information on how many people take each drug, so **incidence or risk cannot be computed**.
- **No validated causality:** reports list *suspected* drugs; co-medications, disease and indication confound everything.
  When a report lists several drugs and reactions, no drug-reaction pairing is established.
- **Residual duplicates, missing fields, coding differences** (MedDRA terms vary by reporter).
- **Lag:** the newest quarters are incomplete; recent drops in counts may be data lag.
- **Geography:** FAERS over-represents the US.
"""
)

st.markdown("### 4. Machine-learning limits")
st.markdown(
    """
- The label is the **Serious** flag of a *report* (a regulatory category), not a clinical judgment.
- Outcome flags are excluded; outcome-like reaction terms are removed; a time-ordered split is used and model
  selection happens on a validation block so the test block is used only for reporting.
- Reaction terms still carry post-event information, so performance is optimistic for any forward-looking use.
- A model trained on a capped sample may not generalise to other drugs, periods or reporters.
- Scores are uncalibrated; they must not be read as probabilities for a patient.
"""
)

st.markdown("### 5. Privacy and ethics")
st.markdown(
    """
- Only public, aggregated and de-identified data are used; the app **collects and stores no personal health data**
  (the ML form is processed in memory and not saved).
- Any real use would need domain experts (pharmacovigilance, clinical pharmacology), additional validation
  and appropriate ethical / regulatory review.
- Language across the app deliberately says *reported events* and *reporting patterns*, never "this drug causes".
"""
)
