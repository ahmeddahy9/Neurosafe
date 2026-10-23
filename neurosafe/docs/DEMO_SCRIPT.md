# NeuroSafe - 4-minute demo script

Prepare: real data downloaded, `streamlit run app.py` open on Home, browser zoom ~90 %.
(If you only have the synthetic sample, say so out loud on the first screen - the red banner will show it.)

| Time | Screen | What to say / do |
|---|---|---|
| 0:00-0:30 | **Home** | "Doctors and regulators collect side-effect *reports*. NeuroSafe explores public FDA reports for eight CNS sedatives to show how such data can be cleaned, explored and modelled - **educationally**." Point to the disclaimer: reports ≠ causes, no clinical use. |
| 0:30-1:00 | **Data Overview** | Data source: FAERS via openFDA, public-domain (CC0) per openFDA, downloaded on `<date>`, capped non-random sample. Show *Sample size vs API matches*: "so I never compare drug volumes". Show de-duplication numbers and the missing-values chart. |
| 1:00-2:15 | **Exploratory Analysis** | KPIs → top reactions → time trend ("recent quarters are incomplete because of reporting lag"). Open the **stacked Serious vs Non-serious** chart and the **drug × reaction heatmap**: "this compares *reporting patterns*, not safety". **Filter demo:** in the sidebar pick one drug, then Seriousness = Serious, then an age group - show KPIs and charts change; clear everything to show the friendly "no reports match" message. Search the table for a reaction and download the CSV. |
| 2:15-2:45 | **Drug Explorer** | Choose a drug: top reactions, trend, serious share. Read the blue note: "reflects reports submitted, not real-world risk". |
| 2:45-3:35 | **ML Risk Classification** | Task: predict if a *report* is flagged Serious. Safeguards: outcome flags removed, time-ordered split, selection on validation, test used once. Show the metrics table vs the dummy baseline, confusion matrix and ROC/PR. State the numbers you actually see (including if the model barely beats the baseline - that is an honest result). Show the educational form and its "Not for clinical use" warning. |
| 3:35-4:00 | **Methodology & Limitations** | Voluntary reporting, no denominator, duplicates, confounding, optimistic reaction-term leakage. "Real use needs pharmacovigilance experts and validation." Close with the disclaimer. |

Likely questions
* *Why not compute risk per drug?* No denominator; reports aren't exposure.
* *Why is the model not better?* Report fields carry limited information; reaction terms are the only strong signal and are partly post-event.
* *Why a capped sample?* API limits and runtime; the cap's effect is shown explicitly.
