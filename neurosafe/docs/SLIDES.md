# NeuroSafe - 7-slide outline

**1. Title & problem**
- NeuroSafe: Educational Pharmacovigilance Dashboard for CNS Sedatives
- Team / course / supervisor (placeholders), disclaimer line
- Problem: side-effect reports are public but messy; how can students explore them responsibly?

**2. Data source**
- FDA FAERS via openFDA API; public-domain (CC0 per openFDA); download date
- 8 study drugs, quarterly windows, capped non-random sample
- Warnings: voluntary reporting, duplicates, missing data, no denominator

**3. Pipeline**
- Flatten → de-duplicate (report version, case id) → clean (age units, sex, dates, reactions) → missing-value audit
- Reproducible scripts + tests (show counts removed)

**4. Key EDA findings** (fill with YOUR real numbers)
- Reports per drug in the sample (with API totals for context)
- Top reported reactions; Serious vs Non-serious split
- Trend over time and its caveats (lag, cap)
- One screenshot of the heatmap with the phrase "reporting patterns, not risk"

**5. Interactive dashboard**
- Filters: drug, date, sex, age group, seriousness, country
- Screenshot of Exploratory Analysis + Drug Explorer
- Searchable table, CSV export, captions that state what charts cannot show

**6. Machine-learning prototype**
- Task: Serious vs Non-serious report; features list; leakage safeguards
- Time-ordered split, validation-based model selection, test used once
- Table: dummy vs logistic regression vs random forest (your real test metrics), confusion matrix
- Message: "Educational prototype - not for clinical use"

**7. Limitations, ethics & next steps**
- No causality, no incidence, optimistic reaction-term leakage, capped sample
- Privacy: public de-identified data, no user data stored
- Next: disproportionality analysis with experts, more drugs/years, calibration, external validation
- Closing disclaimer
