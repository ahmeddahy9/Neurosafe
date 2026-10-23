# NeuroSafe - Educational Pharmacovigilance Dashboard for CNS Sedatives

> **Disclaimer.** This project is for educational and exploratory pharmacovigilance purposes only. It does not establish causality and must not be used for clinical decision-making.

NeuroSafe is a Streamlit app that downloads a documented sample of official FDA adverse-event
*reports* for selected CNS sedatives / sedative-hypnotics, cleans them in an auditable pipeline,
lets you explore **reporting patterns** interactively, and trains a small, honestly evaluated
*educational prototype* that predicts whether a report is flagged **Serious**.

*(نبذة بالعربية: مشروع تعليمي لتحليل بلاغات الآثار الجانبية للمهدئات من بيانات FDA العامة. النتائج تصف "بلاغات" وليست مخاطر حقيقية، ولا تصلح لأي قرار علاجي.)*

---

## 1. Data source, license and terms

| Item | Details |
|---|---|
| Source | FDA Adverse Event Reporting System (**FAERS**), accessed through the **openFDA Drug Adverse Event API** |
| Official pages | [FAERS](https://www.fda.gov/drugs/drug-approvals-and-databases/fda-adverse-event-reporting-system-faers-database) · [openFDA drug event API](https://open.fda.gov/apis/drug/event/) |
| Endpoint | `https://api.fda.gov/drug/event.json` |
| Download date | Recorded automatically in `data/raw/download_metadata.json` (`download_date_utc`) and shown in the app |
| Query | For every study drug and calendar quarter: `(patient.drug.openfda.generic_name:"<DRUG>" OR patient.drug.medicinalproduct:"<DRUG>") AND receivedate:[<start> TO <end>]`, up to `NEUROSAFE_MAX_PER_WINDOW` (default 300) reports per drug-quarter |
| License / reuse | openFDA states that its data are public domain under a **CC0 1.0** dedication ([license](https://open.fda.gov/license/)); the openFDA service [terms](https://open.fda.gov/terms/) (rate limits etc.) still apply. Re-verify the pages above before redistributing. |
| API key | Optional (raises rate limits). Put it in `.env` as `OPENFDA_API_KEY`; never commit it. |

**Data warnings.** FAERS is a voluntary reporting system. Reports can be duplicated, incomplete,
unverified and biased; a report does not prove that a drug caused an event; there is no denominator
(number of users), so **incidence or "real risk" cannot be computed**. openFDA data lag the source by
about three months or more.

**Sampling.** The download is *capped and non-random* (first N reports returned per drug-quarter).
Per-drug counts in the dashboard therefore are **not** real volumes. The API's own `total` per query
is stored and shown on the *Data Overview* page for comparison.

**Offline sample.** `python scripts/download_data.py --use-sample` copies a small **synthetic**
openFDA-shaped file (`data/sample/`) for UI/pipeline testing when the API is unreachable. It is random,
has no real signal, and the app shows a red banner whenever it is in use. Never present it as an analysis.

---

## 2. Quick start

Requires **Python 3.10+** (developed on 3.13).

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
python scripts/download_data.py      # official data (internet needed)  |  --use-sample for offline UI testing
python scripts/preprocess_data.py    # clean -> data/processed/reports.csv
python scripts/train_model.py        # train + evaluate -> models/
streamlit run app.py
```

Run the tests: `pytest -q` (unit tests, pipeline/model checks and a headless smoke test of every page, with and without data).

### Reproducing results
1. Keep the same `NEUROSAFE_START_DATE`, `NEUROSAFE_END_DATE`, drug list and cap (all recorded in `data/raw/download_metadata.json`).
2. openFDA is updated continuously, so a later download may differ slightly; keep `data/raw/` if you need exact reproducibility.
3. `RANDOM_STATE = 42` fixes the model's randomness.

### Changing drugs or the period
* Drugs: edit `SEDATIVE_DRUGS` in `src/config.py` (or `python scripts/download_data.py --drugs DIAZEPAM ZOLPIDEM`).
* Period / cap: `.env` (`NEUROSAFE_START_DATE`, `NEUROSAFE_END_DATE`, `NEUROSAFE_MAX_PER_WINDOW`) or CLI flags `--start --end --cap`.
* After changing anything, re-run the three scripts.
* `python scripts/download_data.py --dry-run` prints the queries without calling the API.

### Where files are written
| Path | Content | In Git? |
|---|---|---|
| `data/raw/faers_raw.jsonl`, `download_metadata.json` | trimmed raw records + provenance | no |
| `data/processed/reports.csv`, `preprocess_report.json` | clean table + cleaning audit | no |
| `models/seriousness_model.joblib`, `metrics.json` | fitted pipeline + test metrics | no |
| `data/sample/` | tiny SYNTHETIC test file | yes |

---

## 3. Project layout

```text
neurosafe/
├── app.py                       # Home (disclaimer, goals, citation)
├── pages/
│   ├── 1_Data_Overview.py       # shape, audit, data dictionary, missing values, preview
│   ├── 2_Exploratory_Analysis.py# KPIs, charts, filters, searchable table + CSV download
│   ├── 3_Drug_Explorer.py       # per-drug reactions, trend, serious share
│   ├── 4_ML_Risk_Classification.py
│   ├── 5_Methodology_and_Limitations.py
│   └── 6_About_Team.py          # placeholders to edit
├── src/                         # logic (no Streamlit except utils.py)
│   ├── config.py  data_loader.py  preprocessing.py  features.py
│   └── modeling.py  visualizations.py  utils.py
├── scripts/                     # download_data.py  preprocess_data.py  train_model.py  make_sample_data.py
├── tests/                       # preprocessing, pipeline/model, page smoke tests
├── notebooks/01_eda.ipynb
├── data/{raw,processed,sample}  models/  docs/
├── requirements.txt  .env.example  .gitignore  .streamlit/config.toml
```

---

## 4. Data pipeline

1. **Flatten** each openFDA record to one row; keep only reports whose drug list contains a study drug (whole-word match on product/generic/substance name).
2. **De-duplicate**: highest `report_version` per `report_id`, then latest report per manufacturer case number (`companynumb`). The counts removed are stored in `preprocess_report.json` and shown in the app. Duplicates without a shared identifier remain undetected.
3. **Clean**: names upper-cased; age converted to years from its unit and set to NaN if missing/negative/>120; sex 1/2 → Male/Female else Unknown; dates parsed; reaction terms normalised and de-duplicated per report.
4. **Derive**: `report_year`, `report_quarter`, `age_group`, `selected_drug`, `n_reactions`, `n_drugs_total`, ...
5. **Audit**: missing-value table/chart; nothing is invented.

The full column list is the data dictionary on the *Data Overview* page (`config.DATA_DICTIONARY`).

---

## 5. Machine-learning model (educational prototype)

* **Task:** classify a de-duplicated report as `Serious` vs `Non-serious` (from the `serious` field). Reports with unknown label are excluded, not guessed.
* **Leakage controls:** outcome flags (death, hospitalisation, life-threatening, disabling, congenital, other) are *never* features; outcome-like reaction terms (e.g. DEATH, HOSPITALISATION) are removed; preprocessing sits inside a scikit-learn `Pipeline`; the calendar *year* is not used (quarter-of-year only) because a time-ordered split would make it an extrapolation.
* **Split:** oldest 80 % → training period, newest 20 % → test. A validation block (newest 20 % of the training period) is used to **select** between models, so the test block is touched only for reporting. Group-aware fallback by `case_id` if dates are unusable.
* **Models:** `DummyClassifier(prior)` baseline, `LogisticRegression(class_weight="balanced")`, `RandomForest(class_weight="balanced_subsample")`. Selection metric: validation PR-AUC.
* **Metrics shown (test only):** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, ROC/PR curves, baseline comparison, top coefficients/importances (associations, not causes).
* **Honesty guard:** if there are fewer than 300 labelled reports or fewer than 30 in the minority class, no model is trained and the ML page explains why.
* **Residual leakage:** reaction terms are partly post-event information (severity of a listed reaction correlates with seriousness), so reported performance is optimistic for any real forward-looking use.

---

## 6. Privacy and ethics

* Only public, de-identified, aggregated data; the app collects and stores **no personal health data** (the ML form is evaluated in memory).
* FAERS limitations: under-reporting, reporting bias, duplicates, missing data, confounding.
* An association between a drug and a reported event does not prove causation.
* Results are not medical advice or treatment guidance.
* Any real use needs domain experts, further validation and ethical/regulatory review.

---

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `download_data.py` prints HTTP 429 / retries | Add an `OPENFDA_API_KEY` to `.env`, lower `--cap`, or wait; the script backs off automatically. Without a key openFDA allows 240 requests/min and 1,000/day per IP. |
| "Failed after 5 attempts" / no internet | Check network/proxy; use `--use-sample` for offline UI testing only. |
| A window returns 0 matches | Normal for very recent quarters (openFDA lags). HTTP 404 from openFDA means "no match". |
| App says "No processed data found" | Run the three scripts in order. |
| ML page says insufficient data | Increase `--cap`, widen dates or add drugs, then run `train_model.py` again. |
| `ModuleNotFoundError: src` | Run commands from the project root (`neurosafe/`). |
| Page shows old data after re-running scripts | Refresh the browser (`R`); cached loaders watch file modification time. |

---

## 8. Status and honesty notes

* All unit, pipeline and page smoke tests pass on the synthetic sample. The live openFDA downloader was written against the documented API but could **not** be exercised against `api.fda.gov` from the authoring environment (network blocked); its logic is covered by mocked tests. Run `python scripts/download_data.py` once on your machine and check the printed per-window `matches=` / `fetched=` lines.
* No metrics in this repository come from real FAERS data until you run the pipeline on a real download. Metrics on the sample are meaningless by construction.

## 9. Presentation material
`docs/DEMO_SCRIPT.md` (3-5 minute demo) and `docs/SLIDES.md` (7-slide outline).
