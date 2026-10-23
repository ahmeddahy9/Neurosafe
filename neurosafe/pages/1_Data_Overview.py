import pandas as pd
import streamlit as st

from src import config, data_loader, preprocessing
from src.utils import data_banner, disclaimer_box, kpi_card, require_data, setup_page
from src.visualizations import CAPTIONS, missing_bar

setup_page("Data Overview", "🗂️")
disclaimer_box()
data_banner()
df = require_data()

c1, c2, c3, c4 = st.columns(4)
kpi_card(c1, "Rows (reports)", f"{len(df):,}")
kpi_card(c2, "Columns", f"{df.shape[1]}")
kpi_card(c3, "Date range", f"{df['receive_date'].min():%Y-%m} → {df['receive_date'].max():%Y-%m}")
kpi_card(c4, "Study drugs found", f"{df['selected_drugs'].str.split('|').explode().nunique()}")

st.markdown("### Pipeline audit")
audit = data_loader.read_json(config.PREPROCESS_REPORT_FILE)
if audit:
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Raw records", f"{audit['n_raw_records']:,}")
    a2.metric("No study drug match", f"{audit['n_no_study_drug_match']:,}")
    a3.metric("Removed: older versions", f"{audit['removed_same_report_id']:,}")
    a4.metric("Removed: same case id", f"{audit['removed_same_case_id']:,}")
    st.caption("De-duplication: the highest `report_version` per `report_id` is kept, then the most recently "
               "received report per manufacturer case number. Duplicates without a shared identifier cannot be detected.")
else:
    st.info("Audit file not found (run `scripts/preprocess_data.py`).")

meta = data_loader.data_source_info()
windows = pd.DataFrame(meta.get("windows", []))
if not windows.empty:
    st.markdown("### Sample size vs. matches reported by the API")
    summ = windows.groupby("drug").agg(api_total_matches=("api_total_matches", "sum"), fetched=("fetched", "sum")).reset_index()
    st.dataframe(summ, hide_index=True)
    st.caption("The download is capped per drug and quarter, so the sample is **not** proportional to real report volumes. "
               "`api_total_matches` is the count the API itself reports for each query.")

st.markdown("### Missing values")
miss = preprocessing.missing_value_summary(df)
col_a, col_b = st.columns([2, 3])
col_a.dataframe(miss, hide_index=True)
col_b.plotly_chart(missing_bar(miss))
st.caption(CAPTIONS["missing"] + " Missing values are never imputed in the dashboard; the model imputes the median age inside its training pipeline only.")

st.markdown("### Data dictionary")
st.dataframe(pd.DataFrame(config.DATA_DICTIONARY, columns=["column", "type", "description"]),
             hide_index=True)

st.markdown("### Preview")
st.dataframe(df.head(200))
st.caption("First 200 rows of the processed table.")
