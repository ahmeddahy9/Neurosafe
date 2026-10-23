import pandas as pd
import streamlit as st

from src.features import explode_reactions
from src.utils import data_banner, disclaimer_box, kpi_card, pct, require_data, setup_page, sidebar_filters
from src import visualizations as viz
from src.visualizations import CAPTIONS

setup_page("Exploratory Analysis", "📊")
disclaimer_box()
data_banner()
df = require_data()
f = sidebar_filters(df)
reports, pairs = f.reports, f.pairs

# ---- KPIs -----------------------------------------------------------------
known = reports[reports["serious_label"].isin(["Serious", "Non-serious"])]
serious_pct = (known["serious_label"].eq("Serious").mean() * 100) if len(known) else float("nan")
drug_counts = pairs["drug"].value_counts()
k1, k2, k3, k4 = st.columns(4)
kpi_card(k1, "Reports (selection)", f"{len(reports):,}")
kpi_card(k2, "Study drugs", f"{pairs['drug'].nunique()}")
kpi_card(k3, "Serious share (labelled)", "n/a" if pd.isna(serious_pct) else pct(serious_pct))
kpi_card(k4, "Most-reported drug", drug_counts.index[0] if len(drug_counts) else "n/a")
st.caption("KPIs describe the filtered *sample* of reports. 'Serious share' is the share of reports flagged serious "
           "among reports with a known flag - not a risk or incidence rate. A report naming two study drugs is counted once in the report KPI and under each drug in drug charts.")

# ---- Top drugs / reactions -------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    d = drug_counts.rename_axis("drug").reset_index(name="count")
    st.plotly_chart(viz.top_bar(d, "drug", "Top reported drugs (report mentions)"))
    st.caption(CAPTIONS["top_drugs"])
with c2:
    rx = explode_reactions(reports)
    r = rx["reaction"].value_counts().rename_axis("reaction").reset_index(name="count")
    st.plotly_chart(viz.top_bar(r, "reaction", "Top 10 reported reactions", color="#5FA8A0"))
    st.caption(CAPTIONS["top_reactions"])

# ---- Time -------------------------------------------------------------------
st.markdown("#### Reports over time")
t1, t2 = st.columns([1, 3])
freq = t1.radio("Granularity", ["Quarter", "Month"], horizontal=True)
split_by_drug = t1.checkbox("One line per drug", value=False)
src = pairs if split_by_drug else reports
st.plotly_chart(viz.time_series(src, freq, by="drug" if split_by_drug else None))
st.caption(CAPTIONS["time"])

# ---- Serious vs non-serious ----------------------------------------------
s1, s2 = st.columns(2)
with s1:
    norm = st.checkbox("Show as % of each drug's reports", value=False)
    st.plotly_chart(viz.stacked_serious_by_drug(pairs, normalize=norm))
    st.caption(CAPTIONS["stacked"])
with s2:
    ser = reports[reports["serious_label"] == "Serious"]
    sr = explode_reactions(ser)["reaction"].value_counts().rename_axis("reaction").reset_index(name="count")
    st.plotly_chart(viz.top_bar(sr, "reaction", "Top reactions in Serious reports", color=viz.SERIOUS_COLOR))
    st.caption(CAPTIONS["serious_events"])

# ---- Heatmap -----------------------------------------------------------------
st.markdown("#### Drug × reaction patterns")
n_top = st.slider("Number of top reactions", 5, 25, 12)
st.plotly_chart(viz.heatmap_drug_reaction(pairs, n_top))
st.caption(CAPTIONS["heatmap"])

# ---- Demographics -------------------------------------------------------------
st.markdown("#### Demographics")
d1, d2, d3 = st.columns(3)
with d1:
    st.plotly_chart(viz.age_histogram(reports))
    st.caption(CAPTIONS["age"])
with d2:
    st.plotly_chart(viz.sex_bar(reports))
    st.caption(CAPTIONS["sex"])
with d3:
    st.plotly_chart(viz.country_bar(reports))
    st.caption(CAPTIONS["country"])

# ---- Table ------------------------------------------------------------------
st.markdown("#### Filtered data")
show_cols = ["report_id", "receive_date", "selected_drugs", "serious_label", "age_years", "age_group",
             "sex", "country", "n_reactions", "reactions"]
q = st.text_input("Search (drug, reaction, country, report id...)", "")
view = reports[show_cols]
if q:
    mask = view.astype(str).apply(lambda col: col.str.contains(q, case=False, regex=False)).any(axis=1)
    view = view[mask]
st.dataframe(view, hide_index=True)
st.download_button("Download filtered data (CSV)", view.to_csv(index=False).encode("utf-8"),
                   file_name="neurosafe_filtered_reports.csv", mime="text/csv")
st.caption(f"{len(view):,} rows shown. Column headers can be clicked to sort.")
