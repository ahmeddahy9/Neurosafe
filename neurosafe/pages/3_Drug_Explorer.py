import pandas as pd
import streamlit as st

from src.features import explode_drugs, explode_reactions
from src.utils import data_banner, disclaimer_box, kpi_card, pct, require_data, setup_page
from src import visualizations as viz
from src.visualizations import CAPTIONS

setup_page("Drug Explorer", "💊")
disclaimer_box()
data_banner()
df = require_data()

pairs_all = explode_drugs(df)
drugs = sorted(pairs_all["drug"].unique())
drug = st.selectbox("Choose a study drug", drugs)
d = pairs_all[pairs_all["drug"] == drug]

st.info("Results reflect **reports submitted** for this drug in a capped sample - not the real-world risk, "
        "frequency or safety of the drug. Co-medications, indication and reporting habits are not controlled.")

known = d[d["serious_label"].isin(["Serious", "Non-serious"])]
k1, k2, k3, k4 = st.columns(4)
kpi_card(k1, "Reports mentioning drug", f"{len(d):,}")
kpi_card(k2, "Serious share (labelled)", pct(known["serious_label"].eq("Serious").mean() * 100) if len(known) else "n/a")
kpi_card(k3, "Median age (years)", f"{d['age_years'].median():.0f}" if d["age_years"].notna().any() else "n/a")
kpi_card(k4, "Median reactions / report", f"{d['n_reactions'].median():.0f}")

c1, c2 = st.columns(2)
with c1:
    r = explode_reactions(d)["reaction"].value_counts().rename_axis("reaction").reset_index(name="count")
    st.plotly_chart(viz.top_bar(r, "reaction", f"Top reported reactions - {drug}", n=12, color="#5FA8A0"))
    st.caption(CAPTIONS["top_reactions"])
with c2:
    st.plotly_chart(viz.time_series(d, "Quarter", title=f"Reports over time - {drug}"))
    st.caption(CAPTIONS["time"])

c3, c4 = st.columns(2)
with c3:
    comp = d["serious_label"].value_counts().rename_axis("label").reset_index(name="reports")
    st.dataframe(comp.assign(share=(comp["reports"] / comp["reports"].sum() * 100).round(1).astype(str) + "%"),
                 hide_index=True)
    st.caption("Seriousness flag distribution within this drug's reports. It is a reporting category, not a measured risk.")
with c4:
    st.plotly_chart(viz.age_histogram(d))
    st.caption(CAPTIONS["age"])

# Reactions that are over-represented relative to the other study drugs (descriptive only)
st.markdown("#### Reaction share vs. the other study drugs")
others = pairs_all[pairs_all["drug"] != drug]
if len(d) >= 30 and len(others) >= 30:
    a = explode_reactions(d).groupby("reaction")["report_id"].nunique() / d["report_id"].nunique() * 100
    b = explode_reactions(others).groupby("reaction")["report_id"].nunique() / others["report_id"].nunique() * 100
    cmp_ = pd.concat([a.rename(drug), b.rename("Other study drugs")], axis=1).fillna(0)
    cmp_ = cmp_[cmp_[drug] >= 2].assign(difference=lambda x: (x[drug] - x["Other study drugs"]).round(1)).round(1)
    st.dataframe(cmp_.sort_values("difference", ascending=False).head(12).rename_axis("reaction").reset_index(),
                 hide_index=True)
    st.caption("Percent of reports listing each reaction (columns are % of reports). Differences describe reporting patterns "
               "only; they are not safety rankings and no statistical test or disproportionality signal is claimed here.")
else:
    st.caption("Not enough reports to compare this drug with the others (need at least 30 on each side).")
