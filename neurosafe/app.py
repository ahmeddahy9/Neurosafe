"""NeuroSafe - Home page. Run with: streamlit run app.py"""
import streamlit as st

from src import config, data_loader
from src.utils import data_banner, disclaimer_box, get_reports, setup_page

setup_page("NeuroSafe", "🧠")
st.subheader("Educational Pharmacovigilance Dashboard for CNS Sedatives")
disclaimer_box()
data_banner()

left, right = st.columns([3, 2], gap="large")
with left:
    st.markdown(
        """
**What is this?**
NeuroSafe explores publicly reported adverse-event *reports* for a selected set of CNS
sedatives / sedative-hypnotics (benzodiazepines, "Z-drugs" and a sedating anaesthetic adjunct).
It combines data cleaning, interactive exploration and a small educational machine-learning
prototype.

**Goals**
1. Download a documented sample of official FAERS reports (via openFDA) with a reproducible script.
2. Clean the data transparently (de-duplication, age conversion, missing-value audit).
3. Explore *reporting patterns*: which reactions are reported, when, for whom.
4. Train and honestly evaluate a prototype that predicts whether a report is flagged **Serious**.
5. Document the limits so nobody mistakes reports for risk.

**How to read the results.** Always read them as *reported events / reporting patterns*.
A report is not proof that a drug caused an event, and counts are not incidence rates.
        """
    )
    st.markdown("**Pages (left sidebar):** Data Overview - Exploratory Analysis - Drug Explorer - "
                "ML Risk Classification - Methodology & Limitations - About / Team")
with right:
    st.markdown("**Study drugs (editable in `src/config.py`)**")
    st.code(", ".join(config.SEDATIVE_DRUGS), language="text")
    st.caption("Not a medically exhaustive list.")

    df = get_reports()
    if df is None:
        st.info("No processed data yet. Run the three scripts in `scripts/` (see README).")
    else:
        st.metric("Reports in the processed dataset", f"{len(df):,}")

st.markdown("---")
st.markdown("#### Data source & citation")
st.markdown(
    f"""
- {config.CITATION}
- Official pages: [FAERS]({config.FAERS_URL}) - [openFDA drug event API]({config.OPENFDA_DOCS_URL})
- Reuse terms: openFDA states its data are public domain under a CC0 1.0 dedication ([license]({config.OPENFDA_LICENSE_URL}), [terms]({config.OPENFDA_TERMS_URL})).
  FAERS reports may be duplicated, incomplete or unverified; see *Methodology & Limitations*.
"""
)
info = data_loader.data_source_info()
if info and not info.get("is_sample"):
    st.caption(f"Downloaded on {info.get('download_date_utc', 'n/a')} UTC - API last updated: {info.get('api_last_updated', 'n/a')}")
