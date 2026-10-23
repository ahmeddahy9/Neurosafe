"""Streamlit helpers: page setup, cached loading, sidebar filters, shared widgets."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from . import config, data_loader
from .features import explode_drugs

_CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1250px;}
.ns-kpi {background:#F3F7FA; border:1px solid #DCE6ED; border-radius:12px; padding:14px 16px;}
.ns-kpi .v {font-size:1.7rem; font-weight:700; color:#2F6F8F; line-height:1.2;}
.ns-kpi .l {font-size:.82rem; color:#5B6B79; text-transform:uppercase; letter-spacing:.04em;}
.ns-disclaimer {background:#FFF6E5; border-left:5px solid #E3B04B; padding:12px 16px; border-radius:6px; margin:.5rem 0 1rem 0;}
.ns-synthetic {background:#FBE9E7; border-left:5px solid #B85C56; padding:12px 16px; border-radius:6px; margin:.5rem 0 1rem 0;}
</style>
"""


def setup_page(title: str, icon: str = "🧠") -> None:
    st.set_page_config(page_title=f"NeuroSafe - {title}", page_icon=icon, layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### 🧠 NeuroSafe")
        st.caption("Educational pharmacovigilance dashboard for CNS sedatives")
    st.title(f"{icon} {title}")


def disclaimer_box() -> None:
    st.markdown(
        f'<div class="ns-disclaimer"><b>Disclaimer.</b> {config.DISCLAIMER}<br>'
        f'<span dir="rtl">{config.DISCLAIMER_AR}</span></div>',
        unsafe_allow_html=True,
    )


def data_banner() -> None:
    """Red banner for synthetic data; neutral provenance line for real downloads."""
    info = data_loader.data_source_info()
    if info.get("is_sample"):
        st.markdown(
            '<div class="ns-synthetic"><b>SYNTHETIC SAMPLE DATA - UI TEST ONLY.</b> '
            "These records are randomly generated, are not FAERS data and contain no real "
            "signal. Nothing shown here may be presented as an analysis.</div>",
            unsafe_allow_html=True,
        )
    elif info:
        st.caption(
            f"Data: FAERS via openFDA - downloaded {info.get('download_date_utc', 'n/a')[:10]} - "
            f"window {info.get('date_window', ['?', '?'])[0]} to {info.get('date_window', ['?', '?'])[1]}. "
            "Capped, non-random sample."
        )


def kpi_card(col, label: str, value: str) -> None:
    col.markdown(f'<div class="ns-kpi"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading processed data...")
def _cached_reports(path: str, mtime: float) -> pd.DataFrame | None:
    return data_loader.load_reports(path)


def get_reports() -> pd.DataFrame | None:
    path = config.REPORTS_FILE
    if not path.exists():
        return None
    return _cached_reports(str(path), path.stat().st_mtime)


def require_data() -> pd.DataFrame:
    """Return the processed table or show a friendly message and stop the page."""
    df = get_reports()
    if df is None or df.empty:
        st.warning(
            "No processed data found yet. Build it from the project folder:\n\n"
            "```bash\npython scripts/download_data.py        # or: --use-sample (synthetic, UI test only)\n"
            "python scripts/preprocess_data.py\npython scripts/train_model.py\n```"
        )
        st.stop()
    return df


@st.cache_resource(show_spinner=False)
def _cached_model(path: str, mtime: float):
    import joblib
    return joblib.load(path)


def get_model():
    path = config.MODEL_FILE
    if not path.exists():
        return None
    return _cached_model(str(path), path.stat().st_mtime)


def get_metrics() -> dict | None:
    return data_loader.read_json(config.METRICS_FILE)


# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
@dataclass
class Filtered:
    reports: pd.DataFrame     # one row per report (reports mentioning >= 1 selected drug)
    pairs: pd.DataFrame       # one row per (report, selected drug)
    drugs: list[str]


def sidebar_filters(df: pd.DataFrame) -> Filtered:
    st.sidebar.header("Filters")
    all_drugs = sorted(explode_drugs(df)["drug"].unique())
    drugs = st.sidebar.multiselect("Drug", all_drugs, default=all_drugs)

    d = df.dropna(subset=["receive_date"])
    if not d.empty:
        lo, hi = d["receive_date"].min().date(), d["receive_date"].max().date()
        picked = st.sidebar.date_input("Date range", value=(lo, hi), min_value=lo, max_value=hi)
        start, end = (picked[0], picked[-1]) if isinstance(picked, (tuple, list)) and len(picked) else (lo, hi)
    else:
        start = end = None

    sexes = sorted(df["sex"].unique())
    sex = st.sidebar.multiselect("Sex", sexes, default=sexes)
    groups = [g for g in config.AGE_GROUP_ORDER if g in set(df["age_group"])]
    age = st.sidebar.multiselect("Age group", groups, default=groups)
    ser_opts = [s for s in ["Serious", "Non-serious", "Unknown"] if s in set(df["serious_label"])]
    ser = st.sidebar.multiselect("Seriousness", ser_opts, default=ser_opts)
    countries = df["country"].value_counts().index.tolist()
    country = st.sidebar.multiselect("Country (empty = all)", countries, default=[])

    m = pd.Series(True, index=df.index)
    if start is not None:
        m &= df["receive_date"].between(pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)) | df["receive_date"].isna()
    m &= df["sex"].isin(sex) & df["age_group"].isin(age) & df["serious_label"].isin(ser)
    if country:
        m &= df["country"].isin(country)
    filt = df[m]
    pairs = explode_drugs(filt)
    pairs = pairs[pairs["drug"].isin(drugs)]
    reports = filt[filt["report_id"].isin(pairs["report_id"])]

    st.sidebar.markdown("---")
    st.sidebar.metric("Reports in selection", f"{len(reports):,}")
    if reports.empty:
        st.warning("No reports match the current filters. Widen the filters in the sidebar.")
        st.stop()
    return Filtered(reports=reports, pairs=pairs, drugs=drugs)


def pct(x: float) -> str:
    return f"{x:.1f}%"
