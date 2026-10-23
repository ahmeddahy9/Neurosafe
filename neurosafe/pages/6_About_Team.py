import streamlit as st

from src.utils import disclaimer_box, setup_page

setup_page("About / Team", "👥")
disclaimer_box()

st.markdown(
    """
**Project:** NeuroSafe - Educational Pharmacovigilance Dashboard for CNS Sedatives

| Field | Value |
|---|---|
| Course | `[Course name - edit in pages/6_About_Team.py]` |
| University / Faculty | `[University and faculty]` |
| Supervisor | `[Dr. Name]` |
| Academic year | `[Year]` |

**Team members** (placeholders - replace before submission)

| # | Name | Role |
|---|---|---|
| 1 | `[Student name]` | Data engineering & cleaning |
| 2 | `[Student name]` | EDA & visualisation |
| 3 | `[Student name]` | Machine learning |
| 4 | `[Student name]` | Documentation & testing |

**Data:** FDA FAERS via openFDA (public domain, CC0 1.0 per openFDA).
**Tech:** Python, Streamlit, pandas, scikit-learn, Plotly.
"""
)
st.caption("نبذة: مشروع تعليمي لتحليل بلاغات الآثار الجانبية للمهدئات - لا يُستخدم لأي قرار طبي.")
