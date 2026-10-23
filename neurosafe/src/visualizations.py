"""Plotly figure builders with one shared visual identity (no Streamlit here)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import config
from .features import explode_reactions

SERIOUS_COLOR = "#B85C56"
NONSERIOUS_COLOR = "#5B8FB9"
UNKNOWN_COLOR = "#B8BEC6"
PRIMARY = "#2F6F8F"
COLORWAY = ["#2F6F8F", "#5FA8A0", "#E3B04B", "#B85C56", "#8A8FBF", "#7C9A6D", "#C48A5A", "#6C7A89"]
SERIOUS_MAP = {"Serious": SERIOUS_COLOR, "Non-serious": NONSERIOUS_COLOR, "Unknown": UNKNOWN_COLOR}

CAPTIONS = {
    "top_drugs": "Shows how many reports in this *sample* mention each study drug. It does NOT show which drug is more common in use or more risky: counts depend on how many patients take a drug, on the download cap, and on reporting habits.",
    "top_reactions": "Most frequently listed reaction terms (MedDRA) among the filtered reports. A term appearing often means it is often reported, not that the drug causes it.",
    "time": "Reports by receive date. Rises or falls can come from reporting campaigns, publicity, data lag in the latest quarters, or the download cap, not only from real-world changes.",
    "stacked": "Share of Serious vs Non-serious *reports* per drug. 'Serious' follows the reporter/regulatory definition and does not mean the drug caused the outcome. It is not an incidence rate.",
    "heatmap": "Each cell = % of a drug's reports that list the reaction. Compares reporting patterns only; it cannot rank drug safety because each drug has different users, indications, co-medication and reporting volume.",
    "age": "Distribution of reported patient age (years) where age was valid. Missing or implausible ages are excluded, so this may not represent all reports.",
    "sex": "Reports by sex as recorded. 'Unknown' means not reported.",
    "country": "Country where the event occurred (ISO code). FAERS is dominated by US reports; other countries are under-represented, so do not compare countries.",
    "missing": "Percentage of values missing or 'Unknown' per column. Models and charts that rely on these columns are weaker when the share is high.",
    "serious_events": "Most frequent reaction terms among Serious reports only. Descriptive; no causal link to the drug can be drawn.",
}


def _style(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.01, font=dict(size=16)) if title else None,
        template="plotly_white", colorway=COLORWAY, height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        font=dict(family="Inter, Segoe UI, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def empty_figure(message: str = "No data for the current filters") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=15, color="#6C7A89"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, height=260)


def top_bar(counts: pd.DataFrame, label_col: str, title: str, n: int = 10, color: str = PRIMARY) -> go.Figure:
    """Horizontal bar chart of the top ``n`` rows of a (label, count) table."""
    if counts.empty:
        return empty_figure()
    d = counts.sort_values("count", ascending=False).head(n).iloc[::-1]
    fig = px.bar(d, x="count", y=label_col, orientation="h", text="count")
    fig.update_traces(marker_color=color, textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_title=None, xaxis_title="Reports")
    return _style(fig, title, height=max(300, 34 * len(d) + 110))


def time_series(df: pd.DataFrame, freq: str = "Quarter", by: str | None = None, title: str = "Reports over time") -> go.Figure:
    """Counts per quarter or month; ``by`` (e.g. 'drug') draws one line per group."""
    d = df.dropna(subset=["receive_date"])
    if d.empty:
        return empty_figure()
    key = "report_quarter" if freq == "Quarter" else "report_month"
    groups = [key] + ([by] if by else [])
    g = d.groupby(groups).size().reset_index(name="reports").sort_values(key)
    fig = px.line(g, x=key, y="reports", color=by, markers=True)
    fig.update_layout(xaxis_title=None, yaxis_title="Reports")
    return _style(fig, title)


def stacked_serious_by_drug(pairs: pd.DataFrame, normalize: bool = False) -> go.Figure:
    if pairs.empty:
        return empty_figure()
    g = pairs.groupby(["drug", "serious_label"]).size().reset_index(name="reports")
    fig = px.bar(g, x="drug", y="reports", color="serious_label", color_discrete_map=SERIOUS_MAP,
                 category_orders={"serious_label": ["Serious", "Non-serious", "Unknown"]})
    fig.update_layout(barmode="stack", xaxis_title=None, yaxis_title="Reports", legend_title=None)
    if normalize:
        fig.update_layout(barnorm="percent", yaxis_title="% of the drug's reports")
    return _style(fig, "Serious vs Non-serious reports by drug")


def heatmap_drug_reaction(pairs: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """% of each drug's reports that list each of the overall top reactions."""
    if pairs.empty:
        return empty_figure()
    rx = explode_reactions(pairs)
    if rx.empty:
        return empty_figure("No reaction terms available")
    top = rx["reaction"].value_counts().head(top_n).index
    totals = pairs.groupby("drug")["report_id"].nunique()
    cnt = (rx[rx["reaction"].isin(top)].groupby(["drug", "reaction"])["report_id"].nunique()
           .unstack(fill_value=0).reindex(index=totals.index, columns=top, fill_value=0))
    pct = cnt.div(totals, axis=0) * 100
    fig = go.Figure(go.Heatmap(z=pct.values, x=list(pct.columns), y=list(pct.index),
                               colorscale="Blues", colorbar=dict(title="% of reports"),
                               hovertemplate="%{y} / %{x}<br>%{z:.1f}% of reports<extra></extra>"))
    fig.update_layout(xaxis_tickangle=-40)
    return _style(fig, "Drug x top reported reactions (% of the drug's reports)", height=max(340, 46 * len(pct) + 190))


def age_histogram(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["age_years"])
    if d.empty:
        return empty_figure("No valid ages")
    fig = px.histogram(d, x="age_years", nbins=30, color="serious_label", color_discrete_map=SERIOUS_MAP)
    fig.update_layout(xaxis_title="Age (years)", yaxis_title="Reports", legend_title=None, barmode="stack")
    return _style(fig, "Age distribution of reports")


def sex_bar(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return empty_figure()
    g = df["sex"].value_counts().reset_index()
    g.columns = ["sex", "count"]
    fig = px.bar(g, x="sex", y="count", text="count")
    fig.update_traces(marker_color=PRIMARY, textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title=None, yaxis_title="Reports")
    return _style(fig, "Reports by sex")


def country_bar(df: pd.DataFrame, top: int = 12, min_countries: int = 3) -> go.Figure:
    known = df[df["country"].notna() & ~df["country"].isin(["", "Unknown"])]
    if known["country"].nunique() < min_countries:
        return empty_figure("Not enough country information in the current selection")
    g = known["country"].value_counts().head(top).reset_index()
    g.columns = ["country", "count"]
    return top_bar(g, "country", "Reports by country of occurrence (top)", n=top)


def missing_bar(summary: pd.DataFrame) -> go.Figure:
    if summary.empty:
        return empty_figure()
    d = summary.sort_values("missing_pct")
    fig = px.bar(d, x="missing_pct", y="column", orientation="h", text="missing_pct")
    fig.update_traces(marker_color="#C48A5A", texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="% missing / Unknown", yaxis_title=None, xaxis_range=[0, max(105, d["missing_pct"].max() + 8)])
    return _style(fig, "Missing data by column", height=max(300, 40 * len(d) + 100))


def confusion_matrix_fig(cm: list[list[int]]) -> go.Figure:
    labels = ["Non-serious", "Serious"]
    z = np.array(cm)
    fig = go.Figure(go.Heatmap(z=z, x=[f"Pred: {l}" for l in labels], y=[f"True: {l}" for l in labels],
                               colorscale="Blues", showscale=False, text=z, texttemplate="%{text}",
                               textfont=dict(size=18)))
    fig.update_yaxes(autorange="reversed")
    return _style(fig, "Confusion matrix (test set, threshold 0.5)", height=340)


def roc_pr_fig(roc: dict | None, pr: dict | None, pos_rate: float | None = None) -> go.Figure:
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC curve", "Precision-Recall curve"))
    if roc:
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="model", line=dict(color=PRIMARY)), 1, 1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance", line=dict(dash="dash", color=UNKNOWN_COLOR)), 1, 1)
    if pr:
        fig.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name="model", showlegend=False, line=dict(color=PRIMARY)), 1, 2)
        if pos_rate is not None:
            fig.add_trace(go.Scatter(x=[0, 1], y=[pos_rate, pos_rate], mode="lines", name="prevalence", line=dict(dash="dash", color=UNKNOWN_COLOR)), 1, 2)
    fig.update_xaxes(title_text="False positive rate", row=1, col=1)
    fig.update_yaxes(title_text="True positive rate", row=1, col=1)
    fig.update_xaxes(title_text="Recall", row=1, col=2)
    fig.update_yaxes(title_text="Precision", row=1, col=2)
    return _style(fig, None, height=340)


def importance_bar(items: list[dict]) -> go.Figure:
    if not items:
        return empty_figure("No importance available")
    d = pd.DataFrame(items).iloc[::-1]
    colors = [SERIOUS_COLOR if v > 0 else NONSERIOUS_COLOR for v in d["value"]]
    fig = go.Figure(go.Bar(x=d["value"], y=d["feature"], orientation="h", marker_color=colors))
    fig.update_layout(xaxis_title=d["kind"].iloc[0], yaxis_title=None)
    return _style(fig, "Strongest associations in the model", height=max(320, 28 * len(d) + 110))
