#!/usr/bin/env python
"""
Advanced Streamlit Dashboard (Demo-Only, No External Imports)

This file is self-contained:
- NO imports from cfpipe or .config
- Generates synthetic data so you can see the full UI immediately
- Later, you can replace the synthetic loader with real data loading

Run:
    streamlit run src/cfpipe/dashboard.py
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# --------------------------- DEMO DATA -------------------------------- #

def fabricate_demo_data(n_chunks: int = 350, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    aspects_pool = [
        "battery", "screen", "camera", "performance",
        "price", "design", "durability", "sound",
        "connectivity", "software"
    ]
    sentiments = ["positive", "negative"]
    absa_rows, chunk_rows = [], []

    start_unix = 1700000000  # arbitrary base timestamp
    for i in range(n_chunks):
        chunk_id = f"ch_{i}"
        unix_time = start_unix + int(rng.integers(0, 60 * 60 * 24 * 180))  # within ~6 months
        aspects = rng.choice(aspects_pool, size=rng.integers(1, 4), replace=False).tolist()
        sentiment = rng.choice(sentiments, p=[0.6, 0.4])
        base_score = rng.uniform(0.45, 0.95) if sentiment == "positive" else rng.uniform(0.05, 0.55)
        sentiment_score = float(np.clip(base_score + rng.normal(0, 0.06), 0.0, 1.0))

        absa_rows.append({
            "chunk_id": chunk_id,
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "aspects": aspects
        })
        text = f"This review discusses {', '.join(aspects)} and feels {sentiment} overall."
        chunk_rows.append({
            "chunk_id": chunk_id,
            "unix_time": unix_time,
            "text": text
        })

    return pd.DataFrame(absa_rows), pd.DataFrame(chunk_rows)


# --------------------------- LOAD (DEMO) ------------------------------- #

@st.cache_data(show_spinner=True)
def get_demo_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    return fabricate_demo_data()

df_absa, df_chunks = get_demo_data()

# Normalize
df_absa["aspects"] = df_absa["aspects"].apply(lambda v: v if isinstance(v, list) else ([] if pd.isna(v) else [v]))
df = df_absa.merge(df_chunks, on="chunk_id", how="left")

# Time fields
df["ts"] = pd.to_datetime(df["unix_time"], unit="s", errors="coerce")
df["month"] = df["ts"].dt.to_period("M").astype(str)
df["date"] = df["ts"].dt.date

# Signed sentiment
def to_signed(row):
    sent = row.get("sentiment", "")
    mult = 1.0 if sent == "positive" else (-1.0 if sent == "negative" else 0.0)
    return mult * (row.get("sentiment_score", 0.0) or 0.0)

df["signed_score"] = df.apply(to_signed, axis=1)

# --------------------------- UI LAYOUT -------------------------------- #

st.set_page_config(page_title="Crowdsourced Product Feedback Dashboard (Demo)", layout="wide")
st.title("📊 Crowdsourced Product Feedback Dashboard (Demo)")
st.info("Demo Mode: Synthetic data is displayed so you can preview the UI.")

# Sidebar filters
st.sidebar.header("Filters")
all_aspects: List[str] = sorted({a for lst in df["aspects"] for a in lst})
chosen_aspects = st.sidebar.multiselect("Filter by aspect(s)", all_aspects, default=[])

sentiment_choices = ["positive", "negative"]
sentiment_filter = st.sidebar.multiselect("Sentiment", sentiment_choices, default=sentiment_choices)

date_min, date_max = df["date"].min(), df["date"].max()
date_range = st.sidebar.date_input("Date range", (date_min, date_max), min_value=date_min, max_value=date_max)
search_text = st.sidebar.text_input("Search review text", "")
min_occ = st.sidebar.slider("Min occurrences for aspect bars", 1, 30, 5)
show_positive_bar = st.sidebar.checkbox("Show positive aspects bar", True)
smooth_trend = st.sidebar.checkbox("3-point moving average (trend)", False)
show_heatmap = st.sidebar.checkbox("Show aspect co-occurrence heatmap", False)

# Apply filters
filtered = df.copy()
if chosen_aspects:
    filtered = filtered[filtered["aspects"].apply(lambda xs: any(a in xs for a in chosen_aspects))]
filtered = filtered[filtered["sentiment"].isin(sentiment_filter)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]
if search_text.strip():
    needle = search_text.lower()
    filtered = filtered[filtered["text"].fillna("").str.lower().str.contains(needle)]

if filtered.empty:
    st.warning("No data matches current filters.")
    st.stop()

# 1) Sentiment distribution
st.subheader("1. Sentiment Distribution")
dist = filtered["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="count")
dist["percent"] = 100 * dist["count"] / dist["count"].sum()
pie = alt.Chart(dist).mark_arc(innerRadius=50).encode(
    theta="count:Q",
    color=alt.Color("sentiment:N", legend=alt.Legend(title="Sentiment")),
    tooltip=["sentiment", "count", alt.Tooltip("percent:Q", format=".1f")]
)
st.altair_chart(pie, use_container_width=True)

# helper
def aspect_frequency(sub: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind in ["positive", "negative"]:
        sub = sub[sub["sentiment"] == kind]
    counts = {}
    for lst in sub["aspects"]:
        for a in lst:
            counts.setdefault(a, {"count": 0, "signed_vals": []})
            counts[a]["count"] += 1
            # approximate signed average for ranking
            signed_mean = sub[sub["aspects"].apply(lambda xs: a in xs)]["signed_score"].mean()
            counts[a]["signed_vals"].append(signed_mean)
    rows = [{"aspect": a, "count": v["count"], "avg_signed": (np.mean(v["signed_vals"]) if v["signed_vals"] else 0.0)}
            for a, v in counts.items()]
    out = pd.DataFrame(rows)
    return out[out["count"] >= min_occ].sort_values("count", ascending=False)

# 2) Aspect bars
st.subheader("2. Aspect Bars")
neg_df = aspect_frequency(filtered, "negative")
if neg_df.empty:
    st.info("No negative aspects meet the threshold.")
else:
    neg_chart = alt.Chart(neg_df.head(15)).mark_bar(color="#d62728").encode(
        x=alt.X("count:Q", title="Occurrences"),
        y=alt.Y("aspect:N", sort="-x"),
        tooltip=["aspect", "count"]
    )
    st.markdown("**Top Negative Aspects**")
    st.altair_chart(neg_chart, use_container_width=True)

if show_positive_bar:
    pos_df = aspect_frequency(filtered, "positive")
    if pos_df.empty:
        st.info("No positive aspects meet the threshold.")
    else:
        pos_chart = alt.Chart(pos_df.head(15)).mark_bar(color="#2ca02c").encode(
            x=alt.X("avg_signed:Q", title="Avg Signed Sentiment"),
            y=alt.Y("aspect:N", sort="-x"),
            tooltip=["aspect", "count", alt.Tooltip("avg_signed:Q", format=".3f")]
        )
        st.markdown("**Top Positive Aspects (by avg signed score)**")
        st.altair_chart(pos_chart, use_container_width=True)

# 3) Trend over time
st.subheader("3. Sentiment Trend Over Time (Monthly)")
trend = filtered.groupby("month", dropna=True)["signed_score"].mean().reset_index().rename(columns={"signed_score": "avg_signed"})
if smooth_trend and len(trend) >= 3:
    trend["smoothed"] = trend["avg_signed"].rolling(3, center=True, min_periods=1).mean()
    y_field = "smoothed"
else:
    y_field = "avg_signed"

trend_chart = alt.Chart(trend).mark_line(point=True).encode(
    x=alt.X("month:N", title="Month"),
    y=alt.Y(f"{y_field}:Q", title="Avg Signed Sentiment"),
    tooltip=["month", alt.Tooltip(y_field, format=".3f")]
)
st.altair_chart(trend_chart, use_container_width=True)

# 4) Co-occurrence heatmap (optional)
if show_heatmap:
    st.subheader("4. Aspect Co-occurrence Heatmap")
    aspects = sorted({a for lst in filtered["aspects"] for a in lst})
    idx = {a: i for i, a in enumerate(aspects)}
    mat = np.zeros((len(aspects), len(aspects)), dtype=int)
    for lst in filtered["aspects"]:
        unique = list(set(lst))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                ai, aj = idx[unique[i]], idx[unique[j]]
                mat[ai, aj] += 1
                mat[aj, ai] += 1
    co_df = pd.DataFrame(mat, index=aspects, columns=aspects)
    if co_df.sum().sum() == 0:
        st.info("Not enough aspect pairs for co-occurrence.")
    else:
        melt = co_df.reset_index().melt(id_vars="index", var_name="aspect_b", value_name="count").rename(columns={"index": "aspect_a"})
        heat = alt.Chart(melt).mark_rect().encode(
            x=alt.X("aspect_a:N", sort=None, title="Aspect A"),
            y=alt.Y("aspect_b:N", sort=None, title="Aspect B"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="inferno"), title="Co-occurrence"),
            tooltip=["aspect_a", "aspect_b", "count"]
        ).properties(height=500)
        st.altair_chart(heat, use_container_width=True)

# 5) Review explorer
st.subheader("5. Review Explorer")
exploded = filtered.explode("aspects")
if chosen_aspects:
    exploded = exploded[exploded["aspects"].isin(chosen_aspects)]
view_cols = ["chunk_id", "sentiment", "sentiment_score", "signed_score", "aspects", "text", "ts"]
exploded_view = exploded[view_cols].drop_duplicates()
st.write(f"Rows: {len(exploded_view)}")
st.dataframe(exploded_view, use_container_width=True)

st.download_button(
    "Download filtered CSV",
    data=exploded_view.to_csv(index=False),
    file_name="filtered_demo_reviews.csv",
    mime="text/csv"
)

# Summary metrics
st.subheader("Summary Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", f"{len(filtered):,}")
c2.metric("Distinct Aspects", f"{len({a for lst in filtered['aspects'] for a in lst}):,}")
c3.metric("Avg Signed", f"{filtered['signed_score'].mean():.3f}")
pos_ratio = (filtered["sentiment"] == "positive").mean() * 100
c4.metric("Positive %", f"{pos_ratio:.1f}%")

st.success("✅ Demo dashboard loaded successfully.")
