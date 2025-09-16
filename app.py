import os
from datetime import datetime, timedelta
import pandas as pd
import requests
import streamlit as st

# --------------------
# CONFIG
# --------------------
NEWSAPI_KEY = st.secrets.get("NEWSAPI_KEY", os.getenv("NEWSAPI_KEY", ""))

BRANDS = ["Zomato", "Swiggy"]

# --------------------
# HELPERS
# --------------------
@st.cache_data(ttl=60*60)
def fetch_mentions(brand: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch last-24h mentions from NewsAPI."""
    if not NEWSAPI_KEY:
        return pd.DataFrame()
    url = (f"https://newsapi.org/v2/everything?q={requests.utils.quote(brand)}"
           f"&language=en&from={start_date}&to={end_date}"
           f"&sortBy=publishedAt&apiKey={NEWSAPI_KEY}")
    r = requests.get(url, timeout=30).json()
    rows = []
    for a in r.get("articles", []):
        rows.append({
            "brand": brand,
            "title": a.get("title"),
            "source": (a.get("source") or {}).get("name"),
            "url": a.get("url"),
            "published": a.get("publishedAt"),
            "snippet": a.get("description"),
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=60*60)
def build_dataset():
    """Fetch mentions & summaries for all brands."""
    end = datetime.utcnow().date()
    start = end - timedelta(days=1)
    start_s, end_s = start.isoformat(), end.isoformat()

    dfs = [fetch_mentions(b, start_s, end_s) for b in BRANDS]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # “Summary” = top 5 headlines joined (free pseudo-summary)
    summaries = []
    for b in BRANDS:
        subset = df[df["brand"] == b].head(5)
        joined = " | ".join([t for t in subset["title"].dropna().tolist()])
        summaries.append({"brand": b, "summary": joined if joined else "(no mentions)"})
    summary_df = pd.DataFrame(summaries)
    return df, summary_df

# --------------------
# UI
# --------------------
st.set_page_config(page_title="Zomato vs Swiggy Tracker", page_icon="🍴", layout="wide")
st.title("🍴 Zomato vs Swiggy — Press Tracker (last 24h)")

if st.button("🔄 Refresh now"):
    st.cache_data.clear()

with st.spinner("Fetching latest mentions..."):
    mentions_df, summaries_df = build_dataset()

col1, col2 = st.columns(2)
col1.metric("Total Mentions", f"{len(mentions_df):,}")
if not mentions_df.empty:
    sov = mentions_df.groupby("brand").size().sort_values(ascending=False)
    col2.metric("Top Share of Voice", sov.index[0])

if not mentions_df.empty:
    st.subheader("📊 Share of Voice")
    sov_pct = (sov / sov.sum() * 100).round(1)
    st.bar_chart(sov_pct)

st.subheader("📰 All Mentions (24h)")
if not mentions_df.empty:
    st.dataframe(mentions_df[["brand","title","source","url","published"]], use_container_width=True)
else:
    st.info("No mentions found.")

st.subheader("🤖 AI-Lite Summaries")
for _, r in summaries_df.iterrows():
    st.markdown(f"### {r['brand']}\n{r['summary']}")
