# streamlit_app.py
# HumourScope — Reddit Humour Norms (real data only)

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st

# Project helpers
from humourscope.data_loader import load_from_file, load_hf_sample
from humourscope.lexicon import EN_STOP, TOKEN_RE, extract_entities, tokenize_meaningful

# ====================== Page & CSS ======================

st.set_page_config(
    page_title="HumourScope Reddit Humour Norms",
    layout="wide",
    page_icon="😂",
)

st.markdown(
    """
<style>
  /* Hide Streamlit header/toolbar and reduce padding */
  header[data-testid="stHeader"] { display: none; }
  div[data-testid="stToolbar"] { display: none; }
  .block-container { padding-top: 1rem; }

  /* Dark palette polish */
  body, .stApp { background-color: #0f172a; color: #e5e7eb; }
  section[data-testid="stSidebar"] { background: #111827; }
  .stPlotlyChart, .stDataFrame { background: #111827; border-radius: 10px; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<h1>😂 HumourScope — Reddit Humour Norms</h1>
<p>Explore humour, sentiment, and tone across Reddit communities</p>
""",
    unsafe_allow_html=True,
)

# ====================== Sidebar ======================

st.sidebar.header("Inputs")

subs = st.sidebar.text_input(
    "Filter by Subreddits (comma-separated; optional)",
    "funny, antimemes, ProgrammerHumor",
)

data_source = st.sidebar.selectbox(
    "Data source",
    ["Hugging Face (public dataset)", "Local file (CSV/Parquet; Pushshift export)"],
)

# HF options
hf_dataset = "SocialGrep/one-million-reddit-jokes"
hf_n = 5000
if data_source == "Hugging Face (public dataset)":
    st.sidebar.caption("Dataset: SocialGrep/one-million-reddit-jokes (mostly r/Jokes)")
    hf_n = st.sidebar.slider("Rows to load", 1000, 30000, 5000, step=1000)

# Local file options
uploaded = None
local_path = ""
if data_source == "Local file (CSV/Parquet; Pushshift export)":
    uploaded = st.sidebar.file_uploader("Upload CSV/Parquet", type=["csv", "parquet"])
    local_path = st.sidebar.text_input(
        "…or path on disk",
        "data/exports/reddit_2025-08_funny_antimemes_ProgrammerHumor.parquet",
    )

run = st.sidebar.button("Fetch & Analyse")

# ====================== Cleaning ======================

URL_RE = re.compile(r"https?://\S+")
USER_RE = re.compile(r"\bu/\w+")
SUB_RE = re.compile(r"\br/\w+")
WS_RE = re.compile(r"\s+")
BOILER_RE = re.compile(
    r"(i am a bot|action (?:was )?performed automatically|"
    r"contact (?:the )?moderators(?: of this subreddit)?|"
    r"message (?:the )?moderators|"
    r"www\.reddit\.com|reddit\.com|wiki|rules)",
    re.I,
)


def clean_text(s: str) -> str:
    s = s or ""
    s = URL_RE.sub(" ", s)
    s = BOILER_RE.sub(" ", s)
    s = USER_RE.sub(" ", s)
    s = SUB_RE.sub(" ", s)
    s = s.lower()
    s = re.sub(r"[^\w\s!?.,;:()\-']", " ", s)
    s = WS_RE.sub(" ", s).strip()
    return s


def clean_text_series(col: pd.Series) -> pd.Series:
    return col.fillna("").map(clean_text)


# ====================== Scoring & Topics ======================


def lightweight_humor_score(texts: List[str]) -> pd.Series:
    """Fast heuristic if transformers are unavailable."""
    scores = []
    for t in texts:
        t2 = t.lower() if isinstance(t, str) else ""
        s = 0.05 + (len(t2) % 10) * 0.01
        if any(k in t2 for k in ("lol", "lmao", "rofl")):
            s += 0.12
        s += 0.10 * t2.count("!")
        if any(e in t2 for e in ("😂", "😅", "🤣")):
            s += 0.08
        if any(k in t2 for k in ("printf", "console.log")):
            s += 0.06
        scores.append(max(0.05, min(0.95, s)))
    return pd.Series(scores)


def try_hf_humor_probs(texts: List[str]) -> pd.Series:
    """
    Use a public sentiment model (positive ≈ humour proxy).
    Falls back to a heuristic if transformers/weights unavailable.
    """
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        mdl.eval()

        probs: list[float] = []
        with torch.inference_mode():
            for i in range(0, len(texts), 16):
                batch = [t if isinstance(t, str) else "" for t in texts[i : i + 16]]
                enc = tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                logits = mdl(**enc).logits
                pos_p = logits.softmax(dim=1)[:, 1].cpu().tolist()
                probs.extend(pos_p)
        return pd.Series(probs)
    except Exception:
        return lightweight_humor_score(texts)


def safe_bertopic(texts: List[str]) -> pd.DataFrame:
    """Try BERTopic; otherwise n-gram frequencies with strong stopwording."""
    # --- sanitize input ---
    texts = [t for t in texts if isinstance(t, str)]
    texts = [t.strip() for t in texts if t and t.strip()]
    if len(texts) < 20:
        st.info("Not enough text for topic extraction yet.")
        return pd.DataFrame({"Top terms": [], "Docs": []})

    try:
        # ---- Primary: BERTopic ----
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer

        sample = texts[:1500] if len(texts) > 1500 else texts
        emb = SentenceTransformer("all-MiniLM-L6-v2")

        # light vectorizer for topic labels
        vec = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=5)
        topic_model = BERTopic(
            vectorizer_model=vec,
            calculate_probabilities=False,
            verbose=False,
        )
        _topics, _ = topic_model.fit_transform(
            sample,
            emb.encode(sample, normalize_embeddings=True, show_progress_bar=False),
        )
        info = topic_model.get_topic_info().rename(columns={"Name": "Top terms", "Count": "Docs"})
        return info.sort_values("Docs", ascending=False)

    except Exception:
        # ---- Fallback: frequent phrases (2–3grams) with strong stop-wording ----
        try:
            from sklearn.feature_extraction.text import CountVectorizer

            vec = CountVectorizer(
                ngram_range=(2, 3),
                stop_words=list(EN_STOP),
                token_pattern=TOKEN_RE.pattern,
                min_df=5,
                max_df=0.30,  # trim overly common phrases
            )
            X = vec.fit_transform(texts)
            if X.shape[1] == 0:
                st.info("No salient phrases (empty vocabulary).")
                return pd.DataFrame({"Top terms": [], "Docs": []})

            freqs = X.sum(axis=0).A1
            vocab = vec.get_feature_names_out()
            out = (
                pd.DataFrame({"Top terms": vocab, "Docs": freqs})
                .sort_values("Docs", ascending=False)
                .head(20)
            )
            return out

        except Exception as e2:
            st.info(f"Topic fallback unavailable: {e2}")
            return pd.DataFrame({"Top terms": [], "Docs": []})


def humour_mix_plot(df: pd.DataFrame):
    agg = df.groupby("subreddit")["humor_prob"].mean().reset_index()
    return px.bar(
        agg,
        x="subreddit",
        y="humor_prob",
        title="Avg humour probability by subreddit",
        range_y=[0, 1],
    )


# ====================== Main ======================
CFG = {"displaylogo": False, "responsive": True}
if run:
    sub_list = [s.strip() for s in subs.split(",") if s.strip()]

    # Load real data per source
    if data_source == "Hugging Face (public dataset)":
        with st.spinner(f"Loading HF dataset: {hf_dataset} ..."):
            df = load_hf_sample(dataset=hf_dataset, split="train", n=hf_n)
            st.success(f"Loaded {len(df):,} rows from HF dataset")

    elif data_source == "Local file (CSV/Parquet; Pushshift export)":
        if uploaded is not None:
            tmpdir = Path("data/uploads")
            tmpdir.mkdir(parents=True, exist_ok=True)
            tmp_path = tmpdir / uploaded.name
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            path = tmp_path
        else:
            path = Path(local_path)

        with st.spinner(f"Loading file: {path} ..."):
            if not path.exists():
                st.error(f"File not found: {path}")
                st.stop()
            df = load_from_file(path)
            st.success(f"Loaded {len(df):,} rows from {path.name}")

    else:
        st.error("Unknown data source")
        st.stop()

    # Optional subreddit filter (non-destructive)
    if sub_list:
        if "subreddit" in df.columns and df["subreddit"].notna().any():
            filtered = df[df["subreddit"].astype(str).isin(sub_list)]
            if filtered.empty:
                st.warning("No rows match those subreddits in this dataset; keeping all rows.")
            else:
                df = filtered
        else:
            st.info("Dataset has no subreddit labels; showing all rows.")

    # Optional: drop obvious AutoModerator / boilerplate comments
    automod_hints = ("i am a bot", "performed automatically", "contact the moderators")
    mask = ~df["text"].astype(str).str.lower().str.contains("|".join(automod_hints))
    df = df[mask]

    # Clean text for modeling/plots
    df["text_clean"] = clean_text_series(df["text"])

    st.caption("Preview of normalized input")
    st.dataframe(df.head(), width="stretch")

    # Score humour
    with st.spinner("Scoring humour..."):
        df["humor_prob"] = try_hf_humor_probs(df["text_clean"].tolist())
        df["humor_label"] = (df["humor_prob"] >= 0.5).astype(int)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(humour_mix_plot(df), width="stretch", config=CFG, theme=None)
    with c2:
        st.subheader("Humour share by community")
        pies = (
            df.groupby("subreddit")["humor_label"]
            .mean()
            .reset_index()
            .rename(columns={"humor_label": "humor_share"})
        )
        if len(pies) and pies["humor_share"].notna().any():
            st.plotly_chart(
                px.pie(
                    pies,
                    names="subreddit",
                    values="humor_share",
                    title="Share of comments likely humorous (≥ 0.5)",
                ),
                width="stretch",
                config=CFG,
                theme=None,
            )
        else:
            st.info("No humour labels available to plot yet.")

    # Counts stacked bar
    st.subheader("Humorous vs non-humorous counts by community")
    counts = (
        df.assign(label=df["humor_label"].map({1: "humorous", 0: "not humorous"}))
        .groupby(["subreddit", "label"])
        .size()
        .reset_index(name="n")
    )
    if len(counts):
        st.plotly_chart(
            px.bar(counts, x="subreddit", y="n", color="label", barmode="stack"),
            width="stretch",
            config=CFG,
            theme=None,
        )

    # Topic templates / phrases
    st.subheader("Template clusters / frequent phrases")
    info = safe_bertopic(df["text_clean"].tolist())
    st.dataframe(info.head(20), width="stretch")

    # Entities (shared helper; returns empty if spaCy unavailable)
    st.subheader("In-group entities")
    ents = extract_entities(df["text"].tolist()[:1500])
    if not ents.empty:
        st.dataframe(ents.head(30), width="stretch")
    else:
        st.write("No entities extracted (or NER unavailable).")

    # ---------------- Compare communities ----------------
    st.subheader("Compare communities")
    uniq = sorted(df["subreddit"].astype(str).unique().tolist())

    if len(uniq) >= 2:
        colA, colB = st.columns(2)
        with colA:
            a = st.selectbox("Community A", uniq, index=0)
        with colB:
            b = st.selectbox("Community B", uniq, index=1)

        cc1, cc2 = st.columns(2)

        def profile_block(data: pd.DataFrame, name: str) -> None:
            subdf = data[data["subreddit"] == name]
            st.markdown(f"#### r/{name}")
            st.metric("Avg humour prob", f"{subdf['humor_prob'].mean():.2f}")
            st.metric("Comments analysed", f"{len(subdf):,}")

            tokens = (
                subdf["text_clean"]
                .astype(str)
                .map(tokenize_meaningful)
                .explode()
                .value_counts()
                .head(15)
                .reset_index()
                .rename(columns={"index": "token", 0: "freq"})
            )
            if not {"token", "freq"}.issubset(tokens.columns):
                tokens.columns = ["token", "freq"]
            st.dataframe(tokens, width="stretch")

        with cc1:
            profile_block(df, a)
        with cc2:
            profile_block(df, b)
    else:
        st.info("Need at least two subreddits to compare.")

    # Export
    st.subheader("Export")
    export_dir = Path("data/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    export_name = f"humourscope_{'_'.join(uniq) if uniq else 'all'}.csv"
    path = export_dir / export_name
    df.to_csv(path, index=False)
    st.download_button(
        "Download CSV",
        data=path.read_bytes(),
        file_name=export_name,
        mime="text/csv",
    )

else:
    st.info("Choose a data source, then click **Fetch & Analyse**.")
