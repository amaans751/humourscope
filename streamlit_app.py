# streamlit_app.py
# HumourScope — Reddit Humour Norms (Demo-ready)

import os
import re
from pathlib import Path
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------- UI & Page Setup ----------------

st.set_page_config(page_title="HumourScope", layout="wide")
st.title("HumourScope — Reddit Humour Norms")

st.sidebar.header("Inputs")
subs = st.sidebar.text_input("Subreddits (comma-separated)", "funny, antimemes, ProgrammerHumor")
window = st.sidebar.selectbox(
    "Time window (ignored in Demo)", ["day", "week", "month", "year"], index=2
)
limit_posts = st.sidebar.slider("Top posts per subreddit (ignored in Demo)", 20, 200, 80, step=20)
demo_mode = st.sidebar.toggle("Demo mode (no Reddit API)", value=True)
run = st.sidebar.button("Fetch & Analyse")

# ---------------- Utilities ----------------


def have_creds() -> bool:
    """Detect if Reddit creds are present via env or Streamlit secrets."""
    if os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET"):
        return True
    try:
        _ = st.secrets["REDDIT_CLIENT_ID"]
        _ = st.secrets["REDDIT_CLIENT_SECRET"]
        return True
    except Exception:
        return False


URL_RE = re.compile(r"https?://\S+")
USER_RE = re.compile(r"\bu/\w+")
SUB_RE = re.compile(r"\br/\w+")
WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    s = (s or "").lower()
    s = URL_RE.sub(" ", s)
    s = USER_RE.sub(" ", s)
    s = SUB_RE.sub(" ", s)
    s = re.sub(r"[^\w\s!?.,;:()\-]", " ", s)
    s = WS_RE.sub(" ", s).strip()
    return s


def clean_text_series(col: pd.Series) -> pd.Series:
    return col.fillna("").map(clean_text)


def synthetic_demo_df(rows: int = 240) -> pd.DataFrame:
    """Generate a small but varied dataset so charts have signal."""
    examples = [
        (
            "funny",
            "I told my computer I needed a break; it said No problem, I'll crash.",
            77,
        ),
        ("funny", "My boss told me to have a good day, so I went home.", 64),
        ("antimemes", "This is not a meme. It's an anti-meme.", 23),
        ("antimemes", "A picture of nothing. Title: nothing. Comments: nothing.", 12),
        (
            "ProgrammerHumor",
            "Unit tests pass on Friday; prod deploy fails on Monday 😅",
            85,
        ),
        ("ProgrammerHumor", "printf('hello world'); // works in prod!", 48),
    ]
    data = []
    for i in range(rows):
        sub, txt, ups = examples[i % len(examples)]
        ups2 = max(0, int(ups * (0.8 + (i % 7) * 0.05)))
        t = txt
        if i % 5 == 0:
            t += " 😂"
        if i % 6 == 0:
            t = t.replace(".", "!").replace("'", "")
        data.append(
            {
                "subreddit": sub,
                "post_id": f"p{i//10}",
                "post_title": f"Demo post {i//10} in r/{sub}",
                "post_ups": ups2 + 5,
                "created_utc": 1_700_000_000 + i * 60,
                "comment_id": f"c{i}",
                "text": t,
                "comment_ups": ups2,
                "is_submitter": False,
                "parent_id": f"t1_{i-1}" if i else "t3_root",
                "depth": 0 if i % 9 else 1,
                "permalink": f"https://reddit.com/r/{sub}/comments/demo/{i}",
            }
        )
    df = pd.DataFrame(data)
    df["text_clean"] = clean_text_series(df["text"])
    return df


def lightweight_humor_score(texts: List[str]) -> pd.Series:
    """
    Tiny, fast heuristic so demo works without HF/torch.
    Signals: emojis, exclamation, 'lol', 'lmao', programmer & anti-meme cues.
    Outputs ~probabilities in [0.05, 0.95].
    """
    scores = []
    for t in texts:
        t2 = t.lower()
        s = 0.0
        s += 0.12 if ("lol" in t2 or "lmao" in t2 or "rofl" in t2) else 0.0
        s += 0.10 * t.count("!")
        s += 0.08 if ("😂" in t or "😅" in t or "🤣" in t) else 0.0
        s += 0.08 if "i told my computer" in t2 else 0.0
        s += 0.06 if ("anti-meme" in t2 or "anti meme" in t2) else 0.0
        s += 0.06 if ("printf" in t2 or "console.log" in t2) else 0.0
        s += 0.04 if ("boss" in t2 and "day" in t2) else 0.0
        scores.append(max(0.05, min(0.95, s)))
    return pd.Series(scores)


def try_hf_humor_probs(texts: List[str]) -> pd.Series:
    """Try HF model; fallback to lightweight heuristic on any failure."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained("Humor-Research/humor-detection-comb-23")
        mdl = AutoModelForSequenceClassification.from_pretrained(
            "Humor-Research/humor-detection-comb-23"
        ).to(device)
        mdl.eval()
        probs = []
        with torch.inference_mode():
            for i in range(0, len(texts), 16):
                batch = texts[i : i + 16]
                enc = tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(device)
                logits = mdl(**enc).logits
                p = logits.softmax(dim=1)[:, 1].detach().cpu().tolist()
                probs.extend(p)
        return pd.Series(probs)
    except Exception as e:
        st.info(f"Using lightweight humour heuristic (HF model unavailable: {e})")
        return lightweight_humor_score(texts)


def safe_bertopic(texts: List[str]) -> pd.DataFrame:
    """Try BERTopic; fallback to frequent phrase counts."""
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer

        sample = texts[:1500] if len(texts) > 1500 else texts
        emb = SentenceTransformer("all-MiniLM-L6-v2")
        vec = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=5)
        topic_model = BERTopic(vectorizer_model=vec, calculate_probabilities=False, verbose=False)
        topics, _ = topic_model.fit_transform(
            sample,
            emb.encode(sample, normalize_embeddings=True, show_progress_bar=False),
        )
        info = topic_model.get_topic_info()
        info = info.rename(columns={"Name": "Top terms", "Count": "Docs"})
        return info.sort_values("Docs", ascending=False)
    except Exception as e:
        from sklearn.feature_extraction.text import CountVectorizer

        vec = CountVectorizer(ngram_range=(2, 3), stop_words="english", min_df=3)
        X = vec.fit_transform(texts)
        freqs = X.sum(axis=0).A1
        vocab = vec.get_feature_names_out()
        df = pd.DataFrame({"Top terms": vocab, "Docs": freqs}).sort_values("Docs", ascending=False)
        st.info(f"BERTopic unavailable, showing frequent phrases instead ({e})")
        return df.head(20)


def safe_entities(texts: List[str]) -> pd.DataFrame:
    """Try spaCy NER; fallback to empty on failure (with notice)."""
    try:
        import spacy

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            os.system("python -m spacy download en_core_web_sm >/dev/null 2>&1")
            nlp = spacy.load("en_core_web_sm")
        rows = []
        for t in texts[:1500]:
            doc = nlp(t)
            for ent in doc.ents:
                rows.append((ent.text, ent.label_))
        df = pd.DataFrame(rows, columns=["entity", "label"])
        if df.empty:
            return pd.DataFrame(columns=["entity", "label", "count"])
        return (
            df.value_counts(["entity", "label"])
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
    except Exception as e:
        st.info(f"spaCy NER unavailable, skipping entities ({e})")
        return pd.DataFrame(columns=["entity", "label", "count"])


def humour_mix_plot(df: pd.DataFrame):
    agg = df.groupby("subreddit")["humor_prob"].mean().reset_index()
    return px.bar(
        agg,
        x="subreddit",
        y="humor_prob",
        title="Avg humour probability by subreddit",
        range_y=[0, 1],
    )


# ---------------- Main Run Block ----------------

if run:
    sub_list = [s.strip() for s in subs.split(",") if s.strip()]

    if demo_mode or not have_creds():
        st.warning("Running in **Demo mode** (no Reddit API creds). Using a synthetic sample.")
        df = synthetic_demo_df(rows=300)
    else:
        # Live mode: use your ingestion module
        from src.humourscope.ingest import fetch_comments  # type: ignore
        from src.humourscope.preprocess import clean_text as clean_text_fn  # type: ignore

        with st.spinner("Fetching comments…"):
            frames = [fetch_comments(s, limit_posts=limit_posts, lookback=window) for s in sub_list]
            df = pd.concat(frames, ignore_index=True)
            df = df.rename(columns={"comment_body": "text"})
            df["text_clean"] = df["text"].fillna("").map(clean_text_fn)

    st.success(f"Prepared {len(df):,} comments")
    st.write(df.head())

    # Humour probabilities (HF or heuristic)
    with st.spinner("Scoring humour…"):
        df["humor_prob"] = try_hf_humor_probs(df["text_clean"].tolist())
        df["humor_label"] = (df["humor_prob"] >= 0.5).astype(int)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(humour_mix_plot(df), use_container_width=True)

    with col2:
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
                    title="Share of comments likely humorous (mean probability ≥ 0.5)",
                ),
                use_container_width=True,
            )
        else:
            st.info("No humour labels available to plot yet.")

    # Optional: stacked bar counts
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
            use_container_width=True,
        )

    # Templates / frequent phrases
    st.subheader("Template clusters / frequent phrases")
    info = safe_bertopic(df["text_clean"].tolist())
    st.dataframe(info.head(20), use_container_width=True)

    # Entities (if available)
    st.subheader("In-group entities")
    ents = safe_entities(df["text"].tolist())
    if not ents.empty:
        st.dataframe(ents.head(30), use_container_width=True)
    else:
        st.write("No entities extracted in demo sample.")

    # Compare communities (robust, no duplicate column names)
    st.subheader("Compare communities")
    uniq = sorted(df["subreddit"].unique().tolist())
    if len(uniq) >= 2:
        colA, colB = st.columns(2)
        with colA:
            a = st.selectbox("Community A", uniq, index=0)
        with colB:
            b = st.selectbox("Community B", uniq, index=1)

        def profile_block(name: str):
            subdf = df[df["subreddit"] == name]
            st.markdown(f"#### r/{name}")
            st.metric("Avg humour prob", f"{subdf['humor_prob'].mean():.2f}")
            st.metric("Comments analysed", f"{len(subdf):,}")

            vc = subdf["text_clean"].astype(str).str.split().explode().value_counts().head(10)
            top_terms = vc.reset_index()
            cols = list(top_terms.columns)
            rename_map = {}
            if "index" in cols:
                rename_map["index"] = "token"
            if "text_clean" in cols:
                rename_map["text_clean"] = "token"
            if "count" in cols:
                rename_map["count"] = "freq"
            if 0 in top_terms.columns:
                rename_map[0] = "freq"
            top_terms = top_terms.rename(columns=rename_map)
            # Ensure final schema is ['token','freq'] with unique names
            if "token" in top_terms.columns and "freq" in top_terms.columns:
                top_terms = top_terms[["token", "freq"]]
            else:
                top_terms.columns = ["token", "freq"]
            st.dataframe(top_terms, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            profile_block(a)
        with c2:
            profile_block(b)
    else:
        st.info("Need at least two subreddits to compare.")

    # Export
    st.subheader("Export")
    export_dir = Path("data/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    export_name = f"humourscope_{'_'.join(uniq)}.csv" if uniq else "humourscope_demo.csv"
    path = export_dir / export_name
    df.to_csv(path, index=False)
    st.download_button(
        "Download CSV", data=path.read_bytes(), file_name=export_name, mime="text/csv"
    )

else:
    st.info(
        "Set inputs and click **Fetch & Analyse**. Demo Mode is ON by default (no API required)."
    )
