# src/humourscope/lexicon.py
from __future__ import annotations

import re
from typing import List

import pandas as pd

# Optional spaCy; we handle absence gracefully
try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover
    spacy = None  # type: ignore

from sklearn.feature_extraction import text as sktext
from sklearn.feature_extraction.text import CountVectorizer

# ---------------- Stopwords & tokenization ----------------

DOMAIN_JUNK = {
    # domain/Reddit crumbs & automod boilerplate
    "https",
    "http",
    "www",
    "reddit",
    "com",
    "wiki",
    "rules",
    "automatically",
    "moderators",
    "subreddit",
    "message",
    "compose",
    "bot",
    "action",
    "performed",
    "contact",
}

# Base English stopwords + domain junk
EN_STOP = frozenset(sktext.ENGLISH_STOP_WORDS).union(DOMAIN_JUNK)

# letters only, >=3 chars
TOKEN_RE = re.compile(r"(?u)\b[a-z]{3,}\b")


def tokenize_meaningful(text: str) -> List[str]:
    """Lowercase tokens excluding stopwords/domain junk; safe on non-str."""
    if not isinstance(text, str):
        return []
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in EN_STOP]


# ---------------- NER helpers (lazy spaCy) ----------------

_nlp = None  # set once


def nlp():
    """Return a cached spaCy English model or raise a helpful error."""
    global _nlp
    if _nlp is not None:
        return _nlp
    if spacy is None:
        raise RuntimeError(
            "spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_sm"
        )
    try:
        _nlp = spacy.load("en_core_web_sm")  # type: ignore[call-arg]
    except OSError as e:
        raise RuntimeError(
            "spaCy model missing. Run: python -m spacy download en_core_web_sm"
        ) from e
    return _nlp


def extract_entities(texts: List[str]) -> pd.DataFrame:
    """Extract entity counts from a list of texts. Returns empty on failure."""
    try:
        n = nlp()
    except Exception:
        # No spaCy available; return empty frame
        return pd.DataFrame(columns=["entity", "label", "count"])

    rows = []
    for t in texts:
        if not isinstance(t, str) or not t:
            continue
        doc = n(t)
        for ent in doc.ents:
            rows.append({"entity": ent.text, "label": ent.label_})

    if not rows:
        return pd.DataFrame(columns=["entity", "label", "count"])

    return (
        pd.DataFrame(rows)
        .value_counts(["entity", "label"])
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


# ---------------- Collocations (frequent phrases) ----------------


def collocations(texts: List[str], top_k: int = 50) -> pd.DataFrame:
    """Return top 2–3 gram phrases with custom stopwords & token pattern."""
    docs = [t.strip() for t in texts if isinstance(t, str) and t.strip()]
    if not docs:
        return pd.DataFrame(columns=["phrase", "count"])

    vec = CountVectorizer(
        ngram_range=(2, 3),
        stop_words=EN_STOP,
        token_pattern=TOKEN_RE.pattern,
        min_df=3,
        max_df=0.30,
    )
    X = vec.fit_transform(docs)
    if X.shape[1] == 0:
        return pd.DataFrame(columns=["phrase", "count"])

    freqs = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    pairs = sorted(zip(vocab, freqs), key=lambda x: x[1], reverse=True)[:top_k]
    return pd.DataFrame(pairs, columns=["phrase", "count"])
