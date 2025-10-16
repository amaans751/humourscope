import re
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r"https?://\S+")
USER_RE = re.compile(r"\bu/\w+")
SUB_RE = re.compile(r"\br/\w+")
WS_RE = re.compile(r"\s+")

# AutoModerator / boilerplate fragments:
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
    s = BOILER_RE.sub(" ", s)  # strip automod boilerplate & domain crumbs
    s = USER_RE.sub(" ", s)
    s = SUB_RE.sub(" ", s)
    s = s.lower()
    s = re.sub(r"[^\w\s!?.,;:()\-']", " ", s)
    s = WS_RE.sub(" ", s).strip()
    return s


def add_style_markers(df: pd.DataFrame, col: str = "text") -> pd.DataFrame:
    def markers(t: str):
        ex = t.count("!")
        q = t.count("?")
        ell = t.count("...")
        caps = sum(1 for w in t.split() if len(w) > 3 and w.isupper())
        return pd.Series({"m_excl": ex, "m_quest": q, "m_ellips": ell, "m_caps": caps})

    feats = df[col].apply(markers)
    return pd.concat([df, feats], axis=1)


def tokenize(texts: Iterable[str]) -> Iterable[str]:
    stop = ENGLISH_STOP_WORDS
    for t in texts:
        yield " ".join([w for w in t.split() if w not in stop])
