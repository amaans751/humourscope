import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer

_nlp = None


def nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # if not present: python -m spacy download en_core_web_sm
            raise RuntimeError("Run: python -m spacy download en_core_web_sm")
    return _nlp


def extract_entities(texts: list[str]) -> pd.DataFrame:
    n = nlp()
    rows = []
    for t in texts:
        doc = n(t)
        for ent in doc.ents:
            rows.append({"entity": ent.text, "label": ent.label_})
    return (
        pd.DataFrame(rows)
        .value_counts(["entity", "label"])
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def collocations(texts: list[str], top_k: int = 50):
    vec = CountVectorizer(ngram_range=(2, 3), stop_words="english", min_df=3)
    X = vec.fit_transform(texts)
    freqs = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    pairs = sorted(zip(vocab, freqs), key=lambda x: x[1], reverse=True)[:top_k]
    return pd.DataFrame(pairs, columns=["phrase", "count"])
