from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer


def build_topic_model(
    texts: list[str], n_grams: tuple[int, int] = (1, 2)
) -> tuple[BERTopic, pd.DataFrame]:
    emb = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = emb.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    vectorizer_model = CountVectorizer(ngram_range=n_grams, stop_words="english", min_df=5)
    topic_model = BERTopic(
        vectorizer_model=vectorizer_model, calculate_probabilities=False, verbose=False
    )
    topics, probs = topic_model.fit_transform(texts, embeddings)
    topic_info = topic_model.get_topic_info()
    docs = pd.DataFrame({"text": texts, "topic": topics})
    return topic_model, docs.merge(topic_info, left_on="topic", right_on="Topic", how="left")
