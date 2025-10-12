from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class BaselineModel:
    pipe: Pipeline


def train_baseline(texts: List[str], labels: List[int]) -> Tuple[BaselineModel, Dict[str, Any]]:
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, n_jobs=1, class_weight="balanced"),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True)
    return BaselineModel(pipe), report


def predict_baseline(model: BaselineModel, texts: List[str]) -> List[int]:
    return model.pipe.predict(texts)


class HFHumor:
    """
    Wrap Humor-Research/humor-detection-comb-23 (binary).
    """

    def __init__(self, model_name: str = "Humor-Research/humor-detection-comb-23"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict_proba(self, texts: List[str]) -> List[float]:
        probs = []
        for i in range(0, len(texts), 16):
            batch = texts[i : i + 16]
            tok = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**tok).logits
            p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().tolist()
            probs.extend(p)
        return probs

    def predict_label(self, texts: List[str], thresh: float = 0.5) -> List[int]:
        return [int(p >= thresh) for p in self.predict_proba(texts)]
