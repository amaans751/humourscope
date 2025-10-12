from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import praw
from tqdm import tqdm

from .config import CACHE_DIR, get_reddit_creds


def _client() -> praw.Reddit:
    c = get_reddit_creds()
    return praw.Reddit(
        client_id=c.client_id, client_secret=c.client_secret, user_agent=c.user_agent
    )


def fetch_comments(subreddit: str, limit_posts: int = 100, lookback: str = "month") -> pd.DataFrame:
    """
    Fetch top posts + comments within a recent window (day|week|month|year|all).
    Returns a DataFrame of comments with metadata.
    """
    reddit = _client()
    sub = reddit.subreddit(subreddit)

    posts = list(sub.top(time_filter=lookback, limit=limit_posts))
    rows = []
    for post in tqdm(posts, desc=f"{subreddit}:posts"):
        post.comments.replace_more(limit=0)
        for c in post.comments.list():
            rows.append(
                {
                    "subreddit": subreddit,
                    "post_id": post.id,
                    "post_title": post.title,
                    "post_ups": post.ups,
                    "created_utc": c.created_utc,
                    "comment_id": c.id,
                    "comment_body": c.body,
                    "comment_ups": c.ups,
                    "is_submitter": c.is_submitter,
                    "parent_id": c.parent_id,
                    "depth": getattr(c, "depth", None),
                    "permalink": f"https://reddit.com{c.permalink}",
                }
            )
        time.sleep(0.2)  # be nice
    df = pd.DataFrame(rows)
    return df


def cache_dataframe(df: pd.DataFrame, name: str) -> Path:
    path = CACHE_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    meta = {"rows": len(df)}
    (CACHE_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return path
