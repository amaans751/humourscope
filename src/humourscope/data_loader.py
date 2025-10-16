# src/humourscope/data_loader.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

# ---------- Common schema ----------
NEEDED_COLS = [
    "subreddit",
    "post_id",
    "comment_id",
    "text",
    "comment_ups",
    "created_utc",
    "permalink",
]


def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Map common Reddit columns to our app schema and enforce order/types."""
    colmap = {
        "body": "text",
        "score": "comment_ups",
        "id": "comment_id",
        "link_id": "post_id",
        "created_utc": "created_utc",
        "permalink": "permalink",
        "subreddit": "subreddit",
        "title": "text",  # some HF datasets only have title
        "selftext": "text",
        "ups": "comment_ups",
    }
    for src, tgt in colmap.items():
        if src in df.columns and tgt not in df.columns:
            df[tgt] = df[src]

    for c in NEEDED_COLS:
        if c not in df.columns:
            df[c] = 0 if c == "comment_ups" else ""

    # Keep a stable column order
    df = df[NEEDED_COLS].copy()

    # Types (lenient – created_utc may be int or timestamp string upstream)
    df["text"] = df["text"].fillna("").astype(str)
    df["comment_id"] = df["comment_id"].astype(str)
    df["post_id"] = df["post_id"].astype(str)
    return df


# ---------- Hugging Face loader ----------
def load_hf_sample(
    dataset: str = "SocialGrep/one-million-reddit-jokes",
    split: str = "train",
    n: int = 5000,
) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    df = pd.DataFrame(ds.select(range(min(n, len(ds)))))
    df = _normalize_cols(df)

    # Ensure we have a subreddit; many HF joke sets lack it
    if "subreddit" not in df.columns or df["subreddit"].fillna("").eq("").all():
        df["subreddit"] = "rJokes"  # <-- add this line
    # If comment_id missing, fallback to post_id
    if "comment_id" not in df or df["comment_id"].fillna("").eq("").all():
        df["comment_id"] = df["post_id"].astype(str)

    return df


# ---------- BigQuery: direct SQL from app ----------
def load_bigquery_query(
    sql: str,
    project_id: str,
    use_bqstorage: bool = True,
) -> pd.DataFrame:
    """
    Run a SQL query against BigQuery public datasets.

    Requires:
      pip install pandas-gbq google-cloud-bigquery pyarrow

    Auth:
      gcloud auth application-default login
      # or set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON
    """
    import pandas_gbq  # lazy import inside function

    df = pandas_gbq.read_gbq(
        sql,
        project_id=project_id,
        dialect="standard",
        use_bqstorage_api=use_bqstorage,
    )
    return _normalize_cols(df)


# ---------- BigQuery: load exported file ----------
def load_from_file(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV/Parquet previously exported from BigQuery Console (or otherwise),
    then normalize to the app schema.
    """
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    return _normalize_cols(df)


# ---------- Helper to build a Reddit comments SQL ----------
def build_reddit_comments_sql(
    table_qualified: str,
    subreddits: Sequence[str],
    start_ts_utc: Optional[int] = None,
    end_ts_utc: Optional[int] = None,
    limit: Optional[int] = 50000,
) -> str:
    """
    Create a SQL query for a reddit comments table.

    Args:
      table_qualified: e.g. "myproj.myds.mytable" or "reddit.reddit_comments.2024_07"
      subreddits: ["funny", "antimemes"]
      start_ts_utc/end_ts_utc: UNIX epoch seconds (optional)
      limit: LIMIT row count (optional)
    """
    subs = ", ".join(f"'{s}'" for s in subreddits) or "'funny'"
    where = [f"subreddit IN ({subs})"]
    if start_ts_utc is not None:
        where.append(f"CAST(created_utc AS INT64) >= {int(start_ts_utc)}")
    if end_ts_utc is not None:
        where.append(f"CAST(created_utc AS INT64) <= {int(end_ts_utc)}")
    where_clause = " AND ".join(where)

    sql = f"""
    SELECT
      subreddit,
      CAST(link_id AS STRING)        AS post_id,
      CAST(id AS STRING)             AS comment_id,
      CAST(body AS STRING)           AS text,
      CAST(score AS INT64)           AS comment_ups,
      CAST(created_utc AS INT64)     AS created_utc,
      CAST(permalink AS STRING)      AS permalink
    FROM `{table_qualified}`
    WHERE {where_clause}
    """.strip()

    if limit:
        sql += f"\nLIMIT {int(limit)}"
    return sql
