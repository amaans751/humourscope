import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

# Try .env first
load_dotenv()

DATA_DIR = Path(os.getenv("HUMOURSCOPE_DATA_DIR", "data"))
CACHE_DIR = DATA_DIR / "cache"
EXPORT_DIR = DATA_DIR / "exports"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


class RedditCreds(BaseModel):
    client_id: str
    client_secret: str
    user_agent: str


def get_reddit_creds() -> RedditCreds:
    """
    Read creds from env OR Streamlit secrets if present.
    Returns creds or raises RuntimeError if not set.
    """
    cid = os.getenv("REDDIT_CLIENT_ID")
    csec = os.getenv("REDDIT_CLIENT_SECRET")
    ua = os.getenv("REDDIT_USER_AGENT", "humourscope/0.1 by unknown")

    # Streamlit secrets fallback (won't import if not running in Streamlit)
    if not (cid and csec):
        try:
            import streamlit as st  # type: ignore

            cid = cid or st.secrets.get("REDDIT_CLIENT_ID")
            csec = csec or st.secrets.get("REDDIT_CLIENT_SECRET")
            ua = st.secrets.get("REDDIT_USER_AGENT", ua)
        except Exception:
            pass

    if not cid or not csec:
        raise RuntimeError("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env or st.secrets")
    return RedditCreds(client_id=cid, client_secret=csec, user_agent=ua)
