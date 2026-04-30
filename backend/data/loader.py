import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the assignment dataset JSON into a clean, flat DataFrame."""
    with open(path) as f:
        raw = json.load(f)

    rows = [_parse_record(r) for r in raw]
    rows = [r for r in rows if r is not None]

    df = pd.DataFrame(rows)
    df = _add_target_columns(df)
    return df


def _parse_record(record: dict) -> Optional[dict]:
    try:
        d = record["data"]
        mc = d.get("metadata_content", {}) or {}
        eng = d.get("engagements", {}) or {}
        ps = d.get("profile_stats", {}) or {}
        media = d.get("media", []) or []

        # Collect all visual summaries from media items
        summaries = [m["summary"] for m in media if m.get("summary")]
        visual_summary = " ".join(summaries).strip()

        # Parse timestamp
        created_at = _parse_datetime(mc.get("created_at"))

        collaborators = mc.get("collaborators") or []
        if not isinstance(collaborators, list):
            collaborators = []

        engagement_rate = eng.get("engagement_rate", 0.0) or 0.0
        views = eng.get("views", 0)
        # views == 0 is valid for static posts, not missing data

        return {
            "post_id": d.get("id", ""),
            "brand": ps.get("username", "unknown"),
            "caption": mc.get("caption", "") or "",
            "visual_summary": visual_summary,
            "media_type": mc.get("media_name", "post"),  # reel / post / album
            "duration": mc.get("duration"),  # seconds; None for non-reels
            "is_collaborated": bool(mc.get("is_collaborated_post", False)),
            "collaborators": collaborators,
            "num_collaborators": len(collaborators),
            "created_at": created_at,
            "followers": ps.get("followers", 0) or 0,
            "views": views if views is not None else 0,
            "likes": eng.get("likes", 0) or 0,
            "comments": eng.get("comments", 0) or 0,
            "shares": eng.get("shares", 0) or 0,
            "engagement_rate": float(engagement_rate),
            "url": d.get("url", ""),
        }
    except Exception:
        return None


def _parse_datetime(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add brand-normalized percentile score and performance tier.

    Two important design decisions:

    1. Separate normalisation by content type:
       Reels use views as the engagement_rate denominator
       (ER = (likes+comments+shares) / views * 100).
       Static posts and albums use followers as the denominator
       (ER = (likes+comments+shares) / followers * 100).
       These are fundamentally different metrics and must NOT be ranked
       against each other.  We compute a within-brand, within-content-type
       percentile rank and combine them into one target column.

    2. Winsorise extreme ER outliers before ranking:
       Several posts have ER > 50% (e.g. 153% for a Pepsi post, 157% for
       Thumsup).  These are almost certainly viral micro-events or data
       anomalies.  If left in, they consume the entire top tier and make
       the target distribution effectively binary (everyone vs one outlier).
       We cap at the 98th percentile per (brand, content_type) group
       *for ranking purposes only* — the raw engagement_rate is preserved.
    """
    df = df.copy()

    # Flag static posts (views == 0 → post or album; views > 0 → reel)
    df["is_static"] = (df["views"] == 0).astype(int)
    # Content type group: 'reel' vs 'static' (post + album)
    df["content_group"] = df["is_static"].map({0: "reel", 1: "static"})

    # Winsorise: cap ER at p98 per (brand, content_group) for ranking
    def _winsorise(series: pd.Series) -> pd.Series:
        cap = series.quantile(0.98)
        return series.clip(upper=cap)

    df["er_for_ranking"] = df.groupby(["brand", "content_group"])[
        "engagement_rate"
    ].transform(_winsorise)

    # Brand+content-type normalised percentile rank (0–1) — primary target
    df["brand_norm_score"] = df.groupby(["brand", "content_group"])[
        "er_for_ranking"
    ].transform(lambda x: x.rank(pct=True, method="average"))

    # Three-tier label
    df["performance_tier"] = df["brand_norm_score"].apply(_score_to_tier)

    return df.drop(columns=["er_for_ranking"])


def _score_to_tier(score: float) -> str:
    if score <= 0.33:
        return "low"
    elif score <= 0.67:
        return "medium"
    return "high"
