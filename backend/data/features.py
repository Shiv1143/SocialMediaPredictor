"""
Feature engineering for structured (non-embedding) features.
These features feed the Ridge regression component of the hybrid model.
"""
import re
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd

# India Standard Time offset: UTC+5:30
_IST = timezone(timedelta(hours=5, minutes=30))

# Matches most Unicode emoji blocks
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F9FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

# Ordered list of feature names — order matters for Ridge coefficients
FEATURE_COLUMNS = [
    # Media type
    "is_reel",
    "is_post",
    "is_album",
    "is_static",          # views == 0 flag (post or album); critical: different ER denominator
    # Duration
    "duration_seconds",
    "duration_short",    # ≤ 15 s
    "duration_medium",   # 16–60 s
    "duration_long",     # > 60 s
    # Collaboration
    "is_collaborated",
    "num_collaborators",
    "is_reel_collab",    # interaction: reel × collaborated (strongest engagement combo)
    # Timing (all in IST — Indian Standard Time UTC+5:30)
    "post_hour_ist",
    "post_day_of_week",
    "post_month",
    "is_weekend",
    "is_prime_time",     # 17:00–22:00 IST — peak Instagram browsing hours in India
    # Audience
    "followers_log",
    "followers_bucket",  # 0=<10k, 1=10k–100k, 2=100k–1M, 3=1M+
    # Caption
    "caption_word_count",
    "caption_char_count",
    "emoji_count",
    "hashtag_count",
    "mention_count",
    "has_question",
    "has_exclamation",
]


def extract_features(row: dict) -> dict:
    """Extract structured features from a single post dict."""
    f = {}

    # --- Media type ---
    mt = (row.get("media_type") or "post").lower()
    f["is_reel"] = int(mt == "reel")
    f["is_post"] = int(mt == "post")
    f["is_album"] = int(mt == "album")
    # is_static captures the different ER denominator (views=0 → followers-based ER)
    # It can be supplied directly or inferred from media_type
    f["is_static"] = int(row.get("is_static", 0) or mt in ("post", "album"))

    # --- Duration ---
    duration = row.get("duration") or 0
    f["duration_seconds"] = float(duration)
    f["duration_short"] = int(0 < duration <= 15)
    f["duration_medium"] = int(15 < duration <= 60)
    f["duration_long"] = int(duration > 60)

    # --- Collaboration ---
    is_collaborated = int(bool(row.get("is_collaborated", False)))
    num_collabs = int(row.get("num_collaborators") or 0)
    f["is_collaborated"] = is_collaborated
    f["num_collaborators"] = num_collabs
    # Interaction: reel + collaboration is the strongest organic-reach combo in Instagram
    f["is_reel_collab"] = int(f["is_reel"] and is_collaborated)

    # --- Timing (converted to IST) ---
    # All timestamps in the dataset are UTC. Indian social media activity peaks
    # 17:00–22:00 IST. Using raw UTC hours would make the peak appear at 11:30–16:30
    # UTC, shifting the feature by 5.5 hours and losing the prime-time signal.
    created_at = row.get("created_at")
    if created_at and isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except Exception:
            created_at = None

    if created_at and hasattr(created_at, "hour"):
        # Convert to IST regardless of original tzinfo
        if created_at.tzinfo is not None:
            ist_dt = created_at.astimezone(_IST)
        else:
            # Treat naive datetimes as UTC
            ist_dt = created_at.replace(tzinfo=timezone.utc).astimezone(_IST)
        f["post_hour_ist"] = ist_dt.hour
        f["post_day_of_week"] = ist_dt.weekday()
        f["post_month"] = ist_dt.month
        f["is_weekend"] = int(ist_dt.weekday() >= 5)
        f["is_prime_time"] = int(17 <= ist_dt.hour <= 22)
    else:
        # Sensible defaults when timestamp is missing (Indian evening default)
        f["post_hour_ist"] = 18
        f["post_day_of_week"] = 2
        f["post_month"] = 6
        f["is_weekend"] = 0
        f["is_prime_time"] = 1

    # --- Audience ---
    followers = int(row.get("followers") or 0)
    f["followers_log"] = float(np.log1p(followers))
    if followers < 10_000:
        f["followers_bucket"] = 0
    elif followers < 100_000:
        f["followers_bucket"] = 1
    elif followers < 1_000_000:
        f["followers_bucket"] = 2
    else:
        f["followers_bucket"] = 3

    # --- Caption text ---
    caption = row.get("caption") or ""
    words = caption.split()
    f["caption_word_count"] = len(words)
    f["caption_char_count"] = len(caption)
    f["emoji_count"] = len(_EMOJI_RE.findall(caption))
    f["hashtag_count"] = caption.count("#")
    f["mention_count"] = caption.count("@")
    f["has_question"] = int("?" in caption)
    f["has_exclamation"] = int("!" in caption)

    return f


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a feature matrix DataFrame from a posts DataFrame."""
    rows = [extract_features(row.to_dict()) for _, row in df.iterrows()]
    feat_df = pd.DataFrame(rows, index=df.index)
    # Ensure column order matches FEATURE_COLUMNS
    return feat_df[FEATURE_COLUMNS]


def build_text_for_embedding(row: dict) -> str:
    """
    Build a rich text representation of a post for sentence-transformer embedding.
    Combines brand identity, media type, caption, and pre-computed visual summary.

    Token-budget note:
      all-MiniLM-L6-v2 has a hard limit of 256 tokens (~1 token ≈ 4-5 chars for
      English/Romanized Hindi).  We budget ~900 chars total to stay safely under
      the limit while preserving the most signal-dense content (caption first,
      then visual summary).  Texts exceeding this are truncated *here* rather than
      silently inside the model, so the truncation is deterministic and visible.
    """
    MAX_CAPTION_CHARS = 300
    MAX_VISUAL_CHARS = 500
    MAX_TOTAL_CHARS = 900

    parts = []

    brand = (row.get("brand") or "").replace("_", " ").strip()
    if brand:
        parts.append(f"Brand: {brand}.")

    media_type = (row.get("media_type") or "").strip()
    if media_type:
        parts.append(f"Content type: {media_type}.")

    duration = row.get("duration")
    if duration:
        parts.append(f"Duration: {duration}s.")

    if row.get("is_collaborated"):
        nc = row.get("num_collaborators", 1)
        parts.append(f"Collab post ({nc} collaborator(s)).")

    caption = (row.get("caption") or "").strip()
    if caption:
        # Caption carries more signal than visual summary — give it priority
        cap_truncated = caption[:MAX_CAPTION_CHARS]
        if len(caption) > MAX_CAPTION_CHARS:
            cap_truncated += "…"
        parts.append(f"Caption: {cap_truncated}")

    visual = (row.get("visual_summary") or "").strip()
    if visual:
        vis_truncated = visual[:MAX_VISUAL_CHARS]
        if len(visual) > MAX_VISUAL_CHARS:
            vis_truncated += "…"
        parts.append(f"Visual: {vis_truncated}")

    text = " ".join(parts)
    # Hard cap on total length as final safety net
    if len(text) > MAX_TOTAL_CHARS:
        text = text[:MAX_TOTAL_CHARS] + "…"
    return text


# Human-readable descriptions for each feature (used in explanation output)
FEATURE_DESCRIPTIONS = {
    "is_reel": "Content is a Reel (video, views-based ER)",
    "is_post": "Content is a static image post (followers-based ER)",
    "is_album": "Content is a carousel/album (followers-based ER)",
    "is_static": "Static content (post or album) — different engagement denominator than reels",
    "duration_seconds": "Video duration in seconds",
    "duration_short": "Short video (≤15 s) — snappy, high completion rate",
    "duration_medium": "Medium video (15–60 s) — standard format",
    "duration_long": "Long video (>60 s) — requires sustained viewer attention",
    "is_collaborated": "Post features a collaboration (expands reach to collab audience)",
    "num_collaborators": "Number of collaborators tagged",
    "is_reel_collab": "Reel with collaboration — historically the highest-reach combo",
    "post_hour_ist": "Hour of day (IST) the post was published",
    "post_day_of_week": "Day of the week (0=Mon … 6=Sun)",
    "post_month": "Month of the year",
    "is_weekend": "Posted on a weekend (Saturday or Sunday)",
    "is_prime_time": "Posted in prime time (17:00–22:00 IST) — peak Indian Instagram hours",
    "followers_log": "Brand's follower count (log-scaled)",
    "followers_bucket": "Follower tier (<10 k / 10 k–100 k / 100 k–1 M / 1 M+)",
    "caption_word_count": "Number of words in the caption",
    "caption_char_count": "Number of characters in the caption",
    "emoji_count": "Number of emojis in the caption",
    "hashtag_count": "Number of hashtags (#) in the caption",
    "mention_count": "Number of @mentions in the caption",
    "has_question": "Caption contains a question (drives comments)",
    "has_exclamation": "Caption uses exclamation marks (energetic tone)",
}
