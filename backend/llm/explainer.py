"""
Optional LLM explanation layer (Layer 3B).

Uses a locally-running Ollama model to generate a rich natural-language
explanation that a content manager can act on — instead of raw coefficient
numbers.

Configuration (environment variables):
  OLLAMA_HOST   URL of the Ollama server (default: http://localhost:11434)
  OLLAMA_MODEL  Model to use (default: llama3.2)

Falls back to a structured template explanation if Ollama is not reachable
or if the call fails for any reason.

The LLM is ONLY used for explanation generation — it never affects the
predicted score, tier, or confidence. Those come from Ridge + KNN.

Note: The OpenAI image vision path in main.py (_vision_summary) is separate
and still uses OPENAI_API_KEY if set.
"""
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_MODEL = "llama3.2"


def _ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", _DEFAULT_HOST).rstrip("/")


def _ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", _DEFAULT_MODEL)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def generate_explanation(
    post: dict,
    prediction: dict,
    similar_posts: list,
    key_factors: list,
    brand_stats: dict,
    brand_alignment_score: Optional[float] = None,
) -> str:
    """
    Generate a human-readable explanation for the prediction.

    Tries Ollama first; falls back to a structured template if Ollama is
    not running or returns an error.
    """
    try:
        return await _ollama_explanation(
            post, prediction, similar_posts, key_factors,
            brand_stats, brand_alignment_score
        )
    except Exception as e:
        logger.warning("Ollama explanation failed (%s), using template fallback", e)

    return _template_explanation(
        post, prediction, similar_posts, key_factors,
        brand_stats, brand_alignment_score
    )


# ---------------------------------------------------------------------------
# Ollama path
# ---------------------------------------------------------------------------

async def _ollama_explanation(
    post, prediction, similar_posts, key_factors,
    brand_stats, brand_alignment_score
) -> str:
    host = _ollama_host()
    model = _ollama_model()

    brand = post.get("brand", "unknown").replace("_", " ")
    tier = prediction.get("performance_tier", "medium")
    est_er = prediction.get("predicted_engagement_rate", 0)
    brand_median = brand_stats.get("median", 0)
    content_group = prediction.get("model_details", {}).get("content_group", "reel")

    # Build similar posts context (top 3)
    sim_context = ""
    for i, sp in enumerate(similar_posts[:3], 1):
        sim_context += (
            f"  {i}. '{sp['caption_snippet']}' "
            f"→ {sp['engagement_rate']}% ER ({sp['performance_tier']})\n"
        )

    # Positive / negative factor context
    positives = [f for f in key_factors if f["impact"] == "positive"][:3]
    negatives = [f for f in key_factors if f["impact"] == "negative"][:3]
    factor_context = ""
    if positives:
        factor_context += "Positive signals: " + ", ".join(f["description"] for f in positives) + ".\n"
    if negatives:
        factor_context += "Risks: " + ", ".join(f["description"] for f in negatives) + ".\n"

    # Off-strategy note
    alignment_note = ""
    if brand_alignment_score is not None and brand_alignment_score < 0.35:
        alignment_note = (
            f"Note: This post's content is atypical for {brand} "
            f"(brand alignment score: {brand_alignment_score:.2f}). "
            "It may be exploring a new content direction."
        )

    # Collaborator tier note
    collab_note = ""
    collab_tier = post.get("collaborator_tier", 0)
    if collab_tier > 0:
        tier_labels = {1: "micro-influencer", 2: "mid-tier creator",
                       3: "macro influencer", 4: "mega celebrity"}
        collab_note = f"Collaborator type: {tier_labels.get(collab_tier, 'unknown')}."

    prompt = f"""You are an expert Instagram content strategist for Indian beverage brands.

Brand: {brand}
Post type: {content_group} ({post.get('media_type', 'reel')}, {post.get('duration') or 'N/A'}s)
Caption: {(post.get('caption') or '')[:300]}
Visual: {(post.get('visual_summary') or 'Not provided')[:200]}
{collab_note}

Prediction: {tier.upper()} performance
Estimated engagement rate: {est_er:.2f}% (brand median: {brand_median:.2f}%)

Most similar historical posts:
{sim_context}
{factor_context}
{alignment_note}

Write a 3-4 sentence explanation a content manager can act on. Be specific about:
1. Why this post is predicted to perform {tier}
2. What the most similar historical posts tell us
3. One concrete thing they could change to improve performance

Do NOT use bullet points. Write in plain, direct prose. Keep it under 120 words."""

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "num_predict": 200,
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["response"].strip()


# ---------------------------------------------------------------------------
# Template fallback (Ollama not available)
# ---------------------------------------------------------------------------

def _template_explanation(
    post, prediction, similar_posts, key_factors,
    brand_stats, brand_alignment_score
) -> str:
    brand = post.get("brand", "unknown").replace("_", " ").title()
    tier = prediction.get("performance_tier", "medium")
    est_er = prediction.get("predicted_engagement_rate", 0)
    brand_median = brand_stats.get("median", 0)
    content_group = prediction.get("model_details", {}).get("content_group", "reel")

    tier_label = {"low": "below average", "medium": "average", "high": "above average"}[tier]

    positives = [f["description"] for f in key_factors if f["impact"] == "positive"][:2]
    negatives = [f["description"] for f in key_factors if f["impact"] == "negative"][:2]

    parts = [
        f"This {content_group} is predicted to perform {tier_label} for {brand} "
        f"(est. {est_er:.2f}% ER vs brand median {brand_median:.2f}%)."
    ]

    if positives:
        parts.append("Strengths: " + " and ".join(positives) + ".")
    if negatives:
        parts.append("Watch out for: " + " and ".join(negatives) + ".")

    if similar_posts:
        top = similar_posts[0]
        parts.append(
            f"Most similar historical post: '{top['caption_snippet']}' "
            f"({top['performance_tier']}, {top['engagement_rate']}% ER)."
        )

    if brand_alignment_score is not None and brand_alignment_score < 0.35:
        parts.append(
            f"This content is atypical for {brand}'s usual style "
            f"(alignment: {brand_alignment_score:.2f}) — results may be harder to predict."
        )

    collab_tier = post.get("collaborator_tier", 0)
    if collab_tier >= 3:
        parts.append(
            "A high-profile collaborator has been factored in — "
            "this may significantly boost organic reach beyond the base prediction."
        )

    return " ".join(parts)
