"""
Optional LLM explanation layer (Layer 3B).

Priority order for generating explanations:
  1. Ollama  — if a local Ollama server is running (OLLAMA_HOST)
  2. Local HuggingFace model via `transformers` — already installed, no sudo,
     downloads model on first use and caches it (~900 MB for the default model)
  3. Template fallback — structured text, always available, no LLM needed

Environment variables:
  OLLAMA_HOST        Ollama server URL  (default: http://localhost:11434)
  OLLAMA_MODEL       Ollama model name  (default: llama3.2)
  LOCAL_LLM_MODEL    HF model for local inference
                     (default: Qwen/Qwen2.5-1.5B-Instruct)
  LOCAL_LLM_ENABLED  Set to "false" to skip local model entirely (default: true)

The LLM is ONLY used for explanation text — it never affects predicted score,
tier, or confidence. Those come from Ridge + KNN.
"""
import logging
import os
import threading
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local-model pipeline cache (loaded once, reused across requests)
# ---------------------------------------------------------------------------
_local_pipe = None
_local_pipe_lock = threading.Lock()
_local_pipe_loading = False  # True while background load is in progress


def preload_local_model():
    """
    Start loading the local model in a background thread.

    Call this at server startup so the model is ready before the first
    prediction request arrives. Falls back to template while loading.
    """
    if not _local_llm_enabled():
        return

    def _load():
        global _local_pipe_loading
        _local_pipe_loading = True
        try:
            logger.info("Pre-warming local LLM in background …")
            _run_local_pipeline("Say hi.")   # trigger load + cache
            logger.info("Local LLM warm and ready.")
        except Exception as e:
            logger.warning("Local LLM pre-warm failed: %s", e)
        finally:
            _local_pipe_loading = False

    t = threading.Thread(target=_load, daemon=True)
    t.start()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_MODEL = "qwen3.5:4b"


def _ollama_host() -> str:
    return os.getenv("OLLAMA_HOST", _DEFAULT_HOST).rstrip("/")


def _ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", _DEFAULT_MODEL)


def _local_model_name() -> str:
    return os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


def _local_llm_enabled() -> bool:
    return os.getenv("LOCAL_LLM_ENABLED", "true").lower() not in ("false", "0", "no")


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
    use_local_llm: bool = False,
) -> str:
    """
    Generate a human-readable explanation for the prediction.

    Priority:
      1. Ollama (if server is reachable) — fast, streaming-friendly
      2. Local HuggingFace model — only when use_local_llm=True (slow, ~8-10s)
      3. Template fallback — always available, instant

    The local LLM is disabled by default for request-path calls to keep
    latency predictable. It is only used when explicitly opted-in (e.g.
    background pre-generation) or when the caller can afford the wait.
    """
    args = (post, prediction, similar_posts, key_factors, brand_stats, brand_alignment_score)

    # 1. Try Ollama (fast, <2s when running)
    try:
        return await _ollama_explanation(*args)
    except Exception as e:
        logger.debug("Ollama not available (%s), falling back", e)

    # 2. Local transformers model — only when explicitly allowed
    if use_local_llm and _local_llm_enabled():
        try:
            return await _local_explanation(*args)
        except Exception as e:
            logger.warning("Local model explanation failed (%s), using template fallback", e)

    # 3. Template fallback (instant)
    return _template_explanation(*args)


# ---------------------------------------------------------------------------
# Ollama path
# ---------------------------------------------------------------------------

async def _ollama_explanation(
    post, prediction, similar_posts, key_factors,
    brand_stats, brand_alignment_score
) -> str:
    host = _ollama_host()
    model = _ollama_model()
    prompt = _build_prompt(post, prediction, similar_posts, key_factors,
                           brand_stats, brand_alignment_score)

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.4,
                    "num_predict": 150,
                },
            },
        )
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "").strip()
        if not content:
            raise ValueError("Empty response from Ollama")
        return content


# ---------------------------------------------------------------------------
# Local HuggingFace model via transformers (no server, no sudo)
# ---------------------------------------------------------------------------

async def _local_explanation(
    post, prediction, similar_posts, key_factors,
    brand_stats, brand_alignment_score
) -> str:
    """
    Generate explanation using a local HuggingFace model.

    The pipeline is loaded once (first call takes ~5-10 seconds to load model
    weights) and reused for all subsequent requests. Uses Apple MPS GPU on Mac
    for fast inference — a 120-word response takes ~3-5 seconds.

    Default model: Qwen/Qwen2.5-1.5B-Instruct (~900 MB, downloads once)
    Override: LOCAL_LLM_MODEL=<any HF model ID>
    """
    import asyncio

    # Build the prompt (same as Ollama path)
    prompt = _build_prompt(post, prediction, similar_posts, key_factors,
                           brand_stats, brand_alignment_score)

    # Run the blocking pipeline call in a thread so it doesn't block the event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_local_pipeline, prompt)
    return result


def _run_local_pipeline(prompt: str) -> str:
    """Blocking call — runs in a thread pool executor."""
    global _local_pipe

    if _local_pipe is None:
        with _local_pipe_lock:
            if _local_pipe is None:  # double-checked locking
                import torch
                from transformers import pipeline

                model_name = _local_model_name()
                logger.info("Loading local LLM: %s (first call only) …", model_name)

                if torch.backends.mps.is_available():
                    device = "mps"
                    dtype = torch.float16
                elif torch.cuda.is_available():
                    device = "cuda"
                    dtype = torch.float16
                else:
                    device = "cpu"
                    dtype = torch.float32

                _local_pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    device=device,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
                logger.info("Local LLM ready on %s.", device)

    messages = [{"role": "user", "content": prompt}]
    output = _local_pipe(
        messages,
        max_new_tokens=180,
        temperature=0.4,
        do_sample=True,
        pad_token_id=_local_pipe.tokenizer.eos_token_id,
    )
    # transformers returns the full conversation; extract the last assistant turn
    generated = output[0]["generated_text"]
    if isinstance(generated, list):
        # Chat format — last element is the assistant response
        return generated[-1]["content"].strip()
    return str(generated).strip()


def _build_prompt(
    post, prediction, similar_posts, key_factors,
    brand_stats, brand_alignment_score
) -> str:
    """Shared prompt builder used by both Ollama and local model paths."""
    brand = post.get("brand", "unknown").replace("_", " ")
    tier = prediction.get("performance_tier", "medium")
    est_er = prediction.get("predicted_engagement_rate", 0)
    brand_median = brand_stats.get("median", 0)
    content_group = prediction.get("model_details", {}).get("content_group", "reel")

    sim_context = ""
    for i, sp in enumerate(similar_posts[:2], 1):
        sim_context += f"  {i}. '{sp['caption_snippet']}' → {sp['engagement_rate']}% ER ({sp['performance_tier']})\n"

    positives = [f for f in key_factors if f["impact"] == "positive"][:2]
    negatives = [f for f in key_factors if f["impact"] == "negative"][:2]
    factor_context = ""
    if positives:
        factor_context += "Positive signals: " + ", ".join(f["description"] for f in positives) + ".\n"
    if negatives:
        factor_context += "Risks: " + ", ".join(f["description"] for f in negatives) + ".\n"

    alignment_note = ""
    if brand_alignment_score is not None and brand_alignment_score < 0.35:
        alignment_note = f"Note: atypical content for {brand} (brand alignment: {brand_alignment_score:.2f}).\n"

    collab_note = ""
    collab_tier = post.get("collaborator_tier", 0)
    if collab_tier > 0:
        tier_labels = {1: "micro-influencer", 2: "mid-tier creator", 3: "macro influencer", 4: "mega celebrity"}
        collab_note = f"Collaborator type: {tier_labels.get(collab_tier, 'unknown')}.\n"

    return (
        f"You are an expert Instagram content strategist for Indian beverage brands.\n\n"
        f"Brand: {brand}\n"
        f"Post type: {content_group} ({post.get('media_type', 'reel')}, {post.get('duration') or 'N/A'}s)\n"
        f"Caption: {(post.get('caption') or '')[:300]}\n"
        f"Visual: {(post.get('visual_summary') or 'Not provided')[:200]}\n"
        f"{collab_note}"
        f"\nPrediction: {tier.upper()} performance\n"
        f"Estimated engagement rate: {est_er:.2f}% (brand median: {brand_median:.2f}%)\n"
        f"\nMost similar historical posts:\n{sim_context}"
        f"{factor_context}"
        f"{alignment_note}"
        f"\nWrite a 3-4 sentence explanation a content manager can act on. Be specific about:\n"
        f"1. Why this post is predicted to perform {tier}\n"
        f"2. What the most similar historical posts tell us\n"
        f"3. One concrete thing they could change to improve performance\n\n"
        f"Do NOT use bullet points. Write in plain, direct prose. Keep it under 120 words."
    )


# ---------------------------------------------------------------------------
# Template fallback (no LLM available)
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
