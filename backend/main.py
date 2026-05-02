import base64
import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import BASE_DIR, MODELS_DIR
from backend.llm.explainer import generate_explanation, preload_local_model
from backend.models.predictor import HybridPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Social Media Performance Predictor",
    description="Predicts Instagram post engagement for beverage brands.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance — loaded once at startup
_predictor: Optional[HybridPredictor] = None

# BLIP vision model cache — loaded once in background at startup
_blip_processor = None
_blip_model = None
_blip_lock = threading.Lock()


def _preload_blip():
    """Load BLIP image-captioning model in a background thread at startup."""
    global _blip_processor, _blip_model
    try:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        model_id = os.getenv("BLIP_MODEL", "Salesforce/blip-image-captioning-base")
        logger.info("Pre-warming BLIP vision model (%s) …", model_id)
        with _blip_lock:
            _blip_processor = BlipProcessor.from_pretrained(model_id)
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            _blip_model = BlipForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16
            ).to(device)
        logger.info("BLIP vision model ready on %s.", device)
    except Exception as e:
        logger.warning("BLIP pre-warm failed: %s", e)


@app.on_event("startup")
async def startup():
    global _predictor
    model_path = MODELS_DIR / "hybrid_predictor.joblib"
    if model_path.exists():
        logger.info("Loading trained model …")
        _predictor = HybridPredictor.load(model_path)
        logger.info("Model ready.")
    else:
        logger.warning(
            "No trained model found at %s — run `python train.py` first.", model_path
        )
    # Pre-warm local LLM and vision model in background
    preload_local_model()
    threading.Thread(target=_preload_blip, daemon=True).start()


def _get_predictor() -> HybridPredictor:
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run `python train.py` first, then restart the server.",
        )
    return _predictor


# ------------------------------------------------------------------
# Static frontend
# ------------------------------------------------------------------

_frontend_dir = BASE_DIR / "frontend"
if _frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    index = _frontend_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "API is running. See /docs for the API reference."}


# ------------------------------------------------------------------
# Health & metadata
# ------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health():
    from backend.llm.explainer import _local_pipe, _local_pipe_loading
    llm_status = "ready" if _local_pipe is not None else ("loading" if _local_pipe_loading else "unavailable")
    vision_status = "ready" if _blip_model is not None else "loading"
    return {
        "status": "ok",
        "model_loaded": _predictor is not None,
        "llm_status": llm_status,
        "vision_status": vision_status,
    }


@app.get("/brands", tags=["meta"])
async def list_brands():
    """Return all brands the model knows about."""
    if _predictor is None:
        from backend.config import KNOWN_BRANDS
        return {"brands": KNOWN_BRANDS}
    brands = sorted(b for b in _predictor.brand_stats if not b.startswith("_"))
    return {"brands": brands}


@app.get("/api/dataset/stats", tags=["meta"])
async def dataset_stats():
    """Per-brand engagement statistics derived from the training dataset."""
    p = _get_predictor()
    out = {}
    for brand, stats in p.brand_stats.items():
        if brand.startswith("_"):
            continue
        out[brand] = {
            "n_posts": stats.get("n_posts"),
            "median_engagement_rate": round(stats.get("median", 0), 2),
            "mean_engagement_rate": round(stats.get("mean", 0), 2),
            "p33_engagement_rate": round(stats.get("p33", 0), 2),
            "p67_engagement_rate": round(stats.get("p67", 0), 2),
        }
    return {"brand_stats": out}


@app.get("/api/evaluation", tags=["evaluation"])
async def get_evaluation():
    """Return pre-computed evaluation results (generated by train.py)."""
    eval_path = MODELS_DIR / "evaluation_results.json"
    if not eval_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation results not found. Run `python train.py` first.",
        )
    with open(eval_path) as f:
        return json.load(f)


# ------------------------------------------------------------------
# Prediction — JSON body
# ------------------------------------------------------------------

class PredictRequest(BaseModel):
    caption: str
    brand: str
    media_type: str = "reel"                         # reel | post | album
    duration: Optional[int] = None                   # seconds (reels only)
    is_collaborated: bool = False
    collaborators: Optional[List[str]] = []
    collaborator_follower_count: Optional[int] = None  # used for tier-based score boost
    posted_at: Optional[str] = None                  # ISO-8601 datetime string
    visual_summary: Optional[str] = None
    followers: Optional[int] = None                  # override brand default


class SeedPost(BaseModel):
    caption: str
    engagement_rate: float
    media_type: str = "reel"
    duration: Optional[int] = None
    is_collaborated: bool = False
    collaborators: Optional[List[str]] = []
    created_at: Optional[str] = None
    visual_summary: Optional[str] = None


class RegisterBrandRequest(BaseModel):
    brand: str
    followers: int
    seed_posts: List[SeedPost]


@app.post("/api/predict", tags=["prediction"])
async def predict_json(request: PredictRequest):
    """
    Predict engagement for a new post draft.

    - **caption**: Post caption text (required)
    - **brand**: Brand username, e.g. `cocacola_india` (required)
    - **media_type**: `reel` | `post` | `album` (default: `reel`)
    - **duration**: Video length in seconds (reels only)
    - **is_collaborated**: Whether the post is a collab
    - **collaborators**: List of collaborator usernames
    - **collaborator_follower_count**: Follower count of the main collaborator (enables tier-based score boost)
    - **posted_at**: Planned publish time (ISO-8601); affects time-of-day features
    - **visual_summary**: Text description of the creative (from a vision model or manual)
    - **followers**: Override the brand's default follower count
    """
    p = _get_predictor()
    post = _build_post_dict(request)
    try:
        result = p.predict(post)
        result = await _enrich_explanation(post, result, p)
        return result
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/predict/stream", tags=["prediction"])
async def predict_stream(request: PredictRequest):
    """
    Streaming version of /api/predict.

    Returns a stream of newline-delimited JSON events:
      1. {"event": "prediction", ...full prediction object...}   — emitted immediately (~330ms)
      2. {"event": "explanation_token", "token": "..."}          — one per LLM token as generated
      3. {"event": "done"}                                        — stream complete

    The frontend can show the prediction result instantly and fill in the
    explanation text progressively as tokens arrive.
    """
    p = _get_predictor()
    post = _build_post_dict(request)

    try:
        result = p.predict(post)
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc))

    async def event_stream():
        # ── Event 1: emit prediction immediately ──
        yield json.dumps({"event": "prediction", **result}) + "\n"

        # ── Event 2+: stream LLM explanation tokens ──
        exp = result.get("explanation", {})
        brand = post.get("brand", "unknown")
        brand_stats = p.brand_stats.get(brand) or p.brand_stats.get("_global", {})
        alignment = result.get("model_details", {}).get("brand_alignment_score")

        from backend.llm.explainer import _build_prompt, _ollama_host, _ollama_model, _template_explanation
        prompt = _build_prompt(
            post, result,
            exp.get("similar_posts", []),
            exp.get("key_factors", []),
            brand_stats, alignment,
        )
        host = _ollama_host()
        model = _ollama_model()

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST", f"{host}/api/chat",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": True,
                        "think": False,
                        "options": {"temperature": 0.4, "num_predict": 150},
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                yield json.dumps({"event": "explanation_token", "token": token}) + "\n"
                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.warning("Streaming explanation failed (%s), sending template", e)
            fallback = _template_explanation(
                post, result, exp.get("similar_posts", []),
                exp.get("key_factors", []), brand_stats, alignment,
            )
            yield json.dumps({"event": "explanation_token", "token": fallback}) + "\n"

        yield json.dumps({"event": "done"}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/api/analyze-media", tags=["prediction"])
async def analyze_media(
    media_file: Optional[UploadFile] = File(None),
    media_url: Optional[str] = Form(None),
):
    """
    Analyze an uploaded image/video or a URL and return an auto-generated
    visual description. Accepts either a file upload or a public URL.
    """
    try:
        if media_url and media_url.strip():
            summary = await _vision_summary_from_url(media_url.strip())
        elif media_file:
            summary = await _vision_summary(media_file)
        else:
            return {"visual_summary": "", "error": "Provide media_file or media_url"}
        return {"visual_summary": summary}
    except Exception as exc:
        logger.warning("Media analysis failed: %s", exc)
        return {"visual_summary": ""}


@app.post("/brands/register", tags=["brands"])
async def register_brand(request: RegisterBrandRequest):
    """
    Register a new brand (cold-start) or add seed posts to an existing one.

    Provide at least 10 real posts with their actual engagement rates to get
    meaningful within-brand predictions. With fewer posts, cross-brand similarity
    dominates and accuracy is lower.

    After registration, the model is saved to disk automatically.
    """
    p = _get_predictor()
    if len(request.seed_posts) < 3:
        raise HTTPException(
            status_code=422,
            detail="At least 3 seed posts are required. Provide 10+ for best results.",
        )
    try:
        seed_dicts = [sp.model_dump() for sp in request.seed_posts]
        result = p.add_brand(request.brand, request.followers, seed_dicts)
        return {
            "status": "registered",
            **result,
            "message": (
                f"Brand '{request.brand}' registered with {len(request.seed_posts)} posts. "
                "Predictions for this brand will now use within-brand similarity."
            ),
        }
    except Exception as exc:
        logger.exception("Brand registration error")
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------------------------------------------
# Prediction — multipart form (with optional image upload)
# ------------------------------------------------------------------

@app.post("/api/predict/upload", tags=["prediction"])
async def predict_upload(
    caption: str = Form(...),
    brand: str = Form(...),
    media_type: str = Form("reel"),
    duration: Optional[int] = Form(None),
    is_collaborated: bool = Form(False),
    collaborators: str = Form(""),
    collaborator_follower_count: Optional[int] = Form(None),
    posted_at: Optional[str] = Form(None),
    media_file: Optional[UploadFile] = File(None),
    media_url: Optional[str] = Form(None),
    visual_summary: Optional[str] = Form(None),
):
    """
    Same as `/api/predict` but accepts `multipart/form-data`.

    Creative input priority (first wins):
      1. `visual_summary` — user-typed description (overrides everything)
      2. `media_url`      — public URL to an image or video
      3. `media_file`     — uploaded file
    """
    p = _get_predictor()

    # User-provided description takes highest priority
    if visual_summary and visual_summary.strip():
        final_summary = visual_summary.strip()
    elif media_url and media_url.strip():
        try:
            final_summary = await _vision_summary_from_url(media_url.strip())
        except Exception as exc:
            logger.warning("Vision summary from URL failed: %s", exc)
            final_summary = ""
    elif media_file:
        try:
            final_summary = await _vision_summary(media_file)
        except Exception as exc:
            logger.warning("Vision summary failed: %s", exc)
            final_summary = ""
    else:
        final_summary = ""

    collab_list = [c.strip() for c in collaborators.split(",") if c.strip()]

    post = {
        "caption": caption,
        "brand": brand,
        "media_type": media_type,
        "duration": duration,
        "is_collaborated": is_collaborated or bool(collab_list),
        "collaborators": collab_list,
        "num_collaborators": len(collab_list),
        "collaborator_follower_count": collaborator_follower_count,
        "created_at": posted_at,
        "visual_summary": final_summary,
        "followers": p.brand_stats.get(brand, {}).get("median_followers", 100_000),
    }

    try:
        result = p.predict(post)
        result = await _enrich_explanation(post, result, p)
        result["visual_summary_used"] = final_summary
        return result
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

async def _enrich_explanation(post: dict, result: dict, p: HybridPredictor) -> dict:
    """
    Replace the template explanation summary with an LLM-generated one when
    OPENAI_API_KEY is set. Falls back gracefully; never blocks the prediction.
    """
    try:
        exp = result.get("explanation", {})
        brand = post.get("brand", "unknown")
        brand_stats = p.brand_stats.get(brand) or p.brand_stats.get("_global", {})
        alignment = result.get("model_details", {}).get("brand_alignment_score")
        llm_summary = await generate_explanation(
            post=post,
            prediction=result,
            similar_posts=exp.get("similar_posts", []),
            key_factors=exp.get("key_factors", []),
            brand_stats=brand_stats,
            brand_alignment_score=alignment,
        )
        result["explanation"]["summary"] = llm_summary
    except Exception as e:
        logger.warning("Explanation enrichment failed: %s", e)
    return result


def _build_post_dict(req: PredictRequest) -> dict:
    p = _get_predictor()
    collab_list = req.collaborators or []
    return {
        "caption": req.caption,
        "brand": req.brand,
        "media_type": req.media_type,
        "duration": req.duration,
        "is_collaborated": req.is_collaborated or bool(collab_list),
        "collaborators": collab_list,
        "num_collaborators": len(collab_list),
        "collaborator_follower_count": req.collaborator_follower_count,
        "created_at": req.posted_at,
        "visual_summary": req.visual_summary or "",
        "followers": req.followers or p.brand_stats.get(req.brand, {}).get("median_followers", 100_000),
    }


_VISION_PROMPT = (
    "Analyze this Instagram creative briefly for an engagement prediction system. "
    "Describe: 1) what's shown (people, objects, setting), "
    "2) any text overlays or captions visible on screen, "
    "3) brand elements or logos, "
    "4) overall mood and energy. "
    "Keep it under 120 words."
)


async def _vision_summary_from_url(url: str) -> str:
    """
    Fetch media from a public URL and run it through the same vision pipeline
    as an uploaded file. Supports images and videos (mp4, mov, webm, etc.).
    """
    VIDEO_EXTS = {"mp4", "mov", "avi", "webm", "mkv", "m4v"}
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        contents = resp.content
        content_type = resp.headers.get("content-type", "").split(";")[0].strip()

    # Detect video by content-type or URL extension
    ext = url.lower().split("?")[0].rsplit(".", 1)[-1] if "." in url else ""
    is_video = content_type.startswith("video/") or ext in VIDEO_EXTS

    if is_video:
        frames = _extract_video_frames(contents)
        frame_mime = "image/jpeg"
    else:
        frames = [contents]
        frame_mime = content_type or "image/jpeg"

    if not frames:
        return ""

    summary = await _ollama_vision(frames, is_video)
    if summary:
        return summary

    if not is_video:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            summary = await _openai_vision(frames[0], frame_mime, api_key)
            if summary:
                return summary

    summary = await _blip_vision(frames, is_video)
    return summary or ""


async def _vision_summary(media: UploadFile) -> str:
    """
    Auto-generate a visual summary from an uploaded image or video file.

    Priority:
      1. Ollama vision model (llava) — local, free, works for both image and video frames
      2. OpenAI GPT-4o-mini Vision — if OPENAI_API_KEY is set (images only)
      3. Empty string — graceful fallback when neither is available
    """
    contents = await media.read()
    content_type = media.content_type or ""
    is_video = content_type.startswith("video/") or media.filename.lower().rsplit(".", 1)[-1] in (
        "mp4", "mov", "avi", "webm", "mkv"
    )

    # Extract frames: for video pull 3 keyframes; for image use the file directly
    if is_video:
        frames = _extract_video_frames(contents)
        frame_mime = "image/jpeg"
    else:
        frames = [contents]
        frame_mime = content_type or "image/jpeg"

    if not frames:
        return ""

    # 1. Ollama llava (if running)
    summary = await _ollama_vision(frames, is_video)
    if summary:
        return summary

    # 2. OpenAI GPT-4o-mini Vision (if API key set, images only)
    if not is_video:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            summary = await _openai_vision(frames[0], frame_mime, api_key)
            if summary:
                return summary

    # 3. Local BLIP model (always available after startup warm-up)
    summary = await _blip_vision(frames, is_video)
    if summary:
        return summary

    return ""


def _extract_video_frames(video_bytes: bytes, n_frames: int = 3) -> list:
    """
    Extract n evenly-spaced keyframes from a video as JPEG bytes.
    Returns an empty list if cv2 is not installed or the video can't be decoded.
    """
    try:
        import cv2
        import numpy as np
        import tempfile, os as _os

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            tmp_path = f.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total == 0:
                cap.release()
                return []

            frames = []
            for i in range(n_frames):
                # Sample at 20%, 50%, 80% through the video
                pos = int(total * (i + 1) / (n_frames + 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret:
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    frames.append(bytes(buf))
            cap.release()
            return frames
        finally:
            _os.unlink(tmp_path)

    except ImportError:
        logger.warning("opencv-python-headless not installed — cannot extract video frames")
        return []
    except Exception as e:
        logger.warning("Video frame extraction failed: %s", e)
        return []


async def _ollama_vision(frames: list, is_video: bool) -> str:
    """Describe frames using the Ollama vision model (llava or similar)."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_VISION_MODEL", "llava")

    descriptions = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, frame_bytes in enumerate(frames):
            b64 = base64.b64encode(frame_bytes).decode()
            prompt = (
                f"[Frame {i+1}/{len(frames)} of a video reel] {_VISION_PROMPT}"
                if is_video and len(frames) > 1
                else _VISION_PROMPT
            )
            try:
                resp = await client.post(
                    f"{host}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "images": [b64],
                        "stream": False,
                        "options": {"num_predict": 150, "temperature": 0.2},
                    },
                )
                resp.raise_for_status()
                descriptions.append(resp.json()["response"].strip())
            except Exception as e:
                logger.debug("Ollama vision frame %d failed: %s", i + 1, e)
                return ""  # If Ollama is not up, bail immediately

    if not descriptions:
        return ""
    if len(descriptions) == 1:
        return descriptions[0]
    # Merge multi-frame descriptions into one video summary
    return " | ".join(f"[{i+1}] {d}" for i, d in enumerate(descriptions))


async def _openai_vision(image_bytes: bytes, mime: str, api_key: str) -> str:
    """Describe a single image using OpenAI GPT-4o-mini Vision (fallback)."""
    try:
        from openai import AsyncOpenAI
        b64 = base64.b64encode(image_bytes).decode()
        client = AsyncOpenAI(api_key=api_key)
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": _VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
            max_tokens=200,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        logger.warning("OpenAI vision fallback failed: %s", e)
        return ""


async def _blip_vision(frames: list, is_video: bool) -> str:
    """
    Describe frames using the local BLIP image-captioning model.
    No API key, no server — runs entirely on-device using Apple MPS.
    """
    if _blip_processor is None or _blip_model is None:
        logger.debug("BLIP model not yet loaded, skipping local vision")
        return ""

    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_blip, frames, is_video)


def _run_blip(frames: list, is_video: bool) -> str:
    """Blocking BLIP inference — runs in thread pool."""
    try:
        import torch
        from PIL import Image
        import io

        device = next(_blip_model.parameters()).device
        captions = []

        for i, frame_bytes in enumerate(frames):
            image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
            # Conditional captioning with a context hint gives more relevant output
            context = "an instagram creative showing"
            inputs = _blip_processor(
                image, text=context, return_tensors="pt"
            ).to(device, torch.float16)

            with torch.no_grad():
                out = _blip_model.generate(**inputs, max_new_tokens=80)
            caption = _blip_processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)

        if not captions:
            return ""
        if len(captions) == 1:
            return captions[0]
        # For video: combine frame captions into one description
        return " | ".join(f"[Frame {i+1}] {c}" for i, c in enumerate(captions))

    except Exception as e:
        logger.warning("BLIP inference failed: %s", e)
        return ""
