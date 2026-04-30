"""
Hybrid predictor: Ridge Regression (structured features) + KNN Embedding Similarity.

Design rationale:
  - With only 35–120 posts per brand, gradient boosting would badly overfit.
  - Ridge regression with L2 regularization handles small N, gives coefficients
    we can interpret directly as feature contributions.
  - Sentence-transformer embeddings capture semantic content (caption + visual
    summary) and let us find structurally similar past posts for explanation.
  - The KNN component also acts as a graceful fallback: for unseen brands or
    unusual posts, we still return the performance of the most semantically
    similar historical posts.
"""
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from backend.config import (
    EMBEDDING_MODEL_NAME,
    HYBRID_KNN_WEIGHT,
    HYBRID_RIDGE_WEIGHT,
    KNN_K,
    MODELS_DIR,
)
from backend.data.features import (
    FEATURE_COLUMNS,
    FEATURE_DESCRIPTIONS,
    build_feature_matrix,
    build_text_for_embedding,
    extract_features,
)

logger = logging.getLogger(__name__)


class HybridPredictor:
    def __init__(
        self,
        ridge_weight: float = HYBRID_RIDGE_WEIGHT,
        knn_weight: float = HYBRID_KNN_WEIGHT,
    ):
        self.ridge_weight = ridge_weight
        self.knn_weight = knn_weight

        self.ridge_pipeline: Optional[Pipeline] = None
        self.embedding_model: Optional[SentenceTransformer] = None

        # KNN store (populated at training time)
        self.train_embeddings: Optional[np.ndarray] = None
        self.train_records: Optional[list] = None
        self.train_targets: Optional[np.ndarray] = None  # brand_norm_score

        # Per-brand engagement statistics for denormalization and context
        self.brand_stats: dict = {}

        self.is_trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> dict:
        logger.info(f"Training HybridPredictor on {len(df)} records …")

        self._compute_brand_stats(df)

        # 1. Ridge regression on structured features
        X = build_feature_matrix(df)
        y = df["brand_norm_score"].values

        self.ridge_pipeline = Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))]
        )
        self.ridge_pipeline.fit(X[FEATURE_COLUMNS], y)
        ridge_train_r2 = self.ridge_pipeline.score(X[FEATURE_COLUMNS], y)

        # 2. Sentence-transformer embeddings
        logger.info(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' …")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        texts = [build_text_for_embedding(r.to_dict()) for _, r in df.iterrows()]
        logger.info("Computing embeddings for training corpus …")
        self.train_embeddings = self.embedding_model.encode(
            texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True
        )

        # 3. Store records + targets for KNN lookup
        self.train_records = df.to_dict("records")
        self.train_targets = y

        self.is_trained = True
        logger.info(f"Training complete. Ridge R² (in-sample): {ridge_train_r2:.3f}")

        return {"n_samples": len(df), "ridge_r2_train": round(ridge_train_r2, 4)}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, post: dict) -> dict:
        """
        Predict engagement for a new post.

        Required keys: caption, brand, media_type
        Optional keys: duration, is_collaborated, collaborators,
                       created_at, followers, visual_summary
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained. Run train.py first.")

        brand = post.get("brand", "unknown")
        known_brand = brand in self.brand_stats and not brand.startswith("_")

        # Fill in brand's median followers if not provided
        if not post.get("followers"):
            fallback_followers = self.brand_stats.get(brand, {}).get(
                "median_followers",
                self.brand_stats.get("_global", {}).get("median_followers", 100_000),
            )
            post = {**post, "followers": fallback_followers}

        # Determine content group for target denormalization
        media_type = (post.get("media_type") or "reel").lower()
        content_group = "reel" if media_type == "reel" else "static"

        # ---------- Ridge component ----------
        features = extract_features(post)
        X = pd.DataFrame([features])[FEATURE_COLUMNS]
        ridge_score = float(np.clip(self.ridge_pipeline.predict(X)[0], 0.0, 1.0))

        # ---------- KNN component (brand-aware) ----------
        # For known brands, same-brand neighbours are the most relevant signal.
        # Cross-brand neighbours from stylistically different brands (Red Bull vs
        # Pepsi) are nearly uninformative — they share an embedding neighbourhood
        # but not an engagement neighbourhood.
        #
        # Strategy:
        #  • If the brand is known AND has ≥ MIN_SAME_BRAND_POSTS posts in the
        #    training corpus, compute two KNN scores:
        #      – same-brand KNN  (K nearest from same brand only)
        #      – cross-brand KNN (K nearest from ALL posts)
        #    and blend them  (same-brand heavily weighted).
        #  • If the brand is unknown or has too few training posts, fall back to
        #    pure cross-brand KNN.
        MIN_SAME_BRAND_POSTS = 5

        text = build_text_for_embedding(post)
        query_emb = self.embedding_model.encode(
            [text], normalize_embeddings=True
        )[0]
        sims = (self.train_embeddings @ query_emb).astype(float)

        # All-corpus KNN (cross-brand)
        cross_k_idx = np.argsort(sims)[-KNN_K:][::-1]
        cross_k_sims = sims[cross_k_idx]
        cross_k_targets = self.train_targets[cross_k_idx]
        cross_weights = np.clip(cross_k_sims, 0, None)
        if cross_weights.sum() > 1e-9:
            knn_cross = float(np.average(cross_k_targets, weights=cross_weights))
        else:
            knn_cross = float(np.mean(self.train_targets))

        # Same-brand KNN
        same_brand_mask = np.array(
            [r.get("brand") == brand for r in self.train_records], dtype=bool
        )
        n_same_brand = same_brand_mask.sum()

        if known_brand and n_same_brand >= MIN_SAME_BRAND_POSTS:
            same_sims = np.where(same_brand_mask, sims, -np.inf)
            k_same = min(KNN_K, n_same_brand)
            same_k_idx = np.argsort(same_sims)[-k_same:][::-1]
            same_k_sims = sims[same_k_idx]
            same_k_targets = self.train_targets[same_k_idx]
            same_weights = np.clip(same_k_sims, 0, None)
            if same_weights.sum() > 1e-9:
                knn_same = float(np.average(same_k_targets, weights=same_weights))
            else:
                knn_same = float(np.mean(same_k_targets))
            # 70% same-brand + 30% cross-brand for known brands
            knn_score = float(np.clip(0.7 * knn_same + 0.3 * knn_cross, 0.0, 1.0))
            # Use same-brand neighbours for explanation (more informative)
            top_k_idx = same_k_idx
            top_k_sims = same_k_sims
        else:
            knn_score = float(np.clip(knn_cross, 0.0, 1.0))
            top_k_idx = cross_k_idx
            top_k_sims = cross_k_sims

        # ---------- Hybrid ----------
        final_score = float(
            np.clip(
                self.ridge_weight * ridge_score + self.knn_weight * knn_score,
                0.0,
                1.0,
            )
        )

        tier = _score_to_tier(final_score)
        estimated_er = self._denormalize_er(final_score, brand, content_group)
        confidence = self._confidence(top_k_sims, ridge_score, knn_score)

        # Reduce confidence for unknown brands — predictions are less reliable
        if not known_brand:
            confidence *= 0.6

        explanation = self._explain(
            post, features, sims, top_k_idx, top_k_sims,
            ridge_score, knn_score, final_score, brand,
            content_group=content_group,
        )

        warnings = []
        if not known_brand:
            warnings.append(
                f"Brand '{brand}' was not seen during training. "
                "Prediction uses cross-brand similarity only and should be treated "
                "as a rough estimate. Add training data for this brand for better accuracy."
            )
        if content_group == "static":
            warnings.append(
                "Static posts (images/albums) use a followers-based engagement rate. "
                "This is a different metric than reel ER (views-based) and cannot be "
                "directly compared."
            )

        result = {
            "predicted_engagement_rate": round(estimated_er, 4),
            "brand_normalized_score": round(final_score, 4),
            "performance_tier": tier,
            "confidence": round(confidence, 4),
            "explanation": explanation,
            "model_details": {
                "ridge_score": round(ridge_score, 4),
                "knn_score": round(knn_score, 4),
                "ridge_weight": self.ridge_weight,
                "knn_weight": self.knn_weight,
                "content_group": content_group,
                "known_brand": known_brand,
            },
        }
        if warnings:
            result["warnings"] = warnings
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _denormalize_er(self, score: float, brand: str, content_group: str = "reel") -> float:
        """
        Map a brand-normalised score (0–1) back to an estimated engagement rate.

        Uses the content-type-specific percentile curve so that a score of 0.7
        for a reel maps to a reel-appropriate ER, and a score of 0.7 for a static
        post maps to a static-appropriate ER.  This is necessary because
        reel ER (views-based) and static ER (followers-based) are different metrics.
        """
        stats = self.brand_stats.get(brand) or self.brand_stats.get("_global", {})
        ctype_stats = stats.get("content_types", {}).get(content_group, {})
        percs = ctype_stats.get("percentiles") or stats.get("percentiles", [])
        if not percs:
            return score * 10.0
        idx = score * (len(percs) - 1)
        lo, hi = int(idx), min(int(idx) + 1, len(percs) - 1)
        frac = idx - lo
        return float(percs[lo] * (1 - frac) + percs[hi] * frac)

    def _confidence(self, top_sims: np.ndarray, ridge: float, knn: float) -> float:
        """
        Confidence heuristic (0–1):
          40 % → max cosine similarity of nearest neighbour
          30 % → mean cosine similarity of top-K neighbours
          30 % → agreement between Ridge and KNN scores
        """
        max_sim = float(top_sims.max()) if len(top_sims) > 0 else 0.5
        avg_sim = float(top_sims.mean()) if len(top_sims) > 0 else 0.5
        agreement = 1.0 - abs(ridge - knn)
        return float(np.clip(0.4 * max_sim + 0.3 * avg_sim + 0.3 * agreement, 0, 1))

    def _explain(
        self, post, features, all_sims, top_k_idx, top_k_sims,
        ridge_score, knn_score, final_score, brand,
        content_group: str = "reel",
    ) -> dict:
        """Build the explanation dict returned to the caller."""

        # --- Ridge feature contributions ---
        ridge_step = self.ridge_pipeline.named_steps["ridge"]
        scaler_step = self.ridge_pipeline.named_steps["scaler"]
        X_raw = pd.DataFrame([features])[FEATURE_COLUMNS]
        X_scaled = scaler_step.transform(X_raw)[0]
        contributions = ridge_step.coef_ * X_scaled

        factor_list = []
        for feat, contrib in zip(FEATURE_COLUMNS, contributions):
            if abs(contrib) < 0.005:
                continue
            factor_list.append(
                {
                    "feature": feat,
                    "contribution": round(float(contrib), 4),
                    "impact": "positive" if contrib > 0 else "negative",
                    "description": _feature_label(feat, features.get(feat, 0)),
                }
            )
        factor_list.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        top_factors = factor_list[:6]

        # --- Similar posts ---
        similar_posts = []
        for idx, sim in zip(top_k_idx, top_k_sims):
            rec = self.train_records[idx]
            cap = rec.get("caption", "")
            snippet = (cap[:90] + "…") if len(cap) > 90 else cap
            similar_posts.append(
                {
                    "brand": rec.get("brand", ""),
                    "caption_snippet": snippet,
                    "engagement_rate": round(float(rec.get("engagement_rate", 0)), 2),
                    "performance_tier": rec.get("performance_tier", ""),
                    "similarity_score": round(float(sim), 3),
                    "media_type": rec.get("media_type", ""),
                    "url": rec.get("url", ""),
                }
            )

        # --- Brand context ---
        brand_stats = self.brand_stats.get(brand) or self.brand_stats.get("_global", {})
        estimated_er = self._denormalize_er(final_score, brand, content_group)
        brand_median = brand_stats.get("median", 0.0)

        tier = _score_to_tier(final_score)
        tier_label = {"low": "below average", "medium": "average", "high": "above average"}[tier]
        summary = (
            f"This post is predicted to perform {tier_label} for {brand.replace('_', ' ')}. "
            f"Estimated engagement rate: {estimated_er:.2f}% "
            f"(brand median: {brand_median:.2f}%). "
            f"Prediction driven {int(self.knn_weight * 100)}% by content similarity "
            f"and {int(self.ridge_weight * 100)}% by structural features."
        )

        return {
            "summary": summary,
            "key_factors": top_factors,
            "similar_posts": similar_posts,
            "brand_context": {
                "brand": brand,
                "brand_median_er": round(brand_median, 2),
                "estimated_er": round(estimated_er, 2),
                "performance_tier": tier,
            },
        }

    def _compute_brand_stats(self, df: pd.DataFrame):
        """
        Build per-brand, per-content-type engagement statistics for denormalization.

        We store separate percentile curves for reels (views-based ER) and static
        content (followers-based ER) within each brand.  At denormalization time,
        we pick the right curve based on the incoming post's content type.
        """
        for brand in df["brand"].unique():
            sub = df[df["brand"] == brand]
            followers = sub["followers"].values
            self.brand_stats[brand] = {
                "median_followers": float(np.median(followers)),
                "n_posts": len(sub),
                "content_types": {},
            }
            # Per-content-type stats (reel vs static)
            for ctype in ["reel", "static"]:
                csub = sub[sub["content_group"] == ctype]["engagement_rate"]
                if len(csub) == 0:
                    continue
                er = csub.values
                self.brand_stats[brand]["content_types"][ctype] = {
                    "mean": float(er.mean()),
                    "median": float(np.median(er)),
                    "p33": float(np.percentile(er, 33)),
                    "p67": float(np.percentile(er, 67)),
                    "percentiles": np.percentile(er, np.linspace(0, 100, 101)).tolist(),
                    "n": len(csub),
                }
            # Convenience: overall median (used in explanation)
            all_er = sub["engagement_rate"].values
            self.brand_stats[brand]["median"] = float(np.median(all_er))
            self.brand_stats[brand]["mean"] = float(all_er.mean())

        # Global fallback
        all_er = df["engagement_rate"].values
        self.brand_stats["_global"] = {
            "mean": float(all_er.mean()),
            "median": float(np.median(all_er)),
            "percentiles": np.percentile(all_er, np.linspace(0, 100, 101)).tolist(),
            "content_types": {
                "reel": {},
                "static": {},
            },
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None):
        if path is None:
            path = MODELS_DIR / "hybrid_predictor.joblib"
        joblib.dump(self, path)
        logger.info(f"Model saved → {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "HybridPredictor":
        if path is None:
            path = MODELS_DIR / "hybrid_predictor.joblib"
        return joblib.load(path)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _score_to_tier(score: float) -> str:
    if score <= 0.33:
        return "low"
    elif score <= 0.67:
        return "medium"
    return "high"


def _feature_label(feat: str, value) -> str:
    desc = FEATURE_DESCRIPTIONS.get(feat, feat)
    # Append current value for numeric features to give context
    if feat in ("duration_seconds", "caption_word_count", "emoji_count",
                "hashtag_count", "mention_count", "num_collaborators"):
        desc += f" (value: {int(value)})"
    elif feat in ("post_hour",):
        desc += f" (hour {int(value)}:00)"
    return desc
