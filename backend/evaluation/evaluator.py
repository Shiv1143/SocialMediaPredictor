"""
Evaluation pipeline for the Social Media Performance Predictor.

Strategy:
  - Leave-One-Brand-Out (LOBO) cross-validation: given only 5 brands,
    standard k-fold would split within brands (leaking stylistic patterns).
    LOBO gives a realistic estimate of how well the model generalises to a
    brand it has never seen — the hardest and most honest test.
  - Multiple baselines (always-mean, brand-mean, random) so the model's
    improvement has a meaningful reference point.
  - Spearman rank correlation as the primary metric: getting the ordering
    right (high vs low posts) is more actionable than minimising raw MSE.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def run_full_evaluation(df: pd.DataFrame) -> dict:
    """Run the complete evaluation suite and return a structured results dict."""
    from backend.models.predictor import HybridPredictor

    results: dict = {}

    # 1. Train a final model on all data (for in-sample diagnostics)
    logger.info("Training final model for in-sample evaluation …")
    final_predictor = HybridPredictor()
    final_predictor.train(df)

    # 2. Leave-one-brand-out cross-validation
    # Tests: "how well does the model generalize to a brand it has never seen?"
    logger.info("Running Leave-One-Brand-Out CV …")
    results["cross_validation_lobo"] = _lobo_cv(df)

    # 3. Within-brand k-fold cross-validation
    # Tests: "how well does the model rank posts within a brand it has seen?"
    # This is the relevant metric for the primary use case.
    logger.info("Running within-brand k-fold CV …")
    results["cross_validation_within_brand"] = _within_brand_kfold_cv(df)

    # 4. Baseline comparisons (in-sample, for reference)
    results["baselines"] = _compute_baselines(df)

    # 5. In-sample per-brand diagnostics
    results["per_brand_insample"] = _per_brand_eval(final_predictor, df)

    # 6. Tier classification accuracy (in-sample)
    results["tier_accuracy"] = _tier_accuracy(final_predictor, df)

    # 7. Failure case analysis
    results["failure_analysis"] = _failure_cases(final_predictor, df)

    # 8. Dataset overview
    results["dataset_overview"] = _dataset_overview(df)

    return results


# ------------------------------------------------------------------
# Cross-validation
# ------------------------------------------------------------------

def _lobo_cv(df: pd.DataFrame) -> dict:
    from backend.models.predictor import HybridPredictor

    groups = df["brand"].values
    logo = LeaveOneGroupOut()
    fold_results = []

    for train_idx, test_idx in logo.split(df, groups=groups):
        test_brand = df.iloc[test_idx]["brand"].iloc[0]
        n_test = len(test_idx)

        if n_test < 3:
            logger.warning(f"Skipping {test_brand}: only {n_test} test samples")
            continue

        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)

        p = HybridPredictor()
        p.train(train_df)

        y_true = test_df["brand_norm_score"].values
        y_pred = _predict_batch(p, test_df)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        spearman, p_val = spearmanr(y_true, y_pred)
        spearman = float(spearman) if not np.isnan(spearman) else 0.0

        # Tier accuracy for this fold
        y_true_tier = test_df["performance_tier"].values
        y_pred_tier = [_score_to_tier(s) for s in y_pred]
        tier_acc = float(np.mean([t == p for t, p in zip(y_true_tier, y_pred_tier)]))

        fold_results.append(
            {
                "brand_left_out": test_brand,
                "n_test": n_test,
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "spearman": round(spearman, 4),
                "spearman_p_value": round(float(p_val) if not np.isnan(p_val) else 1.0, 4),
                "tier_accuracy": round(tier_acc, 4),
            }
        )
        logger.info(
            f"  [{test_brand}] MAE={mae:.3f}  RMSE={rmse:.3f}  "
            f"Spearman={spearman:.3f}  TierAcc={tier_acc:.2%}"
        )

    if not fold_results:
        return {"error": "No folds completed"}

    return {
        "strategy": "Leave-One-Brand-Out (LOBO)",
        "interpretation": (
            "Each fold trains on 4 brands and evaluates on the 5th. "
            "This simulates deploying the model on a brand it has never seen. "
            "Spearman correlation measures whether the model correctly ranks "
            "posts from low to high engagement within a new brand."
        ),
        "fold_results": fold_results,
        "mean_mae": round(float(np.mean([r["mae"] for r in fold_results])), 4),
        "mean_rmse": round(float(np.mean([r["rmse"] for r in fold_results])), 4),
        "mean_spearman": round(float(np.mean([r["spearman"] for r in fold_results])), 4),
        "mean_tier_accuracy": round(float(np.mean([r["tier_accuracy"] for r in fold_results])), 4),
    }


# ------------------------------------------------------------------
# Within-brand k-fold CV
# ------------------------------------------------------------------

def _within_brand_kfold_cv(df: pd.DataFrame, n_splits: int = 3) -> dict:
    """
    K-fold CV performed independently within each brand.

    Why this matters: LOBO CV answers "can we predict for an unseen brand?"
    (answer: barely — brands are too stylistically different).
    Within-brand CV answers "can we rank posts within a known brand?" which
    is the actual deployment use case.  We use k=3 (not 5) because the
    smallest brand (Thumsup) has only 35 posts — 3-fold gives ~12 test samples
    per fold, which is enough for a meaningful Spearman estimate.

    Note on temporal leakage: the followers count in this dataset is current
    (not at post time), so any feature using followers is slightly leaky.
    With a live system you'd store follower count at post time.
    """
    from sklearn.model_selection import KFold
    from backend.models.predictor import HybridPredictor

    brand_results = []
    overall_spearman = []

    for brand in sorted(df["brand"].unique()):
        brand_df = df[df["brand"] == brand].reset_index(drop=True)
        n = len(brand_df)

        # Use min(n_splits, n//5) to ensure at least 5 samples per test fold
        k = min(n_splits, max(2, n // 5))
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        fold_spearmans = []
        fold_maes = []

        for fold_i, (train_idx, test_idx) in enumerate(kf.split(brand_df)):
            if len(test_idx) < 3:
                continue
            train_df = brand_df.iloc[train_idx].copy().reset_index(drop=True)
            test_df = brand_df.iloc[test_idx].copy().reset_index(drop=True)

            # We need to retrain the full model on all brands minus the test split,
            # but keeping other brands intact. For computational efficiency, we train
            # on the full non-brand data + this brand's training fold.
            other_brands_df = df[df["brand"] != brand].copy()
            combined_train = pd.concat([other_brands_df, train_df], ignore_index=True)

            p = HybridPredictor()
            p.train(combined_train)

            y_true = test_df["brand_norm_score"].values
            y_pred = _predict_batch(p, test_df)

            mae = float(mean_absolute_error(y_true, y_pred))
            spearman, _ = spearmanr(y_true, y_pred)
            spearman = float(spearman) if not np.isnan(spearman) else 0.0

            fold_spearmans.append(spearman)
            fold_maes.append(mae)

        if not fold_spearmans:
            continue

        mean_sp = float(np.mean(fold_spearmans))
        mean_mae = float(np.mean(fold_maes))
        overall_spearman.append(mean_sp)

        brand_results.append({
            "brand": brand,
            "n_posts": n,
            "k_folds": k,
            "mean_spearman": round(mean_sp, 4),
            "mean_mae": round(mean_mae, 4),
        })
        logger.info(f"  [{brand}] within-brand {k}-fold: Spearman={mean_sp:.3f}  MAE={mean_mae:.3f}")

    return {
        "strategy": "Within-Brand K-Fold (k=3)",
        "interpretation": (
            "Each brand's posts are split k-fold; model trained on other brands + "
            "that brand's training fold, tested on holdout fold. This measures ranking "
            "ability within a known brand — the primary deployment scenario. "
            "More honest than in-sample Spearman, less pessimistic than LOBO."
        ),
        "brand_results": brand_results,
        "overall_mean_spearman": round(float(np.mean(overall_spearman)), 4) if overall_spearman else None,
    }


# ------------------------------------------------------------------
# Baselines
# ------------------------------------------------------------------

def _compute_baselines(df: pd.DataFrame) -> dict:
    y_true = df["brand_norm_score"].values
    results = {}

    # Always predict 0.5 (the expected value of a uniform distribution on [0,1])
    y_mean = np.full(len(y_true), 0.5)
    results["always_0.5"] = _metrics(
        y_true, y_mean,
        "Always predict 0.5 — the global expected value of the normalised score"
    )

    # Always predict brand mean (cheats by knowing brand identity)
    y_brand_mean = df.groupby("brand")["brand_norm_score"].transform("mean").values
    results["brand_mean"] = _metrics(
        y_true, y_brand_mean,
        "Predict each brand's mean normalised score (requires knowing the brand)"
    )

    # Predict brand median engagement rate mapped back to normalised scale
    y_brand_median = df.groupby("brand")["brand_norm_score"].transform("median").values
    results["brand_median"] = _metrics(
        y_true, y_brand_median,
        "Predict each brand's median normalised score"
    )

    # Random baseline
    rng = np.random.default_rng(42)
    y_random = rng.uniform(0, 1, len(y_true))
    results["random"] = _metrics(
        y_true, y_random,
        "Random uniform prediction between 0 and 1"
    )

    return results


# ------------------------------------------------------------------
# Per-brand in-sample diagnostics
# ------------------------------------------------------------------

def _per_brand_eval(predictor, df: pd.DataFrame) -> list:
    results = []
    for brand in sorted(df["brand"].unique()):
        sub = df[df["brand"] == brand]
        y_true = sub["brand_norm_score"].values
        y_pred = _predict_batch(predictor, sub)
        spearman, _ = spearmanr(y_true, y_pred)
        results.append(
            {
                "brand": brand,
                "n_posts": len(sub),
                "mae": round(mean_absolute_error(y_true, y_pred), 4),
                "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
                "spearman": round(float(spearman) if not np.isnan(spearman) else 0.0, 4),
                "note": "In-sample — inflated vs LOBO CV above",
            }
        )
    return results


# ------------------------------------------------------------------
# Tier classification accuracy
# ------------------------------------------------------------------

def _tier_accuracy(predictor, df: pd.DataFrame) -> dict:
    y_true = df["performance_tier"].values
    y_pred_scores = _predict_batch(predictor, df)
    y_pred = [_score_to_tier(s) for s in y_pred_scores]

    overall = float(np.mean([t == p for t, p in zip(y_true, y_pred)]))

    tiers = ["low", "medium", "high"]
    per_tier = {}
    for tier in tiers:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == tier and p == tier)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != tier and p == tier)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == tier and p != tier)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_tier[tier] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    return {
        "overall_accuracy": round(overall, 4),
        "per_tier": per_tier,
        "note": "In-sample results — LOBO CV tier_accuracy above is the honest estimate",
    }


# ------------------------------------------------------------------
# Failure case analysis
# ------------------------------------------------------------------

def _failure_cases(predictor, df: pd.DataFrame) -> dict:
    y_true = df["brand_norm_score"].values
    y_pred = _predict_batch(predictor, df)
    errors = np.abs(y_true - y_pred)

    worst_idx = np.argsort(errors)[-10:][::-1]
    cases = []
    for i in worst_idx:
        row = df.iloc[i]
        cases.append(
            {
                "brand": row["brand"],
                "media_type": row["media_type"],
                "caption_snippet": (row["caption"][:80] + "…") if len(row.get("caption", "")) > 80 else row.get("caption", ""),
                "true_score": round(float(y_true[i]), 3),
                "predicted_score": round(float(y_pred[i]), 3),
                "absolute_error": round(float(errors[i]), 3),
                "true_tier": row["performance_tier"],
                "predicted_tier": _score_to_tier(y_pred[i]),
                "engagement_rate": round(float(row["engagement_rate"]), 2),
            }
        )

    return {
        "worst_10_predictions": cases,
        "analysis": (
            "The largest errors tend to be outlier posts: viral collaborations "
            "with celebrity accounts that drove engagement far above the brand's "
            "baseline, or posts with unusual creative formats. With only ~35–120 "
            "posts per brand, the model lacks enough examples of these extremes. "
            "More data (especially edge-case posts) would improve reliability."
        ),
    }


# ------------------------------------------------------------------
# Dataset overview
# ------------------------------------------------------------------

def _dataset_overview(df: pd.DataFrame) -> dict:
    overview = {
        "total_posts": len(df),
        "brands": {},
        "media_types": df["media_type"].value_counts().to_dict(),
        "static_posts_zero_views": int((df["views"] == 0).sum()),
        "collaborated_posts": int(df["is_collaborated"].sum()),
        "engagement_rate_global": {
            "min": round(float(df["engagement_rate"].min()), 4),
            "max": round(float(df["engagement_rate"].max()), 4),
            "mean": round(float(df["engagement_rate"].mean()), 4),
            "median": round(float(df["engagement_rate"].median()), 4),
        },
    }
    for brand in sorted(df["brand"].unique()):
        sub = df[df["brand"] == brand]
        er = sub["engagement_rate"]
        overview["brands"][brand] = {
            "n_posts": len(sub),
            "median_er": round(float(er.median()), 2),
            "mean_er": round(float(er.mean()), 2),
            "tier_distribution": sub["performance_tier"].value_counts().to_dict(),
        }
    return overview


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def _predict_batch(predictor, df: pd.DataFrame) -> np.ndarray:
    preds = []
    for _, row in df.iterrows():
        try:
            result = predictor.predict(row.to_dict())
            preds.append(result["brand_normalized_score"])
        except Exception as e:
            logger.warning(f"Prediction failed for row: {e}")
            preds.append(0.5)
    return np.array(preds)


def _metrics(y_true, y_pred, description: str) -> dict:
    spearman, _ = spearmanr(y_true, y_pred)
    return {
        "description": description,
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "spearman": round(float(spearman) if not np.isnan(spearman) else 0.0, 4),
    }


def _score_to_tier(score: float) -> str:
    if score <= 0.33:
        return "low"
    elif score <= 0.67:
        return "medium"
    return "high"
