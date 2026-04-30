"""
Training script for the Social Media Performance Predictor.

Usage:
    python train.py

What it does:
  1. Loads and parses assignment-dataset.json
  2. Trains the HybridPredictor (Ridge + Embedding KNN)
  3. Saves the model to saved_models/hybrid_predictor.joblib
  4. Runs the full evaluation suite (LOBO CV, baselines, failure analysis)
  5. Saves evaluation results to saved_models/evaluation_results.json
"""
import json
import logging
import sys
from pathlib import Path

# Make sure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from backend.config import DATASET_PATH, MODELS_DIR
from backend.data.loader import load_dataset
from backend.evaluation.evaluator import run_full_evaluation
from backend.models.predictor import HybridPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    logger.info(f"Loading dataset: {DATASET_PATH}")
    if not DATASET_PATH.exists():
        logger.error("Dataset file not found. Expected at: %s", DATASET_PATH)
        sys.exit(1)

    df = load_dataset(DATASET_PATH)
    logger.info(
        f"Loaded {len(df)} posts across {df['brand'].nunique()} brands: "
        + ", ".join(f"{b}({n})" for b, n in df["brand"].value_counts().items())
    )

    # ------------------------------------------------------------------
    # 2. Train
    # ------------------------------------------------------------------
    logger.info("Training HybridPredictor …")
    predictor = HybridPredictor()
    train_info = predictor.train(df)
    logger.info(f"Training info: {train_info}")

    # ------------------------------------------------------------------
    # 3. Save model
    # ------------------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    predictor.save()

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    logger.info("Running evaluation suite (this includes LOBO CV — takes a few minutes) …")
    eval_results = run_full_evaluation(df)

    eval_path = MODELS_DIR / "evaluation_results.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2, default=str)
    logger.info(f"Evaluation results saved → {eval_path}")

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    lobo = eval_results.get("cross_validation_lobo", {})
    wb   = eval_results.get("cross_validation_within_brand", {})
    baselines = eval_results.get("baselines", {})
    tier = eval_results.get("tier_accuracy", {})

    print("\n" + "=" * 65)
    print("  EVALUATION SUMMARY")
    print("=" * 65)

    print("\n[1] Leave-One-Brand-Out CV  (cold-start — unseen brand)")
    print(f"    Mean MAE      : {lobo.get('mean_mae', 'N/A')}")
    print(f"    Mean Spearman : {lobo.get('mean_spearman', 'N/A')}")
    print(f"    Mean Tier Acc : {lobo.get('mean_tier_accuracy', 'N/A')}")
    print("    Per-brand:")
    for fold in lobo.get("fold_results", []):
        print(
            f"      {fold['brand_left_out']:<22} "
            f"MAE={fold['mae']:.3f}  "
            f"Spearman={fold['spearman']:.3f}  "
            f"TierAcc={fold['tier_accuracy']:.0%}"
        )

    print(f"\n[2] Within-Brand K-Fold CV  (known brand — primary use case)")
    print(f"    Overall Mean Spearman: {wb.get('overall_mean_spearman', 'N/A')}")
    for br in wb.get("brand_results", []):
        print(f"      {br['brand']:<22} Spearman={br['mean_spearman']:.3f}  MAE={br['mean_mae']:.3f}  k={br['k_folds']}")

    print("\n[3] Baselines (in-sample reference)")
    for name, b in baselines.items():
        print(f"    {name:<15} MAE={b['mae']:.3f}  Spearman={b['spearman']:.3f}")

    print(f"\n[4] Tier Classification (in-sample)")
    print(f"    Overall accuracy: {tier.get('overall_accuracy', 'N/A')}")

    print("\n" + "=" * 65)
    print("  Done.  Start the API with:")
    print("  uvicorn backend.main:app --reload --port 8000")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
