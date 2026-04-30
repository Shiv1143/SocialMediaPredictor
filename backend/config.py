from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATASET_PATH = BASE_DIR / "Consuma - AI Engineer - Problem Statement" / "assignment-dataset.json"
MODELS_DIR = BASE_DIR / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
KNN_K = 7

# Hybrid model weights (ridge + KNN)
HYBRID_RIDGE_WEIGHT = 0.35
HYBRID_KNN_WEIGHT = 0.65

KNOWN_BRANDS = [
    "cocacola_india",
    "redbullindia",
    "pepsiindia",
    "sprite_india",
    "thumsupofficial",
]
