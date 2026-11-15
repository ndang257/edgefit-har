"""Project-wide constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

DATA_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ZIP_NAME = "uci_har.zip"

RANDOM_STATE = 42

# Final holdout: complete subjects never seen during model development.
TEST_SUBJECTS = list(range(25, 31))
CV_FOLDS = 4

# Edge-oriented reduced models keep the top-N features by Random Forest importance.
N_TOP_FEATURES = 25

# Small PyTorch MLP (reference model on the same engineered features).
MLP_HIDDEN = (128, 64)
MLP_DROPOUT = 0.15
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_EPOCHS = 12
MLP_BATCH_SIZE = 128
MLP_VAL_SIZE = 0.2

# Random Forest hyperparameter search (grouped CV, development subjects only).
RF_PARAM_DIST = {
    "n_estimators": [100, 150, 200],
    "max_depth": [None, 15, 25],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}
HPO_ITERATIONS = 6

# CPU inference benchmark.
BENCH_WARMUP = 20
BENCH_REPEATS = 200
BENCH_BATCH = 1000
