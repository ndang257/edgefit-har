"""Train, evaluate and benchmark every candidate model end to end.

Steps
1. Load UCI HAR (downloaded on first run) and describe the classes.
2. Hold out subjects 25-30 as the final test set; 4-fold StratifiedGroupKFold on the rest.
3. Logistic Regression, Random Forest and LightGBM on all 561 features, then Random Forest
   and LightGBM on the 25 most important features (edge-oriented).
4. A small PyTorch MLP on the same features, with subject-aware validation for early selection.
5. Compare by Macro F1, then class-level errors of the best model.
6. Permutation importance, CPU size/latency benchmark of every model.
7. Grouped-CV hyperparameter search for the Random Forest, logged to MLflow.

Usage:
    python scripts/train.py            # full run (~6 min on 2 CPU cores), writes reports/
    python scripts/train.py --quick    # small settings for a fast check, writes reports/quick/
"""

import argparse

from edgefit.config import MODELS_DIR, REPORTS_DIR, ROOT
from edgefit.data import load_har
from edgefit.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small settings, writes to reports/quick/")
    args = parser.parse_args()

    out_dir = REPORTS_DIR / "quick" if args.quick else REPORTS_DIR
    X, y, groups, label_map = load_har()
    run(X, y, groups, label_map, out_dir=out_dir, models_dir=MODELS_DIR, mlflow_dir=ROOT / "mlruns", quick=args.quick)


if __name__ == "__main__":
    main()
