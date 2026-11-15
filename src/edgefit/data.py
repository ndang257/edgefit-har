"""Download and load the UCI HAR dataset."""

import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from edgefit.config import DATA_DIR, DATA_URL, ZIP_NAME

REQUIRED = [
    "features.txt",
    "activity_labels.txt",
    "train/X_train.txt",
    "train/y_train.txt",
    "train/subject_train.txt",
    "test/X_test.txt",
    "test/y_test.txt",
    "test/subject_test.txt",
]


def _find_root(folder: Path) -> Path | None:
    """The dataset root is the folder that contains features.txt (layout differs between downloads)."""
    for path in sorted(folder.rglob("features.txt")):
        if all((path.parent / f).exists() for f in REQUIRED):
            return path.parent
    return None


def download_har(data_dir: Path = DATA_DIR) -> Path:
    """Download (if needed) and extract UCI HAR. Returns the dataset root folder.

    The official archive contains the data as a nested zip, so nested archives are extracted too.
    You can also place the downloaded zip at ``data/raw/uci_har.zip`` yourself.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    root = _find_root(data_dir)
    if root is not None:
        return root

    zip_path = data_dir / ZIP_NAME
    if not zip_path.exists():
        print(f"Downloading UCI HAR to {zip_path} ...")
        urllib.request.urlretrieve(DATA_URL, zip_path)

    extract_dir = data_dir / "extracted"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    for nested in list(extract_dir.rglob("*.zip")):
        target = nested.parent / nested.stem
        if not target.exists():
            with zipfile.ZipFile(nested) as zf:
                zf.extractall(target)

    root = _find_root(data_dir)
    if root is None:
        raise FileNotFoundError(f"UCI HAR files not found after extracting {zip_path}")
    return root


def make_unique(names: list[str]) -> list[str]:
    """Append __dup2, __dup3, ... to repeated names (UCI HAR has duplicated feature names)."""
    seen: dict[str, int] = {}
    out = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}__dup{seen[name]}")
    return out


def sanitize_feature_names(names: list[str]) -> list[str]:
    """Make names unique and LightGBM-compatible: tBodyAcc-mean()-X -> tBodyAcc_mean_X."""
    clean = [re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_]+", "_", n)).strip("_") for n in make_unique(names)]
    return make_unique(clean)


def load_har(root: Path | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[int, str]]:
    """Load the original train and test files into one table.

    Returns the feature matrix (10,299 x 561), activity ids (1-6), subject ids (1-30)
    and a mapping from activity id to name. The original 70/30 split is ignored; the
    project defines its own subject holdout.
    """
    root = Path(root) if root is not None else download_har()
    features = pd.read_csv(root / "features.txt", sep=r"\s+", header=None, names=["id", "feature"])
    names = sanitize_feature_names(features.feature.tolist())
    labels = pd.read_csv(root / "activity_labels.txt", sep=r"\s+", header=None, names=["id", "activity"])
    label_map = dict(zip(labels.id, labels.activity, strict=True))

    parts = []
    for split in ("train", "test"):
        X = pd.read_csv(root / split / f"X_{split}.txt", sep=r"\s+", header=None, names=names)
        y = pd.read_csv(root / split / f"y_{split}.txt", header=None)[0]
        g = pd.read_csv(root / split / f"subject_{split}.txt", header=None)[0]
        parts.append((X, y, g))

    X = pd.concat([p[0] for p in parts], ignore_index=True)
    y = pd.concat([p[1] for p in parts], ignore_index=True).rename("activity")
    groups = pd.concat([p[2] for p in parts], ignore_index=True).rename("subject")
    return X, y, groups, label_map
