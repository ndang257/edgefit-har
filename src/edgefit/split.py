"""Subject-aware splits: no person appears on both sides of any split."""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from edgefit.config import CV_FOLDS, MLP_VAL_SIZE, RANDOM_STATE, TEST_SUBJECTS


def subject_holdout(X: pd.DataFrame, y: pd.Series, groups: pd.Series, test_subjects=TEST_SUBJECTS):
    """Hold out complete subjects as the final test set. Returns train and test (X, y, groups)."""
    mask = groups.isin(test_subjects)
    train = tuple(obj[~mask].reset_index(drop=True) for obj in (X, y, groups))
    test = tuple(obj[mask].reset_index(drop=True) for obj in (X, y, groups))
    assert_disjoint(train[2], test[2])
    return train, test


def assert_disjoint(groups_a, groups_b) -> None:
    overlap = set(groups_a) & set(groups_b)
    if overlap:
        raise AssertionError(f"Subjects on both sides of the split: {sorted(overlap)}")


def grouped_cv(n_splits: int = CV_FOLDS) -> StratifiedGroupKFold:
    """Stratified CV that keeps every subject inside a single fold."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def describe_folds(cv, X, y, groups) -> pd.DataFrame:
    """One row per fold with its validation subjects; raises if a subject leaks across."""
    rows = []
    for fold, (tr, va) in enumerate(cv.split(X, y, groups)):
        assert_disjoint(groups.iloc[tr], groups.iloc[va])
        rows.append(
            {
                "fold": fold,
                "train_rows": len(tr),
                "val_rows": len(va),
                "val_subjects": sorted(map(int, groups.iloc[va].unique())),
            }
        )
    return pd.DataFrame(rows)


def validation_split(X, y, groups, val_size: float = MLP_VAL_SIZE):
    """Subject-aware train/validation split of the development data (for the MLP)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=RANDOM_STATE)
    tr, va = next(gss.split(X, y, groups))
    assert_disjoint(groups.iloc[tr], groups.iloc[va])
    return tr, va
