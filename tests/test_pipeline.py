import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from edgefit.benchmark import benchmark_model
from edgefit.data import load_har, make_unique, sanitize_feature_names
from edgefit.models import MLP, predict_mlp, train_mlp
from edgefit.pipeline import run
from edgefit.split import assert_disjoint, describe_folds, grouped_cv, subject_holdout, validation_split

ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]


def synthetic_har(n_subjects=30, per_class=6, n_features=40, seed=0):
    """Small HAR-like data: 6 separable activities, every subject does every activity."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3, size=(6, n_features))
    X, y, g = [], [], []
    for s in range(1, n_subjects + 1):
        offset = rng.normal(scale=0.3, size=n_features)  # person-specific shift
        for c in range(6):
            X.append(centers[c] + offset + rng.normal(size=(per_class, n_features)))
            y += [c + 1] * per_class
            g += [s] * per_class
    cols = [f"f{i}" for i in range(n_features)]
    return (
        pd.DataFrame(np.vstack(X), columns=cols),
        pd.Series(y, name="activity"),
        pd.Series(g, name="subject"),
        dict(enumerate(ACTIVITIES, 1)),
    )


@pytest.fixture
def har():
    return synthetic_har()


def test_feature_names_are_unique_and_clean():
    raw = ["tBodyAcc-mean()-X", "fBodyAcc-bandsEnergy()-1,8", "fBodyAcc-bandsEnergy()-1,8", "angle(X,gravityMean)"]
    clean = sanitize_feature_names(raw)
    assert clean == [
        "tBodyAcc_mean_X",
        "fBodyAcc_bandsEnergy_1_8",
        "fBodyAcc_bandsEnergy_1_8_dup2",
        "angle_X_gravityMean",
    ]
    assert len(set(clean)) == len(clean)
    assert make_unique(["a", "a", "a"]) == ["a", "a__dup2", "a__dup3"]


def test_load_har_reads_original_layout(tmp_path, har):
    X, y, g, label_map = har
    root = tmp_path / "UCI HAR Dataset"
    for split, idx in (("train", slice(0, 500)), ("test", slice(500, None))):
        (root / split).mkdir(parents=True)
        np.savetxt(root / split / f"X_{split}.txt", X.iloc[idx].to_numpy(), fmt="%.6f")
        np.savetxt(root / split / f"y_{split}.txt", y.iloc[idx].to_numpy(), fmt="%d")
        np.savetxt(root / split / f"subject_{split}.txt", g.iloc[idx].to_numpy(), fmt="%d")
    (root / "features.txt").write_text("\n".join(f"{i + 1} {c}-mean()" for i, c in enumerate(X.columns)))
    (root / "activity_labels.txt").write_text("\n".join(f"{i} {a}" for i, a in label_map.items()))

    X2, y2, g2, labels2 = load_har(root)
    assert X2.shape == X.shape and list(X2.columns)[0] == "f0_mean"
    np.testing.assert_allclose(X2.to_numpy(), X.to_numpy(), atol=1e-5)
    assert y2.tolist() == y.tolist() and g2.tolist() == g.tolist() and labels2 == label_map


def test_subject_holdout_and_cv_keep_people_separate(har):
    X, y, g, _ = har
    (X_tr, y_tr, g_tr), (X_te, _, g_te) = subject_holdout(X, y, g, test_subjects=[25, 26, 27, 28, 29, 30])
    assert set(g_te) == {25, 26, 27, 28, 29, 30} and set(g_tr).isdisjoint(g_te)
    assert len(X_tr) + len(X_te) == len(X)
    folds = describe_folds(grouped_cv(), X_tr, y_tr, g_tr)
    assert len(folds) == 4
    assert sorted(s for subs in folds.val_subjects for s in subs) == sorted(set(g_tr))
    tr, va = validation_split(X_tr, y_tr, g_tr)
    assert set(g_tr.iloc[tr]).isdisjoint(g_tr.iloc[va])
    with pytest.raises(AssertionError):
        assert_disjoint([1, 2], [2, 3])


def test_mlp_trains_and_predicts_activity_ids(har):
    X, y, _, _ = har
    model, history, seconds = train_mlp(X.iloc[::2], y.iloc[::2], X.iloc[1::2], y.iloc[1::2], epochs=2)
    assert isinstance(model, MLP) and len(history) == 2 and seconds > 0
    assert set(np.unique(predict_mlp(model, X))) <= set(range(1, 7))


def test_benchmark_reports_size_and_latency(tmp_path, har):
    X, y, _, _ = har
    model = LogisticRegression(max_iter=500).fit(X, y)
    r = benchmark_model("LR test", model, X, tmp_path, warmup=2, repeats=5)
    assert r["size_mb"] > 0 and r["single_window_latency_ms"] > 0 and r["batch_size"] == min(len(X), 1000)
    assert (tmp_path / "LR_test.joblib").exists()


def test_end_to_end_quick_run(tmp_path, har):
    X, y, g, label_map = har
    res = run(
        X,
        y,
        g,
        label_map,
        out_dir=tmp_path / "reports",
        models_dir=tmp_path / "models",
        mlflow_dir=tmp_path / "mlruns",
        quick=True,
    )
    assert all(res["checks"].values())
    decision = pd.read_csv(tmp_path / "reports" / "decision_table.csv")
    assert len(decision) == 7 and decision.macro_f1.is_monotonic_decreasing
    assert decision.size_mb.notna().all()
    assert json.loads((tmp_path / "reports" / "results.json").read_text())["best_model"] == decision.model[0]
    assert (tmp_path / "reports" / "figures" / "tradeoff.png").exists()
