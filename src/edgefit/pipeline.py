"""End-to-end experiment: split, train all candidates, evaluate, benchmark, tune, track.

Kept in the package (not only in scripts/train.py) so the tests can run it on synthetic data.
"""

import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV  # noqa: E402

from edgefit import config as C  # noqa: E402
from edgefit.benchmark import benchmark_model, environment  # noqa: E402
from edgefit.evaluation import (  # noqa: E402
    class_report,
    plot_class_distribution,
    plot_confusion,
    plot_mlp_history,
    plot_permutation_importance,
    plot_tradeoff,
    scores,
    top_confusions,
)
from edgefit.models import classical_models, predict_mlp, train_mlp  # noqa: E402
from edgefit.split import describe_folds, grouped_cv, subject_holdout, validation_split  # noqa: E402

warnings.filterwarnings("ignore", message=".*feature names.*")

FULL = {
    "n_estimators": 300,
    "mlp_epochs": C.MLP_EPOCHS,
    "hpo_iter": C.HPO_ITERATIONS,
    "param_dist": C.RF_PARAM_DIST,
    "perm_repeats": 10,
    "bench": {"warmup": C.BENCH_WARMUP, "repeats": C.BENCH_REPEATS},
}
QUICK = {
    "n_estimators": 30,
    "mlp_epochs": 3,
    "hpo_iter": 2,
    "param_dist": {
        "n_estimators": [20, 30],
        "max_depth": [None, 15],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt"],
    },
    "perm_repeats": 2,
    "bench": {"warmup": 3, "repeats": 20},
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(
    X_all,
    y_all,
    groups_all,
    label_map,
    out_dir: Path,
    models_dir: Path,
    mlflow_dir: Path,
    quick: bool = False,
    test_subjects=C.TEST_SUBJECTS,
) -> dict:
    s = QUICK if quick else FULL
    out_dir, models_dir, mlflow_dir = Path(out_dir), Path(models_dir), Path(mlflow_dir)
    fig_dir = out_dir / "figures"
    for d in (out_dir, fig_dir, models_dir, mlflow_dir):
        d.mkdir(parents=True, exist_ok=True)
    names = y_all.map(label_map)

    # 1. Data understanding -------------------------------------------------------------
    freq = names.value_counts().rename_axis("activity").reset_index(name="count")
    freq["share"] = freq["count"] / freq["count"].sum()
    freq.to_csv(out_dir / "class_distribution.csv", index=False)
    plot_class_distribution(freq, fig_dir / "class_distribution.png")
    per_subject = pd.crosstab(groups_all, names)
    per_subject.to_csv(out_dir / "windows_per_subject.csv")
    log(f"Data: {X_all.shape[0]} windows, {X_all.shape[1]} features, {groups_all.nunique()} subjects")

    # 2. Subject-aware holdout + grouped CV ----------------------------------------------
    (X_tr, y_tr, g_tr), (X_te, y_te, g_te) = subject_holdout(X_all, y_all, groups_all, test_subjects)
    cv = grouped_cv()
    folds = describe_folds(cv, X_tr, y_tr, g_tr)
    folds.to_csv(out_dir / "cv_folds.csv", index=False)
    log(
        f"Holdout: {len(X_tr)} train rows ({g_tr.nunique()} subjects) / {len(X_te)} test rows "
        f"(subjects {sorted(map(int, g_te.unique()))})"
    )

    fitted, feats, rows = {}, {}, []

    def fit_eval(name, model, cols):
        t0 = time.perf_counter()
        model.fit(X_tr[cols], y_tr)
        train_time = time.perf_counter() - t0
        fitted[name], feats[name] = model, list(cols)
        rows.append({"model": name, "train_time_s": train_time, **scores(y_te, model.predict(X_te[cols]))})
        log(f"{name:28s} macro F1 {rows[-1]['macro_f1']:.4f}  ({train_time:.1f}s)")

    # 3. Classical models on all features, then reduced-feature edge models --------------
    all_cols = list(X_tr.columns)
    base = classical_models(s["n_estimators"])
    for fam, model in base.items():
        fit_eval(f"{fam} ({len(all_cols)})", clone(model), all_cols)

    rf_full = fitted[f"Random Forest ({len(all_cols)})"]
    importances = pd.Series(rf_full.feature_importances_, index=all_cols).sort_values(ascending=False)
    top = importances.head(C.N_TOP_FEATURES).index.tolist()
    importances.head(C.N_TOP_FEATURES).rename("impurity_importance").to_csv(out_dir / "top_features_impurity.csv")
    for fam in ("Random Forest", "LightGBM"):
        fit_eval(f"{fam} ({len(top)})", clone(base[fam]), top)

    # 4. PyTorch MLP (validation subjects carved out of the development data) -----------
    tr_idx, va_idx = validation_split(X_tr, y_tr, g_tr)
    mlp, history, mlp_time = train_mlp(
        X_tr.iloc[tr_idx], y_tr.iloc[tr_idx], X_tr.iloc[va_idx], y_tr.iloc[va_idx], epochs=s["mlp_epochs"]
    )
    history.to_csv(out_dir / "mlp_history.csv", index=False)
    plot_mlp_history(history, fig_dir / "mlp_training.png")
    mlp_name = f"PyTorch MLP ({len(all_cols)})"
    fitted[mlp_name], feats[mlp_name] = mlp, all_cols
    rows.append({"model": mlp_name, "train_time_s": mlp_time, **scores(y_te, predict_mlp(mlp, X_te[all_cols]))})
    log(f"{mlp_name:28s} macro F1 {rows[-1]['macro_f1']:.4f}  ({mlp_time:.1f}s)")

    # 5. Comparison + class-level analysis of the best model ----------------------------
    comparison = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    best = comparison.loc[0, "model"]
    model = fitted[best]
    pred_best = predict_mlp(model, X_te[feats[best]]) if best == mlp_name else model.predict(X_te[feats[best]])
    class_report(y_te, pred_best, label_map).to_csv(out_dir / "best_model_class_report.csv")
    top_confusions(y_te, pred_best, label_map).to_csv(out_dir / "best_model_top_confusions.csv", index=False)
    plot_confusion(y_te, pred_best, label_map, f"Confusion matrix: {best}", fig_dir / "confusion_matrix.png")

    # 6a. Permutation importance on the 25-feature Random Forest -------------------------
    rf_top_name = f"Random Forest ({len(top)})"
    perm = permutation_importance(
        fitted[rf_top_name],
        X_te[top],
        y_te,
        scoring="f1_macro",
        n_repeats=s["perm_repeats"],
        random_state=C.RANDOM_STATE,
        n_jobs=-1,
    )
    perm_df = pd.DataFrame(
        {"perm_mean": perm.importances_mean, "perm_std": perm.importances_std, "impurity_rank": range(1, len(top) + 1)},
        index=pd.Index(top, name="feature"),
    ).sort_values("perm_mean", ascending=False)
    perm_df.to_csv(out_dir / "permutation_importance.csv")
    plot_permutation_importance(perm_df, n=min(20, len(top)), path=fig_dir / "permutation_importance.png")
    rank_corr = perm_df["impurity_rank"].corr(perm_df["perm_mean"].rank(ascending=False), method="spearman")

    # 7. Grouped-CV hyperparameter search (development subjects only) --------------------
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=C.RANDOM_STATE, n_jobs=-1),
        s["param_dist"],
        n_iter=s["hpo_iter"],
        scoring="f1_macro",
        cv=cv,
        random_state=C.RANDOM_STATE,
    )
    search.fit(X_tr, y_tr, groups=g_tr)
    tuned_name = f"Random Forest tuned ({len(all_cols)})"
    fitted[tuned_name], feats[tuned_name] = search.best_estimator_, all_cols
    tuned_row = {
        "model": tuned_name,
        "train_time_s": search.refit_time_,
        **scores(y_te, search.best_estimator_.predict(X_te)),
    }
    pd.DataFrame(search.cv_results_).to_csv(out_dir / "hpo_cv_results.csv", index=False)
    log(f"{tuned_name:28s} macro F1 {tuned_row['macro_f1']:.4f}  (grouped-CV {search.best_score_:.4f})")

    # 6b. Edge benchmark (CPU, single thread) -------------------------------------------
    env = environment(device="cuda" if next(mlp.parameters()).is_cuda else "cpu")
    bench = []
    for name, m in fitted.items():
        r = benchmark_model(name, m, X_te[feats[name]], models_dir, **s["bench"])
        bench.append({**r, "n_features": len(feats[name])})
    bench_df = pd.DataFrame(bench)

    decision = (
        pd.concat([comparison, pd.DataFrame([tuned_row])], ignore_index=True)
        .merge(bench_df, on="model")
        .sort_values("macro_f1", ascending=False)
        .reset_index(drop=True)
    )
    decision = decision[
        [
            "model",
            "macro_f1",
            "accuracy",
            "train_time_s",
            "n_features",
            "size_mb",
            "single_window_latency_ms",
            "batch_latency_ms",
        ]
    ]
    decision.to_csv(out_dir / "decision_table.csv", index=False)
    plot_tradeoff(decision, fig_dir / "tradeoff.png")
    plt.close("all")

    # 8. Experiment tracking ---------------------------------------------------------------
    mlflow.set_tracking_uri(f"sqlite:///{(mlflow_dir / 'mlflow.db').resolve()}")
    exp_name = "EdgeFit_UCI_HAR" + ("_quick" if quick else "")
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name, artifact_location=(mlflow_dir / "artifacts").resolve().as_uri())
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name="rf_grouped_cv_hpo") as mlrun:
        mlflow.log_params(
            {
                "test_subjects": str(list(test_subjects)),
                "cv": f"StratifiedGroupKFold({C.CV_FOLDS})",
                "primary_metric": "macro_f1",
                "n_iter": s["hpo_iter"],
                **{f"best_{k}": v for k, v in search.best_params_.items()},
            }
        )
        mlflow.log_metrics(
            {
                "cv_macro_f1": search.best_score_,
                "test_macro_f1": tuned_row["macro_f1"],
                "test_accuracy": tuned_row["accuracy"],
            }
        )
        mlflow.log_artifact(str(out_dir / "decision_table.csv"))
        run_id = mlrun.info.run_id

    # Reproducibility checks ---------------------------------------------------------------
    checks = {
        "train_test_subjects_disjoint": set(g_tr).isdisjoint(set(g_te)),
        "test_subjects_as_configured": sorted(map(int, g_te.unique())) == sorted(test_subjects),
        "mlp_validation_within_development_data": set(g_tr.iloc[va_idx]).isdisjoint(set(g_te)),
        "hpo_used_grouped_cv": search.cv is cv,
        "all_models_benchmarked": set(fitted) == set(bench_df.model),
    }
    assert all(checks.values()), checks

    results = {
        "quick": quick,
        "n_windows": int(len(X_all)),
        "n_features": int(X_all.shape[1]),
        "n_subjects": int(groups_all.nunique()),
        "test_subjects": [int(t) for t in test_subjects],
        "train_rows": int(len(X_tr)),
        "test_rows": int(len(X_te)),
        "best_model": best,
        "top_features": top,
        "impurity_vs_permutation_spearman": float(rank_corr),
        "mlp": {
            "best_epoch": int(history.val_macro_f1.idxmax() + 1),
            "best_val_macro_f1": float(history.val_macro_f1.max()),
            "train_subjects": sorted(map(int, g_tr.iloc[tr_idx].unique())),
            "val_subjects": sorted(map(int, g_tr.iloc[va_idx].unique())),
        },
        "hpo": {
            "best_params": search.best_params_,
            "cv_macro_f1": float(search.best_score_),
            "test_macro_f1": float(tuned_row["macro_f1"]),
            "mlflow_run_id": run_id,
        },
        "environment": env,
        "benchmark_notes": "CPU, single thread, median of repeated single-window predictions after warm-up, "
        f"one batch of up to {C.BENCH_BATCH} test windows. Timings are hardware-dependent.",
        "checks": checks,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))
    log(f"Done. Results in {out_dir}")
    return results
