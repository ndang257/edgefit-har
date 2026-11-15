"""Metrics and plots. Macro F1 is the primary metric: every activity counts equally."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def scores(y_true, y_pred) -> dict:
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_f1": f1_score(y_true, y_pred, average="macro")}


def class_report(y_true, y_pred, label_map: dict) -> pd.DataFrame:
    ids = sorted(label_map)
    report = classification_report(
        y_true, y_pred, labels=ids, target_names=[label_map[i] for i in ids], output_dict=True
    )
    return pd.DataFrame(report).T


def top_confusions(y_true, y_pred, label_map: dict, n: int = 5) -> pd.DataFrame:
    ids = sorted(label_map)
    names = [label_map[i] for i in ids]
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=ids), index=names, columns=names)
    pairs = cm.stack().rename_axis(["true", "predicted"]).reset_index(name="count")
    return pairs[pairs.true != pairs.predicted].nlargest(n, "count").reset_index(drop=True)


def plot_confusion(y_true, y_pred, label_map: dict, title: str, path=None):
    ids = sorted(label_map)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred, labels=ids), display_labels=[label_map[i] for i in ids]
    ).plot(cmap="Blues", ax=ax, colorbar=False)
    ax.set(title=title, xlabel="Predicted", ylabel="True")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_class_distribution(freq: pd.DataFrame, path=None):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(freq["activity"], freq["count"], color="#2a7f62")
    ax.set(title="Windows per activity", xlabel="", ylabel="Number of windows")
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_permutation_importance(perm: pd.DataFrame, n: int = 20, path=None):
    top = perm.head(n)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(top.index, top["perm_mean"], xerr=top["perm_std"], color="#2a7f62")
    ax.set(title=f"Permutation importance (Random Forest, 25 features) - top {n}", xlabel="Mean drop in Macro F1")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_mlp_history(history: pd.DataFrame, path=None):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(history.epoch, history.train_loss, "o-", color="#888888", label="Training loss")
    ax1.set(xlabel="Epoch", ylabel="Training loss")
    ax2 = ax1.twinx()
    ax2.plot(history.epoch, history.val_macro_f1, "o-", color="#2a7f62", label="Validation Macro F1")
    ax2.set_ylabel("Validation Macro F1")
    best = history.loc[history.val_macro_f1.idxmax()]
    ax2.axvline(best.epoch, ls="--", color="#2a7f62", alpha=0.5)
    fig.legend(loc="lower right", bbox_to_anchor=(0.86, 0.2))
    ax1.set_title("PyTorch MLP training (best epoch kept)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig


def plot_tradeoff(decision: pd.DataFrame, path=None):
    """Macro F1 against single-window CPU latency; marker area grows with model size."""
    d = decision.dropna(subset=["single_window_latency_ms"])
    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = 60 + 40 * np.sqrt(d["size_mb"])
    colors = ["#2a7f62" if n == 25 else "#4a6fa5" for n in d["n_features"]]
    ax.scatter(d["single_window_latency_ms"], d["macro_f1"], s=sizes, c=colors, alpha=0.8, edgecolor="white")
    for _, r in d.iterrows():
        ax.annotate(
            f"{r['model']}\n{r['size_mb']:.2f} MB",
            (r["single_window_latency_ms"], r["macro_f1"]),
            textcoords="offset points",
            xytext=(8, -4),
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set(
        xlabel="Single-window CPU latency (ms, log scale)",
        ylabel="Macro F1 on unseen subjects",
        title="Macro F1 vs. CPU latency per window (marker size = model size)",
    )
    ax.scatter([], [], c="#4a6fa5", label="561 features")
    ax.scatter([], [], c="#2a7f62", label="25 features")
    ax.legend(loc="lower left", frameon=False)
    ax.set_xlim(right=ax.get_xlim()[1] * 6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    return fig
