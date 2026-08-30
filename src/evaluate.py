"""
Metrics and plots for model comparison.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # write files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import REPORT_DIR


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    m = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall (Sensitivity)": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m["Specificity"] = tn / (tn + fp) if (tn + fp) else 0.0
    m["ROC-AUC"] = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
    m["TP"], m["FP"], m["TN"], m["FN"] = int(tp), int(fp), int(tn), int(fn)
    return m


def metrics_table(results: dict) -> pd.DataFrame:
    """results: {model_name: metrics_dict} -> sorted DataFrame."""
    df = pd.DataFrame(results).T
    order = [
        "Accuracy", "Balanced Acc", "Precision", "Recall (Sensitivity)",
        "Specificity", "F1", "ROC-AUC", "MCC", "TP", "FP", "TN", "FN",
    ]
    df = df[[c for c in order if c in df.columns]]
    return df.sort_values("F1", ascending=False)


def print_classification_reports(y_true, preds: dict) -> str:
    chunks = []
    for name, y_pred in preds.items():
        chunks.append(f"\n=== {name} ===")
        chunks.append(
            classification_report(
                y_true, y_pred, target_names=["No ASD traits", "ASD traits"],
                zero_division=0,
            )
        )
    text = "\n".join(chunks)
    return text


# ------------------------------------------------------------------ plots
def plot_confusion_matrices(y_true, preds: dict, path=None):
    path = path or REPORT_DIR / "confusion_matrices.png"
    n = len(preds)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (name, y_pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=10)
        ax.set_xticks([0, 1], ["Pred No", "Pred Yes"], fontsize=8)
        ax.set_yticks([0, 1], ["True No", "True Yes"], fontsize=8)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12,
                )
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Confusion matrices (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_roc_curves(y_true, probas: dict, path=None):
    path = path or REPORT_DIR / "roc_curves.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in probas.items():
        if proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_model_comparison(table: pd.DataFrame, path=None):
    path = path or REPORT_DIR / "model_comparison.png"
    cols = ["Accuracy", "Precision", "Recall (Sensitivity)", "F1"]
    sub = table[cols]
    ax = sub.plot(kind="bar", figsize=(11, 5.5), ylim=(0, 1.05), rot=25)
    ax.set_title("Model comparison on the test set")
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_names, name, path=None):
    """Works for tree ensembles (feature_importances_) and linear models (coef_)."""
    path = path or REPORT_DIR / f"feature_importance_{name.replace(' ', '_')}.png"
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    if hasattr(clf, "feature_importances_"):
        vals = clf.feature_importances_
        label = "Gini importance"
    elif hasattr(clf, "coef_"):
        vals = np.abs(clf.coef_).ravel()
        label = "|coefficient|"
    else:
        return None

    idx = np.argsort(vals)[-20:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_names[i] for i in idx], vals[idx])
    ax.set_xlabel(label)
    ax.set_title(f"Top features — {name}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_class_balance(y, path=None):
    path = path or REPORT_DIR / "class_balance.png"
    counts = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["No ASD traits", "ASD traits"], counts.values, color=["#4C72B0", "#DD8452"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_title("Class balance")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ------------------------------------------------ production-grade plots
def plot_pr_curves(y_true, probas: dict, path=None):
    """Precision-recall curves. More informative than ROC when classes skew."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    path = path or REPORT_DIR / "precision_recall_curves.png"
    baseline = np.mean(y_true)
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in probas.items():
        if proba is None:
            continue
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.axhline(baseline, ls="--", c="k", lw=1,
               label=f"Always-yes baseline ({baseline:.3f})")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-recall curves")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_calibration(y_true, probas: dict, path=None):
    """Are the confidence numbers shown in the UI actually trustworthy?"""
    from metrics_advanced import calibration_points

    path = path or REPORT_DIR / "calibration.png"
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
    for name, proba in probas.items():
        if proba is None:
            continue
        prob_true, prob_pred = calibration_points(y_true, proba)
        if prob_true.size:
            ax.plot(prob_pred, prob_true, marker="o", ms=4, label=name)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (reliability diagram)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_threshold_sweep(sweep, recommended: dict, name: str, path=None):
    """Sensitivity/specificity trade-off with the chosen operating point."""
    path = path or REPORT_DIR / "threshold_sweep.png"
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(sweep["threshold"], sweep["sensitivity"], label="Sensitivity")
    ax.plot(sweep["threshold"], sweep["specificity"], label="Specificity")
    ax.plot(sweep["threshold"], sweep["ppv"], label="PPV", alpha=0.7)
    ax.axvline(recommended["threshold"], c="k", ls=":", lw=1.5,
               label=f"Chosen = {recommended['threshold']:.3f}")
    ax.axvline(0.5, c="gray", ls="--", lw=1, alpha=0.6, label="Default 0.5")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Operating characteristics — {name}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_prevalence_curve(sensitivity: float, specificity: float, path=None):
    """How PPV collapses as the screened population gets less enriched."""
    path = path or REPORT_DIR / "ppv_vs_prevalence.png"
    prev = np.linspace(0.005, 0.75, 300)
    tp = sensitivity * prev
    fp = (1 - specificity) * (1 - prev)
    ppv = np.divide(tp, tp + fp, out=np.zeros_like(prev), where=(tp + fp) > 0)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(prev * 100, ppv, lw=2, color="#2F6F62")
    ax.axvline(1.7, ls="--", c="#A9741F", lw=1.4,
               label="General population (~1.7%)")
    ax.axvline(69, ls="--", c="gray", lw=1.4, label="This dataset (~69%)")
    ax.set_xlabel("Prevalence in the screened population (%)")
    ax.set_ylabel("Positive predictive value")
    ax.set_ylim(0, 1.02)
    ax.set_title("PPV depends on who you screen, not just on the model")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
