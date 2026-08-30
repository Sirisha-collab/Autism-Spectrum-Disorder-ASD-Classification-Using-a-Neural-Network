"""
Metrics that actually matter for a screening instrument.

Accuracy is close to useless here. A screening tool is judged on whether it
catches cases (sensitivity), how many false alarms it creates at the prevalence
where it will really be used (PPV), and whether its probabilities mean anything
(calibration). This module computes all of that with uncertainty attached.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

RNG = np.random.default_rng(42)


# ------------------------------------------------------- uncertainty
def bootstrap_ci(
    y_true, y_pred, y_proba=None, n_boot: int = 2000, alpha: float = 0.05
) -> dict:
    """
    Percentile bootstrap confidence intervals on the test set.

    A point estimate from 211 test rows is not a result. Report the interval.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = None if y_proba is None else np.asarray(y_proba)
    n = len(y_true)

    stats: dict[str, list[float]] = {
        "Sensitivity": [], "Specificity": [], "PPV": [], "NPV": [],
        "F1": [], "ROC-AUC": [],
    }

    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        stats["Sensitivity"].append(tp / (tp + fn) if tp + fn else np.nan)
        stats["Specificity"].append(tn / (tn + fp) if tn + fp else np.nan)
        stats["PPV"].append(tp / (tp + fp) if tp + fp else np.nan)
        stats["NPV"].append(tn / (tn + fn) if tn + fn else np.nan)
        stats["F1"].append(f1_score(yt, yp, zero_division=0))
        if y_proba is not None:
            stats["ROC-AUC"].append(roc_auc_score(yt, y_proba[idx]))

    out = {}
    for name, values in stats.items():
        arr = np.asarray([v for v in values if not np.isnan(v)])
        if arr.size == 0:
            continue
        out[name] = {
            "estimate": float(np.mean(arr)),
            "lo": float(np.percentile(arr, 100 * alpha / 2)),
            "hi": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        }
    return out


def repeated_cv(pipe, X, y, n_splits: int = 5, n_repeats: int = 6, scoring="f1") -> dict:
    """
    Repeated stratified k-fold. A single 5-fold run has enough variance that two
    models can swap places by chance; repeating it gives an honest interval.
    """
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )
    scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "lo": float(np.percentile(scores, 2.5)),
        "hi": float(np.percentile(scores, 97.5)),
        "n_fits": int(len(scores)),
    }


# ------------------------------------------------------- calibration
def calibration_metrics(y_true, y_proba, n_bins: int = 10) -> dict:
    """
    Brier score and expected calibration error.

    The UI shows a confidence percentage. If the model says 80% and is right 55%
    of the time, that number is a lie to the parent reading it. This measures it.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    brier = float(brier_score_loss(y_true, y_proba))

    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_proba, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_proba[mask].mean())

    return {
        "Brier": brier,
        "ECE": float(ece),
        "AP (PR-AUC)": float(average_precision_score(y_true, y_proba)),
    }


def calibration_points(y_true, y_proba, n_bins: int = 10):
    try:
        prob_true, prob_pred = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy="quantile"
        )
        return prob_true, prob_pred
    except Exception:
        return np.array([]), np.array([])


# ------------------------------------------------------- thresholds
def threshold_sweep(y_true, y_proba, min_sensitivity: float = 0.95) -> pd.DataFrame:
    """
    Every decision threshold with its operating characteristics.

    0.5 is an arbitrary default, not a clinical choice. For a screen you pick the
    threshold that hits a required sensitivity, then report what it costs in
    false positives.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    rows = []
    for t in np.unique(np.round(np.concatenate([y_proba, [0.0, 1.0]]), 3)):
        pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        rows.append(
            {
                "threshold": float(t),
                "sensitivity": sens,
                "specificity": spec,
                "ppv": tp / (tp + fp) if tp + fp else 0.0,
                "npv": tn / (tn + fn) if tn + fn else 0.0,
                "youden_j": sens + spec - 1,
                "n_flagged": int(tp + fp),
            }
        )

    df = pd.DataFrame(rows).drop_duplicates("threshold").sort_values("threshold")
    return df.reset_index(drop=True)


def recommend_threshold(sweep: pd.DataFrame, min_sensitivity: float = 0.95) -> dict:
    """
    Highest specificity among thresholds meeting the sensitivity floor.

    Screening is asymmetric: a missed case delays intervention during the years
    when it helps most, while a false positive costs one assessment that rules
    it out. So sensitivity is a constraint, not something to trade away.
    """
    eligible = sweep[sweep["sensitivity"] >= min_sensitivity]
    if eligible.empty:
        best = sweep.loc[sweep["youden_j"].idxmax()]
        note = (
            f"No threshold reaches {min_sensitivity:.0%} sensitivity; "
            "falling back to max Youden's J."
        )
    else:
        best = eligible.loc[eligible["specificity"].idxmax()]
        note = f"Highest specificity at >= {min_sensitivity:.0%} sensitivity."

    return {
        "threshold": float(best["threshold"]),
        "sensitivity": float(best["sensitivity"]),
        "specificity": float(best["specificity"]),
        "ppv_in_sample": float(best["ppv"]),
        "note": note,
    }


# ------------------------------------------------------- prevalence
def prevalence_adjusted(
    sensitivity: float, specificity: float, prevalences=(0.01, 0.017, 0.03, 0.10, 0.69)
) -> pd.DataFrame:
    """
    PPV and NPV recomputed at prevalences the tool would really meet.

    This is the most important table in the project. Sensitivity and specificity
    are properties of the test, but PPV depends on how common the condition is
    in whoever you screen. This dataset is roughly 69% positive because it was
    collected from families who already suspected something. Deploy the same
    model in a general population clinic and the PPV collapses, even though
    sensitivity and specificity are unchanged.
    """
    rows = []
    for p in prevalences:
        tp = sensitivity * p
        fn = (1 - sensitivity) * p
        tn = specificity * (1 - p)
        fp = (1 - specificity) * (1 - p)
        rows.append(
            {
                "prevalence": p,
                "PPV": tp / (tp + fp) if tp + fp else float("nan"),
                "NPV": tn / (tn + fn) if tn + fn else float("nan"),
                "false_alarms_per_1000": round(fp * 1000, 1),
                "missed_per_1000": round(fn * 1000, 1),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------- fairness
def subgroup_metrics(y_true, y_pred, groups: pd.Series, min_n: int = 20) -> pd.DataFrame:
    """
    Sensitivity and specificity broken out by subgroup.

    A model can look strong overall while missing cases in one group. For a
    paediatric screening tool that is a real harm, so it gets measured, and
    groups too small to support an estimate are reported as such rather than
    given a misleading number.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = pd.Series(groups).reset_index(drop=True)

    rows = []
    for value, idx in groups.groupby(groups).groups.items():
        idx = np.asarray(list(idx))
        yt, yp = y_true[idx], y_pred[idx]
        n = len(idx)

        if n < min_n or len(np.unique(yt)) < 2:
            rows.append(
                {"group": value, "n": n, "sensitivity": np.nan,
                 "specificity": np.nan, "note": "too few rows to estimate"}
            )
            continue

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        rows.append(
            {
                "group": value,
                "n": n,
                "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
                "specificity": tn / (tn + fp) if tn + fp else np.nan,
                "note": "",
            }
        )

    df = pd.DataFrame(rows).sort_values("n", ascending=False)
    reliable = df.dropna(subset=["sensitivity"])
    if len(reliable) > 1:
        gap = reliable["sensitivity"].max() - reliable["sensitivity"].min()
        df.attrs["sensitivity_gap"] = float(gap)
    return df.reset_index(drop=True)


# ------------------------------------------------------- model comparison
def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """
    McNemar's test on the discordant pairs of two models.

    Comparing two accuracies on the same test set by eyeballing which is bigger
    ignores that they are not independent samples. This asks whether the
    difference is larger than chance.
    """
    y_true = np.asarray(y_true)
    a_right = np.asarray(pred_a) == y_true
    b_right = np.asarray(pred_b) == y_true

    n01 = int(np.sum(a_right & ~b_right))
    n10 = int(np.sum(~a_right & b_right))

    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0,
                "note": "models agree on every test row"}

    try:
        from scipy.stats import binomtest
        p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
    except Exception:
        chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p = float(np.exp(-chi2 / 2))

    return {
        "n01": n01,
        "n10": n10,
        "p_value": float(p),
        "note": "significant at 0.05" if p < 0.05 else "not distinguishable",
    }
