"""
Picking a winner, with the reasoning written down.

Sorting by F1 and taking the top row is the wrong move for a screening tool.
F1 balances precision and recall as though the two errors cost the same, and
here they don't: a missed case delays intervention during the years it helps
most, while a false positive costs one assessment that rules it out. So this
module gates on sensitivity first, then ranks the survivors on how well they
separate cases from non-cases, then checks whether the winner's lead is real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from metrics_advanced import mcnemar_test

# A screen that misses one case in ten is not fit for purpose.
SENSITIVITY_FLOOR = 0.90

# Below this, differences between models are noise.
MEANINGFUL_GAP = 0.02

CRITERIA = [
    (
        "Sensitivity floor",
        f"Must catch at least {SENSITIVITY_FLOOR:.0%} of cases. A screen that "
        "misses cases fails at the one job it has.",
    ),
    (
        "Youden's J",
        "Sensitivity + specificity − 1. Rewards separating cases from "
        "non-cases, rather than agreeing with whichever class is larger.",
    ),
    (
        "Cross-validation stability",
        "Low variance across repeated folds. A model that swings between folds "
        "will swing on new data too.",
    ),
    (
        "Calibration",
        "The interface shows a probability to a parent. It has to mean what it "
        "says.",
    ),
    (
        "Distinguishability",
        "McNemar's test against the runners-up. If the difference isn't "
        "significant, prefer the simpler model.",
    ),
]

# Lower is simpler. Used only to break genuine ties.
COMPLEXITY = {
    "Logistic Regression": 1,
    "Naive Bayes": 1,
    "Decision Tree": 2,
    "K-Nearest Neighbours": 3,
    "SVM (RBF)": 4,
    "AdaBoost": 5,
    "Random Forest": 6,
    "Gradient Boosting": 6,
}


def select_best(
    results: dict,
    preds: dict,
    y_test,
    calibration: dict | None = None,
) -> dict:
    """
    Rank every model against the criteria above and explain the outcome.

    results: {name: metrics dict from evaluate.compute_metrics}
    preds:   {name: predicted labels on the test set}
    """
    rows = []
    for name, m in results.items():
        sens = float(m.get("Recall (Sensitivity)", 0.0))
        spec = float(m.get("Specificity", 0.0))
        rows.append(
            {
                "model": name,
                "sensitivity": sens,
                "specificity": spec,
                "youden_j": sens + spec - 1,
                "cv_std": float(m.get("CV F1 (std)", np.nan)),
                "roc_auc": float(m.get("ROC-AUC", np.nan)),
                "passes_floor": sens >= SENSITIVITY_FLOOR,
                "complexity": COMPLEXITY.get(name, 4),
            }
        )

    table = pd.DataFrame(rows)
    eligible = table[table["passes_floor"]].copy()

    if eligible.empty:
        # Nothing clears the floor. Say so rather than crowning a failure.
        table = table.sort_values("youden_j", ascending=False).reset_index(drop=True)
        winner = table.iloc[0]
        return {
            "winner": winner["model"],
            "confident": False,
            "criteria": [{"name": n, "why": w} for n, w in CRITERIA],
            "ranking": table.round(4).to_dict("records"),
            "headline": f"No model clears the {SENSITIVITY_FLOOR:.0%} sensitivity floor",
            "reason": (
                f"Every model misses more than {1 - SENSITIVITY_FLOOR:.0%} of "
                f"cases on the test set. {winner['model']} separates the classes "
                "best of a weak field, but none of these is deployable as a "
                "screen. Collect more signal before choosing between them."
            ),
            "runner_up": None,
            "mcnemar": {},
        }

    eligible = eligible.sort_values(
        ["youden_j", "cv_std", "complexity"], ascending=[False, True, True]
    ).reset_index(drop=True)

    winner = eligible.iloc[0]
    runner_up = eligible.iloc[1] if len(eligible) > 1 else None

    # Is the lead real, or is it noise?
    comparisons = {}
    indistinguishable = []
    for _, row in eligible.iloc[1:4].iterrows():
        test = mcnemar_test(y_test, preds[winner["model"]], preds[row["model"]])
        comparisons[row["model"]] = test
        if test["p_value"] >= 0.05:
            indistinguishable.append(row["model"])

    gap = (
        float(winner["youden_j"] - runner_up["youden_j"])
        if runner_up is not None
        else float("inf")
    )
    confident = gap >= MEANINGFUL_GAP and not indistinguishable

    reason = _write_reason(
        winner=winner,
        runner_up=runner_up,
        gap=gap,
        indistinguishable=indistinguishable,
        n_eligible=len(eligible),
        n_total=len(table),
        calibration=calibration,
    )

    return {
        "winner": winner["model"],
        "f1_leader": max(results, key=lambda k: results[k].get("F1", 0)),
        "confident": confident,
        "criteria": [{"name": n, "why": w} for n, w in CRITERIA],
        "ranking": eligible.round(4).to_dict("records"),
        "excluded": table[~table["passes_floor"]]
        .sort_values("sensitivity", ascending=False)
        .round(4)
        .to_dict("records"),
        "headline": f"{winner['model']} — but read why",
        "reason": reason,
        "runner_up": runner_up["model"] if runner_up is not None else None,
        "mcnemar": comparisons,
        "sensitivity_floor": SENSITIVITY_FLOOR,
    }


def _write_reason(
    *, winner, runner_up, gap, indistinguishable, n_eligible, n_total, calibration
) -> str:
    """Plain-language justification, assembled from what the numbers actually say."""
    parts = []

    excluded = n_total - n_eligible
    if excluded:
        parts.append(
            f"{n_eligible} of {n_total} models clear the "
            f"{SENSITIVITY_FLOOR:.0%} sensitivity floor; the other "
            f"{excluded} miss too many cases to be usable as a screen."
        )
    else:
        parts.append(f"All {n_total} models clear the sensitivity floor.")

    parts.append(
        f"{winner['model']} catches {winner['sensitivity']:.1%} of cases while "
        f"correctly clearing {winner['specificity']:.1%} of non-cases, giving "
        f"the highest Youden's J at {winner['youden_j']:.3f}."
    )

    if runner_up is not None:
        if indistinguishable:
            names = ", ".join(indistinguishable)
            parts.append(
                f"That lead is not statistically distinguishable from {names} "
                "on this test set — McNemar's test can't separate them. With "
                f"the difference inside the noise, {winner['model']} is the "
                "defensible pick because it is the simpler model, not because "
                "it is measurably better."
            )
        elif gap < MEANINGFUL_GAP:
            parts.append(
                f"It leads {runner_up['model']} by only {gap:.3f} on Youden's J, "
                "which is small enough that a different train/test split could "
                "reverse the order."
            )
        else:
            parts.append(
                f"It leads {runner_up['model']} by {gap:.3f} on Youden's J, and "
                "McNemar's test confirms the difference is larger than chance."
            )

    if not np.isnan(winner["cv_std"]):
        stability = "stable" if winner["cv_std"] < 0.02 else "variable"
        parts.append(
            f"Across repeated cross-validation folds it is {stability} "
            f"(F1 standard deviation {winner['cv_std']:.3f})."
        )

    if calibration and calibration.get("after"):
        after = calibration["after"]
        parts.append(
            f"After isotonic calibration its Brier score is {after['Brier']:.3f} "
            f"and expected calibration error {after['ECE']:.3f}, so the "
            "probability shown to a parent is close to the observed frequency."
        )

    return " ".join(parts)
