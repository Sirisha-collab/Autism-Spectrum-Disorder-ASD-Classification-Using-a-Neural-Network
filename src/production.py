"""
Turns a fitted model into something defensible: calibrated probabilities, a
chosen operating threshold, uncertainty intervals, subgroup checks and a written
model card.

Called at the end of train.py.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from config import MODEL_DIR, REPORT_DIR
from evaluate import (
    plot_calibration,
    plot_pr_curves,
    plot_prevalence_curve,
    plot_threshold_sweep,
)
from model_selection import select_best
from metrics_advanced import (
    bootstrap_ci,
    calibration_metrics,
    mcnemar_test,
    prevalence_adjusted,
    recommend_threshold,
    subgroup_metrics,
    threshold_sweep,
)

# Roughly 1 in 60 children, the figure commonly cited for general-population
# prevalence. Change it here if your write-up uses a different source.
GENERAL_POPULATION_PREVALENCE = 0.017
MIN_SENSITIVITY = 0.95


def calibrate(pipe, X_train, y_train):
    """
    Wrap the fitted pipeline so its predict_proba is trustworthy.

    Tree ensembles push probabilities toward 0 and 1; SVMs need Platt scaling.
    The UI shows this number to a parent, so it has to mean what it says.
    """
    try:
        calibrated = CalibratedClassifierCV(pipe, method="isotonic", cv=5)
        calibrated.fit(X_train, y_train)
        return calibrated, "isotonic"
    except Exception:
        try:
            calibrated = CalibratedClassifierCV(pipe, method="sigmoid", cv=5)
            calibrated.fit(X_train, y_train)
            return calibrated, "sigmoid"
        except Exception:
            return pipe, "none"


def evaluate_production(
    *,
    best_name: str,
    best_pipe,
    fitted: dict,
    results: dict,
    preds: dict,
    probas: dict,
    X_train,
    y_train,
    X_test,
    y_test,
    df_test_raw: pd.DataFrame,
    drop_leakage: bool,
    drop_behaviour: bool,
) -> dict:
    """Run the full production evaluation and write every artifact."""
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_model": best_name,
        "leakage_column_dropped": drop_leakage,
        "behaviour_items_dropped": drop_behaviour,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    print("\n" + "=" * 70)
    print("PRODUCTION EVALUATION")
    print("=" * 70)

    # --- 0. choose the model, on screening criteria not F1 --------------
    selection = select_best(results=results, preds=preds, y_test=y_test)
    if selection["winner"] != best_name:
        print(
            f"\nF1 ranked {best_name} first, but on screening criteria the "
            f"choice is {selection['winner']}. Deploying the latter."
        )
        best_name = selection["winner"]
        best_pipe = fitted[best_name]
    report["best_model"] = best_name
    report["f1_ranked_first"] = selection.get("f1_leader")

    # --- 1. uncertainty on the headline numbers ------------------------
    ci = bootstrap_ci(y_test, preds[best_name], probas.get(best_name))
    report["bootstrap_ci"] = ci
    print("\n95% bootstrap confidence intervals (test set):")
    for metric, v in ci.items():
        print(f"  {metric:<14} {v['estimate']:.3f}  [{v['lo']:.3f}, {v['hi']:.3f}]")

    # --- 2. calibration ------------------------------------------------
    calibrated, method = calibrate(best_pipe, X_train, y_train)
    report["calibration_method"] = method

    cal_proba = None
    if hasattr(calibrated, "predict_proba"):
        cal_proba = calibrated.predict_proba(X_test)[:, 1]
        before = (
            calibration_metrics(y_test, probas[best_name])
            if probas.get(best_name) is not None
            else None
        )
        after = calibration_metrics(y_test, cal_proba)
        report["calibration"] = {"before": before, "after": after}
        print(f"\nCalibration ({method}):")
        if before:
            print(f"  Brier  {before['Brier']:.4f} -> {after['Brier']:.4f}")
            print(f"  ECE    {before['ECE']:.4f} -> {after['ECE']:.4f}")
        print(f"  PR-AUC {after['AP (PR-AUC)']:.4f}")

    # --- 3. operating threshold ----------------------------------------
    proba_for_threshold = cal_proba if cal_proba is not None else probas.get(best_name)
    recommended = {"threshold": 0.5, "note": "no probabilities available"}

    if proba_for_threshold is not None:
        sweep = threshold_sweep(y_test, proba_for_threshold)
        recommended = recommend_threshold(sweep, MIN_SENSITIVITY)
        sweep.round(4).to_csv(REPORT_DIR / "threshold_sweep.csv", index=False)
        plot_threshold_sweep(sweep, recommended, best_name)
        print(f"\nOperating threshold: {recommended['threshold']:.3f}")
        print(f"  {recommended['note']}")
        print(
            f"  sensitivity {recommended['sensitivity']:.3f}   "
            f"specificity {recommended['specificity']:.3f}"
        )
    report["operating_point"] = recommended

    # --- 4. prevalence adjustment --------------------------------------
    sens = recommended.get("sensitivity", ci.get("Sensitivity", {}).get("estimate", 0))
    spec = recommended.get("specificity", ci.get("Specificity", {}).get("estimate", 0))
    prev_table = prevalence_adjusted(sens, spec)
    prev_table.round(4).to_csv(REPORT_DIR / "prevalence_adjusted.csv", index=False)
    plot_prevalence_curve(sens, spec)

    print("\nPPV / NPV by prevalence of the screened population:")
    print(prev_table.round(3).to_string(index=False))

    at_pop = prev_table.iloc[
        (prev_table["prevalence"] - GENERAL_POPULATION_PREVALENCE).abs().argmin()
    ]
    report["prevalence_adjusted"] = prev_table.round(4).to_dict("records")
    print(
        f"\n  At {at_pop['prevalence']:.1%} prevalence the PPV is "
        f"{at_pop['PPV']:.1%} — of every 100 children flagged, roughly "
        f"{round(at_pop['PPV'] * 100)} would turn out to have the condition."
    )

    # --- 5. subgroup fairness ------------------------------------------
    report["subgroups"] = {}
    for col in ["Sex", "Ethnicity"]:
        if col not in df_test_raw.columns:
            continue
        table = subgroup_metrics(y_test, preds[best_name], df_test_raw[col])
        table.round(4).to_csv(
            REPORT_DIR / f"subgroup_{col.lower()}.csv", index=False
        )
        gap = table.attrs.get("sensitivity_gap")
        report["subgroups"][col] = {
            "rows": table.round(4).where(pd.notna(table), None).to_dict("records"),
            "sensitivity_gap": gap,
        }
        print(f"\nSensitivity by {col}:")
        print(table.round(3).to_string(index=False))
        if gap is not None:
            verdict = "acceptable" if gap < 0.10 else "REVIEW — gap above 10 points"
            print(f"  Largest gap between groups: {gap:.3f} ({verdict})")

    # --- 5b. explain the choice ------------------------------------------
    selection["reason"] = select_best(
        results=results, preds=preds, y_test=y_test,
        calibration=report.get("calibration"),
    )["reason"]
    report["selection"] = selection
    print("\n" + "-" * 70)
    print(f"SELECTED: {selection['winner']}")
    print("-" * 70)
    import textwrap
    for line in textwrap.wrap(selection["reason"], 68):
        print("  " + line)
    if not selection["confident"]:
        print("\n  Note: the margin over the runner-up is within noise.")

    # --- 6. is the winner actually better? ------------------------------
    others = [n for n in preds if n != best_name][:3]
    report["mcnemar"] = {}
    if others:
        print(f"\nMcNemar vs {best_name} (are the differences real?):")
        for other in others:
            result = mcnemar_test(y_test, preds[best_name], preds[other])
            report["mcnemar"][other] = result
            print(
                f"  vs {other:<22} p={result['p_value']:.4f}  {result['note']}"
            )

    # --- 7. extra plots -------------------------------------------------
    plot_pr_curves(y_test, probas)
    plot_calibration(y_test, probas)

    # --- 8. save the deployable bundle ----------------------------------
    bundle = {
        "model": calibrated,
        "uncalibrated_model": best_pipe,
        "model_name": best_name,
        "feature_columns": list(X_train.columns),
        "drop_leakage": drop_leakage,
        "drop_behaviour": drop_behaviour,
        "threshold": recommended["threshold"],
        "calibration_method": method,
        "metrics": {
            "sensitivity": sens,
            "specificity": spec,
            "bootstrap_ci": ci,
        },
        "trained_at": report["generated_at"],
        "version": "1.0.0",
    }
    import joblib

    joblib.dump(bundle, MODEL_DIR / "best_model.joblib")

    (REPORT_DIR / "production_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    write_model_card(report, sens, spec, at_pop)
    return report


def write_model_card(report: dict, sens: float, spec: float, at_pop) -> None:
    """A model card — the document a reviewer will ask for."""
    ci = report.get("bootstrap_ci", {})

    def fmt(name: str) -> str:
        v = ci.get(name)
        return (
            f"{v['estimate']:.3f} (95% CI {v['lo']:.3f}–{v['hi']:.3f})"
            if v
            else "n/a"
        )

    subgroup_lines = []
    for col, data in report.get("subgroups", {}).items():
        gap = data.get("sensitivity_gap")
        subgroup_lines.append(
            f"- **{col}** — largest sensitivity gap between groups: "
            + (f"{gap:.3f}" if gap is not None else "not estimable")
        )

    card = f"""# Model card — Q-CHAT-10 traits screening

Generated {report['generated_at']} · version 1.0.0

## Intended use

Flags toddlers whose Q-CHAT-10 responses warrant a referral for a full
developmental assessment. It is a triage aid for parents and primary care, not a
diagnostic instrument. It must not be used to deny, delay or replace an
assessment, and a negative result never rules autism out.

**Out of scope:** children outside roughly 12–36 months, adults, any use as
evidence in an educational or clinical decision, and any use without a clinician
in the loop.

## Model

- Estimator: {report['best_model']}
- Probability calibration: {report.get('calibration_method', 'none')}
- Decision threshold: {report['operating_point']['threshold']:.3f}
  ({report['operating_point'].get('note', '')})
- Training rows: {report['n_train']} · test rows: {report['n_test']}

## Performance (test set, 95% bootstrap CIs)

| Metric | Value |
|---|---|
| Sensitivity | {fmt('Sensitivity')} |
| Specificity | {fmt('Specificity')} |
| PPV (in sample) | {fmt('PPV')} |
| NPV (in sample) | {fmt('NPV')} |
| ROC-AUC | {fmt('ROC-AUC')} |

## The number that matters most

The in-sample PPV is misleading. This dataset is roughly 69% positive because it
was collected from families who already suspected a developmental difference.
Applied at a general-population prevalence near {at_pop['prevalence']:.1%}, the
same sensitivity and specificity yield a PPV of **{at_pop['PPV']:.1%}** and an
NPV of **{at_pop['NPV']:.1%}**.

In practice: screening 1,000 children at that prevalence produces about
{at_pop['false_alarms_per_1000']:.0f} false alarms and misses about
{at_pop['missed_per_1000']:.1f} cases. Sensitivity and specificity are
properties of the test; PPV is a property of who you screen.

## Known limitations

1. **Target leakage.** The label is `Yes` exactly when the Q-CHAT-10 score
   exceeds 3, and that score is the sum of A1–A10. The target is an arithmetic
   function of the features, so near-perfect scores reflect the construction of
   the dataset, not detection ability. Run `--no-behaviour` for the honest
   baseline on background variables alone.
2. **Selection bias.** Participants self-selected through a screening app. They
   are not a random sample of toddlers.
3. **Self-reported labels.** "ASD traits" means "above the screening cut-off",
   not a clinician's diagnosis.
4. **Small and unevenly distributed subgroups.** Several ethnicity groups have
   too few rows to estimate performance, so bias in those groups is unmeasured
   rather than absent.
5. **No temporal or external validation.** Performance on a different
   population, era or collection method is unknown.

## Fairness

{chr(10).join(subgroup_lines) if subgroup_lines else '- No subgroup columns available in this run.'}

Groups below 20 rows are reported as not estimable rather than given a number
that would look precise and mean nothing.

## Ethical considerations

Results are shown to parents, who may be anxious and are not trained to read a
probability. The interface therefore reports the raw Q-CHAT-10 score alongside
the model output, never displays a diagnosis, and routes every outcome —
positive or negative — toward a conversation with a clinician.

Screening data about children is sensitive. Nothing in this project should be
deployed with real user data without ethics approval, informed consent, a lawful
basis for processing, and a retention policy.
"""
    (REPORT_DIR / "MODEL_CARD.md").write_text(card)
    print(f"\nModel card written to {REPORT_DIR / 'MODEL_CARD.md'}")
