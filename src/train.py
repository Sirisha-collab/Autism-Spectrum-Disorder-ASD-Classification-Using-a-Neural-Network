"""
Train and compare every model in the registry.

Usage (from the project root):
    python src/train.py
    python src/train.py --csv "data/Toddler Autism dataset July 2018.csv"
    python src/train.py --keep-leakage      # include Qchat-10-Score
    python src/train.py --tune              # GridSearchCV on the main models
"""
from __future__ import annotations

import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from config import (
    CV_FOLDS,
    DEFAULT_CSV,
    MODEL_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    TEST_SIZE,
)
from data_prep import build_preprocessor, dataset_summary, feature_names, load_data, make_xy
from evaluate import (
    compute_metrics,
    metrics_table,
    plot_class_balance,
    plot_confusion_matrices,
    plot_feature_importance,
    plot_model_comparison,
    plot_roc_curves,
    print_classification_reports,
)
from models import PARAM_GRIDS, get_models
from production import evaluate_production

warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    p = argparse.ArgumentParser(description="Train ASD-traits screening models.")
    p.add_argument("--csv", default=str(DEFAULT_CSV), help="path to the dataset CSV")
    p.add_argument(
        "--keep-leakage",
        action="store_true",
        help="keep Qchat-10-Score as a feature (inflates every metric)",
    )
    p.add_argument(
        "--no-behaviour",
        action="store_true",
        help="drop A1..A10 and train on demographics only (the hard task)",
    )
    p.add_argument("--tune", action="store_true", help="run GridSearchCV")
    p.add_argument("--test-size", type=float, default=TEST_SIZE)
    return p.parse_args()


def main():
    args = parse_args()
    drop_leakage = not args.keep_leakage

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    df = load_data(args.csv)
    print(dataset_summary(df))

    X, y = make_xy(df, drop_leakage=drop_leakage, drop_behaviour=args.no_behaviour)
    print(f"\nFeatures used ({X.shape[1]}): {list(X.columns)}")
    print(f"Leakage column Qchat-10-Score: {'EXCLUDED' if drop_leakage else 'INCLUDED'}")
    if args.no_behaviour:
        print("A1..A10 EXCLUDED - demographics-only run (expect realistic, lower scores)")
    plot_class_balance(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE, stratify=y
    )
    df_test_raw = df.loc[X_test.index]
    print(f"Train: {len(X_train)} rows   Test: {len(X_test)} rows")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    results, preds, probas, fitted = {}, {}, {}, {}

    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    for name, estimator in get_models().items():
        pipe = Pipeline(
            [("prep", build_preprocessor(X_train)), ("clf", estimator)]
        )

        if args.tune and name in PARAM_GRIDS:
            search = GridSearchCV(
                pipe, PARAM_GRIDS[name], cv=cv, scoring="f1", n_jobs=-1
            )
            search.fit(X_train, y_train)
            pipe = search.best_estimator_
            best = search.best_params_
        else:
            pipe.fit(X_train, y_train)
            best = None

        y_pred = pipe.predict(X_test)
        y_proba = (
            pipe.predict_proba(X_test)[:, 1]
            if hasattr(pipe.named_steps["clf"], "predict_proba")
            else None
        )

        m = compute_metrics(y_test, y_pred, y_proba)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1)
        m["CV F1 (mean)"] = cv_scores.mean()
        m["CV F1 (std)"] = cv_scores.std()

        results[name] = m
        preds[name] = y_pred
        probas[name] = y_proba
        fitted[name] = pipe

        line = (
            f"{name:<24} acc={m['Accuracy']:.3f}  f1={m['F1']:.3f}  "
            f"recall={m['Recall (Sensitivity)']:.3f}  "
            f"auc={m['ROC-AUC']:.3f}  cv_f1={cv_scores.mean():.3f}±{cv_scores.std():.3f}"
        )
        print(line)
        if best:
            print(f"{'':<24} best params: {best}")

    # ------------------------------------------------------------- reports
    table = metrics_table(results)
    full = table.join(
        pd.DataFrame(results).T[["CV F1 (mean)", "CV F1 (std)"]]
    )

    print("\n" + "=" * 70)
    print("METRICS COMPARISON")
    print("=" * 70)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(full.round(4).to_string())

    reports_text = print_classification_reports(y_test, preds)
    print(reports_text)

    full.round(4).to_csv(REPORT_DIR / "metrics_comparison.csv")
    (REPORT_DIR / "classification_reports.txt").write_text(reports_text)

    plot_confusion_matrices(y_test, preds)
    plot_roc_curves(y_test, probas)
    plot_model_comparison(table)

    best_name = table.index[0]
    best_model = fitted[best_name]

    names = feature_names(best_model.named_steps["prep"])
    for n in ["Random Forest", "Decision Tree", "Logistic Regression"]:
        if n in fitted:
            plot_feature_importance(
                fitted[n], feature_names(fitted[n].named_steps["prep"]), n
            )

    for name, pipe in fitted.items():
        joblib.dump(pipe, MODEL_DIR / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib")

    production = evaluate_production(
        best_name=best_name,
        best_pipe=best_model,
        fitted=fitted,
        results=results,
        preds=preds,
        probas=probas,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        df_test_raw=df_test_raw,
        drop_leakage=drop_leakage,
        drop_behaviour=args.no_behaviour,
    )

    (REPORT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "best_model": best_name,
                "metrics": {k: float(v) for k, v in results[best_name].items()},
                "leakage_column_dropped": drop_leakage,
                "behaviour_items_dropped": args.no_behaviour,
                "n_train": len(X_train),
                "n_test": len(X_test),
            },
            indent=2,
        )
    )

    deployed = production.get("best_model", best_name)
    print("\n" + "=" * 70)
    print(f"DEPLOYED MODEL: {deployed}")
    print(f"  F1 ranking put {best_name} first (F1 = {table.loc[best_name, 'F1']:.4f})")
    print(f"Saved to   : {MODEL_DIR / 'best_model.joblib'}")
    print(f"Plots + CSV: {REPORT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
