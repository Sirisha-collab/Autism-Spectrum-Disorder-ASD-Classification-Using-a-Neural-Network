"""
Interactive Q-CHAT-10 screening: asks the 10 behaviour questions plus a few
background questions, then runs the trained model.

Usage (from the project root):
    python src/predict.py                       # interactive
    python src/predict.py --model models/Random_Forest.joblib
    python src/predict.py --demo                # run a canned example
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from config import MODEL_DIR, QCHAT_CUTOFF
from questionnaire import DEMOGRAPHIC_QUESTIONS, QCHAT10, score_answers

DISCLAIMER = """
------------------------------------------------------------------
This is a SCREENING aid built on the public Q-CHAT-10 toddler
dataset. It does not diagnose autism. Only a qualified clinician
(paediatrician, developmental psychologist, psychiatrist) can make
a diagnosis, using a full assessment. Use the result only as a
prompt to seek professional advice.
------------------------------------------------------------------
"""


def ask_choice(prompt: str, options: list[str]) -> int:
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"   {i}. {opt}")
    while True:
        raw = input("   Your answer [1-%d]: " % len(options)).strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print("   Please enter a number from the list.")


def ask_int(prompt: str, lo: int, hi: int) -> int:
    while True:
        raw = input(f"\n{prompt} [{lo}-{hi}]: ").strip()
        if raw.isdigit() and lo <= int(raw) <= hi:
            return int(raw)
        print("   Please enter a whole number in range.")


def collect_answers() -> tuple[dict, dict]:
    print("=" * 66)
    print("  Q-CHAT-10  —  Autism traits screening questionnaire (toddlers)")
    print("=" * 66)
    print(DISCLAIMER)

    answer_idx = {}
    for feat, question, options, _ in QCHAT10:
        answer_idx[feat] = ask_choice(f"[{feat}] {question}", options)

    demo = {}
    for field, question, kind, options in DEMOGRAPHIC_QUESTIONS:
        if kind == "int":
            demo[field] = ask_int(question, 6, 60)
        else:
            demo[field] = options[ask_choice(question, options)]

    return answer_idx, demo


DEMO_ANSWERS = (
    {"A1": 0, "A2": 1, "A3": 1, "A4": 3, "A5": 4,
     "A6": 0, "A7": 1, "A8": 2, "A9": 0, "A10": 3},
    {
        "Age_Mons": 28,
        "Sex": "m",
        "Ethnicity": "White European",
        "Jaundice": "No",
        "Family_mem_with_ASD": "No",
        "Who completed the test": "family member",
    },
)


def build_row(answer_idx: dict, demo: dict, feature_columns: list[str]) -> tuple[pd.DataFrame, dict]:
    scores = score_answers(answer_idx)
    row = {f"A{i}": scores[f"A{i}"] for i in range(1, 11)}
    row.update(demo)
    row["Qchat-10-Score"] = scores["total"]

    # Keep only the columns the model was trained on, in the right order.
    frame = pd.DataFrame([{c: row.get(c) for c in feature_columns}])
    return frame, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODEL_DIR / "best_model.joblib"))
    ap.add_argument("--demo", action="store_true", help="use a built-in example")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(
            f"Model not found: {model_path}\nRun 'python src/train.py' first."
        )

    bundle = joblib.load(model_path)
    if isinstance(bundle, dict):
        model = bundle["model"]
        model_name = bundle.get("model_name", "model")
        feature_columns = bundle["feature_columns"]
    else:  # a bare pipeline saved per-model
        model = bundle
        model_name = model_path.stem.replace("_", " ")
        feature_columns = list(model.named_steps["prep"].feature_names_in_)

    if args.demo:
        answer_idx, demo = DEMO_ANSWERS
        print("Running the built-in demo case...\n")
    else:
        answer_idx, demo = collect_answers()

    frame, scores = build_row(answer_idx, demo, feature_columns)

    pred = int(model.predict(frame)[0])
    proba = None
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba = float(model.predict_proba(frame)[0][1])

    print("\n" + "=" * 66)
    print("  RESULT")
    print("=" * 66)
    print("Item scores:", " ".join(f"A{i}={scores[f'A{i}']}" for i in range(1, 11)))
    print(f"Q-CHAT-10 total score : {scores['total']} / 10")
    print(f"Clinical cut-off      : {QCHAT_CUTOFF} (score above this -> refer)")
    print(f"Model used            : {model_name}")
    print(f"Model prediction      : {'ASD traits likely' if pred else 'ASD traits unlikely'}")
    if proba is not None:
        print(f"Model confidence      : {proba:.1%}")

    print()
    if scores["total"] > QCHAT_CUTOFF or pred == 1:
        print("The responses suggest traits that warrant a professional")
        print("developmental assessment. Please talk to your paediatrician.")
    else:
        print("The responses do not indicate elevated ASD traits on this")
        print("screening tool. If you still have concerns about your child's")
        print("development, raise them with your paediatrician anyway —")
        print("a screening questionnaire can miss things.")
    print(DISCLAIMER)


if __name__ == "__main__":
    main()
