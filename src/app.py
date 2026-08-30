"""
Optional web UI for the same questionnaire.

Run from the project root:
    streamlit run src/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import pandas as pd
import streamlit as st

from config import MODEL_DIR, QCHAT_CUTOFF
from questionnaire import DEMOGRAPHIC_QUESTIONS, QCHAT10, score_answers

st.set_page_config(page_title="Q-CHAT-10 Screening", page_icon="🧩", layout="centered")


@st.cache_resource
def load_model(path):
    bundle = joblib.load(path)
    if isinstance(bundle, dict):
        return bundle["model"], bundle.get("model_name", "model"), bundle["feature_columns"]
    return bundle, Path(path).stem, list(bundle.named_steps["prep"].feature_names_in_)


st.title("Autism Traits Screening (Q-CHAT-10)")
st.info(
    "This is a screening aid, not a diagnosis. Only a qualified clinician can "
    "diagnose autism after a full assessment. Use this result only as a prompt "
    "to seek professional advice."
)

available = sorted(MODEL_DIR.glob("*.joblib"))
if not available:
    st.error("No trained model found. Run `python src/train.py` first.")
    st.stop()

choice = st.sidebar.selectbox(
    "Model", available, format_func=lambda p: p.stem.replace("_", " ")
)
model, model_name, feature_columns = load_model(choice)

st.subheader("Behaviour questions")
answers = {}
for feat, question, options, _ in QCHAT10:
    answers[feat] = options.index(
        st.radio(f"**{feat}.** {question}", options, key=feat, horizontal=False)
    )

st.subheader("Background")
demo = {}
cols = st.columns(2)
for i, (field, question, kind, options) in enumerate(DEMOGRAPHIC_QUESTIONS):
    with cols[i % 2]:
        if kind == "int":
            demo[field] = st.number_input(question, 6, 60, 24, key=field)
        else:
            demo[field] = st.selectbox(question, options, key=field)

if st.button("Run screening", type="primary", use_container_width=True):
    scores = score_answers(answers)
    row = {f"A{i}": scores[f"A{i}"] for i in range(1, 11)}
    row.update(demo)
    row["Qchat-10-Score"] = scores["total"]
    frame = pd.DataFrame([{c: row.get(c) for c in feature_columns}])

    pred = int(model.predict(frame)[0])
    proba = (
        float(model.predict_proba(frame)[0][1])
        if hasattr(model.named_steps["clf"], "predict_proba")
        else None
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Q-CHAT-10 score", f"{scores['total']} / 10")
    c2.metric("Cut-off", QCHAT_CUTOFF)
    c3.metric("Confidence", f"{proba:.0%}" if proba is not None else "—")

    if pred == 1 or scores["total"] > QCHAT_CUTOFF:
        st.warning(
            f"**{model_name}** flags elevated ASD traits. Please arrange a "
            "developmental assessment with a paediatrician."
        )
    else:
        st.success(
            f"**{model_name}** does not flag elevated ASD traits. If you still "
            "have concerns about development, raise them with a paediatrician "
            "anyway — screening tools can miss things."
        )

    with st.expander("Item-level scores"):
        st.dataframe(
            pd.DataFrame(
                {"Item": [f"A{i}" for i in range(1, 11)],
                 "Score": [scores[f"A{i}"] for i in range(1, 11)]}
            ),
            hide_index=True,
        )
