# Autism Traits Screening — ML Model Comparison + Q-CHAT-10 Questionnaire

A complete, VS Code–ready Python project built on the Toddler Autism (Q-CHAT-10)
dataset. It trains and compares eight classifiers, produces a full metrics
report with plots, and ships an interactive questionnaire that runs the trained
model on a new child's answers.

> **This is a screening aid, not a diagnostic tool.** The label in this dataset is
> "ASD traits present", meaning the child scored above the Q-CHAT-10 referral
> cut-off — it is not a clinical autism diagnosis. Only a qualified clinician can
> diagnose autism, after a full assessment. Keep the disclaimer in the UI.

---

## 1. Project layout

```
autism-screening-ml/
├── data/
│   ├── Toddler Autism dataset July 2018.csv   <- put YOUR Kaggle file here
│   └── sample_synthetic.csv                   <- fake data, so it runs out of the box
├── models/                                    <- trained pipelines (.joblib)
├── reports/                                   <- metrics CSV, plots, text reports
├── src/                                       ML layer (Python)
│   ├── config.py            paths, column names, constants
│   ├── data_prep.py         loading, cleaning, preprocessing pipeline
│   ├── models.py            the 8 classifiers + hyperparameter grids
│   ├── evaluate.py          metrics + all plots
│   ├── train.py             main training / comparison script
│   ├── questionnaire.py     the 10 Q-CHAT items and their scoring rules
│   ├── predict.py           interactive CLI questionnaire -> prediction
│   └── make_sample_data.py  generates the synthetic CSV
├── api/
│   └── main.py                                FastAPI service
├── web/                                       Next.js 15 + React 19 frontend
│   ├── app/
│   │   ├── layout.tsx        shell, fonts, disclaimer
│   │   ├── page.tsx          the questionnaire
│   │   ├── metrics/page.tsx  model comparison table
│   │   └── globals.css       design tokens
│   ├── components/
│   │   ├── ItemLedger.tsx    the sticky A1-A10 score strip
│   │   ├── QuestionCard.tsx
│   │   └── ResultPanel.tsx
│   └── lib/                  typed API client
├── .vscode/                 launch configs (F5 to run)
└── requirements.txt
```

## 2. Setup in VS Code

```bash
# open the folder in VS Code, then in the integrated terminal:
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Then `Ctrl+Shift+P` → **Python: Select Interpreter** → pick `.venv`.

Drop your Kaggle CSV into `data/`. If your filename differs from
`Toddler Autism dataset July 2018.csv`, either rename it or pass `--csv`.

## 3. Running

```bash
# train and compare all models
python src/train.py

# point at a specific file
python src/train.py --csv "data/Toddler Autism dataset July 2018.csv"

# with hyperparameter tuning (GridSearchCV, slower)
python src/train.py --tune

# the honest, non-trivial version — see section 5
python src/train.py --no-behaviour

# interactive questionnaire
python src/predict.py
python src/predict.py --demo
python src/predict.py --model models/Random_Forest.joblib

# API + web UI (two terminals)
uvicorn api.main:app --reload --port 8000     # terminal 1
cd web && npm install && npm run dev          # terminal 2
```

Then open http://localhost:3000 for the app, or http://localhost:8000/docs for
the auto-generated API docs.

Or just press **F5** in VS Code and pick a configuration.

## 4. What you get

### Screening metrics, not just accuracy

Accuracy is close to useless for a screen. The pipeline reports what a reviewer
will actually ask about:

| What | Where | Why it's there |
|---|---|---|
| 95% bootstrap CIs on sensitivity, specificity, PPV, NPV, AUC | `production_report.json` | a point estimate from ~200 test rows isn't a result |
| Calibration: Brier score, ECE, reliability diagram | `calibration.png` | the UI shows a confidence % to a parent; it has to mean something |
| Isotonic/Platt recalibration of the deployed model | in the saved bundle | raw tree and SVM probabilities are badly calibrated |
| Threshold sweep + chosen operating point | `threshold_sweep.png/.csv` | 0.5 is a library default, not a clinical choice |
| **PPV/NPV at real prevalence** | `ppv_vs_prevalence.png` | the single most important table in the project |
| Precision-recall curves | `precision_recall_curves.png` | more honest than ROC under class skew |
| Sensitivity by sex and ethnicity | `subgroup_*.csv` | a model can look strong overall and miss cases in one group |
| McNemar tests between models | `production_report.json` | "0.97 beats 0.96" needs a significance test on paired predictions |
| Repeated stratified CV | `metrics_advanced.py` | one 5-fold run has enough variance to swap the rankings |
| `MODEL_CARD.md` | `reports/` | intended use, limitations, fairness, ethics |

**The prevalence table is the one to put on a slide.** Sensitivity and
specificity are properties of the test; PPV is a property of the population you
screen. This dataset is ~69% positive because families self-selected into it.
Run the same model at general-population prevalence (~1.7%) and the PPV
collapses — that is normal for screening instruments, and it is why the result
routes the parent to an assessment instead of stating a conclusion.



**Models compared:** Logistic Regression, Decision Tree, Random Forest,
Gradient Boosting, AdaBoost, SVM (RBF), K-Nearest Neighbours, Naive Bayes.

**Metrics per model:** accuracy, balanced accuracy, precision, recall
(sensitivity), specificity, F1, ROC-AUC, MCC, confusion-matrix counts, and
5-fold cross-validated F1 with standard deviation.

**Artifacts written to `reports/`:**

| File | Contents |
|---|---|
| `metrics_comparison.csv` | the full metrics table, sorted by F1 |
| `classification_reports.txt` | per-class precision/recall/F1 for every model |
| `model_comparison.png` | grouped bar chart of the headline metrics |
| `confusion_matrices.png` | one matrix per model |
| `roc_curves.png` | overlaid ROC curves with AUCs |
| `feature_importance_*.png` | top-20 features for RF / DT / LogReg |
| `class_balance.png` | target distribution |
| `summary.json` | best model + its metrics |

The best model (by F1) is saved to `models/best_model.joblib`; every individual
model is saved too, so the questionnaire can use any of them.

## 5. The leakage problem — read this before you present the project

In this dataset:

- `Qchat-10-Score` is literally `A1 + A2 + ... + A10`.
- `Class/ASD Traits` is literally `Yes` if `Qchat-10-Score > 3`, else `No`.

So the target is a **deterministic arithmetic function of the input features**.
Any model that sees `Qchat-10-Score` scores 100% on everything, and most models
that see `A1..A10` get very close to 100% too, because they only need to learn a
threshold on a sum. Those numbers are real, but they measure "did the model
learn addition", not "can the model detect autism traits".

This project handles that in three ways:

1. `Qchat-10-Score` is **dropped by default**. Use `--keep-leakage` to see the
   inflated version for comparison.
2. `train.py` prints a leakage check showing what fraction of rows satisfy
   `Qchat-10-Score == sum(A1..A10)`.
3. `--no-behaviour` drops `A1..A10` as well and trains on age, sex, ethnicity,
   jaundice, family history and respondent only. Accuracy falls to roughly the
   majority-class baseline with ROC-AUC near 0.5 — which is the correct,
   informative result: the demographics in this dataset carry almost no signal
   about the outcome on their own.

A good write-up presents all three runs and explains the gap. That is a much
stronger project than reporting 100% accuracy without comment.

### Which model wins, and why

`src/model_selection.py` picks the deployed model on screening criteria rather
than "sort by F1, take row one". The rule, in order: a hard 90% sensitivity
floor (a screen that misses cases fails at its only job), then Youden's J, then
cross-validation stability, then calibration, then McNemar's test against the
runners-up — and if the difference isn't significant, the simpler model wins.

The reasoning is generated from the numbers, saved into
`production_report.json`, printed in the terminal, and rendered on `/metrics`
by `components/ModelVerdict.tsx`. When F1's leader and the selection disagree,
the UI says so explicitly and explains why F1 is the wrong criterion here.

### Explaining a single prediction

`src/explain.py` computes SHAP values for each result and the panel renders them
as diverging bars. Answers pushing toward a referral go right, answers pushing
away go left.

Three details that matter:

- **Explainer picked per estimator.** `TreeExplainer` for forests and boosting,
  `LinearExplainer` for logistic regression, `KernelExplainer` (on a k-means
  summarised background) for SVM, KNN and Naive Bayes.
- **One-hot columns are summed back to their source.** `Ethnicity` appears once
  rather than as nine near-zero rows, and `bin__A4` is shown as "Points to share
  interest".
- **The background sample ships inside the `.joblib`.** The API never needs the
  training data at request time.

Bars scale to the largest contribution in that prediction, not to an absolute
unit, because SHAP magnitudes are in the model's internal units — log-odds for
logistic regression, probability for trees. Ranking and direction are the
meaningful parts.

If `shap` isn't installed the API returns `contributions: null` and the panel
hides the section. The screening still works.

## 4a. Tests

```bash
pip install pytest httpx
pytest
```

36 tests covering the Q-CHAT scoring rules item by item (including the reversed
scoring on item 10), consent enforcement, request validation, the age
eligibility gate, the prediction contract, and the SHAP feature-name mapping.

## 5a. The web app

**Backend** — FastAPI (`api/main.py`):

| Endpoint | Purpose |
|---|---|
| `GET /api/health` `/api/ready` | liveness and readiness, for a container orchestrator |
| `GET /api/questions` | items, options, scoring indices, background questions, age range |
| `GET /api/models` | every trained `.joblib`, and which one won |
| `GET /api/metrics` | comparison table plus the full production report |
| `GET /api/model-card` | the generated model card |
| `POST /api/predict` | validates, scores, runs the calibrated model, audits |

Production concerns handled: strict Pydantic validation with `extra="forbid"`,
required consent flag, per-item option-range checks, request IDs on every
response, structured logging with latency, in-memory rate limiting, an
append-only audit log (`logs/screenings.jsonl`, scores and outcome only — no
identifiers), and in-memory model caching.

**Frontend** — Next.js 15 App Router, React 19, TypeScript, Tailwind v4.

Interface details worth knowing about:

- **Live service status** in the header, polling `/api/health` every 30s. The
  app tells you the backend is down instead of letting a failed submit be the
  discovery.
- **Dark mode** via `prefers-color-scheme`, because this gets filled in at 11pm.
- **Skeleton loaders** shaped like the content they replace, so the layout
  doesn't jump.
- **One orchestrated motion moment.** The score counts up and the SHAP bars
  stagger in when the result lands, completing the arc the ledger has been
  building across ten questions. Everything is suppressed under
  `prefers-reduced-motion`.
- **The PPV curve is rendered as inline SVG** on `/metrics`, so the project's
  central finding is a picture rather than a row in a table.

The questionnaire is built for the person actually using it: a consent screen
that states plainly what the tool can't tell you, draft auto-save so an
interrupted parent doesn't lose ten answers, arrow-key navigable radiogroups
with proper ARIA, a live progress bar, an age eligibility gate that says when
the score isn't interpretable, and a print stylesheet so the result sheet can be
taken to an appointment. Errors name what went wrong and how to fix it.

Items are fetched from the API rather than hardcoded, so editing
`src/questionnaire.py` updates the UI with no frontend change.

`next.config.ts` proxies `/api/*` to the FastAPI port, so there are no CORS
issues in development and no API URL baked into the client. To point at a
deployed backend, set `NEXT_PUBLIC_API_URL` in `web/.env.local`.

The sticky strip at the top is the interface's one deliberate flourish: ten
cells, one per item, filling in as you answer, with the running total beside
them. It makes the scoring rule visible — and makes the leakage in section 5
something you can watch happen rather than something buried in a footnote.

## 6. The questionnaire

`src/questionnaire.py` holds the ten Q-CHAT-10 items with their real response
options and the official scoring rule: for items 1–9 the three least typical
responses score 1; for item 10 the three most frequent responses score 1. A
total above 3 is the standard referral cut-off.

`predict.py` (CLI) and the web app both consume it. The CLI asks the ten questions plus six background questions, converts the
answers into the same `A1..A10` feature format the model was trained on, and
reports both the raw Q-CHAT-10 score and the model's prediction with its
confidence. Because of the leakage described above, the two will usually agree —
that agreement is expected, not a sign of a strong model.

## 7. Extending it

- Add a model: one entry in `get_models()` in `src/models.py`. Everything else
  updates automatically.
- Add a metric: one line in `compute_metrics()` in `src/evaluate.py`.
- Handle class imbalance: pass `class_weight="balanced"` to the tree/linear
  models, or add SMOTE via `imbalanced-learn`.
- Explainability: SHAP values on the Random Forest pipeline make a strong
  addition for a report or viva. Return them from `/api/predict` and render a
  per-item contribution bar in `ResultPanel.tsx`.
- Persistence: add SQLite or Postgres behind the API to store completed
  screenings, then build a `/history` route.
- Deployment: the FastAPI service containerises cleanly; the frontend deploys to
  Vercel with `NEXT_PUBLIC_API_URL` pointed at it.
