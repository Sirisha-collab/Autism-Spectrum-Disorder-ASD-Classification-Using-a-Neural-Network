# Autism Screening — ML Model Comparison + CHAT-10 Questionnaire

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
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Drop your Kaggle CSV into `data/`. 

## 3. Running

```bash
# API + web UI (two terminals)
uvicorn api.main:app --reload --port 8000     # terminal 1
cd web && npm install && npm run dev          # terminal 2
```

**Models compared:** Logistic Regression, Decision Tree, Random Forest,
Gradient Boosting, AdaBoost, SVM (RBF), K-Nearest Neighbours, Naive Bayes.

**Metrics per model:** accuracy, balanced accuracy, precision, recall
(sensitivity), specificity, F1, ROC-AUC, MCC, confusion-matrix counts, and
5-fold cross-validated F1 with standard deviation.
