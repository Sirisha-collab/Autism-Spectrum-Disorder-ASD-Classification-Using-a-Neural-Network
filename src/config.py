"""
Central configuration for the autism screening ML project.
Edit paths / settings here instead of hunting through the other files.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

# Put your Kaggle CSV in the data/ folder. If the filename differs,
# change it here (or pass --csv on the command line).
DEFAULT_CSV = DATA_DIR / "Toddler Autism dataset July 2018.csv"

for _d in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- columns
BEHAVIOUR_COLS = [f"A{i}" for i in range(1, 11)]
TARGET_COL = "Class/ASD Traits"
SCORE_COL = "Qchat-10-Score"
ID_COL = "Case_No"

NUMERIC_COLS = ["Age_Mons"]
CATEGORICAL_COLS = [
    "Sex",
    "Ethnicity",
    "Jaundice",
    "Family_mem_with_ASD",
    "Who completed the test",
]

# Columns that leak the answer. Qchat-10-Score IS the sum of A1..A10, and the
# label is a threshold on that sum -> keeping it makes every model look perfect.
LEAKY_COLS = [SCORE_COL]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# The Q-CHAT-10 clinical referral cut-off used by the original screening tool.
QCHAT_CUTOFF = 3
