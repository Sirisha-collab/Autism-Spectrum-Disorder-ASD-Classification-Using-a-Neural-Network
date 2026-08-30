"""
Loading, cleaning and preprocessing for the Toddler Autism (Q-CHAT-10) dataset.
"""
from __future__ import annotations

import re
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    BEHAVIOUR_COLS,
    CATEGORICAL_COLS,
    ID_COL,
    LEAKY_COLS,
    NUMERIC_COLS,
    SCORE_COL,
    TARGET_COL,
)

# Kaggle exports vary slightly in spelling/spacing. Map whatever we find onto
# the canonical names used everywhere else in the project.
_CANONICAL = {
    "case_no": ID_COL,
    "caseno": ID_COL,
    "age_mons": "Age_Mons",
    "agemons": "Age_Mons",
    "age": "Age_Mons",
    "qchat10score": SCORE_COL,
    "qchat_10_score": SCORE_COL,
    "sex": "Sex",
    "gender": "Sex",
    "ethnicity": "Ethnicity",
    "jaundice": "Jaundice",
    "jundice": "Jaundice",
    "family_mem_with_asd": "Family_mem_with_ASD",
    "familymemwithasd": "Family_mem_with_ASD",
    "austim": "Family_mem_with_ASD",
    "whocompletedthetest": "Who completed the test",
    "who_completed_the_test": "Who completed the test",
    "classasdtraits": TARGET_COL,
    "class_asd_traits": TARGET_COL,
    "class": TARGET_COL,
    "classasd": TARGET_COL,
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the canonical schema, tolerant of Kaggle variants."""
    rename = {}
    for col in df.columns:
        key = _norm(col)
        if re.fullmatch(r"a([1-9]|10)", key):
            rename[col] = f"A{key[1:]}"
        elif key in {_norm(k): _norm(k) for k in _CANONICAL}:
            pass
    # second pass against the lookup table using normalised keys
    lookup = {_norm(k): v for k, v in _CANONICAL.items()}
    for col in df.columns:
        key = _norm(col)
        if col in rename:
            continue
        if key in lookup:
            rename[col] = lookup[key]
    return df.rename(columns=rename)


def load_data(csv_path) -> pd.DataFrame:
    """Read the CSV and apply basic cleaning."""
    df = pd.read_csv(csv_path)
    df = standardise_columns(df)

    missing = [c for c in BEHAVIOUR_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns missing from {csv_path}: {missing}\n"
            f"Found instead: {list(df.columns)}"
        )

    # Trim whitespace and normalise casing on every text column.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Yes/No -> consistent casing
    for col in ["Jaundice", "Family_mem_with_ASD", TARGET_COL]:
        if col in df.columns:
            df[col] = df[col].str.capitalize()

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].str.lower().map({"m": "m", "f": "f"}).fillna(df["Sex"])

    # '?' is used for unknown ethnicity in some versions
    df = df.replace({"?": pd.NA, "nan": pd.NA, "": pd.NA})

    df = df.drop_duplicates()
    return df


def make_xy(df: pd.DataFrame, drop_leakage: bool = True, drop_behaviour: bool = False):
    """
    Split the dataframe into a feature matrix X and binary target y.

    drop_leakage=True removes Qchat-10-Score, which is the arithmetic sum of
    A1..A10 and therefore trivially determines the label.

    drop_behaviour=True additionally removes A1..A10, leaving only demographics.
    This is the genuinely hard version of the task and gives realistic scores.
    """
    y = (df[TARGET_COL].astype(str).str.lower().str[0] == "y").astype(int)

    drop = [TARGET_COL]
    if ID_COL in df.columns:
        drop.append(ID_COL)
    if drop_leakage:
        drop += [c for c in LEAKY_COLS if c in df.columns]
    if drop_behaviour:
        drop += [c for c in BEHAVIOUR_COLS if c in df.columns]

    X = df.drop(columns=drop)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """One-hot encode categoricals, scale numerics, pass A1..A10 through."""
    numeric = [c for c in NUMERIC_COLS if c in X.columns]
    categorical = [c for c in CATEGORICAL_COLS if c in X.columns]
    binary = [c for c in BEHAVIOUR_COLS if c in X.columns]
    if SCORE_COL in X.columns:
        numeric = numeric + [SCORE_COL]

    try:  # scikit-learn >= 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older versions
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", ohe),
                    ]
                ),
                categorical,
            ),
            (
                "bin",
                SimpleImputer(strategy="most_frequent"),
                binary,
            ),
        ],
        remainder="drop",
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Readable feature names after fitting the ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"f{i}" for i in range(preprocessor.transform_shape_[1])]


def dataset_summary(df: pd.DataFrame) -> str:
    """Short text report used by train.py and the README."""
    lines = [
        f"Rows: {len(df)}   Columns: {df.shape[1]}",
        f"Target distribution:\n{df[TARGET_COL].value_counts().to_string()}",
        f"Missing values per column:\n"
        f"{df.isna().sum()[lambda s: s > 0].to_string() or '  (none)'}",
    ]
    if SCORE_COL in df.columns:
        recomputed = df[BEHAVIOUR_COLS].sum(axis=1)
        match = (recomputed == df[SCORE_COL]).mean()
        lines.append(
            f"Leakage check: Qchat-10-Score equals sum(A1..A10) for "
            f"{match:.1%} of rows."
        )
    return "\n".join(lines)
