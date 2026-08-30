"""
The classifiers compared in this project. Add or remove entries here and the
training script, metrics tables and plots all update automatically.
"""
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import RANDOM_STATE


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(probability=True, random_state=RANDOM_STATE),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
    }


# Small grids for the tuning step. Keys are prefixed with 'clf__' because the
# estimator sits inside a Pipeline.
PARAM_GRIDS = {
    "Logistic Regression": {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l2"],
    },
    "Decision Tree": {
        "clf__max_depth": [3, 5, 8, None],
        "clf__min_samples_leaf": [1, 3, 5, 10],
        "clf__criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [6, 10, None],
        "clf__min_samples_leaf": [1, 2, 4],
    },
    "SVM (RBF)": {
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", 0.1],
    },
    "K-Nearest Neighbours": {
        "clf__n_neighbors": [3, 5, 7, 11],
        "clf__weights": ["uniform", "distance"],
    },
}
