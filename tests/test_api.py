"""API contract tests. Run after 'python src/train.py'."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_DEMOGRAPHICS = {
    "Age_Mons": 22,
    "Sex": "m",
    "Ethnicity": "asian",
    "Jaundice": "No",
    "Family_mem_with_ASD": "No",
    "Who completed the test": "family member",
}
LOW_TRAIT = {f"A{i}": 0 for i in range(1, 10)} | {"A10": 4}
HIGH_TRAIT = {f"A{i}": 4 for i in range(1, 10)} | {"A10": 0}


def payload(answers, **overrides):
    body = {
        "answers": answers,
        "demographics": VALID_DEMOGRAPHICS,
        "consent": True,
    }
    body.update(overrides)
    return body


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_questions_shape():
    r = client.get("/api/questions")
    assert r.status_code == 200
    body = r.json()
    assert len(body["items"]) == 10
    assert body["cutoff"] == 3
    assert all(len(item["options"]) == 5 for item in body["items"])


def test_request_id_header():
    assert "X-Request-ID" in client.get("/api/health").headers


def test_consent_is_required():
    r = client.post("/api/predict", json=payload(LOW_TRAIT, consent=False))
    assert r.status_code == 422


def test_incomplete_answers_rejected():
    r = client.post("/api/predict", json=payload({"A1": 0}))
    assert r.status_code == 422


def test_out_of_range_option_rejected():
    r = client.post("/api/predict", json=payload(LOW_TRAIT | {"A1": 99}))
    assert r.status_code == 422


def test_unknown_demographic_field_rejected():
    body = payload(LOW_TRAIT)
    body["demographics"] = VALID_DEMOGRAPHICS | {"injected": "x"}
    assert client.post("/api/predict", json=body).status_code == 422


@pytest.mark.skipif(
    not (ROOT / "models" / "best_model.joblib").exists(),
    reason="no trained model; run 'python src/train.py'",
)
class TestPredictions:
    def test_low_trait_scores_zero(self):
        r = client.post("/api/predict", json=payload(LOW_TRAIT))
        assert r.status_code == 200
        body = r.json()
        assert body["qchat_score"] == 0
        assert body["above_cutoff"] is False
        assert body["screening_id"]

    def test_high_trait_scores_ten(self):
        r = client.post("/api/predict", json=payload(HIGH_TRAIT))
        body = r.json()
        assert body["qchat_score"] == 10
        assert body["above_cutoff"] is True
        assert body["prediction"] == 1

    def test_probability_within_bounds(self):
        body = client.post("/api/predict", json=payload(HIGH_TRAIT)).json()
        if body["probability"] is not None:
            assert 0.0 <= body["probability"] <= 1.0

    def test_age_outside_validation_window_is_flagged(self):
        body = payload(LOW_TRAIT)
        body["demographics"] = VALID_DEMOGRAPHICS | {"Age_Mons": 34}
        r = client.post("/api/predict", json=body)
        assert r.json()["eligibility"]["status"] == "outside_validation_window"

    def test_age_out_of_range_is_flagged(self):
        body = payload(LOW_TRAIT)
        body["demographics"] = VALID_DEMOGRAPHICS | {"Age_Mons": 60}
        r = client.post("/api/predict", json=body)
        assert r.json()["eligibility"]["status"] == "out_of_range"

    def test_interpretation_is_present(self):
        body = client.post("/api/predict", json=payload(HIGH_TRAIT)).json()
        assert len(body["interpretation"]) > 40
