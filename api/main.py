"""
FastAPI service for the Q-CHAT-10 screening tool.

    uvicorn api.main:app --reload --port 8000

Interactive docs at http://localhost:8000/docs
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from config import MODEL_DIR, QCHAT_CUTOFF, REPORT_DIR
from questionnaire import DEMOGRAPHIC_QUESTIONS, QCHAT10, score_answers

# --------------------------------------------------------------- logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("screening-api")

AUDIT_LOG = ROOT / "logs" / "screenings.jsonl"
AUDIT_LOG.parent.mkdir(exist_ok=True)

ITEM_IDS = [feat for feat, *_ in QCHAT10]
OPTION_COUNT = {feat: len(options) for feat, _q, options, _s in QCHAT10}

# Q-CHAT-10 was validated on toddlers. Outside this window the score is not
# interpretable, so the API says so rather than returning a confident number.
MIN_AGE_MONTHS, MAX_AGE_MONTHS = 12, 36
VALIDATED_AGE_RANGE = (18, 24)

RATE_LIMIT_REQUESTS = 30
RATE_LIMIT_WINDOW_S = 60
_hits: dict[str, deque] = defaultdict(deque)

app = FastAPI(
    title="Q-CHAT-10 Screening API",
    version="1.0.0",
    description=(
        "Serves the Q-CHAT-10 questionnaire and runs calibrated scikit-learn "
        "models. Screening aid only — not a diagnostic tool."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_CACHE: dict[str, Any] = {}


# --------------------------------------------------------------- middleware
@app.middleware("http")
async def observability(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    started = time.perf_counter()

    if request.method == "POST":
        client = request.client.host if request.client else "unknown"
        now = time.time()
        window = _hits[client]
        while window and now - window[0] > RATE_LIMIT_WINDOW_S:
            window.popleft()
        if len(window) >= RATE_LIMIT_REQUESTS:
            log.warning("[%s] rate limit hit for %s", request_id, client)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Too many requests. Try again in a minute."},
            )
        window.append(now)

    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    log.info(
        "[%s] %s %s -> %s (%.1fms)",
        request_id, request.method, request.url.path,
        response.status_code, elapsed_ms,
    )
    return response


# --------------------------------------------------------------- schemas
class Demographics(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    age_months: Annotated[int, Field(ge=1, le=120, alias="Age_Mons")]
    sex: Annotated[str, Field(alias="Sex")]
    ethnicity: Annotated[str, Field(alias="Ethnicity")]
    jaundice: Annotated[Literal["Yes", "No"], Field(alias="Jaundice")]
    family_asd: Annotated[Literal["Yes", "No"], Field(alias="Family_mem_with_ASD")]
    completed_by: Annotated[str, Field(alias="Who completed the test")]

    def to_features(self) -> dict[str, Any]:
        return {
            "Age_Mons": self.age_months,
            "Sex": self.sex,
            "Ethnicity": self.ethnicity,
            "Jaundice": self.jaundice,
            "Family_mem_with_ASD": self.family_asd,
            "Who completed the test": self.completed_by,
        }


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: dict[str, int] = Field(
        ..., description="Chosen option index per item, e.g. {'A1': 0, ...}"
    )
    demographics: Demographics
    model: str | None = Field(None, description="Model stem; defaults to best.")
    consent: bool = Field(
        ...,
        description="Must be true. Confirms the user was shown the disclaimer.",
    )

    @field_validator("answers")
    @classmethod
    def validate_answers(cls, value: dict[str, int]) -> dict[str, int]:
        missing = set(ITEM_IDS) - set(value)
        if missing:
            raise ValueError(f"missing answers for {sorted(missing)}")
        unknown = set(value) - set(ITEM_IDS)
        if unknown:
            raise ValueError(f"unknown items {sorted(unknown)}")
        for item, chosen in value.items():
            if not 0 <= chosen < OPTION_COUNT[item]:
                raise ValueError(
                    f"{item}: option {chosen} out of range "
                    f"(0-{OPTION_COUNT[item] - 1})"
                )
        return value

    @field_validator("consent")
    @classmethod
    def require_consent(cls, value: bool) -> bool:
        if not value:
            raise ValueError("consent is required before scoring")
        return value


class Interval(BaseModel):
    estimate: float
    lo: float
    hi: float


class PredictResponse(BaseModel):
    screening_id: str
    completed_at: str
    prediction: int
    label: str
    probability: float | None
    threshold: float
    qchat_score: int
    cutoff: int
    above_cutoff: bool
    item_scores: dict[str, int]
    model_name: str
    model_version: str
    calibration: str
    eligibility: dict[str, Any]
    performance: dict[str, Any]
    interpretation: str


# --------------------------------------------------------------- helpers
def _model_files() -> list[Path]:
    return sorted(p for p in MODEL_DIR.glob("*.joblib") if p.stem != "best_model")


def _load(stem: str | None) -> dict[str, Any]:
    key = stem or "__best__"
    if key in _CACHE:
        return _CACHE[key]

    path = MODEL_DIR / f"{stem}.joblib" if stem else MODEL_DIR / "best_model.joblib"
    if not path.exists():
        if stem:
            raise HTTPException(404, f"No model named {stem}.")
        raise HTTPException(
            503, "No trained model available. Run 'python src/train.py' first."
        )

    bundle = joblib.load(path)
    if isinstance(bundle, dict):
        loaded = {
            "model": bundle["model"],
            "name": bundle.get("model_name", path.stem),
            "features": bundle["feature_columns"],
            "threshold": float(bundle.get("threshold", 0.5)),
            "calibration": bundle.get("calibration_method", "none"),
            "version": bundle.get("version", "unknown"),
            "metrics": bundle.get("metrics", {}),
        }
    else:
        loaded = {
            "model": bundle,
            "name": path.stem.replace("_", " "),
            "features": list(bundle.named_steps["prep"].feature_names_in_),
            "threshold": 0.5,
            "calibration": "none",
            "version": "unversioned",
            "metrics": {},
        }

    _CACHE[key] = loaded
    log.info("loaded model %s (threshold %.3f)", loaded["name"], loaded["threshold"])
    return loaded


def _eligibility(age_months: int) -> dict[str, Any]:
    lo, hi = VALIDATED_AGE_RANGE
    if lo <= age_months <= hi:
        return {"status": "validated", "message": ""}
    if MIN_AGE_MONTHS <= age_months <= MAX_AGE_MONTHS:
        return {
            "status": "outside_validation_window",
            "message": (
                f"Q-CHAT-10 was validated on children {lo}\u2013{hi} months old. "
                f"At {age_months} months the score is still informative but less "
                "reliable, so weigh it against your own observations."
            ),
        }
    return {
        "status": "out_of_range",
        "message": (
            f"This tool covers children {MIN_AGE_MONTHS}\u2013{MAX_AGE_MONTHS} "
            f"months. At {age_months} months the result is not interpretable — "
            "speak to a clinician instead of relying on this score."
        ),
    }


def _interpretation(flagged: bool, eligibility: dict) -> str:
    if eligibility["status"] == "out_of_range":
        return (
            "This screen does not apply at this age. Take any developmental "
            "concerns directly to your paediatrician."
        )
    if flagged:
        return (
            "The responses meet the referral threshold. Book a developmental "
            "assessment with your paediatrician and bring these answers. Most "
            "children who screen positive turn out not to be autistic — an "
            "assessment either opens access to support or settles the question."
        )
    return (
        "The responses do not meet the referral threshold. If you have ongoing "
        "concerns about your child's development, raise them with your "
        "paediatrician anyway. A ten-item screen can miss things, and what you "
        "notice day to day carries more weight than this score."
    )


def _audit(record: dict) -> None:
    """Append-only log. No free text, no identifiers — scores and outcome only."""
    try:
        with AUDIT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        log.warning("could not write audit record")


# --------------------------------------------------------------- routes
@app.get("/api/health", tags=["ops"])
def health():
    return {
        "status": "ok",
        "models_available": len(_model_files()),
        "best_model_ready": (MODEL_DIR / "best_model.joblib").exists(),
        "metrics_ready": (REPORT_DIR / "metrics_comparison.csv").exists(),
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@app.get("/api/ready", tags=["ops"])
def ready():
    if not (MODEL_DIR / "best_model.joblib").exists():
        raise HTTPException(503, "No trained model. Run 'python src/train.py'.")
    _load(None)
    return {"status": "ready"}


@app.get("/api/questions", tags=["questionnaire"])
def questions():
    return {
        "items": [
            {
                "id": feat,
                "question": question,
                "options": options,
                "scoringOptions": sorted(scoring),
            }
            for feat, question, options, scoring in QCHAT10
        ],
        "demographics": [
            {"id": field, "question": q, "kind": kind, "options": options}
            for field, q, kind, options in DEMOGRAPHIC_QUESTIONS
        ],
        "cutoff": QCHAT_CUTOFF,
        "ageRange": {
            "min": MIN_AGE_MONTHS,
            "max": MAX_AGE_MONTHS,
            "validated": list(VALIDATED_AGE_RANGE),
        },
    }


@app.get("/api/models", tags=["models"])
def models():
    best = None
    if (MODEL_DIR / "best_model.joblib").exists():
        best = _load(None)["name"]
    return {
        "models": [
            {"stem": p.stem, "name": p.stem.replace("_", " ")} for p in _model_files()
        ],
        "best": best,
    }


@app.get("/api/metrics", tags=["models"])
def metrics():
    csv = REPORT_DIR / "metrics_comparison.csv"
    if not csv.exists():
        raise HTTPException(404, "No metrics yet. Run 'python src/train.py'.")

    df = pd.read_csv(csv, index_col=0)
    payload: dict[str, Any] = {
        "columns": list(df.columns),
        "rows": [{"model": idx, **row} for idx, row in df.round(4).iterrows()],
    }

    production = REPORT_DIR / "production_report.json"
    if production.exists():
        payload["production"] = json.loads(production.read_text())

    prevalence = REPORT_DIR / "prevalence_adjusted.csv"
    if prevalence.exists():
        payload["prevalence"] = pd.read_csv(prevalence).round(4).to_dict("records")

    return payload


@app.get("/api/model-card", tags=["models"])
def model_card():
    card = REPORT_DIR / "MODEL_CARD.md"
    if not card.exists():
        raise HTTPException(404, "No model card yet. Run 'python src/train.py'.")
    return {"markdown": card.read_text(encoding="utf-8")}


@app.post("/api/predict", response_model=PredictResponse, tags=["questionnaire"])
def predict(req: PredictRequest, request: Request):
    scores = score_answers(req.answers)
    bundle = _load(req.model)
    eligibility = _eligibility(req.demographics.age_months)

    row: dict[str, Any] = {f"A{i}": scores[f"A{i}"] for i in range(1, 11)}
    row.update(req.demographics.to_features())
    row["Qchat-10-Score"] = scores["total"]
    frame = pd.DataFrame([{c: row.get(c) for c in bundle["features"]}])

    model = bundle["model"]
    threshold = bundle["threshold"]

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(frame)[0][1])
        prediction = int(probability >= threshold)
    else:
        probability = None
        prediction = int(model.predict(frame)[0])

    flagged = prediction == 1 or scores["total"] > QCHAT_CUTOFF
    screening_id = str(uuid.uuid4())
    completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    ci = bundle["metrics"].get("bootstrap_ci", {})

    _audit(
        {
            "screening_id": screening_id,
            "at": completed_at,
            "request_id": getattr(request.state, "request_id", None),
            "qchat_score": scores["total"],
            "prediction": prediction,
            "probability": probability,
            "model": bundle["name"],
            "model_version": bundle["version"],
            "age_months": req.demographics.age_months,
            "eligibility": eligibility["status"],
        }
    )

    return PredictResponse(
        screening_id=screening_id,
        completed_at=completed_at,
        prediction=prediction,
        label="Elevated ASD traits" if flagged else "No elevated ASD traits",
        probability=probability,
        threshold=threshold,
        qchat_score=scores["total"],
        cutoff=QCHAT_CUTOFF,
        above_cutoff=scores["total"] > QCHAT_CUTOFF,
        item_scores={f"A{i}": scores[f"A{i}"] for i in range(1, 11)},
        model_name=bundle["name"],
        model_version=bundle["version"],
        calibration=bundle["calibration"],
        eligibility=eligibility,
        performance={
            "sensitivity": ci.get("Sensitivity"),
            "specificity": ci.get("Specificity"),
            "npv": ci.get("NPV"),
        },
        interpretation=_interpretation(flagged, eligibility),
    )
