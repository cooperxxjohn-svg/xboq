"""
Award Probability Model — T3-4.

Predicts the probability of winning a tender bid using a logistic regression
model trained on historical bid outcomes from rate_history.

Features used:
  - bid_readiness_score  (0-100, from BidSynthesis)
  - price_delta_pct      (% above/below benchmark; positive = more expensive)
  - competitor_count     (estimated number of bidders)
  - rfi_count            (open RFIs — proxy for uncertainty)
  - blocker_count        (blocking items — direct risk signal)
  - trade_confidence_mean (0-1, data quality indicator)

Pure numpy — no sklearn/xgboost dependency.

Heuristic weights (used before enough training data):
  sigmoid(0.03 * readiness - 0.02 * price_delta - 0.15 * competitors + 1.5)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL_PATH = Path.home() / ".xboq" / "award_model.json"
_LOW_CONFIDENCE_THRESHOLD = 10
_MEDIUM_CONFIDENCE_THRESHOLD = 30

# Heuristic weights: [readiness, price_delta, competitors, rfi_count, blocker_count, trade_confidence, bias]
_HEURISTIC_WEIGHTS = [0.03, -0.02, -0.15, -0.01, -0.05, 0.2, 1.5]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AwardPrediction:
    probability_pct: int              # 0-100
    confidence: str                   # "low" | "medium" | "high"
    key_drivers: List[dict] = field(default_factory=list)
    trained_on: int = 0               # number of historical bids used
    model_version: str = "heuristic"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _dot(weights: List[float], features: List[float]) -> float:
    return sum(w * f for w, f in zip(weights, features))


# ---------------------------------------------------------------------------
# AwardPredictor
# ---------------------------------------------------------------------------

class AwardPredictor:
    """
    Logistic regression win probability predictor.

    Training uses gradient descent (200 steps) on historical records with
    won=True/False outcomes.  Falls back to calibrated heuristic weights
    when fewer than 3 training examples are available.
    """

    def __init__(self):
        self._weights: Optional[List[float]] = None
        self._trained_on: int = 0
        self._load_model()

    # ── Model persistence ────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if _MODEL_PATH.exists():
            try:
                data = json.loads(_MODEL_PATH.read_text())
                self._weights = data.get("weights")
                self._trained_on = data.get("trained_on", 0)
            except Exception as exc:
                logger.debug("award_predictor: could not load model: %s", exc)

    def _save_model(self) -> None:
        try:
            _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            _MODEL_PATH.write_text(json.dumps({
                "weights": self._weights,
                "trained_on": self._trained_on,
            }))
        except Exception as exc:
            logger.debug("award_predictor: could not save model: %s", exc)

    # ── Feature extraction ───────────────────────────────────────────────────

    def extract_features(self, payload: dict, competitor_count: int = 3) -> dict:
        """Extract numeric features from a job payload."""
        readiness = float(payload.get("bid_readiness_score") or 0.0)
        price_delta = float(
            (payload.get("benchmark_comparison") or {}).get("pct_diff") or 0.0
        )
        rfis = payload.get("rfis") or []
        rfi_count = len([r for r in rfis if r.get("source") != "meeting"])
        blockers = payload.get("blockers") or []
        blocker_count = len(blockers)
        confidences = [
            float(v)
            for v in (payload.get("trade_confidence") or {}).values()
            if isinstance(v, (int, float))
        ]
        trade_conf_mean = (sum(confidences) / len(confidences)) if confidences else 0.5

        return {
            "bid_readiness_score": readiness,
            "price_delta_pct": price_delta,
            "competitor_count": float(competitor_count),
            "rfi_count": float(rfi_count),
            "blocker_count": float(blocker_count),
            "trade_confidence_mean": trade_conf_mean,
        }

    def _feature_vector(self, features: dict) -> List[float]:
        """Return [readiness, price_delta, competitors, rfi_count, blockers, trade_conf, 1.0]."""
        return [
            features.get("bid_readiness_score", 0.0),
            features.get("price_delta_pct", 0.0),
            features.get("competitor_count", 3.0),
            features.get("rfi_count", 0.0),
            features.get("blocker_count", 0.0),
            features.get("trade_confidence_mean", 0.5),
            1.0,    # bias
        ]

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, historical_records: List[dict]) -> None:
        """
        Fit logistic regression weights on historical bid records.

        Each record must have the same keys as extract_features() output
        plus a boolean "won" field.
        """
        if len(historical_records) < 3:
            logger.debug("award_predictor: not enough records (%d) to train", len(historical_records))
            return

        X: List[List[float]] = []
        y: List[float] = []
        for rec in historical_records:
            if "won" not in rec:
                continue
            X.append(self._feature_vector(rec))
            y.append(1.0 if rec["won"] else 0.0)

        if len(X) < 3:
            return

        n_features = len(X[0])
        w = [0.0] * n_features
        lr = 0.01

        for _ in range(200):
            grad = [0.0] * n_features
            for xi, yi in zip(X, y):
                pred = _sigmoid(_dot(w, xi))
                err = pred - yi
                for j in range(n_features):
                    grad[j] += err * xi[j]
            for j in range(n_features):
                w[j] -= lr * grad[j] / len(X)

        self._weights = w
        self._trained_on = len(X)
        self._save_model()

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, features: dict) -> AwardPrediction:
        """Return award probability for the given feature dict."""
        fv = self._feature_vector(features)
        weights = self._weights if self._weights else _HEURISTIC_WEIGHTS
        trained_on = self._trained_on

        raw_prob = _sigmoid(_dot(weights, fv))
        prob_pct = max(0, min(100, round(raw_prob * 100)))

        confidence: str
        if trained_on < _LOW_CONFIDENCE_THRESHOLD:
            confidence = "low"
        elif trained_on < _MEDIUM_CONFIDENCE_THRESHOLD:
            confidence = "medium"
        else:
            confidence = "high"

        # Build key drivers (feature impact vs neutral baseline)
        key_drivers = []
        feature_names = [
            ("bid_readiness_score", "readiness_score"),
            ("price_delta_pct", "price_competitiveness"),
            ("competitor_count", "competitor_count"),
            ("rfi_count", "open_rfis"),
            ("blocker_count", "blockers"),
            ("trade_confidence_mean", "data_quality"),
        ]
        for (feat_key, driver_name), w in zip(feature_names, weights):
            val = features.get(feat_key, 0.0)
            impact_pct = round(w * val * 100 / max(abs(_dot(weights, fv)), 0.01), 1)
            key_drivers.append({
                "factor": driver_name,
                "value": round(float(val), 2),
                "impact": f"{'+' if impact_pct >= 0 else ''}{impact_pct}%",
            })

        # Sort by absolute impact descending
        key_drivers.sort(key=lambda d: abs(float(d["impact"].strip("%+"))), reverse=True)

        return AwardPrediction(
            probability_pct=prob_pct,
            confidence=confidence,
            key_drivers=key_drivers[:4],     # top 4 drivers
            trained_on=trained_on,
            model_version="trained" if self._weights else "heuristic",
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_predictor: Optional[AwardPredictor] = None


def _get_predictor() -> AwardPredictor:
    global _predictor
    if _predictor is None:
        _predictor = AwardPredictor()
    return _predictor


def predict_award(payload: dict, competitor_count: int = 3) -> AwardPrediction:
    """Module-level convenience function."""
    predictor = _get_predictor()
    features = predictor.extract_features(payload, competitor_count)
    return predictor.predict(features)


def train_from_history() -> int:
    """
    Retrain the model from all rate_history records with a won field.

    Returns number of training records used.
    """
    try:
        from src.analysis.rate_history import load_all_records
        records = load_all_records()
        labelled = [r for r in records if "won" in r]
        if labelled:
            predictor = _get_predictor()
            predictor.train(labelled)
            return len(labelled)
    except Exception as exc:
        logger.debug("train_from_history failed: %s", exc)
    return 0
