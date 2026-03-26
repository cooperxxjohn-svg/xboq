"""
QA Workflow — T4-1.

Human reviewer approves/rejects/corrects extracted line items.
When all items are resolved, the job earns a "QA Verified" badge.

Storage: ~/.xboq/qa_reviews/{job_id}.json
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

_REVIEWS_DIR = Path.home() / ".xboq" / "qa_reviews"
_lock = threading.RLock()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class QAItem:
    item_id: str
    description: str
    trade: str
    quantity: float
    unit: str
    source_page: int
    status: str = "pending"          # "pending"|"approved"|"rejected"|"corrected"
    corrected_quantity: Optional[float] = None
    reviewer_note: str = ""
    reviewed_by: str = ""
    reviewed_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "QAItem":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class QAJob:
    job_id: str
    items: List[QAItem] = field(default_factory=list)
    created_at: str = ""
    verified: bool = False
    verified_at: str = ""
    verified_by: str = ""

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "items": [i.to_dict() for i in self.items],
            "created_at": self.created_at,
            "verified": self.verified,
            "verified_at": self.verified_at,
            "verified_by": self.verified_by,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QAJob":
        items = [QAItem.from_dict(i) for i in d.get("items", [])]
        return cls(
            job_id=d.get("job_id", ""),
            items=items,
            created_at=d.get("created_at", ""),
            verified=d.get("verified", False),
            verified_at=d.get("verified_at", ""),
            verified_by=d.get("verified_by", ""),
        )


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _qa_path(job_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in job_id)
    return _REVIEWS_DIR / f"{safe}.json"


def _load(job_id: str) -> Optional[QAJob]:
    p = _qa_path(job_id)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return QAJob.from_dict(json.load(f))


def _save(qa_job: QAJob) -> None:
    _REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    p = _qa_path(qa_job.job_id)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(qa_job.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_qa_job(job_id: str, payload: dict) -> QAJob:
    """
    Initialise a QA job from a pipeline payload.

    Pulls candidate items from payload["boq_items"] (if available) or
    from build_review_queue flagged items. Creates one QAItem per BOQ item.
    """
    with _lock:
        existing = _load(job_id)
        if existing is not None:
            return existing

        items: List[QAItem] = []

        # Primary source: boq_items
        for raw in payload.get("boq_items") or []:
            items.append(QAItem(
                item_id=raw.get("item_id") or str(uuid.uuid4())[:8],
                description=raw.get("description", "")[:200],
                trade=raw.get("trade", "unknown"),
                quantity=float(raw.get("quantity") or raw.get("qty") or 0.0),
                unit=raw.get("unit", ""),
                source_page=int(raw.get("source_page") or raw.get("page") or 0),
            ))

        # Fallback: review queue flagged items
        if not items:
            try:
                from src.analysis.review_queue import build_review_queue
                flags = build_review_queue(
                    quantity_reconciliation=payload.get("quantity_reconciliation") or [],
                    conflicts=payload.get("conflicts") or [],
                    pages_skipped=(payload.get("run_coverage") or {}).get("pages_skipped") or [],
                    toxic_summary=payload.get("toxic_pages"),
                    risk_results=payload.get("risk_results") or [],
                )
                for f in flags:
                    items.append(QAItem(
                        item_id=f.get("source_key", str(uuid.uuid4())[:8]),
                        description=f.get("title", ""),
                        trade="unknown",
                        quantity=0.0,
                        unit="",
                        source_page=int((f.get("page_refs") or [0])[0]),
                    ))
            except Exception:
                pass

        qa_job = QAJob(
            job_id=job_id,
            items=items,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        _save(qa_job)
        return qa_job


def submit_review(
    job_id: str,
    item_id: str,
    status: str,
    corrected_quantity: Optional[float] = None,
    note: str = "",
    reviewer: str = "",
) -> QAJob:
    """
    Apply a review decision to a single QA item.

    status must be one of: "approved" | "rejected" | "corrected"
    """
    valid = {"approved", "rejected", "corrected"}
    if status not in valid:
        raise ValueError(f"status must be one of {valid}, got {status!r}")

    with _lock:
        qa_job = _load(job_id)
        if qa_job is None:
            raise KeyError(f"QA job '{job_id}' not found")

        matched = False
        for item in qa_job.items:
            if item.item_id == item_id:
                item.status = status
                item.corrected_quantity = corrected_quantity
                item.reviewer_note = note
                item.reviewed_by = reviewer
                item.reviewed_at = datetime.now(timezone.utc).isoformat()
                matched = True
                break

        if not matched:
            raise KeyError(f"Item '{item_id}' not found in QA job '{job_id}'")

        _save(qa_job)
        return qa_job


def mark_verified(job_id: str, verified_by: str = "") -> QAJob:
    """Mark the entire QA job as verified (all items reviewed)."""
    with _lock:
        qa_job = _load(job_id)
        if qa_job is None:
            raise KeyError(f"QA job '{job_id}' not found")
        qa_job.verified = True
        qa_job.verified_at = datetime.now(timezone.utc).isoformat()
        qa_job.verified_by = verified_by
        _save(qa_job)
        return qa_job


def get_qa_status(job_id: str) -> Optional[QAJob]:
    """Return current QA job state, or None if not created yet."""
    with _lock:
        return _load(job_id)


def is_verified(job_id: str) -> bool:
    """True only when the QA job exists and has been explicitly verified."""
    qa_job = get_qa_status(job_id)
    return qa_job is not None and qa_job.verified


def pending_count(job_id: str) -> int:
    """Number of items still in 'pending' status."""
    qa_job = get_qa_status(job_id)
    if qa_job is None:
        return 0
    return sum(1 for i in qa_job.items if i.status == "pending")
