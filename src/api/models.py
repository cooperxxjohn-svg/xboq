"""
xBOQ.ai — SQLAlchemy ORM models.

Tables
------
  jobs          Job metadata (status, org, progress, etc.)
                Payload JSON is stored on disk, not in DB (payloads are 5–50 MB)

  audit_log     Immutable compliance log.
                NEVER delete or update rows — append-only by convention.
                Captures: job lifecycle, auth events, rate changes, membership changes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.api.db import Base


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# JobModel
# ---------------------------------------------------------------------------

class JobModel(Base):
    """Persistent record for each analysis job."""

    __tablename__ = "jobs"

    job_id:          Mapped[str]            = mapped_column(String(64), primary_key=True)
    status:          Mapped[str]            = mapped_column(String(32), default="queued", nullable=False)
    org_id:          Mapped[str]            = mapped_column(String(128), default="local", nullable=False)
    project_name:    Mapped[str]            = mapped_column(String(256), default="", nullable=False)
    run_mode:        Mapped[str]            = mapped_column(String(64), default="demo_fast", nullable=False)
    created_at:      Mapped[datetime]       = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    completed_at:    Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    progress:        Mapped[float]          = mapped_column(Float, default=0.0, nullable=False)
    progress_message: Mapped[str]           = mapped_column(Text, default="", nullable=False)
    # Column name matches old sqlite3 schema ("errors"/"output_files") for
    # zero-downtime migration. Python attribute uses _json suffix to avoid
    # name collision with the @property accessors below.
    errors_json:       Mapped[str]          = mapped_column("errors", Text, default="[]", nullable=False)
    output_files_json: Mapped[str]          = mapped_column("output_files", Text, default="{}", nullable=False)
    payload_path:    Mapped[str]            = mapped_column(Text, default="", nullable=False)
    queue_position:  Mapped[int]            = mapped_column(Integer, default=0, nullable=False)

    __table_args__ = (
        Index("idx_jobs_org_created", "org_id", "created_at"),
    )

    # Convenience accessors (not columns)

    @property
    def errors(self) -> List[str]:
        try:
            return json.loads(self.errors_json or "[]")
        except Exception:
            return []

    @errors.setter
    def errors(self, value: List[str]) -> None:
        self.errors_json = json.dumps(value, default=str)

    @property
    def output_files(self) -> Dict[str, str]:
        try:
            return json.loads(self.output_files_json or "{}")
        except Exception:
            return {}

    @output_files.setter
    def output_files(self, value: Dict[str, str]) -> None:
        self.output_files_json = json.dumps(value, default=str)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "org_id": self.org_id,
            "project_name": self.project_name,
            "run_mode": self.run_mode,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "errors": self.errors,
            "output_files": self.output_files,
            "payload_path": self.payload_path,
            "queue_position": self.queue_position,
        }


# ---------------------------------------------------------------------------
# AuditLogModel  (append-only — no UPDATE or DELETE in application code)
# ---------------------------------------------------------------------------

class AuditLogModel(Base):
    """
    Immutable compliance audit log.

    Every significant action is appended here with enough context to answer:
      - Who performed the action?
      - What resource was affected?
      - When did it happen?
      - What changed?

    Convention: NEVER update or delete rows from this table.
    """

    __tablename__ = "audit_log"

    id:            Mapped[int]             = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    event_type:    Mapped[str]             = mapped_column(String(64), nullable=False)   # e.g. "job.created"
    org_id:        Mapped[str]             = mapped_column(String(128), default="", nullable=False)
    user_id:       Mapped[str]             = mapped_column(String(256), default="", nullable=False)
    resource_type: Mapped[str]             = mapped_column(String(64), default="", nullable=False)  # "job"|"rate"|"member"|"auth"
    resource_id:   Mapped[str]             = mapped_column(String(256), default="", nullable=False)
    detail_json:   Mapped[str]             = mapped_column(Text, default="{}", nullable=False)
    ip_address:    Mapped[str]             = mapped_column(String(64), default="", nullable=False)
    created_at:    Mapped[datetime]        = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (
        Index("idx_audit_org_created", "org_id", "created_at"),
        Index("idx_audit_event_type", "event_type"),
    )

    @property
    def detail(self) -> Dict[str, Any]:
        try:
            return json.loads(self.detail_json or "{}")
        except Exception:
            return {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event_type": self.event_type,
            "org_id": self.org_id,
            "user_id": self.user_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "detail": self.detail,
            "ip_address": self.ip_address,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# EstimatorFeedbackModel  — correction loop
# ---------------------------------------------------------------------------

class EstimatorFeedbackModel(Base):
    """
    Records every estimator correction to a quantity or rate.

    Purpose: ground-truth dataset for calibration and fine-tuning.
    Every correction stores the original AI value alongside the human
    correction so we can track accuracy drift over time.

    Convention: NEVER delete rows — corrections are immutable history.
    """

    __tablename__ = "estimator_feedback"

    id:             Mapped[int]   = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    job_id:         Mapped[str]   = mapped_column(String(64), nullable=False)
    org_id:         Mapped[str]   = mapped_column(String(128), default="local", nullable=False)
    user_id:        Mapped[str]   = mapped_column(String(256), default="", nullable=False)

    # What was corrected
    field_type:     Mapped[str]   = mapped_column(String(32), nullable=False)   # "quantity" | "rate" | "unit" | "description"
    trade:          Mapped[str]   = mapped_column(String(64), default="", nullable=False)
    item_ref:       Mapped[str]   = mapped_column(String(256), default="", nullable=False)   # BOQ item ref / description

    # The correction itself
    ai_value:       Mapped[str]   = mapped_column(Text, default="", nullable=False)    # original AI output
    human_value:    Mapped[str]   = mapped_column(Text, default="", nullable=False)    # corrected value
    unit:           Mapped[str]   = mapped_column(String(32), default="", nullable=False)
    confidence_was: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # AI confidence at time of correction

    # Context — helps understand *why* the correction was needed
    drawing_ref:    Mapped[str]   = mapped_column(String(256), default="", nullable=False)
    rule_applied:   Mapped[str]   = mapped_column(String(256), default="", nullable=False)
    correction_note: Mapped[str]  = mapped_column(Text, default="", nullable=False)

    created_at:     Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (
        Index("idx_feedback_org_job", "org_id", "job_id"),
        Index("idx_feedback_trade", "trade"),
        Index("idx_feedback_field_type", "field_type"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "org_id": self.org_id,
            "user_id": self.user_id,
            "field_type": self.field_type,
            "trade": self.trade,
            "item_ref": self.item_ref,
            "ai_value": self.ai_value,
            "human_value": self.human_value,
            "unit": self.unit,
            "confidence_was": self.confidence_was,
            "drawing_ref": self.drawing_ref,
            "rule_applied": self.rule_applied,
            "correction_note": self.correction_note,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# ProjectMemberModel  — replaces ~/.xboq/project_members/ flat files
# ---------------------------------------------------------------------------

class ProjectMemberModel(Base):
    """
    Project membership — who has what role on which project.

    Replaces the file-based ~/.xboq/project_members/{project_id}.json
    store for multi-instance / PostgreSQL deployments.
    """

    __tablename__ = "project_members"

    id:          Mapped[int]      = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    project_id:  Mapped[str]      = mapped_column(String(256), nullable=False)
    user_id:     Mapped[str]      = mapped_column(String(256), nullable=False)
    role:        Mapped[str]      = mapped_column(String(32), nullable=False)   # viewer | editor | admin
    added_by:    Mapped[str]      = mapped_column(String(256), default="", nullable=False)
    added_at:    Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (
        Index("idx_members_project", "project_id"),
        Index("idx_members_user", "user_id"),
        # Enforce one role per (project, user) pair
        Index("idx_members_unique", "project_id", "user_id", unique=True),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "user_id": self.user_id,
            "role": self.role,
            "added_by": self.added_by,
            "added_at": self.added_at.isoformat() if self.added_at else None,
        }


# ---------------------------------------------------------------------------
# OrgRateModel / ProjectRateModel  — replaces ~/.xboq/org_rates/ flat files
# ---------------------------------------------------------------------------

class OrgRateModel(Base):
    """
    Organisation-level rate overrides.

    Replaces ~/.xboq/org_rates/{org_id}.json for multi-instance deployments.
    """

    __tablename__ = "org_rates"

    id:           Mapped[int]      = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    org_id:       Mapped[str]      = mapped_column(String(128), nullable=False)
    material_key: Mapped[str]      = mapped_column(String(64), nullable=False)
    rate_inr:     Mapped[float]    = mapped_column(Float, nullable=False)
    unit:         Mapped[str]      = mapped_column(String(32), default="", nullable=False)
    notes:        Mapped[str]      = mapped_column(Text, default="", nullable=False)
    updated_at:   Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)
    updated_by:   Mapped[str]      = mapped_column(String(256), default="", nullable=False)

    __table_args__ = (
        Index("idx_org_rates_org_key", "org_id", "material_key", unique=True),
    )


class ProjectRateModel(Base):
    """
    Project-level rate overrides (highest priority — overrides org rates).

    Replaces ~/.xboq/project_rates/{project_id}.json.
    """

    __tablename__ = "project_rates"

    id:           Mapped[int]      = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    project_id:   Mapped[str]      = mapped_column(String(256), nullable=False)
    org_id:       Mapped[str]      = mapped_column(String(128), default="local", nullable=False)
    material_key: Mapped[str]      = mapped_column(String(64), nullable=False)
    rate_inr:     Mapped[float]    = mapped_column(Float, nullable=False)
    unit:         Mapped[str]      = mapped_column(String(32), default="", nullable=False)
    notes:        Mapped[str]      = mapped_column(Text, default="", nullable=False)
    updated_at:   Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)
    updated_by:   Mapped[str]      = mapped_column(String(256), default="", nullable=False)

    __table_args__ = (
        Index("idx_project_rates_proj_key", "project_id", "material_key", unique=True),
    )


# ---------------------------------------------------------------------------
# AccuracyBenchmarkModel  — ground truth comparison results
# ---------------------------------------------------------------------------

class RateHistoryModel(Base):
    """
    Time-series rate history for all tracked materials.

    Records:
    - Project/org rate overrides (actual procurement prices)
    - DSR / SOR official updates (scraped from public sources)
    - Manual market rate entries

    Powers the rate trend chart and staleness warnings on BOQ line items.
    """

    __tablename__ = "rate_history"

    id:           Mapped[int]      = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    material_key: Mapped[str]      = mapped_column(String(64), nullable=False)
    rate_inr:     Mapped[float]    = mapped_column(Float, nullable=False)
    source:       Mapped[str]      = mapped_column(String(64), default="manual", nullable=False)   # dsr_2023 | cpwd_circular | project_override | market
    region:       Mapped[str]      = mapped_column(String(64), default="national", nullable=False)
    org_id:       Mapped[str]      = mapped_column(String(128), default="system", nullable=False)
    notes:        Mapped[str]      = mapped_column(Text, default="", nullable=False)
    recorded_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (
        Index("idx_rate_history_key_date", "material_key", "recorded_at"),
        Index("idx_rate_history_org", "org_id"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "material_key": self.material_key,
            "rate_inr": self.rate_inr,
            "source": self.source,
            "region": self.region,
            "org_id": self.org_id,
            "notes": self.notes,
            "recorded_at": self.recorded_at.isoformat() if self.recorded_at else None,
        }


class AccuracyBenchmarkModel(Base):
    """
    Records the outcome of a ground-truth accuracy comparison.

    When a tender has a published BOQ (ground truth), the pipeline compares
    AI-extracted quantities against it and records per-trade deltas here.
    This feeds the /api/accuracy dashboard.
    """

    __tablename__ = "accuracy_benchmarks"

    id:              Mapped[int]   = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    job_id:          Mapped[str]   = mapped_column(String(64), nullable=False)
    org_id:          Mapped[str]   = mapped_column(String(128), default="local", nullable=False)
    tender_ref:      Mapped[str]   = mapped_column(String(256), default="", nullable=False)  # e.g. "nbcc_2024_001"
    tender_type:     Mapped[str]   = mapped_column(String(64), default="", nullable=False)   # "hospital" | "residential" | "industrial" etc

    # Aggregate accuracy
    qty_delta_pct:   Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # median quantity delta %
    rate_delta_pct:  Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # median rate delta %
    item_recall:     Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # fraction of BOQ items detected
    item_precision:  Mapped[float] = mapped_column(Float, default=0.0, nullable=False)  # fraction of AI items that are correct

    # Per-trade breakdown (JSON)
    trade_deltas_json: Mapped[str] = mapped_column(Text, default="{}", nullable=False)

    # Pipeline metadata
    run_mode:        Mapped[str]   = mapped_column(String(64), default="", nullable=False)
    page_count:      Mapped[int]   = mapped_column(Integer, default=0, nullable=False)
    processing_secs: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    created_at:      Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (
        Index("idx_accuracy_org", "org_id"),
        Index("idx_accuracy_tender_type", "tender_type"),
    )

    @property
    def trade_deltas(self) -> Dict[str, Any]:
        try:
            return json.loads(self.trade_deltas_json or "{}")
        except Exception:
            return {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "org_id": self.org_id,
            "tender_ref": self.tender_ref,
            "tender_type": self.tender_type,
            "qty_delta_pct": self.qty_delta_pct,
            "rate_delta_pct": self.rate_delta_pct,
            "item_recall": self.item_recall,
            "item_precision": self.item_precision,
            "trade_deltas": self.trade_deltas,
            "run_mode": self.run_mode,
            "page_count": self.page_count,
            "processing_secs": self.processing_secs,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
