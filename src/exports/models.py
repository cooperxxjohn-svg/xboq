"""
Data models for XBOQ exports.

Typed dataclasses for:
- RFI items with evidence
- Trade gaps with coverage
- Blockers with impact
- Assumptions log
- Export bundles
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Trade(str, Enum):
    CIVIL = "civil"
    STRUCTURAL = "structural"
    ARCHITECTURAL = "architectural"
    MEP = "mep"
    FINISHES = "finishes"
    GENERAL = "general"


class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConfidenceBand(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Evidence:
    """Traceability data for claims."""
    sheet_ids: List[str] = field(default_factory=list)
    page_numbers: List[int] = field(default_factory=list)
    detected_entities: List[str] = field(default_factory=list)  # e.g., door tags, room names
    search_performed: str = ""  # what was looked for
    not_found: str = ""  # what wasn't found
    snippet: str = ""  # text excerpt if available

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sheet_ids": self.sheet_ids,
            "page_numbers": self.page_numbers,
            "detected_entities": self.detected_entities,
            "search_performed": self.search_performed,
            "not_found": self.not_found,
            "snippet": self.snippet,
        }

    def summary(self) -> str:
        """Human-readable evidence summary."""
        parts = []
        if self.page_numbers:
            parts.append(f"Pages: {', '.join(map(str, self.page_numbers))}")
        if self.detected_entities:
            parts.append(f"Found: {', '.join(self.detected_entities[:5])}")
        if self.not_found:
            parts.append(f"Missing: {self.not_found}")
        return " | ".join(parts) if parts else "No evidence"


@dataclass
class RFIItem:
    """Enhanced RFI with full traceability."""
    id: str
    title: str
    trade: str
    priority: str  # critical, high, medium, low
    missing_info: str
    why_it_matters: str  # cost/time risk
    evidence: Evidence
    suggested_resolution: str
    issue_type: str = ""
    package: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "trade": self.trade,
            "priority": self.priority,
            "missing_info": self.missing_info,
            "why_it_matters": self.why_it_matters,
            "evidence": self.evidence.to_dict(),
            "suggested_resolution": self.suggested_resolution,
            "issue_type": self.issue_type,
            "package": self.package,
        }

    def to_csv_row(self) -> Dict[str, str]:
        """Flatten for CSV export."""
        return {
            "RFI ID": self.id,
            "Title": self.title,
            "Trade": self.trade.title(),
            "Priority": self.priority.upper(),
            "Missing Info": self.missing_info,
            "Why It Matters": self.why_it_matters,
            "Evidence Pages": ", ".join(map(str, self.evidence.page_numbers)),
            "Detected Items": ", ".join(self.evidence.detected_entities[:10]),
            "Suggested Resolution": self.suggested_resolution,
        }

    def to_email_text(self) -> str:
        """Format for email draft."""
        return f"""RFI: {self.title}
Priority: {self.priority.upper()}

Issue:
{self.missing_info}

Impact:
{self.why_it_matters}

Evidence:
- Pages referenced: {', '.join(map(str, self.evidence.page_numbers)) or 'N/A'}
- Items detected: {', '.join(self.evidence.detected_entities[:5]) or 'N/A'}

Requested Action:
{self.suggested_resolution}
"""


@dataclass
class Blocker:
    """Critical blocker with severity and fix action."""
    id: str
    description: str
    severity: str  # critical, high, medium
    impact: str
    evidence: Evidence
    recommended_fix: str
    fix_score_delta: int = 0  # how much score improves if fixed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "severity": self.severity,
            "impact": self.impact,
            "evidence": self.evidence.to_dict(),
            "recommended_fix": self.recommended_fix,
            "fix_score_delta": self.fix_score_delta,
        }


@dataclass
class TradeGap:
    """Trade-level gap summary with coverage metrics."""
    trade: str
    rfi_count: int
    high_priority_count: int
    gaps: List[str]
    scope_coverage_pct: float  # 0-100
    priceable_items: int
    blocked_items: int
    missing_dependencies: List[str]
    cost_risk: str  # high, medium, low
    schedule_risk: str
    confidence: str  # high, medium, low
    next_action: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade": self.trade,
            "rfi_count": self.rfi_count,
            "high_priority_count": self.high_priority_count,
            "gaps": self.gaps,
            "scope_coverage_pct": self.scope_coverage_pct,
            "priceable_items": self.priceable_items,
            "blocked_items": self.blocked_items,
            "missing_dependencies": self.missing_dependencies,
            "cost_risk": self.cost_risk,
            "schedule_risk": self.schedule_risk,
            "confidence": self.confidence,
            "next_action": self.next_action,
        }


@dataclass
class PricingReadinessRow:
    """Single row in pricing readiness sheet."""
    trade: str
    scope_coverage_pct: float
    priceable_items: int
    blocked_items: int
    top_missing: List[str]
    next_action: str
    confidence: str
    cost_risk: str
    schedule_risk: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Trade": self.trade.title(),
            "Scope Coverage %": f"{self.scope_coverage_pct:.0f}%",
            "Priceable Items": self.priceable_items,
            "Blocked Items": self.blocked_items,
            "Top Missing": ", ".join(self.top_missing[:3]),
            "Next Action": self.next_action,
            "Confidence": self.confidence.title(),
            "Cost Risk": self.cost_risk.title(),
            "Schedule Risk": self.schedule_risk.title(),
        }


@dataclass
class Assumption:
    """Logged assumption for bid packet."""
    id: str
    category: str  # scope, specification, pricing, timeline
    description: str
    accepted: bool = False
    source: str = "auto-suggested"  # auto-suggested, user-added
    impact_if_wrong: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "description": self.description,
            "accepted": self.accepted,
            "source": self.source,
            "impact_if_wrong": self.impact_if_wrong,
        }


@dataclass
class DrawingSetMeta:
    """Metadata about the analyzed drawing set."""
    total_pages: int
    total_sheets: int
    disciplines_detected: List[str]
    scale_status: str  # "all_scaled", "partial", "none"
    pages_without_scale: int
    revision_dates: List[str]
    file_names: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_pages": self.total_pages,
            "total_sheets": self.total_sheets,
            "disciplines_detected": self.disciplines_detected,
            "scale_status": self.scale_status,
            "pages_without_scale": self.pages_without_scale,
            "revision_dates": self.revision_dates,
            "file_names": self.file_names,
        }


@dataclass
class ExportBundle:
    """Container for all export file paths."""
    project_id: str
    created_at: datetime

    # Pricing readiness
    pricing_xlsx: Optional[str] = None
    pricing_csv: Optional[str] = None

    # RFI pack
    rfi_csv: Optional[str] = None
    rfi_email_txt: Optional[str] = None
    rfi_html: Optional[str] = None

    # Bid packet
    bid_packet_html: Optional[str] = None
    bid_packet_zip: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat(),
            "pricing_xlsx": self.pricing_xlsx,
            "pricing_csv": self.pricing_csv,
            "rfi_csv": self.rfi_csv,
            "rfi_email_txt": self.rfi_email_txt,
            "rfi_html": self.rfi_html,
            "bid_packet_html": self.bid_packet_html,
            "bid_packet_zip": self.bid_packet_zip,
        }
