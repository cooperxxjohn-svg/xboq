"""
Export Facade — single import point for all xBOQ export functionality.

Consolidates three export packages into one namespace:

    src/export/           — file-based exports (Excel BOQ, PDF summary, Word RFI)
    src/exports/          — operational deliverables (bid packet, RFI pack, pricing readiness)
    src/analysis/export/  — QTO Excel exporter (openpyxl, trade-wise)

Usage:
    from src.export_facade import (
        export_boq_excel,
        export_pdf_summary,
        export_rfi_word,
        export_qto_excel,
        build_bid_readiness_packet,
        build_rfi_pack,
        build_pricing_readiness_sheet,
        ExportBundle,
    )
"""

from __future__ import annotations

# ── File-based exports (src/export/) ──────────────────────────────────────────
try:
    from src.export import (
        export_boq_excel,
        export_pdf_summary,
        export_rfi_word,
        PlanExporter,
        export_plan,
    )
    _HAS_FILE_EXPORTS = True
except ImportError:
    _HAS_FILE_EXPORTS = False
    export_boq_excel = None  # type: ignore[assignment]
    export_pdf_summary = None  # type: ignore[assignment]
    export_rfi_word = None  # type: ignore[assignment]
    PlanExporter = None  # type: ignore[assignment]
    export_plan = None  # type: ignore[assignment]

# ── Operational deliverables (src/exports/) ───────────────────────────────────
try:
    from src.exports import (
        build_bid_readiness_packet,
        build_rfi_pack,
        build_pricing_readiness_sheet,
        ExportBundle,
        RFIItem,
        TradeGap,
        Blocker as ExportsBlocker,
        Assumption,
        Evidence,
        PricingReadinessRow,
    )
    _HAS_OP_EXPORTS = True
except ImportError:
    _HAS_OP_EXPORTS = False
    build_bid_readiness_packet = None  # type: ignore[assignment]
    build_rfi_pack = None  # type: ignore[assignment]
    build_pricing_readiness_sheet = None  # type: ignore[assignment]
    ExportBundle = None  # type: ignore[assignment]
    RFIItem = None  # type: ignore[assignment]
    TradeGap = None  # type: ignore[assignment]
    ExportsBlocker = None  # type: ignore[assignment]
    Assumption = None  # type: ignore[assignment]
    Evidence = None  # type: ignore[assignment]
    PricingReadinessRow = None  # type: ignore[assignment]

# ── QTO Excel exporter (src/analysis/export/) ─────────────────────────────────
try:
    from src.analysis.export.excel_exporter import generate_qto_excel
    _HAS_QTO_EXPORT = True
except ImportError:
    try:
        # Attempt alternate name
        from src.analysis.export.excel_exporter import export_qto_excel as generate_qto_excel  # type: ignore
        _HAS_QTO_EXPORT = True
    except ImportError:
        _HAS_QTO_EXPORT = False
        generate_qto_excel = None  # type: ignore[assignment]

# Expose with a consistent name
export_qto_excel = generate_qto_excel


# ── Status ────────────────────────────────────────────────────────────────────

def available_exporters() -> dict:
    """Return a dict showing which export packages are importable."""
    return {
        "file_exports": _HAS_FILE_EXPORTS,   # Excel BOQ, PDF, Word
        "op_exports":   _HAS_OP_EXPORTS,     # bid packet, RFI pack
        "qto_export":   _HAS_QTO_EXPORT,     # openpyxl trade-wise
    }


# ── Public API ────────────────────────────────────────────────────────────────

__all__ = [
    # File exports
    "export_boq_excel",
    "export_pdf_summary",
    "export_rfi_word",
    "PlanExporter",
    "export_plan",
    # Operational deliverables
    "build_bid_readiness_packet",
    "build_rfi_pack",
    "build_pricing_readiness_sheet",
    "ExportBundle",
    "RFIItem",
    "TradeGap",
    "ExportsBlocker",
    "Assumption",
    "Evidence",
    "PricingReadinessRow",
    # QTO Excel
    "export_qto_excel",
    "generate_qto_excel",
    # Status
    "available_exporters",
]
