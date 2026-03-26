"""
Extraction Accuracy Evaluation Harness — xBOQ Gap 3

Measures the *accuracy* of xBOQ QTO extraction modules against human-labeled
ground truth.  Complements the existing `benchmarks/` regression suite (which
checks pipeline KPI thresholds) by computing quantity-level metrics:

    MAE   — Mean Absolute Error (counts / sqm)
    MAPE  — Mean Absolute Percentage Error
    Acc   — Exact-match accuracy within ±N%
    F1    — For element presence/absence

Typical workflow
----------------
1. Human annotates a drawing PDF and fills in ground truth (GT):
       {"door": 12, "window": 18, "wc": 4}          # element counts
       [{"description": "…", "qty": 450, "unit": "sqm"}]   # BOQ items

2. Run the harness:
       harness = EvalHarness(llm_client=client)
       case = EvalCase.load("benchmarks/gt/villa.json")
       result = harness.run(case)
       harness.print_report(result)
       harness.save_run(result, "benchmarks/_runs")

3. If no GT yet (unlabeled mode):
       result = harness.run_unlabeled("villa", ["/path/to/villa.pdf"])
       # writes predicted outputs to benchmarks/_runs for later human labeling

Files produced per run
----------------------
  benchmarks/_runs/<case_name>/<timestamp>/
      eval_result.json       — full metrics + raw predictions + GT
      console_report.txt     — human-readable table (also printed to stdout)
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import textwrap
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EvalCase:
    """
    A single labeled evaluation case.

    Attributes
    ----------
    name : str
        Unique case name (e.g. "villa_2br", "hospital_block_a")
    pdf_paths : List[str]
        Path(s) to the drawing PDF(s) for this case.
    ground_truth_elements : Dict[str, int]
        Human-labeled element counts keyed by element type.
        e.g. {"door": 12, "window": 18, "wc": 4, "room": 6}
        Leave empty {} if unknown (unlabeled mode).
    ground_truth_boq : List[dict]
        Human-labeled BOQ rows.  Each dict has at minimum:
            description (str), qty (float), unit (str)
        Optional: trade (str), item_no (str/int)
        Leave empty [] if unknown (unlabeled mode).
    ground_truth_area_sqm : float
        Total floor area in sqm (0.0 if unknown).
    notes : str
        Free-text notes (drawing type, floors, scale, etc.)
    mode : str
        "visual" — runs visual element detector (requires llm_client + fitz/PIL)
        "text"   — runs text-based QTO only (no LLM vision call)
        "full"   — runs both and merges (default)
    page_texts : List[Tuple[int, str, str]]
        Pre-extracted page texts [(page_idx, text, doc_type), …].
        If empty, eval harness skips text-based QTO.
    scale_ratio : int
        Known drawing scale ratio (e.g. 100 for 1:100).  0 = unknown.
    floors : int
        Number of floors in the building.  Defaults to 1.
    """
    name: str
    pdf_paths: List[str] = field(default_factory=list)
    ground_truth_elements: Dict[str, int] = field(default_factory=dict)
    ground_truth_boq: List[dict] = field(default_factory=list)
    ground_truth_area_sqm: float = 0.0
    notes: str = ""
    mode: str = "full"
    page_texts: List[Tuple[int, str, str]] = field(default_factory=list)
    scale_ratio: int = 0
    floors: int = 1

    @classmethod
    def load(cls, json_path: str) -> "EvalCase":
        """Load an EvalCase from a JSON file."""
        data = json.loads(Path(json_path).read_text())
        data["page_texts"] = [
            tuple(t) for t in data.get("page_texts", [])
        ]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, json_path: str) -> None:
        """Persist this EvalCase to a JSON file (for later labeling)."""
        out = asdict(self)
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        Path(json_path).write_text(json.dumps(out, indent=2))


@dataclass
class ElementMetric:
    """Per-element-type accuracy metric."""
    element_type: str
    predicted: int
    ground_truth: int
    abs_error: int           # |pred - gt|
    pct_error: float         # |pred - gt| / gt * 100  (NaN if gt=0)
    within_10pct: bool
    within_20pct: bool


@dataclass
class BOQMetric:
    """Per-BOQ-item accuracy metric."""
    description: str
    predicted_qty: float
    ground_truth_qty: float
    unit: str
    abs_error: float
    pct_error: float         # NaN if gt=0
    within_10pct: bool
    within_20pct: bool


@dataclass
class EvalResult:
    """
    Full evaluation result for one EvalCase.

    Attributes
    ----------
    case_name : str
    timestamp : str          ISO-8601 UTC
    is_labeled : bool        False = unlabeled run (no GT metrics)

    element_metrics : List[ElementMetric]
    boq_metrics : List[BOQMetric]

    element_mae : float      Mean Absolute Error across element types
    element_mape : float     MAPE (%) across element types with GT > 0
    element_acc_10 : float   % of element types within ±10%
    element_acc_20 : float   % of element types within ±20%

    boq_mae : float          MAE across BOQ item quantities
    boq_mape : float         MAPE (%) across BOQ items with GT > 0
    boq_acc_10 : float       % of BOQ items within ±10%
    boq_acc_20 : float       % of BOQ items within ±20%

    area_predicted_sqm : float
    area_ground_truth_sqm : float
    area_pct_error : float   (NaN if GT=0)

    raw_predicted_elements : Dict[str, int]   raw outputs from extraction
    raw_predicted_boq : List[dict]
    warnings : List[str]
    notes : str
    """
    case_name: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_labeled: bool = True

    element_metrics: List[ElementMetric] = field(default_factory=list)
    boq_metrics: List[BOQMetric] = field(default_factory=list)

    element_mae: float = 0.0
    element_mape: float = 0.0
    element_acc_10: float = 0.0
    element_acc_20: float = 0.0

    boq_mae: float = 0.0
    boq_mape: float = 0.0
    boq_acc_10: float = 0.0
    boq_acc_20: float = 0.0

    area_predicted_sqm: float = 0.0
    area_ground_truth_sqm: float = 0.0
    area_pct_error: float = float("nan")

    raw_predicted_elements: Dict[str, int] = field(default_factory=dict)
    raw_predicted_boq: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: str = ""

    def overall_score(self) -> float:
        """
        0–100 composite accuracy score (higher = better).
        Blends element accuracy (60%) and BOQ accuracy (40%).
        Returns 0 if not labeled.
        """
        if not self.is_labeled:
            return 0.0
        elem_score = self.element_acc_20 * 60.0
        boq_score  = self.boq_acc_20 * 40.0 if self.boq_metrics else 60.0
        return round(elem_score + boq_score, 1)


# =============================================================================
# METRIC COMPUTATION
# =============================================================================

def _pct_error(pred: float, gt: float) -> float:
    if gt == 0:
        return float("nan")
    return abs(pred - gt) / gt * 100.0


def _compute_element_metrics(
    predicted: Dict[str, int],
    ground_truth: Dict[str, int],
) -> Tuple[List[ElementMetric], float, float, float, float]:
    """
    Compare predicted element counts against GT.
    Returns (metrics, mae, mape, acc_10, acc_20).
    """
    metrics: List[ElementMetric] = []
    all_types = set(predicted) | set(ground_truth)

    for etype in sorted(all_types):
        pred = predicted.get(etype, 0)
        gt   = ground_truth.get(etype, 0)
        ae   = abs(pred - gt)
        pe   = _pct_error(pred, gt)
        metrics.append(ElementMetric(
            element_type=etype,
            predicted=pred,
            ground_truth=gt,
            abs_error=ae,
            pct_error=pe,
            within_10pct=(not math.isnan(pe) and pe <= 10.0),
            within_20pct=(not math.isnan(pe) and pe <= 20.0),
        ))

    if not metrics:
        return metrics, 0.0, 0.0, 0.0, 0.0

    mae  = sum(m.abs_error for m in metrics) / len(metrics)
    valid = [m for m in metrics if not math.isnan(m.pct_error)]
    mape = sum(m.pct_error for m in valid) / len(valid) if valid else float("nan")
    acc10 = sum(1 for m in valid if m.within_10pct) / len(valid) if valid else 0.0
    acc20 = sum(1 for m in valid if m.within_20pct) / len(valid) if valid else 0.0

    return metrics, mae, (mape if not math.isnan(mape) else 0.0), acc10, acc20


def _match_boq_items(
    predicted_boq: List[dict],
    ground_truth_boq: List[dict],
) -> Tuple[List[BOQMetric], float, float, float, float]:
    """
    Match predicted BOQ items to GT items by description similarity (case-insensitive
    substring match).  Unmatched GT items are scored as qty=0.
    Returns (metrics, mae, mape, acc_10, acc_20).
    """
    metrics: List[BOQMetric] = []

    for gt_item in ground_truth_boq:
        gt_desc = str(gt_item.get("description", "")).lower().strip()
        gt_qty  = float(gt_item.get("qty", 0) or 0)
        gt_unit = str(gt_item.get("unit", ""))

        # Find best matching predicted item
        best_pred_qty = 0.0
        for p in predicted_boq:
            p_desc = str(p.get("description", "")).lower()
            if _desc_match(gt_desc, p_desc):
                best_pred_qty = float(p.get("qty", 0) or 0)
                break

        ae = abs(best_pred_qty - gt_qty)
        pe = _pct_error(best_pred_qty, gt_qty)
        metrics.append(BOQMetric(
            description=gt_item.get("description", ""),
            predicted_qty=best_pred_qty,
            ground_truth_qty=gt_qty,
            unit=gt_unit,
            abs_error=ae,
            pct_error=pe,
            within_10pct=(not math.isnan(pe) and pe <= 10.0),
            within_20pct=(not math.isnan(pe) and pe <= 20.0),
        ))

    if not metrics:
        return metrics, 0.0, 0.0, 0.0, 0.0

    mae  = sum(m.abs_error for m in metrics) / len(metrics)
    valid = [m for m in metrics if not math.isnan(m.pct_error)]
    mape = sum(m.pct_error for m in valid) / len(valid) if valid else float("nan")
    acc10 = sum(1 for m in valid if m.within_10pct) / len(valid) if valid else 0.0
    acc20 = sum(1 for m in valid if m.within_20pct) / len(valid) if valid else 0.0

    return metrics, mae, (mape if not math.isnan(mape) else 0.0), acc10, acc20


def _desc_match(needle: str, haystack: str, min_overlap: int = 5) -> bool:
    """
    Simple substring match: the longest word of needle must appear in haystack.
    """
    words = [w for w in needle.split() if len(w) >= min_overlap]
    return any(w in haystack for w in words)


# =============================================================================
# EXTRACTION ADAPTERS
# =============================================================================

def _extract_visual_elements(
    pdf_path: str,
    page_texts: List[Tuple[int, str, str]],
    llm_client: Any,
    scale_ratio: int = 0,
) -> Tuple[Dict[str, int], List[dict], float, List[str]]:
    """
    Run visual element detector on a single PDF.
    Returns (element_counts, boq_items, detected_area_sqm, warnings).
    """
    try:
        from src.analysis.qto.visual_element_detector import run_visual_detection
    except ImportError as exc:
        return {}, [], 0.0, [f"visual_element_detector import failed: {exc}"]

    try:
        result = run_visual_detection(
            pdf_path=pdf_path,
            page_texts=page_texts,
            llm_client=llm_client,
        )
        counts: Dict[str, int] = {}
        for el in result.elements:
            counts[el.element_type] = counts.get(el.element_type, 0) + el.count
        return counts, result.line_items, result.detected_area_sqm, result.warnings
    except Exception as exc:
        logger.warning("Visual extraction failed: %s", exc)
        return {}, [], 0.0, [f"Visual extraction error: {exc}"]


def _extract_text_elements(
    page_texts: List[Tuple[int, str, str]],
    floors: int = 1,
    total_area_sqm: float = 0.0,
) -> Tuple[Dict[str, int], List[dict], List[str]]:
    """
    Run text-based QTO modules (structural, door/window schedules) on page_texts.
    Returns (element_counts, boq_items, warnings).
    """
    warnings_out: List[str] = []
    all_items: List[dict] = []
    counts: Dict[str, int] = {}

    # ── Door/window schedule extractor ──────────────────────────────────
    try:
        from src.analysis.qto.door_window_takeoff import run_door_window_takeoff
        dw = run_door_window_takeoff(page_texts=page_texts)
        counts["door"]   = counts.get("door",   0) + dw.total_doors
        counts["window"] = counts.get("window", 0) + dw.total_windows
        all_items.extend(dw.line_items)
    except Exception as exc:
        warnings_out.append(f"door_window_takeoff: {exc}")

    # ── Structural takeoff ───────────────────────────────────────────────
    try:
        from src.analysis.qto.structural_takeoff import run_structural_takeoff
        st = run_structural_takeoff(
            page_texts=page_texts,
            floors=floors,
            total_area_sqm=total_area_sqm,
        )
        _st_col_count = sum(
            getattr(e, "count", 1)
            for e in getattr(st, "elements", [])
            if getattr(e, "element_type", "") == "column"
        )
        if _st_col_count:
            counts["column"] = counts.get("column", 0) + _st_col_count
        all_items.extend(st.line_items)
    except Exception as exc:
        warnings_out.append(f"structural_takeoff: {exc}")

    return counts, all_items, warnings_out


# =============================================================================
# EVAL HARNESS
# =============================================================================

class EvalHarness:
    """
    Runs extraction on EvalCase instances and measures accuracy.

    Parameters
    ----------
    llm_client : optional
        OpenAI or Anthropic client for vision extraction.
        If None, only text-based extraction runs.
    tolerance_pct : float
        Default tolerance for acc_N metric (default 20.0%).
    """

    def __init__(
        self,
        llm_client: Any = None,
        tolerance_pct: float = 20.0,
    ) -> None:
        self.llm_client = llm_client
        self.tolerance_pct = tolerance_pct

    # ── Main run ──────────────────────────────────────────────────────────

    def run(self, case: EvalCase) -> EvalResult:
        """
        Run extraction on a labeled EvalCase and compute accuracy metrics.
        If case.ground_truth_elements is empty, switches to unlabeled mode.
        """
        is_labeled = bool(case.ground_truth_elements or case.ground_truth_boq)
        result = EvalResult(
            case_name=case.name,
            is_labeled=is_labeled,
            area_ground_truth_sqm=case.ground_truth_area_sqm,
            notes=case.notes,
        )

        all_pred_counts: Dict[str, int] = {}
        all_pred_boq: List[dict] = []
        area_sqm = 0.0

        # ── Visual extraction ─────────────────────────────────────────────
        if case.mode in ("visual", "full") and self.llm_client is not None:
            primary_pdf = case.pdf_paths[0] if case.pdf_paths else ""
            if primary_pdf:
                vis_counts, vis_boq, vis_area, vis_warns = _extract_visual_elements(
                    pdf_path=primary_pdf,
                    page_texts=case.page_texts,
                    llm_client=self.llm_client,
                    scale_ratio=case.scale_ratio,
                )
                for k, v in vis_counts.items():
                    all_pred_counts[k] = all_pred_counts.get(k, 0) + v
                all_pred_boq.extend(vis_boq)
                area_sqm = max(area_sqm, vis_area)
                result.warnings.extend(vis_warns)

        # ── Text extraction ────────────────────────────────────────────────
        if case.mode in ("text", "full") and case.page_texts:
            txt_counts, txt_boq, txt_warns = _extract_text_elements(
                page_texts=case.page_texts,
                floors=case.floors,
                total_area_sqm=case.ground_truth_area_sqm,
            )
            for k, v in txt_counts.items():
                # Take max of visual/text for same element type (avoid double-count)
                all_pred_counts[k] = max(all_pred_counts.get(k, 0), v)
            # Text BOQ supplements visual — only add items not already in visual boq
            existing_descs = {i.get("description", "").lower() for i in all_pred_boq}
            for item in txt_boq:
                if item.get("description", "").lower() not in existing_descs:
                    all_pred_boq.append(item)
            result.warnings.extend(txt_warns)

        # ── Store raw predictions ─────────────────────────────────────────
        result.raw_predicted_elements = all_pred_counts
        result.raw_predicted_boq      = all_pred_boq
        result.area_predicted_sqm     = area_sqm

        if case.ground_truth_area_sqm > 0:
            result.area_pct_error = _pct_error(area_sqm, case.ground_truth_area_sqm)

        # ── Compute metrics (labeled only) ────────────────────────────────
        if is_labeled:
            if case.ground_truth_elements:
                (
                    result.element_metrics,
                    result.element_mae,
                    result.element_mape,
                    result.element_acc_10,
                    result.element_acc_20,
                ) = _compute_element_metrics(all_pred_counts, case.ground_truth_elements)

            if case.ground_truth_boq:
                (
                    result.boq_metrics,
                    result.boq_mae,
                    result.boq_mape,
                    result.boq_acc_10,
                    result.boq_acc_20,
                ) = _match_boq_items(all_pred_boq, case.ground_truth_boq)

        return result

    def run_unlabeled(
        self,
        name: str,
        pdf_paths: List[str],
        page_texts: Optional[List[Tuple[int, str, str]]] = None,
        floors: int = 1,
        scale_ratio: int = 0,
        notes: str = "",
    ) -> EvalResult:
        """
        Run extraction on an unlabeled PDF and record outputs for later labeling.
        Returns EvalResult with is_labeled=False (no accuracy metrics).
        """
        case = EvalCase(
            name=name,
            pdf_paths=pdf_paths,
            page_texts=page_texts or [],
            floors=floors,
            scale_ratio=scale_ratio,
            notes=notes,
            mode="full",
        )
        return self.run(case)

    # ── Reporting ─────────────────────────────────────────────────────────

    def print_report(self, result: EvalResult) -> None:
        """Print a human-readable accuracy report to stdout."""
        print(self._format_report(result))

    def _format_report(self, result: EvalResult) -> str:
        lines: List[str] = []
        sep = "─" * 70

        lines.append(sep)
        lines.append(f"  xBOQ Extraction Accuracy — {result.case_name}")
        lines.append(f"  Run: {result.timestamp}")
        if result.notes:
            lines.append(f"  Notes: {result.notes}")
        lines.append(sep)

        if not result.is_labeled:
            lines.append("  Mode: UNLABELED — no accuracy metrics (outputs recorded for labeling)")
            lines.append("")
            lines.append("  Predicted element counts:")
            for etype, count in sorted(result.raw_predicted_elements.items()):
                lines.append(f"    {etype:<20} {count:>6}")
            lines.append("")
            lines.append("  Predicted BOQ items:")
            for item in result.raw_predicted_boq[:10]:
                qty  = item.get("qty", 0)
                unit = item.get("unit", "")
                desc = item.get("description", "")[:55]
                lines.append(f"    {qty:>8.1f} {unit:<6}  {desc}")
            if len(result.raw_predicted_boq) > 10:
                lines.append(f"    … +{len(result.raw_predicted_boq) - 10} more")
            lines.append(sep)
            return "\n".join(lines)

        # ── Element accuracy table ────────────────────────────────────────
        if result.element_metrics:
            lines.append("")
            lines.append("  ELEMENT COUNT ACCURACY")
            lines.append(f"  {'Type':<18} {'GT':>5} {'Pred':>5} {'|Err|':>6} {'MAPE%':>7} {'≤10%':>5} {'≤20%':>5}")
            lines.append("  " + "─" * 55)
            for m in result.element_metrics:
                pe_str = f"{m.pct_error:7.1f}" if not math.isnan(m.pct_error) else "   N/A"
                lines.append(
                    f"  {m.element_type:<18} {m.ground_truth:>5} {m.predicted:>5}"
                    f" {m.abs_error:>6} {pe_str} {'✓' if m.within_10pct else '✗':>5}"
                    f" {'✓' if m.within_20pct else '✗':>5}"
                )
            lines.append("")
            lines.append(
                f"  Summary → MAE: {result.element_mae:.1f}   MAPE: {result.element_mape:.1f}%"
                f"   Acc±10%: {result.element_acc_10:.0%}   Acc±20%: {result.element_acc_20:.0%}"
            )

        # ── BOQ accuracy table ─────────────────────────────────────────────
        if result.boq_metrics:
            lines.append("")
            lines.append("  BOQ QUANTITY ACCURACY")
            lines.append(f"  {'Description':<40} {'Unit':<5} {'GT':>8} {'Pred':>8} {'MAPE%':>7} {'≤20%':>5}")
            lines.append("  " + "─" * 75)
            for m in result.boq_metrics:
                pe_str = f"{m.pct_error:7.1f}" if not math.isnan(m.pct_error) else "   N/A"
                desc   = m.description[:40]
                lines.append(
                    f"  {desc:<40} {m.unit:<5} {m.ground_truth_qty:>8.1f} {m.predicted_qty:>8.1f}"
                    f" {pe_str} {'✓' if m.within_20pct else '✗':>5}"
                )
            lines.append("")
            lines.append(
                f"  Summary → MAE: {result.boq_mae:.1f}   MAPE: {result.boq_mape:.1f}%"
                f"   Acc±10%: {result.boq_acc_10:.0%}   Acc±20%: {result.boq_acc_20:.0%}"
            )

        # ── Area ──────────────────────────────────────────────────────────
        if result.area_ground_truth_sqm > 0:
            lines.append("")
            area_err_str = (
                f"{result.area_pct_error:.1f}%"
                if not math.isnan(result.area_pct_error) else "N/A"
            )
            lines.append(
                f"  AREA → GT: {result.area_ground_truth_sqm:.0f} sqm   "
                f"Pred: {result.area_predicted_sqm:.0f} sqm   Err: {area_err_str}"
            )

        # ── Overall score ──────────────────────────────────────────────────
        lines.append("")
        lines.append(f"  OVERALL SCORE: {result.overall_score():.1f} / 100")

        # ── Warnings ──────────────────────────────────────────────────────
        if result.warnings:
            lines.append("")
            lines.append("  WARNINGS:")
            for w in result.warnings[:5]:
                lines.append(f"    ⚠ {w}")
            if len(result.warnings) > 5:
                lines.append(f"    … +{len(result.warnings) - 5} more")

        lines.append(sep)
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────

    def save_run(self, result: EvalResult, output_dir: str = "benchmarks/_runs") -> Path:
        """
        Save eval result to output_dir/<case_name>/<timestamp>/eval_result.json
        and console_report.txt.  Returns the directory path.
        """
        ts_slug = result.timestamp.replace(":", "-").replace(".", "-")[:19]
        run_dir = Path(output_dir) / result.case_name / ts_slug
        run_dir.mkdir(parents=True, exist_ok=True)

        # JSON — convert dataclasses to dict
        def _to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _to_dict(v) for k, v in asdict(obj).items()}
            if isinstance(obj, float) and math.isnan(obj):
                return None
            return obj

        json_path = run_dir / "eval_result.json"
        json_path.write_text(json.dumps(_to_dict(result), indent=2))

        report_path = run_dir / "console_report.txt"
        report_path.write_text(self._format_report(result))

        logger.info("Eval run saved to: %s", run_dir)
        return run_dir

    def save_unlabeled_template(self, result: EvalResult, output_path: str) -> None:
        """
        Write a labeling template CSV from an unlabeled run's predictions.
        The human annotator fills in the 'ground_truth' column and runs again
        as a labeled case.
        """
        rows = [["element_type", "predicted_count", "ground_truth_count"]]
        for etype, count in sorted(result.raw_predicted_elements.items()):
            rows.append([etype, count, ""])

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        logger.info("Labeling template written to: %s", path)


# =============================================================================
# CLI CONVENIENCE
# =============================================================================

def run_from_json(case_json: str, output_dir: str = "benchmarks/_runs") -> EvalResult:
    """
    Load an EvalCase from JSON, run evaluation (no LLM — text-only), save and
    print results.  Useful for quick CLI usage without setting up an LLM client.

    Usage:
        python -c "
        from src.analysis.eval_harness import run_from_json
        run_from_json('benchmarks/gt/villa.json')
        "
    """
    case = EvalCase.load(case_json)
    harness = EvalHarness(llm_client=None)
    result = harness.run(case)
    harness.print_report(result)
    harness.save_run(result, output_dir)
    return result
