"""
Training Pack — bundle inputs, outputs, ground truth, and feedback for
paired dataset export.

Pure function, no Streamlit dependency.

Output structure (in ZIP):
    inputs/         — input PDF hashes (or filenames)
    outputs/        — our CSV + JSON outputs
    ground_truth/   — GT JSONs
    diff/           — diff report JSON
    context/        — bid context JSON, project metadata
    feedback/       — feedback.jsonl
    README.md       — auto-generated summary

Sprint 20: Pilot Conversion + Paired Dataset Capture.
"""

import io
import json
import hashlib
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Helpers ──────────────────────────────────────────────────────────────

def _hash_file(path: Path) -> str:
    """SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# Heavy payload keys to strip from the training pack (save space).
_STRIP_KEYS = frozenset({
    "ocr_text_cache", "ocr_bbox_meta", "diagnostics",
})


# ── Build ZIP ────────────────────────────────────────────────────────────

def build_training_pack(
    project_id: str,
    run_id: str,
    payload: Dict[str, Any],
    gt_diff: Optional[Dict[str, Any]] = None,
    gt_boq: Optional[List[dict]] = None,
    gt_schedules: Optional[List[dict]] = None,
    gt_quantities: Optional[List[dict]] = None,
    feedback_entries: Optional[List[dict]] = None,
    project_metadata: Optional[Dict[str, Any]] = None,
    csv_buffers: Optional[Dict[str, str]] = None,
    input_pdf_paths: Optional[List[Path]] = None,
) -> bytes:
    """
    Build a training pack ZIP.

    Args:
        project_id: Project identifier.
        run_id: Run identifier.
        payload: Full analysis payload dict.
        gt_diff: Ground truth diff result (from compute_gt_diff).
        gt_boq/gt_schedules/gt_quantities: Parsed GT data lists.
        feedback_entries: Feedback JSONL entries.
        project_metadata: Project metadata dict.
        csv_buffers: Dict of {filename: csv_string} for CSV exports.
        input_pdf_paths: Paths to input PDFs (will store hashes, not files).

    Returns:
        ZIP file content as bytes.
    """
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # ── inputs/ — file manifest with hashes ──
        input_manifest = []
        for pdf_path in (input_pdf_paths or []):
            p = Path(pdf_path)
            entry: Dict[str, Any] = {
                "filename": p.name,
                "size_bytes": p.stat().st_size if p.exists() else 0,
            }
            if p.exists():
                entry["sha256"] = _hash_file(p)
            input_manifest.append(entry)
        zf.writestr(
            "inputs/manifest.json",
            json.dumps(input_manifest, indent=2),
        )

        # ── outputs/ — slim payload + CSV exports ──
        slim_payload = {
            k: v for k, v in payload.items() if k not in _STRIP_KEYS
        }
        zf.writestr(
            "outputs/analysis.json",
            json.dumps(slim_payload, indent=2, default=str),
        )
        for fname, csv_content in (csv_buffers or {}).items():
            if isinstance(csv_content, bytes):
                zf.writestr(f"outputs/{fname}", csv_content)
            else:
                zf.writestr(f"outputs/{fname}", csv_content)

        # ── ground_truth/ ──
        if gt_boq:
            zf.writestr(
                "ground_truth/gt_boq.json",
                json.dumps(gt_boq, indent=2, default=str),
            )
        if gt_schedules:
            zf.writestr(
                "ground_truth/gt_schedules.json",
                json.dumps(gt_schedules, indent=2, default=str),
            )
        if gt_quantities:
            zf.writestr(
                "ground_truth/gt_quantities.json",
                json.dumps(gt_quantities, indent=2, default=str),
            )

        # ── diff/ ──
        if gt_diff:
            zf.writestr(
                "diff/gt_diff.json",
                json.dumps(gt_diff, indent=2, default=str),
            )

        # ── context/ ──
        context: Dict[str, Any] = {
            "project_id": project_id,
            "run_id": run_id,
            "exported_at": datetime.now().isoformat(),
            "readiness_score": payload.get("readiness_score"),
            "decision": payload.get("decision"),
        }
        if project_metadata:
            context["project_metadata"] = project_metadata
        zf.writestr(
            "context/bid_context.json",
            json.dumps(context, indent=2, default=str),
        )

        # ── feedback/ ──
        if feedback_entries:
            lines = [json.dumps(e, default=str) for e in feedback_entries]
            zf.writestr("feedback/feedback.jsonl", "\n".join(lines) + "\n")

        # ── README ──
        readme = _build_readme(
            project_id, run_id, gt_diff, len(feedback_entries or []),
        )
        zf.writestr("README.md", readme)

    buf.seek(0)
    return buf.getvalue()


def _build_readme(
    project_id: str,
    run_id: str,
    gt_diff: Optional[Dict[str, Any]],
    feedback_count: int,
) -> str:
    """Build a README for the training pack."""
    match_rate = "N/A"
    if gt_diff and gt_diff.get("overall_match_rate") is not None:
        match_rate = f"{gt_diff['overall_match_rate']:.1%}"

    return (
        f"# xBOQ Training Pack\n\n"
        f"Project: {project_id}\n"
        f"Run: {run_id}\n"
        f"Exported: {datetime.now().isoformat()}\n\n"
        f"## Contents\n"
        f"- `inputs/` — Input file manifest (SHA256 hashes)\n"
        f"- `outputs/` — xBOQ analysis outputs (JSON + CSV)\n"
        f"- `ground_truth/` — Human ground truth data\n"
        f"- `diff/` — Automated diff report (match rate: {match_rate})\n"
        f"- `context/` — Bid context and project metadata\n"
        f"- `feedback/` — User feedback entries ({feedback_count} entries)\n\n"
        f"## Schema\n"
        f"See templates in the xBOQ project for canonical column definitions.\n\n"
        f"Generated by xBOQ Training Pack Exporter (Sprint 20)\n"
    )


# ── Disk save ────────────────────────────────────────────────────────────

def save_training_pack_to_disk(
    project_id: str,
    run_id: str,
    zip_bytes: bytes,
    projects_dir: Path,
) -> Path:
    """
    Save training pack ZIP to projects_dir/<pid>/datasets/<run_id>/training_pack.zip.

    Returns path to the saved ZIP.
    """
    ds_dir = Path(projects_dir) / project_id / "datasets" / run_id
    ds_dir.mkdir(parents=True, exist_ok=True)
    path = ds_dir / "training_pack.zip"
    with open(path, "wb") as f:
        f.write(zip_bytes)
    return path
