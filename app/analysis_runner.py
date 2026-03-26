"""
Analysis Runner - Backward Compatibility Wrapper

This module re-exports from src.analysis.pipeline for backward compatibility.
All analysis pipeline code has been moved to src/analysis/pipeline.py
"""

import sys
from pathlib import Path

# Add src to path (use .resolve() to handle symlinks)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Re-export everything from the new location
from src.analysis.pipeline import (
    AnalysisStage,
    AnalysisResult,
    run_analysis_pipeline,
    save_uploaded_files,
    generate_project_id,
    load_pdf_pages,
    run_ocr_extraction,
    extract_text_from_pdf,
)

__all__ = [
    "AnalysisStage",
    "AnalysisResult",
    "run_analysis_pipeline",
    "save_uploaded_files",
    "generate_project_id",
    "load_pdf_pages",
    "run_ocr_extraction",
    "extract_text_from_pdf",
]


# =============================================================================
# TEST (delegates to pipeline module)
# =============================================================================

if __name__ == "__main__":
    from src.analysis.pipeline import PROJECT_ROOT as PIPELINE_ROOT
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analysis_runner.py <pdf_path>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    project_id = generate_project_id()
    output_dir = PROJECT_ROOT / "out" / project_id

    def progress_cb(stage_id: str, message: str, progress: float):
        print(f"[{stage_id}] {message} ({progress*100:.0f}%)")

    result = run_analysis_pipeline(
        input_files=[pdf_path],
        project_id=project_id,
        output_dir=output_dir,
        progress_callback=progress_cb,
    )

    if result.success:
        print(f"\n✅ Analysis complete!")
        print(f"   Project: {result.project_id}")
        print(f"   Output: {result.output_dir}")
        print(f"   Files: {result.files_generated}")
    else:
        print(f"\n❌ Analysis failed: {result.error_message}")
        print(result.stack_trace)
