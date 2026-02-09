"""
Index Adapter

Maps runner's ProjectIndexer interface to real ingest module.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

# Import real module
from src.ingest import PlanIngester, ingest_plan


class ProjectIndexer:
    """
    Adapter for indexing drawings in a project.

    Runner expects:
        indexer = ProjectIndexer(project_dir)
        result = indexer.index_drawings(drawings_dir)

    We map this to the real ingest module.
    """

    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)
        self.ingester = PlanIngester()

    def index_drawings(self, drawings_dir: Path) -> Dict[str, Any]:
        """
        Index all drawings in the directory.

        Args:
            drawings_dir: Path to drawings folder

        Returns:
            Dictionary with:
                - total_pages: int
                - files: list of file info
                - ingested: list of IngestedPlan data
        """
        drawings_dir = Path(drawings_dir)

        # Find all drawing files
        patterns = ["*.pdf", "*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
        files = []
        for pattern in patterns:
            files.extend(drawings_dir.glob(pattern))

        result = {
            "total_pages": 0,
            "files": [],
            "ingested": [],
            "errors": [],
        }

        for file_path in sorted(files):
            file_info = {
                "path": str(file_path),
                "name": file_path.name,
                "suffix": file_path.suffix.lower(),
            }
            result["files"].append(file_info)

            try:
                # Use real ingester - get page count from the ingested result
                ingested = self.ingester.ingest(file_path)

                # Get ACTUAL page count from ingested data
                page_count = getattr(ingested, 'total_pages', 1)

                # For PDFs, also verify with fitz directly if page_count seems wrong
                if file_path.suffix.lower() == ".pdf" and page_count <= 1:
                    try:
                        import fitz
                        doc = fitz.open(str(file_path))
                        actual_pages = len(doc)
                        doc.close()
                        if actual_pages > page_count:
                            page_count = actual_pages
                    except Exception:
                        pass  # Keep ingested page_count if fitz fails

                result["total_pages"] += page_count
                file_info["page_count"] = page_count

                # Store ingested data
                ingested_data = {
                    "plan_id": ingested.plan_id,
                    "source_path": str(ingested.source_path),
                    "plan_type": ingested.plan_type.value if ingested.plan_type else "unknown",
                    "image_shape": list(ingested.image.shape) if ingested.image is not None else None,
                    "vector_text_count": len(ingested.vector_texts) if ingested.vector_texts else 0,
                }
                result["ingested"].append(ingested_data)

            except Exception as e:
                result["errors"].append({
                    "file": str(file_path),
                    "error": str(e),
                })

        return result


def index_project(project_dir: Path, drawings_dir: Path = None) -> Dict[str, Any]:
    """
    Convenience function to index a project's drawings.

    Args:
        project_dir: Project directory
        drawings_dir: Drawings directory (optional, defaults to project_dir/drawings)

    Returns:
        Index result dictionary
    """
    if drawings_dir is None:
        drawings_dir = Path(project_dir) / "drawings"

    indexer = ProjectIndexer(project_dir)
    return indexer.index_drawings(drawings_dir)
