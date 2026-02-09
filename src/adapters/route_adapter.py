"""
Route Adapter

Maps runner's PageRouter interface to page classification logic.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional


class PageRouter:
    """
    Adapter for routing/classifying drawing pages.

    Runner expects:
        router = PageRouter()
        result = router.route_pages(pages)

    This classifies pages as floor_plan, structural, electrical, etc.
    """

    # Page type patterns based on filename or extracted text
    PAGE_TYPE_PATTERNS = {
        "floor_plan": [
            r"floor\s*plan",
            r"fp\d*",
            r"ground\s*floor",
            r"first\s*floor",
            r"basement",
            r"typical\s*floor",
            r"layout",
        ],
        "structural": [
            r"structural",
            r"rcc",
            r"foundation",
            r"footing",
            r"beam",
            r"column",
            r"slab\s*layout",
        ],
        "section": [
            r"section",
            r"sec\s*[a-z]",
            r"cross\s*section",
        ],
        "elevation": [
            r"elevation",
            r"elev",
            r"front\s*view",
            r"side\s*view",
        ],
        "electrical": [
            r"electrical",
            r"elec",
            r"wiring",
            r"lighting",
            r"power\s*layout",
        ],
        "plumbing": [
            r"plumbing",
            r"water\s*supply",
            r"drainage",
            r"sanitary",
        ],
        "site_plan": [
            r"site\s*plan",
            r"plot\s*plan",
            r"site\s*layout",
        ],
        "door_window_schedule": [
            r"door.*schedule",
            r"window.*schedule",
            r"opening.*schedule",
            r"d/w\s*schedule",
        ],
        "finish_schedule": [
            r"finish.*schedule",
            r"specification",
            r"material.*schedule",
        ],
    }

    def __init__(self):
        # Compile patterns
        self.compiled_patterns = {}
        for page_type, patterns in self.PAGE_TYPE_PATTERNS.items():
            self.compiled_patterns[page_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def classify_page(self, filename: str, text_content: str = "") -> str:
        """
        Classify a single page.

        Args:
            filename: Page filename
            text_content: Extracted text from page (optional)

        Returns:
            Page type string
        """
        # Combine filename and text for matching
        combined = f"{filename} {text_content}".lower()

        for page_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(combined):
                    return page_type

        return "unknown"

    def route_pages(self, pages: List[Dict]) -> Dict[str, Any]:
        """
        Route multiple pages.

        Args:
            pages: List of page info dicts with 'path' and optionally 'text'

        Returns:
            Routing result with classified pages
        """
        result = {
            "routed": True,
            "types": {},
            "pages": [],
        }

        type_counts = {}

        for page in pages:
            path = Path(page.get("path", ""))
            text = page.get("text", "")

            page_type = self.classify_page(path.name, text)

            type_counts[page_type] = type_counts.get(page_type, 0) + 1

            result["pages"].append({
                "path": str(path),
                "type": page_type,
            })

        result["types"] = type_counts
        return result

    def route_directory(self, drawings_dir: Path) -> Dict[str, Any]:
        """
        Route all pages in a directory.

        Args:
            drawings_dir: Path to drawings directory

        Returns:
            Routing result
        """
        drawings_dir = Path(drawings_dir)

        # Find all files
        patterns = ["*.pdf", "*.png", "*.jpg", "*.jpeg"]
        files = []
        for pattern in patterns:
            files.extend(drawings_dir.glob(pattern))

        pages = [{"path": str(f)} for f in sorted(files)]
        return self.route_pages(pages)


def route_project_pages(drawings_dir: Path) -> Dict[str, Any]:
    """
    Convenience function to route pages in a directory.

    Args:
        drawings_dir: Drawings directory

    Returns:
        Routing result
    """
    router = PageRouter()
    return router.route_directory(drawings_dir)
