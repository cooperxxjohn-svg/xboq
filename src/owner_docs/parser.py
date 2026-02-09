"""
Owner Docs Parser - Parse PDF documents from owner_docs folder.

Extracts text and classifies document types.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
from enum import Enum


class DocType(Enum):
    """Type of owner document."""
    TENDER_NOTICE = "tender_notice"
    BID_FORM = "bid_form"
    OWNER_BOQ = "owner_boq"
    SPECIFICATIONS = "specifications"
    CONTRACT_CONDITIONS = "contract_conditions"
    ADDENDUM = "addendum"
    DRAWINGS_INDEX = "drawings_index"
    SCHEDULE_OF_RATES = "schedule_of_rates"
    UNKNOWN = "unknown"


@dataclass
class ParsedDocument:
    """Parsed document content."""
    file_path: str
    file_name: str
    doc_type: DocType
    text_content: str
    pages: List[str] = field(default_factory=list)
    page_count: int = 0
    tables: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "doc_type": self.doc_type.value,
            "page_count": self.page_count,
            "metadata": self.metadata,
        }


class OwnerDocsParser:
    """Parse owner documents folder."""

    # Document type detection patterns
    DOC_TYPE_PATTERNS = {
        DocType.TENDER_NOTICE: [
            r'notice\s+inviting\s+tender',
            r'nit\s+no',
            r'tender\s+notice',
            r'invitation\s+to\s+bid',
            r'request\s+for\s+proposal',
            r'rfp\s+no',
        ],
        DocType.BID_FORM: [
            r'bid\s+form',
            r'tender\s+form',
            r'form\s+of\s+tender',
            r'schedule\s+of\s+quantities',
            r'bill\s+of\s+quantities',
        ],
        DocType.OWNER_BOQ: [
            r'bill\s+of\s+quantit',
            r'boq',
            r'schedule\s+of\s+quantit',
            r'item\s+no.*description.*unit.*qty',
            r'sr\.?\s*no.*description.*unit.*quantity',
        ],
        DocType.SPECIFICATIONS: [
            r'technical\s+specification',
            r'general\s+specification',
            r'particular\s+specification',
            r'specification\s+clause',
            r'material\s+specification',
        ],
        DocType.CONTRACT_CONDITIONS: [
            r'general\s+conditions\s+of\s+contract',
            r'gcc',
            r'particular\s+conditions',
            r'special\s+conditions',
            r'conditions\s+of\s+contract',
            r'terms\s+and\s+conditions',
        ],
        DocType.ADDENDUM: [
            r'addendum',
            r'corrigendum',
            r'amendment',
            r'clarification',
            r'pre-?bid\s+meeting',
        ],
        DocType.SCHEDULE_OF_RATES: [
            r'schedule\s+of\s+rates',
            r'sor',
            r'dsr',
            r'cpwd\s+rates',
            r'pwd\s+rates',
        ],
    }

    def __init__(self):
        self.compiled_patterns = {}
        for doc_type, patterns in self.DOC_TYPE_PATTERNS.items():
            self.compiled_patterns[doc_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def parse_folder(self, folder_path: Path) -> List[ParsedDocument]:
        """Parse all documents in folder."""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            return []

        documents = []

        # Find all PDFs
        pdf_files = list(folder_path.glob("*.pdf")) + list(folder_path.glob("**/*.pdf"))

        for pdf_path in pdf_files:
            doc = self.parse_document(pdf_path)
            if doc:
                documents.append(doc)

        # Also check for text/yaml files
        for ext in ["*.txt", "*.yaml", "*.yml"]:
            for file_path in folder_path.glob(ext):
                doc = self._parse_text_file(file_path)
                if doc:
                    documents.append(doc)

        return documents

    def parse_document(self, file_path: Path) -> Optional[ParsedDocument]:
        """Parse a single PDF document."""
        try:
            text_content, pages = self._extract_pdf_text(file_path)

            if not text_content:
                return None

            doc_type = self._classify_document(text_content, file_path.name)

            # Extract tables if BOQ
            tables = []
            if doc_type == DocType.OWNER_BOQ:
                tables = self._extract_tables(text_content)

            return ParsedDocument(
                file_path=str(file_path),
                file_name=file_path.name,
                doc_type=doc_type,
                text_content=text_content,
                pages=pages,
                page_count=len(pages),
                tables=tables,
            )

        except Exception as e:
            import logging
            logging.warning(f"Failed to parse {file_path}: {e}")
            return None

    def _extract_pdf_text(self, file_path: Path) -> tuple:
        """Extract text from PDF using available libraries."""
        text_content = ""
        pages = []

        try:
            # Try pdfplumber first
            import pdfplumber

            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    pages.append(page_text)
                    text_content += page_text + "\n\n"

        except ImportError:
            try:
                # Fallback to PyPDF2
                import PyPDF2

                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        pages.append(page_text)
                        text_content += page_text + "\n\n"

            except ImportError:
                # No PDF library available
                import logging
                logging.warning("No PDF library available (pdfplumber or PyPDF2)")
                return "", []

        return text_content, pages

    def _parse_text_file(self, file_path: Path) -> Optional[ParsedDocument]:
        """Parse a text/yaml file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()

            doc_type = self._classify_document(text_content, file_path.name)

            return ParsedDocument(
                file_path=str(file_path),
                file_name=file_path.name,
                doc_type=doc_type,
                text_content=text_content,
                pages=[text_content],
                page_count=1,
            )

        except Exception as e:
            import logging
            logging.warning(f"Failed to parse {file_path}: {e}")
            return None

    def _classify_document(self, text: str, filename: str) -> DocType:
        """Classify document type based on content and filename."""
        text_lower = text.lower()
        filename_lower = filename.lower()

        # Check filename hints first
        if "addend" in filename_lower or "corrig" in filename_lower:
            return DocType.ADDENDUM
        if "boq" in filename_lower or "bill" in filename_lower:
            return DocType.OWNER_BOQ
        if "spec" in filename_lower:
            return DocType.SPECIFICATIONS
        if "tender" in filename_lower or "nit" in filename_lower:
            return DocType.TENDER_NOTICE

        # Check content patterns
        best_match = DocType.UNKNOWN
        best_score = 0

        for doc_type, patterns in self.compiled_patterns.items():
            score = 0
            for pattern in patterns:
                if pattern.search(text_lower):
                    score += 1

            if score > best_score:
                best_score = score
                best_match = doc_type

        return best_match

    def _extract_tables(self, text: str) -> List[Dict]:
        """Extract table-like structures from text."""
        tables = []

        # Look for BOQ-style tables
        lines = text.split('\n')
        current_table = []
        in_table = False

        for line in lines:
            # Detect table start (header row)
            if re.search(r'(item|sr\.?\s*no|sl\.?\s*no).*description.*unit', line, re.IGNORECASE):
                in_table = True
                current_table = [line]
            elif in_table:
                # Check if line looks like table data
                if re.match(r'^\s*\d+', line) or re.search(r'\d+\.\d+|\d+,\d+', line):
                    current_table.append(line)
                elif len(line.strip()) < 5:
                    # Empty line might end table
                    if len(current_table) > 2:
                        tables.append({
                            "type": "boq",
                            "rows": current_table,
                        })
                    current_table = []
                    in_table = False

        # Don't forget last table
        if current_table and len(current_table) > 2:
            tables.append({
                "type": "boq",
                "rows": current_table,
            })

        return tables

    def get_documents_by_type(
        self, documents: List[ParsedDocument], doc_type: DocType
    ) -> List[ParsedDocument]:
        """Filter documents by type."""
        return [d for d in documents if d.doc_type == doc_type]
