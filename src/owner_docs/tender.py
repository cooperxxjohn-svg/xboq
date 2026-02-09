"""
Tender Parser - Extract tender/NIT information.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

from .parser import ParsedDocument, DocType


@dataclass
class TenderInfo:
    """Extracted tender information."""
    reference: Optional[str] = None
    project_name: Optional[str] = None
    owner_name: Optional[str] = None
    location: Optional[str] = None
    submission_date: Optional[str] = None
    submission_time: Optional[str] = None
    opening_date: Optional[str] = None
    completion_months: Optional[int] = None
    emd_amount: Optional[float] = None
    tender_fee: Optional[float] = None
    estimated_cost: Optional[float] = None
    eligibility_criteria: List[str] = field(default_factory=list)
    contact_info: Dict[str, str] = field(default_factory=dict)
    pre_bid_date: Optional[str] = None
    source_file: str = ""
    source_page: int = 0

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "project_name": self.project_name,
            "owner_name": self.owner_name,
            "location": self.location,
            "submission_date": self.submission_date,
            "submission_time": self.submission_time,
            "opening_date": self.opening_date,
            "completion_months": self.completion_months,
            "emd_amount": self.emd_amount,
            "tender_fee": self.tender_fee,
            "estimated_cost": self.estimated_cost,
            "eligibility_criteria": self.eligibility_criteria,
            "contact_info": self.contact_info,
            "pre_bid_date": self.pre_bid_date,
            "source_file": self.source_file,
        }


class TenderParser:
    """Extract tender information from documents."""

    # Patterns for extracting tender info
    PATTERNS = {
        "reference": [
            r'tender\s*(?:no|number|ref)\.?\s*[:=]?\s*([A-Z0-9/-]+)',
            r'nit\s*(?:no|number)\.?\s*[:=]?\s*([A-Z0-9/-]+)',
            r'ref(?:erence)?\.?\s*(?:no)?\.?\s*[:=]?\s*([A-Z0-9/-]+)',
        ],
        "project_name": [
            r'name\s+of\s+(?:work|project)\s*[:=]?\s*(.+?)(?:\n|$)',
            r'project\s*[:=]?\s*(.+?)(?:\n|$)',
            r'subject\s*[:=]?\s*(.+?)(?:\n|$)',
        ],
        "owner_name": [
            r'(?:employer|owner|client|organization)\s*[:=]?\s*(.+?)(?:\n|$)',
            r'issued\s+by\s*[:=]?\s*(.+?)(?:\n|$)',
        ],
        "location": [
            r'location\s*[:=]?\s*(.+?)(?:\n|$)',
            r'site\s*[:=]?\s*(.+?)(?:\n|$)',
            r'at\s+([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)?)',
        ],
        "submission_date": [
            r'submission\s+(?:date|deadline)\s*[:=]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'last\s+date\s*[:=]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'due\s+(?:date|on)\s*[:=]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(?:on\s+or\s+before|before)\s*[:=]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ],
        "completion_months": [
            r'completion\s+(?:period|time)\s*[:=]?\s*(\d+)\s*months?',
            r'duration\s*[:=]?\s*(\d+)\s*months?',
            r'(\d+)\s*months?\s+(?:from|of)\s+(?:date|start)',
            r'time\s+(?:of|for)\s+completion\s*[:=]?\s*(\d+)',
        ],
        "emd_amount": [
            r'emd\s*[:=]?\s*(?:rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)',
            r'earnest\s+money\s*[:=]?\s*(?:rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)',
            r'bid\s+security\s*[:=]?\s*(?:rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)',
        ],
        "tender_fee": [
            r'tender\s+fee\s*[:=]?\s*(?:rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)',
            r'cost\s+of\s+tender\s*[:=]?\s*(?:rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)',
            r'document\s+fee\s*[:=]?\s*(?:rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)',
        ],
        "estimated_cost": [
            r'estimated\s+cost\s*[:=]?\s*(?:rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)\s*(?:lakh|lac|crore)?',
            r'approximate\s+(?:cost|value)\s*[:=]?\s*(?:rs\.?|₹)?\s*([\d,]+(?:\.\d+)?)',
        ],
        "pre_bid_date": [
            r'pre[- ]?bid\s+(?:meeting|conference)\s*[:=]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        ],
    }

    def __init__(self):
        self.compiled_patterns = {}
        for field_name, patterns in self.PATTERNS.items():
            self.compiled_patterns[field_name] = [
                re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns
            ]

    def extract(self, documents: List[ParsedDocument]) -> Optional[TenderInfo]:
        """Extract tender info from documents."""
        # Find tender notice documents
        tender_docs = [d for d in documents if d.doc_type == DocType.TENDER_NOTICE]

        # Also check other docs if no dedicated tender notice
        if not tender_docs:
            tender_docs = [d for d in documents if d.doc_type in [
                DocType.BID_FORM, DocType.CONTRACT_CONDITIONS
            ]]

        if not tender_docs:
            return None

        # Combine text from all tender docs
        combined_text = "\n\n".join(d.text_content for d in tender_docs)

        tender_info = TenderInfo()
        if tender_docs:
            tender_info.source_file = tender_docs[0].file_name

        # Extract each field
        for field_name, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(combined_text)
                if match:
                    value = match.group(1).strip()

                    if field_name in ["emd_amount", "tender_fee", "estimated_cost"]:
                        # Parse numeric value
                        value = self._parse_amount(value, combined_text, match)
                    elif field_name == "completion_months":
                        value = int(value)

                    setattr(tender_info, field_name, value)
                    break

        # Extract eligibility criteria
        tender_info.eligibility_criteria = self._extract_eligibility(combined_text)

        return tender_info

    def _parse_amount(self, value: str, full_text: str, match) -> float:
        """Parse amount string to float."""
        # Remove commas
        value = value.replace(",", "")

        try:
            amount = float(value)

            # Check for lakh/crore multiplier in context
            context = full_text[match.start():match.end() + 20].lower()
            if "crore" in context:
                amount *= 10000000
            elif "lakh" in context or "lac" in context:
                amount *= 100000

            return amount
        except ValueError:
            return 0.0

    def _extract_eligibility(self, text: str) -> List[str]:
        """Extract eligibility criteria."""
        criteria = []

        # Look for eligibility section
        eligibility_match = re.search(
            r'eligibility\s*(?:criteria|requirements?)?\s*[:=]?\s*(.+?)(?=\n\n|\n[A-Z])',
            text,
            re.IGNORECASE | re.DOTALL
        )

        if eligibility_match:
            section = eligibility_match.group(1)

            # Split into individual criteria
            for line in section.split('\n'):
                line = line.strip()
                if line and len(line) > 10:
                    # Clean up numbering
                    line = re.sub(r'^[\d.)\s]+', '', line)
                    if line:
                        criteria.append(line)

        # Also look for specific patterns
        patterns = [
            r'minimum\s+annual\s+turnover\s*[:=]?\s*(.+?)(?:\n|$)',
            r'experience\s+(?:of|in)\s+similar\s+(?:work|project)s?\s*[:=]?\s*(.+?)(?:\n|$)',
            r'(?:valid\s+)?registration\s+(?:with|under)\s+(.+?)(?:\n|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                criteria.append(match.group(0).strip())

        return criteria[:10]  # Limit
