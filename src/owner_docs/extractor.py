"""
Contract Extractor - Extract contract terms from owner documents.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .parser import ParsedDocument, DocType
from .tender import TenderInfo


@dataclass
class InclusionItem:
    """Single inclusion item."""
    description: str
    category: str = ""
    source_file: str = ""
    page_number: int = 0

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "category": self.category,
            "source_file": self.source_file,
            "page_number": self.page_number,
        }


@dataclass
class ExclusionItem:
    """Single exclusion item."""
    description: str
    category: str = ""
    impact: str = ""
    source_file: str = ""
    page_number: int = 0

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "category": self.category,
            "impact": self.impact,
            "source_file": self.source_file,
            "page_number": self.page_number,
        }


@dataclass
class LDClause:
    """Liquidated damages clause."""
    rate_percent: float
    period: str  # day/week/month
    max_percent: float
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "rate_percent": self.rate_percent,
            "period": self.period,
            "max_percent": self.max_percent,
            "notes": self.notes,
        }


@dataclass
class Milestone:
    """Payment milestone."""
    description: str
    timeline: str
    payment_percent: float

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "timeline": self.timeline,
            "payment_percent": self.payment_percent,
        }


@dataclass
class TestingRequirement:
    """Testing requirement."""
    item: str
    test_type: str
    frequency: str
    third_party: bool = False

    def to_dict(self) -> dict:
        return {
            "item": self.item,
            "test_type": self.test_type,
            "frequency": self.frequency,
            "third_party": self.third_party,
        }


@dataclass
class ContractTerms:
    """Extracted contract terms."""
    contract_type: Optional[str] = None
    gst_terms: Optional[str] = None
    payment_terms: Optional[str] = None
    retention_percent: Optional[float] = None
    dlp_months: Optional[int] = None
    ld_clause: Optional[LDClause] = None
    milestones: List[Milestone] = field(default_factory=list)
    inclusions: List[InclusionItem] = field(default_factory=list)
    exclusions: List[ExclusionItem] = field(default_factory=list)
    required_makes: Dict[str, List[str]] = field(default_factory=dict)
    testing_requirements: List[TestingRequirement] = field(default_factory=list)
    mom_clauses: List[str] = field(default_factory=list)
    insurance_requirements: List[str] = field(default_factory=list)
    alternates: List[Dict] = field(default_factory=list)
    allowances: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "contract_type": self.contract_type,
            "gst_terms": self.gst_terms,
            "payment_terms": self.payment_terms,
            "retention_percent": self.retention_percent,
            "dlp_months": self.dlp_months,
            "ld_clause": self.ld_clause.to_dict() if self.ld_clause else None,
            "milestones": [m.to_dict() for m in self.milestones],
            "inclusions_count": len(self.inclusions),
            "exclusions_count": len(self.exclusions),
            "required_makes": self.required_makes,
            "testing_requirements": [t.to_dict() for t in self.testing_requirements],
            "mom_clauses": self.mom_clauses[:10],
            "alternates": self.alternates,
            "allowances": self.allowances,
        }


class ContractExtractor:
    """Extract contract terms from documents."""

    # LD clause patterns
    LD_PATTERNS = [
        r'(?:ld|liquidated\s+damages?)\s*(?:@|at|of)?\s*([\d.]+)\s*%\s*(?:per|/)\s*(day|week|month)',
        r'([\d.]+)\s*%\s*(?:per|/)\s*(day|week|month)\s*(?:as\s+)?(?:ld|liquidated)',
        r'penalty\s*(?:@|at|of)?\s*([\d.]+)\s*%\s*(?:per|/)\s*(day|week|month)',
    ]

    LD_MAX_PATTERNS = [
        r'(?:max(?:imum)?|subject\s+to)\s*([\d.]+)\s*%',
        r'not\s+(?:to\s+)?exceed\s*([\d.]+)\s*%',
        r'(?:limited|capped)\s+(?:to|at)\s*([\d.]+)\s*%',
    ]

    # Retention patterns
    RETENTION_PATTERNS = [
        r'retention\s*(?:@|at|of)?\s*([\d.]+)\s*%',
        r'([\d.]+)\s*%\s*(?:as\s+)?retention',
        r'withhold\s*([\d.]+)\s*%',
    ]

    # DLP patterns
    DLP_PATTERNS = [
        r'(?:dlp|defect\s+liability)\s*(?:period)?\s*[:=]?\s*(\d+)\s*months?',
        r'(\d+)\s*months?\s*(?:dlp|defect\s+liability)',
        r'maintenance\s+period\s*[:=]?\s*(\d+)\s*months?',
    ]

    # Inclusion indicators
    INCLUSION_INDICATORS = [
        'shall include', 'includes', 'including', 'inclusive of',
        'work includes', 'scope includes', 'contractor shall',
        'following are included', 'in the scope',
    ]

    # Exclusion indicators
    EXCLUSION_INDICATORS = [
        'shall exclude', 'excludes', 'excluding', 'exclusive of',
        'not included', 'not in scope', 'owner shall provide',
        'outside scope', 'by others', 'not part of',
    ]

    def __init__(self):
        self.ld_regex = [re.compile(p, re.IGNORECASE) for p in self.LD_PATTERNS]
        self.ld_max_regex = [re.compile(p, re.IGNORECASE) for p in self.LD_MAX_PATTERNS]
        self.retention_regex = [re.compile(p, re.IGNORECASE) for p in self.RETENTION_PATTERNS]
        self.dlp_regex = [re.compile(p, re.IGNORECASE) for p in self.DLP_PATTERNS]

    def extract(
        self, documents: List[ParsedDocument], tender_info: Optional[TenderInfo]
    ) -> ContractTerms:
        """Extract contract terms from documents."""
        terms = ContractTerms()

        # Find relevant documents
        contract_docs = [d for d in documents if d.doc_type in [
            DocType.CONTRACT_CONDITIONS, DocType.TENDER_NOTICE,
            DocType.BID_FORM, DocType.SPECIFICATIONS
        ]]

        # Combine text
        combined_text = "\n\n".join(d.text_content for d in contract_docs)

        # Extract LD clause
        terms.ld_clause = self._extract_ld_clause(combined_text)

        # Extract retention
        terms.retention_percent = self._extract_retention(combined_text)

        # Extract DLP
        terms.dlp_months = self._extract_dlp(combined_text)

        # Extract GST terms
        terms.gst_terms = self._extract_gst_terms(combined_text)

        # Extract payment terms
        terms.payment_terms = self._extract_payment_terms(combined_text)

        # Extract contract type
        terms.contract_type = self._extract_contract_type(combined_text)

        # Extract milestones
        terms.milestones = self._extract_milestones(combined_text)

        # Extract inclusions
        terms.inclusions = self._extract_inclusions(documents)

        # Extract exclusions
        terms.exclusions = self._extract_exclusions(documents)

        # Extract required makes
        terms.required_makes = self._extract_required_makes(combined_text)

        # Extract testing requirements
        terms.testing_requirements = self._extract_testing(combined_text)

        # Extract MoM clauses
        terms.mom_clauses = self._extract_mom(combined_text)

        # Extract alternates
        terms.alternates = self._extract_alternates(combined_text)

        # Extract allowances
        terms.allowances = self._extract_allowances(combined_text)

        return terms

    def _extract_ld_clause(self, text: str) -> Optional[LDClause]:
        """Extract LD clause."""
        for pattern in self.ld_regex:
            match = pattern.search(text)
            if match:
                rate = float(match.group(1))
                period = match.group(2).lower()

                # Find max
                max_percent = 10.0  # Default
                for max_pattern in self.ld_max_regex:
                    max_match = max_pattern.search(text)
                    if max_match:
                        max_percent = float(max_match.group(1))
                        break

                return LDClause(
                    rate_percent=rate,
                    period=period,
                    max_percent=max_percent,
                    notes=match.group(0)[:100],
                )

        return None

    def _extract_retention(self, text: str) -> Optional[float]:
        """Extract retention percentage."""
        for pattern in self.retention_regex:
            match = pattern.search(text)
            if match:
                return float(match.group(1))
        return None

    def _extract_dlp(self, text: str) -> Optional[int]:
        """Extract DLP months."""
        for pattern in self.dlp_regex:
            match = pattern.search(text)
            if match:
                return int(match.group(1))
        return None

    def _extract_gst_terms(self, text: str) -> Optional[str]:
        """Extract GST terms."""
        patterns = [
            r'gst\s*(?:@|at)?\s*([\d.]+)\s*%',
            r'gst\s+(?:is\s+)?(?:included|inclusive)',
            r'gst\s+(?:is\s+)?(?:excluded|exclusive|extra)',
            r'plus\s+gst',
            r'(?:rate|amount)s?\s+(?:are\s+)?(?:inclusive|exclusive)\s+of\s+gst',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)

        return None

    def _extract_payment_terms(self, text: str) -> Optional[str]:
        """Extract payment terms."""
        patterns = [
            r'payment\s+(?:shall\s+be\s+made|terms?)\s*[:=]?\s*(.+?)(?:\n|$)',
            r'running\s+(?:account\s+)?bills?\s*[:=]?\s*(.+?)(?:\n|$)',
            r'(monthly|weekly|milestone[- ]?based)\s+(?:payment|billing)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1 if len(match.groups()) > 0 else 0).strip()[:200]

        return None

    def _extract_contract_type(self, text: str) -> Optional[str]:
        """Extract contract type."""
        types = {
            'item rate': ['item rate', 'unit rate', 'schedule of rates'],
            'lump sum': ['lump sum', 'lumpsum', 'fixed price'],
            'cost plus': ['cost plus', 'cost\+', 'reimbursement'],
            'epc': ['epc', 'turnkey', 'design build'],
            'percentage': ['percentage', 'percent basis'],
        }

        text_lower = text.lower()
        for contract_type, patterns in types.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return contract_type

        return None

    def _extract_milestones(self, text: str) -> List[Milestone]:
        """Extract payment milestones."""
        milestones = []

        # Look for milestone table patterns
        patterns = [
            r'(\d+)\s*%\s*(?:on|upon|at)\s+(.+?)(?:\n|$)',
            r'(?:milestone|stage)\s*(\d+)\s*[:=]?\s*(.+?)\s*-?\s*(\d+)\s*%',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    milestones.append(Milestone(
                        description=match[1] if len(match) > 2 else match[1],
                        timeline=match[1] if len(match) > 2 else "",
                        payment_percent=float(match[0] if match[0].isdigit() else match[2]),
                    ))

        return milestones[:15]  # Limit

    def _extract_inclusions(self, documents: List[ParsedDocument]) -> List[InclusionItem]:
        """Extract inclusions."""
        inclusions = []

        for doc in documents:
            lines = doc.text_content.split('\n')
            in_inclusion_section = False

            for i, line in enumerate(lines):
                line_lower = line.lower()

                # Check if entering inclusion section
                if any(ind in line_lower for ind in self.INCLUSION_INDICATORS):
                    in_inclusion_section = True

                    # Check if single line inclusion
                    if len(line) > 30:
                        inclusions.append(InclusionItem(
                            description=line.strip()[:200],
                            source_file=doc.file_name,
                            page_number=i // 50,  # Approximate
                        ))

                elif in_inclusion_section:
                    # Check for list item
                    if re.match(r'^[\d.)\-•]\s+', line):
                        clean = re.sub(r'^[\d.)\-•]\s+', '', line).strip()
                        if clean and len(clean) > 5:
                            inclusions.append(InclusionItem(
                                description=clean[:200],
                                source_file=doc.file_name,
                            ))
                    elif not line.strip():
                        in_inclusion_section = False

        return inclusions[:100]  # Limit

    def _extract_exclusions(self, documents: List[ParsedDocument]) -> List[ExclusionItem]:
        """Extract exclusions."""
        exclusions = []

        for doc in documents:
            lines = doc.text_content.split('\n')
            in_exclusion_section = False

            for i, line in enumerate(lines):
                line_lower = line.lower()

                # Check if entering exclusion section
                if any(ind in line_lower for ind in self.EXCLUSION_INDICATORS):
                    in_exclusion_section = True

                    if len(line) > 30:
                        exclusions.append(ExclusionItem(
                            description=line.strip()[:200],
                            impact="Verify scope boundary",
                            source_file=doc.file_name,
                        ))

                elif in_exclusion_section:
                    if re.match(r'^[\d.)\-•]\s+', line):
                        clean = re.sub(r'^[\d.)\-•]\s+', '', line).strip()
                        if clean and len(clean) > 5:
                            exclusions.append(ExclusionItem(
                                description=clean[:200],
                                impact="Excluded from contractor scope",
                                source_file=doc.file_name,
                            ))
                    elif not line.strip():
                        in_exclusion_section = False

        return exclusions[:50]  # Limit

    def _extract_required_makes(self, text: str) -> Dict[str, List[str]]:
        """Extract required makes/brands."""
        makes = {}

        patterns = [
            r'([a-z\s]+?)(?:\s+make|\s+brand)\s*[:=]?\s*([A-Za-z,/\s]+(?:\s+or\s+equivalent)?)',
            r'approved\s+makes?\s+(?:for\s+)?([a-z\s]+?)\s*[:=]?\s*([A-Za-z,/\s]+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for item, brand_list in matches:
                item = item.strip().lower()
                brands = [b.strip() for b in re.split(r'[,/]|\s+or\s+', brand_list)]
                brands = [b for b in brands if b and len(b) > 2]
                if brands:
                    makes[item] = brands

        return makes

    def _extract_testing(self, text: str) -> List[TestingRequirement]:
        """Extract testing requirements."""
        tests = []

        patterns = [
            r'(cube\s+test|slump\s+test|rebar\s+test|pressure\s+test|megger\s+test)',
            r'(?:testing|test)\s+(?:of|for)\s+([a-z\s]+)',
            r'third\s+party\s+(?:testing|inspection)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                tests.append(TestingRequirement(
                    item=match.strip() if match else "General",
                    test_type=match.strip(),
                    frequency="As per IS codes",
                    third_party="third party" in match.lower() if match else False,
                ))

        return tests[:20]  # Limit

    def _extract_mom(self, text: str) -> List[str]:
        """Extract method of measurement clauses."""
        mom_clauses = []

        patterns = [
            r'(?:method\s+of\s+measurement|mom)\s*[:=]?\s*(.+?)(?:\n|$)',
            r'(?:measurement|measuring)\s+shall\s+be\s+(.+?)(?:\n|$)',
            r'(?:no\s+)?deduction\s+(?:for|of)\s+(.+?)(?:\n|$)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.strip():
                    mom_clauses.append(match.strip()[:150])

        return mom_clauses[:20]

    def _extract_alternates(self, text: str) -> List[Dict]:
        """Extract alternate items."""
        alternates = []

        patterns = [
            r'alternate\s*(?:no\.?)?\s*(\d+)\s*[:=]?\s*(.+?)(?:\n|$)',
            r'option\s*(?:no\.?)?\s*(\d+)\s*[:=]?\s*(.+?)(?:\n|$)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for num, desc in matches:
                alternates.append({
                    "alternate_no": num,
                    "description": desc.strip()[:200],
                })

        return alternates

    def _extract_allowances(self, text: str) -> List[Dict]:
        """Extract allowances."""
        allowances = []

        patterns = [
            r'allowance\s+(?:for|of)\s+(.+?)\s*[:=]?\s*(?:rs\.?|₹)?\s*([\d,]+)',
            r'provisional\s+sum\s+(?:for|of)\s+(.+?)\s*[:=]?\s*(?:rs\.?|₹)?\s*([\d,]+)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for item, amount in matches:
                allowances.append({
                    "item": item.strip(),
                    "amount": float(amount.replace(",", "")),
                })

        return allowances
