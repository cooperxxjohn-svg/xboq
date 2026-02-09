"""
Reporting Module for Structural Estimator
Provides structured output formats, table builders, and Excel export.

Modules:
- estimator_output: Data schema and output builder
- boq_builder: BOQ generation with dependency tracking
- scope_checklist: Scope tracking (detected vs missing)
- requirements_extractor: Extract requirements from notes
- exporter: Excel and JSON export
- synonyms_india: Indian construction term mappings
- scope_completion: High recall scope completion engine
- text_extractor: Multi-region text extraction
"""

from .estimator_output import (
    EstimatorOutput,
    ProjectInfo,
    MaterialSpecs,
    ColumnEntry,
    FootingEntry,
    BeamEntry,
    BOQItem,
    ScopeChecklist,
    Requirement,
    AssumptionEntry,
    QtyStatus,
    ConfidenceLevel,
    build_estimator_output
)

from .boq_builder import (
    build_boq_items,
    attach_boq_items
)

from .scope_checklist import (
    build_scope_checklist,
    attach_scope_checklist
)

from .requirements_extractor import (
    extract_requirements,
    extract_requirements_from_foundation_data,
    attach_requirements
)

from .exporter import (
    build_columns_df,
    build_footings_df,
    build_beams_df,
    build_requirements_df,
    build_scope_df,
    build_boq_df,
    build_assumptions_df,
    build_summary_df,
    export_to_excel,
    export_to_json
)

from .synonyms_india import (
    INDIAN_SYNONYMS,
    SYNONYM_TO_STANDARD,
    BOQ_TEMPLATES,
    normalize_term,
    find_synonyms,
    detect_terms_in_text,
    get_probable_items_from_text,
    expand_search_terms
)

from .scope_completion import (
    complete_scope,
    generate_missing_items_report,
    EnhancedBOQItem,
    ItemEvidence,
    InferenceRule,
    INFERENCE_RULES
)

from .text_extractor import (
    ExtractedText,
    TextRegion,
    extract_notes_and_schedules,
    extract_from_pdf_page,
    extract_section_from_text,
    find_region_by_header,
    merge_extracted_text
)

__all__ = [
    # Output schema
    'EstimatorOutput',
    'ProjectInfo',
    'MaterialSpecs',
    'ColumnEntry',
    'FootingEntry',
    'BeamEntry',
    'BOQItem',
    'ScopeChecklist',
    'Requirement',
    'AssumptionEntry',
    'QtyStatus',
    'ConfidenceLevel',
    'build_estimator_output',

    # BOQ builder
    'build_boq_items',
    'attach_boq_items',

    # Scope checklist
    'build_scope_checklist',
    'attach_scope_checklist',

    # Requirements extractor
    'extract_requirements',
    'extract_requirements_from_foundation_data',
    'attach_requirements',

    # Exporter / Table builders
    'build_columns_df',
    'build_footings_df',
    'build_beams_df',
    'build_requirements_df',
    'build_scope_df',
    'build_boq_df',
    'build_assumptions_df',
    'build_summary_df',
    'export_to_excel',
    'export_to_json',

    # Indian synonyms
    'INDIAN_SYNONYMS',
    'SYNONYM_TO_STANDARD',
    'BOQ_TEMPLATES',
    'normalize_term',
    'find_synonyms',
    'detect_terms_in_text',
    'get_probable_items_from_text',
    'expand_search_terms',

    # Scope completion
    'complete_scope',
    'generate_missing_items_report',
    'EnhancedBOQItem',
    'ItemEvidence',
    'InferenceRule',
    'INFERENCE_RULES',

    # Text extractor
    'ExtractedText',
    'TextRegion',
    'extract_notes_and_schedules',
    'extract_from_pdf_page',
    'extract_section_from_text',
    'find_region_by_header',
    'merge_extracted_text'
]
