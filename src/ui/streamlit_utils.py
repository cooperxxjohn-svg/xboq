"""
XBOQ Streamlit UI Utilities
Prevents PyArrow/Streamlit crashes and provides consistent UI components.
"""

import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Optional


# =============================================================================
# DATAFRAME SANITIZATION
# =============================================================================

def sanitize_df_for_streamlit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize a DataFrame for Streamlit display to prevent Arrow conversion errors.

    Handles:
    - Mixed types in columns (strings in numeric columns)
    - Lone '-' characters that cause Arrow errors
    - NaN/None values
    - Special characters

    Args:
        df: Input DataFrame (may contain mixed types)

    Returns:
        Sanitized DataFrame safe for st.dataframe()
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    df = df.copy()

    for col in df.columns:
        # Replace problematic values with None first
        df[col] = df[col].replace(['-', '—', '–', '', ' ', 'nan', 'None', 'NaN'], None)

        # Try numeric conversion safely (will keep strings as strings)
        df[col] = pd.to_numeric(df[col], errors='ignore')

        # Ensure all values are either numeric or string (no mixed object types)
        if df[col].dtype == 'object':
            # Convert everything to string, handling None
            df[col] = df[col].apply(lambda x: str(x) if x is not None and pd.notna(x) else '')

    return df


def df_to_styled(
    df: pd.DataFrame,
    status_column: Optional[str] = None,
    confidence_column: Optional[str] = None
):
    """
    Apply consistent styling to a DataFrame.

    Args:
        df: DataFrame to style
        status_column: Column name containing status values for row coloring
        confidence_column: Column name containing confidence values

    Returns:
        Styled DataFrame
    """
    df = sanitize_df_for_streamlit(df)

    def highlight_status(row):
        """Color rows based on status."""
        if status_column and status_column in row.index:
            status = str(row[status_column]).lower()
            if status in ['computed', 'detected', 'high']:
                return ['background-color: #d4edda'] * len(row)
            elif status in ['partial', 'inferred', 'med', 'needs_review']:
                return ['background-color: #fff3cd'] * len(row)
            elif status in ['unknown', 'missing', 'low']:
                return ['background-color: #f8d7da'] * len(row)
        return [''] * len(row)

    def highlight_confidence(val):
        """Color confidence values."""
        try:
            # Handle percentage strings
            if isinstance(val, str) and '%' in val:
                val = float(val.replace('%', '')) / 100
            val = float(val)
            if val >= 0.8:
                return 'background-color: #d4edda'
            elif val >= 0.5:
                return 'background-color: #fff3cd'
            else:
                return 'background-color: #f8d7da'
        except:
            return ''

    styled = df.style

    if status_column and status_column in df.columns:
        styled = styled.apply(highlight_status, axis=1)

    if confidence_column and confidence_column in df.columns:
        styled = styled.applymap(highlight_confidence, subset=[confidence_column])

    return styled


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_confidence_badge(confidence: float) -> str:
    """Render a confidence badge as HTML."""
    if confidence >= 0.8:
        color = "#28a745"
        icon = "🟢"
    elif confidence >= 0.5:
        color = "#ffc107"
        icon = "🟡"
    else:
        color = "#dc3545"
        icon = "🔴"

    return f'{icon} {confidence:.0%}'


def render_source_badge(source: str) -> str:
    """Render a source badge as HTML."""
    badges = {
        "explicit": ("📋", "#28a745", "Detected"),
        "synonym": ("🔤", "#17a2b8", "Synonym"),
        "inferred": ("🔍", "#6c757d", "Inferred"),
        "camelot": ("📊", "#007bff", "Schedule"),
        "heuristic": ("🔎", "#6610f2", "Visual"),
        "pdf_text": ("📝", "#20c997", "Text"),
        "ocr": ("👁️", "#fd7e14", "OCR"),
    }

    icon, color, label = badges.get(source.lower(), ("❓", "#6c757d", source))
    return f'{icon} {label}'


def render_status_badge(status: str) -> str:
    """Render a status badge."""
    badges = {
        "detected": ("✅", "#28a745"),
        "inferred": ("🔍", "#ffc107"),
        "missing": ("❌", "#dc3545"),
        "needs_review": ("⚠️", "#fd7e14"),
        "computed": ("✅", "#28a745"),
        "partial": ("⚠️", "#ffc107"),
        "unknown": ("❓", "#dc3545"),
    }

    icon, color = badges.get(status.lower(), ("❓", "#6c757d"))
    return f'{icon} {status.title()}'


def render_severity_badge(severity: str) -> str:
    """Render a severity badge."""
    badges = {
        "low": ("🟢", "#28a745"),
        "med": ("🟡", "#ffc107"),
        "high": ("🔴", "#dc3545"),
    }

    icon, color = badges.get(severity.lower(), ("❓", "#6c757d"))
    return f'{icon} {severity.upper()}'


def render_needs_review_banner(
    message: str = "This takeoff needs human review",
    reasons: Optional[List[str]] = None
):
    """Render a 'needs review' warning banner."""
    st.warning(f"⚠️ **{message}**")

    if reasons:
        with st.expander("Why review is needed"):
            for reason in reasons:
                st.write(f"- {reason}")


def render_evidence_viewer(evidence_list: List[Dict[str, Any]], title: str = "Evidence"):
    """
    Render an evidence viewer component.

    Args:
        evidence_list: List of Evidence objects or dicts
        title: Section title
    """
    if not evidence_list:
        st.caption("No evidence available")
        return

    with st.expander(f"🔍 {title} ({len(evidence_list)} sources)"):
        for i, ev in enumerate(evidence_list):
            # Handle both Evidence objects and dicts
            if hasattr(ev, 'page'):
                page = ev.page
                source = ev.source.value if hasattr(ev.source, 'value') else str(ev.source)
                snippet = ev.snippet
            else:
                page = ev.get('page', 0)
                source = ev.get('source', 'unknown')
                snippet = ev.get('snippet', '')

            col1, col2 = st.columns([1, 4])
            with col1:
                st.caption(f"Page {page + 1}")
                st.caption(render_source_badge(source))
            with col2:
                if snippet:
                    st.code(snippet[:200] + "..." if len(snippet) > 200 else snippet)
                else:
                    st.caption("No snippet available")

            if i < len(evidence_list) - 1:
                st.divider()


# =============================================================================
# DATA CONVERSION HELPERS
# =============================================================================

def boq_items_to_df(boq_items: List[Any], include_confidence_reason: bool = True) -> pd.DataFrame:
    """Convert BOQ items to DataFrame for display."""
    if not boq_items:
        return pd.DataFrame()

    rows = []
    for item in boq_items:
        # Handle both BOQItem objects and dicts
        if hasattr(item, 'to_export_dict'):
            row = item.to_export_dict()
            rows.append(row)
        elif hasattr(item, 'item_name'):
            row = {
                "S.No": getattr(item, 'id', ''),
                "System": getattr(item, 'system', '').title(),
                "Subsystem": getattr(item, 'subsystem', '').title(),
                "Description": item.item_name,
                "Unit": getattr(item, 'unit', '-') or '-',
                "Qty": item.qty if item.qty is not None else '-',
                "Qty Status": getattr(item, 'qty_status', 'unknown'),
                "Dependencies": ", ".join(getattr(item, 'dependencies', [])) or '-',
                "Confidence": f"{getattr(item, 'confidence', 0):.0%}",
                "Source": getattr(item, 'source', 'explicit'),
            }
            # Add confidence_reason if available
            if include_confidence_reason and hasattr(item, 'confidence_reason'):
                row["Confidence Reason"] = getattr(item, 'confidence_reason', '') or '-'
            rows.append(row)
        else:
            # Dict format
            rows.append({
                "S.No": item.get('id', ''),
                "Description": item.get('item_name', ''),
                "Unit": item.get('unit', '-'),
                "Qty": item.get('qty', '-'),
                "Qty Status": item.get('qty_status', 'unknown'),
            })

    return sanitize_df_for_streamlit(pd.DataFrame(rows))


def scope_items_to_df(scope_items: List[Any], include_confidence_reason: bool = True) -> pd.DataFrame:
    """Convert scope items to DataFrame for display."""
    if not scope_items:
        return pd.DataFrame()

    rows = []
    for item in scope_items:
        if hasattr(item, 'category'):
            row = {
                "Category": item.category.value.title() if hasattr(item.category, 'value') else str(item.category),
                "Trade": item.trade,
                "Status": item.status.value if hasattr(item.status, 'value') else str(item.status),
                "Reason": item.reason,
                "Confidence": f"{item.confidence:.0%}",
                "Pages": ", ".join(str(p + 1) for p in item.pages_found) if item.pages_found else "-",
            }
            # Add confidence_reason if available
            if include_confidence_reason and hasattr(item, 'confidence_reason'):
                row["Confidence Reason"] = getattr(item, 'confidence_reason', '') or '-'
            rows.append(row)
        else:
            rows.append({
                "Category": item.get('category', ''),
                "Trade": item.get('trade', ''),
                "Status": item.get('status', ''),
                "Confidence": item.get('confidence', ''),
            })

    return sanitize_df_for_streamlit(pd.DataFrame(rows))


def conflicts_to_df(conflicts: List[Any], include_evidence: bool = True) -> pd.DataFrame:
    """Convert conflicts to DataFrame for display."""
    if not conflicts:
        return pd.DataFrame()

    rows = []
    for c in conflicts:
        if hasattr(c, 'type'):
            row = {
                "Type": c.type.value.replace('_', ' ').title() if hasattr(c.type, 'value') else str(c.type),
                "Severity": c.severity.value.upper() if hasattr(c.severity, 'value') else str(c.severity),
                "Description": c.description,
                "Resolution": c.suggested_resolution,
            }
            # Add evidence snippet if available
            if include_evidence and hasattr(c, 'evidence') and c.evidence:
                snippets = []
                for ev in c.evidence[:2]:  # Limit to first 2 evidence items
                    if hasattr(ev, 'snippet') and ev.snippet:
                        snippets.append(ev.snippet[:100])
                row["Evidence"] = " | ".join(snippets) if snippets else "-"
            rows.append(row)
        else:
            rows.append({
                "Type": c.get('type', ''),
                "Severity": c.get('severity', ''),
                "Description": c.get('description', ''),
            })

    return sanitize_df_for_streamlit(pd.DataFrame(rows))


def coverage_to_df(coverage: List[Any], boq_items: List[Any], include_breakdown: bool = True) -> pd.DataFrame:
    """Convert coverage records to DataFrame for display."""
    if not coverage:
        return pd.DataFrame()

    # Build BOQ lookup - handle both pydantic models and dicts
    boq_lookup = {}
    for item in boq_items:
        if hasattr(item, 'id'):
            # Pydantic model
            item_id = item.id
            item_name = item.item_name
        elif isinstance(item, dict):
            # Dict
            item_id = item.get('id', '')
            item_name = item.get('item_name', '')
        else:
            continue
        boq_lookup[item_id] = item_name

    rows = []
    for c in coverage:
        if hasattr(c, 'boq_item_id'):
            sources = [s.value if hasattr(s, 'value') else str(s) for s in c.sources_used]
            boq_name = boq_lookup.get(c.boq_item_id, c.boq_item_id)
            row = {
                "BOQ Item": boq_name[:50] if boq_name else "-",
                "Coverage Score": f"{c.coverage_score:.0%}",
                "Pages Used": ", ".join(str(p + 1) for p in c.pages_used) if c.pages_used else "-",
                "Sources": ", ".join(sources) if sources else "-",
                "Evidence Count": len(c.contributed_by),
            }

            # Add coverage breakdown if available
            if include_breakdown and hasattr(c, 'coverage_breakdown') and c.coverage_breakdown:
                breakdown_parts = []
                for name, value in c.coverage_breakdown.items():
                    sign = "+" if value >= 0 else ""
                    breakdown_parts.append(f"{name}={sign}{value:.2f}")
                row["Breakdown"] = " | ".join(breakdown_parts) if breakdown_parts else "-"

            rows.append(row)

    return sanitize_df_for_streamlit(pd.DataFrame(rows))
