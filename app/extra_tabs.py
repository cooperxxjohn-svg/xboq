"""
Sprint 46 — Extra Tabs
======================
Render functions for tabs 22–30 added in Sprint 46.
Each function is called inside a `with preview_tabs[N]:` block in demo_page.py.

Tabs:
  22 — 📁 Projects     (multi-project workspace)
  23 — 📄 Bid Report   (PDF / Word / Bid Packet download)
  24 — 🏗️ Prelims      (site staff, equipment, facilities BOQ)
  25 — 📬 Scope Pkgs   (subcontractor scope packages)
  26 — 🔄 Addenda      (revision / amendment tracker)
  27 — 🔍 Reconcile    (drawing-vs-owner BOQ cross-check)
  28 — 💵 Cash Flow    (S-curve & monthly spend)
  29 — 📊 Benchmark    (eval harness accuracy dashboard)
  30 — 🔀 Compare      (multi-tender side-by-side)
"""

from __future__ import annotations

import io
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────
_APP_DIR = Path(__file__).resolve().parent
_ROOT = _APP_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _inr(v: float) -> str:
    if v >= 1e7:  return f"₹{v/1e7:.2f} Cr"
    if v >= 1e5:  return f"₹{v/1e5:.1f} L"
    return f"₹{v:,.0f}"


def _badge(label: str, color: str = "#334155") -> str:
    return (
        f'<span style="background:{color};color:#f1f5f9;'
        f'border-radius:4px;padding:2px 8px;font-size:0.78rem;'
        f'font-weight:600">{label}</span>'
    )


def _section(title: str) -> None:
    st.markdown(f"#### {title}")


# =============================================================================
# Tab 22 — 📁 Projects
# =============================================================================

def render_projects_tab(payload: dict) -> None:
    """Multi-project workspace: list, create, and link analysis runs."""
    try:
        from analysis.projects import (
            create_project, list_projects, load_project,
            save_run, list_runs,
        )
    except ImportError:
        st.info("Projects module not available.")
        return

    st.markdown("### 📁 Project Workspace")
    st.caption("Track multiple tenders. Each project stores its analysis runs for comparison.")

    # ── Link current run to a project ────────────────────────────────────────
    pid = payload.get("project_id", "")
    if pid:
        st.info(f"Current analysis: **{pid}**")
        all_projs = list_projects()
        if all_projs:
            names = [f"{p['name']} ({p['project_id'][:8]}…)" for p in all_projs]
            chosen = st.selectbox("Link this run to an existing project:", ["— skip —"] + names, key="tab22_link")
            if chosen != "— skip —":
                idx = names.index(chosen)
                proj_id = all_projs[idx]["project_id"]
                if st.button("💾 Save run to project", key="tab22_save_run"):
                    try:
                        save_run(
                            project_id=proj_id,
                            run_id=pid,
                            payload_path="",
                            run_metadata={
                                "readiness_score": payload.get("readiness_score", 0),
                                "decision": payload.get("decision", ""),
                                "rfi_count": len(payload.get("rfis", [])),
                            },
                        )
                        st.success("Run linked to project ✓")
                    except Exception as e:
                        st.error(f"Could not save run: {e}")

    st.divider()

    col_list, col_new = st.columns([2, 1])

    # ── Project list ──────────────────────────────────────────────────────────
    with col_list:
        _section("Your Projects")
        projects = list_projects()
        if not projects:
            st.info("No projects yet. Create one on the right →")
        for proj in projects:
            score_badge = ""
            runs = list_runs(proj["project_id"])
            last_score = None
            if runs:
                last_score = runs[0].get("readiness_score")
                col = "#22c55e" if last_score and last_score >= 75 else "#f59e0b" if last_score and last_score >= 50 else "#ef4444"
                score_badge = _badge(f"Score {last_score}", col)

            with st.expander(
                f"📋 {proj['name']}  •  {proj.get('bid_date','No date')}  •  {len(runs)} run(s)",
                expanded=False,
            ):
                c1, c2 = st.columns(2)
                c1.markdown(f"**Owner/Client:** {proj.get('owner','—')}")
                c1.markdown(f"**Created:** {proj.get('created_at','')[:10]}")
                c2.markdown(f"**Trades in scope:** {', '.join(proj.get('trades_in_scope') or []) or 'All'}")
                c2.markdown(f"**Notes:** {proj.get('notes','—')[:80]}")
                if score_badge:
                    st.markdown(score_badge, unsafe_allow_html=True)

                if runs:
                    st.markdown("**Run history:**")
                    for run in runs[:5]:
                        sc = run.get("readiness_score", "—")
                        dec = run.get("decision", "")
                        ts  = run.get("timestamp", "")[:16]
                        st.markdown(f"- `{ts}` — Score **{sc}** {dec}")

    # ── Create new project ────────────────────────────────────────────────────
    with col_new:
        _section("New Project")
        with st.form("tab22_new_proj"):
            name     = st.text_input("Project Name *")
            owner    = st.text_input("Client / Owner")
            bid_date = st.text_input("Bid Date (e.g. 2026-04-15)")
            notes    = st.text_area("Notes", height=80)
            trades   = st.multiselect(
                "Trades in scope",
                ["Civil", "Structural", "MEP", "Finishes", "Electrical",
                 "Plumbing", "HVAC", "Facade", "Landscaping", "Prelims"],
            )
            submitted = st.form_submit_button("Create Project ➕")
            if submitted:
                if not name.strip():
                    st.error("Project name is required.")
                else:
                    try:
                        proj = create_project(
                            name=name.strip(),
                            owner=owner.strip(),
                            bid_date=bid_date.strip(),
                            notes=notes.strip(),
                            trades_in_scope=trades or None,
                        )
                        st.success(f"✓ Created: {proj['project_id'][:12]}…")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")


# =============================================================================
# Tab 23 — 📄 Bid Report
# =============================================================================

def render_bid_report_tab(payload: dict) -> None:
    """One-click export: PDF summary, RFI Word doc, Bid Packet HTML/ZIP."""
    st.markdown("### 📄 Bid Report Exports")
    st.caption("Generate professional documents from the current analysis.")

    pid   = payload.get("project_id", "project")
    rfis  = payload.get("rfis", [])
    trade_summary = {
        t.get("trade", t.get("name", "General")): t
        for t in payload.get("trade_coverage", [])
        if isinstance(t, dict)
    }
    bid_gate = {
        "status": payload.get("decision", "REVIEW"),
        "score":  payload.get("readiness_score", 0),
        "blockers": [b.get("title", str(b)) for b in payload.get("blockers", [])[:5]],
    }
    metrics = {
        "pages":    payload.get("processing_stats", {}).get("total_pages", 0),
        "rooms":    len(payload.get("qto_rooms", [])),
        "openings": len(payload.get("line_items", [])),
    }

    # ── PDF Summary ───────────────────────────────────────────────────────────
    st.markdown("#### 📕 PDF Bid Summary")
    st.caption("2-page A4 report: readiness scorecard, trade coverage, top RFIs.")
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("Generate PDF", key="tab23_pdf"):
            try:
                from export.pdf_export import export_pdf_summary
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
                    out = export_pdf_summary(payload, Path(tf.name))
                with open(out, "rb") as f:
                    pdf_bytes = f.read()
                st.download_button(
                    "⬇ Download PDF",
                    data=pdf_bytes,
                    file_name=f"bid_summary_{pid}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except ImportError:
                st.error("reportlab not installed. Run: pip install reportlab")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
                st.code(traceback.format_exc())

    st.divider()

    # ── RFI Word Document ─────────────────────────────────────────────────────
    st.markdown("#### 📘 RFI Word Document")
    st.caption(f"Formal .docx with all {len(rfis)} RFIs, sorted by priority.")
    if st.button("Generate Word Doc", key="tab23_word"):
        try:
            from export.word_export import export_rfi_word
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tf:
                out = export_rfi_word(payload, Path(tf.name))
            with open(out, "rb") as f:
                docx_bytes = f.read()
            st.download_button(
                "⬇ Download .docx",
                data=docx_bytes,
                file_name=f"rfis_{pid}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        except ImportError:
            st.error("python-docx not installed. Run: pip install python-docx")
        except Exception as e:
            st.error(f"Word generation failed: {e}")

    st.divider()

    # ── Bid Packet (HTML + ZIP) ───────────────────────────────────────────────
    st.markdown("#### 📦 Bid Readiness Packet")
    st.caption("Full HTML report + JSON bundle (RFIs, assumptions, summary).")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬇ Download HTML Report", key="tab23_html"):
            try:
                from exports.bid_packet import get_bid_packet_buffer
                buf = get_bid_packet_buffer(pid, rfis, trade_summary, bid_gate, metrics, format="html")
                st.download_button(
                    "⬇ bid_readiness.html",
                    data=buf,
                    file_name=f"bid_readiness_{pid}.html",
                    mime="text/html",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Error: {e}")
    with c2:
        if st.button("⬇ Download ZIP Bundle", key="tab23_zip"):
            try:
                from exports.bid_packet import get_bid_packet_buffer
                buf = get_bid_packet_buffer(pid, rfis, trade_summary, bid_gate, metrics, format="zip")
                st.download_button(
                    "⬇ bid_packet.zip",
                    data=buf,
                    file_name=f"bid_packet_{pid}.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # ── RFI Pack (CSV + Email drafts) ─────────────────────────────────────────
    st.markdown("#### 📋 RFI Tracker Pack")
    st.caption("CSV tracker, per-trade email drafts, HTML print view.")
    c1, c2, c3 = st.columns(3)
    with c1:
        try:
            from exports.rfi_pack import get_rfi_pack_buffer
            buf = get_rfi_pack_buffer(rfis, trade_summary, pid, format="csv")
            st.download_button("⬇ RFI Tracker CSV", data=buf,
                file_name=f"rfi_tracker_{pid}.csv", mime="text/csv",
                use_container_width=True)
        except Exception as e:
            st.error(f"CSV: {e}")
    with c2:
        try:
            from exports.rfi_pack import get_rfi_pack_buffer
            buf = get_rfi_pack_buffer(rfis, trade_summary, pid, format="txt")
            st.download_button("⬇ Email Draft .txt", data=buf,
                file_name=f"rfi_email_{pid}.txt", mime="text/plain",
                use_container_width=True)
        except Exception as e:
            st.error(f"TXT: {e}")
    with c3:
        try:
            from exports.rfi_pack import get_rfi_pack_buffer
            buf = get_rfi_pack_buffer(rfis, trade_summary, pid, format="html")
            st.download_button("⬇ RFI Pack HTML", data=buf,
                file_name=f"rfi_pack_{pid}.html", mime="text/html",
                use_container_width=True)
        except Exception as e:
            st.error(f"HTML: {e}")

    st.divider()

    # ── Email Drafts ──────────────────────────────────────────────────────────
    st.markdown("#### ✉️ Email Drafts")
    st.caption("Pre-written RFI emails per trade and exclusion/clarification notice.")
    try:
        from exports.email_drafts import generate_all_email_drafts
        assumptions = payload.get("assumptions", [])
        drafts = generate_all_email_drafts(
            rfis=rfis,
            assumptions=assumptions,
            project_name=pid,
        )
        for fname, content in drafts.items():
            with st.expander(f"📧 {fname}"):
                st.code(content, language="")
                st.download_button(
                    f"⬇ {fname}",
                    data=content.encode(),
                    file_name=fname,
                    mime="text/plain",
                    key=f"tab23_email_{fname}",
                )
    except Exception as e:
        st.warning(f"Email drafts unavailable: {e}")


# =============================================================================
# Tab 24 — 🏗️ Prelims
# =============================================================================

def render_prelims_tab(payload: dict) -> None:
    """Preliminary cost estimator: staff, equipment, site facilities."""
    st.markdown("### 🏗️ Prelims BOQ")
    st.caption("Site establishment, staff, equipment and facilities cost estimate.")

    # ── Inputs ────────────────────────────────────────────────────────────────
    area_sqm   = payload.get("plan_graph", {}).get("floor_area_sqm", 0) or 0
    n_floors   = payload.get("plan_graph", {}).get("floors", 4) or 4
    boq_items  = payload.get("line_items", payload.get("boq_items", []))

    # Estimate project value from BOQ
    proj_value_est = sum(
        float(i.get("amount", 0) or i.get("total_amount", 0) or 0)
        for i in boq_items
    )
    if proj_value_est < 1e5:
        proj_value_est = 5_000_000.0  # default 50L

    col_inp, col_out = st.columns([1, 2])
    with col_inp:
        _section("Inputs")
        proj_value   = st.number_input("Project Value (₹)", value=float(round(proj_value_est, -4)), step=500000.0, format="%.0f", key="prelims_val")
        duration     = st.number_input("Duration (months)", value=18, min_value=1, max_value=120, key="prelims_dur")
        built_area   = st.number_input("Built-up Area (sqm)", value=float(max(area_sqm, 500.0)), step=100.0, format="%.0f", key="prelims_area")
        floors_in    = st.number_input("Floors", value=int(n_floors), min_value=1, max_value=60, key="prelims_floors")
        proj_type    = st.selectbox("Project Type", ["residential", "commercial", "institutional", "industrial", "infrastructure"], key="prelims_type")

    if st.button("Calculate Prelims", key="prelims_calc"):
        try:
            from prelims.calculator     import PrelimsCalculator
            from prelims.staff_costs    import StaffCostsCalculator
            from prelims.equipment_costs import EquipmentCostsCalculator
            from prelims.site_facilities import SiteFacilitiesCalculator

            calc  = PrelimsCalculator(proj_value, int(duration), float(built_area), proj_type)
            staff_items = StaffCostsCalculator().calculate(int(duration), proj_value, proj_type)
            equip_items = EquipmentCostsCalculator().calculate(int(duration), float(built_area), proj_type, int(floors_in))
            site_items  = SiteFacilitiesCalculator().calculate(int(duration), float(built_area), proj_type)
            ins_items   = calc.calculate_insurance_bonds(proj_value, int(duration))
            misc_items  = calc.calculate_miscellaneous(proj_value, int(duration))

            all_items = staff_items + equip_items + site_items + ins_items + misc_items

            with col_out:
                _section("Prelims BOQ")
                total = sum(i.amount for i in all_items)
                pct   = total / proj_value * 100 if proj_value else 0

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Prelims", _inr(total))
                m2.metric("% of Project Value", f"{pct:.1f}%")
                m3.metric("Items", str(len(all_items)))

                # Group by category
                cats: Dict[str, List] = {}
                for item in all_items:
                    cats.setdefault(item.category, []).append(item)

                for cat, items in cats.items():
                    cat_total = sum(i.amount for i in items)
                    with st.expander(f"**{cat.replace('_',' ').title()}** — {_inr(cat_total)}", expanded=True):
                        rows = [{"Description": i.description, "Unit": i.unit,
                                 "Qty": round(i.quantity, 2), "Rate": _inr(i.rate),
                                 "Amount": _inr(i.amount)} for i in items]
                        st.dataframe(rows, use_container_width=True)

                # Benchmark validation
                bench = calc.validate_prelims(total)
                bench_col = "#22c55e" if bench["status"] == "OK" else "#f59e0b"
                st.markdown(
                    f'<div style="background:#1e293b;border-left:4px solid {bench_col};'
                    f'border-radius:6px;padding:0.6rem 1rem;margin-top:0.5rem">'
                    f'<b>Benchmark check:</b> {bench["status"]} — {bench["recommendation"]}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # Download
                rows_dl = [i.to_dict() for i in all_items]
                st.download_button(
                    "⬇ Download Prelims BOQ (JSON)",
                    data=json.dumps(rows_dl, indent=2),
                    file_name=f"prelims_{payload.get('project_id','project')}.json",
                    mime="application/json",
                    use_container_width=True,
                )

                try:
                    import csv, io as _io
                    buf = _io.StringIO()
                    w = csv.DictWriter(buf, fieldnames=["description","unit","quantity","rate","amount","category"])
                    w.writeheader()
                    for i in all_items:
                        w.writerow(i.to_dict())
                    st.download_button(
                        "⬇ Download Prelims BOQ (CSV)",
                        data=buf.getvalue().encode(),
                        file_name=f"prelims_{payload.get('project_id','project')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                except Exception:
                    pass

        except ImportError as e:
            st.error(f"Prelims modules not available: {e}")
        except Exception as e:
            st.error(f"Error calculating prelims: {e}")
            st.code(traceback.format_exc())
    else:
        with col_out:
            st.info("Fill in the inputs and click **Calculate Prelims**.")


# =============================================================================
# Tab 25 — 📬 Scope Packages
# =============================================================================

def render_scope_packages_tab(payload: dict) -> None:
    """Subcontractor scope packages: split BOQ by trade, compare quotes."""
    st.markdown("### 📬 Subcontractor Scope Packages")
    st.caption("Split your BOQ by trade to generate sub-bid invitation packages.")

    line_items = payload.get("line_items", payload.get("boq_items", []))
    if not line_items:
        st.warning("No BOQ line items found in this analysis. Run a full pipeline first.")
        return

    # ── Group by trade ────────────────────────────────────────────────────────
    trade_map: Dict[str, List[dict]] = {}
    for item in line_items:
        trade = str(item.get("trade", item.get("package", "General"))).title()
        trade_map.setdefault(trade, []).append(item)

    trades = sorted(trade_map.keys())
    _section(f"BOQ Split — {len(trades)} trades, {len(line_items)} items")

    # Summary table
    summary_rows = []
    for tr in trades:
        items = trade_map[tr]
        amt = sum(float(i.get("amount", 0) or 0) for i in items)
        summary_rows.append({"Trade": tr, "Items": len(items), "Est. Value": _inr(amt)})
    st.dataframe(summary_rows, use_container_width=True)

    st.divider()
    selected_trade = st.selectbox("Select trade to view / download scope package:", trades, key="tab25_trade")

    if selected_trade:
        items = trade_map[selected_trade]
        st.markdown(f"**{selected_trade}** — {len(items)} items")
        show_rows = [
            {
                "Item No": i.get("item_no", i.get("unified_item_no", "—")),
                "Description": str(i.get("description", ""))[:80],
                "Unit": i.get("unit", ""),
                "Qty": i.get("qty", i.get("quantity", "")),
                "Rate (₹)": i.get("rate", i.get("unit_rate", "")),
                "Amount": _inr(float(i.get("amount", 0) or 0)),
            }
            for i in items
        ]
        st.dataframe(show_rows, use_container_width=True)

        # Download this trade as CSV
        import csv, io as _io
        buf = _io.StringIO()
        if show_rows:
            w = csv.DictWriter(buf, fieldnames=list(show_rows[0].keys()))
            w.writeheader()
            w.writerows(show_rows)
        st.download_button(
            f"⬇ Download {selected_trade} Scope (CSV)",
            data=buf.getvalue().encode(),
            file_name=f"scope_{selected_trade.lower().replace(' ','_')}_{payload.get('project_id','')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="tab25_dl_csv",
        )

        # Download full scope package JSON
        scope_pkg = {
            "project_id": payload.get("project_id", ""),
            "trade": selected_trade,
            "item_count": len(items),
            "items": items,
            "instructions": [
                "Price all items inclusive of GST unless stated.",
                "Validity: 90 days from date of submission.",
                "Queries: raise an RFI before submission.",
            ],
        }
        st.download_button(
            f"⬇ Download {selected_trade} Scope Package (JSON)",
            data=json.dumps(scope_pkg, indent=2, default=str),
            file_name=f"scope_pkg_{selected_trade.lower().replace(' ','_')}.json",
            mime="application/json",
            use_container_width=True,
            key="tab25_dl_json",
        )

    st.divider()

    # ── Quote leveler ─────────────────────────────────────────────────────────
    _section("Quote Comparison (optional)")
    st.caption("Paste or upload subcontractor quotes to level and compare them.")
    try:
        from quotes.leveler import QuoteLeveler
        from quotes.recommender import QuoteRecommender
        from quotes.parser import SubcontractorQuote, QuoteLineItem

        if "tab25_quotes" not in st.session_state:
            st.session_state["tab25_quotes"] = []

        with st.expander("Add a quote manually"):
            with st.form("tab25_add_quote"):
                sub_name  = st.text_input("Subcontractor name")
                pkg       = st.text_input("Package / Trade")
                total_amt = st.number_input("Total quoted amount (₹)", min_value=0.0, step=10000.0)
                excl      = st.text_area("Exclusions (one per line)")
                incl      = st.text_area("Inclusions (one per line)")
                sub_ok    = st.form_submit_button("Add Quote")
                if sub_ok and sub_name:
                    q = SubcontractorQuote(
                        subcontractor_name=sub_name,
                        package=pkg,
                        total_amount=total_amt,
                        exclusions=[e.strip() for e in excl.splitlines() if e.strip()],
                        inclusions=[i.strip() for i in incl.splitlines() if i.strip()],
                    )
                    st.session_state["tab25_quotes"].append(q)
                    st.success(f"Added quote from {sub_name}")

        quotes = st.session_state["tab25_quotes"]
        if len(quotes) >= 2:
            if st.button("Level & Compare Quotes", key="tab25_level"):
                leveled = QuoteLeveler().level(quotes, line_items)
                rec     = QuoteRecommender().recommend(leveled)
                st.metric("Recommended Bidder", rec.get("recommended_bidder","—"))
                st.metric("Recommended Amount", _inr(rec.get("recommended_amount", 0)))
                st.metric("Savings vs Highest", _inr(rec.get("highest_amount",0) - rec.get("recommended_amount",0)))
                st.markdown("**Reasons:**")
                for r in rec.get("reasons", []):
                    st.markdown(f"- {r}")
                if rec.get("risks"):
                    st.markdown("**Risks:**")
                    for r in rec["risks"]:
                        st.markdown(f"- ⚠️ {r}")
        elif quotes:
            st.info(f"{len(quotes)} quote added. Add at least 2 to compare.")
        else:
            st.info("Add 2+ subcontractor quotes above to compare.")

    except ImportError:
        st.info("Quote leveler module not available.")
    except Exception as e:
        st.error(f"Quote comparison error: {e}")


# =============================================================================
# Tab 26 — 🔄 Addenda
# =============================================================================

def render_addenda_tab(payload: dict) -> None:
    """Revision & amendment tracker: detect what changed and estimate impact."""
    st.markdown("### 🔄 Addenda & Revision Tracker")
    st.caption("Track drawing revisions and estimate their cost impact on your bid.")

    # ── Revision history from current payload ─────────────────────────────────
    try:
        from revision.detector import RevisionDetector
        from revision.tracker  import RevisionTracker
        from revision.impact   import ImpactAnalyzer

        pages     = payload.get("diagnostics", {}).get("page_index", [])
        scope_reg = payload.get("scope_register", {"items": []})
        boq_items = payload.get("boq_items", payload.get("line_items", []))

        detector = RevisionDetector()
        tracker  = RevisionTracker()

        # Build revision history
        tables  = detector.detect_all(pages) if pages else []
        history = tracker.build_history(pages, tables) if pages else None

        col_stat, col_detail = st.columns([1, 2])

        with col_stat:
            _section("Revision Summary")
            if history:
                st.metric("Total Sheets", len(history.sheets))
                st.metric("Sheets with Revisions", len(history.sheets) - len(history.sheets_without_revisions))
                st.metric("Latest Revision", history.latest_revision or "—")
                st.metric("Latest Date", history.latest_date or "—")
                gaps = tracker.find_revision_gaps(history)
                if gaps:
                    st.warning(f"⚠️ {len(gaps)} sheet(s) may be missing revisions")
                    for g in gaps[:5]:
                        st.markdown(f"  - `{g}`")
            else:
                st.info("No drawing page index in this payload. Revision detection requires a full pipeline run with drawings.")

        with col_detail:
            _section("Sheet Revision Log")
            if history and history.sheets:
                rows = []
                for sid, sr in history.sheets.items():
                    rows.append({
                        "Sheet": sid,
                        "Name": sr.sheet_name[:40],
                        "Current Rev": sr.current_revision or "—",
                        "Date": sr.current_date or "—",
                        "Rev Count": sr.revision_count,
                    })
                st.dataframe(rows, use_container_width=True)

                # Impact analysis
                if st.button("Analyze Revision Impact", key="tab26_impact"):
                    analyzer = ImpactAnalyzer()
                    try:
                        impact = analyzer.analyze(history, None, scope_reg, boq_items)
                        c1, c2, c3 = st.columns(3)
                        c1.metric("High Impact Items", impact.high_impact_count, delta_color="inverse")
                        c2.metric("Medium Impact", impact.medium_impact_count)
                        c3.metric("Cost Impact Est.", _inr(impact.total_cost_impact or 0))
                        if impact.recommendations:
                            st.markdown("**Recommendations:**")
                            for r in impact.recommendations:
                                st.markdown(f"- {r}")
                    except Exception as ie:
                        st.warning(f"Impact analysis incomplete: {ie}")
            else:
                st.info("No sheet revision data found.")

    except ImportError as e:
        st.info(f"Revision module not available: {e}")
    except Exception as e:
        st.error(f"Revision tracker error: {e}")
        st.code(traceback.format_exc())

    st.divider()

    # ── Manual addenda upload ─────────────────────────────────────────────────
    _section("Upload Addendum / Corrigendum")
    st.caption("Upload a new version of drawings or BOQ to detect scope changes.")
    st.info("🔜 Drag-and-drop addendum comparison coming soon. Upload a second PDF and compare side-by-side.")

    addendum_data = payload.get("addendum_index", [])
    if addendum_data:
        st.markdown(f"**{len(addendum_data)} addenda detected in this tender document:**")
        for a in addendum_data[:10]:
            st.markdown(f"- {a}")


# =============================================================================
# Tab 27 — 🔍 Reconcile
# =============================================================================

def render_reconcile_tab(payload: dict) -> None:
    """Drawing-vs-owner BOQ reconciliation: match, compare, flag discrepancies."""
    st.markdown("### 🔍 Drawing ↔ Owner BOQ Reconciliation")
    st.caption("Match quantities extracted from drawings against the owner's BOQ. Flag discrepancies.")

    try:
        from alignment.matcher    import BOQMatcher
        from alignment.comparator import BOQComparator
        from alignment.reconciler import BOQReconciler
    except ImportError as e:
        st.error(f"Alignment modules not available: {e}")
        return

    # Data sources
    drawings_boq = payload.get("line_items", [])
    owner_boq    = payload.get("boq_items", [])

    col_src = st.columns(2)
    col_src[0].metric("Drawings BOQ items", len(drawings_boq))
    col_src[1].metric("Owner BOQ items",    len(owner_boq))

    if not drawings_boq and not owner_boq:
        st.warning("No BOQ items found. Run a full pipeline with a drawings PDF and BOQ document.")
        return
    if not drawings_boq:
        st.warning("No drawings-derived items. Structural/QTO takeoff may be needed.")
        return
    if not owner_boq:
        st.warning("No owner BOQ extracted. Upload a bill-of-quantities document.")
        return

    tolerance = st.slider("Discrepancy tolerance (%)", 0, 30, 10, key="tab27_tol")

    if st.button("Run Reconciliation", key="tab27_run", type="primary"):
        try:
            with st.spinner("Matching items…"):
                matcher    = BOQMatcher()
                comparator = BOQComparator(tolerance_percent=float(tolerance))
                reconciler = BOQReconciler()

                matches      = matcher.match(drawings_boq, owner_boq)
                discrepancies = comparator.compare(matches)
                unified_boq, notes = reconciler.reconcile(matches, discrepancies)
                summary = reconciler.generate_reconciliation_summary(unified_boq)

            # ── Stats ─────────────────────────────────────────────────────────
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Matched", summary.get("matched", 0))
            c2.metric("Drawings Only", summary.get("drawings_only", 0))
            c3.metric("Owner Only", summary.get("owner_only", 0))
            c4.metric("Total Items", summary.get("total_items", 0))

            # ── Discrepancy breakdown ─────────────────────────────────────────
            if discrepancies:
                sev_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
                for d in discrepancies:
                    sev_counts[d.severity] = sev_counts.get(d.severity, 0) + 1

                sev_cols = st.columns(4)
                colors = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
                for i, (sev, cnt) in enumerate(sev_counts.items()):
                    sev_cols[i].metric(f"{colors[sev]} {sev.title()}", cnt)

                st.markdown("#### Discrepancies")
                disc_rows = [
                    {
                        "Description": d.description[:60],
                        "Unit": d.unit,
                        "Drawings Qty": round(d.drawings_qty, 2),
                        "Owner Qty": round(d.owner_qty, 2),
                        "Diff %": f"{d.difference_percent:+.1f}%",
                        "Severity": d.severity.upper(),
                        "Cause": d.possible_cause[:50],
                    }
                    for d in sorted(discrepancies, key=lambda x: ["critical","high","medium","low"].index(x.severity))
                ]
                st.dataframe(disc_rows, use_container_width=True)

            # ── Unified BOQ ───────────────────────────────────────────────────
            if unified_boq:
                with st.expander(f"Unified BOQ ({len(unified_boq)} items)", expanded=False):
                    uni_rows = [
                        {
                            "Item": u.get("unified_item_no",""),
                            "Description": str(u.get("description",""))[:60],
                            "Unit": u.get("unit",""),
                            "Qty": u.get("quantity",""),
                            "Qty Source": u.get("quantity_source",""),
                            "Status": u.get("status",""),
                            "Package": u.get("package",""),
                        }
                        for u in unified_boq
                    ]
                    st.dataframe(uni_rows, use_container_width=True)

                # Download
                st.download_button(
                    "⬇ Download Unified BOQ (JSON)",
                    data=json.dumps(unified_boq, indent=2, default=str),
                    file_name=f"unified_boq_{payload.get('project_id','')}.json",
                    mime="application/json",
                    use_container_width=True,
                )

            if notes:
                with st.expander("Reconciliation notes"):
                    for n in notes:
                        st.markdown(f"- {n}")

        except Exception as e:
            st.error(f"Reconciliation failed: {e}")
            st.code(traceback.format_exc())


# =============================================================================
# Tab 28 — 💵 Cash Flow
# =============================================================================

def render_cash_flow_tab(payload: dict) -> None:
    """S-curve and monthly spend profile from BOQ line items."""
    st.markdown("### 💵 Cash Flow & S-Curve")
    st.caption("Monthly spend projection from BOQ. Each trade is assigned a timing band.")

    try:
        from analysis.cash_flow import compute_cash_flow, fmt_inr, TRADE_DISPLAY
    except ImportError as e:
        st.error(f"Cash flow module not available: {e}")
        return

    line_items = payload.get("line_items", payload.get("boq_items", []))
    boq_total  = sum(float(i.get("amount", 0) or 0) for i in line_items)

    c_inp, c_out = st.columns([1, 2])
    with c_inp:
        _section("Inputs")
        duration  = st.slider("Project duration (months)", 6, 60, 18, key="cf_dur")
        val_input = st.number_input(
            "Total project value (₹) — 0 = use BOQ total",
            value=0.0, step=1_000_000.0, format="%.0f", key="cf_val",
        )
        override = val_input if val_input > 0 else (boq_total if boq_total > 0 else None)

        if line_items:
            st.caption(f"BOQ total: {fmt_inr(boq_total)} across {len(line_items)} items")
        else:
            st.warning("No BOQ items — using ₹1 Cr placeholder.")

    with c_out:
        try:
            result = compute_cash_flow(line_items, duration, override)

            # ── Headline metrics ───────────────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Project Value", fmt_inr(result.total_value))
            m2.metric("Peak Spend Month",    f"M{result.peak_month}")
            m3.metric("Front-half Spend",    f"{result.front_half_pct}%")

            # ── S-Curve (using st.line_chart) ──────────────────────────────────
            _section("Cumulative S-Curve (%)")
            chart_data = {
                "Cumulative %": result.cumulative_pcts[1:],  # drop the leading 0
            }
            try:
                import pandas as pd
                df = pd.DataFrame(chart_data, index=[f"M{i+1}" for i in range(result.duration_months)])
                st.line_chart(df, use_container_width=True)
            except ImportError:
                st.line_chart(chart_data)

            # ── Monthly bar chart ──────────────────────────────────────────────
            _section("Monthly Spend")
            try:
                import pandas as pd
                spend_df = pd.DataFrame(
                    {"Spend (₹)": result.monthly_spends},
                    index=[f"M{i+1}" for i in range(result.duration_months)],
                )
                st.bar_chart(spend_df, use_container_width=True)
            except ImportError:
                st.bar_chart({"Spend": result.monthly_spends})

            # ── Trade breakdown ────────────────────────────────────────────────
            with st.expander("Trade breakdown"):
                trade_rows = [
                    {
                        "Trade": TRADE_DISPLAY.get(k, k.title()),
                        "Est. Value": fmt_inr(v),
                        "% of Total": f"{v/result.total_value*100:.1f}%" if result.total_value else "—",
                    }
                    for k, v in sorted(result.trade_totals.items(), key=lambda x: -x[1])
                ]
                st.dataframe(trade_rows, use_container_width=True)

            # ── Monthly detail table ───────────────────────────────────────────
            with st.expander("Monthly detail"):
                monthly_rows = [
                    {
                        "Month": m.label,
                        "Spend": fmt_inr(m.planned_spend),
                        "Cumulative": fmt_inr(m.cumulative_spend),
                        "S-Curve %": f"{m.cumulative_pct:.1f}%",
                    }
                    for m in result.months
                ]
                st.dataframe(monthly_rows, use_container_width=True)

            # ── Download ───────────────────────────────────────────────────────
            dl_data = [
                {
                    "Month": m.month, "Label": m.label,
                    "Planned_Spend_INR": round(m.planned_spend, 2),
                    "Cumulative_Spend_INR": round(m.cumulative_spend, 2),
                    "Cumulative_Pct": round(m.cumulative_pct, 2),
                }
                for m in result.months
            ]
            st.download_button(
                "⬇ Download S-Curve Data (JSON)",
                data=json.dumps(dl_data, indent=2),
                file_name=f"cash_flow_{payload.get('project_id','project')}.json",
                mime="application/json",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"Cash flow calculation failed: {e}")
            st.code(traceback.format_exc())


# =============================================================================
# Tab 29 — 📊 Benchmark
# =============================================================================

def render_benchmark_tab(payload: dict) -> None:
    """Eval harness accuracy dashboard: historical benchmark runs."""
    st.markdown("### 📊 Accuracy Benchmark Dashboard")
    st.caption("Shows MAE, MAPE, and Acc±10%/±20% across saved eval harness runs.")

    bench_dir = _ROOT / "benchmarks" / "_runs"

    try:
        from analysis.eval_harness import EvalHarness
        harness = EvalHarness()
    except ImportError as e:
        st.warning(f"EvalHarness not available: {e}")
        harness = None

    # ── Load saved runs from disk ──────────────────────────────────────────────
    run_files = sorted(bench_dir.glob("**/*.json")) if bench_dir.exists() else []

    if not run_files:
        st.info(f"No benchmark runs found in `{bench_dir}`. Run `EvalHarness.run()` to generate results.")
        _section("Run a Quick Benchmark")
        st.caption("Select a benchmark case to evaluate against current pipeline.")
        gt_dir = _ROOT / "benchmarks" / "gt"
        gt_files = list(gt_dir.glob("*.json")) if gt_dir.exists() else []
        if gt_files:
            chosen = st.selectbox("Ground truth case:", [f.name for f in gt_files], key="bench_gt")
            if st.button("Run Eval", key="bench_run") and harness:
                try:
                    from analysis.eval_harness import EvalCase
                    case = EvalCase.load(gt_dir / chosen)
                    with st.spinner("Running evaluation…"):
                        result = harness.run(case)
                    harness.save_run(result, bench_dir)
                    st.success("Benchmark complete! Reload to see results.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Benchmark failed: {e}")
                    st.code(traceback.format_exc())
        return

    # ── Parse and display runs ────────────────────────────────────────────────
    runs = []
    for f in run_files:
        try:
            data = json.loads(f.read_text())
            runs.append(data)
        except Exception:
            pass

    if not runs:
        st.warning("Benchmark run files found but could not be parsed.")
        return

    # Aggregate stats
    _section(f"Results — {len(runs)} run(s)")

    rows = []
    for r in runs:
        rows.append({
            "Case": r.get("case_name", r.get("name", "—")),
            "Mode": r.get("mode", "—"),
            "MAE Doors": round(r.get("door_mae", r.get("mae", 0)), 2),
            "MAE Windows": round(r.get("window_mae", 0), 2),
            "MAPE %": round(r.get("mape", r.get("mape_pct", 0)), 1),
            "Acc ±10%": f"{r.get('acc_10', r.get('accuracy_10pct', 0)):.0%}" if isinstance(r.get('acc_10', r.get('accuracy_10pct')), float) else "—",
            "Acc ±20%": f"{r.get('acc_20', r.get('accuracy_20pct', 0)):.0%}" if isinstance(r.get('acc_20', r.get('accuracy_20pct')), float) else "—",
            "Date": r.get("timestamp", r.get("run_at", ""))[:16],
        })

    st.dataframe(rows, use_container_width=True)

    # Overall averages
    if rows:
        avg_mape = sum(float(str(r["MAPE %"]).replace("%","")) for r in rows if r["MAPE %"] != "—") / len(rows)
        m1, m2 = st.columns(2)
        m1.metric("Avg MAPE across all runs", f"{avg_mape:.1f}%")
        m2.metric("Total benchmark cases", len(runs))

    # Raw JSON for deep inspection
    with st.expander("Raw benchmark data"):
        selected = st.selectbox("Select run:", [r.get("case_name", f"Run {i}") for i, r in enumerate(runs)], key="bench_sel")
        sel_idx  = [r.get("case_name", f"Run {i}") for i, r in enumerate(runs)].index(selected)
        st.json(runs[sel_idx])


# =============================================================================
# Tab 30 — 🔀 Compare
# =============================================================================

def render_compare_tab(payload: dict) -> None:
    """Multi-tender comparison: pin current analysis, compare side-by-side."""
    st.markdown("### 🔀 Multi-Tender Comparison")
    st.caption("Pin analyses from multiple tenders and compare them side-by-side.")

    PINNED_KEY = "tab30_pinned_analyses"
    if PINNED_KEY not in st.session_state:
        st.session_state[PINNED_KEY] = []

    pinned: List[dict] = st.session_state[PINNED_KEY]

    # ── Pin current analysis ───────────────────────────────────────────────────
    pid  = payload.get("project_id", "")
    name = payload.get("tender_name", payload.get("drawing_overview", {}).get("project_name", pid))

    already_pinned = any(p.get("project_id") == pid for p in pinned)
    c1, c2 = st.columns([3, 1])
    if not already_pinned and pid:
        if c1.button(f"📌 Pin current analysis ({pid[:12]}…)", key="tab30_pin"):
            pinned.append({
                "project_id":      pid,
                "name":            name or pid,
                "readiness_score": payload.get("readiness_score", 0),
                "decision":        payload.get("decision", "—"),
                "rfi_count":       len(payload.get("rfis", [])),
                "blocker_count":   len(payload.get("blockers", [])),
                "gap_count":       len(payload.get("gaps", [])),
                "boq_items":       len(payload.get("line_items", payload.get("boq_items", []))),
                "trade_coverage":  len(payload.get("trade_coverage", [])),
                "pages":           payload.get("processing_stats", {}).get("total_pages", 0),
                "bid_readiness_label": payload.get("bid_synthesis", {}).get("bid_readiness_label","—"),
                "estimated_cost":  payload.get("bid_synthesis", {}).get("estimated_cost_inr", 0),
                "cost_per_sqm":    payload.get("bid_synthesis", {}).get("cost_per_sqm", 0),
                "contingency_pct": payload.get("bid_synthesis", {}).get("recommended_contingency_pct", 0),
            })
            st.success("Pinned ✓")
            st.rerun()
    elif already_pinned:
        c1.info(f"Current analysis already pinned.")

    if c2.button("Clear all pins", key="tab30_clear"):
        st.session_state[PINNED_KEY] = []
        st.rerun()

    if not pinned:
        st.info("No analyses pinned yet. Run multiple tenders, pin each one, then compare here.")
        return

    st.divider()
    _section(f"Comparison — {len(pinned)} tender(s) pinned")

    # ── Side-by-side comparison table ─────────────────────────────────────────
    METRICS = [
        ("Readiness Score",      "readiness_score",      lambda v: f"{v}/100"),
        ("Decision",             "decision",              lambda v: str(v)),
        ("Readiness Label",      "bid_readiness_label",  lambda v: str(v)),
        ("RFIs",                 "rfi_count",             lambda v: str(v)),
        ("Blockers",             "blocker_count",         lambda v: str(v)),
        ("Gaps",                 "gap_count",             lambda v: str(v)),
        ("BOQ Items",            "boq_items",             lambda v: str(v)),
        ("Trade Coverage",       "trade_coverage",        lambda v: f"{v} trades"),
        ("Pages Processed",      "pages",                 lambda v: str(v)),
        ("Est. Cost",            "estimated_cost",        lambda v: _inr(v) if v else "—"),
        ("Cost / sqm",           "cost_per_sqm",          lambda v: _inr(v) if v else "—"),
        ("Recommended Contingency", "contingency_pct",   lambda v: f"{v:.1f}%" if v else "—"),
    ]

    # Build table: rows = metrics, columns = tenders
    col_headers = ["Metric"] + [p.get("name", p["project_id"])[:20] for p in pinned]
    table_rows  = []
    for label, key, fmt in METRICS:
        row = {"Metric": label}
        for p in pinned:
            row[p.get("name", p["project_id"])[:20]] = fmt(p.get(key, 0) or 0)
        table_rows.append(row)

    st.dataframe(table_rows, use_container_width=True)

    # ── Score comparison bar ───────────────────────────────────────────────────
    _section("Readiness Score Comparison")
    score_data = {p.get("name", p["project_id"])[:20]: p.get("readiness_score", 0) for p in pinned}
    try:
        import pandas as pd
        st.bar_chart(pd.DataFrame({"Readiness Score": score_data}))
    except ImportError:
        st.bar_chart(score_data)

    # ── Individual pinned cards ────────────────────────────────────────────────
    st.divider()
    _section("Pinned Tenders")
    cols = st.columns(min(len(pinned), 3))
    for i, p in enumerate(pinned):
        col = cols[i % len(cols)]
        score = p.get("readiness_score", 0)
        col_css = "#22c55e" if score >= 75 else "#f59e0b" if score >= 50 else "#ef4444"
        col.markdown(
            f'<div style="background:#1e293b;border-left:4px solid {col_css};'
            f'border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.5rem">'
            f'<div style="font-weight:700;font-size:0.95rem">{p.get("name","")[:24]}</div>'
            f'<div style="color:{col_css};font-size:1.4rem;font-weight:800">{score}/100</div>'
            f'<div style="color:#94a3b8;font-size:0.8rem">{p.get("decision","")}'
            f' · {p.get("rfi_count",0)} RFIs · {p.get("blocker_count",0)} blockers</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if col.button("❌ Unpin", key=f"tab30_unpin_{i}"):
            pinned.pop(i)
            st.rerun()
