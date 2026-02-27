# xBOQ Architecture

Single-page overview of all components and their module paths.

---

## 1. Orchestration

| Concern | Module |
|---------|--------|
| Pipeline entry | `src/analysis/pipeline.py` → `run_analysis_pipeline()` |
| Pipeline cache | `src/analysis/pipeline_cache.py` |
| Job / project CRUD | `src/analysis/projects.py` (`~/.xboq/projects/<id>/`) |
| Collaboration | `src/analysis/collaboration.py` (JSONL append-only) |
| Company playbooks | `src/analysis/company_playbooks.py` (`~/.xboq/playbooks/`) |
| Owner profiles | `src/analysis/owner_profiles.py` |
| Storage layer | `src/storage/` (7 modules) |
| Auth | `src/auth/` (5 modules) |
| Run modes | `src/models/analysis_models.py` → `RunMode` (DEMO_FAST / STANDARD_REVIEW / FULL_AUDIT) |

---

## 2. Document Intelligence

| Concern | Module | Key Classes |
|---------|--------|-------------|
| Page classification | `src/analysis/page_index.py` | `PageIndex`, `IndexedPage` — two-pass: text-layer then OCR |
| Page prioritization | `src/analysis/page_selection.py` | `SelectedPages` — Tier 1/2/3 within budget |
| Table extraction | `src/analysis/table_router.py` | `TableExtractionResult` — fallback chain: pdfplumber → camelot lattice → camelot stream → OCR rows |
| OCR fallback | `src/analysis/ocr_fallback.py` | Tesseract with PSM 6 (blocks) / PSM 11 (sparse) |
| Surya OCR | `src/analysis/surya_ocr.py` | Alternative OCR engine |
| Toxic page isolation | `src/analysis/toxic_pages.py` | Retry at low DPI → mark toxic → skip |
| Multi-doc handling | `src/analysis/multi_doc.py` | Cross-document processing |
| Normalization | `src/analysis/normalize.py` | Data normalization |

---

## 3. Extractors

All live under `src/analysis/extractors/`. Routing is by `doc_type` from PageIndex.

| Extractor | File | Routes from doc_types |
|-----------|------|----------------------|
| BOQ | `extract_boq.py` | `boq` |
| Schedules | `extract_schedule_tables.py` | `schedule` (doors, windows, finishes) |
| Requirements | `extract_notes.py` | `notes`, `legend`, `spec`, `conditions`, `addendum` |
| Drawings | `extract_drawings_minimal.py` | `plan`, `detail`, `section`, `elevation` |
| Commercial terms | `extract_commercial_terms.py` | (commercial document pages) |

Combined output: `ExtractionResult` dataclass (`extractors/__init__.py`).

Routing table defined in `extractors/__init__.py` → `EXTRACTOR_ROUTING`.

---

## 4. Decisioning

| Concern | Module |
|---------|--------|
| RFI generation | `src/analysis/rfi_engine.py` — checklist-driven, ~30-50 checks per trade |
| RFI clustering | `src/analysis/rfi_clustering.py` |
| Conflict detection | `src/analysis/conflicts.py` |
| Dependency reasoning | `src/analysis/dependency_reasoner.py` |
| Delta detection | `src/analysis/delta_detector.py` |
| Supersedes detection | `src/analysis/supersedes_detector.py` |
| Review queue | `src/analysis/review_queue.py` |
| Approval workflow | `src/analysis/approval_states.py` |
| Bulk actions | `src/analysis/bulk_actions.py` |
| Reconciliation | `src/analysis/reconciler.py`, `src/analysis/recon_actions.py` |
| QA scoring | `src/analysis/qa_score.py` |
| Quality dashboard | `src/analysis/quality_dashboard.py` |

---

## 5. Specialized Analysis

| Concern | Module |
|---------|--------|
| Quantities | `src/analysis/quantities.py` |
| Quantity reconciliation | `src/analysis/quantity_reconciliation.py` |
| Finish takeoff | `src/analysis/finish_takeoff.py` |
| Plan graph | `src/analysis/plan_graph.py` → `PlanSetGraph` entity aggregation |
| Highlights | `src/analysis/highlights.py` |
| LLM enrichment | `src/analysis/llm_enrichment.py` |
| Addendum adapter | `src/analysis/addendum_adapter.py` |
| Determinism config | `src/analysis/determinism.py` |
| Estimating playbook | `src/analysis/estimating_playbook.py` |
| Pricing guidance | `src/analysis/pricing_guidance.py` |

---

## 6. Domain Modules (under `src/`)

| Domain | Path | Modules |
|--------|------|---------|
| BOQ processing | `src/boq/` | 13 modules |
| Structural | `src/structural/` | 15 modules |
| Estimator | `src/estimator/` | 8 modules |
| Openings (doors/windows) | `src/openings/` | 11 modules |
| Finishes | `src/finishes/` | 6 modules |
| RFI management | `src/rfi/` | 8 modules |
| Scope | `src/scope/` | 9 modules |
| Pricing | `src/pricing/` | 8 modules |
| Project management | `src/project/` | 13 modules |
| MEP | `src/mep/` | 9 modules |
| Risk | `src/risk/` | 8 modules |
| Reporting | `src/reporting/` | 13 modules |
| Measurement rules | `src/measurement_rules/` | 10 modules |
| Preliminaries | `src/prelims/` | 8 modules |
| Details | `src/details/` | 7 modules |
| Owner docs | `src/owner_docs/` | 10 modules |
| Adapters | `src/adapters/` | 17 modules |
| UI utilities | `src/ui/` | 6 modules |

---

## 7. Deliverables / Exports

| Output | Module |
|--------|--------|
| 7-tab Streamlit UI | `app/demo_page.py` (345KB) |
| Submission ZIP | `app/submission_pack.py` |
| Bid summary PDF | `app/bid_summary_pdf.py` |
| Bid summary | `app/bid_summary.py` |
| DOCX exports | `app/docx_exports.py` |
| Evidence appendix | `app/evidence_appendix_pdf.py` |
| BOQ editor UI | `app/boq_app.py` |
| Structural UI | `app/structural_app.py` |
| Openings UI | `app/openings_app.py` |
| Risk checklist | `app/risk_checklist.py` |
| Pilot intake | `app/pilot_docs.py` |
| Multi-format exports | `src/exports/` (9 modules) |

---

## 8. Learning Hooks

| Concern | Module |
|---------|--------|
| Feedback collection | `src/analysis/feedback.py` |
| Ground truth intake | `src/analysis/ground_truth.py` |
| Ground truth diff | `src/analysis/ground_truth_diff.py` |
| Evaluation metrics | `src/analysis/evaluation.py` |
| Training pack | `src/analysis/training_pack.py` |
| Meeting agenda gen | `src/analysis/meeting_agenda.py` |

RAG integration is planned for a future sprint.

---

## 9. Data Models

Core Pydantic models in `src/models/analysis_models.py`:

- **Enums**: `Severity`, `BidImpact`, `RiskLevel`, `Trade`, `SheetType`, `Discipline`, `RunMode`
- **Models**: `RFIItem`, `EvidenceRef`, `PlanSetGraph`, `PlanSheet`, `RunCoverage`
- **ExtractionResult**: `src/analysis/extractors/__init__.py` (dataclass)
- **AnalysisResult / AnalysisStage**: `src/analysis/pipeline.py`

---

## 10. Tests & Scripts

| Asset | Path |
|-------|------|
| Pipeline tests | `tests/test_tender_pipeline.py` (294KB, comprehensive) |
| BOQ schema tests | `tests/test_boq_schema.py` |
| Smoke render test | `scripts/smoke_render_results.py` |
| Regression runner | `scripts/run_regression.py` |
| Pilot batch ingest | `scripts/pilot_batch_ingest.py` |
| Demo scripts | `scripts/run_bid_demo_standalone.py`, `scripts/run_demo.sh` |
| Synthetic generators | `tests/synthetic_generator.py`, `tests/synthetic_structural.py` |
