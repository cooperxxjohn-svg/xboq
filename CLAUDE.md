# CLAUDE.md — xBOQ Repo Guide

## What is xBOQ?

AI bid engineer copilot for contractors.
Analyses RCC construction tenders (India-first) — ingests PDFs (BOQ, drawings,
schedules, specs), extracts structured data, generates RFIs, and produces
submission packs.

**Current-pack intelligence first, history / RAG later.**

---

## Key Entry Points

| Area | File |
|------|------|
| **UI entry** | `app/demo_page.py` |
| **Pipeline orchestrator** | `src/analysis/pipeline.py` → `run_analysis_pipeline()` |
| **Page indexing** | `src/analysis/page_index.py` → `PageIndex`, `IndexedPage` |
| **Page selection** | `src/analysis/page_selection.py` → `SelectedPages` |
| **Extractors** | `src/analysis/extractors/` (BOQ, schedules, notes, drawings, commercial) |
| **Table router** | `src/analysis/table_router.py` → `TableExtractionResult` |
| **OCR fallback** | `src/analysis/ocr_fallback.py` |
| **RFI engine** | `src/analysis/rfi_engine.py` → checklist-driven RFI generation |
| **Pilot batch ingest** | `scripts/pilot_batch_ingest.py` |
| **Pilot inventory** | `scripts/pilot_inventory.py` |
| **Pilot ops** | `scripts/pilot_ops.py` |
| **Projects** | `src/analysis/projects.py` (CRUD, `~/.xboq/projects/`) |
| **Company playbooks** | `src/analysis/company_playbooks.py` (`~/.xboq/playbooks/`) |
| **Submission pack** | `app/submission_pack.py` (ZIP builder) |
| **Bid summary** | `app/bid_summary.py`, `app/bid_summary_pdf.py` |
| **DOCX exports** | `app/docx_exports.py` |
| **Data models** | `src/models/analysis_models.py` (Pydantic: RFIItem, EvidenceRef, RunMode, Trade, etc.) |
| **Toxic page isolation** | `src/analysis/toxic_pages.py` |
| **Review queue** | `src/analysis/review_queue.py` |
| **QA scoring** | `src/analysis/qa_score.py` |
| **Run artifacts** | `src/ops/run_artifacts.py` — structured debug output |
| **Collaboration** | `src/analysis/collaboration.py` (JSONL append-only) |

---

## Pipeline Flow

```
PDF files
  → page_index.py    (classify every page: doc_type + discipline)
  → page_selection.py (tier pages within OCR budget: DEMO_FAST=80, STANDARD_REVIEW=220, FULL_AUDIT=all)
  → extractors/      (route by doc_type → BOQ/schedule/notes/drawings/commercial)
  → table_router.py  (fallback chain: pdfplumber → camelot lattice → camelot stream → OCR rows)
  → rfi_engine.py    (checklist-driven RFI generation with evidence)
  → pipeline.py      (assemble payload, processing_stats, extraction_diagnostics)
  → demo_page.py     (7-tab Streamlit UI rendering)
```

---

## Important Payload Keys

```
processing_stats          — total_pages, deep_processed_pages, ocr_pages, skipped_pages, selection_mode
extraction_diagnostics    — boq_pages_attempted/parsed, schedule_pages_attempted/parsed, methods_used
boq_stats                 — counts, coverage per trade
commercial_terms          — parsed contract terms
requirements_by_trade     — requirements grouped by trade
rfis                      — generated RFIs with evidence
blockers                  — blocking items with bid impact + cost risk
structural_takeoff        — structural extraction results
estimating_playbook       — company estimating strategies
quantity_reconciliation   — cross-check quantities
plan_graph                — rooms, openings, columns, beams, footings
qa_score                  — overall quality score
```

---

## Non-Negotiable Rules

1. **Never break backward compatibility** — all changes must be additive
2. **Never claim data is missing when coverage is incomplete** — check `selection_mode` and skip counts first
3. **Use `.get()` with defaults** for all payload access — keys may be absent in older runs
4. **Use `_make_widget_key()`** for all Streamlit widget keys (see `app/demo_page.py:964`)
5. **Use `safe_str()`** when rendering user text to avoid encoding/overlap issues
6. **Add unit + smoke tests** for any behavior changes
7. **Prefer minimal patches over refactors** — do not modify extraction/scoring/selection/UI logic without a dedicated sprint ticket
8. **Never crash the UI** — wrap tab renders with `_safe_tab()` (see `app/demo_page.py:4850`)

---

## Standard Commands

```bash
# Run pipeline tests
pytest tests/test_tender_pipeline.py -q

# Smoke-test UI rendering with synthetic payloads
python scripts/smoke_render_results.py

# Launch Streamlit UI
streamlit run app/demo_page.py

# Run regression suite (needs benchmarks/manifest.json — see benchmarks/README.md)
python scripts/run_regression.py

# Run all tests
pytest tests/ -q

# Pilot ops (inventory, dry-run, run, summary)
python scripts/pilot_ops.py --help

# Initialize a dataset case for ground truth capture
python scripts/init_dataset_case.py --tenant acme --project hospital_300pg
```

---

## Project Layout

```
xboq.ai/
├── app/                     # Streamlit UI components (15 modules)
├── src/
│   ├── analysis/            # Core analysis pipeline (48 modules)
│   │   └── extractors/      # Per-type extraction (5 extractors + __init__)
│   ├── models/              # Pydantic data models
│   ├── ops/                 # Operational helpers (run artifacts, etc.)
│   ├── boq/                 # BOQ processing (13 modules)
│   ├── structural/          # Structural analysis (15 modules)
│   ├── estimator/           # Cost estimation (8 modules)
│   ├── exports/             # Multi-format export (9 modules)
│   ├── scoring/             # Bid scoring (6 modules)
│   ├── openings/            # Door/window analysis (11 modules)
│   ├── finishes/            # Finish takeoff (6 modules)
│   ├── rfi/                 # RFI management (8 modules)
│   ├── scope/               # Scope definition (9 modules)
│   ├── pricing/             # Pricing logic (8 modules)
│   ├── project/             # Project management (13 modules)
│   ├── mep/                 # MEP systems (9 modules)
│   ├── risk/                # Risk assessment (8 modules)
│   ├── reporting/           # Report generation (13 modules)
│   ├── storage/             # Persistence layer (7 modules)
│   ├── auth/                # Authentication (5 modules)
│   ├── ui/                  # UI utilities (6 modules)
│   └── adapters/            # Format adapters (17 modules)
├── tests/                   # Test suite
├── scripts/                 # Utility & demo scripts
├── docs/                    # Architecture, debugging, workflow guides
├── prompts/skills/          # Claude prompt playbooks
├── benchmarks/              # Regression benchmark scaffolding
├── rules/                   # YAML rule files
├── rates/                   # Rate data
├── templates/               # Template files
└── requirements.txt         # Dependencies (Python 3.10+)
```

---

## Formatting (Opt-In)

No `pyproject.toml` with ruff/black is configured by default.
If you add one, ensure it does NOT auto-reformat existing code on save.
Use `--check` mode only until the team opts in to reformatting.
