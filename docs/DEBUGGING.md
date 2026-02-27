# xBOQ Debugging Runbook

"If X then check Y" checklists for common issues.

---

## BOQ Items Count Too Low or Zero

**Symptoms**: `boq_items` list is empty or has fewer items than expected.

1. **Check page_selection tiering**
   - Payload → `processing_stats.selection_mode` (should be `standard_review` or `full_audit` for large docs)
   - Payload → `processing_stats.deep_processed_pages` vs `total_pages`
   - If `selection_mode` is `demo_fast`, BOQ pages may have been skipped (budget = 80 pages)
   - File: `src/analysis/page_selection.py` — Tier 1 includes `boq` doc_type

2. **Check extraction_diagnostics**
   - Payload → `extraction_diagnostics.boq_pages_attempted` — how many pages routed to BOQ extractor
   - Payload → `extraction_diagnostics.boq_pages_parsed` — how many yielded items
   - If attempted == 0: page_index did not classify any pages as `boq`
   - If attempted > 0 but parsed == 0: table extraction failed on those pages

3. **Check table_router diagnostics**
   - Payload → `extraction_diagnostics.table_methods_used` — which methods succeeded
   - Fallback chain: pdfplumber → camelot lattice → camelot stream → OCR rows → none
   - If method is `none` for BOQ pages, all extraction failed — check if pages are image-only
   - File: `src/analysis/table_router.py`

4. **Check OCR routing**
   - Payload → `processing_stats.ocr_pages` — number of pages requiring OCR
   - File: `src/analysis/ocr_fallback.py` — Tesseract config may need tuning for this document
   - Try: run with `full_audit` mode to process all pages

**UI location**: Coverage tab → diagnostics expander → BOQ section

**Test to run**: `pytest tests/test_tender_pipeline.py -q -k boq`

---

## Schedules Missing (Doors / Windows / Finishes)

**Symptoms**: `schedules` list is empty or missing expected schedule types.

1. **Check extraction_diagnostics**
   - Payload → `extraction_diagnostics.schedule_pages_attempted` / `schedule_pages_parsed`
   - If attempted == 0: no pages classified as `schedule` by page_index

2. **Check page classification**
   - Payload → page_index results — look for pages classified as `schedule`
   - File: `src/analysis/page_index.py` — the `doc_type` enum includes `schedule`
   - Common issue: schedule is embedded in drawing sheets (classified as `plan` or `detail`)

3. **Check table extraction methods**
   - Payload → `extraction_diagnostics.table_methods_used`
   - Schedule extractor relies on `table_router` — same fallback chain as BOQ
   - File: `src/analysis/extractors/extract_schedule_tables.py`

**UI location**: Coverage tab → Schedule section

**Test to run**: `pytest tests/test_tender_pipeline.py -q -k schedule`

---

## Pages Skipped — Confusion About What Was Processed

**Symptoms**: User sees high skip count or doesn't understand coverage.

1. **Check processing_stats**
   - `total_pages` — from page_index (total in PDF)
   - `deep_processed_pages` — pages that went through full extraction
   - `skipped_pages` — total minus deep_processed
   - `selection_mode` — `demo_fast` (80), `standard_review` (220), `full_audit` (all)
   - `selected_pages_count` — pages chosen by page_selection

2. **Understand tiering**
   - Tier 1 (always processed): cover, index, legend, notes, schedule, boq, conditions, addendum
   - Tier 2 (budget permitting): plan, detail, section, elevation (round-robin across disciplines)
   - Tier 3 (remaining budget): unknown, spec (sample)
   - File: `src/analysis/page_selection.py`

3. **Check toxic pages**
   - Payload → `processing_stats.toxic_pages` — pages that failed OCR even after retry
   - File: `src/analysis/toxic_pages.py` — retry at 72 DPI, then mark toxic

**UI location**: Review Queue → skipped panel; Coverage tab → processing stats expander

---

## UI Issues

### Duplicate Widget Keys

- **Cause**: Two Streamlit widgets with the same `key=` parameter
- **Fix**: Use `_make_widget_key(tab_name, item_id, loop_index)` for all widget keys
- **File**: `app/demo_page.py:964` — `_make_widget_key()` definition
- **Pattern**: `st.checkbox(..., key=_make_widget_key("review", rfi_id, idx))`

### Overlap / Strange Rendering

- **Cause**: Non-string or encoding-unsafe values rendered directly
- **Fix**: Wrap with `safe_str()` before rendering
- **Files**: `app/demo_page.py`, `app/docx_exports.py`, `app/bid_summary_pdf.py`, `app/evidence_appendix_pdf.py`

### Tab Failures / Crashes

- **Cause**: Unhandled exception in a tab render function
- **Fix**: Wrap tab rendering with `_safe_tab(tab_name, render_fn, *args, **kwargs)`
- **File**: `app/demo_page.py:4850` — `_safe_tab()` definition
- **Behavior**: Shows friendly error message instead of crashing entire app

### General UI Debugging

- Run `python scripts/smoke_render_results.py` to test all tabs with synthetic payloads
- Check for `st.error` or `st.warning` calls that might indicate handled errors
- Launch with `streamlit run app/demo_page.py` and test each tab manually

---

## Performance / Timing Issues

### Slow Pages

1. **Check processing_stats**
   - `extraction_times` in payload — per-extractor timing
   - Look for extractors taking > 5s per page

2. **Check toxic page retries**
   - Payload → `processing_stats.toxic_pages`
   - Each toxic page retry adds ~30s (RETRY_TIMEOUT_S in `toxic_pages.py`)
   - File: `src/analysis/toxic_pages.py`

3. **Check table_router method selection**
   - OCR row reconstruction is slowest — check if many pages fall through to OCR
   - Payload → `extraction_diagnostics.table_methods_used`

### Cache Hit Rate

- File: `src/analysis/pipeline_cache.py`
- Check if pipeline_cache is being used (cache hit = skip re-processing)
- Demo cache: `demo_cache/` directory

### Run Mode Optimization

- `demo_fast` (80 pages) — fastest, for demos
- `standard_review` (220 pages) — production default
- `full_audit` (all pages) — slowest, full coverage
- File: `src/analysis/page_selection.py` — budget constants

---

## RFIs Missing or Low Quality

1. **Check extraction coverage**
   - RFIs are generated from extracted data — if extraction is poor, RFIs will be poor
   - Check `boq_items`, `schedules`, `requirements` counts first

2. **Check RFI engine checklist**
   - File: `src/analysis/rfi_engine.py` — ~30-50 checklist items across trades
   - Each check has a condition that must be met to generate an RFI
   - Target: >15 RFIs on 300+ page tender

3. **Check evidence references**
   - Each RFI should have `EvidenceRef` linking to source pages
   - Missing evidence = check `src/models/analysis_models.py` → `EvidenceRef`

**UI location**: RFIs tab → grouped RFIs expander

---

## Quick Reference: Payload Key → Module

| Payload Key | Source Module |
|-------------|-------------|
| `processing_stats` | `src/analysis/pipeline.py` → `_build_processing_stats()` |
| `extraction_diagnostics` | `src/analysis/extractors/__init__.py` → `ExtractionResult` |
| `boq_items` / `boq_stats` | `src/analysis/extractors/extract_boq.py` |
| `schedules` | `src/analysis/extractors/extract_schedule_tables.py` |
| `requirements` / `requirements_by_trade` | `src/analysis/extractors/extract_notes.py` |
| `commercial_terms` | `src/analysis/extractors/extract_commercial_terms.py` |
| `rfis` / `blockers` | `src/analysis/rfi_engine.py` |
| `quantity_reconciliation` | `src/analysis/quantity_reconciliation.py` |
| `structural_takeoff` | `src/structural/` |
| `estimating_playbook` | `src/analysis/estimating_playbook.py` |
| `qa_score` | `src/analysis/qa_score.py` |

---

## Pilot Scorecard Triage

**Symptoms**: Pilot batch run completed but some tenders show errors or low scores.

1. **Read the scorecard**
   ```bash
   python scripts/pilot_ops.py --input <output_dir> --mode summary
   ```
   Or open `pilot_scorecard.csv` directly.

2. **Check status column**
   - `OK` — pipeline completed, check quality metrics
   - `ERROR` — pipeline crashed, check `error` column for traceback
   - `PIPELINE_FAIL` — pipeline ran but returned `success=False`

3. **Triage by impact**
   - Sort by `qa_score` ascending — lowest quality first
   - Sort by `boq_items` — zeroes indicate BOQ extraction failure
   - Sort by `duration_sec` — slowest may have toxic page issues

4. **Common scorecard patterns**

   | Pattern | Payload Key | Likely Cause | Fix |
   |---------|-------------|-------------|-----|
   | `boq_items = 0` | `extraction_diagnostics.boq_pages_attempted` | Page classification missed BOQ pages | Add keywords to `page_index.py` |
   | `rfis = 0` | `boq_items`, `requirements` | No data to generate RFIs from | Fix extraction first |
   | `status = ERROR` | `error` column | Pipeline crash | Read traceback, check for missing deps |
   | `qa_score < 0.3` | All extraction counts | Mostly image-only PDF, OCR failed | Try `full_audit` mode |
   | `duration > 300s` | `processing_stats.toxic_pages` | Toxic page retries | Check `toxic_pages.py`, reduce DPI |

5. **Fix top 2 patterns only** — do not try to fix everything at once
6. **Rerun failed tenders**
   ```bash
   python scripts/pilot_batch_ingest.py --include "Tender Name" --input <dir>
   ```

**Script**: `scripts/pilot_ops.py --mode summary`

---

## Scripts & Tests to Run

```bash
# Full pipeline test suite
pytest tests/test_tender_pipeline.py -q

# Smoke-test UI rendering
python scripts/smoke_render_results.py

# Run regression suite (needs benchmarks/manifest.json)
python scripts/run_regression.py

# Launch UI for manual testing
streamlit run app/demo_page.py

# Pilot operations
python scripts/pilot_ops.py --help
```
