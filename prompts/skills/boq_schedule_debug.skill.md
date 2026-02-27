# Skill: BOQ & Schedule Extraction Debug

## When to Use

- BOQ item count is zero or suspiciously low
- Schedule tables (doors, windows, finishes) are missing
- Extraction diagnostics show pages attempted but not parsed
- Table router is falling through to OCR or "none"

## Inputs Required

Paste one or more of:
1. The `extraction_diagnostics` section from the analysis payload
2. The `processing_stats` section (especially `selection_mode`, `deep_processed_pages`)
3. The `table_methods_used` breakdown
4. Any error messages from the pipeline run
5. (Optional) The raw text from a BOQ/schedule page for comparison

## Procedure

1. **Check page classification**
   - Open `src/analysis/page_index.py`
   - Verify the doc_type classification for the problematic pages
   - Look at `EXTRACTOR_ROUTING` in `src/analysis/extractors/__init__.py` to confirm routing

2. **Check page selection**
   - Open `src/analysis/page_selection.py`
   - Confirm `boq` and `schedule` are in Tier 1 (always selected)
   - Check if `selection_mode` budget was sufficient

3. **Trace table extraction**
   - Open `src/analysis/table_router.py`
   - Check the fallback chain: pdfplumber → camelot lattice → camelot stream → OCR rows
   - For each failed method, check the confidence threshold and why it was skipped

4. **Check the specific extractor**
   - BOQ: `src/analysis/extractors/extract_boq.py`
   - Schedules: `src/analysis/extractors/extract_schedule_tables.py`
   - Look at parsing logic and minimum row/column requirements

5. **Run targeted tests**
   ```bash
   pytest tests/test_tender_pipeline.py -q -k boq
   pytest tests/test_tender_pipeline.py -q -k schedule
   ```

6. **If OCR is the issue**
   - Check `src/analysis/ocr_fallback.py` — Tesseract PSM config
   - Try running with `full_audit` mode to process all pages
   - Check `src/analysis/toxic_pages.py` for pages that failed OCR entirely

## Definition of Done

- Root cause identified (page classification, table extraction, or OCR)
- Fix applied and targeted test passes
- `extraction_diagnostics` shows improved attempted/parsed counts
- Full pipeline test suite still green: `pytest tests/test_tender_pipeline.py -q`

## Tests to Run

```bash
pytest tests/test_tender_pipeline.py -q -k boq
pytest tests/test_tender_pipeline.py -q -k schedule
pytest tests/test_tender_pipeline.py -q  # full suite
python scripts/smoke_render_results.py
```

## Expected Response Format

- Files changed (with line numbers)
- Commands run and their output
- Before/after extraction_diagnostics comparison
- Manual QA: re-run pipeline on affected document and verify counts
