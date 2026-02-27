# Skill: Performance Optimization

## When to Use

- Pipeline takes too long on a tender (target: <5 min for standard_review)
- Many toxic page retries slowing down processing
- Cache hit rate is low despite repeated runs
- Need to optimize for demo speed (demo_fast mode)

## Inputs Required

1. `processing_stats` from the slow run (especially timing data)
2. `extraction_diagnostics` — `table_methods_used` breakdown
3. Run mode used (`demo_fast`, `standard_review`, `full_audit`)
4. Total page count and OCR page count
5. (Optional) Timing per extractor from `extraction_times`

## Procedure

1. **Profile the run**
   - Check `processing_stats`:
     - `total_pages` vs `deep_processed_pages` — are we processing too many?
     - `ocr_pages` — OCR is ~10x slower than text-layer extraction
     - `toxic_pages` — each retry costs ~30s (`src/analysis/toxic_pages.py` RETRY_TIMEOUT_S)

2. **Identify bottleneck extractors**
   - Check `extraction_times` in payload
   - Rank extractors by time spent
   - Common bottlenecks:
     - `table_router` OCR row reconstruction (slowest fallback)
     - `extract_boq` on scanned pages
     - `page_index` OCR pass on large documents

3. **Check table_router method distribution**
   - `extraction_diagnostics.table_methods_used`:
     - `pdfplumber` — fastest
     - `camelot_lattice` — medium
     - `camelot_stream` — slow
     - `ocr_rows` — slowest
     - `none` — wasted time with no result
   - If many pages fall through to `ocr_rows` or `none`, investigate why pdfplumber fails

4. **Check page selection efficiency**
   - File: `src/analysis/page_selection.py`
   - Tier 1 pages should be minimal but high-value
   - Review if unnecessary pages are in Tier 1
   - Consider if `demo_fast` budget (80) is appropriate for the use case

5. **Improve cache hit rate**
   - File: `src/analysis/pipeline_cache.py`
   - Check if cache is enabled and working
   - Cache key should include: file hash + run_mode + relevant config
   - Pre-warm cache for demo: `scripts/prewarm_demo_cache.py`

6. **Reduce toxic page impact**
   - File: `src/analysis/toxic_pages.py`
   - Current retry: normal DPI → low DPI (72) → mark toxic
   - Each retry has 30s timeout
   - Consider: skip known-toxic page patterns earlier
   - Check `ocr_fallback.py` Tesseract config optimization

7. **Run mode tuning**
   - `demo_fast` (80 pages) — for quick demos
   - `standard_review` (220 pages) — balanced
   - `full_audit` (all pages) — only when needed
   - File: `src/analysis/page_selection.py`

## Definition of Done

- Identified top 3 time sinks with evidence
- Applied targeted optimizations (no broad refactors)
- Re-run shows measurable improvement (compare timing)
- No extraction quality regression: run `python scripts/run_regression.py`
- Pipeline tests green: `pytest tests/test_tender_pipeline.py -q`

## Tests to Run

```bash
pytest tests/test_tender_pipeline.py -q
python scripts/run_regression.py  # if benchmarks configured
python scripts/smoke_render_results.py
```

## Expected Response Format

- Bottleneck analysis table (component → time → % of total)
- Changes made (file:line) with rationale
- Before/after timing comparison
- Regression suite results (no quality degradation)
