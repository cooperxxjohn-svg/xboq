# Skill: Regression Suite

## When to Use

- After any change to extraction, scoring, or pipeline logic
- Before merging a PR that touches `src/analysis/`
- When setting up benchmark cases for a new tender type
- Comparing pipeline performance across code versions

## Inputs Required

1. `benchmarks/manifest.json` — mapping of case names to input file paths and run modes
2. `benchmarks/expected_metrics.yaml` — KPI thresholds per case (optional)
3. Access to tender files referenced in the manifest

## Procedure

1. **Set up manifest** (if not already done)
   ```bash
   cp benchmarks/manifest.example.json benchmarks/manifest.json
   # Edit manifest.json to point to your local tender files
   ```

2. **Set up expected metrics** (optional)
   ```bash
   # Edit benchmarks/expected_metrics.yaml with thresholds
   # See template for format
   ```

3. **Run regression**
   ```bash
   python scripts/run_regression.py
   ```

4. **Review results**
   - Check `benchmarks/_runs/<case>/<timestamp>/` for artifacts
   - Check `benchmarks/_runs/regression_report.csv` for KPI summary
   - Look for PASS/WARN/FAIL per KPI per case

5. **Compare against baseline**
   - If `expected_metrics.yaml` has thresholds:
     - PASS: KPI meets or exceeds threshold
     - WARN: KPI is within 10% of threshold
     - FAIL: KPI is below threshold by >10%
   - Without thresholds: results are informational only

6. **Investigate failures**
   - For each FAIL, follow the debugging runbook in `docs/DEBUGGING.md`
   - Cross-reference with `extraction_diagnostics` in the saved payload

## KPIs Tracked

| KPI | Source |
|-----|--------|
| `pages_total` | `processing_stats.total_pages` |
| `deep_processed_pages` | `processing_stats.deep_processed_pages` |
| `ocr_pages` | `processing_stats.ocr_pages` |
| `boq_items_count` | `len(boq_items)` |
| `commercial_terms_count` | `len(commercial_terms)` |
| `requirements_count` | `len(requirements)` |
| `rfis_count` | `len(rfis)` |
| `blockers_count` | `len(blockers)` |
| `toxic_pages_count` | `processing_stats.toxic_pages` |
| `qa_score` | `qa_score` |
| `boq_pages_attempted` | `extraction_diagnostics.boq_pages_attempted` |
| `boq_pages_parsed` | `extraction_diagnostics.boq_pages_parsed` |
| `schedule_pages_attempted` | `extraction_diagnostics.schedule_pages_attempted` |
| `schedule_pages_parsed` | `extraction_diagnostics.schedule_pages_parsed` |
| `table_methods_used` | `extraction_diagnostics.table_methods_used` |

## Definition of Done

- All cases in manifest run without crashes
- Regression report CSV generated
- All KPIs at PASS or WARN level (no regressions)
- If FAIL: root cause identified and documented

## Tests to Run

```bash
python scripts/run_regression.py
pytest tests/test_tender_pipeline.py -q
```

## Expected Response Format

- Regression report table (case × KPI × PASS/WARN/FAIL)
- Files changed (if fixes were needed)
- Commands run and output
- Recommendation: safe to merge or needs investigation
