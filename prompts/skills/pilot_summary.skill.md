# Skill: Pilot Summary

## When to Use

- After completing pilot batch ingest and need a concise summary
- Preparing a 1-page tender overview for stakeholders
- Need to rank tenders by quality, readiness, or complexity
- Generating a pilot output pack for review

## Inputs Required

1. Path to pilot output directory (containing `pilot_scorecard.csv`)
2. (Optional) Path to inventory output (`inventory_rollup.csv`)
3. (Optional) Specific tender names to focus on
4. Target audience (internal team vs client vs investor)

## Procedure

1. **Read the scorecard**
   ```bash
   python scripts/pilot_ops.py --input <output_dir> --mode summary
   ```
   - Parse `pilot_scorecard.csv` for per-tender metrics
   - Identify top/bottom performers by `qa_score`

2. **Read inventory rollup** (if available)
   - Parse `inventory_rollup.csv` for file counts, page counts, BOQ source
   - Cross-reference with scorecard results

3. **For each tender, extract key facts**
   - Load `analysis.json` from the output directory
   - Pull:
     - `processing_stats.total_pages` — document size
     - `boq_items` count — extraction success
     - `rfis` count — issue detection
     - `blockers` count — blocking items
     - `commercial_terms` — contract term coverage
     - `qa_score` — overall quality
     - `processing_stats.selection_mode` — how much was processed

4. **Build 1-page summary**
   - Header: pilot name, date, tender count
   - Table: tender × key metrics
   - Highlights: best/worst tenders
   - Issues: common gaps across tenders
   - Recommendations: what to fix for production readiness

5. **Save summary**
   - Write to `<output_dir>/pilot_summary.md`
   - Include both human-readable and structured data

## Definition of Done

- 1-page summary generated with all tenders covered
- Tender ranking by quality score
- Top 3 issues identified across all tenders
- Summary saved to output directory
- No new code changes required (read-only operation)

## Tests to Run

```bash
pytest tests/test_tender_pipeline.py -q -k pilot
python scripts/pilot_ops.py --input <output_dir> --mode summary
```

## Expected Response Format

```markdown
# Pilot Summary: <Customer Name>
Date: <date>
Tenders: <count>

## Tender Rankings
| Rank | Tender | Pages | BOQ Items | RFIs | QA Score |
|------|--------|-------|-----------|------|----------|
| 1    | ...    | ...   | ...       | ...  | ...      |

## Key Findings
- ...

## Common Issues
1. ...
2. ...

## Recommendations
- ...
```
