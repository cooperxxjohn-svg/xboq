# Pilot Tender Workflow

Step-by-step guide for processing new pilot tenders through xBOQ.

---

## Overview

```
Inventory → Dry-Run → Run 1 Tender → Batch All → Scorecard → Fix → Rerun → Output Pack
```

---

## Step 1: Inventory First

Before processing anything, scan all tender folders for metadata only (no OCR, no pipeline).

```bash
python scripts/pilot_inventory.py \
  --input ~/path/to/tender_folders/ \
  --output ~/path/to/inventory_output/
```

Or use the ops wrapper:

```bash
python scripts/pilot_ops.py --input ~/path/to/tender_folders/ --mode inventory
```

**What it produces:**
- Per-tender `manifest.json` — every file with type, size, page count, Excel previews
- Per-tender `summary.md` — human-readable overview
- Master `inventory_rollup.md` + `inventory_rollup.csv`

**What to look for:**
- Where the BOQ lives (Excel vs PDF vs NOT FOUND)
- Drawing count (embedded in PDFs or trapped in RARs?)
- Conditions/specs files present
- Addenda count
- Missing components flagged
- Recommended processing order

---

## Step 2: Dry-Run Classification

Run the batch ingest in dry-run mode to verify file classification without running the pipeline.

```bash
python scripts/pilot_batch_ingest.py \
  --input ~/path/to/tender_folders/ \
  --input-type folder \
  --dry-run \
  --tenant pilot_customer
```

Or:

```bash
python scripts/pilot_ops.py --input ~/path/to/tender_folders/ --mode dry-run
```

**What to check:**
- File routing summary per tender (how many PDFs routed to BOQ, drawings, conditions, etc.)
- Any files classified as `unknown` — may need keyword additions to `src/pilot/file_router.py`
- Excel files detected as BOQ sources

---

## Step 3: Run One Tender First

Pick the simplest tender (recommended by inventory) and run it end-to-end.

```bash
python scripts/pilot_batch_ingest.py \
  --input ~/path/to/tender_folders/ \
  --input-type folder \
  --mode full_audit \
  --tenant pilot_customer \
  --include "Tender Documents 3" \
  --limit 1
```

**What to check:**
- Pipeline completes without crash
- Output directory has: `analysis.json`, `deep_analysis.json`, `plan_graph.json`, `file_routing.json`, `eval.json`
- Scorecard row shows reasonable metrics (BOQ items > 0, RFIs > 0)

---

## Step 4: Batch Run All

Once one tender works, run all tenders.

```bash
python scripts/pilot_batch_ingest.py \
  --input ~/path/to/tender_folders/ \
  --input-type folder \
  --mode full_audit \
  --tenant pilot_customer
```

Or:

```bash
python scripts/pilot_ops.py \
  --input ~/path/to/tender_folders/ \
  --output ~/path/to/pilot_output/ \
  --tenant pilot_customer \
  --mode full_audit
```

---

## Step 5: Read the Scorecard

The batch ingest writes `pilot_scorecard.csv` to the output directory.

```bash
python scripts/pilot_ops.py --input ~/path/to/pilot_output/ --mode summary
```

**Scorecard columns:**
- `tender_name`, `status`, `pages_total`, `deep_processed_pages`
- `boq_items`, `rfis`, `blockers`, `commercial_terms`
- `qa_score`, `duration_sec`, `error`

**What to look for:**
- Any tenders with `status = ERROR`
- Tenders with `boq_items = 0` (BOQ extraction failed)
- Tenders with `qa_score < 0.5` (low quality)
- Tenders with high `duration_sec` (performance issues)

---

## Step 6: Fix Top 2 Failure Patterns Only

Do NOT try to fix everything. Focus on the top 2 most impactful patterns.

Common patterns and fixes:

| Pattern | Likely Fix |
|---------|-----------|
| BOQ items = 0 | Check `extraction_diagnostics.boq_pages_attempted` — if 0, page classification missed BOQ pages. Add keywords to `page_index.py`. |
| File routing wrong | Add keywords to `src/pilot/file_router.py` → `_normalize_name()` or keyword lists |
| OCR timeout / toxic pages | Check `processing_stats.toxic_pages` — consider reducing DPI or skipping known-bad patterns |
| Missing conditions | Conditions may be embedded in main tender doc — check page classification |
| Schedule rows = 0 | Schedule pages may be classified as `plan` instead of `schedule` |

After fixing, run tests:

```bash
pytest tests/test_tender_pipeline.py -q
```

---

## Step 7: Rerun Failed Tenders Only

Use `--include` to rerun only the tenders that failed:

```bash
python scripts/pilot_batch_ingest.py \
  --input ~/path/to/tender_folders/ \
  --input-type folder \
  --mode full_audit \
  --tenant pilot_customer \
  --include "Tender Documents 2" \
  --include "Tender Documents 4"
```

---

## Step 8: Prepare Pilot Output Pack

After all tenders pass, prepare the output for review:

1. **Scorecard**: `pilot_scorecard.csv` — overall summary
2. **Per-tender outputs**: `<output>/<tender>/analysis.json` — full payloads
3. **Inventory rollup**: `inventory_rollup.md` — what each tender contains
4. **Known issues**: Document any tenders with missing BOQ, low scores, or known gaps

Optional — launch the UI to visually inspect results:

```bash
streamlit run app/demo_page.py
```

Load a specific project from the output directory and click through all 7 tabs.

---

## Quick Reference

| Step | Command | Time |
|------|---------|------|
| Inventory | `python scripts/pilot_ops.py --input DIR --mode inventory` | ~1s |
| Dry-run | `python scripts/pilot_ops.py --input DIR --mode dry-run` | ~2s |
| Run 1 | `pilot_batch_ingest.py --include NAME --limit 1` | 1-5 min |
| Run all | `python scripts/pilot_ops.py --input DIR --mode full_audit` | 5-30 min |
| Summary | `python scripts/pilot_ops.py --input OUTPUT --mode summary` | instant |
| Rerun | `pilot_batch_ingest.py --include NAME1 --include NAME2` | varies |
