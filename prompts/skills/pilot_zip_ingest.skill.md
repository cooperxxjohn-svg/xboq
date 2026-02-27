# Skill: Pilot ZIP Ingest

## When to Use

- Processing a new pilot tender ZIP (India RCC construction)
- Need to classify mixed PDF/DOCX/XLSX documents
- Need to validate and produce a scorecard for intake quality
- Running batch ingest for multiple tenders

## Inputs Required

1. Path to the ZIP file or extracted folder
2. Tender metadata (project name, owner, bid date) if available
3. Expected document types (BOQ, drawings, specs, schedules)
4. (Optional) Previous scorecard for comparison

## Procedure

1. **Extract and classify documents**
   - Use `scripts/pilot_batch_ingest.py` or `scripts/pilot_intake.py`
   - Each file is classified by extension and content:
     - PDF → run through `page_index.py` for page-level classification
     - DOCX → extract text, classify as spec/conditions/notes
     - XLSX → check for BOQ structure (item, description, unit, qty, rate columns)

2. **Create project**
   - Use `src/analysis/projects.py` → `create_project()`
   - Or: `python scripts/new_project.py --name "Tender Name" --owner "Owner"`

3. **Run pipeline**
   ```bash
   # Standard review mode for pilot
   python run_full_project.py --input <path> --mode standard_review
   ```

4. **Produce intake scorecard**
   - Check `processing_stats`:
     - Total pages ingested
     - Pages classified by doc_type
     - OCR vs text-layer ratio
   - Check `extraction_diagnostics`:
     - BOQ items extracted
     - Schedules found
     - Requirements captured
     - Commercial terms parsed
   - Check `rfis` count and quality

5. **Validate outputs**
   - Review submission pack structure (`app/submission_pack.py`)
   - Check each export format (PDF, DOCX, CSV)
   - Verify RFI evidence references point to real pages

6. **Document findings**
   - Note any document types not handled well
   - Note OCR quality issues (scanned vs digital)
   - Note missing data that should have been extracted

## Definition of Done

- All documents in ZIP classified and processed
- Scorecard produced with: page count, extraction counts, RFI count, coverage %
- No pipeline crashes (check for `_safe_tab` error catches)
- Results saved under `~/.xboq/projects/<project_id>/`
- Pilot intake doc updated if needed: `docs/PILOT_INTAKE.md`

## Tests to Run

```bash
pytest tests/test_tender_pipeline.py -q
python scripts/smoke_render_results.py
streamlit run app/demo_page.py  # manual visual check
```

## Expected Response Format

- Document classification summary table
- Extraction scorecard (items per category)
- List of issues / items needing manual review
- Recommended next steps (re-run at higher mode, manual page review, etc.)
