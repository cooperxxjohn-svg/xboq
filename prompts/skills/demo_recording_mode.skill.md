# Skill: Demo Recording Mode

## When to Use

- Preparing for a YC demo or investor presentation
- Recording a screen capture of the tool in action
- Need clean, non-noisy UI screens
- Verifying all exports generate correctly for demo

## Inputs Required

1. Which demo scenario to run (e.g., standard tender review)
2. Demo tender file path (or use cached demo data)
3. Export formats needed (PDF, DOCX, ZIP submission pack)
4. Any specific features to highlight

## Procedure

1. **Pre-warm the demo cache**
   ```bash
   python scripts/prewarm_demo_cache.py
   ```
   - Ensures demo runs are instant (no pipeline wait time)
   - Cached results are stored in `demo_cache/`

2. **Verify demo data is clean**
   - Check `demo_inputs/` for the demo tender files
   - Run pipeline in `demo_fast` mode:
     ```bash
     python scripts/run_bid_demo_standalone.py
     ```
   - Or use the demo scripts:
     ```bash
     bash scripts/run_demo.sh
     bash scripts/run_demo_ready.sh
     ```

3. **Check UI is clean**
   ```bash
   streamlit run app/demo_page.py
   ```
   - Click through all 7 tabs:
     1. Executive Summary — readiness score, decision, sub-scores
     2. Missing Dependencies — evidence-backed items
     3. Flagged Areas/Risks — severity-ranked
     4. RFIs — grouped and exportable
     5. Trade Coverage & Priceability — coverage matrix
     6. Assumptions/Exclusions — structured lists
     7. Drawing Set Overview/Audit Trail — page classification summary
   - Verify no error messages, warnings, or debug output visible
   - Verify all expanders open cleanly

4. **Test all exports**
   - Bid Summary PDF: `app/bid_summary_pdf.py`
   - Bid Summary DOCX: `app/docx_exports.py`
   - Evidence Appendix: `app/evidence_appendix_pdf.py`
   - Submission ZIP: `app/submission_pack.py`
   - Download each and verify content/formatting

5. **Check for noisy screens**
   - No debug prints in console
   - No "TODO" or "WIP" labels visible in UI
   - No placeholder data or "lorem ipsum"
   - Processing stats section is informative but not overwhelming
   - RFI evidence references are complete (no "page N/A")

6. **Run smoke tests**
   ```bash
   python scripts/smoke_render_results.py
   pytest tests/test_tender_pipeline.py -q
   ```

7. **Record checklist**
   - [ ] Cache pre-warmed
   - [ ] All 7 tabs render cleanly
   - [ ] All exports generate without error
   - [ ] No noisy debug output
   - [ ] Demo scenario runs end-to-end in <30s
   - [ ] Screen resolution appropriate for recording

## Definition of Done

- All 7 UI tabs render cleanly with demo data
- All export formats generate correctly
- No error messages, warnings, or debug output visible
- Demo runs in <30s from cached data
- Smoke tests pass

## Tests to Run

```bash
python scripts/smoke_render_results.py
pytest tests/test_tender_pipeline.py -q
python scripts/run_bid_demo_standalone.py
```

## Expected Response Format

- Demo readiness checklist (all items checked)
- Screenshots or descriptions of each tab
- List of any issues found and fixes applied
- Export verification results
- Recommended recording setup (resolution, flow sequence)
