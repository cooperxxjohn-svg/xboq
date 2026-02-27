# Skill: UI Stability

## When to Use

- Streamlit app crashes or shows error messages in any tab
- Duplicate widget key errors in console
- Visual overlap or garbled text rendering
- New UI feature added — need to verify stability
- Before a demo or presentation

## Inputs Required

1. Error message or traceback from Streamlit
2. (Optional) Screenshot of the rendering issue
3. Which tab / section is affected
4. Whether the issue is reproducible with synthetic data or only real tenders

## Procedure

1. **Run smoke test first**
   ```bash
   python scripts/smoke_render_results.py
   ```
   - Exit 0 = all synthetic payloads render without exception
   - Exit 1 = at least one payload caused an exception — read the traceback

2. **Check widget keys**
   - Search for `st.` widget calls missing `key=` parameter
   - All widget keys must use `_make_widget_key(tab, item_id, idx)`
   - File: `app/demo_page.py:964` — `_make_widget_key()` definition
   - Common fix: add `key=_make_widget_key("tab_name", identifier, loop_index)`

3. **Check safe_tab wrapping**
   - Every tab render function must be called via `_safe_tab()`
   - File: `app/demo_page.py:4850` — `_safe_tab()` definition
   - Pattern: `_safe_tab("Tab Name", render_tab_fn, payload, ...)`
   - If a tab is not wrapped, an exception in that tab crashes the entire app

4. **Check safe_str usage**
   - All user-sourced text rendered in UI must go through `safe_str()`
   - Files using safe_str: `demo_page.py`, `docx_exports.py`, `bid_summary_pdf.py`, `evidence_appendix_pdf.py`
   - Common issue: encoding errors with non-ASCII characters (Hindi/Devanagari in India tenders)

5. **Check payload access patterns**
   - All payload field access must use `.get()` with defaults
   - Never assume a key exists — older runs may lack newer keys
   - Pattern: `payload.get("boq_stats", {}).get("total", 0)`

6. **Test with edge case payloads**
   - Empty payload `{}`
   - Payload with only `processing_stats`
   - Payload with very large lists (1000+ BOQ items)
   - Payload with unicode / special characters

7. **Visual check**
   ```bash
   streamlit run app/demo_page.py
   ```
   - Click through all 7 tabs
   - Expand all expanders
   - Check for overlapping text, missing sections, broken layouts

## Definition of Done

- `python scripts/smoke_render_results.py` exits 0
- No duplicate widget key errors
- All tabs render without crash (even with minimal payload)
- Visual check shows no overlapping or garbled text
- `pytest tests/test_tender_pipeline.py -q` still green

## Tests to Run

```bash
python scripts/smoke_render_results.py
pytest tests/test_tender_pipeline.py -q
streamlit run app/demo_page.py  # manual
```

## Expected Response Format

- List of widget key fixes (file:line → change)
- List of safe_tab / safe_str additions
- Smoke test before/after output
- Screenshot or description of visual verification
