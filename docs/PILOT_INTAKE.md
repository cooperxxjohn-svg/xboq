# Pilot Customer Intake: Processing, Storage & Using Data to Improve the System

## What you have (7 drawing sets + document zip)

- **7 drawing sets** = 7 PDFs (or 7 folders of PDFs) from the pilot.
- **Document zip** = likely conditions, BOQ, addenda, or other tender docs in one ZIP.

---

## 1. How we process it

### Option A: Demo UI (Streamlit)

- **Upload**: User drags multiple files into the file uploader. All selected PDFs are treated as one tender set.
- **Save**: `save_uploaded_files()` writes files to:
  - **`<repo>/uploads/<project_id>/`** — one folder per run, files named as uploaded (e.g. `Drawing_Set_01.pdf`, …).
- **Run**: `run_analysis_pipeline(input_files=saved_paths, project_id=..., output_dir=...)` runs the full pipeline (load → index → select → OCR/extract → graph → dependency reasoner → RFI engine → export).
- **Output**: Results go to **`<repo>/out/<project_id>/`**:
  - `analysis.json` — full payload (readiness, blockers, RFIs, trade coverage, quantities, commercial terms, etc.)
  - `deep_analysis.json`, `plan_graph.json`, evaluation logs
  - `.xboq_cache/` — OCR and extraction cache for this input set

So: **one project_id = one “tender” = N PDFs in that run.** For 7 drawing sets you can either:

- Create **one project** and upload all 7 PDFs (plus any from the doc zip) in a single run, or  
- Create **7 projects** (one per drawing set) if you want to analyze each set separately.

### Option B: CLI (batch / scripted)

- **`run_demo_analysis.py`**  
  - Single or multiple inputs:  
    `python run_demo_analysis.py --project_id pilot_xyz --input ./path/to/file1.pdf --input ./path/to/file2.pdf ...`  
  - Output: `--output` (default `./out/<project_id>/`) gets `analysis.json` and pipeline outputs.

- **`run_full_project.py`**  
  - Expects an **input directory** of drawings (PDFs/images).  
  - Usage: `python run_full_project.py --project_id pilot_xyz --input_dir /path/to/extracted_zip/`  
  - It copies/looks for PDFs in that dir and runs the full 22-phase pipeline. Output: `out/<project_id>/`.

**ZIP handling**: There is no built-in “upload a zip” in the UI. You (or the pilot) **unzip first**, then either:

- Point `--input_dir` at the extracted folder, or  
- Upload the extracted PDFs in the Streamlit UI.

---

## 2. Where we store it

| What | Location | Notes |
|------|----------|--------|
| **Uploaded PDFs (demo)** | `<repo>/uploads/<project_id>/` | One dir per project; filenames preserved |
| **Pipeline output (demo)** | `<repo>/out/<project_id>/` | analysis.json, deep_analysis, plan_graph, cache, logs |
| **Project metadata & run history** | `~/.xboq/projects/<project_id>/` | metadata.json, runs/<run_id>.json (when using projects API) |
| **Feedback (RFI/quantity verdicts)** | Per-run: `feedback.jsonl` in output dir; or via `StorageBackend.append_feedback(project_id)` | Pilot corrections (correct/wrong/edited) |
| **Collaboration (comments, assign, due date)** | `~/.xboq/projects/<project_id>/` via `collaboration.jsonl` (when using storage) | In-app collaboration state |

So for a pilot:

- **Files**: repo `uploads/` and `out/` (or wherever `--output` / `--input_dir` point).
- **Identity of the pilot run**: `project_id` (and optionally `run_id` in `~/.xboq/projects/` if you use the projects API).

---

## 3. How we use it to improve the system

Today the system **does** capture data that can improve the product; the main gap is **automated use** of that data.

### Already in place

- **Feedback capture**  
  - In the UI, estimators can mark RFIs as **correct** / **wrong** / **edited** (and optionally quantities).  
  - Stored in `feedback.jsonl` (per run/output dir) or via `StorageBackend.append_feedback`.  
  - Schema: `feedback_type`, `item_id`, `verdict`, `corrected_value`, `notes`, `timestamp`, etc.

- **Quality dashboard metrics**  
  - `quality_dashboard.compute_quality_metrics(feedback_entries, run_history)` produces:
    - Correct / wrong / edited counts (e.g. per RFI type)
    - Correction rate, “noisy” checks (most often marked wrong)
    - Run trends (readiness_score, rfis_count over time).  
  - So you **can** inspect which checks or RFIs are unreliable.

- **Training / pilot pack**  
  - `training_pack.build_training_pack(...)` bundles:
    - Input manifest (file hashes, names)
    - Slim payload (no heavy caches)
    - Ground truth diff, feedback.jsonl, context/metadata  
  - Output: a ZIP for **paired dataset export** (e.g. for fine-tuning or evaluation).  
  - Pilot runs can be exported as training packs and used offline to improve models or rules.

- **Pilot metadata on projects**  
  - `create_project(..., pilot_mode=True, company_name=..., trades_in_scope=...)`  
  - So you can mark and filter pilot projects and later aggregate by pilot or trade.

### Not yet wired end-to-end

- **Automatic retraining / threshold tuning**  
  - Feedback and quality metrics are not yet fed back into:
    - RFI rule thresholds, or  
    - Prompt or model selection.  
  - That would be the next step: e.g. “if correction rate for RFI type X > Y, relax threshold” or “send these items to human review”.

- **Central aggregation of pilot data**  
  - Feedback and runs live per project/run. There is no built-in job that:
    - Scans all pilot projects (e.g. `pilot_mode=True`),
    - Loads all `feedback.jsonl`,
    - Computes quality metrics per pilot or globally, or  
    - Exports training packs for every pilot run.  
  - You can do this with a one-off script (see below).

---

## 4. Recommended pilot intake workflow (7 sets + doc zip)

1. **Unzip** the document zip (and any zipped drawing sets) into one folder, e.g. `~/pilot_customer_jan2026/`.
2. **One project per tender** (or one per drawing set if you want separate reports):
   - Create project: e.g. `pilot_company_abc_tender_01` with `pilot_mode=True`, `company_name="Pilot Customer"`.
3. **Run analysis**:
   - **UI**: Upload all PDFs for that tender from the folder (multi-select). Run analysis. Results in `out/<project_id>/`.
   - **CLI**:  
     `python run_demo_analysis.py --project_id pilot_company_abc_tender_01 --input ~/pilot_customer_jan2026/file1.pdf --input ... --output ./out/pilot_company_abc_tender_01`  
   - Or use the small **pilot intake script** below to unzip and run from a single zip.
4. **Collect feedback** in the UI (correct/wrong/edited on RFIs) so `feedback.jsonl` is populated.
5. **Export training packs** for runs you care about (via existing “export training pack” in app or script calling `build_training_pack`).
6. **Periodically**: Run a script over `~/.xboq/projects/` and `out/` to aggregate feedback and run history, compute `compute_quality_metrics` per pilot or globally, and use that to decide which rules or prompts to adjust.

---

## 5. Quick script: unzip + run pipeline (optional)

A small script `scripts/pilot_intake.py` can:

- Take a path to the pilot’s zip (and optionally a project_id).
- Unzip to a temp or chosen directory.
- Collect all PDFs and run `run_analysis_pipeline` (or `run_demo_analysis`).
- Write results to `out/<project_id>/`.

That gives you a single command to “ingest this zip and produce analysis” for the 7 sets + doc zip (either one run per zip or one run per extracted folder).

---

## Summary

| Question | Answer |
|----------|--------|
| **Processing** | Multi-PDF supported. UI: upload → `uploads/<project_id>/` → pipeline → `out/<project_id>/`. CLI: `--input` / `--input_dir`. Unzip first if customer sends a zip. |
| **Storage** | Uploads and outputs under repo (or custom paths); project/run metadata and collaboration under `~/.xboq/projects/`; feedback in output dir or via storage. |
| **Using it to improve** | Feedback + quality metrics + training packs exist; use them to find noisy checks and to export paired data. Next step: wire metrics back into thresholds/prompts and/or aggregate pilot feedback centrally. |

If you want, next we can add `scripts/pilot_intake.py` (unzip + run) and a tiny “aggregate pilot feedback” script that lists pilot projects and runs `compute_quality_metrics` over their feedback.
