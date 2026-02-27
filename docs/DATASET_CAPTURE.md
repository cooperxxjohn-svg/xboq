# Dataset Capture — Pre-RAG Learning Data Collection

How xBOQ collects training and learning data now, before RAG integration.

---

## Why Capture Data Now?

Every pilot run produces valuable signal:
- What the pipeline extracted vs what the estimator actually needed
- Which RFIs were useful vs noisy
- What the estimator corrected or added
- How the BOQ was refined before submission

Capturing this systematically now means we'll have high-quality paired datasets
when RAG and fine-tuning become available.

---

## What to Capture

### 1. Refined Outputs

After an estimator reviews pipeline results, capture the refined version:
- Corrected BOQ items (quantities, units, descriptions)
- Edited RFIs (accepted, rejected, modified)
- Adjusted commercial terms
- Final bid pricing decisions

**Where it lives today**: `feedback.jsonl` in each run's output directory.

### 2. Playbook Used

Which company playbook was active during analysis:
- Trade-specific markup percentages
- Standard assumptions and exclusions
- Preferred materials / approved makes
- Regional rate adjustments

**Where it lives today**: `~/.xboq/playbooks/<playbook_id>/`

### 3. Approvals and Edits

Track the review workflow:
- Which blockers were resolved (and how)
- Which RFIs were sent vs deferred
- Approval decisions with timestamps
- Comments and notes from reviewers

**Where it lives today**: `collaboration.jsonl` per project, `approval_states` in payload.

### 4. Exported Artifacts

The final deliverables sent to the client:
- Submission pack ZIP
- Bid summary PDF
- RFI letters
- Assumptions/exclusions DOCX

**Where it lives today**: `out/<project_id>/` output directory.

### 5. Uploaded Ground Truth Files

When available, the "correct answer" for comparison:
- Client-provided BOQ Excel (the official BOQ)
- Final awarded quantities
- Post-tender reconciliation data

**Where it lives today**: Not yet systematic — use `scripts/init_dataset_case.py` to scaffold.

### 6. Diff Reports (Future)

Comparison between pipeline output and ground truth:
- BOQ items matched vs missed
- Quantity accuracy (within ±10%?)
- RFI relevance scoring

**Where it will live**: `datasets/<tenant>/<project>/<run>/feedback/`

---

## Standard Dataset Case Structure

Use `scripts/init_dataset_case.py` to create:

```
datasets/<tenant>/<project>/<run>/
├── inputs/          # Original tender files (or symlinks)
├── outputs/         # Pipeline output (analysis.json, etc.)
├── playbook/        # Playbook snapshot used for this run
├── ground_truth/    # Human-verified correct answers
└── feedback/        # Estimator corrections and comments
```

```bash
python scripts/init_dataset_case.py \
  --tenant acme_construction \
  --project hospital_wing_b \
  --run 2026-02-27_full_audit
```

---

## Collection Workflow

### During Pilot Runs

1. Run pipeline as normal (`pilot_batch_ingest.py` or `pilot_ops.py`)
2. Open results in Streamlit UI
3. Estimator reviews and corrects items (feedback captured automatically)
4. Export submission pack
5. After review: `init_dataset_case.py` to scaffold, then copy/link relevant files

### After Pilot Feedback

1. Collect the official BOQ from the client → `ground_truth/`
2. Copy the `feedback.jsonl` → `feedback/`
3. Snapshot the active playbook → `playbook/`
4. Run ground truth diff (when available): `ground_truth_diff.py`

---

## What This Enables Later

| Use Case | Data Required | Status |
|----------|--------------|--------|
| **Fine-tune extraction** | Input PDFs + ground truth BOQ | Scaffold ready |
| **Improve RFI rules** | RFI feedback (correct/wrong/edited) | Capturing now |
| **Train RAG** | Paired (tender → analysis) examples | Scaffold ready |
| **Benchmark regressions** | Fixed inputs + expected metrics | `benchmarks/` ready |
| **Quality dashboards** | Aggregated feedback across pilots | `quality_dashboard.py` exists |
| **Playbook optimization** | Playbook + outcome pairs | Scaffold ready |

---

## Key Modules

| Module | Purpose |
|--------|---------|
| `src/analysis/feedback.py` | Feedback capture (correct/wrong/edited) |
| `src/analysis/ground_truth.py` | Ground truth intake |
| `src/analysis/ground_truth_diff.py` | Diff reports |
| `src/analysis/training_pack.py` | Export paired training data |
| `src/analysis/evaluation.py` | Evaluation metrics |
| `src/analysis/quality_dashboard.py` | Quality metrics over time |
| `scripts/init_dataset_case.py` | Scaffold dataset case directories |
