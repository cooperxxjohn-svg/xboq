# Benchmarks

Regression benchmark scaffolding for xBOQ pipeline.

**No real tender data is committed to this repo.**

---

## Setup

### 1. Create a manifest

```bash
cp benchmarks/manifest.example.json benchmarks/manifest.json
```

Edit `manifest.json` to point to your local tender files:

```json
{
  "cases": {
    "hospital_300pg": {
      "paths": ["/path/to/hospital_tender.pdf"],
      "mode": "standard_review"
    },
    "residential_50pg": {
      "paths": ["/path/to/residential.pdf"],
      "mode": "demo_fast"
    }
  }
}
```

### 2. Place tender assets

Tender PDFs can live anywhere on your machine. The manifest just references
absolute paths. Do NOT commit tender files to this repo.

Recommended location: `~/xboq_benchmarks/` (outside repo).

### 3. Set expected metrics (optional)

Edit `benchmarks/expected_metrics.yaml` to define pass/fail thresholds per case:

```yaml
hospital_300pg:
  min_boq_items: 50
  min_commercial_terms: 5
  min_requirements: 20
  min_rfis: 15
  max_toxic_pages: 5
```

### 4. Run the regression suite

```bash
python scripts/run_regression.py
```

If `manifest.json` is missing, the script prints setup instructions and exits 0.

---

## Output

Results are saved to `benchmarks/_runs/<case_name>/<timestamp>/`:
- `payload.json` — full pipeline output
- `regression_report.csv` — KPI summary with PASS/WARN/FAIL

The `_runs/` directory is gitignored by default.

---

## KPIs Tracked

| KPI | Description |
|-----|-------------|
| `pages_total` | Total pages in document |
| `deep_processed_pages` | Pages fully extracted |
| `ocr_pages` | Pages requiring OCR |
| `boq_items_count` | BOQ line items extracted |
| `commercial_terms_count` | Commercial terms parsed |
| `requirements_count` | Requirements captured |
| `rfis_count` | RFIs generated |
| `blockers_count` | Blocking issues found |
| `toxic_pages_count` | Pages that failed extraction |
| `qa_score` | Overall quality score |
| `boq_pages_attempted` | Pages routed to BOQ extractor |
| `boq_pages_parsed` | Pages that yielded BOQ items |
| `schedule_pages_attempted` | Pages routed to schedule extractor |
| `schedule_pages_parsed` | Pages that yielded schedule rows |
