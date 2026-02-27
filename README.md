# XBOQ Pre-Construction Engine

**India-first BOQ & Scope Tool for RCC Construction**

XBOQ is an automated Bill of Quantities (BOQ) extraction engine designed specifically for Indian residential RCC construction. It handles the complete preconstruction workflow from drawing intake to bid submission.

---

## Quick Start

### 1. Create a New Project

```bash
python scripts/new_project.py --project_id villa_whitefield
```

This creates:
```
data/projects/villa_whitefield/
├── drawings/           # Put PDF/DWG/images here
├── owner_docs/         # Specs, schedules
├── quotes/             # SC quotes
├── owner_inputs.yaml   # Fill in project info
└── project_intake.md   # Checklist
```

### 2. Add Drawings and Fill Inputs

1. Copy drawings into `drawings/` folder
2. Edit `owner_inputs.yaml` with project details
3. Complete `project_intake.md` checklist

### 3. Run the Full Pipeline

```bash
python run_full_project.py --project_id villa_whitefield --profile typical
```

**Profiles:**
- `conservative` - Lower estimates (0.8x)
- `typical` - Standard Indian residential
- `premium` - Higher-end specs (1.2x)

### 4. Review Outputs

```
output/villa_whitefield/
├── boq/
│   ├── boq_output.csv          # Complete BOQ
│   ├── finishes_boq.csv        # Finish quantities
│   └── openings_schedule.csv   # Doors & windows
├── scope/
│   ├── room_areas.csv          # Room areas
│   └── scope_summary.json      # Scope data
├── risk/
│   ├── bid_strategy.md         # Commercial strategy
│   ├── risk_pricing.csv        # Package risks
│   └── sensitivity_report.md   # Rate sensitivity
├── bid_book/
│   ├── exclusions.md           # Exclusions
│   ├── assumptions.md          # Assumptions
│   └── clarifications.md       # Clarifications letter
├── packages/
│   └── *.csv                   # Trade packages
├── debug/
│   └── *_combined.png          # Visual overlays
└── pipeline_result.json        # Run summary
```

### 5. Validate Results

```bash
python scripts/new_validation.py --project_id villa_whitefield
# Fill in validation.csv and compare with manual takeoff
```

---

## Smoke Test

Before processing real drawings, verify the setup:

```bash
python -m src.smoke_test
```

This checks:
- All modules import correctly
- Rule files are valid YAML
- Dependencies (OpenCV, PDF support)
- Output directories writable

---

## Project Workflow

### A. Intake
```bash
python scripts/new_project.py --project_id <id>
```

### B. Process
```bash
python run_full_project.py --project_id <id>
```

### C. Validate
```bash
python scripts/new_validation.py --project_id <id>
# Compare engine values with manual takeoff
```

### D. Log Issues
```bash
python scripts/log_issue.py --project <id> --type AREA --desc "Living room 10% off"
```

### E. Version Rules
```bash
# Create snapshot before changes
python scripts/snapshot_rules.py --tag v2

# Compare versions
python scripts/diff_rules.py --from v1 --to v2
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Room Detection** | Extracts rooms, labels, areas from plans |
| **Scale Detection** | Auto-detects scale from drawings |
| **Openings Schedule** | Doors, windows, ventilators |
| **Finish Takeoff** | Floor, wall, ceiling, skirting |
| **IS 1200 Measurement** | CPWD-style deductions |
| **Formwork Derivation** | From RCC quantities |
| **Prelims Calculator** | Quantity-driven prelims |
| **Risk Pricing** | Package-wise contingency |
| **Bid Strategy** | Commercial decision support |
| **Exclusions/Assumptions** | Auto-generated bid docs |

---

## India-Specific Defaults

### Finish Templates (`rules/finish_templates.yaml`)

| Room Type | Floor | Wall | Waterproofing |
|-----------|-------|------|---------------|
| Living/Bedroom | Vitrified 600x600 | Plastic emulsion | — |
| Kitchen | Vitrified 600x600 | Dado 600mm | — |
| Toilet | Anti-skid 300x300 | Tiles 2100mm | ✅ |
| Balcony | Anti-skid 300x300 | Exterior emulsion | ✅ |
| Terrace | IPS | — | ✅ Brick bat coba |
| Pooja | Marble | Plastic emulsion | — |

### Measurement Rules (`rules/measurement_rules.yaml`)

- Plaster deduction: openings > 0.5 sqm
- Paint deduction: openings > 0.1 sqm
- Masonry deduction: openings > 0.1 sqm
- Toilet tile height: 2100mm standard
- Kitchen dado: 600mm standard

### Room Aliases (`rules/room_aliases.yaml`)

Supports regional variations:
- Kitchen: rasoi, rasoi ghar, pak ghar, aduge
- Toilet: shauchalay, snan ghar, guslkhana
- Pooja: mandir, devghar, prayer room
- Utility: servant room, maid room, dhq, ayah room

---

## File Structure

```
floorplan-engine/
├── run_full_project.py       # One-command runner
├── validation_template.csv   # Validation sheet
├── improvement_log.md        # Issue tracking
│
├── scripts/
│   ├── new_project.py        # Create project
│   ├── new_validation.py     # Create validation sheet
│   ├── log_issue.py          # Log issues
│   ├── snapshot_rules.py     # Version rules
│   └── diff_rules.py         # Compare versions
│
├── src/
│   ├── ingest.py             # PDF/DWG intake
│   ├── scale.py              # Scale detection
│   ├── export.py             # Export results
│   │
│   ├── scope/                # Scope extraction
│   ├── boq/                  # BOQ generation
│   ├── risk/                 # Risk & strategy
│   ├── bid_docs/             # Bid documents
│   ├── measurement_rules/    # IS 1200 rules
│   ├── debug/                # Debug overlays
│   └── smoke_test.py         # System check
│
├── rules/
│   ├── room_aliases.yaml     # Room name mappings
│   ├── finish_templates.yaml # Finish specs
│   ├── measurement_rules.yaml # Deduction rules
│   ├── scale_assumptions.yaml # Scale defaults
│   ├── rate_library.yaml     # DSR/CPWD rates
│   └── snapshots/            # Rule versions
│       └── v1/               # Initial snapshot
│
├── data/projects/            # Project data
└── output/                   # Pipeline outputs
```

---

## BOQ Schema

| Column | Description |
|--------|-------------|
| `item_code` | Unique ID (e.g., FLR-VIT-01) |
| `description` | Full description |
| `qty` | Quantity |
| `unit` | sqm, cum, rmt, nos, kg |
| `rate` | Unit rate (INR) |
| `amount` | qty × rate |
| `package` | Trade package |
| `confidence` | 0.0 - 1.0 |
| `derived_from` | Data source |

---

## Bid Strategy Output

The bid strategy sheet (`risk/bid_strategy.md`) provides:

1. **Safe Packages** - Low risk, price aggressively (5% margin)
2. **Risky Packages** - Protect margin (12-15%)
3. **Quote Requirements** - SC quotes needed before submission
4. **Top 10 Risk Drivers** - What could go wrong
5. **Rate Sensitivity** - Steel/cement/labour impact
6. **Pricing Recommendations** - Margin guidance
7. **Go/No-Go Assessment** - Should we bid?

---

## Decision Logic & Blockers

XBOQ uses a three-tier decision system to assess bid readiness:

### Decision Levels

| Decision | Score | Meaning | Action |
|----------|-------|---------|--------|
| **PASS** | ≥70 | Reasonably complete, safe to bid | Proceed with pricing |
| **CONDITIONAL** | ≥40 | Some trades blocked, but scope is usable | Review RFIs, proceed with caution |
| **NO-GO** | <40 | Drawing set is too incomplete to bid safely | Request more drawings |

### How Decisions Are Made

1. **Score Components** (weighted average):
   - Completeness (25%): Missing dependencies penalize score
   - Measurement (20%): Scale coverage + drawing presence
   - Coverage (35%): Disciplines, sheet types, entities detected
   - Blocker (20%): Severity and bid impact of blockers

2. **Trade-Specific Blockers**:
   - Blockers only affect their specific trades (not global kill switches)
   - Example: Missing MEP drawings blocks MEP trade but not Architectural
   - Scale issues reduce confidence but don't completely block pricing

3. **CONDITIONAL triggers**:
   - At least one trade has >50% coverage
   - Some useful content detected (sheet types, doors, rooms)
   - Score between 40-70

### Blocker Types

| ID | Issue | Affected Trades | Bid Impact |
|----|-------|-----------------|------------|
| BLK-0013 | Scale not detected | All measured trades | Forces allowance |
| BLK-0003 | No MEP drawings | MEP only | Clarification needed |
| BLK-0010 | Doors without schedule | Architectural | Clarification needed |
| BLK-0011 | Windows without schedule | Architectural | Clarification needed |
| BLK-0012 | Rooms without finish schedule | Finishes | Clarification needed |
| BLK-0007 | No structural drawings | Structural | Clarification needed |

### OCR Fallback for Scanned Drawings

For PDFs with no embedded text layer (scanned drawings), XBOQ automatically uses OCR to detect:
- Scale labels (1:100, SCALE = 1:50, NTS)
- Sheet titles and numbers (A-101, FLOOR PLAN)
- Discipline markers (STRUCTURAL, ELECTRICAL)

OCR is triggered automatically when page text is empty or too short.

### Example: Garage Drawings.pdf

**Before (v1 - Global NO-GO)**:
```json
{
  "decision": "NO-GO",
  "readiness_score": 39,
  "trade_coverage": {
    "architectural": "100%",  // Wrong - template data
    "mep": "37.5%"
  }
}
```

**After (v2 - CONDITIONAL)**:
```json
{
  "decision": "CONDITIONAL",
  "readiness_score": 52,
  "trade_coverage": {
    "architectural": "67%",   // Based on actual detected content
    "structural": "0%",       // No structural drawings found
    "mep": "0%"               // No MEP drawings found
  },
  "recommendation": "Proceed with architectural scope. Request structural and MEP drawings via RFI."
}
```

The engine now extracts whatever is available and clearly indicates what's blocked vs. what can be priced.

---

## Troubleshooting

### Smoke test fails
```bash
python -m src.smoke_test --verbose
```
Check which modules/files are missing.

### Scale detection wrong
- Check `rules/scale_assumptions.yaml`
- Add explicit scale in `owner_inputs.yaml`
- Review `debug/*_scale.png` overlay

### Room labels wrong
- Update `rules/room_aliases.yaml`
- Check `debug/*_rooms.png` overlay

### Areas don't match
- Review `rules/measurement_rules.yaml`
- Check deduction thresholds
- Fill validation sheet and compare

### Pipeline crashes
```bash
python scripts/log_issue.py --project <id> --type CRASH --desc "Error message"
```

---

## Development

### Run Tests
```bash
python -m pytest tests/ -v
```

### Add Room Alias
1. Edit `rules/room_aliases.yaml`
2. Add to appropriate category
3. Snapshot: `python scripts/snapshot_rules.py --tag v2`

### Add Finish Template
1. Edit `rules/finish_templates.yaml`
2. Add room type with specs
3. Add CPWD mapping to `rates/cpwd_mapping.csv`

---

## License

Proprietary - All Rights Reserved

## Support

Contact: support@xboq.in
