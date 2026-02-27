# XBOQ Improvement Log

Track issues found during validation and their resolution.
Use `python scripts/log_issue.py` to add entries.

---

## Quick Add Command

```bash
python scripts/log_issue.py --project <id> --type <TYPE> --desc "description"
```

---

## Issue Categories

| Category | Description |
|----------|-------------|
| SCALE | Scale detection errors (wrong ratio, missed scale bar) |
| ROOM | Room detection errors (missed room, wrong label, wrong boundary) |
| AREA | Area calculation errors (wrong sqm, deduction issues) |
| OPENING | Door/window detection errors |
| QTY | Quantity estimation errors (BOQ values) |
| RULE | Rule configuration issues |
| CRASH | Pipeline crashes or failures |

---

## Log Table

| Date | Project | Type | Description | Root Cause | Resolution | Status |
|------|---------|------|-------------|------------|------------|--------|
<!-- Add new entries below this line -->
| 2026-02-03 | - | - | Log initialized | - | - | - |

---

## Summary Statistics

- Total Issues: 0
- Open Issues: 0
- Resolved: 0

### By Category
- SCALE: 0
- ROOM: 0
- AREA: 0
- OPENING: 0
- QTY: 0
- RULE: 0
- CRASH: 0

---

## Resolution Patterns

### Scale Issues
- **Symptoms**: Areas 10x too large/small
- **Common causes**: OCR misread, wrong unit assumption
- **Fix**: Check `rules/scale_assumptions.yaml`

### Room Detection Issues
- **Symptoms**: Missing/merged rooms, wrong labels
- **Common causes**: Low contrast, unusual naming
- **Fix**: Update `rules/room_aliases.yaml`

### Quantity Issues
- **Symptoms**: BOQ doesn't match manual takeoff
- **Common causes**: Deduction rules, measurement method
- **Fix**: Review `rules/measurement_rules.yaml`

---

*Last Updated: 2026-02-03*

