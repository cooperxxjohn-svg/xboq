# BID SUMMARY: DEMO-2024-001

**Project**: Residential Villa - Whitefield, Bangalore
**Generated**: 02-Feb-2026 22:34

---
## 🟡 BID STATUS: SUBMITTABLE WITH RESERVATIONS

**Review reservations before submission.**
---

## KEY METRICS

| Metric | Value |
|--------|-------|
| **BOQ Total** | ₹11,830,440.00 |
| **Prelims** | ₹4,336,500.19 (36.7%) |
| **Grand Total** | ₹16,166,940.19 |
| **Built-up Area** | 4,500 sqm (48,438 sqft) |
| **Rate per Sqft** | ₹333.77 |
| **Gate Score** | 62.7/100 |
| **Reservations** | 6 |
| **High Priority RFIs** | 3 |
| **Provisional Value** | ₹1,570,000.00 |

## BOQ SUMMARY BY PACKAGE

| Package | Items | Value (INR) | % of Total |
|---------|-------|-------------|------------|
| MASONRY | 4 | ₹4,140,600.00 | 35.0% |
| RCC | 4 | ₹3,247,490.00 | 27.5% |
| FLOORING | 3 | ₹1,029,750.00 | 8.7% |
| PLUMBING | 1 | ₹850,000.00 | 7.2% |
| DOORS_WINDOWS | 4 | ₹837,400.00 | 7.1% |
| ELECTRICAL | 1 | ₹720,000.00 | 6.1% |
| EXTERNAL | 2 | ₹589,500.00 | 5.0% |
| WATERPROOF | 2 | ₹415,700.00 | 3.5% |
| **TOTAL** | **21** | **₹11,830,440.00** | **100%** |

## TOP RESERVATIONS

1. 🔴 **RES-001**: MEP drawings not provided - plumbing and electrical on provisional allowance
2. 🟠 **RES-002**: Scale confidence 78% on floor plans - quantities may vary
3. 🟠 **RES-003**: Approval drawings only - not GFC (Good For Construction)
4. 🔴 **RES-004**: Soil investigation report not provided - foundation design unverified
5. 🟠 **RES-005**: External works scope partially defined

## HIGH PRIORITY RFIs

- **RFI-001**: Confirm column C5 dimensions - mismatch between structural (450x450) and architectural (400x400) drawings
- **RFI-002**: Provide MEP drawings for plumbing and electrical layout - currently using provisional allowances
- **RFI-005**: Provide soil investigation report for foundation design verification

## PHASE STATUS

| Phase | Description | Status |
|-------|-------------|--------|
| PHASE16 | Owner Docs Parser | ⏭️ Skipped |
| PHASE17 | Owner Inputs Engine | ✅ Completed |
| PHASE18 | BOQ Alignment | ⏭️ Skipped |
| PHASE19 | Pricing Engine | ✅ Completed |
| PHASE20 | Quote Leveling | ⏭️ Skipped |
| PHASE21 | Prelims Generator | ✅ Completed |
| PHASE22 | Bid Book Export | ✅ Completed |
| PHASE23 | Bid Gate | ✅ Completed |
| PHASE24 | Clarifications Letter | ✅ Completed |
| PHASE25 | Package Outputs | ✅ Completed |

## OUTPUT LOCATIONS

```
output/bid_demo/
├── summary.md                    # This file
├── bid_engine_results.json       # Complete results
├── phase17_owner_inputs/
├── phase19_pricing/
│   ├── priced_boq.json
│   └── rate_analysis.json
├── phase21_prelims/
├── phase22_bidbook/
│   ├── bid_summary.json
│   └── priced_boq.csv
├── phase23_bid_gate/
│   ├── bid_gate_report.md
│   └── bid_gate_report.json
├── phase24_clarifications/
│   └── clarifications_letter.md
└── phase25_packages/
    ├── rcc/
    ├── masonry/
    ├── flooring/
    └── ...
```

## NEXT STEPS

1. ✅ Review reservations in `bid_gate_report.md`
2. ✅ Document all reservations in clarifications letter
3. 🔄 Obtain subcontractor quotes from `packages/` RFQ sheets
4. 🔄 Proceed with submission, noting reservations
