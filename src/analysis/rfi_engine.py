"""
RFI Engine — Checklist-driven RFI generation with evidence.

Evaluates a discipline checklist (~30-50 items) against:
- ExtractionResult (requirements, schedules, BOQ items, callouts)
- PageIndex (page classification counts)
- SelectedPages (what was analyzed)
- PlanSetGraph (entities, aggregates)

Two check types:
1. Missing-field checks: required info not found
2. Conflict checks: contradictions across pages

Every RFI uses the existing RFIItem model with full EvidenceRef.
Target: >15 RFIs on a 300+ page tender.
"""

import os
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict

from src.models.analysis_models import (
    RFIItem, EvidenceRef, Trade, Severity,
    PlanSetGraph, create_rfi_id,
    RunCoverage, CoverageStatus, SelectionMode,
)

# Imported lazily to avoid circular imports — only used in generate_rfis()
# from .package_classifier import PackageType, PackageClassification

from .page_index import PageIndex
from .page_selection import SelectedPages
from .extractors import ExtractionResult

DEBUG = os.environ.get("DEBUG_PIPELINE", "0") == "1"


# =============================================================================
# CHECKLIST DEFINITION
# =============================================================================

# Each check: (check_id, check_fn_name, trade, default_priority, question_template, why_template)
# check_fn returns: (should_fire: bool, evidence: EvidenceRef)
CHECKLIST = [
    # ---- Architectural ----
    ("CHK-A-001", "chk_door_schedule_missing", Trade.ARCHITECTURAL, Severity.HIGH,
     "Please issue a complete Door Schedule for all {n_tags} door openings identified on the plan drawings. "
     "The schedule shall specify for each door mark: (1) opening dimensions — clear width × height in mm, "
     "(2) door leaf type (flush solid-core / panelled / full-glazed / louvred / fire-rated), "
     "(3) core construction and leaf thickness in mm, "
     "(4) frame material and profile (pressed steel / hardwood / aluminium / UPVC) with section dimensions, "
     "(5) complete ironmongery set (mortice lock grade, lever handle pattern, overhead closer EN rating, "
     "floor stopper, tower bolt, vision panel if any, kick plate where applicable), "
     "(6) surface finish on both faces (paint grade / veneer species / HPL laminate / PU coat), and "
     "(7) acoustic rating (dB) and/or fire rating (FD30 / FD60 / FD90) where required by the NBC or client brief.",
     "A standard flush door (Rs 4,500–7,000) vs a fire-rated steel door (Rs 18,000–35,000) carries a "
     "300–500% cost difference per opening. With {n_tags} unspecified openings, the unclarified cost "
     "exposure is significant and pricing accuracy cannot be achieved without this schedule."),

    ("CHK-A-002", "chk_window_schedule_missing", Trade.ARCHITECTURAL, Severity.HIGH,
     "Please issue a complete Window Schedule for all {n_tags} window openings identified on the plan drawings. "
     "The schedule shall specify for each window mark: (1) opening dimensions — clear width × height in mm, "
     "(2) frame system (aluminium section series / UPVC profile class / steel / timber) with wall-thickness "
     "and transom/mullion details, "
     "(3) glass specification (thickness in mm, type: float / toughened / laminated / reflective / DGU with "
     "air-gap), (4) operation type per panel (fixed / sliding / top-hung / bottom-hung / casement / louvred), "
     "(5) hardware (lever handle, espagnolette, friction stay arm, tilt-turn fittings as applicable), "
     "(6) sealant system and weather-strip detail, and "
     "(7) acoustic / thermal / solar performance class where the brief specifies HVAC load or noise criteria.",
     "Window and glazing costs range from Rs 650–2,200 per sqft depending on specification. "
     "With {n_tags} unspecified openings, pricing ranges can vary by Rs 8–25 lakhs on a typical mid-rise "
     "project — making the window schedule a critical pricing document."),

    ("CHK-A-003", "chk_finish_schedule_missing", Trade.ARCHITECTURAL, Severity.HIGH,
     "Please issue a complete Room Finish Schedule for all {n_rooms} rooms and spaces identified on the drawings. "
     "For each room or space, the schedule shall specify: "
     "(1) Floor finish — material (tile / stone / timber / epoxy / carpet), tile or stone size and grade/series, "
     "adhesive type, and any surface treatment (polished / honed / anti-slip); "
     "(2) Wall finish — plaster type and thickness, paint system (primer + number of top coats, sheen level), "
     "or tile/stone specification with grout colour and joint width where applicable; "
     "(3) Ceiling finish — soffit plaster / gypsum board (thickness and grid) / mineral fibre / exposed RCC, "
     "finished ceiling height from FFL in mm; and "
     "(4) Any special finishes — anti-static flooring, chemical-resistant coating, acoustic treatment, "
     "feature wall material, or wet-area waterproofing membrane beneath tiles.",
     "Finish costs range from Rs 350/sqft (standard emulsion walls, vitrified floor) to Rs 1,500+/sqft "
     "(imported marble / stone cladding / teak flooring). Without a finish schedule for {n_rooms} rooms, "
     "bidders assume a single specification — causing a 3–4× spread in finish pricing across different "
     "contractors and making bid comparison impossible."),

    ("CHK-A-004", "chk_no_general_notes", Trade.ARCHITECTURAL, Severity.MEDIUM,
     "No General Notes or Specification Notes drawing has been identified in the document set. "
     "Please provide a General Notes sheet or project specification document that covers at minimum: "
     "(1) applicable Indian Standards and codes (IS 456, IS 1200, IS 2212, NBC 2016 as applicable), "
     "(2) approved materials list with minimum grades — concrete grade, mortar mix ratio, brick/block "
     "strength class, steel grade, paint brand tier; "
     "(3) minimum workmanship requirements — surface tolerances, curing periods, test frequencies; "
     "(4) inspection and hold-point schedule for structural elements; "
     "(5) concrete mix design basis (nominal / design mix) and minimum cube test frequency; and "
     "(6) any project-specific deviations from IS standards or client special requirements.",
     "Without general notes, each contractor independently assumes material grades and workmanship "
     "standards — leading to non-comparable bids and significant post-award disputes when the contractor's "
     "assumed grade differs from the client's expectation."),

    ("CHK-A-005", "chk_no_legend", Trade.ARCHITECTURAL, Severity.MEDIUM,
     "No Drawing Legend or Abbreviations key has been identified in the drawing set. "
     "Please provide a Legend sheet that defines: "
     "(1) all material hatching patterns used in plans, sections, and details with their material meaning "
     "(e.g., brick hatch = 230mm BW, diagonal hatch = RCC, dot pattern = PCC); "
     "(2) all standard symbols used across the set (door swing convention, window type markers, "
     "sanitary fixture symbols, electrical point symbols, structural section marks); "
     "(3) all abbreviations used in drawing notes and schedules "
     "(e.g., FFL = finished floor level, SFL = structural floor level, GL = ground level, "
     "THK = thickness, DPC = damp proof course, BW = brick wall); and "
     "(4) revision cloud convention and drawing status codes.",
     "Misinterpretation of hatching (e.g., treating an RCC element as masonry) or abbreviations "
     "causes scope misquotes — particularly affecting structural, waterproofing, and external works "
     "where material confusion directly drives quantity and rate errors."),

    ("CHK-A-006", "chk_room_dimensions_missing", Trade.ARCHITECTURAL, Severity.MEDIUM,
     "Room dimensions are missing or incomplete — {n_rooms} rooms have been identified but only "
     "{n_dims} dimension callouts are present across the plan pages. "
     "Please issue revised plan drawings that include for every room and space: "
     "(1) internal clear dimensions — width × depth measured from finished face of wall, "
     "(2) door and window centring dimensions referenced from the nearest column or wall face, "
     "(3) sill height and lintel/head height for all openings (measured from FFL), "
     "(4) floor-to-floor and floor-to-ceiling clear heights shown on at least one section per floor, "
     "(5) any critical hold dimensions for built-in furniture, equipment plinths, or service trenches, and "
     "(6) overall building dimensions (external envelope) on the ground floor plan.",
     "Area take-off errors from missing dimensions typically cause ±10–20% variance in floor-finish, "
     "wall-plaster, and false-ceiling quantities. On a 2,000 sqm floor plate, this translates to a "
     "pricing error of Rs 3–12 lakhs per floor — compounding across multiple levels."),

    # ---- Structural ----
    ("CHK-S-001", "chk_no_structural", Trade.STRUCTURAL, Severity.HIGH,
     "No structural drawings have been identified in the provided document set. "
     "Please issue the complete Structural Drawing Package, which shall include at minimum: "
     "(1) Foundation Plan showing all footing / pile locations, cap dimensions, and founding level, "
     "with a Footing Schedule (mark, plan size L×B, depth below GL, reinforcement bars and ties, "
     "concrete grade, and applicable SBC assumption); "
     "(2) Column Layout Plan with Column Schedule (mark, plan dimensions, reinforcement at each "
     "floor level, splice locations, and concrete grade per zone); "
     "(3) Beam Layout Plan with Beam Schedule (mark, depth × width, top bars, bottom bars, "
     "stirrup dia and spacing at supports and midspan); "
     "(4) Slab Schedule (thickness, main bar dia and spacing, distribution bar, top reinforcement "
     "over supports, and any post-tensioning or flat-slab punching shear details); "
     "(5) Staircase structural details with going, riser, waist slab thickness, and reinforcement; and "
     "(6) Any special elements — retaining walls, shear walls, transfer beams, cantilevers — with "
     "full reinforcement details.",
     "RCC and structural steelwork represent 25–35% of total project cost. Without structural drawings, "
     "the entire structural trade must be carried as a provisional sum — making the bid technically "
     "incomplete and exposing the client to unrealistic pricing assumptions."),

    ("CHK-S-002", "chk_no_foundation", Trade.STRUCTURAL, Severity.MEDIUM,
     "Foundation details have not been found in the drawing set. "
     "Please provide the following foundation documentation: "
     "(1) Foundation Plan clearly showing footing / pile type, positions, and mark references; "
     "(2) Footing Schedule with mark, plan dimensions (L×B or diameter), depth below existing GL, "
     "reinforcement (main bars + ties), and design concrete grade; "
     "(3) Plinth beam / ground beam schedule with size and reinforcement; "
     "(4) Founding level SBC assumption and the soil investigation / geotechnical report reference; "
     "(5) Details of any hard standing, pedestal, haunch, or under-reaming requirement; and "
     "(6) Anti-termite treatment specification and waterproofing / DPC details at plinth level.",
     "Foundation type (isolated footing vs. raft vs. pile) drives excavation depth, shuttering "
     "complexity, and concrete volumes. A 500mm error in assumed founding depth changes earthwork "
     "quantities by 20–40% on a typical residential block — a cost difference of Rs 5–15 lakhs."),

    ("CHK-S-003", "chk_bbs_missing", Trade.STRUCTURAL, Severity.MEDIUM,
     "A Bar Bending Schedule (BBS) has not been found in the document set. "
     "Please either provide the BBS or clarify the contractual position on its preparation: "
     "(1) If owner/consultant-supplied: provide the complete BBS for all structural elements, "
     "listing for each mark — bar diameter, number of bars, cutting length (mm), shape code "
     "(as per IS 2502), and total weight (kg) per element type; "
     "(2) If contractor-prepared: confirm (a) the applicable IS measurement standard (IS 1200 Part 16), "
     "(b) the permitted wastage and lapping allowance percentage, "
     "(c) whether hooks, cranks, chairs, and spacers are to be included in the quoted weight, and "
     "(d) the steel grade and source (TISCO / JSW / Vizag / approved equivalent) to be applied.",
     "Reinforcement steel is typically 3.5–5.5 kg/sqft BUA and represents 15–25% of project cost. "
     "Without a BBS, contractors use thumb-rule rates that vary by ±30% — creating major pricing "
     "uncertainty that leads to either inflated bids or post-award steel weight disputes."),

    # ---- MEP ----
    ("CHK-M-001", "chk_no_mep", Trade.MEP, Severity.MEDIUM,
     "No Mechanical, Electrical, or Plumbing (MEP) drawings have been identified in the document set. "
     "Please issue the complete MEP drawing package or clarify the scope boundary between the civil "
     "contract and any separate MEP packages. If MEP is within this contract scope, provide: "
     "(1) Electrical Single Line Diagram (SLD) showing supply source, metering, main LT panel (MDB), "
     "sub-distribution boards (SDBs/DBs), feeder cable sizes, and fault levels; "
     "(2) Lighting and power layout per floor showing DB locations, conduit routing, and point schedule; "
     "(3) Plumbing riser diagram and floor layout showing pipe sizes, material, and fixture connections; "
     "(4) Soil, waste, and vent (SWV) drainage riser diagram; "
     "(5) HVAC layout or ventilation schedule (if in scope); and "
     "(6) Fire-fighting riser schematic and floor layout.",
     "MEP typically represents 25–40% of a building project's total cost. Without these drawings, "
     "the entire MEP scope must be carried as a provisional sum — the single largest source of "
     "post-contract variation orders and cost disputes on building projects."),

    ("CHK-M-002", "chk_no_electrical", Trade.ELECTRICAL, Severity.MEDIUM,
     "No electrical layout drawings have been identified. Please provide the complete electrical drawing set: "
     "(1) Single Line Diagram (SLD) showing supply authority metering point, main LT panel, "
     "all sub-distribution boards with their floor / zone allocation, feeder cable sizes, "
     "MCB / MCCB / RCCB ratings, and connected load per DB; "
     "(2) Floor-wise layout plans showing: conduit routing (surface / concealed, with dia), "
     "DB location, lighting point positions and circuit allocation, power outlet positions "
     "(16A, 6A, 3-pin industrial), data and telephone point positions, and safety points "
     "(earth leakage, emergency lighting, fire alarm call points); "
     "(3) External yard lighting layout, earthing layout, and lightning protection details; "
     "(4) DB schedules listing every way — circuit description, conductor size, MCB rating, "
     "RCCB grouping, and connected load in kW; and "
     "(5) Cable schedule for all feeders above 6 sqmm — conductor size, insulation type, "
     "conduit / tray / direct burial, and route length.",
     "Electrical work ranges from Rs 45–90/sqft for standard residential to Rs 150–250/sqft "
     "for commercial or healthcare. Without drawings, the scope cannot be substantiated and "
     "the contractor carries 100% of scope risk — reflected as a high contingency in the bid."),

    ("CHK-M-003", "chk_no_plumbing", Trade.PLUMBING, Severity.MEDIUM,
     "No plumbing and drainage drawings have been identified. Please provide: "
     "(1) Water supply riser diagram showing incoming supply connection, break-pressure tank, "
     "overhead storage tank (capacity in kL), booster pump set, and distribution to each floor "
     "with pipe sizes (dia × material: GI / CPVC / PPR) and isolation valve positions; "
     "(2) Floor-wise plumbing layout showing hot and cold water pipe routing, "
     "individual fixture connections, and floor drain positions; "
     "(3) Soil, Waste, and Vent (SWV) riser diagram showing pipe sizes (dia × material: PVC-SWR / CI), "
     "vent pipe termination height, and clean-out access positions; "
     "(4) Site drainage layout showing gully traps, inspection chambers (IC) / manholes with "
     "invert levels, catch pits, and connection to external municipal sewer or on-site STP; and "
     "(5) Sanitary fixture schedule confirming brand, model number, flush valve / cistern capacity, "
     "and tap / mixer flow rate for each fixture type.",
     "Plumbing work is typically 4–7% of project cost. Pipe material (CPVC vs. PPR vs. GI), "
     "riser design, and fixture grade significantly affect pricing — an unspecified riser design "
     "can cause ±40% variation in plumbing cost and create post-award disputes on material substitutions."),

    ("CHK-M-004", "chk_no_hvac", Trade.MEP, Severity.LOW,
     "No HVAC or mechanical ventilation drawings have been identified. "
     "Please clarify and provide the following: "
     "(1) Confirm whether HVAC is included in this contract or is a separate specialised package; "
     "(2) If in scope — provide equipment schedule (unit type: split / cassette / AHU / VRF, "
     "capacity in TR or kW, brand tier, outdoor unit position); "
     "ductwork layout with duct sizes (mm), diffuser / grille positions, and static pressure basis; "
     "chilled water / refrigerant pipe routing and insulation details; "
     "(3) Fresh air and exhaust ventilation layout for toilets, basements, kitchens, and server rooms "
     "(if applicable) with fan duty points (CMH × Pa); and "
     "(4) Room-wise cooling load calculation or design basis report confirming U-values and internal loads.",
     "HVAC, if in scope, typically represents 8–15% of project cost. Uncertainty on equipment capacity "
     "and ductwork complexity creates a pricing range of ±50%, making it essential to define the scope "
     "boundary clearly before bids are submitted."),

    ("CHK-M-005", "chk_no_fire", Trade.MEP, Severity.MEDIUM,
     "No fire fighting or fire protection system drawings have been identified. Please provide: "
     "(1) Fire-fighting riser schematic showing pump room location, underground fire reserve sump "
     "(capacity in kL), overhead tank, jockey pump, main duty pump and standby pump specifications, "
     "main riser dia, landing valves, and zone valve positions; "
     "(2) Floor-wise layout showing sprinkler head positions (ordinary hazard / extra hazard, "
     "with heads-per-zone count and coverage area), hose reel cabinet positions, "
     "internal and external fire hydrant positions, and manual call point locations; "
     "(3) Fire detection and alarm layout — smoke detector, heat detector, and "
     "public address / voice evacuation panel location per floor; "
     "(4) Gaseous suppression system details for server rooms, LT rooms, or generator enclosures "
     "if applicable (agent type, nozzle positions, cylinder room location); and "
     "(5) Applicable NBC / TAC / local fire NOC norms to confirm system specification basis.",
     "Full sprinkler system vs. hydrant-only vs. combined gaseous suppression represents a 3× cost "
     "difference. Without layouts, a provisional sum leads to disputes at detailed pricing stage — "
     "fire systems are typically 2–5% of project cost and cannot be reliably allowanced without scope clarity."),

    # ---- Cross-discipline ----
    ("CHK-X-001", "chk_scale_missing", Trade.GENERAL, Severity.MEDIUM,
     "Scale notation is missing or illegible on {n_missing} of {n_total} drawing pages. "
     "Please reissue these drawings with one of the following: "
     "(1) A printed scale bar and/or numeric scale ratio on every sheet "
     "(e.g., 1:100, 1:50, 1:20 — confirmed for the stated print size A0/A1/A2); "
     "(2) OR a written confirmation that all required dimensions are fully annotated on the drawings "
     "and that physical scale measurement is not needed for quantity take-off; "
     "AND confirm the intended print size for each affected sheet so that any physical scale bars "
     "can be applied correctly if drawings are printed at other than 1:1 PDF scale.",
     "Area and length measurements taken from unscaled drawings carry errors of 15–30% depending "
     "on print size. A 10% area error on a 5,000 sqm project translates to a Rs 20–50 lakh pricing "
     "error in floor finishes and false ceiling alone — across {n_missing} unscaled sheets."),

    ("CHK-X-002", "chk_no_site_plan", Trade.CIVIL, Severity.MEDIUM,
     "No site plan or key plan has been found in the document set. Please provide: "
     "(1) Site boundary plan showing plot dimensions, total plot area, road frontage, "
     "and orientation (north point); "
     "(2) Statutory setbacks and FAR / ground coverage compliance diagram; "
     "(3) Approach roads, entry/exit gate positions, and security cabin location; "
     "(4) Existing utilities on and around the site — HT/LT overhead or underground lines, "
     "water main (size and depth), sewer manhole positions, telecom ducts — with clearance dimensions; "
     "(5) External development scope — compound wall (length, height, type), paving area (sqm), "
     "landscaping zone, storm-water drainage channels, sump / STP location; and "
     "(6) Point of supply connections — power transformer / HT room position, water supply tap-off "
     "point, and sewer connection chamber — with approximate route lengths to the building.",
     "External works are commonly 5–15% of project cost on standalone projects. Without a site plan, "
     "compound wall lengths, road areas, utility connection lengths, and external drainage cannot be "
     "quantified — leading to lump-sum provisional allowances that frequently become major variation orders."),

    ("CHK-X-003", "chk_no_sections", Trade.GENERAL, Severity.MEDIUM,
     "No section drawings (longitudinal or transverse) have been found in the drawing set. "
     "Please provide at least one section through each principal building zone showing: "
     "(1) Floor-to-floor height at every level — measured from structural floor level (SFL) to SFL, "
     "and confirmed finished floor-to-ceiling clear height; "
     "(2) Structural slab thickness and principal beam depths (affects concrete volume and shuttering); "
     "(3) Parapet height above terrace floor level and coping detail; "
     "(4) Staircase profile — going, riser, number of flights per floor, headroom clearance; "
     "(5) Plinth height from natural ground level and approach ramp / step detail; and "
     "(6) Terrace parapet height, roof slope (if applicable), and any terrace garden build-up detail.",
     "Floor-to-floor height controls column lengths, shuttering area, and wall-plaster quantities. "
     "A 100mm height error per floor on an 8-storey building compounds to 800mm — affecting RCC, "
     "masonry, and finishing quantities by 5–8% per floor and potentially Rs 20–50 lakhs at project level."),

    ("CHK-X-004", "chk_no_elevations", Trade.ARCHITECTURAL, Severity.LOW,
     "No elevation drawings have been found. Please provide elevations of all building faces showing: "
     "(1) Overall building height from FGL to top of parapet / coping in mm; "
     "(2) External wall finish material for each zone — texture paint / sand-faced plaster / "
     "stone cladding / ACP panel / curtain wall / face brick — with demarcation lines and dimensions; "
     "(3) Window and door head / sill levels, lintel projection depth and underside level; "
     "(4) Sunshade / chajja projection in mm and soffit level; "
     "(5) Any external architectural features — canopy, signage band, cornice, louvre grille, "
     "external railing — with dimensions and material; and "
     "(6) External cladding fixing system detail (if ACP, stone, or GRC — confirm sub-frame system).",
     "External elevation finishes range from Rs 45/sqft (two-coat weather-coat paint) to "
     "Rs 600+/sqft (natural stone cladding on sub-frame). Without elevations, the external finishes "
     "scope — typically 3–8% of project cost — cannot be quantified or priced accurately."),

    ("CHK-X-005", "chk_no_boq", Trade.GENERAL, Severity.HIGH,
     "No Bill of Quantities (BOQ) document has been identified in the tender set. "
     "Please confirm one of the following and provide the relevant document: "
     "(1) Priced BOQ contract — issue the complete BOQ in the prescribed format with all trade "
     "sections, item descriptions, units, and quantities filled in, for contractors to rate only; "
     "(2) Open / unpriced BOQ contract — issue the BOQ format and confirm that contractors are "
     "required to prepare their own quantities from drawings; "
     "(3) Lump-sum tender — confirm that the total price is based on drawings and specification "
     "only, with no BOQ, and specify the measurement standard to be used for variation valuation. "
     "In all cases, confirm the trade-wise breakdown required and whether the contractor's BOQ "
     "will form part of the signed contract document.",
     "Without a BOQ format, each contractor structures their bid differently — making bid comparison "
     "impossible, enabling scope gaps between packages, and creating a high risk of variation orders "
     "for items the client assumed were included but no contractor priced."),

    ("CHK-X-006", "chk_boq_missing_quantities", Trade.GENERAL, Severity.HIGH,
     "The BOQ contains {n_missing} items (out of {n_total} total) where the quantity column is "
     "blank, zero, or illegible. Please reissue the BOQ with confirmed quantities for all items, "
     "supported by measurement sheets or quantity take-off schedules from which the quantities "
     "were derived. Where any item is intentionally provisional (e.g., contingency, PC sum, "
     "or owner-supply item), label it explicitly as 'Provisional' or 'PC Item' and state the "
     "basis of the estimated quantity (e.g., 'estimated from tender drawings, subject to remeasure').",
     "Unquantified BOQ items force contractors to either count quantities themselves from drawings "
     "(adding bid cost and error risk) or include a wide contingency. {n_missing} unquantified items "
     "may represent 10–30% of total contract value depending on which trades are affected."),

    ("CHK-X-007", "chk_no_conditions", Trade.GENERAL, Severity.MEDIUM,
     "No General Conditions of Contract (GCC), Special Conditions of Contract (SCC), or "
     "Particular Conditions document has been found in the tender set. "
     "Please issue the complete contract conditions document covering: "
     "(1) Payment mechanism — milestone-based / running account (RA) bills / monthly interim, "
     "with payment certification period and payment period after certification; "
     "(2) Retention money percentage, retention cap, and conditions for first and second moiety release; "
     "(3) Liquidated damages rate and cap; "
     "(4) Defect Liability Period duration and contractor obligations during DLP; "
     "(5) Insurance requirements (CAR policy, WC, third-party liability); "
     "(6) Force majeure definition and relief provisions; "
     "(7) Dispute resolution procedure (conciliation / arbitration / court jurisdiction); and "
     "(8) Any special site conditions — access restrictions, noise curfew, safety induction requirements, "
     "or client-supplied materials list.",
     "Contract conditions collectively represent 3–7% of contract value in direct cost terms "
     "(retention lock-up, insurance premiums, PBG charges, DLP overheads). Bidding without knowing "
     "these terms forces contractors to include risk premiums or makes serious bidders decline to participate."),

    ("CHK-X-008", "chk_no_addendum", Trade.GENERAL, Severity.LOW,
     "No addenda or corrigenda to the original tender documents have been identified. "
     "Please confirm and provide: "
     "(1) Whether any addenda, corrigenda, or pre-bid meeting minutes have been issued "
     "since the original tender documents were released — and if so, provide all such documents "
     "with their issue dates and the specific clauses / drawings they supersede; "
     "(2) The current revision status of all tender drawings — provide a drawing register "
     "confirming each drawing number, title, and current revision code; and "
     "(3) The final cut-off date after which no further pre-bid amendments will be issued, "
     "so that contractors can confirm they are pricing the current and complete scope.",
     "Bidding on superseded documents is one of the most common causes of post-award contractor "
     "claims. Without confirmation that all addenda have been received and that the drawing register "
     "is current, contractors cannot certify that their bid reflects the latest scope — "
     "creating a legal and commercial risk for both client and bidder."),

    # ---- Schedule field checks ----
    ("CHK-SCH-001", "chk_schedule_missing_sizes", Trade.ARCHITECTURAL, Severity.HIGH,
     "{n_missing} door or window schedule items have a mark reference but no opening size has been "
     "specified in the schedule or confirmed on the drawings. "
     "Please provide confirmed opening sizes — clear width × clear height in mm — for each of these "
     "{n_missing} items. Where a standard IS module size applies (per IS 1948 for timber or "
     "IS 1081 for steel), confirm the IS module reference and nominal dimension. "
     "Where sizes vary on different floors or zones, specify per-floor dimensions clearly. "
     "If any opening is non-standard (wider than 1800mm or taller than 2400mm), confirm "
     "the structural lintel or transom arrangement that supports the opening.",
     "Opening size is the primary variable for door and window pricing. A 900×2100mm standard "
     "door (Rs 5,000–8,000) versus a 1200×2400mm premium door (Rs 12,000–22,000) represents "
     "a 2–3× cost difference per opening — multiplied across {n_missing} unspecified items."),

    ("CHK-SCH-002", "chk_schedule_missing_qty", Trade.ARCHITECTURAL, Severity.MEDIUM,
     "{n_missing} door or window schedule items have a mark reference and size but no confirmed "
     "quantity (number of units) has been stated. "
     "Please confirm the total count for each mark type, cross-referenced to the floor plans. "
     "For marks that appear on multiple floors, state the per-floor count and the cumulative total. "
     "For 'typical floor' marks, confirm the total number of floors to which the typical condition applies. "
     "If any mark type was identified on a drawing but is now omitted, confirm whether it has been "
     "superseded by another mark or deleted from scope.",
     "Missing quantities prevent reliable cost estimation for those item types. Contractors must either "
     "recount from drawings (adding bid preparation cost and error risk) or include contingencies — "
     "both leading to uncompetitive or risk-laden bids."),

    # ---- Conflict checks ----
    ("CHK-C-001", "chk_conflicting_material_specs", Trade.GENERAL, Severity.HIGH,
     "Conflicting material specifications have been detected across the tender documents: {conflicts}. "
     "Please issue a written Clarification Note confirming: "
     "(1) The governing specification for each conflict — identified by document reference "
     "(drawing number / BOQ item number / specification clause number); "
     "(2) Whether a formal Amendment or Addendum will be issued to supersede the non-governing "
     "document, or whether the Clarification Note itself constitutes the amendment; "
     "(3) If a rate revision is required for any BOQ item affected by the specification clarification "
     "(e.g., if the governing spec is a higher-grade material than one of the conflicting documents "
     "specified), provide the revised rate basis or request re-pricing; and "
     "(4) The deadline for submitting revised rates if re-pricing is required.",
     "Conflicting specifications force contractors to either bid the higher (safer but uncompetitive) "
     "specification or the lower (competitive but risky) — creating a systematic bias in bid evaluation. "
     "Without clarification, post-award disputes over the correct specification are almost inevitable "
     "and frequently result in variation orders valued at 2–8% of contract value."),

    ("CHK-C-002", "chk_duplicate_tags_different_sizes", Trade.ARCHITECTURAL, Severity.HIGH,
     "The following door or window tags appear with conflicting size callouts at different locations "
     "in the drawing set: {conflicts}. "
     "Please reissue the affected drawings with a definitive correction and: "
     "(1) Confirm the correct dimension for each conflicting tag; "
     "(2) Update the Door / Window Schedule to reflect the governing dimension; "
     "(3) If the same mark is intentionally used for different sizes on different floors or zones, "
     "assign separate mark designations (e.g., D1A, D1B) and update both the schedule and all "
     "plan drawings accordingly; and "
     "(4) Issue a revision cloud on all affected sheets with revision note and date.",
     "A tag used at 10 locations — each potentially requiring a different-size door or frame — "
     "creates a pricing error of Rs 50,000–3,00,000 depending on the size and specification spread. "
     "Contractors who notice the conflict must either price the worst case (expensive and uncompetitive) "
     "or assume the cheaper option (risky if incorrect)."),

    # ---- Commercial / Contract ----
    ("CHK-COM-001", "chk_no_ld_clause", Trade.COMMERCIAL, Severity.HIGH,
     "No Liquidated Damages (LD) clause has been found in the conditions of contract. "
     "Please confirm the following in the tender conditions or by written clarification: "
     "(1) The applicable LD rate — as a percentage of contract value per week of delay "
     "(industry standard: 0.5%–1.0% per week); "
     "(2) The maximum cap on total LD deductions as a percentage of contract value "
     "(typically 10%); "
     "(3) The completion milestone(s) to which LD applies — sectional completion dates "
     "(if any) and the final completion date; "
     "(4) Whether LD is the client's sole and exclusive remedy for delay, or whether the "
     "client reserves the right to claim general or unliquidated damages in addition; and "
     "(5) The grace period (if any) between the contract completion date and the date "
     "from which LD starts to accrue.",
     "LD rate is a direct input to the contractor's risk pricing model. An uncapped LD clause, "
     "or a rate exceeding 1%/week, typically causes contractors to add 2–5% risk contingency to "
     "their bid price or to decline to submit — both detrimental to value for money for the client."),

    ("CHK-COM-002", "chk_no_retention", Trade.COMMERCIAL, Severity.HIGH,
     "No retention money clause has been found in the conditions of contract. "
     "Please confirm the following: "
     "(1) Retention deduction rate — percentage to be deducted from each running account (RA) bill "
     "(typical range: 5%–10% of gross bill value); "
     "(2) Retention cap — maximum total retention to be held as a percentage of contract value "
     "(typically 5%); "
     "(3) Conditions for release of the first moiety of retention (typically on issue of "
     "Completion Certificate or Taking-Over Certificate); "
     "(4) Conditions for release of the second moiety (typically on expiry of DLP and issue of "
     "Final Certificate); and "
     "(5) Whether the contractor may substitute a Retention Bond / bank guarantee in lieu of cash "
     "retention — and if so, the required bank guarantee format and validity.",
     "At a 10% retention on monthly RA bills, a contractor on a 24-month project may have "
     "Rs 50–80 lakhs locked in retention at peak construction. This working capital cost must be "
     "funded at 12–18% p.a. and is typically reflected as 0.5–1.5% of contract value in the bid price."),

    ("CHK-COM-003", "chk_no_warranty_dlp", Trade.COMMERCIAL, Severity.MEDIUM,
     "No warranty period or Defect Liability Period (DLP) clause has been found in the tender conditions. "
     "Please confirm: "
     "(1) DLP duration from the date of the Completion Certificate — the standard for building works "
     "is 12 months; extended periods (24 months) are common for waterproofing, specialty finishes, "
     "and MEP systems; "
     "(2) Contractor obligations during DLP — response time to attend defects (24 hrs / 48 hrs), "
     "who bears the cost of defect rectification (contractor-responsible vs. fair wear and tear); "
     "(3) Whether a DLP Performance Bond / bank guarantee is required — if so, the percentage of "
     "contract value and validity period (must cover full DLP plus 3–6 months claim period); and "
     "(4) The procedure for issuing the Final Certificate at DLP end, including the defect list "
     "close-out process.",
     "DLP duration and obligations directly affect post-project overhead costs. A 24-month DLP "
     "with a dedicated site representative costs 1.5–3% of contract value more than a 12-month "
     "DLP — this differential must be priced into the bid and cannot be estimated without clarity."),

    ("CHK-COM-004", "chk_no_bid_validity", Trade.COMMERCIAL, Severity.MEDIUM,
     "No Bid Validity Period has been stated in the tender documents. "
     "Please confirm: "
     "(1) The required validity of the bid from the date of submission — the standard range is "
     "90–180 days for building projects; "
     "(2) Whether the client reserves the right to request an extension of bid validity, "
     "and the procedure for doing so (contractor's right to decline or to revise the bid price "
     "if validity is extended beyond a stated trigger); "
     "(3) Whether any price escalation is permitted on material costs if contract award is "
     "delayed beyond a stated period within the validity window; and "
     "(4) The consequence if a contractor declines to extend their bid validity "
     "(e.g., forfeiture of EMD).",
     "Bid validity directly affects material price risk exposure. Steel, cement, and aluminium "
     "prices can move ±15–20% over a 6-month period. A 180-day validity typically adds 2–4% to "
     "bid prices relative to a 90-day validity — purely as a price-risk contingency."),

    ("CHK-COM-005", "chk_no_emd", Trade.COMMERCIAL, Severity.HIGH,
     "No Earnest Money Deposit (EMD) or Bid Security amount has been stated in the tender documents. "
     "Please confirm the following before bid submission: "
     "(1) The EMD amount — in absolute rupees or as a percentage of estimated contract value "
     "(typical range: 1%–3%); "
     "(2) Acceptable forms of EMD — demand draft (DD), pay order, bank guarantee (BG), "
     "or online NEFT / RTGS transfer; "
     "(3) EMD validity — must cover at least the bid validity period plus 30 days; "
     "(4) Conditions under which EMD will be forfeited (withdrawal after submission, "
     "failure to execute contract after award); "
     "(5) Timeline for refund to unsuccessful bidders after contract award; and "
     "(6) Whether the successful bidder's EMD is converted to part-security against the PBG "
     "or refunded in full on submission of the Performance Bank Guarantee.",
     "Arranging EMD — particularly BG amounts above Rs 25 lakhs — requires advance bank "
     "arrangements that can take 5–10 working days. Without knowing the EMD amount, bidders "
     "cannot complete bid preparation and may be forced to seek an extension of the bid deadline."),

    ("CHK-COM-006", "chk_no_performance_bond", Trade.COMMERCIAL, Severity.MEDIUM,
     "No Performance Bank Guarantee (PBG) or Performance Bond requirement has been stated "
     "in the tender conditions. Please confirm: "
     "(1) PBG percentage of the accepted contract value (typical range: 5%–10%); "
     "(2) Acceptable form — unconditional bank guarantee from a scheduled commercial bank, "
     "minimum bank rating, and whether a parent company guarantee is acceptable as an alternative; "
     "(3) PBG validity — must cover the full contract period plus the DLP period plus "
     "3–6 months for any claims; "
     "(4) Conditions under which the PBG may be invoked by the client; "
     "(5) Whether the PBG quantum may be reduced on satisfactory milestone achievement; and "
     "(6) Timeline for return of the PBG after issue of the Final Certificate.",
     "Bank guarantee charges are typically 1%–2% per annum of the guaranteed amount. "
     "A 10% PBG on a 24-month contract costs approximately 2%–4% of the PBG face value in "
     "bank charges — a direct bid cost that cannot be estimated without knowing the required percentage."),

    ("CHK-COM-007", "chk_no_mobilization_advance", Trade.COMMERCIAL, Severity.LOW,
     "No mobilisation advance clause has been found in the tender conditions. "
     "Please confirm whether a mobilisation advance is available and if so: "
     "(1) Advance amount — as a percentage of contract value (typical range: 5%–15%); "
     "(2) Conditions for advance release — bank guarantee requirement, mobilisation milestones "
     "(e.g., site establishment, equipment deployment, materials stacked); "
     "(3) Recovery schedule — from which RA bill number, at what percentage deduction per bill, "
     "and the target recovery date; "
     "(4) Whether the advance is interest-free or carries an interest charge "
     "(and the applicable rate if interest-bearing); and "
     "(5) Bank guarantee format and validity required to secure the advance.",
     "Availability of a mobilisation advance affects the contractor's initial financing cost. "
     "Without an advance on a project above Rs 5 crore, the contractor must arrange working capital "
     "at 12%–18% p.a. — typically priced into the bid as a hidden 1%–2% premium."),

    ("CHK-COM-008", "chk_no_insurance", Trade.COMMERCIAL, Severity.MEDIUM,
     "No insurance requirements have been specified in the tender conditions. "
     "Please confirm the following mandatory insurance policies to be arranged by the contractor: "
     "(1) Contractor's All Risk (CAR) Policy — sum insured basis (reinstatement value of works), "
     "third-party liability limit (minimum Rs 1 crore per occurrence), deductible amount, "
     "and policy period (from site handover to completion plus 30 days); "
     "(2) Workmen's Compensation (WC) / Employees' Compensation Policy — statutory minimum "
     "per the Employees' Compensation Act 1923, covering all workmen on site; "
     "(3) Whether the employer / client / owner is to be named as co-insured or additional insured "
     "on the CAR policy; "
     "(4) Any project-specific additional insurance — marine cargo for imported equipment, "
     "erection all-risk for plant and machinery, professional indemnity if design responsibility "
     "is included; and "
     "(5) Timeline for submitting evidence of insurance — before site mobilisation "
     "or before the first RA bill.",
     "Insurance premiums represent 0.3%–0.8% of contract value. Without knowing the coverage "
     "requirements (particularly the CAR sum insured basis and TPL limit), contractors either "
     "under-insure (creating risk exposure) or over-insure (unnecessary bid cost) — "
     "both leading to post-award adjustment demands."),

    ("CHK-COM-009", "chk_no_escalation", Trade.COMMERCIAL, Severity.MEDIUM,
     "No price escalation or price variation clause has been found in the conditions of contract. "
     "Please confirm one of the following: "
     "(1) Fixed-price / firm-price contract — confirm explicitly that the contract is lump-sum "
     "firm price with no price adjustment for material or labour cost changes during execution; "
     "(2) Variable-price contract with formula escalation — provide: the escalation formula and "
     "the specific indices to be used (RBI WPI Sub-indices, CPWD cost indices, or project-specific "
     "basket); the base date for recording the index (typically date of tender submission); "
     "the frequency of escalation calculations; and any ceiling on total escalation payable; or "
     "(3) Cost-plus arrangement — confirm the basis for establishing actual costs and the "
     "fee / overhead and profit percentage.",
     "On a 24-month project, steel, cement, and copper prices can collectively move 20%–40%. "
     "Without an escalation clause, contractors must include a risk contingency of 4%–8% of "
     "contract value — effectively a premium the client pays to transfer price risk to the contractor. "
     "An explicit firm-price confirmation is equally important, as its absence creates ambiguity."),

    ("CHK-COM-010", "chk_ld_rate_high", Trade.COMMERCIAL, Severity.HIGH,
     "The Liquidated Damages rate stated in the tender conditions is {ld_rate}% {cadence_str}"
     "which appears to exceed the typical industry threshold of 0.5%–1.0% per week "
     "(or equivalent on a per-day or per-month basis). "
     "Please confirm the following: "
     "(1) Whether the stated LD rate of {ld_rate}% is correct as drafted "
     "(confirm it is not a typographic error — e.g., 0.5% intended vs 5% stated); "
     "(2) The overall cap on total LD liability as a percentage of contract value; "
     "(3) Whether the LD rate applies to the full contract value or only to the value of the "
     "undelivered portion; and "
     "(4) Whether the client is willing to negotiate the LD rate to within industry norms, "
     "noting that an above-market LD rate will be reflected directly in bid prices or may "
     "cause qualified contractors to decline to bid.",
     "An LD rate above 1%/week (equivalent to >52% p.a. of contract value) significantly "
     "increases contractor risk exposure and is typically reflected as a 3%–7% premium in bid "
     "prices — or causes qualified contractors with alternative work to decline — resulting in "
     "either inflated costs or a narrower, less competitive bidder field for the client."),
]


# =============================================================================
# CHECK FUNCTIONS
# =============================================================================

def _make_evidence(
    pages: List[int] = None,
    snippets: List[str] = None,
    detected_entities: Dict[str, Any] = None,
    search_attempts: Dict[str, Any] = None,
    confidence: float = 0.6,
    confidence_reason: str = "",
    budget: int = 80,
) -> EvidenceRef:
    """Helper to construct EvidenceRef."""
    if not confidence_reason and confidence < 0.7:
        confidence_reason = (
            f"Not found within analyzed pages (OCR cap {budget}); "
            "may exist elsewhere in the set."
        )
    return EvidenceRef(
        pages=pages or [],
        snippets=snippets or [],
        detected_entities=detected_entities or {},
        search_attempts=search_attempts or {},
        confidence=confidence,
        confidence_reason=confidence_reason,
    )


class CheckContext:
    """All data needed by check functions."""
    def __init__(
        self,
        extracted: ExtractionResult,
        page_index: PageIndex,
        selected: SelectedPages,
        plan_graph: Optional[PlanSetGraph],
        run_coverage: Optional[RunCoverage] = None,
        package_classification: Optional[Any] = None,
    ):
        self.extracted = extracted
        self.page_index = page_index
        self.selected = selected
        self.plan_graph = plan_graph
        self.run_coverage = run_coverage
        self.budget = selected.budget_total if selected else 80

        # Package type from Stage 0 classifier (may be None for legacy callers)
        self.package_classification = package_classification
        # Store the PackageType value string for easy comparison
        self.package_type_value: str = (
            package_classification.package_type.value
            if package_classification is not None
            else ""
        )

        # Pre-compute useful aggregates
        self.type_counts = page_index.counts_by_type if page_index else {}
        self.disc_counts = page_index.counts_by_discipline if page_index else {}

        # Tags from plan_graph
        self.door_tags = set(plan_graph.all_door_tags) if plan_graph else set()
        self.window_tags = set(plan_graph.all_window_tags) if plan_graph else set()
        self.room_names = set(plan_graph.all_room_names) if plan_graph else set()

        # Tags from callouts
        for c in extracted.callouts:
            if c.get("callout_type") == "tag":
                tag = c["text"]
                if tag.startswith(("D", "DR")):
                    self.door_tags.add(tag)
                elif tag.startswith(("W", "WN")):
                    self.window_tags.add(tag)

        # Schedule info
        self.schedule_types_found: Set[str] = set()
        for s in extracted.schedules:
            self.schedule_types_found.add(s.get("schedule_type", "unknown"))

        # Drawing pages with dimensions
        self.dimension_callouts = [
            c for c in extracted.callouts if c.get("callout_type") == "dimension"
        ]

        # Material callouts with grades
        self.material_callouts = [
            c for c in extracted.callouts if c.get("callout_type") == "material"
        ]

        # Commercial terms (Sprint 19)
        self.commercial_terms = getattr(extracted, 'commercial_terms', [])
        self.commercial_term_types = {t.get("term_type") for t in self.commercial_terms}

    def is_covered(self, *doc_types: str) -> CoverageStatus:
        """Check if the given doc_types were fully covered during extraction."""
        if not self.run_coverage or self.run_coverage.selection_mode == SelectionMode.FULL_READ:
            return CoverageStatus.NOT_FOUND_AFTER_SEARCH
        for dt in doc_types:
            if self.run_coverage.is_doc_type_covered(dt) == CoverageStatus.UNKNOWN_NOT_PROCESSED:
                return CoverageStatus.UNKNOWN_NOT_PROCESSED
        return CoverageStatus.NOT_FOUND_AFTER_SEARCH


# =============================================================================
# CHECK-TO-DOC_TYPE MAPPING (for coverage gating)
# =============================================================================

_CHECK_DOC_TYPES: Dict[str, List[str]] = {
    "CHK-A-001": ["schedule"],      # door schedule
    "CHK-A-002": ["schedule"],      # window schedule
    "CHK-A-003": ["schedule"],      # finish schedule
    "CHK-A-004": ["notes"],         # general notes
    "CHK-A-005": ["legend"],        # legend
    "CHK-S-003": ["notes", "schedule"],  # BBS
    "CHK-X-005": ["boq"],           # BOQ
    "CHK-X-007": ["conditions"],    # tender conditions
    "CHK-X-008": ["addendum"],      # addendum
    "CHK-SCH-001": ["schedule"],    # schedule sizes
    "CHK-SCH-002": ["schedule"],    # schedule quantities
    # Commercial checks — need conditions/spec pages
    "CHK-COM-001": ["conditions", "spec"],
    "CHK-COM-002": ["conditions", "spec"],
    "CHK-COM-003": ["conditions", "spec"],
    "CHK-COM-004": ["conditions", "spec"],
    "CHK-COM-005": ["conditions", "spec"],
    "CHK-COM-006": ["conditions", "spec"],
    "CHK-COM-007": ["conditions", "spec"],
    "CHK-COM-008": ["conditions", "spec"],
    "CHK-COM-009": ["conditions", "spec"],
    "CHK-COM-010": ["conditions", "spec"],
}

# ---------------------------------------------------------------------------
# Drawing-only checks — should be suppressed for TENDER packages and downgraded
# to LOW severity for MIXED packages (Fix 5: RFI gating table).
# ---------------------------------------------------------------------------
_DRAWING_ONLY_CHECKS: frozenset = frozenset({
    "CHK-M-001",  # no MEP drawings overall
    "CHK-M-002",  # no electrical layout
    "CHK-M-003",  # no plumbing drawings
    "CHK-M-004",  # no HVAC drawings
    "CHK-M-005",  # no fire fighting drawings
    "CHK-S-001",  # no structural drawings
    "CHK-X-002",  # no site plan
    "CHK-X-003",  # no sections
    "CHK-X-004",  # no elevations
})

_MIXED_SUFFIX = (
    " [Drawings may be in a separate package — verify before issuing this RFI.]"
)

# Mapping from check_id → (dependency_name, trade) for dedup against blockers.
# When a Blocker already covers the same (dependency, trade) pair we drop the
# redundant standalone checklist RFI (Fix 6: dedup + linkage).
_ISSUE_DEP_MAP: Dict[str, Tuple[str, Trade]] = {
    "CHK-A-001": ("door_schedule",       Trade.ARCHITECTURAL),
    "CHK-A-002": ("window_schedule",     Trade.ARCHITECTURAL),
    "CHK-A-003": ("finish_schedule",     Trade.FINISHES),
    "CHK-S-001": ("structural_drawings", Trade.STRUCTURAL),
    "CHK-M-001": ("mep_drawings",        Trade.MEP),
    "CHK-X-002": ("site_plan",           Trade.CIVIL),
    "CHK-X-003": ("section_drawings",    Trade.ARCHITECTURAL),
    "CHK-X-004": ("elevation_drawings",  Trade.ARCHITECTURAL),
}


# --- Individual check functions ---
# Each returns (should_fire: bool, evidence: EvidenceRef, format_kwargs: dict)

def chk_door_schedule_missing(ctx: CheckContext):
    n_tags = len(ctx.door_tags)
    has_schedule = "door" in ctx.schedule_types_found
    has_graph_schedule = ctx.plan_graph.has_door_schedule if ctx.plan_graph else False

    if n_tags > 0 and not has_schedule and not has_graph_schedule:
        pages_with_tags = list({c["source_page"] for c in ctx.extracted.callouts
                                if c.get("callout_type") == "tag" and c["text"].startswith(("D", "DR"))})
        ev = _make_evidence(
            pages=pages_with_tags[:10],
            detected_entities={"door_tags": sorted(ctx.door_tags)[:20]},
            search_attempts={"searched_for": "door schedule page", "schedule_types_found": list(ctx.schedule_types_found)},
            confidence=0.85,
            confidence_reason=f"Found {n_tags} door tags but no door schedule page.",
            budget=ctx.budget,
        )
        return True, ev, {"n_tags": n_tags}
    return False, None, {}


def chk_window_schedule_missing(ctx: CheckContext):
    n_tags = len(ctx.window_tags)
    has_schedule = "window" in ctx.schedule_types_found
    has_graph_schedule = ctx.plan_graph.has_window_schedule if ctx.plan_graph else False

    if n_tags > 0 and not has_schedule and not has_graph_schedule:
        pages_with_tags = list({c["source_page"] for c in ctx.extracted.callouts
                                if c.get("callout_type") == "tag" and c["text"].startswith(("W", "WN"))})
        ev = _make_evidence(
            pages=pages_with_tags[:10],
            detected_entities={"window_tags": sorted(ctx.window_tags)[:20]},
            search_attempts={"searched_for": "window schedule page"},
            confidence=0.85,
            confidence_reason=f"Found {n_tags} window tags but no window schedule page.",
            budget=ctx.budget,
        )
        return True, ev, {"n_tags": n_tags}
    return False, None, {}


def chk_finish_schedule_missing(ctx: CheckContext):
    n_rooms = len(ctx.room_names)
    has_schedule = "finish" in ctx.schedule_types_found
    has_graph_schedule = ctx.plan_graph.has_finish_schedule if ctx.plan_graph else False

    if n_rooms > 0 and not has_schedule and not has_graph_schedule:
        ev = _make_evidence(
            detected_entities={"room_names": sorted(ctx.room_names)[:20]},
            search_attempts={"searched_for": "finish schedule page"},
            confidence=0.75,
            confidence_reason=f"Found {n_rooms} rooms but no finish schedule.",
            budget=ctx.budget,
        )
        return True, ev, {"n_rooms": n_rooms}
    return False, None, {}


def chk_no_general_notes(ctx: CheckContext):
    if ctx.type_counts.get("notes", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "general notes page", "pages_indexed": ctx.page_index.total_pages},
            confidence=0.7,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_legend(ctx: CheckContext):
    if ctx.type_counts.get("legend", 0) == 0:
        has_graph_legend = ctx.plan_graph.has_legend if ctx.plan_graph else False
        if not has_graph_legend:
            ev = _make_evidence(
                search_attempts={"searched_for": "legend/symbols page"},
                confidence=0.65,
                budget=ctx.budget,
            )
            return True, ev, {}
    return False, None, {}


def chk_room_dimensions_missing(ctx: CheckContext):
    n_rooms = len(ctx.room_names)
    n_dims = len(ctx.dimension_callouts)
    # Rough heuristic: if we have rooms but very few dimensions
    if n_rooms > 2 and n_dims < n_rooms:
        ev = _make_evidence(
            detected_entities={"rooms": n_rooms, "dimension_callouts": n_dims},
            search_attempts={"searched_for": "dimension callouts on plan pages"},
            confidence=0.5,
            budget=ctx.budget,
        )
        return True, ev, {"n_rooms": n_rooms, "n_dims": n_dims}
    return False, None, {}


def chk_no_structural(ctx: CheckContext):
    if ctx.disc_counts.get("structural", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "structural drawings", "disciplines_found": list(ctx.disc_counts.keys())},
            confidence=0.8,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_foundation(ctx: CheckContext):
    # Check if any page has foundation keywords
    has_foundation = any(
        p.doc_type == "plan" and "foundation" in (p.title or "").lower()
        for p in ctx.page_index.pages
    )
    if not has_foundation and ctx.disc_counts.get("structural", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "foundation details/layout"},
            confidence=0.6,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_bbs_missing(ctx: CheckContext):
    has_structural = ctx.disc_counts.get("structural", 0) > 0
    # Search for BBS in requirements
    has_bbs = any(
        "bar bending" in r.get("text", "").lower() or "bbs" in r.get("text", "").lower()
        for r in ctx.extracted.requirements
    )
    if has_structural and not has_bbs:
        ev = _make_evidence(
            search_attempts={"searched_for": "bar bending schedule (BBS)", "structural_pages": ctx.disc_counts.get("structural", 0)},
            confidence=0.55,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_mep(ctx: CheckContext):
    mep_count = (
        ctx.disc_counts.get("mechanical", 0) +
        ctx.disc_counts.get("electrical", 0) +
        ctx.disc_counts.get("plumbing", 0)
    )
    if mep_count == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "MEP drawings (M/E/P prefixes)", "disciplines_found": list(ctx.disc_counts.keys())},
            confidence=0.75,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_electrical(ctx: CheckContext):
    if ctx.disc_counts.get("electrical", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "electrical layout drawings"},
            confidence=0.7,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_plumbing(ctx: CheckContext):
    if ctx.disc_counts.get("plumbing", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "plumbing/drainage layout drawings"},
            confidence=0.7,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_hvac(ctx: CheckContext):
    if ctx.disc_counts.get("mechanical", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "HVAC/mechanical layout drawings"},
            confidence=0.5,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_fire(ctx: CheckContext):
    has_fire = ctx.disc_counts.get("fire", 0) > 0
    # Also check in requirements/callouts
    fire_in_req = any("fire" in r.get("text", "").lower() for r in ctx.extracted.requirements)
    if not has_fire and not fire_in_req:
        ev = _make_evidence(
            search_attempts={"searched_for": "fire fighting/protection layout"},
            confidence=0.55,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_scale_missing(ctx: CheckContext):
    if ctx.plan_graph:
        n_total = ctx.plan_graph.pages_with_scale + ctx.plan_graph.pages_without_scale
        n_missing = ctx.plan_graph.pages_without_scale
    else:
        n_total = ctx.page_index.total_pages
        scale_pages = sum(1 for c in ctx.extracted.callouts if c.get("callout_type") == "scale")
        drawing_pages = sum(ctx.type_counts.get(t, 0) for t in ["plan", "detail", "section", "elevation"])
        n_missing = max(drawing_pages - scale_pages, 0)
        n_total = drawing_pages

    if n_total > 0 and n_missing > n_total * 0.3:
        ev = _make_evidence(
            detected_entities={"pages_with_scale": n_total - n_missing, "pages_without_scale": n_missing},
            search_attempts={"searched_for": "scale notation (1:100, NTS, etc.)"},
            confidence=0.7,
            budget=ctx.budget,
        )
        return True, ev, {"n_missing": n_missing, "n_total": n_total}
    return False, None, {}


def chk_no_site_plan(ctx: CheckContext):
    has_site = any(
        "site" in (p.title or "").lower() or p.doc_type == "plan" and "site" in " ".join(p.keywords_hit).lower()
        for p in ctx.page_index.pages
    )
    if not has_site:
        ev = _make_evidence(
            search_attempts={"searched_for": "site plan / site layout"},
            confidence=0.65,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_sections(ctx: CheckContext):
    if ctx.type_counts.get("section", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "section drawings", "types_found": dict(ctx.type_counts)},
            confidence=0.75,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_elevations(ctx: CheckContext):
    if ctx.type_counts.get("elevation", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "elevation drawings"},
            confidence=0.6,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_boq(ctx: CheckContext):
    if ctx.type_counts.get("boq", 0) == 0 and len(ctx.extracted.boq_items) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "Bill of Quantities (BOQ) pages"},
            confidence=0.8,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_boq_missing_quantities(ctx: CheckContext):
    items = ctx.extracted.boq_items
    if len(items) >= 2:
        missing_qty = [item for item in items if item.get("qty") is None]
        n_missing = len(missing_qty)
        n_total = len(items)
        if n_missing > 0 and n_missing / n_total > 0.2:
            pages = list({item["source_page"] for item in missing_qty})
            snippets = [f"{item['item_no']}: {item['description'][:60]}" for item in missing_qty[:5]]
            ev = _make_evidence(
                pages=pages[:10],
                snippets=snippets,
                detected_entities={"items_missing_qty": n_missing, "total_items": n_total},
                confidence=0.8,
                confidence_reason=f"{n_missing}/{n_total} BOQ items have no quantity.",
                budget=ctx.budget,
            )
            return True, ev, {"n_missing": n_missing, "n_total": n_total}
    return False, None, {}


def chk_no_conditions(ctx: CheckContext):
    if ctx.type_counts.get("conditions", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "tender/contract conditions pages"},
            confidence=0.65,
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_addendum(ctx: CheckContext):
    if ctx.type_counts.get("addendum", 0) == 0:
        ev = _make_evidence(
            search_attempts={"searched_for": "addendum/corrigendum pages"},
            confidence=0.4,
            confidence_reason="No addendum found. This is informational — there may not be one.",
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_schedule_missing_sizes(ctx: CheckContext):
    missing = [s for s in ctx.extracted.schedules if not s.get("has_size")]
    if len(missing) > 0:
        pages = list({s["source_page"] for s in missing})
        snippets = [f"{s['mark']}: no size" for s in missing[:5]]
        ev = _make_evidence(
            pages=pages[:10],
            snippets=snippets,
            detected_entities={"items_missing_size": len(missing), "total_schedule_rows": len(ctx.extracted.schedules)},
            confidence=0.8,
            confidence_reason=f"{len(missing)} schedule items have marks but no sizes.",
            budget=ctx.budget,
        )
        return True, ev, {"n_missing": len(missing)}
    return False, None, {}


def chk_schedule_missing_qty(ctx: CheckContext):
    missing = [s for s in ctx.extracted.schedules if not s.get("has_qty")]
    if len(missing) > 0:
        pages = list({s["source_page"] for s in missing})
        snippets = [f"{s['mark']}: no qty" for s in missing[:5]]
        ev = _make_evidence(
            pages=pages[:10],
            snippets=snippets,
            detected_entities={"items_missing_qty": len(missing)},
            confidence=0.7,
            budget=ctx.budget,
        )
        return True, ev, {"n_missing": len(missing)}
    return False, None, {}


def chk_conflicting_material_specs(ctx: CheckContext):
    """Check for conflicting concrete grades across pages."""
    grade_pattern = re.compile(r'\bM-?(\d{2,3})\b', re.IGNORECASE)
    grade_locations: Dict[str, List[int]] = defaultdict(list)

    for c in ctx.material_callouts:
        match = grade_pattern.search(c["text"])
        if match:
            grade = f"M{match.group(1)}"
            grade_locations[grade].append(c["source_page"])

    # Conflict if same structural element has different grades
    if len(grade_locations) >= 2:
        conflicts_str = ", ".join(
            f"{grade} (p.{','.join(str(p+1) for p in pages[:3])})"
            for grade, pages in sorted(grade_locations.items())
        )
        all_pages = []
        for pages in grade_locations.values():
            all_pages.extend(pages)
        ev = _make_evidence(
            pages=sorted(set(all_pages))[:10],
            snippets=[f"{g}: pages {','.join(str(p+1) for p in ps[:3])}" for g, ps in grade_locations.items()],
            detected_entities={"grades_found": dict(grade_locations)},
            confidence=0.6,
            confidence_reason="Multiple concrete grades found — may be intentional for different elements.",
            budget=ctx.budget,
        )
        return True, ev, {"conflicts": conflicts_str}
    return False, None, {}


def chk_duplicate_tags_different_sizes(ctx: CheckContext):
    """Check if same tag has different sizes on different pages."""
    tag_sizes: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    for s in ctx.extracted.schedules:
        mark = s.get("mark")
        size = s.get("size")
        page = s.get("source_page", 0)
        if mark and size:
            tag_sizes[mark][size].append(page)

    conflicts = []
    for mark, sizes in tag_sizes.items():
        if len(sizes) > 1:
            conflict_str = f"{mark}: " + " vs ".join(
                f"{sz} (p.{','.join(str(p+1) for p in pgs[:2])})"
                for sz, pgs in sizes.items()
            )
            conflicts.append(conflict_str)

    if conflicts:
        all_pages = []
        for sizes in tag_sizes.values():
            for pgs in sizes.values():
                all_pages.extend(pgs)
        ev = _make_evidence(
            pages=sorted(set(all_pages))[:10],
            snippets=conflicts[:5],
            confidence=0.85,
            confidence_reason="Same tag has different sizes on different pages.",
            budget=ctx.budget,
        )
        return True, ev, {"conflicts": "; ".join(conflicts[:3])}
    return False, None, {}


# --- Commercial check functions (Sprint 19) ---

def _com_missing_term(ctx: CheckContext, term_type: str, term_label: str):
    """Helper: fire if a commercial term_type is not found."""
    if term_type not in ctx.commercial_term_types:
        conditions_pages = ctx.type_counts.get("conditions", 0) + ctx.type_counts.get("spec", 0)
        ev = _make_evidence(
            search_attempts={
                "searched_for": f"{term_label} in conditions/spec pages",
                "conditions_pages_indexed": conditions_pages,
                "commercial_terms_found": sorted(ctx.commercial_term_types),
            },
            confidence=0.7 if conditions_pages > 0 else 0.5,
            confidence_reason=(
                f"Searched {conditions_pages} conditions/spec pages; "
                f"'{term_label}' not found."
            ) if conditions_pages > 0 else f"No conditions/spec pages found to search for '{term_label}'.",
            budget=ctx.budget,
        )
        return True, ev, {}
    return False, None, {}


def chk_no_ld_clause(ctx: CheckContext):
    return _com_missing_term(ctx, "ld_clause", "liquidated damages / LD clause")


def chk_no_retention(ctx: CheckContext):
    return _com_missing_term(ctx, "retention", "retention clause")


def chk_no_warranty_dlp(ctx: CheckContext):
    return _com_missing_term(ctx, "warranty_dlp", "warranty / defect liability period")


def chk_no_bid_validity(ctx: CheckContext):
    return _com_missing_term(ctx, "bid_validity", "bid validity period")


def chk_no_emd(ctx: CheckContext):
    return _com_missing_term(ctx, "emd_bid_security", "EMD / bid security amount")


def chk_no_performance_bond(ctx: CheckContext):
    return _com_missing_term(ctx, "performance_bond", "performance bank guarantee")


def chk_no_mobilization_advance(ctx: CheckContext):
    return _com_missing_term(ctx, "mobilization_advance", "mobilization advance")


def chk_no_insurance(ctx: CheckContext):
    return _com_missing_term(ctx, "insurance", "insurance / CAR policy")


def chk_no_escalation(ctx: CheckContext):
    return _com_missing_term(ctx, "escalation", "price escalation clause")


def chk_ld_rate_high(ctx: CheckContext):
    """Fire if LD rate exceeds 1% — unusually high."""
    if "ld_clause" not in ctx.commercial_term_types:
        return False, None, {}
    for term in ctx.commercial_terms:
        if term.get("term_type") == "ld_clause":
            value = term.get("value")
            if isinstance(value, (int, float)) and value > 1.0:
                cadence = term.get("cadence") or ""
                cadence_str = f"per {cadence} " if cadence else ""
                snippet = term.get("snippet", "")
                ev = _make_evidence(
                    pages=[term.get("source_page", 0)],
                    snippets=[snippet] if snippet else [],
                    detected_entities={"ld_rate": value, "cadence": cadence},
                    confidence=0.85,
                    confidence_reason=f"LD rate of {value}% {cadence_str}found in conditions.",
                    budget=ctx.budget,
                )
                return True, ev, {"ld_rate": value, "cadence_str": cadence_str}
    return False, None, {}


# Map check IDs to functions
CHECK_FN_MAP = {
    "chk_door_schedule_missing": chk_door_schedule_missing,
    "chk_window_schedule_missing": chk_window_schedule_missing,
    "chk_finish_schedule_missing": chk_finish_schedule_missing,
    "chk_no_general_notes": chk_no_general_notes,
    "chk_no_legend": chk_no_legend,
    "chk_room_dimensions_missing": chk_room_dimensions_missing,
    "chk_no_structural": chk_no_structural,
    "chk_no_foundation": chk_no_foundation,
    "chk_bbs_missing": chk_bbs_missing,
    "chk_no_mep": chk_no_mep,
    "chk_no_electrical": chk_no_electrical,
    "chk_no_plumbing": chk_no_plumbing,
    "chk_no_hvac": chk_no_hvac,
    "chk_no_fire": chk_no_fire,
    "chk_scale_missing": chk_scale_missing,
    "chk_no_site_plan": chk_no_site_plan,
    "chk_no_sections": chk_no_sections,
    "chk_no_elevations": chk_no_elevations,
    "chk_no_boq": chk_no_boq,
    "chk_boq_missing_quantities": chk_boq_missing_quantities,
    "chk_no_conditions": chk_no_conditions,
    "chk_no_addendum": chk_no_addendum,
    "chk_schedule_missing_sizes": chk_schedule_missing_sizes,
    "chk_schedule_missing_qty": chk_schedule_missing_qty,
    "chk_conflicting_material_specs": chk_conflicting_material_specs,
    "chk_duplicate_tags_different_sizes": chk_duplicate_tags_different_sizes,
    # Commercial (Sprint 19)
    "chk_no_ld_clause": chk_no_ld_clause,
    "chk_no_retention": chk_no_retention,
    "chk_no_warranty_dlp": chk_no_warranty_dlp,
    "chk_no_bid_validity": chk_no_bid_validity,
    "chk_no_emd": chk_no_emd,
    "chk_no_performance_bond": chk_no_performance_bond,
    "chk_no_mobilization_advance": chk_no_mobilization_advance,
    "chk_no_insurance": chk_no_insurance,
    "chk_no_escalation": chk_no_escalation,
    "chk_ld_rate_high": chk_ld_rate_high,
}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def generate_rfis(
    extracted: ExtractionResult,
    page_index: PageIndex,
    selected: SelectedPages,
    plan_graph: Optional[PlanSetGraph] = None,
    run_coverage: Optional[RunCoverage] = None,
    package_classification: Optional[Any] = None,
    blockers: Optional[List[Any]] = None,
) -> List[RFIItem]:
    """
    Run the discipline checklist and generate evidence-backed RFIs.

    Args:
        extracted: ExtractionResult from run_extractors.
        page_index: PageIndex from build_page_index.
        selected: SelectedPages from select_pages.
        plan_graph: Optional PlanSetGraph (may be None for small sets).
        run_coverage: Optional RunCoverage for coverage-gated assertions.
        package_classification: Optional PackageClassification (Stage 0).
            When provided, drawing-only checks are suppressed for TENDER
            packages and downgraded to LOW for MIXED packages.
        blockers: Optional list of Blocker objects from dependency_reasoner.
            When provided, checklist RFIs that duplicate a blocker's
            missing_dependency are dropped and linked to the blocker instead.

    Returns:
        List of RFIItem with evidence.
    """
    ctx = CheckContext(
        extracted, page_index, selected, plan_graph,
        run_coverage=run_coverage,
        package_classification=package_classification,
    )
    rfis: List[RFIItem] = []
    rfi_counter = 1

    for check_id, fn_name, trade, priority, question_tpl, why_tpl in CHECKLIST:
        fn = CHECK_FN_MAP.get(fn_name)
        if not fn:
            continue

        try:
            should_fire, evidence, fmt_kwargs = fn(ctx)
        except Exception as e:
            if DEBUG:
                import logging
                logging.getLogger(__name__).warning(f"Check {check_id} failed: {e}")
            continue

        if should_fire and evidence:
            question = question_tpl.format(**fmt_kwargs) if fmt_kwargs else question_tpl
            why = why_tpl.format(**fmt_kwargs) if fmt_kwargs else why_tpl

            # --- Package-type gating (Fix 5: drawing-only checks) ---
            actual_priority = priority
            if check_id in _DRAWING_ONLY_CHECKS and package_classification is not None:
                pkg_val = ctx.package_type_value
                if pkg_val == "tender":
                    continue   # suppress entirely — drawings are not expected
                elif pkg_val == "mixed":
                    actual_priority = Severity.LOW
                    question = question + _MIXED_SUFFIX

            # --- Coverage gating ---
            relevant_types = _CHECK_DOC_TYPES.get(check_id, [])
            cov_status = ctx.is_covered(*relevant_types) if relevant_types else CoverageStatus.NOT_FOUND_AFTER_SEARCH

            if cov_status == CoverageStatus.UNKNOWN_NOT_PROCESSED:
                # Downgrade priority — can't confirm absence
                actual_priority = Severity.LOW
                question = question + " [Coverage gap \u2014 may exist on unprocessed pages]"

            rfi = RFIItem(
                id=create_rfi_id(rfi_counter),
                trade=trade,
                priority=actual_priority,
                question=question,
                why_it_matters=why,
                evidence=evidence,
                suggested_resolution=_suggest_resolution(check_id),
                issue_type=check_id,
                package=_package_for_trade(trade),
                coverage_status=cov_status.value,
            )
            rfis.append(rfi)
            rfi_counter += 1

    # --- Blocker dedup + linkage (Fix 6) ---
    # Drop checklist RFIs that are already covered by a Blocker (same
    # missing_dependency + trade pair).  For the ones not dropped, attach
    # the related_blocker_id so the UI can cross-link them.
    if blockers:
        dep_to_blk_id: Dict[str, str] = {}
        blocker_dep_trade_pairs: Set[Tuple[str, str]] = set()

        for blk in blockers:
            missing_deps = getattr(blk, "missing_dependency", []) or []
            for dep in missing_deps:
                dep_to_blk_id[dep] = blk.id
                trade_val = blk.trade.value if hasattr(blk.trade, "value") else str(blk.trade)
                blocker_dep_trade_pairs.add((dep, trade_val))

        kept: List[RFIItem] = []
        for rfi in rfis:
            dep_trade = _ISSUE_DEP_MAP.get(rfi.issue_type)
            if dep_trade is not None:
                dep_name, dep_trade_enum = dep_trade
                pair = (dep_name, dep_trade_enum.value)
                if pair in blocker_dep_trade_pairs:
                    continue   # blocker already covers this — skip
                # Not a duplicate but share a dep → link to blocker
                if dep_name in dep_to_blk_id:
                    rfi.related_blocker_id = dep_to_blk_id[dep_name]
            kept.append(rfi)
        rfis = kept

    # Re-number IDs sequentially after dedup
    for i, rfi in enumerate(rfis, start=1):
        rfi.id = create_rfi_id(i)

    return rfis


def _suggest_resolution(check_id: str) -> str:
    """Provide a default suggested resolution based on check ID."""
    resolutions = {
        "CHK-A-001": "Request door schedule from architect/consultant.",
        "CHK-A-002": "Request window schedule from architect/consultant.",
        "CHK-A-003": "Request finish schedule from architect/consultant.",
        "CHK-A-004": "Request general notes sheet.",
        "CHK-A-005": "Request legend/symbols sheet.",
        "CHK-A-006": "Request dimensioned floor plans or area statement.",
        "CHK-S-001": "Request structural drawings from structural engineer.",
        "CHK-S-002": "Request foundation details including type, depth, and sizes.",
        "CHK-S-003": "Request bar bending schedule from structural engineer.",
        "CHK-M-001": "Request MEP drawings from MEP consultant.",
        "CHK-M-002": "Request electrical layout drawings.",
        "CHK-M-003": "Request plumbing layout drawings.",
        "CHK-M-004": "Confirm if HVAC is in scope; request drawings if applicable.",
        "CHK-M-005": "Request fire fighting/protection layout from fire consultant.",
        "CHK-X-001": "Request drawings with scale notation or confirm NTS policy.",
        "CHK-X-002": "Request site plan with boundaries, setbacks, and external works.",
        "CHK-X-003": "Request section drawings for floor-to-floor heights and beam depths.",
        "CHK-X-004": "Request elevation drawings for facade details.",
        "CHK-X-005": "Request BOQ in the tender format for pricing.",
        "CHK-X-006": "Clarify missing quantities in BOQ items.",
        "CHK-X-007": "Request tender conditions / general conditions of contract.",
        "CHK-X-008": "Confirm if any addenda/corrigenda were issued.",
        "CHK-SCH-001": "Request complete schedule with sizes for all items.",
        "CHK-SCH-002": "Request quantities in the schedule.",
        "CHK-C-001": "Clarify which material specification governs for each element.",
        "CHK-C-002": "Clarify correct size for each conflicting door/window tag.",
        # Commercial (Sprint 19)
        "CHK-COM-001": "Request liquidated damages clause / rate from the client.",
        "CHK-COM-002": "Request retention percentage and release terms.",
        "CHK-COM-003": "Request warranty / defect liability period details.",
        "CHK-COM-004": "Confirm bid validity period before submission.",
        "CHK-COM-005": "Request EMD / bid security amount and instrument type.",
        "CHK-COM-006": "Request performance bank guarantee percentage and validity.",
        "CHK-COM-007": "Confirm if mobilization advance is available and terms.",
        "CHK-COM-008": "Request insurance requirements (CAR policy, third-party, workmen).",
        "CHK-COM-009": "Confirm if price escalation clause applies to this contract.",
        "CHK-COM-010": "Verify the LD rate — it appears unusually high.",
    }
    return resolutions.get(check_id, "Raise RFI with the architect/consultant for clarification.")


def _package_for_trade(trade: Trade) -> str:
    """Map trade to a bid package name."""
    mapping = {
        Trade.ARCHITECTURAL: "Architectural",
        Trade.STRUCTURAL: "Structural",
        Trade.ELECTRICAL: "Electrical",
        Trade.PLUMBING: "Plumbing",
        Trade.MEP: "MEP",
        Trade.CIVIL: "Civil",
        Trade.FINISHES: "Finishes",
        Trade.GENERAL: "General",
        Trade.COMMERCIAL: "Commercial",
    }
    return mapping.get(trade, "General")


# ---------------------------------------------------------------------------
# Knowledge Base RFI Extension
# ---------------------------------------------------------------------------

# Unit normalisation map — maps aliases to canonical unit names
_UNIT_NORM: Dict[str, str] = {
    # Square metre
    "sqm": "sqm", "m2": "sqm", "smt": "sqm", "sq.m": "sqm", "sq m": "sqm",
    "sqmtr": "sqm", "sq mtr": "sqm",
    # Square foot (non-metric)
    "sqft": "sqft", "sq.ft": "sqft", "sq ft": "sqft", "sft": "sqft",
    # Cubic metre
    "cum": "cum", "m3": "cum", "cu.m": "cum", "cu m": "cum", "cumtr": "cum",
    # Linear metre
    "rmt": "rmt", "rm": "rmt", "lm": "rmt", "lrm": "rmt", "m": "rmt",
    "rmt.": "rmt", "mtr": "rmt",
    # Each / numbers
    "nos": "nos", "no": "nos", "nr": "nos", "each": "nos", "no.": "nos",
    "ea": "nos", "item": "nos",
    # Weight
    "kg": "kg", "kgs": "kg",
    "mt": "mt", "ton": "mt", "tonne": "mt", "tonnes": "mt",
    "mt.": "mt", "mtons": "mt",
    # Liquid
    "ltr": "ltr", "litre": "ltr", "liter": "ltr", "litres": "ltr", "lit": "ltr",
}

# Pairs where BOQ unit vs taxonomy unit constitutes a meaningful error
_UNIT_MISMATCH_PAIRS = frozenset({
    ("sqm", "cum"), ("cum", "sqm"),    # area vs volume
    ("sqft", "sqm"), ("sqm", "sqft"),  # imperial vs metric
    ("sqft", "cum"), ("cum", "sqft"),  # imperial area vs metric volume
    ("rmt", "sqm"), ("sqm", "rmt"),    # linear vs area
    ("rmt", "cum"), ("cum", "rmt"),    # linear vs volume
    ("nos", "sqm"), ("sqm", "nos"),    # count vs area
    ("nos", "cum"), ("cum", "nos"),    # count vs volume
    ("nos", "rmt"), ("rmt", "nos"),    # count vs linear
    ("kg",  "nos"), ("nos",  "kg"),    # weight vs count (rare but real)
})

# Pairs that are HIGH priority (order-of-magnitude billing impact)
_HIGH_PRIO_PAIRS = frozenset({
    ("sqft", "sqm"), ("sqm", "sqft"),
    ("sqm",  "cum"), ("cum",  "sqm"),
    ("sqft", "cum"), ("cum",  "sqft"),
})

_SEVERITY_PRIORITY = {"critical": "CRITICAL", "high": "HIGH",
                      "medium": "MEDIUM",     "low": "LOW"}


def _generate_boq_validation_rfis(
    boq_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Generate RFIs by validating BOQ items against the taxonomy:
      1.  Rate anomalies  (critical / high)  → RFI-KB-8xxx
      2.  Unit mismatches (meaningful errors) → RFI-KB-85xx

    Only surfaces issues with enough confidence to be actionable.
    Wrapped in broad try/except so any taxonomy-load failure is silent.
    """
    if not boq_items:
        return []

    rfis: List[Dict[str, Any]] = []
    import logging as _log
    _logger = _log.getLogger(__name__)

    # ── 1. Rate anomaly RFIs ──────────────────────────────────────────────
    try:
        from src.knowledge_base.rate_validator import RateValidator
        _rv = RateValidator()
        _vr = _rv.validate_items(boq_items, min_match_confidence=0.45)
        _counter = 8000
        for anomaly in _vr.anomalies:
            if anomaly.severity not in ("critical", "high"):
                continue  # medium/low would be too noisy as RFIs
            _counter += 1
            desc_short = anomaly.item_description[:80]
            trade_code = (anomaly.taxonomy_id.split(".")[0].lower()
                          if anomaly.taxonomy_id else "general")
            below = anomaly.deviation_pct < 0
            rfis.append({
                "id": f"RFI-KB-{_counter:04d}",
                "trade": trade_code,
                "priority": _SEVERITY_PRIORITY.get(anomaly.severity, "MEDIUM"),
                "question": (
                    f"Rate anomaly for '{desc_short}': Rs {anomaly.item_rate:,.0f}/"
                    f"{anomaly.item_unit} is {abs(anomaly.deviation_pct):.0f}% "
                    f"{'below' if below else 'above'} expected "
                    f"(Rs {anomaly.expected_min:,.0f}–{anomaly.expected_max:,.0f}/"
                    f"{anomaly.item_unit}). Please confirm rate and scope inclusions."
                ),
                "why_it_matters": (
                    f"{'Under-quoting' if below else 'Over-pricing'} by "
                    f"{abs(anomaly.deviation_pct):.0f}% on '{anomaly.taxonomy_name}'. "
                    "Incorrect rates cause budget overruns or acceptance rejection."
                ),
                "suggested_resolution": (
                    f"Verify scope inclusions for {anomaly.taxonomy_name} and "
                    f"confirm rate conforms to market range "
                    f"Rs {anomaly.expected_min:,.0f}–{anomaly.expected_max:,.0f}/"
                    f"{anomaly.item_unit}."
                ),
                "issue_type": f"rate_anomaly_{anomaly.severity}_{anomaly.taxonomy_id}",
                "package": trade_code,
                "coverage_status": "rate_anomaly",
                "evidence": {
                    "pages": [], "sheets": [],
                    "snippets": [
                        f"'{desc_short}' @ Rs {anomaly.item_rate:,.0f}/{anomaly.item_unit}"
                    ],
                    "detected_entities": {
                        "taxonomy_id": anomaly.taxonomy_id,
                        "taxonomy_name": anomaly.taxonomy_name,
                        "match_confidence": anomaly.match_confidence,
                        "expected_min": anomaly.expected_min,
                        "expected_max": anomaly.expected_max,
                    },
                    "search_attempts": {},
                    "confidence": anomaly.match_confidence,
                    "confidence_reason": (
                        f"Taxonomy match {anomaly.match_confidence:.0%}; "
                        f"rate {abs(anomaly.deviation_pct):.0f}% off range"
                    ),
                },
            })
    except Exception as _e:
        _logger.debug("Rate validation RFIs skipped: %s", _e)

    # ── 2. Unit mismatch RFIs ─────────────────────────────────────────────
    try:
        from src.knowledge_base.matcher import match_boq_batch
        _descs    = [
            (item.get("description") or item.get("item_name", ""))
            for item in boq_items
        ]
        _units    = [item.get("unit", "") for item in boq_items]
        _sections = [item.get("section", "") for item in boq_items]
        _matches  = match_boq_batch(_descs, min_confidence=0.45,
                                    units=_units, sections=_sections)
        _ucounter = 8500
        _seen = set()

        for item, match in zip(boq_items, _matches):
            if not match.matched or not match.unit:
                continue
            boq_unit_raw = (item.get("unit") or "").lower().strip()
            tax_unit_raw = match.unit.lower().strip()
            if not boq_unit_raw:
                continue
            boq_n = _UNIT_NORM.get(boq_unit_raw, boq_unit_raw)
            tax_n = _UNIT_NORM.get(tax_unit_raw, tax_unit_raw)
            if boq_n == tax_n:
                continue
            pair = (boq_n, tax_n)
            if pair not in _UNIT_MISMATCH_PAIRS:
                continue
            issue_key = f"{match.taxonomy_id}:{boq_n}→{tax_n}"
            if issue_key in _seen:
                continue
            _seen.add(issue_key)
            _ucounter += 1
            desc_short = (item.get("description") or item.get("item_name", ""))[:80]
            priority = "HIGH" if pair in _HIGH_PRIO_PAIRS else "MEDIUM"
            rfis.append({
                "id": f"RFI-KB-{_ucounter:04d}",
                "trade": match.trade or "general",
                "priority": priority,
                "question": (
                    f"Unit of measurement appears incorrect for '{desc_short}': "
                    f"BOQ uses '{boq_unit_raw}' but IS 1200 standard unit for "
                    f"{match.canonical_name} is '{tax_unit_raw}'. "
                    "Please confirm correct unit and recalculate quantities."
                ),
                "why_it_matters": (
                    f"Using '{boq_unit_raw}' instead of '{tax_unit_raw}' produces "
                    "incorrect quantity calculations, potentially changing the bill "
                    "value by 10× or more. IS 1200 mandates the standard unit."
                ),
                "suggested_resolution": (
                    f"Revise BOQ unit from '{boq_unit_raw}' to '{tax_unit_raw}' for "
                    f"'{match.canonical_name}' and recalculate all quantities "
                    "per IS 1200."
                ),
                "issue_type": f"unit_mismatch_{match.taxonomy_id}",
                "package": match.trade or "general",
                "coverage_status": "unit_mismatch",
                "evidence": {
                    "pages": [], "sheets": [],
                    "snippets": [
                        f"'{desc_short}' unit='{boq_unit_raw}', expected='{tax_unit_raw}'"
                    ],
                    "detected_entities": {
                        "taxonomy_id": match.taxonomy_id,
                        "taxonomy_name": match.canonical_name,
                        "match_confidence": match.confidence,
                        "boq_unit": boq_unit_raw,
                        "expected_unit": tax_unit_raw,
                    },
                    "search_attempts": {},
                    "confidence": match.confidence,
                    "confidence_reason": (
                        f"Taxonomy match {match.confidence:.0%}; "
                        f"unit mismatch {boq_unit_raw} vs {tax_unit_raw}"
                    ),
                },
            })
    except Exception as _e:
        _logger.debug("Unit mismatch RFIs skipped: %s", _e)

    return rfis


def _generate_room_checklist_rfis(
    boq_items: List[Dict[str, Any]],
    building_type: str = "all",
) -> List[Dict[str, Any]]:
    """
    Detect which room types are present in the BOQ and flag items that are
    missing from each room's required checklist.

    Uses RoomChecklistLoader.match_room_from_keywords() to find rooms, then
    RoomChecklistLoader.check_boq_for_room() to find gaps.

    Generates one RFI per room type (listing ALL missing items) rather than
    one per missing item, to keep the RFI list manageable.

    Returns RFI-KB-9xxx formatted dicts.
    """
    if not boq_items:
        return []

    import logging as _log
    _logger = _log.getLogger(__name__)
    rfis: List[Dict[str, Any]] = []

    try:
        from src.knowledge_base.room_checklists.loader import RoomChecklistLoader
        _loader = RoomChecklistLoader()
        _loader.load_all()
        if _loader.checklist_count == 0:
            return []

        # Build combined BOQ text and description list for searching
        boq_descs = [
            (item.get("description") or item.get("item_name", "")).strip()
            for item in boq_items
        ]
        boq_text = " ".join(boq_descs)

        # Find which room types appear in the BOQ
        matched_rooms = _loader.match_room_from_keywords(boq_text)

        _counter = 9000
        for room in matched_rooms:
            # Skip if this room type doesn't apply to this building type
            if "all" not in room.building_types and building_type not in room.building_types:
                continue

            gap_result = _loader.check_boq_for_room(room.room_type, boq_descs)
            missing = gap_result.get("missing", [])
            if not missing:
                continue

            # Only surface critical and high priority room types as RFIs
            if room.priority not in ("critical", "high"):
                continue

            _counter += 1
            # Format missing items as a readable list
            missing_display = ", ".join(
                m.replace("_", " ").title() for m in missing[:10]
            )
            if len(missing) > 10:
                missing_display += f" (+{len(missing) - 10} more)"

            # Use the rfi_template if defined, otherwise build a generic question
            raw_template = room.rfi_template or (
                f"{room.display_name} detected in BOQ but {len(missing)} "
                "required item(s) appear to be missing: {missing_items}. "
                "Please confirm full scope for this room type."
            )
            try:
                question = raw_template.format(missing_items=missing_display)
            except (KeyError, IndexError):
                question = raw_template  # template has unexpected placeholders

            priority_map = {"critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM"}
            rfis.append({
                "id": f"RFI-KB-{_counter:04d}",
                "trade": "general",
                "priority": priority_map.get(room.priority, "MEDIUM"),
                "question": question,
                "why_it_matters": (
                    f"{room.display_name} requires specific items that are commonly "
                    "omitted from BOQs. Missing items lead to cost overruns during "
                    "execution when subcontractors claim extras."
                ),
                "suggested_resolution": (
                    f"Confirm that the following items are scoped for "
                    f"{room.display_name}: {missing_display}."
                ),
                "issue_type": f"room_gap_{room.room_type}",
                "package": room.room_type,
                "coverage_status": "room_checklist_gap",
                "evidence": {
                    "pages": [], "sheets": [],
                    "snippets": [
                        f"{room.display_name} detected; missing: {missing_display}"
                    ],
                    "detected_entities": {
                        "room_type": room.room_type,
                        "missing_count": len(missing),
                        "missing_items": missing[:15],
                        "common_omissions": room.common_omissions,
                    },
                    "search_attempts": {},
                    "confidence": 0.75,
                    "confidence_reason": "Room type detected via keyword match; required items not found in BOQ",
                },
            })

    except Exception as _e:
        _logger.debug("Room checklist RFIs skipped: %s", _e)

    return rfis


def generate_knowledge_base_rfis(
    scope_gaps: List[Dict[str, Any]],
    boq_items: List[Dict[str, Any]],
    project_params: Dict[str, Any],
    actual_boq_items: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate RFIs from scope gaps using knowledge base rules, and optionally
    validate the full BOQ item list for rate anomalies and unit mismatches.

    This is ADDITIVE to existing checklist-driven RFIs — it picks up:
    • Scope gaps that the 36-check checklist doesn't cover
    • Rate anomalies (critical/high) detected against taxonomy rate ranges
    • Unit-of-measurement errors (sqft vs sqm, sqm vs cum, etc.)

    Args:
        scope_gaps:       List of {"missing_item": ..., "trade": ...} dicts from
                          plan_graph.scope_analysis (already filtered by caller).
        boq_items:        Passed-through to match_gaps_to_rfis (currently used for
                          context; may also be the existing rfi_list from pipeline).
        project_params:   Dict with at least {"building_type": "..."}.
        actual_boq_items: Full list of extracted BOQ item dicts from
                          extraction_result.boq_items. Used for rate + unit
                          validation. Optional — skipped if None or empty.

    Returns List[dict] in RFIItem-compatible format for pipeline merge.
    """
    result: List[Dict[str, Any]] = []

    # ── Scope-gap → RFI rule matching (existing KB behaviour) ──
    try:
        from src.knowledge_base.rfi_rules.loader import match_gaps_to_rfis
        result.extend(match_gaps_to_rfis(scope_gaps, boq_items, project_params))
    except ImportError:
        pass
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "Knowledge base RFI generation failed: %s", e
        )

    # ── BOQ validation: rate anomalies + unit mismatches ──
    if actual_boq_items:
        try:
            validation_rfis = _generate_boq_validation_rfis(actual_boq_items)
            result.extend(validation_rfis)
            if validation_rfis:
                import logging
                logging.getLogger(__name__).info(
                    "BOQ validation: %d additional RFIs "
                    "(%d rate anomaly, %d unit mismatch)",
                    len(validation_rfis),
                    sum(1 for r in validation_rfis if "rate_anomaly" in r.get("coverage_status", "")),
                    sum(1 for r in validation_rfis if "unit_mismatch" in r.get("coverage_status", "")),
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("BOQ validation RFIs skipped: %s", e)

    # ── Room-type completeness: detect rooms in BOQ, flag missing required items ──
    if actual_boq_items:
        try:
            _btype = project_params.get("building_type", "all")
            room_rfis = _generate_room_checklist_rfis(actual_boq_items, _btype)
            result.extend(room_rfis)
            if room_rfis:
                import logging
                logging.getLogger(__name__).info(
                    "Room checklist: %d additional RFIs", len(room_rfis)
                )
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("Room checklist RFIs skipped: %s", e)

    # Sort final list: critical → high → medium → low → unknown
    # Within each severity, preserve discovery order (stable sort).
    _SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    result.sort(key=lambda r: _SEV_ORDER.get(
        (r.get("severity") or r.get("priority") or "low").lower(), 4
    ))

    return result
