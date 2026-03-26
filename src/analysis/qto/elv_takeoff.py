"""
ELV Takeoff — norm-based quantity take-off for Extra Low Voltage systems.

ELV systems are present in 90% of Indian institutional campus tenders
(WAPCOS, EPI, CPWD, IIT, AIIMS) but are seldom extracted from BOQ tables.
This module generates ELV BOQ items entirely from floor-area norms so that
every pipeline run captures these systems automatically.

Systems covered
---------------
1. Fire Alarm System    (smoke detectors, MCPs, panel, cable, hooters)
2. CCTV Surveillance    (cameras, DVR/NVR, cable, monitors)
3. PA System            (speakers, amplifiers, cable)
4. Data Networking      (data points, switches, CAT6, OFC backbone)
5. Structured Wiring    (telephone points + cable)
6. Access Control       (academic / office / hospital / research only)
7. Nurse Call           (hospital / aiims / dispensary only)

All quantities are rounded up (math.ceil) to match industry practice of
ordering in whole-number lots.

Norms reference
---------------
- Fire alarm:  NFPA 72 / NBC 2016 Part 4 (1 detector per 25 sqm)
- CCTV:        IS 16901 / institutional practice (1 camera per 50 sqm)
- PA:          NBC 2016 (1 speaker per 30 sqm)
- Data:        TIA-568 / CPWD IT infrastructure (1 port per 8 sqm for offices)
- Access ctrl: IS 16239 / institutional standard (2 readers per floor)
- Nurse call:  NABH / hospital design (1 station per patient room)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


# =============================================================================
# NORMS
# =============================================================================

_FA_SQM_PER_DETECTOR   = 25.0   # NFPA 72 — 1 smoke detector per 25 sqm
_FA_SQM_PER_MCP        = 300.0  # manual call point coverage area
_FA_MIN_MCP_PER_FLOOR  = 2      # minimum MCPs regardless of area
_FA_CABLE_PER_DETECTOR = 12.0   # metres of FA cable per detector (avg run)
_FA_SQM_PER_HOOTER     = 150.0  # 1 sounder per 150 sqm

_CCTV_SQM_PER_CAMERA   = 50.0   # institutional norm (1 per 50 sqm)
_CCTV_CH_PER_RECORDER  = 16     # 16-channel DVR/NVR
_CCTV_CABLE_PER_CAM    = 20.0   # metres of coaxial/CAT6 per camera
_CCTV_CAM_PER_MONITOR  = 8      # monitoring station covers 8 cameras

_PA_SQM_PER_SPEAKER    = 30.0   # NBC 2016
_PA_SPK_PER_AMP        = 20     # 20 speakers per amplifier zone
_PA_CABLE_PER_SPEAKER  = 15.0   # metres of 2-core PA cable per speaker

_DATA_SQM_PER_POINT_FULL   = 8.0    # TIA-568 office norm
_DATA_SQM_PER_POINT_HOSTEL = 20.0   # 1 port per 20 sqm for hostel
_DATA_SQM_PER_POINT_HOSP   = 13.3   # 0.6× of full (1 per ~13.3 sqm)
_DATA_PORTS_PER_SWITCH      = 24
_DATA_CABLE_PER_POINT       = 20.0  # metres of CAT6 per data point
_DATA_OFC_RISERS            = 2     # OFC runs per floor (up + spare)
_DATA_STOREY_HEIGHT_M       = 3.2   # default storey height for OFC

_TEL_SQM_PER_POINT     = 20.0   # 1 telephone point per 20 sqm
_TEL_CABLE_PER_POINT   = 15.0   # metres of 2-pair telephone cable per point

_AC_READERS_PER_FLOOR  = 2      # access control: 2 doors per floor
_AC_READERS_PER_CTRL   = 8      # readers per controller unit

_NC_SQM_PER_STATION    = 30.0   # nurse call: 1 station per 30 sqm


# =============================================================================
# BUILDING-TYPE SETS
# =============================================================================

_ACCESS_CTRL_TYPES = frozenset({"hospital", "aiims", "office", "academic", "research"})
_NURSE_CALL_TYPES  = frozenset({"hospital", "aiims", "dispensary"})
_NO_DATA_TYPES     = frozenset({"dining", "utility"})
_NO_CCTV_TYPES     = frozenset({"dining", "utility"})

# data-point density modifier by building type
def _data_sqm_per_point(building_type: str) -> float:
    bt = building_type.lower()
    if bt == "hostel":
        return _DATA_SQM_PER_POINT_HOSTEL
    if bt in ("hospital", "aiims"):
        return _DATA_SQM_PER_POINT_HOSP
    # academic / office / research → full density
    return _DATA_SQM_PER_POINT_FULL


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ELVResult:
    """Result of run_elv_takeoff()."""
    line_items: List[dict]          = field(default_factory=list)
    camera_count: int               = 0
    smoke_detector_count: int       = 0
    data_point_count: int           = 0
    warnings: List[str]             = field(default_factory=list)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _item(description: str, unit: str, quantity: float) -> dict:
    """Create a standard ELV BOQ line item dict."""
    return {
        "description": description,
        "trade": "elv",
        "unit": unit,
        "quantity": round(quantity, 2),
        "source": "norm_based",
        "building": "ELV Systems",
    }


def _fire_alarm_items(
    floor_area_sqm: float,
    floors: int,
) -> List[dict]:
    """Generate fire alarm system BOQ items."""
    items: List[dict] = []

    # Smoke detectors
    detectors = math.ceil(floor_area_sqm / _FA_SQM_PER_DETECTOR)
    items.append(_item(
        "Supply, fix and connect addressable smoke detector (photo-electric type) "
        "complete with base, IS 2189 / NFPA 72",
        "Nos",
        detectors,
    ))

    # Manual call points — minimum 2 per floor
    mcp_raw   = math.ceil(floor_area_sqm / _FA_SQM_PER_MCP)
    mcp_floor = _FA_MIN_MCP_PER_FLOOR * floors
    mcps      = max(mcp_raw, mcp_floor)
    items.append(_item(
        "Supply, fix and connect manual call point (break-glass) complete with "
        "mounting box, IS 2189",
        "Nos",
        mcps,
    ))

    # Fire alarm control panel
    # 1 base panel + 1 per every 10 floors above 3
    extra_panels = max(0, (floors - 3) // 10)
    panel_count  = 1 + extra_panels
    items.append(_item(
        "Supply, install and commission microprocessor-based addressable fire "
        "alarm control panel (FACP) with display and battery backup, IS 2189",
        "Nos",
        panel_count,
    ))

    # FA cable
    cable_m = detectors * _FA_CABLE_PER_DETECTOR
    items.append(_item(
        "FR/FRLS 1.5 sqmm 2-core fire alarm cable in surface/concealed conduit, "
        "IS 1554 Part 2",
        "rm",
        cable_m,
    ))

    # Hooters / sounders
    hooters = math.ceil(floor_area_sqm / _FA_SQM_PER_HOOTER)
    items.append(_item(
        "Supply and fix hooter/sounder (electronic alarm siren) for fire alarm "
        "system complete with mounting bracket",
        "Nos",
        hooters,
    ))

    return items


def _cctv_items(floor_area_sqm: float) -> tuple[List[dict], int]:
    """Generate CCTV surveillance BOQ items. Returns (items, camera_count)."""
    items: List[dict] = []

    cameras = math.ceil(floor_area_sqm / _CCTV_SQM_PER_CAMERA)
    items.append(_item(
        "Supply, install and commission IP/HD CCTV camera (2MP, varifocal, "
        "IR night vision) complete with mounting hardware",
        "Nos",
        cameras,
    ))

    recorders = math.ceil(cameras / _CCTV_CH_PER_RECORDER)
    items.append(_item(
        f"Supply, install and commission NVR/DVR ({_CCTV_CH_PER_RECORDER}-channel) "
        "with 2TB HDD for CCTV recording, including power supply and rack",
        "Nos",
        recorders,
    ))

    cable_m = cameras * _CCTV_CABLE_PER_CAM
    items.append(_item(
        "CAT6 UTP/coaxial cable for CCTV including conduit, clamps and termination",
        "rm",
        cable_m,
    ))

    monitors = math.ceil(cameras / _CCTV_CAM_PER_MONITOR)
    items.append(_item(
        '24" LED monitor for CCTV surveillance station complete with stand',
        "Nos",
        monitors,
    ))

    return items, cameras


def _pa_items(floor_area_sqm: float) -> List[dict]:
    """Generate public address system BOQ items."""
    items: List[dict] = []

    speakers = math.ceil(floor_area_sqm / _PA_SQM_PER_SPEAKER)
    items.append(_item(
        "Supply, fix and connect ceiling/wall-mount PA speaker (6W, 100V line) "
        "complete with volume control, IS 2292",
        "Nos",
        speakers,
    ))

    amplifiers = math.ceil(speakers / _PA_SPK_PER_AMP)
    items.append(_item(
        "Supply, install and commission PA amplifier with zone selector, "
        "mic input and emergency override facility",
        "Nos",
        amplifiers,
    ))

    cable_m = speakers * _PA_CABLE_PER_SPEAKER
    items.append(_item(
        "2-core 1.5 sqmm FRLS PA cable in surface/concealed conduit "
        "including clamps and termination",
        "rm",
        cable_m,
    ))

    return items


def _data_networking_items(
    floor_area_sqm: float,
    floors: int,
    building_type: str,
    storey_height_m: float,
) -> tuple[List[dict], int]:
    """
    Generate data networking BOQ items. Returns (items, data_point_count).
    """
    items: List[dict] = []
    sqm_per_point = _data_sqm_per_point(building_type)

    data_points = math.ceil(floor_area_sqm / sqm_per_point)
    items.append(_item(
        "Supply and fix CAT6 data outlet (RJ45, face-plate, back box) "
        "complete with patch cord, TIA-568-C.2",
        "Nos",
        data_points,
    ))

    switches = math.ceil(data_points / _DATA_PORTS_PER_SWITCH)
    items.append(_item(
        f"Supply, install and commission {_DATA_PORTS_PER_SWITCH}-port managed "
        "Gigabit Ethernet network switch (rack-mount) complete with patch panel",
        "Nos",
        switches,
    ))

    cat6_cable_m = data_points * _DATA_CABLE_PER_POINT
    items.append(_item(
        "CAT6 UTP 4-pair cable in PVC conduit for data network "
        "including termination and testing, TIA-568",
        "rm",
        cat6_cable_m,
    ))

    # OFC backbone: floors × storey_height × 2 risers
    ofc_m = floors * storey_height_m * _DATA_OFC_RISERS
    items.append(_item(
        "6-core single-mode OFC backbone cable for inter-floor riser "
        "complete with splice trays, pigtails and connectors",
        "rm",
        round(ofc_m, 2),
    ))

    return items, data_points


def _structured_wiring_items(floor_area_sqm: float) -> List[dict]:
    """Generate structured wiring (telephone) BOQ items."""
    items: List[dict] = []

    tel_points = math.ceil(floor_area_sqm / _TEL_SQM_PER_POINT)
    items.append(_item(
        "Supply and fix telephone outlet (RJ11, face-plate, back box) "
        "complete with krone connection, IS 13252",
        "Nos",
        tel_points,
    ))

    cable_m = tel_points * _TEL_CABLE_PER_POINT
    items.append(_item(
        "2-pair 0.5mm CW1308 telephone cable in PVC conduit "
        "including termination and labelling",
        "rm",
        cable_m,
    ))

    return items


def _access_control_items(floors: int) -> List[dict]:
    """Generate access control BOQ items."""
    items: List[dict] = []

    readers = floors * _AC_READERS_PER_FLOOR
    items.append(_item(
        "Supply, install and commission proximity card reader / biometric "
        "access control reader (125 kHz/13.56 MHz) complete with mounting box",
        "Nos",
        readers,
    ))

    controllers = math.ceil(readers / _AC_READERS_PER_CTRL)
    items.append(_item(
        f"Supply, install and commission {_AC_READERS_PER_CTRL}-door access control "
        "controller with software, power supply and battery backup",
        "Nos",
        controllers,
    ))

    items.append(_item(
        "Supply and fix electric strike / electromagnetic lock for access "
        "controlled door complete with door closer",
        "Nos",
        readers,
    ))

    return items


def _nurse_call_items(floor_area_sqm: float, floors: int) -> List[dict]:
    """Generate nurse call system BOQ items."""
    items: List[dict] = []

    stations = math.ceil(floor_area_sqm / _NC_SQM_PER_STATION)
    items.append(_item(
        "Supply, install and commission patient/nurse call push-button unit "
        "complete with indicator lamp and cable, IS 10791",
        "Nos",
        stations,
    ))

    items.append(_item(
        "Supply, install and commission nurse call master station / annunciator "
        "panel at nurses' station (one per floor)",
        "Nos",
        floors,
    ))

    return items


# =============================================================================
# BUILDING-TYPE ADJUSTMENTS
# =============================================================================

def _resolve_flags(
    building_type: str,
    has_fire_alarm: bool,
    has_cctv: bool,
    has_pa: bool,
    has_data: bool,
) -> dict:
    """
    Apply building-type overrides and return resolved flags and derived booleans.
    """
    bt = building_type.lower()

    # Dining / utility override
    if bt in _NO_DATA_TYPES:
        has_data = False
    if bt in _NO_CCTV_TYPES:
        has_cctv = False

    has_access_control = bt in _ACCESS_CTRL_TYPES
    has_nurse_call     = bt in _NURSE_CALL_TYPES

    return {
        "has_fire_alarm":     has_fire_alarm,
        "has_cctv":           has_cctv,
        "has_pa":             has_pa,
        "has_data":           has_data,
        "has_access_control": has_access_control,
        "has_nurse_call":     has_nurse_call,
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_elv_takeoff(
    floor_area_sqm: float,
    floors: int,
    building_type: str = "hostel",
    has_fire_alarm: bool = True,
    has_cctv: bool = True,
    has_pa: bool = True,
    has_data: bool = True,
    storey_height_m: float = _DATA_STOREY_HEIGHT_M,
) -> ELVResult:
    """
    Generate ELV (Extra Low Voltage) system BOQ items from floor-area norms.

    Parameters
    ----------
    floor_area_sqm : float
        Total gross floor area across all floors (sqm).
    floors : int
        Number of storeys.
    building_type : str
        One of: hostel, academic, office, research, hospital, aiims,
        dispensary, dining, utility.  Controls data density, access control
        and nurse-call inclusion.
    has_fire_alarm : bool
        Include fire alarm system items.
    has_cctv : bool
        Include CCTV surveillance items.
    has_pa : bool
        Include public address system items.
    has_data : bool
        Include data networking + structured wiring items.
    storey_height_m : float
        Floor-to-floor height used for OFC backbone length (default 3.2 m).

    Returns
    -------
    ELVResult
    """
    result   = ELVResult()
    warnings: List[str] = []

    # ── Guards ──────────────────────────────────────────────────────────────
    if floor_area_sqm <= 0:
        warnings.append("floor_area_sqm is zero or negative — no ELV items computed.")
        result.warnings = warnings
        return result

    floors = max(1, int(floors))

    # ── Resolve flags from building type ────────────────────────────────────
    flags = _resolve_flags(
        building_type, has_fire_alarm, has_cctv, has_pa, has_data
    )

    line_items: List[dict] = []

    # ── 1. Fire Alarm ────────────────────────────────────────────────────────
    if flags["has_fire_alarm"]:
        fa_items = _fire_alarm_items(floor_area_sqm, floors)
        line_items.extend(fa_items)
        # smoke detector count — first FA item
        det_item = next(
            (it for it in fa_items if "smoke detector" in it["description"].lower()),
            None,
        )
        result.smoke_detector_count = int(det_item["quantity"]) if det_item else 0
    else:
        warnings.append("Fire alarm system excluded (has_fire_alarm=False).")

    # ── 2. CCTV ──────────────────────────────────────────────────────────────
    if flags["has_cctv"]:
        cctv_items, cam_count = _cctv_items(floor_area_sqm)
        line_items.extend(cctv_items)
        result.camera_count = cam_count
    else:
        warnings.append("CCTV system excluded (has_cctv=False or dining/utility type).")

    # ── 3. PA System ─────────────────────────────────────────────────────────
    if flags["has_pa"]:
        line_items.extend(_pa_items(floor_area_sqm))
    else:
        warnings.append("PA system excluded (has_pa=False).")

    # ── 4. Data Networking ───────────────────────────────────────────────────
    if flags["has_data"]:
        data_items, dp_count = _data_networking_items(
            floor_area_sqm, floors, building_type, storey_height_m
        )
        line_items.extend(data_items)
        result.data_point_count = dp_count

        # Structured wiring (telephone) — always alongside data
        line_items.extend(_structured_wiring_items(floor_area_sqm))
    else:
        warnings.append("Data networking system excluded (has_data=False or dining/utility type).")

    # ── 5. Access Control ────────────────────────────────────────────────────
    if flags["has_access_control"]:
        line_items.extend(_access_control_items(floors))
    # (no warning — it is expected to be absent for hostel/dining)

    # ── 6. Nurse Call ─────────────────────────────────────────────────────────
    if flags["has_nurse_call"]:
        line_items.extend(_nurse_call_items(floor_area_sqm, floors))
    # (no warning — expected to be absent for non-clinical buildings)

    if not line_items:
        warnings.append(
            f"No ELV items generated for building_type='{building_type}' "
            "with all systems disabled."
        )

    result.line_items = line_items
    result.warnings   = warnings

    logger.debug(
        "elv_takeoff: area=%.0f sqm, floors=%d, type=%s, items=%d, "
        "detectors=%d, cameras=%d, data_points=%d",
        floor_area_sqm, floors, building_type, len(line_items),
        result.smoke_detector_count, result.camera_count, result.data_point_count,
    )
    return result
