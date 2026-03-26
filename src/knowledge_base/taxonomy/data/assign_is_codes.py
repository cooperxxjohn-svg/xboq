#!/usr/bin/env python3
"""
Assign IS codes to items with empty is_code_ref fields in:
  - 10_doors_windows.yaml
  - 11_plumbing.yaml
  - 12_electrical.yaml

Mapping is based on item ID prefix (sub-trade segment) and standard_name keywords.
Existing non-empty values are never overwritten.
"""

import yaml
import os
import re
from collections import defaultdict

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def kw(*words):
    """Return a compiled OR regex for the given keywords (case-insensitive)."""
    return re.compile('|'.join(re.escape(w) for w in words), re.IGNORECASE)


# ---------------------------------------------------------------------------
# DOORS & WINDOWS  (10_doors_windows.yaml)
# ---------------------------------------------------------------------------
# Key: 3-segment ID prefix  →  (default_code, [(regex, override_code), ...])
# The first matching regex wins; if none match the default is used.

DW_RULES: dict[str, tuple[str, list]] = {

    # Doors ---------------------------------------------------------------
    "DW.DR.FLS": ("IS 2202", []),          # flush doors (timber)
    "DW.DR.PNL": ("IS 2202", [           # panel doors
        (kw("glass"), "IS 2202, IS 14900"),
    ]),
    "DW.DR.HMD": ("IS 1038", []),          # hollow metal doors → steel
    "DW.DR.FRD": ("IS 3614", []),          # fire-rated doors
    "DW.DR.SLD": ("IS 2202", [            # sliding doors by material
        (kw("Glass", "Frameless", "Toughened"), "IS 2553"),
        (kw("Alumin"), "IS 1948"),
        (kw("UPVC", "PVC"), "IS 4985"),
        (kw("Barn", "Wood"), "IS 2202"),
    ]),
    "DW.DR.FLD": ("IS 2202", [            # folding / bi-fold
        (kw("Alumin", "Glass"), "IS 1948"),
    ]),
    "DW.DR.AUT": ("NBC 2016", []),         # automatic doors
    "DW.DR.RLS": ("IS 6248", []),          # roller shutters / rolling shutters
    "DW.DR.GLS": ("IS 2553", []),          # glass doors (toughened)
    "DW.DR.MAT": ("NBC 2016", [           # door materials (WPC/PVC/FRP/MDF)
        (kw("WPC", "Wood Polymer"), "NBC 2016"),
        (kw("PVC"), "IS 4985"),
        (kw("FRP"), "NBC 2016"),
        (kw("MDF"), "IS 2202"),
    ]),
    "DW.DR.PNS": ("IS 14900", [           # panel/glazed doors
        (kw("alumin"), "IS 1948"),
    ]),
    "DW.DR.SPL": ("IS 1948", [            # specialty / sliding-folding
        (kw("Pocket", "Wood"), "IS 2202"),
        (kw("Revolv"), "NBC 2016"),
        (kw("Fire"), "IS 3614"),
        (kw("Automatic", "Auto"), "NBC 2016"),
    ]),

    # Frames --------------------------------------------------------------
    "DW.FR.ALF": ("IS 1948", []),          # aluminium frames
    "DW.FR.MSH": ("IS 1038", []),          # MS hollow section frames → steel
    "DW.FR.SS":  ("IS 1038", []),          # stainless steel frames
    "DW.FR.UPV": ("IS 4985", []),          # UPVC frames
    "DW.FR.WPC": ("NBC 2016", []),         # WPC frames

    # Windows -------------------------------------------------------------
    "DW.WN.ALC": ("IS 1948", []),          # aluminium casement
    "DW.WN.ALF": ("IS 1948", [            # aluminium fixed / curtain wall
        (kw("Curtain Wall"), "IS 14900, IS 1948"),
    ]),
    "DW.WN.ALT": ("IS 1948", []),          # aluminium top-hung / awning
    "DW.WN.LVR": ("IS 1948", []),          # louvre / aluminium louvre
    "DW.WN.TMB": ("IS 1003", []),          # timber windows
    "DW.WN.UPV": ("IS 4985", []),          # UPVC windows

    # Glass ---------------------------------------------------------------
    "DW.GL.DGU": ("IS 14900", []),         # double-glazed units (float base)
    "DW.GL.FRS": ("IS 14900", []),         # frosted / acid-etched glass
    "DW.GL.LAM": ("IS 2553 Part 2", []),   # laminated glass
    "DW.GL.RFL": ("IS 14900", []),         # reflective glass
    "DW.GL.SPC": ("IS 14900", [           # specialty glass
        (kw("Laminat"), "IS 2553 Part 2"),
        (kw("Toughen", "Temper"), "IS 2553"),
        (kw("Wired"), "IS 14900"),
    ]),
    "DW.GL.TNT": ("IS 14900", []),         # tinted glass

    # Hardware ------------------------------------------------------------
    "DW.HW.HNG": ("IS 1341", []),          # hinges
    "DW.HW.FLS": ("IS 3564", []),          # floor springs
    "DW.HW.LCK": ("IS 1341", [           # locks
        (kw("Digital", "Biometric", "Keypad", "Smart"), "NBC 2016"),
    ]),
    "DW.HW.CLK": ("IS 1341", []),          # cylindrical locks
    "DW.HW.CLS": ("IS 3564", []),          # door closers
    "DW.HW.DCL": ("IS 3564", []),          # door closers (alt prefix)
    "DW.HW.HDL": ("IS 1341", []),          # handles
    "DW.HW.BLT": ("IS 1341", []),          # tower bolts / accessories
    "DW.HW.TBA": ("IS 1341", []),          # tower bolts (alt)
    "DW.HW.ACC": ("IS 1341", [           # accessories
        (kw("Panic", "Crash Bar"), "NBC 2016"),
        (kw("Kick Plate", "Push Plate", "Pull Plate"), "IS 1341"),
        (kw("Door Viewer", "Peephole"), "IS 1341"),
        (kw("Chain"), "IS 1341"),
    ]),
    "DW.HW.DGL": ("NBC 2016", []),         # digital locks
    "DW.HW.FRA": ("IS 4021", []),          # frame accessories / hold-fasts

    # Grills & gates ------------------------------------------------------
    "DW.GR.MS":  ("IS 1200 Part 12", []), # MS grills (measurement standard for windows)
    "DW.GR.SS":  ("IS 1200 Part 12", []), # SS grills
    "DW.GR.CLG": ("IS 6248", []),          # collapsible gates
    "DW.GR.EXG": ("IS 6248", []),          # expandable gates
    "DW.GR.EXM": ("IS 1200 Part 12", []), # expanded metal mesh

    # Mosquito mesh -------------------------------------------------------
    "DW.MM.AL":  ("NBC 2016", []),
    "DW.MM.FBR": ("NBC 2016", []),
    "DW.MM.PLT": ("NBC 2016", []),
    "DW.MM.RET": ("NBC 2016", []),
    "DW.MM.SS":  ("NBC 2016", []),

    # Ventilators ---------------------------------------------------------
    "DW.VN.FXD": ("IS 1948", []),          # fixed louvre ventilators (aluminium)
    "DW.VN.ADJ": ("IS 1948", []),          # adjustable louvre ventilators
    "DW.VN.EXH": ("IS 1948", []),          # exhaust fan openings (aluminium frame)
}

# Also handle the many "additional / extended" sub-trade blocks that share
# the same category logic by normalising their ID prefix to the canonical one.
DW_PREFIX_ALIAS: dict[str, str] = {
    # additional sections reuse same logic as canonical sections
    "DW.DR.FRD": "DW.DR.FRD",   # already canonical
}


# ---------------------------------------------------------------------------
# PLUMBING  (11_plumbing.yaml)
# ---------------------------------------------------------------------------

PLB_RULES: dict[str, tuple[str, list]] = {

    # Water supply pipes --------------------------------------------------
    "PLB.WS.GI":  ("IS 1239", []),
    "PLB.WS.CPV": ("IS 15778", []),
    "PLB.WS.PPR": ("IS 15801", []),
    "PLB.WS.PEX": ("IS 15801", []),       # closest; PEX covered under NBC 2016 / IS 15801 family
    "PLB.WS.SS":  ("NBC 2016", []),        # SS press-fit pipes – no specific BIS, NBC 2016
    "PLB.WS.FIT": ("IS 1239", [          # pipe fittings by material
        (kw("CPVC"), "IS 15778"),
        (kw("PPR"), "IS 15801"),
        (kw("HDPE"), "IS 4984"),
        (kw("UPVC", "PVC"), "IS 4985"),
        (kw("GI", "Galvan"), "IS 1239"),
    ]),
    "PLB.WS.VLV": ("IS 778", [           # valves
        (kw("Ball"), "IS 9890"),
        (kw("Gate"), "IS 778"),
        (kw("Butterfly"), "IS 13095"),
        (kw("Check", "NRV", "Non Return"), "IS 5312"),
        (kw("Float", "Float Valve"), "IS 9890"),
        (kw("Solenoid"), "NBC 2016"),
        (kw("PRV", "Pressure Reduc"), "NBC 2016"),
        (kw("TMV", "Thermostatic"), "NBC 2016"),
    ]),
    "PLB.WS.PPF": ("IS 15801", []),        # PPR fittings

    # Drainage pipes ------------------------------------------------------
    "PLB.DR.HDP": ("IS 4984", []),         # HDPE soil/waste
    "PLB.DR.ACC": ("IS 1726", [          # drain accessories
        (kw("Grease Trap"), "NBC 2016"),
        (kw("Inspection"), "NBC 2016"),
    ]),
    "PLB.DR.TRP": ("IS 1726", [          # traps / drain covers
        (kw("Gully"), "IS 1726"),
        (kw("Floor Trap", "Nahni"), "IS 1726"),
    ]),
    "PLB.DR.GRT": ("NBC 2016", []),        # grease traps
    "PLB.DR.ICH": ("NBC 2016", []),        # inspection chambers

    # Sanitary fixtures ---------------------------------------------------
    "PLB.FX.BSN": ("IS 2556 Part 4", []), # wash basins
    "PLB.FX.EWC": ("IS 2556 Part 2", []), # EWCs / WCs
    "PLB.FX.URN": ("IS 2556 Part 6", []), # urinals
    "PLB.FX.SNK": ("IS 2556", [          # sinks
        (kw("SS", "Steel"), "NBC 2016"),   # kitchen SS sinks not under IS 2556
    ]),
    "PLB.SF.EWC": ("IS 2556 Part 2", []), # EWC (sanitary fixtures sub-trade)
    "PLB.SF.URN": ("IS 2556 Part 6", []),
    "PLB.SF.WB":  ("IS 2556 Part 4", []),
    "PLB.SF.BTH": ("IS 2556", []),         # bathtubs
    "PLB.SF.SHW": ("IS 2548", []),         # shower heads

    # Taps / faucets / mixers / cocks -------------------------------------
    "PLB.CP.TAP": ("IS 1795", []),
    "PLB.CP.MIX": ("IS 1795", []),
    "PLB.CP.SHR": ("IS 2548", []),
    "PLB.CP.COK": ("IS 1795", []),
    "PLB.CP.FLH": ("IS 2556 Part 2", []), # flush valves (for WCs)
    "PLB.CP.HLT": ("IS 2548", []),         # health faucet

    # Valves (standalone) -------------------------------------------------
    "PLB.VL.BTF": ("IS 13095", []),        # butterfly valves
    "PLB.VL.CHK": ("IS 5312", []),         # check / NRV valves
    "PLB.VL.FLT": ("IS 9890", []),         # float valves
    "PLB.VL.PRV": ("NBC 2016", []),        # pressure reducing valves
    "PLB.VL.SOL": ("NBC 2016", []),        # solenoid valves
    "PLB.VL.TMV": ("NBC 2016", []),        # thermostatic mixing valves

    # Tanks ---------------------------------------------------------------
    "PLB.TK.OH":  ("IS 12701", [          # overhead tanks
        (kw("FRP"), "NBC 2016"),
        (kw("SS", "Steel"), "NBC 2016"),
        (kw("Polyethylene", "HDPE", "LLDPE", "Plastic"), "IS 12701"),
        (kw("RCC", "Concrete"), "NBC 2016"),
    ]),
    "PLB.TK.SS":  ("NBC 2016", []),        # SS tanks
    "PLB.TK.UG":  ("NBC 2016", []),        # underground sumps (RCC)
    "PLB.TK.PRS": ("NBC 2016", []),        # pressure booster system

    # Pumps ---------------------------------------------------------------
    "PLB.PM.CEN": ("IS 1520", []),         # centrifugal pumps
    "PLB.PM.SUB": ("IS 8034", []),         # submersible pumps
    "PLB.PM.HPN": ("NBC 2016", []),        # hydropneumatic systems
    "PLB.PM.BST": ("NBC 2016", []),        # booster pumps

    # Accessories ---------------------------------------------------------
    "PLB.AC.TWL": ("NBC 2016", []),        # towel rails / bathroom accessories
    "PLB.FT.EXJ": ("NBC 2016", []),        # expansion joints
    "PLB.FT.FST": ("NBC 2016", []),        # pipe supports / clamps
    "PLB.FT.STR": ("NBC 2016", []),        # strainers

    # Hot water systems ---------------------------------------------------
    "PLB.HW.GYS": ("NBC 2016", []),        # electric geysers
    "PLB.HW.SOL": ("NBC 2016", []),        # solar water heaters
    "PLB.HW.INS": ("NBC 2016", []),        # instant heaters
    "PLB.HW.HTP": ("NBC 2016", []),        # heat pump water heaters
    "PLB.HW.BLR": ("NBC 2016", []),        # central boilers

    # Insulation ----------------------------------------------------------
    "PLB.IN.GWL": ("NBC 2016", []),        # glass wool insulation
    "PLB.IN.NBR": ("NBC 2016", []),        # nitrile rubber insulation
    "PLB.IN.PUF": ("NBC 2016", []),        # PUF insulation

    # Gas piping ----------------------------------------------------------
    "PLB.GAS.PIP": ("NBC 2016", []),       # gas pipes (LP/CNG – NBC 2016 Part 4)
    "PLB.GAS.FIT": ("NBC 2016", []),       # gas fittings

    # Water treatment -----------------------------------------------------
    "PLB.WT.ROP": ("IS 10500", []),        # RO plants
    "PLB.WT.SFT": ("IS 10500", []),        # water softeners
    "PLB.WT.ACF": ("IS 10500", []),        # activated carbon filters
    "PLB.WT.IRN": ("IS 10500", []),        # iron removal filters
    "PLB.WT.SED": ("IS 10500", []),        # sediment filters
    "PLB.WT.UVS": ("IS 10500", []),        # UV disinfection

    # Sewage / STP --------------------------------------------------------
    "PLB.STP.MBB": ("NBC 2016", []),
    "PLB.STP.MBR": ("NBC 2016", []),
    "PLB.STP.PKG": ("NBC 2016", []),
    "PLB.STP.SBR": ("NBC 2016", []),
    "PLB.STP.TTW": ("NBC 2016", []),

    # Rainwater harvesting ------------------------------------------------
    "PLB.RWH.FLT": ("NBC 2016", []),
    "PLB.RWH.PPC": ("NBC 2016", []),
    "PLB.RWH.RTF": ("NBC 2016", []),
    "PLB.RWH.TNK": ("NBC 2016", []),

    # Measurement ---------------------------------------------------------
    "PLB.MM":      ("IS 1200 Part 16", []),
}


# ---------------------------------------------------------------------------
# ELECTRICAL  (12_electrical.yaml)
# ---------------------------------------------------------------------------

ELC_RULES: dict[str, tuple[str, list]] = {

    # Panels --------------------------------------------------------------
    "ELC.PNL.MSB": ("IS 8623", []),
    "ELC.PNL.SMP": ("IS 8623", []),
    "ELC.PNL.MCC": ("IS 8623", []),
    "ELC.PNL.PCC": ("IS 8623", []),
    "ELC.PNL.APC": ("IS 8623", []),        # APFC panel
    "ELC.PNL.APF": ("IS 8623", []),        # APFC panel (alt prefix)
    "ELC.PNL.COS": ("IS 13947", []),       # changeover / ATS switches
    "ELC.DB.DBX": ("IS 8623", []),         # distribution boards
    "ELC.DB.BUS": ("IS 8623", []),         # bus bars / rising mains
    "ELC.DB.RCB": ("IS 8828", []),         # RCBO (similar to MCB standard)
    "ELC.DB.SFU": ("IS 13947", []),        # switch fuse units

    # Wiring / conduits ---------------------------------------------------
    "ELC.WR.CDT": ("IS 9537", [          # conduits
        (kw("PVC", "Rigid"), "IS 9537"),
        (kw("GI", "Steel"), "IS 9537"),
        (kw("Flexible"), "IS 9537"),
    ]),
    "ELC.CD.FLX": ("IS 9537", []),         # flexible conduits
    "ELC.CT.PRF": ("NBC 2016", []),        # cable trays perforated
    "ELC.CT.LDR": ("NBC 2016", []),        # cable trays ladder
    "ELC.CT.RCW": ("NBC 2016", []),        # PVC raceways/trunking
    "ELC.WR.CTR": ("NBC 2016", []),        # cable trays (alt prefix)

    # Cables --------------------------------------------------------------
    "ELC.CB.CTL": ("IS 1554", [          # control / instrumentation cables
        (kw("Instrument", "Screened"), "IS 1554 Part 2"),
    ]),
    "ELC.CB.FBR": ("NBC 2016", []),        # fibre optic cables

    # Points (wiring points) ----------------------------------------------
    "ELC.PT.LGT": ("IS 1200 Part 20", []), # light / fan points (measurement)
    "ELC.PT.PWR": ("IS 1200 Part 20", []), # power socket points
    "ELC.PT.DAT": ("NBC 2016", []),        # data / RJ45 points
    "ELC.PT.MSC": ("IS 1200 Part 20", []), # miscellaneous points

    # Switches / sockets --------------------------------------------------
    "ELC.SW.PLT": ("IS 3854", []),         # modular plates
    "ELC.SW.REG": ("IS 3854", []),         # fan regulators
    "ELC.SW.SKT": ("IS 3854", []),         # sockets / outlets
    "ELC.SW.BLP": ("IS 3854", []),         # bell push switch
    "ELC.SW.FLB": ("IS 3854", []),         # floor boxes

    # Meters / energy metering --------------------------------------------
    "ELC.EM.MTR": ("IS 1248", []),         # energy meters

    # Earthing ------------------------------------------------------------
    "ELC.ER.CHM": ("IS 3043", []),         # chemical earth electrodes
    "ELC.ER.EBA": ("IS 3043", []),         # earth bus bars

    # Lighting fixtures ---------------------------------------------------
    "ELC.FX.LED": ("IS 16107", []),        # LED lighting general
    "ELC.FX.PNL": ("IS 16107", []),        # LED panel lights (alt prefix)
    "ELC.FX.DEC": ("IS 10322", []),        # decorative lighting
    "ELC.FX.EMR": ("IS 10322", []),        # emergency / exit lights
    "ELC.FX.FAN": ("IS 10322", [          # fans
        (kw("Exhaust"), "NBC 2016"),
        (kw("Ceiling Fan", "BLDC"), "IS 374"),
    ]),
    "ELC.FX.FLD": ("IS 16107", []),        # LED flood lights
    "ELC.FX.HBY": ("IS 16107", []),        # LED high bay lights
    "ELC.FX.IND": ("IS 16107", []),        # LED industrial / well glass
    "ELC.FX.STL": ("NBC 2016", []),        # LED street lights
    "ELC.FX.SOL": ("NBC 2016", []),        # solar LED street lights
    "ELC.FX.STP": ("IS 16107", []),        # LED strip lights

    # Lightning protection ------------------------------------------------
    "ELC.LPS.ESE": ("IS 2309", []),        # ESE lightning terminals
    "ELC.LPS.TST": ("IS 2309", []),        # test clamps for lightning

    # UPS / inverters / batteries -----------------------------------------
    "ELC.UPS.ONL": ("IS 16095", []),       # online UPS
    "ELC.UPS.INV": ("IS 16095", []),       # home inverters
    "ELC.UPS.BAT": ("NBC 2016", []),       # SMF batteries
    "ELC.UPS.DGS": ("NBC 2016", []),       # DG synchronization panels

    # Generators ----------------------------------------------------------
    "ELC.DG.015": ("IS 10002", []),
    "ELC.DG.025": ("IS 10002", []),
    "ELC.DG.040": ("IS 10002", []),
    "ELC.DG.062": ("IS 10002", []),
    "ELC.DG.082": ("IS 10002", []),
    "ELC.DG.100": ("IS 10002", []),
    "ELC.DG.125": ("IS 10002", []),
    "ELC.DG.160": ("IS 10002", []),
    "ELC.DG.200": ("IS 10002", []),
    "ELC.DG.250": ("IS 10002", []),
    "ELC.DG.320": ("IS 10002", []),
    "ELC.DG.380": ("IS 10002", []),
    "ELC.DG.500": ("IS 10002", []),
    "ELC.DG.ACC": ("IS 10002", [          # DG accessories
        (kw("AMF", "Synchroniz"), "IS 8623"),
        (kw("Acoustic", "Enclosure"), "NBC 2016"),
    ]),

    # Structured cabling / ICT --------------------------------------------
    "ELC.SC.CAB": ("NBC 2016", []),        # Cat6/Cat6A cables
    "ELC.SC.FPC": ("NBC 2016", []),        # fibre patch cords
    "ELC.SC.IOT": ("NBC 2016", []),        # information outlets
    "ELC.SC.PPL": ("NBC 2016", []),        # patch panels
    "ELC.SC.RCK": ("NBC 2016", []),        # network racks

    # Low voltage / BMS / security ----------------------------------------
    "ELC.LV.CTV": ("NBC 2016", []),        # CCTV cameras
    "ELC.LV.ACS": ("NBC 2016", []),        # access control
    "ELC.LV.FA":  ("NBC 2016", []),        # fire alarm panels
    "ELC.LV.BMS": ("NBC 2016", []),        # BMS DDC controllers
    "ELC.LV.PA":  ("NBC 2016", []),        # PA systems
    "ELC.LV.ICM": ("NBC 2016", []),        # IP intercom
    "ELC.LV.WFI": ("NBC 2016", []),        # WiFi access points

    # CCTV (cc sub-trade) ------------------------------------------------
    "ELC.CC.CAM": ("NBC 2016", []),
    "ELC.CC.NVR": ("NBC 2016", []),
    "ELC.CC.ACS": ("NBC 2016", []),
    "ELC.CC.ANP": ("NBC 2016", []),
    "ELC.CC.BMB": ("NBC 2016", []),
    "ELC.CC.VDP": ("NBC 2016", []),

    # Fire alarm (fa sub-trade) -------------------------------------------
    "ELC.FA.HTR": ("NBC 2016", []),        # hooters / sounders
    "ELC.FA.PAS": ("NBC 2016", []),        # PA system
    "ELC.FA.RIN": ("NBC 2016", []),        # response indicators

    # Solar ---------------------------------------------------------------
    "ELC.SOL.PNL": ("NBC 2016", []),       # solar panels
    "ELC.SOL.INV": ("NBC 2016", []),       # solar inverters (string)
    "ELC.SOL.SIV": ("NBC 2016", []),       # solar on-grid inverters
    "ELC.SOL.BOS": ("NBC 2016", []),       # BoS (cables, mounting structure)
    "ELC.SOL.MNT": ("NBC 2016", []),       # mounting structures
    "ELC.SOL.NMP": ("NBC 2016", []),       # net metering panels

    # EV charging ---------------------------------------------------------
    "ELC.EV.ACH": ("NBC 2016", []),
    "ELC.EV.ACS": ("NBC 2016", []),
    "ELC.EV.DCF": ("NBC 2016", []),
    "ELC.EV.DCH": ("NBC 2016", []),
    "ELC.EV.FND": ("NBC 2016", []),
}

# ---------------------------------------------------------------------------
# Core lookup function
# ---------------------------------------------------------------------------

def get_is_code(item_id: str, standard_name: str, rules: dict) -> str | None:
    """
    Return the IS code for an item, or None if no rule matches.
    Tries longest matching prefix first (up to 3 dot-segments).
    """
    parts = item_id.split(".")
    # Try 3-segment prefix, then 2-segment
    for n in (3, 2):
        prefix = ".".join(parts[:n])
        if prefix in rules:
            default_code, overrides = rules[prefix]
            for pattern, code in overrides:
                if pattern.search(standard_name):
                    return code
            return default_code
    return None


# ---------------------------------------------------------------------------
# YAML traversal
# ---------------------------------------------------------------------------

def process_items(obj, rules: dict, updated: list, samples: list):
    """Recursively walk the loaded YAML object and fill empty is_code_ref."""
    if isinstance(obj, dict):
        if "id" in obj and "is_code_ref" in obj:
            if not obj["is_code_ref"]:  # empty string or None
                code = get_is_code(obj["id"], obj.get("standard_name", ""), rules)
                if code:
                    obj["is_code_ref"] = code
                    updated.append(obj["id"])
                    if len(samples) < 10:
                        samples.append((obj["id"], obj.get("standard_name", ""), code))
        for v in obj.values():
            process_items(v, rules, updated, samples)
    elif isinstance(obj, list):
        for item in obj:
            process_items(item, rules, updated, samples)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FILES = [
    ("10_doors_windows.yaml", DW_RULES),
    ("11_plumbing.yaml",      PLB_RULES),
    ("12_electrical.yaml",    ELC_RULES),
]


def main():
    for filename, rules in FILES:
        path = os.path.join(DATA_DIR, filename)

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        updated: list[str] = []
        samples: list[tuple] = []
        process_items(data, rules, updated, samples)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                data,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=200,
            )

        print(f"\n{'='*60}")
        print(f"File : {filename}")
        print(f"Items updated: {len(updated)}")
        print(f"Sample assignments (first 10):")
        for id_, name, code in samples:
            print(f"  {id_:<30} | {code:<25} | {name[:50]}")

    print("\nDone.")


if __name__ == "__main__":
    main()
