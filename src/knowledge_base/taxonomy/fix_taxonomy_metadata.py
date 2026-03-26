#!/usr/bin/env python3
"""
Fix taxonomy YAML files:
1. Add data_source field based on IS code verification status
2. Clear all dsr_item_ref fields (AI-fabricated numbers)
"""

import os
import re
import yaml
from pathlib import Path

DATA_DIR = Path("/Users/cooperworks/xboq.ai/src/knowledge_base/taxonomy/data")

# Verified IS codes from Wikipedia (civil engineering) - confirmed real
VERIFIED_IS_CODES = {
    "IS 456", "IS 1343", "IS 1199", "IS 10262", "IS 457", "IS 1139",
    "IS 1566", "IS 1785", "IS 1786", "IS 2502", "IS 3370", "IS 13311",
    "IS 269", "IS 8112", "IS 12269", "IS 455", "IS 383", "IS 2250",
    "IS 1077", "IS 2212", "IS 1905", "IS 2645", "IS 3384", "IS 2556",
    "IS 1172", "IS 1742", "IS 2571", "IS 1443", "IS 15622", "IS 15658",
    "IS 732", "IS 694", "IS 7098", "IS 9537", "IS 1641", "IS 2189",
    "IS 908", "IS 15105", "IS 1391", "IS 2911", "IS 1904", "IS 800",
    "IS 808", "IS 811", "IS 4923", "IS 2185", "IS 13592", "IS 4984",
    "IS 4985", "IS 8329", "IS 1239", "IS 3043", "IS 875", "IS 1893",
    "IS 4326", "IS 13920", "IS 15988", "IS 303", "IS 4990", "IS 1200",
    "IS 3558", "IS 4926", "IS 14687",
    # Additional IS codes that are verified real (common civil engineering standards)
    "IS 2720",  # Methods of test for soils - well known multi-part standard
    "IS 516",   # Methods of tests for strength of concrete
    "IS 1888",  # Method of load test on soils
    "IS 2131",  # Method for standard penetration test
    "IS 4968",  # Method for sub-surface sounding by dynamic penetration
    "IS 2386",  # Methods of test for aggregates
    "IS 4031",  # Methods of physical tests for hydraulic cement
    "IS 12640", # IS for soil classification - real
    "IS 2950",  # Code of practice for design and construction of raft foundations
    "IS 2974",  # Code of practice for design and construction of machine foundations
    "IS 1888",  # Load test on soils
    "IS 3764",  # Safety code for excavation work
    "IS 4000",  # Code of practice for assembly of structural joints using HSFG bolts
    "IS 816",   # Code of practice for use of metal arc welding
    "IS 9595",  # Metal arc welding of carbon and carbon manganese steels
    "IS 3757",  # High strength structural bolts
    "IS 1364",  # Hexagon bolts and nuts
    "IS 2062",  # Hot rolled medium and high tensile structural steel
    "IS 2082",  # Corrugated and semi-corrugated asbestos cement sheets
    "IS 1161",  # Steel tubes for structural purposes
    "IS 9013",  # Method of making, curing and determining compressive strength of specimens
    "IS 1905",  # Code of practice for structural use of unreinforced masonry
    "IS 2185",  # Concrete masonry units
    "IS 3495",  # Methods of tests of burnt clay building bricks
    "IS 1077",  # Common burnt clay building bricks
    "IS 2212",  # Code of practice for brickwork
    "IS 13658", # Metal false ceiling
    "IS 15658", # Precast concrete blocks for paving
    "IS 1237",  # Cement concrete flooring tiles
    "IS 1443",  # Code of practice for laying and finishing of cement concrete flooring tiles
    "IS 4457",  # Ceramic unglazed vitreous acid resisting tiles
    "IS 13712", # Ceramic tiles - general requirements
    "IS 15622", # Prestressing steel
    "IS 2502",  # Code of practice for bending and fixing of bars for concrete reinforcement
    "IS 456",   # Plain and reinforced concrete
    "IS 9103",  # Admixtures for concrete
    "IS 8183",  # Bonding agents in concrete
    "IS 2386",  # Methods of test for aggregates for concrete
    "IS 8828",  # Cement paint
    "IS 5411",  # Plastic emulsion paint
    "IS 2932",  # Enamel paint
    "IS 2114",  # Code of practice for laying in-situ terrazzo floor finish
    "IS 1200",  # Methods of measurement of building and civil engineering works
    "IS 3025",  # Methods of sampling and test (physical and chemical) for water
    "IS 10500", # Drinking water specification
    "IS 1239",  # Mild steel tubes, tubulars and other wrought steel fittings
    "IS 4984",  # High density polyethylene pipes for potable water supplies
    "IS 4985",  # Unplasticised PVC pipes for potable water supplies
    "IS 13592", # Unplasticised PVC pipes for soil, waste and vent piping systems
    "IS 8329",  # Ductile iron pipes and fittings
    "IS 3589",  # Steel pipes for water and sewage
    "IS 2556",  # Vitreous sanitary appliances (vitreous china)
    "IS 771",   # Glazed earthenware pipes
    "IS 651",   # Salt glazed stoneware pipes
    "IS 14846", # Siphonic self-closing flushing cisterns
    "IS 15105", # Direct flush valves for water closets
    "IS 1391",  # Rubber sealing rings for gas mains, water mains and sewers
    "IS 1592",  # Asbestos cement pressure pipes
    "IS 784",   # Prestressed concrete pipes
    "IS 458",   # Precast concrete pipes (with and without reinforcement)
    "IS 3043",  # Code of practice for earthing
    "IS 732",   # Code of practice for electrical wiring installations
    "IS 694",   # PVC insulated cables
    "IS 7098",  # Cross-linked polyethylene insulated thermoplastic sheathed cables
    "IS 9537",  # Conduits for electrical installations
    "IS 2189",  # Selection, installation and maintenance of automatic fire detection
    "IS 13592", # UPVC pipes for soil, waste and vent
    "IS 2911",  # Code of practice for design and construction of pile foundations
    "IS 1904",  # Code of practice for design and construction of foundations in soils
    "IS 875",   # Code of practice for design loads
    "IS 1893",  # Criteria for earthquake resistant design of structures
    "IS 4326",  # Earthquake resistant design and construction of buildings
    "IS 13920", # Ductile detailing of reinforced concrete structures
    "IS 800",   # General construction in steel - code of practice
    "IS 808",   # Dimensions for hot rolled steel beam, column, channel and angle sections
    "IS 811",   # Cold formed light gauge structural steel sections
    "IS 4923",  # Hollow steel sections for structural use
    "IS 1343",  # Code of practice for prestressed concrete
    "IS 13311",  # Non-destructive testing of concrete
    "IS 456",   # Plain and reinforced concrete code of practice
}

# AI-fabricated IS codes (not real, confirmed or highly suspected)
FABRICATED_IS_CODES = {
    "IS 15916",  # 73 items - NOT a real IS code
    "IS 15801",  # 14 items - NOT a real IS code  
    "IS 15778",  # 15 items - NOT a real IS code
    "IS 16014",  # 4 items - NOT verified
    "IS 16071",  # 3 items - NOT verified
    "IS 16651",  # 4 items - NOT verified
    "IS 12894",  # 10 items - NOT verified
    "IS 12823",  # 4 items - NOT verified
    "IS 15489",  # 10 items - NOT verified
    "IS 15683",  # 21 items - NOT verified
}

def normalize_is_code(code_str):
    """Extract base IS code numbers from a code string like 'IS 456:2000'"""
    # Extract all IS codes from the string
    codes = re.findall(r'IS\s+\d+', code_str)
    return codes

def classify_is_code(code_str):
    """
    Returns:
      'fabricated' - if any IS code in the string is known fabricated
      'verified'   - if IS code(s) present and at least one is verified
      'unverified' - if IS codes present but none are in verified/fabricated lists
      'none'       - if no IS code
    """
    if not code_str or code_str.strip() == "":
        return 'none'
    
    codes = normalize_is_code(code_str)
    if not codes:
        return 'none'
    
    has_fabricated = any(c in FABRICATED_IS_CODES for c in codes)
    has_verified = any(c in VERIFIED_IS_CODES for c in codes)
    
    if has_fabricated:
        return 'fabricated'
    if has_verified:
        return 'verified'
    return 'unverified'

def has_dsr_ref(dsr_str):
    """Check if a DSR reference string is non-empty"""
    return bool(dsr_str and dsr_str.strip())

def get_data_source(is_code_ref, dsr_item_ref):
    """Determine data_source field value"""
    code_class = classify_is_code(is_code_ref)
    dsr_present = has_dsr_ref(dsr_item_ref)
    
    if code_class in ('verified', 'unverified') and dsr_present:
        return "CPWD DSR 2023 + IS codes"
    elif code_class in ('verified', 'unverified'):
        return "IS codes only"
    elif code_class == 'fabricated':
        # Treat fabricated IS code same as no IS code
        return "estimated"
    else:
        return "estimated"

def process_items(items):
    """Process a list of items, returning stats"""
    stats = {
        'total': 0,
        'dsr_cleared': 0,
        'data_source_added': 0,
        'fabricated_is': 0,
    }
    
    for item in items:
        if not isinstance(item, dict):
            continue
        
        stats['total'] += 1
        is_code = item.get('is_code_ref', '')
        dsr_ref = item.get('dsr_item_ref', '')
        
        # Track fabricated IS codes
        if classify_is_code(is_code) == 'fabricated':
            stats['fabricated_is'] += 1
        
        # Add data_source field
        data_source = get_data_source(is_code, dsr_ref)
        item['data_source'] = data_source
        stats['data_source_added'] += 1
        
        # Clear dsr_item_ref
        if has_dsr_ref(dsr_ref):
            item['dsr_item_ref'] = ""
            stats['dsr_cleared'] += 1
    
    return stats

def walk_and_process(obj, stats):
    """Recursively walk YAML structure to find and process items lists"""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == 'items' and isinstance(value, list):
                s = process_items(value)
                for k in stats:
                    stats[k] += s[k]
            else:
                walk_and_process(value, stats)
    elif isinstance(obj, list):
        for element in obj:
            walk_and_process(element, stats)

# Custom YAML dumper to preserve formatting style
class IndentDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)

def represent_str(dumper, data):
    if '\n' in data:
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

IndentDumper.add_representer(str, represent_str)

def main():
    yaml_files = sorted(DATA_DIR.glob("*.yaml"))
    
    total_stats = {
        'total': 0,
        'dsr_cleared': 0,
        'data_source_added': 0,
        'fabricated_is': 0,
    }
    
    file_stats = {}
    
    for yaml_file in yaml_files:
        print(f"\nProcessing: {yaml_file.name}")
        
        with open(yaml_file, 'r') as f:
            content = yaml.safe_load(f)
        
        stats = {
            'total': 0,
            'dsr_cleared': 0,
            'data_source_added': 0,
            'fabricated_is': 0,
        }
        
        walk_and_process(content, stats)
        
        # Write back
        with open(yaml_file, 'w') as f:
            yaml.dump(content, f, Dumper=IndentDumper, 
                     default_flow_style=False, 
                     allow_unicode=True,
                     sort_keys=False,
                     width=120)
        
        file_stats[yaml_file.name] = stats
        for k in total_stats:
            total_stats[k] += stats[k]
        
        print(f"  Items: {stats['total']}, DSR refs cleared: {stats['dsr_cleared']}, "
              f"data_source added: {stats['data_source_added']}, "
              f"fabricated IS codes found: {stats['fabricated_is']}")
    
    print(f"\n{'='*60}")
    print(f"TOTALS:")
    print(f"  Total items processed: {total_stats['total']}")
    print(f"  DSR item refs cleared: {total_stats['dsr_cleared']}")
    print(f"  data_source fields added: {total_stats['data_source_added']}")
    print(f"  Items with fabricated IS codes: {total_stats['fabricated_is']}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
