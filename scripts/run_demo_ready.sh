#!/bin/bash
# XBOQ Demo Readiness — prewarm cache + validate + start Streamlit
#
# Usage: ./scripts/run_demo_ready.sh
#
# Sets XBOQ_DEMO_MODE=true and starts the app with all demo assets warm.

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   XBOQ DEMO READINESS CHECK${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Prewarm demo cache
echo -e "${BLUE}Step 1: Prewarming demo cache...${NC}"
python3 scripts/prewarm_demo_cache.py || true

# Step 2: Quick smoke test
echo ""
echo -e "${BLUE}Step 2: Running quick test check...${NC}"
python3 -m pytest tests/test_tender_pipeline.py -x -q --tb=short 2>&1 | tail -5

# Step 3: Validate demo assets
echo ""
echo -e "${BLUE}Step 3: Validating demo assets...${NC}"
python3 -c "
import sys
sys.path.insert(0, 'src')
from src.demo.demo_assets import validate_demo_assets
report = validate_demo_assets()
for r in report:
    status = 'READY' if r['cache_found'] else 'MISSING'
    print(f'  [{status}] {r[\"project_id\"]}: {r[\"name\"]}')
"

# Step 4: Start Streamlit in demo mode
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   STARTING DEMO MODE${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

export XBOQ_DEMO_MODE=true
exec streamlit run app/demo_page.py --server.port 8501
