#!/bin/bash
# XBOQ Demo Runner
# Runs full pipeline on input drawings and generates demo PDF

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}   XBOQ DEMO PIPELINE RUNNER${NC}"
echo -e "${BLUE}======================================${NC}"

# Default values
INPUT_DIR="${1:-data/projects/real_test/drawings}"
PROJECT_ID="${2:-demo_$(date +%Y%m%d_%H%M%S)}"
SCALE="${3:-100}"

echo ""
echo "Input Directory: $INPUT_DIR"
echo "Project ID: $PROJECT_ID"
echo "Scale: 1:$SCALE"
echo ""

# Check if input exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    echo ""
    echo "Usage: ./scripts/run_demo.sh [input_dir] [project_id] [scale]"
    echo "Example: ./scripts/run_demo.sh ./my_drawings my_project 100"
    exit 1
fi

# Run full pipeline with MEP enabled
echo -e "${BLUE}Step 1: Running XBOQ Pipeline...${NC}"
python3 run_full_project.py \
    --project_id "$PROJECT_ID" \
    --input_dir "$INPUT_DIR" \
    --enable-mep \
    --scale "$SCALE" \
    --mode full

echo ""
echo -e "${BLUE}Step 2: Generating Demo PDF...${NC}"
python3 scripts/generate_demo_pdf.py --output_dir "out/$PROJECT_ID"

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}   DEMO COMPLETE!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Output directory: out/$PROJECT_ID"
echo ""
echo "Key files:"
echo "  - XBOQ_Demo_Report.pdf    (comprehensive demo PDF)"
echo "  - summary.md              (run summary)"
echo "  - boq/boq_measured.csv    (measured quantities)"
echo "  - boq/boq_inferred.csv    (inferred quantities)"
echo "  - mep/mep_takeoff.csv     (MEP device takeoff)"
echo "  - rfi/rfi_log.md          (RFI list)"
echo "  - bid_gate_report.md      (bid gate assessment)"
echo ""

# Try to open PDF
if command -v open &> /dev/null; then
    open "out/$PROJECT_ID/XBOQ_Demo_Report.pdf"
elif command -v xdg-open &> /dev/null; then
    xdg-open "out/$PROJECT_ID/XBOQ_Demo_Report.pdf"
fi
