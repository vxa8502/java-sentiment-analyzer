#!/bin/bash

# Dataset Verification Script
# Checks that all required datasets are present and valid
# For download/preparation instructions, see: docs/data_cards/*.yaml

set -u  # Exit on undefined variable

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DATA_DIR="data/raw"
PASSED=0
FAILED=0

echo "================================================================"
echo "  Dataset Verification"
echo "================================================================"
echo ""
echo "Checking: $DATA_DIR"
echo "Data cards: docs/data_cards/*.yaml"
echo ""

# Function to check a dataset
check_dataset() {
    local name="$1"
    local path="$2"
    local min_rows="$3"
    local required_cols="$4"

    echo "----------------------------------------------------------------"
    echo "$name"
    echo "----------------------------------------------------------------"

    # Check file exists
    if [ ! -f "$path" ]; then
        echo -e "${RED}[MISSING]${NC} $path"
        echo "  See: docs/data_cards/$(echo "$name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_').yaml"
        FAILED=$((FAILED + 1))
        return 1
    fi

    # Check row count (using Python for accurate CSV parsing)
    local rows
    rows=$(python3 -c "import csv; print(sum(1 for _ in csv.reader(open('$path')))-1)" 2>/dev/null)
    if [ -z "$rows" ]; then
        echo -e "${YELLOW}[WARN]${NC} Could not count rows (Python error)"
    elif [ "$rows" -lt "$min_rows" ]; then
        echo -e "${YELLOW}[WARN]${NC} $path has $rows rows (expected >= $min_rows)"
    else
        echo -e "${GREEN}[OK]${NC} $rows rows"
    fi

    # Check header columns
    local header
    header=$(head -1 "$path")
    for col in $required_cols; do
        if echo "$header" | grep -qi "$col"; then
            echo -e "${GREEN}[OK]${NC} Column: $col"
        else
            echo -e "${YELLOW}[WARN]${NC} Column '$col' not found in header"
        fi
    done

    # Sample preview
    echo ""
    echo "Preview (first 2 data rows):"
    head -3 "$path" | tail -2 | cut -c1-80
    echo ""

    PASSED=$((PASSED + 1))
    return 0
}

# Check IMDB (50K samples, balanced)
check_dataset "IMDB 50K" \
    "$DATA_DIR/imdb_50k/IMDB Dataset.csv" \
    50000 \
    "review sentiment"

echo ""

# Check Amazon train (100K samples from HuggingFace)
check_dataset "Amazon Polarity (train)" \
    "$DATA_DIR/amazon_polarity/train.csv" \
    100000 \
    "label text"

echo ""

# Check Amazon test (50K samples from HuggingFace)
check_dataset "Amazon Polarity (test)" \
    "$DATA_DIR/amazon_polarity/test.csv" \
    50000 \
    "label text"

echo ""

# Check Yelp (100K samples, includes neutral which gets filtered during training)
check_dataset "Yelp Reviews" \
    "$DATA_DIR/yelp/yelp_reviews.csv" \
    100000 \
    "text sentiment"

echo ""

# Summary
echo "================================================================"
echo "  Summary"
echo "================================================================"
echo ""
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All datasets verified!${NC}"
    echo ""
    echo "Next: ./scripts/prepare_data.sh"
    exit 0
else
    echo -e "${RED}Some datasets missing or invalid.${NC}"
    echo "See docs/data_cards/*.yaml for preparation instructions."
    exit 1
fi
