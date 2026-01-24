#!/bin/bash

# Data Preparation Script (One-Time)
# Creates immutable train/test splits for all domains.
# Splits are locked by manifest files - won't be overwritten on subsequent runs.
#
# Usage:
#   ./scripts/prepare_data.sh           # Prepare all domains (skip existing)
#   ./scripts/prepare_data.sh --force   # Force regenerate all splits
#   ./scripts/prepare_data.sh --verify  # Verify existing splits
#
# This script should be run ONCE before training. Running it again will skip
# domains that already have splits, unless --force is specified.

set -e

# Resolve project root (works regardless of where script is called from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "==============================================================="
echo "  Data Preparation (One-Time Split Creation)"
echo "==============================================================="
echo ""

# Parse arguments
FORCE_FLAG=""
VERIFY_ONLY=false
for arg in "$@"; do
    case $arg in
        --force)
            FORCE_FLAG="--force"
            echo "[WARN] Force mode enabled - existing splits will be regenerated"
            echo ""
            ;;
        --verify)
            VERIFY_ONLY=true
            ;;
    esac
done

# Configuration
DATASETS=("imdb_50k" "amazon_polarity" "yelp")
MAX_SAMPLES=50000
RANDOM_SEED=42

# Function to get raw path for a dataset (bash 3.2 compatible)
get_raw_path() {
    case "$1" in
        imdb_50k)        echo "data/raw/imdb_50k/IMDB Dataset.csv" ;;
        amazon_polarity) echo "data/raw/amazon_polarity/train.csv" ;;
        yelp)            echo "data/raw/yelp/yelp_reviews.csv" ;;
        *)               echo "" ;;
    esac
}

# Build project if needed
if [ ! -f "$PROJECT_ROOT/target/sentiment-analyzer-1.0.0.jar" ]; then
    echo "Building project..."
    mvn clean package -DskipTests -q
    echo "[OK] Build complete"
    echo ""
fi

# Verification mode
if [ "$VERIFY_ONLY" = true ]; then
    echo "Verifying existing splits..."
    echo ""

    all_valid=true
    for dataset in "${DATASETS[@]}"; do
        manifest_path="data/processed/$dataset/splits.manifest.json"
        if [ ! -f "$manifest_path" ]; then
            echo "[MISSING] $dataset: No manifest found"
            all_valid=false
        else
            # Run verification via Java (only needs domain and --verify flag)
            MAVEN_OPTS="-Xmx2g" mvn -q exec:java \
                -Dexec.mainClass="sentiment.data.DataPreparer" \
                -Dexec.args="$dataset --verify" \
                -Dexec.cleanupDaemonThreads=false 2>/dev/null

            if [ $? -eq 0 ]; then
                echo "[OK] $dataset: Checksums match"
            else
                echo "[FAIL] $dataset: Checksum mismatch - data may be corrupted"
                all_valid=false
            fi
        fi
    done

    echo ""
    if [ "$all_valid" = true ]; then
        echo "All splits verified successfully."
        exit 0
    else
        echo "Some splits are missing or corrupted. Run ./scripts/prepare_data.sh --force to regenerate."
        exit 1
    fi
fi

# Counters
prepared=0
skipped=0
failed=0

echo "Configuration:"
echo "  Datasets: ${DATASETS[*]}"
echo "  Max samples per dataset: $MAX_SAMPLES"
echo "  Random seed: $RANDOM_SEED"
echo ""

# Prepare each dataset
for dataset in "${DATASETS[@]}"; do
    echo "---------------------------------------------------------------"
    echo "Dataset: $dataset"
    echo "---------------------------------------------------------------"

    raw_path=$(get_raw_path "$dataset")
    manifest_path="data/processed/$dataset/splits.manifest.json"

    # Check if raw data exists
    if [ ! -f "$PROJECT_ROOT/$raw_path" ]; then
        echo "[FAIL] Raw data not found: $raw_path"
        echo "       See docs/data_cards/${dataset}.yaml for download instructions"
        failed=$((failed + 1))
        continue
    fi

    # Check if splits already exist (unless force mode)
    if [ -f "$manifest_path" ] && [ -z "$FORCE_FLAG" ]; then
        echo "[SKIP] Splits already exist at: data/processed/$dataset/"
        echo "       Created: $(grep -o '"created_at"[^,]*' "$manifest_path" | cut -d'"' -f4)"
        skipped=$((skipped + 1))
        echo ""
        continue
    fi

    # Run DataPreparer
    # Note: Use single quotes for path to handle spaces in filenames with Maven exec plugin
    echo "Preparing splits..."
    MAVEN_OPTS="-Xmx4g" mvn -q exec:java \
        -Dexec.mainClass="sentiment.data.DataPreparer" \
        -Dexec.args="$dataset '$raw_path' $MAX_SAMPLES $RANDOM_SEED $FORCE_FLAG" \
        -Dexec.cleanupDaemonThreads=false

    if [ $? -eq 0 ]; then
        echo "[OK] Prepared: data/processed/$dataset/"
        prepared=$((prepared + 1))
    else
        echo "[FAIL] Failed to prepare $dataset"
        failed=$((failed + 1))
    fi

    echo ""
done

echo "==============================================================="
echo "  Data Preparation Complete"
echo "==============================================================="
echo ""
echo "Summary:"
echo "  Prepared: $prepared"
echo "  Skipped (already exist): $skipped"
echo "  Failed: $failed"
echo ""

if [ $failed -gt 0 ]; then
    echo "[WARN] Some datasets failed to prepare. Check the output above."
    exit 1
fi

# Show manifest locations
echo "Manifests created:"
for dataset in "${DATASETS[@]}"; do
    manifest_path="data/processed/$dataset/splits.manifest.json"
    if [ -f "$manifest_path" ]; then
        echo "  - $manifest_path"
    fi
done
echo ""

echo "Next steps:"
echo "  1. Train models: ./scripts/train_all_models.sh"
echo "  2. Evaluate: ./scripts/evaluate_cross_domain.sh"
echo ""
echo "IMPORTANT: These splits are now LOCKED. Training will use these exact"
echo "           test sets, ensuring reproducibility across evaluations."
