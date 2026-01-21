#!/bin/bash
#
# generate_report.sh - Auto-generate FINAL_COMPREHENSIVE_REPORT.md from model metadata
#
# Reads:
#   - results/cross_domain_matrix.json
#   - models/*/*.metadata.json
#   - models/production/sentiment_model.metadata.json
#
# Outputs:
#   - results/FINAL_COMPREHENSIVE_REPORT.md
#
# Usage: ./scripts/generate_report.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/results"
MODELS_DIR="$PROJECT_ROOT/models"
OUTPUT_FILE="$RESULTS_DIR/FINAL_COMPREHENSIVE_REPORT.md"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check dependencies
check_dependencies() {
    if ! command -v jq &> /dev/null; then
        log_error "jq is required but not installed. Install with: brew install jq"
        exit 1
    fi
}

# Verify required files exist
verify_inputs() {
    local missing=0

    if [ ! -f "$RESULTS_DIR/cross_domain_matrix.json" ]; then
        log_error "Missing: results/cross_domain_matrix.json"
        log_error "Run: ./scripts/evaluate_cross_domain.sh first"
        missing=1
    fi

    if [ ! -f "$MODELS_DIR/production/sentiment_model.metadata.json" ]; then
        log_error "Missing: models/production/sentiment_model.metadata.json"
        log_error "Run hyperparameter tuning to create production model first"
        missing=1
    fi

    # Check for at least one model per algorithm
    for algo in svm naive_bayes logistic_regression random_forest; do
        if ! ls "$MODELS_DIR/$algo"/*.metadata.json &> /dev/null; then
            log_warn "No metadata files found for $algo"
        fi
    done

    if [ $missing -eq 1 ]; then
        exit 1
    fi
}

# Get current date
get_date() {
    date "+%B %d, %Y"
}

# Get production model info
get_production_info() {
    local prod_meta="$MODELS_DIR/production/sentiment_model.metadata.json"

    PROD_ALGO=$(jq -r '.model_info.algorithm // .algorithm' "$prod_meta")
    PROD_ACCURACY=$(jq -r '.performance.test_accuracy' "$prod_meta")
    PROD_F1=$(jq -r '.performance.test_f1' "$prod_meta")
    PROD_PRECISION=$(jq -r '.performance.test_precision' "$prod_meta")
    PROD_RECALL=$(jq -r '.performance.test_recall' "$prod_meta")
    PROD_ROC_AUC=$(jq -r '.performance.roc_auc' "$prod_meta")
    PROD_DATASET=$(jq -r '.training_data.dataset' "$prod_meta")
    PROD_SAMPLES=$(jq -r '.training_data.num_samples' "$prod_meta")
    PROD_SIZE_MB=$(jq -r '.model_size_bytes' "$prod_meta" | awk '{printf "%.1f", $1/1048576}')
    PROD_CROSS_DOMAIN_AVG=$(jq -r '.cross_domain_performance.cross_domain_average // empty' "$prod_meta" 2>/dev/null | awk '{printf "%.1f%%", $1*100}')
    if [ -z "$PROD_CROSS_DOMAIN_AVG" ]; then PROD_CROSS_DOMAIN_AVG="N/A"; fi
}

# Build cross-domain performance tables
build_cross_domain_tables() {
    local matrix="$RESULTS_DIR/cross_domain_matrix.json"

    for algo in svm logistic_regression random_forest naive_bayes; do
        local algo_upper=$(echo "$algo" | tr '[:lower:]' '[:upper:]' | sed 's/_/ /g')

        echo "#### $algo_upper"
        echo ""
        echo "| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |"
        echo "|--------------|-----------|-------------|-----------|------------------|"

        for domain in imdb_50k amazon_polarity yelp; do
            local imdb=$(jq -r ".results.${algo}.${domain}.imdb_50k.accuracy" "$matrix" | awk '{printf "%.1f%%", $1*100}')
            local amazon=$(jq -r ".results.${algo}.${domain}.amazon_polarity.accuracy" "$matrix" | awk '{printf "%.1f%%", $1*100}')
            local yelp=$(jq -r ".results.${algo}.${domain}.yelp.accuracy" "$matrix" | awk '{printf "%.1f%%", $1*100}')
            local avg=$(jq -r ".results.${algo}.${domain}.cross_domain_avg" "$matrix" | awk '{printf "%.1f%%", $1*100}')

            # Mark in-domain with asterisk
            case $domain in
                imdb_50k) imdb="${imdb} *" ;;
                amazon_polarity) amazon="${amazon} *" ;;
                yelp) yelp="${yelp} *" ;;
            esac

            # Map domain to display name (portable across BSD/GNU sed)
            local domain_display
            case $domain in
                imdb_50k) domain_display="imdb 50k" ;;
                amazon_polarity) domain_display="amazon polarity" ;;
                yelp) domain_display="yelp" ;;
                *) domain_display="$domain" ;;
            esac
            echo "| $domain_display | $imdb | $amazon | $yelp | $avg |"
        done
        echo ""
    done
}

# Build model comparison table
build_model_comparison_table() {
    echo "| Algorithm | Dataset | Accuracy | F1 | Precision | Recall | Training Time |"
    echo "|-----------|---------|----------|-----|-----------|--------|---------------|"

    for algo_dir in "$MODELS_DIR"/svm "$MODELS_DIR"/logistic_regression "$MODELS_DIR"/random_forest "$MODELS_DIR"/naive_bayes; do
        if [ -d "$algo_dir" ]; then
            for meta in "$algo_dir"/*.metadata.json; do
                if [ -f "$meta" ]; then
                    local algo=$(jq -r '.model_info.algorithm // .algorithm' "$meta")
                    local dataset=$(jq -r '.training_data.dataset' "$meta")
                    local acc=$(jq -r '.performance.test_accuracy' "$meta" | awk '{printf "%.1f%%", $1*100}')
                    local f1=$(jq -r '.performance.test_f1' "$meta" | awk '{printf "%.3f", $1}')
                    local prec=$(jq -r '.performance.test_precision' "$meta" | awk '{printf "%.3f", $1}')
                    local rec=$(jq -r '.performance.test_recall' "$meta" | awk '{printf "%.3f", $1}')
                    local dur=$(jq -r '.training_duration_seconds' "$meta" | awk '{printf "%dm %ds", $1/60, $1%60}')

                    echo "| $algo | $dataset | $acc | $f1 | $prec | $rec | $dur |"
                fi
            done
        fi
    done
}

# Build confusion matrix display for production model
build_production_confusion_matrix() {
    local prod_meta="$MODELS_DIR/production/sentiment_model.metadata.json"
    local cm=$(jq -r '.performance.confusion_matrix' "$prod_meta")

    local tn=$(echo "$cm" | jq -r '.[0][0]')
    local fp=$(echo "$cm" | jq -r '.[0][1]')
    local fn=$(echo "$cm" | jq -r '.[1][0]')
    local tp=$(echo "$cm" | jq -r '.[1][1]')

    echo "| | Predicted Negative | Predicted Positive |"
    echo "|---|---|---|"
    echo "| **Actual Negative** | $tn (TN) | $fp (FP) |"
    echo "| **Actual Positive** | $fn (FN) | $tp (TP) |"
}

# Get best model from cross-domain results
get_best_model() {
    local matrix="$RESULTS_DIR/cross_domain_matrix.json"
    jq -r '.best_generalizing_model.model' "$matrix"
}

get_best_model_accuracy() {
    local matrix="$RESULTS_DIR/cross_domain_matrix.json"
    jq -r '.best_generalizing_model.cross_domain_avg_accuracy' "$matrix" | awk '{printf "%.1f%%", $1*100}'
}

# Count edge cases
count_edge_cases() {
    local edge_dir="$PROJECT_ROOT/data/raw/edge_cases"
    local total=0
    for f in "$edge_dir"/*.csv; do
        if [ -f "$f" ]; then
            local count=$(($(wc -l < "$f") - 1))  # Subtract header
            total=$((total + count))
        fi
    done
    echo $total
}

# Generate the report
generate_report() {
    log_info "Generating report..."

    get_production_info

    cat > "$OUTPUT_FILE" << 'HEADER'
# Sentiment Analysis: Final Comprehensive Evaluation Report

HEADER

    cat >> "$OUTPUT_FILE" << EOF
**Project**: Cross-Domain Sentiment Classification with Edge Case Analysis
**Date**: $(get_date)
**Author**: Victoria Alabi
**Generated**: Auto-generated from model metadata (do not edit manually)

---

## Executive Summary

This report summarizes the training and evaluation of sentiment analysis models across multiple algorithms and domains.

### Production Model

**Algorithm**: $PROD_ALGO
**Training Dataset**: $PROD_DATASET ($PROD_SAMPLES samples)
**Model Size**: ${PROD_SIZE_MB} MB

| Metric | Value |
|--------|-------|
| Test Accuracy | $(echo "$PROD_ACCURACY" | awk '{printf "%.1f%%", $1*100}') |
| Test F1 | $(echo "$PROD_F1" | awk '{printf "%.3f", $1}') |
| Test Precision | $(echo "$PROD_PRECISION" | awk '{printf "%.3f", $1}') |
| Test Recall | $(echo "$PROD_RECALL" | awk '{printf "%.3f", $1}') |
| ROC-AUC | $(echo "$PROD_ROC_AUC" | awk '{printf "%.3f", $1}') |
| Cross-Domain Avg | $PROD_CROSS_DOMAIN_AVG |

### Production Model Confusion Matrix

$(build_production_confusion_matrix)

### Best Generalizing Model

**Model**: $(get_best_model)
**Cross-Domain Average Accuracy**: $(get_best_model_accuracy)

---

## Part 1: Model Comparison (All 12 Experiments)

$(build_model_comparison_table)

---

## Part 2: Cross-Domain Evaluation

Each model was evaluated on all three test domains. Asterisk (*) indicates in-domain evaluation.

$(build_cross_domain_tables)

**Legend**: * = in-domain evaluation

---

## Part 3: Edge Case Evaluation

**Total Edge Cases**: $(count_edge_cases) curated examples across 4 categories:
- Sarcasm
- Mixed Sentiment
- Negation Heavy
- Domain Jargon

EOF

    # Add edge case results if available in metadata
    if jq -e '.edge_case_performance' "$MODELS_DIR/production/sentiment_model.metadata.json" &>/dev/null; then
        cat >> "$OUTPUT_FILE" << 'EOF'
### Production Model Edge Case Performance

EOF
        local prod_meta="$MODELS_DIR/production/sentiment_model.metadata.json"

        echo "| Category | Accuracy |" >> "$OUTPUT_FILE"
        echo "|----------|----------|" >> "$OUTPUT_FILE"

        for category in sarcasm mixed_sentiment negation_heavy domain_jargon; do
            local acc=$(jq -r ".edge_case_performance.${category}.accuracy // \"N/A\"" "$prod_meta")
            if [ "$acc" != "N/A" ] && [ "$acc" != "null" ]; then
                acc=$(echo "$acc" | awk '{printf "%.1f%%", $1*100}')
            fi
            local cat_display=$(echo "$category" | sed 's/_/ /g' | sed 's/\b\(.\)/\u\1/g')
            echo "| $cat_display | $acc |" >> "$OUTPUT_FILE"
        done
        echo "" >> "$OUTPUT_FILE"
    fi

    cat >> "$OUTPUT_FILE" << 'EOF'
---

## Part 4: Reproducibility

All results can be reproduced via:

```bash
# Phase 1: Prepare immutable data splits (run once)
./scripts/prepare_data.sh

# Phase 2: Train all models
./scripts/train_all_models.sh

# Phase 3: Cross-domain evaluation
./scripts/evaluate_cross_domain.sh

# Regenerate this report
./scripts/generate_report.sh
```

### Model Artifacts

EOF

    echo "| File | Algorithm | Dataset | Size |" >> "$OUTPUT_FILE"
    echo "|------|-----------|---------|------|" >> "$OUTPUT_FILE"

    for meta in "$MODELS_DIR"/*/*.metadata.json; do
        if [ -f "$meta" ]; then
            local filename=$(jq -r '.model_file // .artifacts.model_file' "$meta")
            local algo=$(jq -r '.model_info.algorithm // .algorithm' "$meta")
            local dataset=$(jq -r '.training_data.dataset' "$meta")
            local size=$(jq -r '.model_size_bytes' "$meta" | awk '{printf "%.1f MB", $1/1048576}')
            echo "| $filename | $algo | $dataset | $size |" >> "$OUTPUT_FILE"
        fi
    done

    cat >> "$OUTPUT_FILE" << EOF

---

## Metadata

- **Report Generated**: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
- **Git Commit**: $(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
- **Java Version**: $(java -version 2>&1 | head -1 | awk -F '"' '{print $2}')

EOF

    log_info "Report generated: $OUTPUT_FILE"
}

# Validate output
validate_output() {
    log_info "Validating generated report..."

    # Check file exists and has content
    if [ ! -s "$OUTPUT_FILE" ]; then
        log_error "Generated report is empty!"
        exit 1
    fi

    # Check for placeholder text that shouldn't be there
    if grep -q "N/A" "$OUTPUT_FILE"; then
        log_warn "Report contains N/A values - some data may be missing"
    fi

    # Check confusion matrix is 2x2
    if grep -q "3x3\|neutral" "$OUTPUT_FILE"; then
        log_error "Report contains references to 3-class data - check model metadata"
        exit 1
    fi

    log_info "Validation passed"
}

# Main
main() {
    log_info "Starting report generation..."

    check_dependencies
    verify_inputs
    generate_report
    validate_output

    log_info "Done! View report at: $OUTPUT_FILE"
    echo ""
    echo "Preview:"
    head -50 "$OUTPUT_FILE"
}

main "$@"
