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

# Get production model info with error handling
get_production_info() {
    local prod_meta="$MODELS_DIR/production/sentiment_model.metadata.json"

    # Verify file exists and is valid JSON
    if [ ! -f "$prod_meta" ]; then
        log_error "Production metadata not found: $prod_meta"
        exit 1
    fi

    if ! jq empty "$prod_meta" 2>/dev/null; then
        log_error "Production metadata is not valid JSON: $prod_meta"
        exit 1
    fi

    # Extract values with fallbacks for missing fields
    PROD_ALGO=$(jq -r '.model_info.algorithm // .algorithm // "unknown"' "$prod_meta")
    PROD_ACCURACY=$(jq -r '.performance.test_accuracy // 0' "$prod_meta")
    PROD_F1=$(jq -r '.performance.test_f1 // 0' "$prod_meta")
    PROD_PRECISION=$(jq -r '.performance.test_precision // 0' "$prod_meta")
    PROD_RECALL=$(jq -r '.performance.test_recall // 0' "$prod_meta")
    PROD_ROC_AUC=$(jq -r '.performance.roc_auc // 0' "$prod_meta")
    PROD_DATASET=$(jq -r '.training_data.dataset // "unknown"' "$prod_meta")
    PROD_SAMPLES=$(jq -r '.training_data.num_samples // 0' "$prod_meta")

    local size_bytes=$(jq -r '.model_size_bytes // 0' "$prod_meta")
    if [ "$size_bytes" = "null" ] || [ -z "$size_bytes" ]; then
        PROD_SIZE_MB="N/A"
    else
        PROD_SIZE_MB=$(echo "$size_bytes" | awk '{printf "%.1f", $1/1048576}')
    fi

    local cross_avg=$(jq -r '.cross_domain_performance.cross_domain_average // empty' "$prod_meta" 2>/dev/null)
    if [ -z "$cross_avg" ] || [ "$cross_avg" = "null" ]; then
        PROD_CROSS_DOMAIN_AVG="N/A"
    else
        PROD_CROSS_DOMAIN_AVG=$(echo "$cross_avg" | awk '{printf "%.1f%%", $1*100}')
    fi

    # Warn if critical values are missing/zero
    if [ "$PROD_ACCURACY" = "0" ] || [ "$PROD_ACCURACY" = "null" ]; then
        log_warn "Production model accuracy is 0 or missing - metadata may be incomplete"
    fi
}

# Helper function to format accuracy, handling null values properly
format_accuracy() {
    local value="$1"
    if [ "$value" = "null" ] || [ -z "$value" ]; then
        echo "N/A"
    else
        echo "$value" | awk '{printf "%.1f%%", $1*100}'
    fi
}

# Build cross-domain performance tables
build_cross_domain_tables() {
    local matrix="$RESULTS_DIR/cross_domain_matrix.json"

    # Verify matrix file exists and is valid JSON
    if [ ! -f "$matrix" ]; then
        echo "**ERROR: Cross-domain matrix not found**"
        echo ""
        return 1
    fi

    if ! jq empty "$matrix" 2>/dev/null; then
        echo "**ERROR: Cross-domain matrix is not valid JSON**"
        echo ""
        return 1
    fi

    for algo in svm logistic_regression random_forest naive_bayes; do
        local algo_upper=$(echo "$algo" | tr '[:lower:]' '[:upper:]' | sed 's/_/ /g')

        echo "#### $algo_upper"
        echo ""
        echo "| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |"
        echo "|--------------|-----------|-------------|-----------|------------------|"

        for domain in imdb_50k amazon_polarity yelp; do
            # CRITICAL: Check for null values before formatting
            # jq returns "null" (string) when key doesn't exist
            local imdb_raw=$(jq -r ".results.${algo}.${domain}.imdb_50k.accuracy // empty" "$matrix")
            local amazon_raw=$(jq -r ".results.${algo}.${domain}.amazon_polarity.accuracy // empty" "$matrix")
            local yelp_raw=$(jq -r ".results.${algo}.${domain}.yelp.accuracy // empty" "$matrix")
            local avg_raw=$(jq -r ".results.${algo}.${domain}.cross_domain_avg // empty" "$matrix")

            local imdb=$(format_accuracy "$imdb_raw")
            local amazon=$(format_accuracy "$amazon_raw")
            local yelp_val=$(format_accuracy "$yelp_raw")
            local avg=$(format_accuracy "$avg_raw")

            # Mark in-domain with asterisk (only if not N/A)
            case $domain in
                imdb_50k) [ "$imdb" != "N/A" ] && imdb="${imdb} *" ;;
                amazon_polarity) [ "$amazon" != "N/A" ] && amazon="${amazon} *" ;;
                yelp) [ "$yelp_val" != "N/A" ] && yelp_val="${yelp_val} *" ;;
            esac

            # Map domain to display name (portable across BSD/GNU sed)
            local domain_display
            case $domain in
                imdb_50k) domain_display="imdb 50k" ;;
                amazon_polarity) domain_display="amazon polarity" ;;
                yelp) domain_display="yelp" ;;
                *) domain_display="$domain" ;;
            esac
            echo "| $domain_display | $imdb | $amazon | $yelp_val | $avg |"
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

# Generate the report
generate_report() {
    log_info "Generating report..."

    get_production_info

    cat > "$OUTPUT_FILE" << 'HEADER'
# Sentiment Analysis: Final Comprehensive Evaluation Report

HEADER

    cat >> "$OUTPUT_FILE" << EOF
**Project**: Cross-Domain Sentiment Classification
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

EOF

    cat >> "$OUTPUT_FILE" << 'EOF'
---

## Part 3: Reproducibility

All results can be reproduced via:

```bash
# Prepare immutable data splits (run once)
./scripts/prepare_data.sh

# Train all models
./scripts/train_all_models.sh

# Cross-domain evaluation
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

    local warnings=0
    local errors=0

    # Check file exists and has content
    if [ ! -s "$OUTPUT_FILE" ]; then
        log_error "Generated report is empty!"
        exit 1
    fi

    # Check for N/A values that indicate missing data
    local na_count=$(grep -c "N/A" "$OUTPUT_FILE" 2>/dev/null || echo "0")
    if [ "$na_count" -gt 0 ]; then
        log_warn "Report contains $na_count N/A values - some data may be missing"
        warnings=$((warnings + 1))
    fi

    # CRITICAL: Check for suspicious 0.0% values in cross-domain tables
    # Real accuracy should never be exactly 0.0% - this indicates a data problem
    local zero_count=$(grep -E "^\|.*\| 0\.0%" "$OUTPUT_FILE" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$zero_count" -gt 0 ]; then
        log_error "Report contains $zero_count suspicious 0.0% accuracy values!"
        log_error "This usually indicates missing or corrupt evaluation data."
        log_error "Re-run: ./scripts/evaluate_cross_domain.sh"
        errors=$((errors + 1))
    fi

    # Check confusion matrix is 2x2
    if grep -q "3x3\|neutral" "$OUTPUT_FILE"; then
        log_error "Report contains references to 3-class data - check model metadata"
        errors=$((errors + 1))
    fi

    # Check that all expected sections exist
    if ! grep -q "## Executive Summary" "$OUTPUT_FILE"; then
        log_error "Missing Executive Summary section"
        errors=$((errors + 1))
    fi

    if ! grep -q "## Part 1: Model Comparison" "$OUTPUT_FILE"; then
        log_error "Missing Model Comparison section"
        errors=$((errors + 1))
    fi

    if ! grep -q "## Part 2: Cross-Domain Evaluation" "$OUTPUT_FILE"; then
        log_error "Missing Cross-Domain Evaluation section"
        errors=$((errors + 1))
    fi

    if [ "$errors" -gt 0 ]; then
        log_error "Validation FAILED with $errors error(s)"
        exit 1
    fi

    if [ "$warnings" -gt 0 ]; then
        log_warn "Validation passed with $warnings warning(s)"
    else
        log_info "Validation passed"
    fi
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
