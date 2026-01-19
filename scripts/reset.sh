#!/bin/bash
#
# reset.sh - Clean reset of all model artifacts before retraining
#
# This script:
#   1. Deletes all model files (.ser, .metadata.json, feature-importance.json)
#   2. Verifies clean state
#   3. Optionally verifies data pipeline (--verify flag)
#
# Usage:
#   ./scripts/reset.sh           # Reset only
#   ./scripts/reset.sh --verify  # Reset + verify data pipeline
#   ./scripts/reset.sh --dry-run # Show what would be deleted
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/models"
RESULTS_DIR="$PROJECT_ROOT/results"
EDGE_CASES_DIR="$PROJECT_ROOT/data/raw/edge_cases"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} ${BOLD}$1${NC}"; }

DRY_RUN=false
VERIFY=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            ;;
        --verify)
            VERIFY=true
            ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--verify]"
            echo ""
            echo "Options:"
            echo "  --dry-run  Show what would be deleted without deleting"
            echo "  --verify   After reset, verify data pipeline is ready"
            echo ""
            exit 0
            ;;
    esac
done

# Count files before deletion
count_artifacts() {
    local count=0
    for dir in svm naive_bayes logistic_regression random_forest production; do
        if [ -d "$MODELS_DIR/$dir" ]; then
            count=$((count + $(find "$MODELS_DIR/$dir" -name "*.ser" 2>/dev/null | wc -l)))
            count=$((count + $(find "$MODELS_DIR/$dir" -name "*.json" 2>/dev/null | wc -l)))
        fi
    done
    # Count edge case CSVs
    if [ -d "$EDGE_CASES_DIR" ]; then
        count=$((count + $(find "$EDGE_CASES_DIR" -name "*.csv" 2>/dev/null | wc -l)))
    fi
    # Count cross_domain_matrix.json
    if [ -f "$RESULTS_DIR/cross_domain_matrix.json" ]; then
        count=$((count + 1))
    fi
    echo $count
}

# List all artifacts
list_artifacts() {
    echo ""
    echo "Model files (.ser):"
    find "$MODELS_DIR" -name "*.ser" 2>/dev/null | sed 's|'"$PROJECT_ROOT"'/||' | sort || echo "  (none)"

    echo ""
    echo "Metadata files (.json):"
    find "$MODELS_DIR" -name "*.json" ! -name ".gitignore" 2>/dev/null | sed 's|'"$PROJECT_ROOT"'/||' | sort || echo "  (none)"

    echo ""
    echo "Edge case files (.csv):"
    if [ -d "$EDGE_CASES_DIR" ]; then
        find "$EDGE_CASES_DIR" -name "*.csv" 2>/dev/null | sed 's|'"$PROJECT_ROOT"'/||' | sort || echo "  (none)"
    else
        echo "  (none)"
    fi

    echo ""
    echo "Results files:"
    if [ -f "$RESULTS_DIR/cross_domain_matrix.json" ]; then
        echo "  results/cross_domain_matrix.json"
    fi
    echo ""
}

# Delete all model artifacts
delete_artifacts() {
    log_step "Deleting model artifacts..."

    for dir in svm naive_bayes logistic_regression random_forest production; do
        local dir_path="$MODELS_DIR/$dir"
        if [ -d "$dir_path" ]; then
            # Delete .ser files
            find "$dir_path" -name "*.ser" -type f -delete 2>/dev/null || true

            # Delete .json files (but not .gitignore)
            find "$dir_path" -name "*.json" -type f -delete 2>/dev/null || true

            log_info "Cleaned: models/$dir/"
        fi
    done

    # Delete edge case CSVs (will regenerate from new model errors)
    if [ -d "$EDGE_CASES_DIR" ]; then
        find "$EDGE_CASES_DIR" -name "*.csv" -type f -delete 2>/dev/null || true
        log_info "Deleted: data/raw/edge_cases/*.csv (will regenerate)"
    fi

    # Delete cross_domain_matrix.json but keep FINAL_COMPREHENSIVE_REPORT.md
    # (report will be regenerated, but keep as reference)
    if [ -f "$RESULTS_DIR/cross_domain_matrix.json" ]; then
        rm "$RESULTS_DIR/cross_domain_matrix.json"
        log_info "Deleted: results/cross_domain_matrix.json"
    fi
}

# Verify clean state
verify_clean() {
    log_step "Verifying clean state..."

    local remaining=$(count_artifacts)

    if [ "$remaining" -gt 0 ]; then
        log_error "Clean verification failed! $remaining artifacts remaining:"
        list_artifacts
        exit 1
    fi

    # Verify directory structure exists
    for dir in svm naive_bayes logistic_regression random_forest production; do
        if [ ! -d "$MODELS_DIR/$dir" ]; then
            log_warn "Creating missing directory: models/$dir/"
            mkdir -p "$MODELS_DIR/$dir"
        fi

        # Ensure .gitignore exists
        if [ ! -f "$MODELS_DIR/$dir/.gitignore" ]; then
            echo "*.ser" > "$MODELS_DIR/$dir/.gitignore"
            echo "*.json" >> "$MODELS_DIR/$dir/.gitignore"
            echo "!.gitignore" >> "$MODELS_DIR/$dir/.gitignore"
            log_info "Created .gitignore in models/$dir/"
        fi
    done

    log_info "Clean state verified: 0 artifacts, directories intact"
}

# Verify data pipeline
verify_data_pipeline() {
    log_step "Verifying data pipeline..."

    local datasets=("imdb_50k" "amazon_polarity" "yelp")
    local data_dir="$PROJECT_ROOT/data/raw"
    local all_ok=true

    echo ""
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│                   DATA PIPELINE CHECK                   │"
    echo "├─────────────────────────────────────────────────────────┤"

    # Check each dataset directory exists
    for dataset in "${datasets[@]}"; do
        if [ -d "$data_dir/$dataset" ]; then
            local csv_count=$(find "$data_dir/$dataset" -name "*.csv" | wc -l | tr -d ' ')
            echo "│ $dataset: $csv_count CSV file(s) found"
        else
            echo "│ $dataset: MISSING"
            all_ok=false
        fi
    done

    # Check edge cases (OK if empty after reset - will regenerate)
    local edge_dir="$data_dir/edge_cases"
    if [ -d "$edge_dir" ]; then
        local edge_count=$(find "$edge_dir" -name "*.csv" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$edge_count" -eq 0 ]; then
            echo "│ edge_cases: EMPTY (will regenerate in Phase 4)"
        else
            local total_cases=0
            for f in "$edge_dir"/*.csv; do
                if [ -f "$f" ]; then
                    local lines=$(($(wc -l < "$f") - 1))
                    total_cases=$((total_cases + lines))
                fi
            done
            echo "│ edge_cases: $edge_count files, $total_cases total cases"
        fi
    else
        echo "│ edge_cases: MISSING (mkdir data/raw/edge_cases)"
        all_ok=false
    fi

    echo "├─────────────────────────────────────────────────────────┤"

    # Run preprocessing test
    echo "│ Running preprocessing tests...                          │"
    cd "$PROJECT_ROOT"
    if mvn test -Dtest=WekaInstancesConverterTest -q 2>/dev/null; then
        echo "│ WekaInstancesConverterTest: PASSED                      │"
    else
        echo "│ WekaInstancesConverterTest: FAILED                      │"
        all_ok=false
    fi

    echo "└─────────────────────────────────────────────────────────┘"
    echo ""

    if [ "$all_ok" = true ]; then
        log_info "Data pipeline verification: PASSED"
    else
        log_error "Data pipeline verification: FAILED"
        log_error "Fix issues above before training"
        exit 1
    fi
}

# Print summary
print_summary() {
    echo ""
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│                     RESET COMPLETE                      │"
    echo "├─────────────────────────────────────────────────────────┤"
    echo "│ Model artifacts: DELETED                                │"
    echo "│ Edge case CSVs: DELETED (will regenerate)               │"
    echo "│ cross_domain_matrix.json: DELETED                       │"
    echo "│ Directory structure: INTACT                             │"
    echo "│ FINAL_COMPREHENSIVE_REPORT.md: KEPT (will regenerate)   │"
    echo "├─────────────────────────────────────────────────────────┤"
    echo "│                     NEXT STEPS                          │"
    echo "├─────────────────────────────────────────────────────────┤"
    echo "│ 1. ./scripts/train_all_models.sh       (Phase 2)        │"
    echo "│ 2. ./scripts/evaluate_cross_domain.sh  (Phase 3)        │"
    echo "│ 3. ./scripts/generate_edge_cases.sh    (Phase 4a)       │"
    echo "│ 4. Manual: categorize edge cases       (Phase 4b)       │"
    echo "│ 5. ./scripts/evaluate_edge_cases.sh    (Phase 4c)       │"
    echo "│ 6. ./scripts/promote_to_production.sh  (Phase 5)        │"
    echo "│ 7. ./scripts/generate_report.sh        (Phase 6)        │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo ""
}

# Main
main() {
    echo ""
    echo "╔═════════════════════════════════════════════════════════╗"
    echo "║           SENTIMENT ANALYZER - RESET SCRIPT             ║"
    echo "╚═════════════════════════════════════════════════════════╝"
    echo ""

    local artifact_count=$(count_artifacts)
    log_info "Found $artifact_count artifacts to delete"

    if [ "$DRY_RUN" = true ]; then
        log_warn "DRY RUN - No files will be deleted"
        list_artifacts
        exit 0
    fi

    if [ "$artifact_count" -eq 0 ]; then
        log_info "Already clean - no artifacts to delete"
    else
        # Confirm deletion
        echo ""
        list_artifacts
        echo -e "${YELLOW}This will delete all $artifact_count artifacts listed above.${NC}"
        read -p "Continue? [y/N] " -n 1 -r
        echo ""

        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Aborted"
            exit 0
        fi

        delete_artifacts
    fi

    verify_clean

    if [ "$VERIFY" = true ]; then
        verify_data_pipeline
    fi

    print_summary
}

main "$@"
