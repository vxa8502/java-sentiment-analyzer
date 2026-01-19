#!/bin/bash
#
# promote_to_production.sh - Promote best model to production
#
# Reads best_generalizing_model from cross_domain_matrix.json and:
#   - If SVM: Hyperparameter tune -> Retrain -> Feature importance
#   - If non-SVM: Copy model -> Feature importance (perturbation)
#
# Usage:
#   ./scripts/promote_to_production.sh
#   ./scripts/promote_to_production.sh --skip-feature-importance
#   ./scripts/promote_to_production.sh --force-model svm-amazon_polarity
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/models"
RESULTS_DIR="$PROJECT_ROOT/results"
PRODUCTION_DIR="$MODELS_DIR/production"
CROSS_DOMAIN_JSON="$RESULTS_DIR/cross_domain_matrix.json"

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

FORCE_MODEL=""

# Parse arguments
for arg in "$@"; do
    case $arg in
        --force-model=*)
            FORCE_MODEL="${arg#*=}"
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force-model=<model>  Override winner (e.g., svm-amazon_polarity)"
            echo ""
            echo "Examples:"
            echo "  $0                                # Auto-detect winner, full pipeline"
            echo "  $0 --force-model=svm-imdb_50k    # Force specific model"
            echo ""
            exit 0
            ;;
    esac
done

# Check dependencies
check_dependencies() {
    if ! command -v jq &> /dev/null; then
        log_error "jq is required. Install with: brew install jq"
        exit 1
    fi

    if [ ! -f "$CROSS_DOMAIN_JSON" ]; then
        log_error "Missing: $CROSS_DOMAIN_JSON"
        log_error "Run ./scripts/evaluate_cross_domain.sh first"
        exit 1
    fi
}

# Get winning model from cross_domain_matrix.json
get_winner() {
    if [ -n "$FORCE_MODEL" ]; then
        echo "$FORCE_MODEL"
    else
        jq -r '.best_generalizing_model.model' "$CROSS_DOMAIN_JSON"
    fi
}

# Parse model string into algorithm and dataset
# e.g., "svm-amazon_polarity" -> algo=svm, dataset=amazon_polarity
parse_model() {
    local model="$1"
    ALGO=$(echo "$model" | cut -d'-' -f1)
    DATASET=$(echo "$model" | cut -d'-' -f2-)

    # Map algorithm names
    case $ALGO in
        svm) ALGO_DIR="svm"; ALGO_ENUM="SVM" ;;
        naive_bayes) ALGO_DIR="naive_bayes"; ALGO_ENUM="NAIVE_BAYES" ;;
        logistic_regression) ALGO_DIR="logistic_regression"; ALGO_ENUM="LOGISTIC_REGRESSION" ;;
        random_forest) ALGO_DIR="random_forest"; ALGO_ENUM="RANDOM_FOREST" ;;
        *)
            log_error "Unknown algorithm: $ALGO"
            exit 1
            ;;
    esac
}

# Get source model path
get_source_model_path() {
    local model_file="${DATASET}_${ALGO_DIR}_model.ser"
    echo "$MODELS_DIR/$ALGO_DIR/$model_file"
}

# Get source metadata path
get_source_metadata_path() {
    local meta_file="${DATASET}_${ALGO_DIR}_model.metadata.json"
    echo "$MODELS_DIR/$ALGO_DIR/$meta_file"
}

# Tune and retrain SVM
tune_and_retrain_svm() {
    log_step "Hyperparameter tuning SVM (grid search)..."

    # Determine data path based on dataset (must be file path, not directory)
    local data_path
    case $DATASET in
        imdb_50k) data_path="data/raw/imdb_50k/IMDB Dataset.csv" ;;
        amazon_polarity) data_path="data/raw/amazon_polarity/train.csv" ;;
        yelp) data_path="data/raw/yelp/yelp_reviews.csv" ;;
        *)
            log_error "Unknown dataset: $DATASET"
            exit 1
            ;;
    esac

    log_info "Dataset: $DATASET"
    log_info "Data path: $data_path"
    log_info "This will take 5-10x longer than base training..."
    echo ""

    cd "$PROJECT_ROOT"

    # Run training with hyperparameter tuning enabled
    # Args: dataPath outputPath algorithm [maxSamples] [showFeatureImportance] [topFeaturesCount] [enableHyperparameterTuning]
    # IMPORTANT: Use same 50K cap as base models for fair comparison
    local output_file="$PRODUCTION_DIR/sentiment_model.ser"
    mvn -q exec:java -Dexec.mainClass="sentiment.training.TrainModel" \
        -Dexec.args="\"$data_path\" \"$output_file\" SVM 50000 true 100 true"

    log_info "SVM tuned and retrained with feature importance"
}

# Retrain non-SVM model with feature importance
retrain_with_feature_importance() {
    log_step "Retraining $ALGO_ENUM with feature importance..."
    log_warn "This retrains the model to compute feature importance via perturbation"
    log_warn "Expected time: 5-10 minutes depending on algorithm"
    echo ""

    # Determine data path based on dataset (must be file path, not directory)
    local data_path
    case $DATASET in
        imdb_50k) data_path="data/raw/imdb_50k/IMDB Dataset.csv" ;;
        amazon_polarity) data_path="data/raw/amazon_polarity/train.csv" ;;
        yelp) data_path="data/raw/yelp/yelp_reviews.csv" ;;
        *)
            log_error "Unknown dataset: $DATASET"
            exit 1
            ;;
    esac

    log_info "Algorithm: $ALGO_ENUM"
    log_info "Dataset: $DATASET"
    log_info "Data path: $data_path"
    echo ""

    cd "$PROJECT_ROOT"

    # Run training with feature importance enabled (no hyperparameter tuning for non-SVM)
    # Args: dataPath outputPath algorithm [maxSamples] [showFeatureImportance] [topFeaturesCount] [enableHyperparameterTuning]
    # IMPORTANT: Use same 50K cap as base models for fair comparison
    local output_file="$PRODUCTION_DIR/sentiment_model.ser"
    mvn -q exec:java -Dexec.mainClass="sentiment.training.TrainModel" \
        -Dexec.args="\"$data_path\" \"$output_file\" $ALGO_ENUM 50000 true 100 false"

    log_info "$ALGO_ENUM retrained with feature importance"
}

# Print summary
print_summary() {
    local prod_model="$PRODUCTION_DIR/sentiment_model.ser"
    local prod_meta="$PRODUCTION_DIR/sentiment_model.metadata.json"
    local prod_fi="$PRODUCTION_DIR/sentiment_model-feature-importance.json"

    echo ""
    echo "┌─────────────────────────────────────────────────────────┐"
    echo "│               PRODUCTION MODEL READY                    │"
    echo "├─────────────────────────────────────────────────────────┤"

    if [ -f "$prod_model" ]; then
        local size=$(ls -lh "$prod_model" | awk '{print $5}')
        echo "│ Model: sentiment_model.ser ($size)                      │"
    fi

    if [ -f "$prod_meta" ]; then
        local acc=$(jq -r '.performance.test_accuracy' "$prod_meta" 2>/dev/null | awk '{printf "%.1f%%", $1*100}')
        echo "│ Accuracy: $acc                                          │"
    fi

    echo "│ Algorithm: $ALGO_ENUM                                   │"
    echo "│ Dataset: $DATASET                                       │"

    if [ -f "$prod_fi" ]; then
        local fi_status=$(jq -r '.status // "complete"' "$prod_fi")
        if [ "$fi_status" = "pending" ]; then
            echo "│ Feature Importance: PENDING (run API to compute)       │"
        else
            local fi_count=$(jq -r '.top_features | length' "$prod_fi" 2>/dev/null || echo "?")
            echo "│ Feature Importance: $fi_count features                  │"
        fi
    fi

    echo "├─────────────────────────────────────────────────────────┤"
    echo "│ Files:                                                  │"
    echo "│   models/production/sentiment_model.ser                 │"
    echo "│   models/production/sentiment_model.metadata.json       │"
    echo "│   models/production/sentiment_model-feature-importance.json │"
    echo "└─────────────────────────────────────────────────────────┘"
    echo ""
}

# Main
main() {
    echo ""
    echo "╔═════════════════════════════════════════════════════════╗"
    echo "║          PROMOTE TO PRODUCTION                          ║"
    echo "╚═════════════════════════════════════════════════════════╝"
    echo ""

    check_dependencies

    # Get winner
    local winner=$(get_winner)
    local winner_acc=$(jq -r '.best_generalizing_model.cross_domain_avg_accuracy' "$CROSS_DOMAIN_JSON" | awk '{printf "%.1f%%", $1*100}')

    if [ -n "$FORCE_MODEL" ]; then
        log_warn "Forcing model: $FORCE_MODEL (ignoring cross-domain winner)"
    else
        log_info "Best generalizing model: $winner ($winner_acc cross-domain avg)"
    fi

    parse_model "$winner"

    log_info "Algorithm: $ALGO_ENUM"
    log_info "Dataset: $DATASET"
    echo ""

    # Ensure production directory exists
    mkdir -p "$PRODUCTION_DIR"

    # Branch based on algorithm
    if [ "$ALGO" = "svm" ]; then
        log_info "SVM detected - will tune hyperparameters and retrain with feature importance"
        echo ""
        read -p "This will take 5-10x longer than base training. Continue? [y/N] " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Aborted. Use --force-model to pick a different model."
            exit 0
        fi

        tune_and_retrain_svm

    else
        log_warn "Non-SVM model detected - no hyperparameter tuning available"
        log_info "Will retrain with feature importance enabled"
        echo ""

        retrain_with_feature_importance
    fi

    print_summary

    log_info "Next step: ./scripts/generate_report.sh"
}

main "$@"
