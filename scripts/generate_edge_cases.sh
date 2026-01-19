#!/bin/bash
# Generate edge case candidates by extracting model prediction errors
# Usage: ./scripts/generate_edge_cases.sh [algorithm] [domain]
# Or: ./scripts/generate_edge_cases.sh all  # Extract from all models

set -e

# Resolve project root (works regardless of where script is called from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

OUTPUT_DIR="$PROJECT_ROOT/data/raw/edge_cases/candidates"

print_usage() {
    echo "Usage: $0 <algorithm> <domain>"
    echo "   Or: $0 all  # Extract errors from all trained models"
    echo ""
    echo "This script runs ErrorAnalyzer --export on trained models to extract"
    echo "prediction failures for manual categorization into edge case sets."
    echo ""
    echo "Examples:"
    echo "  $0 svm amazon_polarity"
    echo "  $0 all"
    echo ""
    echo "Output: $OUTPUT_DIR/<algorithm>_<domain>_errors.csv"
}

if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Build classpath
echo "Building classpath..."
mvn -q dependency:build-classpath -Dmdep.outputFile=.classpath
CLASSPATH=$(cat .classpath):target/classes

run_error_analysis() {
    local algo=$1
    local domain=$2
    local model_file="$PROJECT_ROOT/models/${algo}/${domain}_${algo}_model.ser"
    local test_file="$PROJECT_ROOT/data/processed/${domain}/test.csv"
    local output_file="${OUTPUT_DIR}/${algo}_${domain}_errors.csv"

    # Validate model exists
    if [ ! -f "$model_file" ]; then
        echo "[ERROR] Model not found: $model_file"
        return 1
    fi

    # Validate test file exists
    if [ ! -f "$test_file" ]; then
        echo "[ERROR] Test file not found: $test_file"
        echo "        Run train_all_models.sh first to generate test splits."
        return 1
    fi

    echo "---------------------------------------------------------------"
    echo "Extracting errors: ${algo} trained on ${domain}"
    echo "---------------------------------------------------------------"

    # Run ErrorAnalyzer and capture the CSV output
    java -cp "$CLASSPATH" sentiment.evaluation.ErrorAnalyzer "$algo" "$domain" "$test_file" --export --top-n 100

    # Move the generated file to output directory
    if [ -f "error_analysis_${algo}_${domain}.csv" ]; then
        mv "error_analysis_${algo}_${domain}.csv" "$output_file"
        echo "Saved: $output_file"
        echo ""
    else
        echo "[WARN] No error file generated for ${algo}/${domain}"
        return 1
    fi
}

if [ "$1" = "all" ]; then
    echo "Extracting prediction errors from all trained models..."
    echo "Output directory: $OUTPUT_DIR"
    echo ""

    ALGORITHMS=("svm" "naive_bayes" "random_forest" "logistic_regression")
    DOMAINS=("imdb_50k" "amazon_polarity" "yelp")

    success_count=0
    fail_count=0

    for algo in "${ALGORITHMS[@]}"; do
        for domain in "${DOMAINS[@]}"; do
            if run_error_analysis "$algo" "$domain"; then
                success_count=$((success_count + 1))
            else
                fail_count=$((fail_count + 1))
            fi
        done
    done

    echo "==============================================================="
    echo "Generation complete: $success_count succeeded, $fail_count failed"
    echo ""

    if [ $success_count -eq 0 ]; then
        echo "[FAIL] No error files generated. Check that models are trained"
        echo "       and test splits exist in data/processed/*/test.csv"
        exit 1
    fi

    echo "Next steps:"
    echo "1. Review CSVs in: $OUTPUT_DIR"
    echo "2. Identify high-confidence errors (confidence > 0.7)"
    echo "3. Categorize into: sarcasm, mixed_sentiment, negation_heavy, domain_jargon"
    echo "4. Add categorized examples to data/raw/edge_cases/<category>.csv"
    echo "5. Run ./scripts/evaluate_edge_cases.sh all to measure improvement"
    echo "==============================================================="
else
    if [ $# -lt 2 ]; then
        print_usage
        exit 1
    fi

    ALGORITHM=$1
    DOMAIN=$2

    if ! run_error_analysis "$ALGORITHM" "$DOMAIN"; then
        exit 1
    fi
fi
