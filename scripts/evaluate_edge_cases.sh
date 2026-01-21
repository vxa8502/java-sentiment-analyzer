#!/bin/bash
# Evaluate models on edge case challenge sets
# Usage: ./scripts/evaluate_edge_cases.sh <algorithm> <domain>
# Or: ./scripts/evaluate_edge_cases.sh all  # Test all models
#
# Results are always persisted to model metadata files.
# Config: Reads algorithms/domains from config/edge-case-evaluation.json

set -e

# Resolve project root (works regardless of where script is called from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <algorithm> <domain>"
    echo "   Or: $0 all  # Evaluate all trained models"
    echo ""
    echo "Results are always persisted to model metadata files."
    echo "Config: Reads from config/edge-case-evaluation.json"
    echo ""
    echo "Examples:"
    echo "  $0 svm imdb_50k"
    echo "  $0 all"
    exit 1
fi

# Build classpath
mvn -q dependency:build-classpath -Dmdep.outputFile=.classpath
CLASSPATH=$(cat .classpath):target/classes

# Try to read algorithms/domains from config, fallback to defaults
CONFIG_FILE="$PROJECT_ROOT/config/edge-case-evaluation.json"
if [ -f "$CONFIG_FILE" ] && command -v jq &> /dev/null; then
    ALGORITHMS=($(jq -r '.algorithms[]' "$CONFIG_FILE"))
    DOMAINS=($(jq -r '.domains[]' "$CONFIG_FILE"))
    echo "Config: Loaded from $CONFIG_FILE"
else
    ALGORITHMS=("svm" "naive_bayes" "random_forest" "logistic_regression")
    DOMAINS=("imdb_50k" "amazon_polarity" "yelp")
    echo "Config: Using defaults (config file not found or jq not installed)"
fi
echo ""

if [ "$1" = "all" ]; then
    echo "Evaluating all trained models on edge cases..."
    echo ""

    for algo in "${ALGORITHMS[@]}"; do
        for domain in "${DOMAINS[@]}"; do
            MODEL_FILE="$PROJECT_ROOT/models/${algo}/${domain}_${algo}_model.ser"

            if [ -f "$MODEL_FILE" ]; then
                echo "==============================================================="
                echo "Testing: ${algo} trained on ${domain}"
                echo "==============================================================="
                java -cp "$CLASSPATH" sentiment.evaluation.EdgeCaseEvaluator "$algo" "$domain"
                echo ""
            fi
        done
    done

    echo "==============================================================="
    echo "Edge case evaluation complete!"
    echo "Results persisted to model metadata files."
    echo "==============================================================="
else
    ALGORITHM=$1
    DOMAIN=$2

    java -cp "$CLASSPATH" sentiment.evaluation.EdgeCaseEvaluator "$ALGORITHM" "$DOMAIN"
fi
