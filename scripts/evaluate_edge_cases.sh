#!/bin/bash
# Evaluate models on edge case challenge sets
# Usage: ./scripts/evaluate_edge_cases.sh <algorithm> <domain>
# Or: ./scripts/evaluate_edge_cases.sh all  # Test all models

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <algorithm> <domain>"
    echo "   Or: $0 all  # Evaluate all trained models"
    echo ""
    echo "Examples:"
    echo "  $0 svm imdb_50k"
    echo "  $0 all"
    exit 1
fi

# Build classpath
mvn -q dependency:build-classpath -Dmdep.outputFile=.classpath
CLASSPATH=$(cat .classpath):target/classes

if [ "$1" = "all" ]; then
    echo "Evaluating all trained models on edge cases..."
    echo ""

    ALGORITHMS=("svm" "naive_bayes" "random_forest" "logistic_regression")
    DOMAINS=("imdb_50k" "amazon_polarity" "yelp")

    for algo in "${ALGORITHMS[@]}"; do
        for domain in "${DOMAINS[@]}"; do
            MODEL_FILE="models/${algo}/${domain}_${algo}_model.ser"

            if [ -f "$MODEL_FILE" ]; then
                echo "•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••"
                echo "Testing: ${algo} trained on ${domain}"
                echo "•••••••••••••••••••••••••••••••••••••••••••••••••••••••••••"
                java -cp "$CLASSPATH" sentiment.evaluation.EdgeCaseEvaluator "$algo" "$domain"
                echo ""
            fi
        done
    done
else
    ALGORITHM=$1
    DOMAIN=$2

    java -cp "$CLASSPATH" sentiment.evaluation.EdgeCaseEvaluator "$ALGORITHM" "$DOMAIN"
fi
