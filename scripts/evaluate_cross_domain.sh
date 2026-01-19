#!/bin/bash

# Cross-Domain Evaluation Script
# Tests all models (SVM, Naive Bayes, Random Forest, Logistic Regression) across all datasets

set -e

# Resolve project root (works regardless of where script is called from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "---------------------------------------------------------------"
echo "  Cross-Domain Evaluation"
echo "---------------------------------------------------------------"
echo ""

# Configuration
MODELS_DIR="$PROJECT_ROOT/models"
TEST_DATA_DIR="$PROJECT_ROOT/data/raw"
OUTPUT_FILE="$PROJECT_ROOT/results/cross_domain_matrix.json"
RESULTS_DIR="$PROJECT_ROOT/results"

# Create results directory if needed
mkdir -p "$RESULTS_DIR"

# Build project if needed
if [ ! -f "$PROJECT_ROOT/target/sentiment-analyzer-1.0.0.jar" ]; then
    echo "Building project..."
    mvn clean package -DskipTests
    echo "[OK] Build complete"
    echo ""
fi

echo "Evaluation configuration:"
echo "  Models directory: $MODELS_DIR"
echo "  Test data directory: $TEST_DATA_DIR"
echo "  Output file: $OUTPUT_FILE"
echo ""

# Check if models exist
if [ ! -d "$MODELS_DIR" ]; then
    echo "[FAIL] Models directory not found: $MODELS_DIR"
    echo "Train models first using: ./scripts/train_all_models.sh"
    exit 1
fi

# Count available models
model_count=$(find "$MODELS_DIR" -name "*.ser" 2>/dev/null | wc -l | tr -d ' ')
if [ "$model_count" -eq 0 ]; then
    echo "[FAIL] No trained models found in $MODELS_DIR"
    echo "Train models first using: ./scripts/train_all_models.sh"
    exit 1
fi

echo "Found $model_count trained model(s)"
echo ""

# Run cross-domain evaluation
echo "Running cross-domain evaluation..."
echo "This will test each model on all 3 datasets (IMDB, Amazon, Yelp)"
echo ""

# Build classpath from Maven dependencies
CLASSPATH=$(mvn -q dependency:build-classpath -Dmdep.outputFile=/dev/stdout)
CLASSPATH="target/classes:target/sentiment-analyzer-1.0.0.jar.original:$CLASSPATH"

# Use full saved test sets (up to 25K per domain, more than any test set has)
MAX_SAMPLES_PER_DOMAIN=25000

java -Xmx8g -cp "$CLASSPATH" \
    sentiment.evaluation.CrossDomainEvaluator \
    "$MODELS_DIR" \
    "$TEST_DATA_DIR" \
    "$OUTPUT_FILE" \
    "$MAX_SAMPLES_PER_DOMAIN"

if [ $? -eq 0 ]; then
    echo ""
    echo "---------------------------------------------------------------"
    echo "  Cross-Domain Evaluation Complete"
    echo "---------------------------------------------------------------"
    echo ""
    echo "Results saved to: $OUTPUT_FILE"
    echo ""
    echo "Key Insight:"
    echo "  The best model is the one with highest CROSS-DOMAIN average,"
    echo "  not the highest single-domain accuracy."
    echo ""
else
    echo "[FAIL] Cross-domain evaluation failed"
    exit 1
fi
