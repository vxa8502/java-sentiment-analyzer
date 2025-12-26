#!/bin/bash
set -e

echo "=== Downloading Pre-Trained Models ==="
echo ""
echo "This will download all 12 trained models (4 algorithms × 3 domains)"
echo "Total size: ~600MB"
echo ""

# GitHub release URL
GITHUB_USER="vxa8502"
REPO_NAME="java-sentiment-analyzer"
VERSION="v1.0.0"
RELEASE_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}/releases/download/${VERSION}/models.tar.gz"

# Check if models already exist
if [ -d "models/svm" ] && [ -d "models/naive_bayes" ] && [ -d "models/random_forest" ] && [ -d "models/logistic_regression" ]; then
    echo " All model directories already present."
    echo ""
    echo "To re-download, delete the model directories first:"
    echo "  rm -rf models/{svm,naive_bayes,random_forest,logistic_regression}"
    echo ""
    exit 0
fi

echo "Downloading models from GitHub Release..."
echo "URL: ${RELEASE_URL}"
echo ""

# Download with progress bar
if command -v wget &> /dev/null; then
    wget --show-progress -O models.tar.gz "$RELEASE_URL"
elif command -v curl &> /dev/null; then
    curl -L --progress-bar -o models.tar.gz "$RELEASE_URL"
else
    echo " Error: Neither wget nor curl found. Please install one of them."
    exit 1
fi

echo ""
echo "Extracting models..."
tar -xzf models.tar.gz

# Clean up tarball
rm models.tar.gz

echo ""
echo " All 12 models downloaded successfully!"
echo ""
echo "Models available:"
echo "  - SVM: 3 models (IMDB, Amazon, Yelp)"
echo "  - Naive Bayes: 3 models"
echo "  - Random Forest: 3 models"
echo "  - Logistic Regression: 3 models"
echo ""
echo "You can now run:"
echo "  ./scripts/evaluate_cross_domain.sh    # Re-run evaluation"
echo "  ./scripts/evaluate_edge_cases.sh all  # Test on edge cases"
