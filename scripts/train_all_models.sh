#!/bin/bash

# Training Pipeline Script
# Trains all model variants (SVM, Naive Bayes, Random Forest, Logistic Regression) on all datasets
# Following David's and Sofia's recommendations for reproducible training

set -e

echo "-----------------------------------------------------------------"
echo "  Multi-Dataset Model Training Pipeline"
echo "-----------------------------------------------------------------"
echo ""

# Configuration
DATASETS=("yelp" "imdb_50k" "amazon_polarity")
ALGORITHMS=("naive_bayes" "svm" "random_forest" "logistic_regression")
MAX_SAMPLES=100000  # Sofia's recommendation: 100K balanced samples
RANDOM_SEED=42

# Counters for progress tracking
total_models=$((${#DATASETS[@]} * ${#ALGORITHMS[@]}))
models_trained=0
models_skipped=0
models_failed=0

# Build project if needed
if [ ! -f "target/sentiment-analyzer-1.0.0.jar" ]; then
    echo "Building project..."
    mvn clean package -DskipTests
    echo "[OK] Build complete"
    echo ""
fi

echo "Training configuration:"
echo "  Datasets: ${DATASETS[*]}"
echo "  Algorithms: ${ALGORITHMS[*]}"
echo "  Max samples per dataset: ${MAX_SAMPLES}"
echo "  Random seed: ${RANDOM_SEED}"
echo "  Total models to check: ${total_models}"
echo ""

# Train each algorithm on each dataset
for dataset in "${DATASETS[@]}"; do
    for algo in "${ALGORITHMS[@]}"; do
        echo "-----------------------------------------------------------------"
        echo "Training: $algo on $dataset"
        echo "-----------------------------------------------------------------"

        # Prepare paths (respect original data format)
        # Adjust maxSamples for memory-constrained environments
        case "$dataset" in
            imdb_50k)
                dataset_path="data/raw/imdb_50k/IMDB Dataset.csv"  # 50K single file
                test_path=""  # No pre-split, training will do 60/20/20
                dataset_max_samples=$MAX_SAMPLES
                ;;
            amazon_polarity)
                dataset_path="data/raw/amazon_polarity/train.csv"  # 3.6M official training set
                test_path="data/raw/amazon_polarity/test.csv"      # 400K official test set
                dataset_max_samples=50000  # Reduced for memory constraints (8GB RAM)
                ;;
            yelp)
                dataset_path="data/raw/yelp/yelp_reviews.csv"  # 100K single file
                test_path=""  # No pre-split, training will do 60/20/20
                dataset_max_samples=$MAX_SAMPLES
                ;;
            *)
                echo "[WARN] Unknown dataset: $dataset"
                continue
                ;;
        esac

        model_path="models/${algo}/${dataset}_${algo}_model.ser"

        # Check if model already exists (skip if it does)
        if [ -f "$model_path" ]; then
            echo "[SKIP] Model already exists: $model_path"
            models_skipped=$((models_skipped + 1))
            echo "Progress: $((models_trained + models_skipped + models_failed))/${total_models} | Trained: ${models_trained} | Skipped: ${models_skipped} | Failed: ${models_failed}"
            echo ""
            continue
        fi

        # Check if dataset exists
        if [ ! -f "$dataset_path" ]; then
            echo "[WARN] Dataset not found: $dataset_path, skipping..."
            models_failed=$((models_failed + 1))
            continue
        fi

        # Train model using TrainModel CLI via Maven exec
        # Args: <dataPath> <outputPath> <algorithm> [maxSamples] [showFeatureImportance] [topFeaturesCount] [enableHyperparameterTuning] [testDataPath]
        # Note: Memory settings are passed via MAVEN_OPTS environment variable (8g for logistic regression, 6g for others)
        heap_size="8g"
        if [ "$algo" = "logistic_regression" ]; then
            heap_size="10g"
        fi

        if [ -n "$test_path" ] && [ -f "$test_path" ]; then
            echo "Using official test set: $test_path (heap: $heap_size)"
            MAVEN_OPTS="-Xmx${heap_size} -XX:+UseG1GC -XX:MaxGCPauseMillis=200" mvn -q exec:java \
                -Dexec.mainClass="sentiment.training.TrainModel" \
                -Dexec.args="\"$dataset_path\" \"$model_path\" \"$algo\" $MAX_SAMPLES false 30 false \"$test_path\"" \
                -Dexec.cleanupDaemonThreads=false
        else
            echo "Using 60/20/20 split (heap: $heap_size)"
            MAVEN_OPTS="-Xmx${heap_size} -XX:+UseG1GC -XX:MaxGCPauseMillis=200" mvn -q exec:java \
                -Dexec.mainClass="sentiment.training.TrainModel" \
                -Dexec.args="\"$dataset_path\" \"$model_path\" \"$algo\" $MAX_SAMPLES false 30 false" \
                -Dexec.cleanupDaemonThreads=false
        fi

        if [ $? -eq 0 ]; then
            echo "[OK] Training complete: $model_path"
            models_trained=$((models_trained + 1))
        else
            echo "[FAIL] Training failed for $algo on $dataset"
            models_failed=$((models_failed + 1))
            # Don't exit - continue with remaining models
        fi

        echo "Progress: $((models_trained + models_skipped + models_failed))/${total_models} | Trained: ${models_trained} | Skipped: ${models_skipped} | Failed: ${models_failed}"
        echo ""
    done
done

echo "-----------------------------------------------------------------"
echo "  Training Complete"
echo "-----------------------------------------------------------------"
echo ""
echo "Summary:"
echo "  Total models: ${total_models}"
echo "  Successfully trained: ${models_trained}"
echo "  Skipped (already exist): ${models_skipped}"
echo "  Failed: ${models_failed}"
echo ""

if [ ${models_failed} -gt 0 ]; then
    echo "[WARN] Some models failed to train. Review the output above for details."
    echo ""
fi

echo "Next steps:"
echo "  1. Run quality assessment: ./scripts/run_quality_reports.sh"
echo "  2. Run cross-domain evaluation: ./scripts/evaluate_cross_domain.sh"
echo "  3. Start API server: docker-compose up"
