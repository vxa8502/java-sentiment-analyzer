#!/bin/bash

# Training Pipeline Script
# Trains all model variants (SVM, Naive Bayes, Random Forest, Logistic Regression) on all datasets

set -e

# Resolve project root (works regardless of where script is called from)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "-----------------------------------------------------------------"
echo "  Multi-Dataset Model Training Pipeline"
echo "-----------------------------------------------------------------"
echo ""

# Configuration
DATASETS=("yelp" "imdb_50k" "amazon_polarity")
ALGORITHMS=("naive_bayes" "svm" "random_forest" "logistic_regression")
RANDOM_SEED=42

# NOTE: Data splits are created by prepare_data.sh.
# Training loads from data/processed/{domain}/train.csv and test.csv.
# MAX_SAMPLES is passed for backward compatibility but is ignored when
# prepared splits exist (the splits already have the configured sample count).
MAX_SAMPLES=50000

# Counters for progress tracking
total_models=$((${#DATASETS[@]} * ${#ALGORITHMS[@]}))
models_trained=0
models_skipped=0
models_failed=0

# Build project if needed
if [ ! -f "$PROJECT_ROOT/target/sentiment-analyzer-1.0.0.jar" ]; then
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

# Verify prepared splits exist for all datasets
echo "Verifying prepared data splits..."
missing_splits=false
for dataset in "${DATASETS[@]}"; do
    manifest_path="$PROJECT_ROOT/data/processed/$dataset/splits.manifest.json"
    if [ ! -f "$manifest_path" ]; then
        echo "[ERROR] No prepared splits for $dataset"
        echo "        Missing: $manifest_path"
        missing_splits=true
    else
        echo "[OK] Found splits for $dataset"
    fi
done

if [ "$missing_splits" = true ]; then
    echo ""
    echo "ERROR: Missing prepared splits. Training requires immutable data splits."
    echo ""
    echo "Run the following to prepare splits:"
    echo "  ./scripts/prepare_data.sh"
    echo ""
    echo "This ensures reproducible cross-domain evaluation."
    exit 1
fi
echo ""

# Train each algorithm on each dataset
for dataset in "${DATASETS[@]}"; do
    for algo in "${ALGORITHMS[@]}"; do
        echo "-----------------------------------------------------------------"
        echo "Training: $algo on $dataset"
        echo "-----------------------------------------------------------------"

        # Prepare paths
        # Raw path is used to infer dataset name; actual data is loaded from
        # prepared splits in data/processed/{domain}/ (created by prepare_data.sh)
        case "$dataset" in
            imdb_50k)
                dataset_path="$PROJECT_ROOT/data/raw/imdb_50k/IMDB Dataset.csv"
                ;;
            amazon_polarity)
                dataset_path="$PROJECT_ROOT/data/raw/amazon_polarity/train.csv"
                ;;
            yelp)
                dataset_path="$PROJECT_ROOT/data/raw/yelp/yelp_reviews.csv"
                ;;
            *)
                echo "[WARN] Unknown dataset: $dataset"
                continue
                ;;
        esac

        model_path="$PROJECT_ROOT/models/${algo}/${dataset}_${algo}_model.ser"

        # Check if model already exists AND is valid (skip if it does)
        if [ -f "$model_path" ]; then
            # CRITICAL: Verify model file is valid, not just exists
            # Check 1: File size > 1KB (empty/corrupt files are tiny)
            file_size=$(stat -f%z "$model_path" 2>/dev/null || stat --printf="%s" "$model_path" 2>/dev/null || echo "0")
            if [ "$file_size" -lt 1024 ]; then
                echo "[WARN] Model file exists but is too small (${file_size} bytes), likely corrupted: $model_path"
                echo "[INFO] Removing corrupted model and retraining..."
                rm -f "$model_path"
                # Also remove metadata if it exists
                rm -f "${model_path%.ser}.metadata.json"
            else
                # Check 2: Corresponding metadata file exists
                meta_path="${model_path%.ser}.metadata.json"
                if [ ! -f "$meta_path" ]; then
                    echo "[WARN] Model exists but metadata is missing: $meta_path"
                    echo "[INFO] Retraining to regenerate metadata..."
                    rm -f "$model_path"
                else
                    echo "[SKIP] Model already exists and validated: $model_path"
                    models_skipped=$((models_skipped + 1))
                    echo "Progress: $((models_trained + models_skipped + models_failed))/${total_models} | Trained: ${models_trained} | Skipped: ${models_skipped} | Failed: ${models_failed}"
                    echo ""
                    continue
                fi
            fi
        fi

        # Check if dataset exists
        if [ ! -f "$dataset_path" ]; then
            echo "[WARN] Dataset not found: $dataset_path, skipping..."
            models_failed=$((models_failed + 1))
            continue
        fi

        # Train model using TrainModel CLI via Maven exec
        # Args: <dataPath> <outputPath> <algorithm> [maxSamples] [showFeatureImportance] [topFeaturesCount] [enableHyperparameterTuning]
        # Note: Memory settings are passed via MAVEN_OPTS environment variable
        heap_size="8g"
        if [ "$algo" = "logistic_regression" ]; then
            heap_size="10g"
        fi

        echo "Using stratified 80/20 split (heap: $heap_size)"

        # Create logs directory if needed
        logs_dir="$PROJECT_ROOT/logs"
        mkdir -p "$logs_dir"
        log_file="$logs_dir/train_${dataset}_${algo}.log"

        # CRITICAL: Do NOT use -q flag - it suppresses error messages
        # Capture output to both console (summary) and log file (full details)
        echo "Training... (full log: $log_file)"
        MAVEN_OPTS="-Xmx${heap_size} -XX:+UseG1GC -XX:MaxGCPauseMillis=200" mvn exec:java \
            -Dexec.mainClass="sentiment.training.TrainModel" \
            -Dexec.args="\"$dataset_path\" \"$model_path\" \"$algo\" $MAX_SAMPLES false 30 false" \
            -Dexec.cleanupDaemonThreads=false 2>&1 | tee "$log_file"

        # Check exit status from PIPESTATUS (since we piped through tee)
        exit_code=${PIPESTATUS[0]}

        if [ $exit_code -eq 0 ]; then
            echo "[OK] Training complete: $model_path"
            models_trained=$((models_trained + 1))
        else
            echo ""
            echo "========================================"
            echo "[FAIL] Training failed for $algo on $dataset"
            echo "========================================"
            echo "  Exit code: $exit_code"
            echo "  Log file: $log_file"
            echo ""
            echo "  Last 10 lines of log:"
            tail -10 "$log_file" | sed 's/^/    /'
            echo "========================================"
            echo ""
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
echo "  1. Run cross-domain evaluation: ./scripts/evaluate_cross_domain.sh"
echo "  2. Generate report: ./scripts/generate_report.sh"
