# Model Training Guide

Training, evaluating, and deploying sentiment analysis models.

## Prerequisites

- Java 21 (JDK)
- Maven 3.9+
- Datasets in `data/raw/` (see Datasets section)

## Quick Start

Train all 12 models (4 algorithms x 3 datasets):

```bash
./scripts/train_all_models.sh
```

Or train a single model:

```bash
mvn exec:java -Dexec.mainClass="sentiment.training.ModelTrainer" \
  -Dexec.args="--algorithm SVM --dataset imdb_50k"
```

**Algorithms:** `SVM`, `NAIVE_BAYES`, `LOGISTIC_REGRESSION`, `RANDOM_FOREST`

**Datasets:** `imdb_50k`, `amazon_polarity`, `yelp`

## Output Structure

Each training run generates:

```
models/<algorithm>/
├── <dataset>_<algorithm>_model.ser           # Trained model
└── <dataset>_<algorithm>_model.metadata.json # Training metrics
```

Example:
```
models/svm/
├── imdb_50k_svm_model.ser
├── imdb_50k_svm_model.metadata.json
├── amazon_polarity_svm_model.ser
├── amazon_polarity_svm_model.metadata.json
├── yelp_svm_model.ser
└── yelp_svm_model.metadata.json
```

## Datasets

| Dataset | Train | Test | Domain |
|---------|-------|------|--------|
| imdb_50k | ~40K | ~10K | Movie reviews |
| amazon_polarity | ~40K | ~10K | Product reviews |
| yelp | ~20K | ~5K | Restaurant reviews |

**Location:** `data/raw/<dataset>/`

**Format:** Binary classification only (positive/negative). All datasets capped at 50K samples with stratified sampling to preserve class distribution, then balanced to 50/50 class ratio. Neutral samples are filtered during loading.

## Training Pipeline

### Phase 1: Train Base Models

```bash
./scripts/train_all_models.sh
```

Trains 12 models (4 algorithms x 3 datasets). Each model is evaluated on its own test set.

### Phase 2: Cross-Domain Evaluation

```bash
./scripts/evaluate_cross_domain.sh
```

Tests each model on all 3 datasets to measure generalization. Results saved to:
- `results/cross_domain_matrix.json`
- Each model's `metadata.json` updated with `cross_domain_performance`

### Phase 3: Edge Case Evaluation

Edge case testing reveals model brittleness that standard test accuracy can mask. A model scoring 90% on clean test data may score 60% on edge cases.

#### What is an Edge Case?

An **edge case** is a **systematic model failure** - a text that multiple algorithms (3+ by default) fail to classify correctly. Single-model failures are excluded as they represent model-specific quirks, not generalizable difficulty.

This threshold is configurable in `config/edge-case-evaluation.json`:
```json
"edge_case_definition": {
  "min_models_failed": 3,
  "rationale": "Errors where 3+ algorithms failed indicate systematic difficulty"
}
```

#### Step 1: Extract prediction errors
```bash
./scripts/generate_edge_cases.sh all
```

Runs `ErrorAnalyzer` on all 12 models, extracting misclassifications from test sets.

#### Step 2: Data quality audit

Raw error files contain label noise from source datasets. We audit a sample to estimate label error rates and filter accordingly.

```bash
# Deduplicate and create stratified audit sample
python scripts/prepare_edge_case_audit.py

# Interactive audit (classify each as label_error/true_error/ambiguous)
python scripts/audit_edge_cases.py
```

Audit results are saved to `data/raw/edge_cases/audit_results.json` and used by downstream scripts to exclude label errors.

#### Step 3: Prepare edge case sample

Filter to systematic failures (3+ models) and create a stratified sample for categorization:

```bash
python scripts/prepare_categorization_sample.py --target-size 200
```

This excludes single-model failures, keeping only texts that multiple algorithms found difficult.

#### Step 4: Categorize edge cases

Manually categorize the sample into edge case types:
- `sarcasm.csv` - Surface sentiment contradicts true intent
- `mixed_sentiment.csv` - Contains both positive and negative aspects
- `negation_heavy.csv` - Complex negation patterns
- `domain_jargon.csv` - Domain-specific terminology

```bash
python scripts/categorize_errors.py --sample
```

**Output CSV format:**
```csv
text,sentiment,num_models_failed,failed_models,notes
"Review text...",NEGATIVE,4,svm-amazon;nb-amazon;lr-amazon;rf-amazon,"..."
```

#### Step 5: Evaluate models on edge cases
```bash
./scripts/evaluate_edge_cases.sh all --persist
```

### Phase 4: Promote to Production

```bash
./scripts/promote_to_production.sh
```

Selects the best-generalizing model based on `cross_domain_matrix.json`, runs hyperparameter tuning (for SVM), and deploys to `models/production/`.

**Options:**
- `--skip-tuning` - Skip SVM hyperparameter tuning (faster, uses default C=0.1)
- `--force-model=<model>` - Override winner (e.g., `svm-amazon_polarity`)

```bash
# Fast promotion without hyperparameter tuning
./scripts/promote_to_production.sh --skip-tuning

# Force a specific model
./scripts/promote_to_production.sh --force-model=naive_bayes-imdb_50k
```

**Output:**
```
models/production/
├── sentiment_model.ser
├── sentiment_model.metadata.json
└── sentiment_model-feature-importance.json
```

## Metadata Format

Every model generates a metadata file for reproducibility:

```json
{
  "algorithm": "SVM",
  "dataset": "amazon_polarity",
  "performance": {
    "confusion_matrix": [[TN, FP], [FN, TP]],
    "test_accuracy": 0.89,
    "test_precision": 0.89,
    "test_recall": 0.89,
    "test_f1": 0.89,
    "roc_auc": 0.94
  },
  "cross_domain_performance": {
    "imdb_50k": {"accuracy": 0.85, "f1": 0.85},
    "amazon_polarity": {"accuracy": 0.89, "f1": 0.89},
    "yelp": {"accuracy": 0.84, "f1": 0.84},
    "cross_domain_average": 0.86
  },
  "preprocessing": {
    "max_features": 5000,
    "min_word_frequency": 2,
    "use_tfidf": true
  },
  "trained_at": "2026-01-18T10:30:00Z"
}
```

## Comparing Models

```bash
# View all model accuracies
for f in models/*/*.metadata.json; do
  echo "$(basename $f): $(jq -r '.performance.test_accuracy' $f)"
done

# Compare cross-domain averages
jq -r '.cross_domain_average' results/cross_domain_matrix.json

# Find best generalizing model
jq -r '.best_generalizing_model' results/cross_domain_matrix.json
```

## Troubleshooting

### Out of Memory

```bash
export MAVEN_OPTS="-Xmx4g"
./scripts/train_all_models.sh
```

### Training Too Slow

- Reduce dataset size in training script
- Use Naive Bayes (fastest algorithm)
- Disable hyperparameter tuning

### Low Accuracy (<75%)

- Check data loading (should be binary, no neutral)
- Verify preprocessing config in `application.yml`
- Try different algorithms

### 3x3 Confusion Matrix

Data pipeline issue - neutral samples not filtered. Check `SimpleDatasetLoader.parseSentiment()`.

## Reset and Retrain

To start fresh:

```bash
./scripts/reset.sh           # Delete all artifacts
./scripts/reset.sh --verify  # Reset + verify data pipeline
./scripts/reset.sh --dry-run # Preview deletions
```

## Configuration

Training parameters in `src/main/resources/application.yml`:

```yaml
sentiment:
  training:
    random-seed: 42          # For reproducibility

  features:
    max-features: 5000       # Final vocabulary size
    min-term-freq: 1         # Minimum term frequency
    use-tfidf: true          # Enable TF-IDF
    use-bigrams: true        # Enable bigram features
    mi-selection-threshold: 50000  # Trigger MI selection when vocab exceeds this
```

### Environment Variables

Override configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTIMENT_RANDOM_SEED` | 42 | Random seed for shuffling, splits, CV |
| `SENTIMENT_MI_THRESHOLD` | 50000 | MI feature selection trigger threshold |
| `SENTIMENT_DATA_IMDB` | `data/raw/imdb_50k/IMDB Dataset.csv` | IMDB dataset path |
| `SENTIMENT_DATA_AMAZON` | `data/raw/amazon_polarity/train.csv` | Amazon dataset path |
| `SENTIMENT_DATA_YELP` | `data/raw/yelp/yelp_reviews.csv` | Yelp dataset path |

Example:
```bash
# Use different random seed for variance analysis
SENTIMENT_RANDOM_SEED=123 ./scripts/train_all_models.sh

# Custom dataset location
SENTIMENT_DATA_AMAZON=/data/custom/amazon.csv ./scripts/promote_to_production.sh
```

## Version Control

**Tracked (in git):**
- `*.metadata.json` - Training metrics
- `*-feature-importance.json` - Feature analysis
- `models/production/` - Production model (via Git LFS)

**Ignored:**
- `models/<algo>/*.ser` - Training model binaries
