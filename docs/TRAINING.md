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
| imdb_50k | 25K | 10K | Movie reviews |
| amazon_polarity | 100K | 20K | Product reviews |
| yelp | 100K | 20K | Restaurant reviews |

**Location:** `data/raw/<dataset>/`

**Format:** Binary classification only (positive/negative). Neutral samples are filtered during loading.

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

```bash
# Generate error candidates from model predictions
./scripts/generate_edge_cases.sh all

# After manual categorization, evaluate
./scripts/evaluate_edge_cases.sh all
```

Edge case categories:
- `sarcasm.csv` - Sarcastic text
- `mixed_sentiment.csv` - Mixed positive/negative
- `negation_heavy.csv` - Heavy use of negation
- `domain_jargon.csv` - Domain-specific language

### Phase 4: Promote to Production

```bash
./scripts/promote_to_production.sh
```

Selects the best-generalizing model based on `cross_domain_matrix.json`, runs hyperparameter tuning (for SVM), and deploys to `models/production/`.

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
  preprocessing:
    max-features: 5000
    min-word-frequency: 2
    use-tfidf: true
    use-bigrams: true
```

## Version Control

**Tracked (in git):**
- `*.metadata.json` - Training metrics
- `*-feature-importance.json` - Feature analysis
- `models/production/` - Production model (via Git LFS)

**Ignored:**
- `models/<algo>/*.ser` - Training model binaries
