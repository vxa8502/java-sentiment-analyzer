# Model Training Guide

Complete guide for training, evaluating, and comparing sentiment analysis models.

---

## Prerequisites

- **Java 21** (JDK)
- **Maven 3.9+**
- **Training Dataset**: [Amazon Customer Reviews Polarity](https://www.kaggle.com/datasets/bhavikardeshna/amazon-customerreviews-polarity)
  - Place in `data/datasets/Reviews.csv`

---

## Quick Start

### Train a Single Model

```bash
mvn exec:java -Dexec.mainClass="sentiment.training.TrainModel" \
  -Dexec.args="./data/datasets/Reviews.csv ./models/svm-model.ser 10000 true 30 false"
```

**Arguments:**
1. `dataPath`: Path to training CSV file
2. `outputPath`: Where to save trained model (.ser file)
3. `maxSamples`: Number of samples to use (0 = all)
4. `showFeatureImportance`: Analyze important features (true/false)
5. `topFeaturesCount`: Number of top features to display
6. `enableHyperparameterTuning`: Run grid search for SVM (slower but more accurate)

**Output:**
- `models/svm-model.ser` - Trained model (binary)
- `models/svm-model.metadata.json` - Training metadata (reproducibility)
- `models/svm-model.feature-importance.json` - Feature analysis

---

## Model Reproducibility

Every training run generates a **metadata file** that captures everything needed to reproduce the model:

**Example: `models/svm-model.metadata.json`**
```json
{
  "model_id": "svm-2024-12-12T10-30-00",
  "algorithm": "SVM",
  "hyperparameters": {
    "algorithm": "SVM"
  },
  "dataset": {
    "source": "amazon_reviews_polarity",
    "path": "./data/datasets/Reviews.csv",
    "samples": {
      "train": 6000,
      "val": 2000,
      "test": 2000
    }
  },
  "preprocessing": {
    "lowercase": true,
    "remove_stopwords": false,
    "max_features": 5000,
    "min_word_frequency": 2,
    "tokenizer": "AdvancedTokenizer"
  },
  "metrics": {
    "test_accuracy": 0.847,
    "test_precision": 0.851,
    "test_recall": 0.843,
    "test_f1": 0.847,
    "confusion_matrix": [[850, 150], [157, 843]],
    "roc_auc": 0.912
  },
  "trained_at": "2024-12-12T10:30:00Z",
  "training_duration_seconds": 42,
  "model_file": "svm-model.ser",
  "model_size_bytes": 3355443
}
```

**Why this matters:** Six months from now, you can trace exactly how this model was built.

---

## Comparing Multiple Models

Train different algorithms and compare their performance using the metadata files:

### 1. Train Multiple Models

```bash
# Train SVM
mvn exec:java -Dexec.mainClass="sentiment.training.TrainModel" \
  -Dexec.args="./data/datasets/Reviews.csv ./models/svm-model.ser 10000 false 30 false"

# Train Naive Bayes
mvn exec:java -Dexec.mainClass="sentiment.training.TrainModel" \
  -Dexec.args="./data/datasets/Reviews.csv ./models/nb-model.ser 10000 false 30 false"

# Train Random Forest
mvn exec:java -Dexec.mainClass="sentiment.training.TrainModel" \
  -Dexec.args="./data/datasets/Reviews.csv ./models/rf-model.ser 10000 false 30 false"
```

### 2. Compare Metadata

```bash
# View all model metrics
jq '.metrics' models/*.metadata.json

# Compare accuracy
jq -r '"\(.algorithm): \(.metrics.test_accuracy)"' models/*.metadata.json

# Compare model sizes
ls -lh models/*.ser
```

**Example comparison:**
```bash
$ jq -r '"\(.algorithm): Acc=\(.metrics.test_accuracy) F1=\(.metrics.test_f1)"' models/*.metadata.json
SVM: Acc=0.847 F1=0.847
NAIVE_BAYES: Acc=0.823 F1=0.823
RANDOM_FOREST: Acc=0.856 F1=0.856
```

**Choosing the best model:**
- **Highest accuracy:** Random Forest (0.856)
- **Fastest training:** Naive Bayes (typically 5x faster)
- **Best balanced:** SVM (good accuracy, moderate speed)

---

## Training Data Split

All training uses stratified 60/20/20 split with fixed seed for reproducibility:

- **60% Training** - Model learns patterns
- **20% Validation** - Hyperparameter tuning (if enabled)
- **20% Test** - Final evaluation (reported in metadata)

---

## Hyperparameter Tuning

Enable grid search for SVM to find optimal parameters:

```bash
mvn exec:java -Dexec.mainClass="sentiment.training.TrainModel" \
  -Dexec.args="./data/datasets/Reviews.csv ./models/svm-tuned.ser 10000 false 30 true"
```

**Grid Search Parameters:**
- Kernel: RBF, Polynomial, Linear
- C: [0.1, 1.0, 10.0]
- Gamma: [0.001, 0.01, 0.1]

**Note:** This increases training time 5-10x but often improves accuracy by 1-3%.

---

## Feature Importance Analysis

Understand what words drive predictions:

```bash
mvn exec:java -Dexec.mainClass="sentiment.training.TrainModel" \
  -Dexec.args="./data/datasets/Reviews.csv ./models/svm-model.ser 10000 true 30 false"
```

**Console Output:**
```
========================================
FEATURE IMPORTANCE ANALYSIS - SVM
========================================
Analysis completed in 1243ms

Statistics:
  Total features: 5000
  Mean absolute weight: 0.123456
  Std deviation: 0.234567
  Median: 0.089012
  95th percentile: 0.567890
========================================

Top 30 Most Influential Features:
  1. excellent      →  +0.8234 (strong positive)
  2. terrible       →  -0.7891 (strong negative)
  3. amazing        →  +0.7654 (strong positive)
  4. awful          →  -0.7123 (strong negative)
  ...
```

**API Access:**
```bash
curl http://localhost:8080/api/v1/model/feature-importance?topFeatures=10
```

---

## Model Versioning Best Practices

### Naming Convention

```
<algorithm>-<date>-<variant>.ser

Examples:
  svm-2024-12-12-baseline.ser
  svm-2024-12-12-tuned.ser
  nb-2024-12-12-10k-samples.ser
```

### Version Control

**DO commit:**
- `*.metadata.json` - Training metadata
- `*.feature-importance.json` - Feature analysis
- `model-comparison-report.md` - Benchmark results

**DO NOT commit:**
- `*.ser` - Large binary model files (use Git LFS or artifact storage)

### Production Deployment

1. Compare models using BenchmarkCLI
2. Choose best model based on accuracy/latency tradeoff
3. Update `application.yml`:
   ```yaml
   sentiment:
     models:
       svm-model-path: ./models/svm-2024-12-12-tuned.ser
   ```
4. Restart application
5. Monitor production metrics at `/api/v1/health`

---

## Troubleshooting

### Issue: Low Accuracy (<75%)

**Possible Causes:**
- Insufficient training data
- Domain mismatch (training on product reviews, testing on tweets)
- Poor feature representation

**Solutions:**
1. Increase `maxSamples` (use more data)
2. Enable hyperparameter tuning
3. Try different algorithms (compare all three)

### Issue: Out of Memory During Training

**Solution:**
```bash
export MAVEN_OPTS="-Xmx4g"
mvn exec:java -Dexec.mainClass="sentiment.training.TrainModel" ...
```

### Issue: Training Takes Too Long

**Solutions:**
1. Reduce `maxSamples` (e.g., 10000 instead of 100000)
2. Disable hyperparameter tuning
3. Use Naive Bayes (fastest algorithm)

---

## Advanced: Custom Datasets

Your CSV must have two columns:

```csv
text,sentiment
"This product is great!",positive
"Terrible quality, very disappointed.",negative
```

**Supported Labels:**
- `positive` / `negative` (binary classification)
- `positive` / `negative` / `neutral` (multi-class)

---

## Performance Benchmarks

### Single Model Training (10K samples)

| Algorithm | Training Time | Test Accuracy | Inference Latency |
|-----------|--------------|---------------|-------------------|
| Naive Bayes | ~5 seconds | 82-83% | 3ms |
| SVM (no tuning) | ~30 seconds | 84-85% | 12ms |
| SVM (with tuning) | ~5 minutes | 85-87% | 12ms |
| Random Forest | ~90 seconds | 85-86% | 28ms |

*Measured on M1 MacBook Pro with Java 21*

---

## Next Steps

1. **Train your first model:**
   ```bash
   mvn exec:java -Dexec.mainClass="sentiment.training.TrainModel" \
     -Dexec.args="./data/datasets/Reviews.csv ./models/my-first-model.ser 5000 true 20 false"
   ```

2. **Inspect metadata:**
   ```bash
   cat models/my-first-model.metadata.json | jq .
   ```

3. **Compare algorithms:**
   - Train SVM, Naive Bayes, and Random Forest
   - Compare metadata using `jq`
   - Choose best model for your use case

4. **Deploy to production:**
   - See [DEPLOYMENT.md](DEPLOYMENT.md) for Docker deployment
   - Monitor metrics at `/api/v1/health`

---

**Last Updated:** 2024-12-12
**Author:** Victoria Alabi
