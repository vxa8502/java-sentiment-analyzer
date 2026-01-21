# Sentiment Analysis: Final Comprehensive Evaluation Report

**Project**: Cross-Domain Sentiment Classification with Edge Case Analysis
**Date**: January 21, 2026
**Author**: Victoria Alabi
**Generated**: Auto-generated from model metadata (do not edit manually)

---

## Executive Summary

This report summarizes the training and evaluation of sentiment analysis models across multiple algorithms and domains.

### Production Model

**Algorithm**: SVM
**Training Dataset**: amazon_polarity (40000 samples)
**Model Size**: 14.4 MB

| Metric | Value |
|--------|-------|
| Test Accuracy | 88.4% |
| Test F1 | 0.884 |
| Test Precision | 0.884 |
| Test Recall | 0.884 |
| ROC-AUC | 0.884 |
| Cross-Domain Avg | 88.2% |

### Production Model Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 4404 (TN) | 599 (FP) |
| **Actual Positive** | 560 (FN) | 4437 (TP) |

### Best Generalizing Model

**Model**: svm-amazon_polarity
**Cross-Domain Average Accuracy**: 88.2%

---

## Part 1: Model Comparison (All 12 Experiments)

| Algorithm | Dataset | Accuracy | F1 | Precision | Recall | Training Time |
|-----------|---------|----------|-----|-----------|--------|---------------|
| SVM | amazon_polarity | 88.4% | 0.884 | 0.884 | 0.884 | 59m 25s |
| SVM | imdb_50k | 89.0% | 0.890 | 0.890 | 0.890 | 159m 30s |
| SVM | yelp | 94.5% | 0.945 | 0.945 | 0.945 | 14m 17s |
| LOGISTIC_REGRESSION | amazon_polarity | 84.3% | 0.843 | 0.843 | 0.843 | 36m 31s |
| LOGISTIC_REGRESSION | imdb_50k | 85.8% | 0.858 | 0.858 | 0.858 | 74m 24s |
| LOGISTIC_REGRESSION | yelp | 92.3% | 0.923 | 0.923 | 0.923 | 8m 30s |
| RANDOM_FOREST | amazon_polarity | 86.9% | 0.869 | 0.869 | 0.869 | 40m 23s |
| RANDOM_FOREST | imdb_50k | 86.8% | 0.868 | 0.868 | 0.868 | 54m 50s |
| RANDOM_FOREST | yelp | 92.5% | 0.925 | 0.926 | 0.925 | 10m 41s |
| NAIVE_BAYES | amazon_polarity | 79.8% | 0.798 | 0.800 | 0.798 | 39m 47s |
| NAIVE_BAYES | imdb_50k | 83.9% | 0.839 | 0.839 | 0.839 | 64m 13s |
| NAIVE_BAYES | yelp | 80.6% | 0.806 | 0.806 | 0.806 | 6m 10s |

---

## Part 2: Cross-Domain Evaluation

Each model was evaluated on all three test domains. Asterisk (*) indicates in-domain evaluation.

#### SVM

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdb 50k | 89.0% * | 81.2% | 84.5% | 82.9% |
| amazon polarity | 84.8% | 88.4% * | 91.5% | 88.2% |
| yelp | 78.1% | 81.4% | 94.5% * | 79.8% |

#### LOGISTIC REGRESSION

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdb 50k | 85.8% * | 77.2% | 80.2% | 78.7% |
| amazon polarity | 80.9% | 84.3% * | 85.8% | 83.4% |
| yelp | 73.8% | 76.0% | 92.3% * | 74.9% |

#### RANDOM FOREST

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdb 50k | 86.8% * | 78.0% | 82.5% | 80.2% |
| amazon polarity | 80.9% | 86.9% * | 89.8% | 85.3% |
| yelp | 70.9% | 80.8% | 92.5% * | 75.8% |

#### NAIVE BAYES

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdb 50k | 83.9% * | 72.6% | 78.1% | 75.3% |
| amazon polarity | 71.1% | 79.8% * | 81.3% | 76.2% |
| yelp | 60.1% | 68.3% | 80.6% * | 64.2% |

**Legend**: * = in-domain evaluation

---

## Part 3: Edge Case Evaluation

**Total Edge Cases**: 213 samples across 5 categories (204 included in metrics)

| Category | Samples | Included |
|----------|---------|----------|
| Mixed Sentiment | 83 | Yes |
| Domain Jargon | 69 | Yes |
| Negation Heavy | 52 | Yes |
| Label Error | 7 | No (n < 30) |
| Sarcasm | 2 | No (n < 30) |

Categories with fewer than 30 samples are excluded from aggregate metrics (confidence intervals too wide for meaningful analysis).

---

## Part 4: Reproducibility

All results can be reproduced via:

```bash
# Phase 1: Prepare immutable data splits (run once)
./scripts/prepare_data.sh

# Phase 2: Train all models
./scripts/train_all_models.sh

# Phase 3: Cross-domain evaluation
./scripts/evaluate_cross_domain.sh

# Regenerate this report
./scripts/generate_report.sh
```

### Model Artifacts

| File | Algorithm | Dataset | Size |
|------|-----------|---------|------|
| amazon_polarity_logistic_regression_model.ser | LOGISTIC_REGRESSION | amazon_polarity | 14.6 MB |
| imdb_50k_logistic_regression_model.ser | LOGISTIC_REGRESSION | imdb_50k | 31.2 MB |
| yelp_logistic_regression_model.ser | LOGISTIC_REGRESSION | yelp | 9.8 MB |
| amazon_polarity_naive_bayes_model.ser | NAIVE_BAYES | amazon_polarity | 14.7 MB |
| imdb_50k_naive_bayes_model.ser | NAIVE_BAYES | imdb_50k | 31.3 MB |
| yelp_naive_bayes_model.ser | NAIVE_BAYES | yelp | 10.0 MB |
| sentiment_model.ser | SVM | amazon_polarity | 14.4 MB |
| amazon_polarity_random_forest_model.ser | RANDOM_FOREST | amazon_polarity | 199.5 MB |
| imdb_50k_random_forest_model.ser | RANDOM_FOREST | imdb_50k | 198.0 MB |
| yelp_random_forest_model.ser | RANDOM_FOREST | yelp | 94.7 MB |
| amazon_polarity_svm_model.ser | SVM | amazon_polarity | 14.4 MB |
| imdb_50k_svm_model.ser | SVM | imdb_50k | 31.0 MB |
| yelp_svm_model.ser | SVM | yelp | 9.6 MB |

---

## Metadata

- **Report Generated**: 2026-01-21T19:55:22Z
- **Git Commit**: df2fc1f
- **Java Version**: 24.0.1

