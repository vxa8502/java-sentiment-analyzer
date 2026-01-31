# Sentiment Analysis: Final Comprehensive Evaluation Report

**Project**: Cross-Domain Sentiment Classification
**Date**: January 30, 2026
**Author**: Victoria Alabi
**Generated**: Auto-generated from model metadata (do not edit manually)

---

## Executive Summary

This report summarizes the training and evaluation of sentiment analysis models across multiple algorithms and domains.

### Production Model

**Algorithm**: SVM
**Training Dataset**: amazon_polarity (40000 samples)
**Model Size**: 15.9 MB

| Metric | Value |
|--------|-------|
| Test Accuracy | 89.6% |
| Test F1 | 0.896 |
| Test Precision | 0.896 |
| Test Recall | 0.896 |
| ROC-AUC | 0.957 |
| Cross-Domain Avg | 88.0% |

### Production Model Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 4478 (TN) | 525 (FP) |
| **Actual Positive** | 519 (FN) | 4478 (TP) |

### Best Generalizing Model

**Model**: svm-amazon_polarity
**Cross-Domain Average Accuracy**: 88.0%

---

## Part 1: Model Comparison (All 12 Experiments)

| Algorithm | Dataset | Accuracy | F1 | Precision | Recall | Training Time |
|-----------|---------|----------|-----|-----------|--------|---------------|
| SVM | amazon_polarity | 89.2% | 0.892 | 0.893 | 0.892 | 97m 22s |
| SVM | imdb_50k | 89.1% | 0.891 | 0.891 | 0.891 | 245m 56s |
| SVM | yelp | 94.0% | 0.940 | 0.940 | 0.940 | 22m 32s |
| LOGISTIC_REGRESSION | amazon_polarity | 84.1% | 0.841 | 0.841 | 0.841 | 53m 28s |
| LOGISTIC_REGRESSION | imdb_50k | 85.5% | 0.855 | 0.855 | 0.855 | 87m 57s |
| LOGISTIC_REGRESSION | yelp | 92.6% | 0.926 | 0.926 | 0.926 | 24m 48s |
| RANDOM_FOREST | amazon_polarity | 86.6% | 0.866 | 0.866 | 0.866 | 101m 40s |
| RANDOM_FOREST | imdb_50k | 86.6% | 0.866 | 0.866 | 0.866 | 223m 51s |
| RANDOM_FOREST | yelp | 92.0% | 0.920 | 0.921 | 0.920 | 37m 47s |
| NAIVE_BAYES | amazon_polarity | 79.2% | 0.792 | 0.796 | 0.792 | 59m 38s |
| NAIVE_BAYES | imdb_50k | 82.9% | 0.829 | 0.830 | 0.829 | 115m 9s |
| NAIVE_BAYES | yelp | 81.3% | 0.813 | 0.816 | 0.813 | 10m 39s |

---

## Part 2: Cross-Domain Evaluation

Each model was evaluated on all three test domains. Asterisk (*) indicates in-domain evaluation.

#### SVM

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdb 50k | 89.1% * | 81.9% | 84.9% | 83.4% |
| amazon polarity | 85.2% | 89.2% * | 90.9% | 88.0% |
| yelp | 78.5% | 82.0% | 94.0% * | 80.3% |

#### LOGISTIC REGRESSION

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdb 50k | 85.5% * | 76.8% | 79.3% | 78.1% |
| amazon polarity | 80.4% | 84.1% * | 84.7% | 82.6% |
| yelp | 75.8% | 77.8% | 92.6% * | 76.8% |

#### RANDOM FOREST

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdb 50k | 86.6% * | 80.2% | 83.1% | 81.6% |
| amazon polarity | 79.6% | 86.6% * | 89.6% | 84.6% |
| yelp | 71.5% | 81.3% | 92.0% * | 76.4% |

#### NAIVE BAYES

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdb 50k | 82.9% * | 73.5% | 79.1% | 76.3% |
| amazon polarity | 69.4% | 79.2% * | 81.3% | 75.4% |
| yelp | 59.8% | 70.9% | 81.3% * | 65.3% |

**Legend**: * = in-domain evaluation

---

## Part 3: Reproducibility

All results can be reproduced via:

```bash
# Prepare immutable data splits (run once)
./scripts/prepare_data.sh

# Train all models
./scripts/train_all_models.sh

# Cross-domain evaluation
./scripts/evaluate_cross_domain.sh

# Regenerate this report
./scripts/generate_report.sh
```

### Model Artifacts

| File | Algorithm | Dataset | Size |
|------|-----------|---------|------|
| amazon_polarity_logistic_regression_model.ser | LOGISTIC_REGRESSION | amazon_polarity | 14.4 MB |
| imdb_50k_logistic_regression_model.ser | LOGISTIC_REGRESSION | imdb_50k | 30.1 MB |
| yelp_logistic_regression_model.ser | LOGISTIC_REGRESSION | yelp | 9.7 MB |
| amazon_polarity_naive_bayes_model.ser | NAIVE_BAYES | amazon_polarity | 14.5 MB |
| imdb_50k_naive_bayes_model.ser | NAIVE_BAYES | imdb_50k | 30.3 MB |
| yelp_naive_bayes_model.ser | NAIVE_BAYES | yelp | 9.9 MB |
| sentiment_model.ser | SVM | amazon_polarity | 15.9 MB |
| amazon_polarity_random_forest_model.ser | RANDOM_FOREST | amazon_polarity | 194.6 MB |
| imdb_50k_random_forest_model.ser | RANDOM_FOREST | imdb_50k | 193.7 MB |
| yelp_random_forest_model.ser | RANDOM_FOREST | yelp | 91.9 MB |
| amazon_polarity_svm_model.ser | SVM | amazon_polarity | 15.9 MB |
| imdb_50k_svm_model.ser | SVM | imdb_50k | 31.6 MB |
| yelp_svm_model.ser | SVM | yelp | 10.3 MB |

---

## Metadata

- **Report Generated**: 2026-01-30T20:40:18Z
- **Git Commit**: 6d7f5e5
- **Java Version**: 24.0.1

