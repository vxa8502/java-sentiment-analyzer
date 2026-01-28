# Sentiment Analysis: Final Comprehensive Evaluation Report

**Project**: Cross-Domain Sentiment Classification
**Date**: January 28, 2026
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
| Test Accuracy | 89.4% |
| Test F1 | 0.894 |
| Test Precision | 0.894 |
| Test Recall | 0.894 |
| ROC-AUC | 0.958 |
| Cross-Domain Avg | 88.2% |

### Production Model Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 4492 (TN) | 511 (FP) |
| **Actual Positive** | 545 (FN) | 4452 (TP) |

### Best Generalizing Model

**Model**: svm-amazon_polarity
**Cross-Domain Average Accuracy**: 88.2%

---

## Part 1: Model Comparison (All 12 Experiments)

| Algorithm | Dataset | Accuracy | F1 | Precision | Recall | Training Time |
|-----------|---------|----------|-----|-----------|--------|---------------|
| SVM | amazon_polarity | 89.3% | 0.893 | 0.893 | 0.893 | 87m 34s |
| SVM | imdb_50k | 89.0% | 0.890 | 0.890 | 0.890 | 130m 22s |
| SVM | yelp | 93.8% | 0.938 | 0.938 | 0.938 | 16m 43s |
| LOGISTIC_REGRESSION | amazon_polarity | 85.0% | 0.850 | 0.851 | 0.850 | 57m 22s |
| LOGISTIC_REGRESSION | imdb_50k | 85.7% | 0.857 | 0.857 | 0.857 | 92m 1s |
| LOGISTIC_REGRESSION | yelp | 92.5% | 0.925 | 0.925 | 0.925 | 21m 8s |
| RANDOM_FOREST | amazon_polarity | 87.0% | 0.870 | 0.870 | 0.870 | 120m 23s |
| RANDOM_FOREST | imdb_50k | 86.5% | 0.865 | 0.865 | 0.865 | 61m 22s |
| RANDOM_FOREST | yelp | 91.6% | 0.916 | 0.917 | 0.916 | 25m 34s |
| NAIVE_BAYES | amazon_polarity | 80.3% | 0.802 | 0.806 | 0.803 | 67m 39s |
| NAIVE_BAYES | imdb_50k | 82.3% | 0.823 | 0.824 | 0.823 | 71m 58s |
| NAIVE_BAYES | yelp | 81.3% | 0.813 | 0.816 | 0.813 | 17m 54s |

---

## Part 2: Cross-Domain Evaluation

Each model was evaluated on all three test domains. Asterisk (*) indicates in-domain evaluation.

#### SVM

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdb 50k | 89.0% * | 83.2% | 85.1% | 84.2% |
| amazon polarity | 85.3% | 89.3% * | 91.2% | 88.2% |
| yelp | 78.9% | 82.1% | 93.8% * | 80.5% |

#### LOGISTIC REGRESSION

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdb 50k | 85.7% * | 77.9% | 79.1% | 78.5% |
| amazon polarity | 81.2% | 85.0% * | 85.3% | 83.3% |
| yelp | 76.3% | 77.7% | 92.5% * | 77.0% |

#### RANDOM FOREST

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdb 50k | 86.5% * | 78.5% | 83.4% | 80.9% |
| amazon polarity | 80.8% | 87.0% * | 89.6% | 85.2% |
| yelp | 70.5% | 81.5% | 91.6% * | 76.0% |

#### NAIVE BAYES

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdb 50k | 82.3% * | 73.8% | 78.8% | 76.3% |
| amazon polarity | 69.7% | 80.3% * | 81.4% | 75.6% |
| yelp | 59.4% | 71.7% | 81.3% * | 65.5% |

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
| imdb_50k_naive_bayes_model.ser | NAIVE_BAYES | imdb_50k | 30.2 MB |
| yelp_naive_bayes_model.ser | NAIVE_BAYES | yelp | 9.8 MB |
| sentiment_model.ser | SVM | amazon_polarity | 15.9 MB |
| amazon_polarity_random_forest_model.ser | RANDOM_FOREST | amazon_polarity | 195.4 MB |
| imdb_50k_random_forest_model.ser | RANDOM_FOREST | imdb_50k | 194.0 MB |
| yelp_random_forest_model.ser | RANDOM_FOREST | yelp | 91.7 MB |
| amazon_polarity_svm_model.ser | SVM | amazon_polarity | 15.9 MB |
| imdb_50k_svm_model.ser | SVM | imdb_50k | 31.6 MB |
| yelp_svm_model.ser | SVM | yelp | 10.3 MB |

---

## Metadata

- **Report Generated**: 2026-01-28T09:10:56Z
- **Git Commit**: 4eea486
- **Java Version**: 24.0.1

