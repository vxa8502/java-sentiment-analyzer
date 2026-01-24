# Sentiment Analysis: Final Comprehensive Evaluation Report

**Project**: Cross-Domain Sentiment Classification
**Date**: January 24, 2026
**Author**: Victoria Alabi
**Generated**: Auto-generated from model metadata (do not edit manually)

---

## Executive Summary

This report summarizes the training and evaluation of sentiment analysis models across multiple algorithms and domains.

### Production Model

**Algorithm**: SVM
**Training Dataset**: amazon_polarity (40000 samples)
**Model Size**: 15.1 MB

| Metric | Value |
|--------|-------|
| Test Accuracy | 88.5% |
| Test F1 | 0.885 |
| Test Precision | 0.885 |
| Test Recall | 0.885 |
| ROC-AUC | 0.950 |
| Cross-Domain Avg | 87.9% |

### Production Model Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 4407 (TN) | 596 (FP) |
| **Actual Positive** | 556 (FN) | 4441 (TP) |

### Best Generalizing Model

**Model**: svm-amazon_polarity
**Cross-Domain Average Accuracy**: 87.9%

---

## Part 1: Model Comparison (All 12 Experiments)

| Algorithm | Dataset | Accuracy | F1 | Precision | Recall | Training Time |
|-----------|---------|----------|-----|-----------|--------|---------------|
| SVM | amazon_polarity | 88.5% | 0.885 | 0.885 | 0.885 | 63m 39s |
| SVM | imdb_50k | 89.0% | 0.890 | 0.890 | 0.890 | 167m 23s |
| SVM | yelp | 93.7% | 0.937 | 0.937 | 0.937 | 13m 34s |
| LOGISTIC_REGRESSION | amazon_polarity | 84.0% | 0.840 | 0.841 | 0.840 | 27m 46s |
| LOGISTIC_REGRESSION | imdb_50k | 86.0% | 0.860 | 0.860 | 0.860 | 73m 17s |
| LOGISTIC_REGRESSION | yelp | 92.2% | 0.922 | 0.922 | 0.922 | 26m 25s |
| RANDOM_FOREST | amazon_polarity | 86.9% | 0.869 | 0.869 | 0.869 | 60m 1s |
| RANDOM_FOREST | imdb_50k | 86.9% | 0.869 | 0.869 | 0.869 | 77m 25s |
| RANDOM_FOREST | yelp | 91.2% | 0.912 | 0.913 | 0.912 | 13m 37s |
| NAIVE_BAYES | amazon_polarity | 79.8% | 0.798 | 0.801 | 0.798 | 26m 29s |
| NAIVE_BAYES | imdb_50k | 84.0% | 0.840 | 0.840 | 0.840 | 58m 11s |
| NAIVE_BAYES | yelp | 80.6% | 0.806 | 0.807 | 0.806 | 8m 31s |

---

## Part 2: Cross-Domain Evaluation

Each model was evaluated on all three test domains. Asterisk (*) indicates in-domain evaluation.

#### SVM

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdb 50k | 89.0% * | 81.4% | 83.8% | 82.6% |
| amazon polarity | 84.9% | 88.5% * | 91.0% | 87.9% |
| yelp | 77.8% | 81.5% | 93.7% * | 79.6% |

#### LOGISTIC REGRESSION

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdb 50k | 86.0% * | 77.2% | 79.0% | 78.1% |
| amazon polarity | 80.9% | 84.0% * | 84.6% | 82.8% |
| yelp | 73.4% | 77.0% | 92.2% * | 75.2% |

#### RANDOM FOREST

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdb 50k | 86.9% * | 78.3% | 81.8% | 80.0% |
| amazon polarity | 81.3% | 86.9% * | 89.7% | 85.5% |
| yelp | 70.7% | 81.0% | 91.2% * | 75.9% |

#### NAIVE BAYES

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdb 50k | 84.0% * | 72.8% | 78.0% | 75.4% |
| amazon polarity | 71.0% | 79.8% * | 81.7% | 76.3% |
| yelp | 59.9% | 68.3% | 80.6% * | 64.1% |

**Legend**: * = in-domain evaluation

---

## Part 3: Reproducibility

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
| amazon_polarity_logistic_regression_model.ser | LOGISTIC_REGRESSION | amazon_polarity | 13.6 MB |
| imdb_50k_logistic_regression_model.ser | LOGISTIC_REGRESSION | imdb_50k | 28.6 MB |
| yelp_logistic_regression_model.ser | LOGISTIC_REGRESSION | yelp | 9.2 MB |
| amazon_polarity_naive_bayes_model.ser | NAIVE_BAYES | amazon_polarity | 13.8 MB |
| imdb_50k_naive_bayes_model.ser | NAIVE_BAYES | imdb_50k | 28.8 MB |
| yelp_naive_bayes_model.ser | NAIVE_BAYES | yelp | 9.4 MB |
| sentiment_model.ser | SVM | amazon_polarity | 15.1 MB |
| amazon_polarity_random_forest_model.ser | RANDOM_FOREST | amazon_polarity | 199.3 MB |
| imdb_50k_random_forest_model.ser | RANDOM_FOREST | imdb_50k | 194.5 MB |
| yelp_random_forest_model.ser | RANDOM_FOREST | yelp | 94.0 MB |
| amazon_polarity_svm_model.ser | SVM | amazon_polarity | 15.1 MB |
| imdb_50k_svm_model.ser | SVM | imdb_50k | 30.1 MB |
| yelp_svm_model.ser | SVM | yelp | 9.8 MB |

---

## Metadata

- **Report Generated**: 2026-01-24T17:41:22Z
- **Git Commit**: 61309b6
- **Java Version**: 24.0.1

