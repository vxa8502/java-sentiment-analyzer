# Sentiment Analysis: Final Comprehensive Evaluation Report

**Project**: Cross-Domain Sentiment Classification with Edge Case Analysis
**Date**: January 20, 2026
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
| Cross-Domain Avg | N/A |

### Production Model Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 4404 (TN) | 599 (FP) |
| **Actual Positive** | 560 (FN) | 4437 (TP) |

### Best Generalizing Model

**Model**: svm-amazon_polarity
**Cross-Domain Average Accuracy**: 88.1%

---

## Part 1: Model Comparison (All 12 Experiments)

| Algorithm | Dataset | Accuracy | F1 | Precision | Recall | Training Time |
|-----------|---------|----------|-----|-----------|--------|---------------|
| SVM | amazon_polarity | 88.7% | 0.887 | 0.887 | 0.887 | 84m 19s |
| SVM | imdb_50k | 89.0% | 0.890 | 0.890 | 0.890 | 342m 32s |
| SVM | yelp | 93.7% | 0.937 | 0.937 | 0.937 | 17m 7s |
| LOGISTIC_REGRESSION | amazon_polarity | 85.1% | 0.851 | 0.851 | 0.851 | 23m 30s |
| LOGISTIC_REGRESSION | imdb_50k | 85.2% | 0.852 | 0.852 | 0.852 | 71m 36s |
| LOGISTIC_REGRESSION | yelp | 92.2% | 0.922 | 0.922 | 0.922 | 43m 21s |
| RANDOM_FOREST | amazon_polarity | 86.9% | 0.869 | 0.869 | 0.869 | 62m 25s |
| RANDOM_FOREST | imdb_50k | 86.5% | 0.865 | 0.865 | 0.865 | 63m 32s |
| RANDOM_FOREST | yelp | 91.7% | 0.917 | 0.917 | 0.917 | 20m 3s |
| NAIVE_BAYES | amazon_polarity | 80.9% | 0.809 | 0.811 | 0.809 | 21m 55s |
| NAIVE_BAYES | imdb_50k | 83.9% | 0.839 | 0.839 | 0.839 | 84m 35s |
| NAIVE_BAYES | yelp | 80.7% | 0.807 | 0.808 | 0.807 | 7m 47s |

---

## Part 2: Cross-Domain Evaluation

Each model was evaluated on all three test domains. Asterisk (*) indicates in-domain evaluation.

#### SVM

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdu 50k | 94.0% * | 81.9% | 84.0% | 83.0% |
| amazon polarity | 85.1% | 88.7% * | 91.1% | 88.1% |
| yelp | 78.8% | 82.1% | 93.7% * | 80.5% |

#### LOGISTIC REGRESSION

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdu 50k | 85.2% * | 77.8% | 79.7% | 78.8% |
| amazon polarity | 81.8% | 85.1% * | 85.9% | 83.9% |
| yelp | 74.8% | 77.9% | 92.2% * | 76.4% |

#### RANDOM FOREST

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdu 50k | 86.5% * | 78.1% | 81.2% | 79.7% |
| amazon polarity | 81.3% | 97.4% * | 89.1% | 85.2% |
| yelp | 71.0% | 81.3% | 91.7% * | 76.2% |

#### NAIVE BAYES

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdu 50k | 83.5% * | 73.1% | 77.9% | 75.5% |
| amazon polarity | 72.1% | 80.9% * | 82.0% | 77.1% |
| yelp | 59.8% | 69.8% | 80.7% * | 64.8% |

**Legend**: * = in-domain evaluation

---

## Part 3: Edge Case Evaluation

**Total Edge Cases**: 419 curated examples across 4 categories:
- Sarcasm
- Mixed Sentiment
- Negation Heavy
- Domain Jargon

---

## Part 4: Reproducibility

All results can be reproduced via:

```bash
# Train all models
./scripts/train_all_models.sh

# Cross-domain evaluation
./scripts/evaluate_cross_domain.sh

# Edge case evaluation
./scripts/evaluate_edge_cases.sh all

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
| imdb_50k_random_forest_model.ser | RANDOM_FOREST | imdb_50k | 197.6 MB |
| yelp_random_forest_model.ser | RANDOM_FOREST | yelp | 94.8 MB |
| amazon_polarity_svm_model.ser | SVM | amazon_polarity | 14.4 MB |
| imdb_50k_svm_model.ser | SVM | imdb_50k | 31.0 MB |
| yelp_svm_model.ser | SVM | yelp | 9.6 MB |

---

## Metadata

- **Report Generated**: 2026-01-20T19:53:08Z
- **Git Commit**: f8b57b2
- **Java Version**: 24.0.1

