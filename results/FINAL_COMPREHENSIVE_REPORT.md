# Sentiment Analysis: Final Comprehensive Evaluation Report

**Project**: Cross-Domain Sentiment Classification with Edge Case Analysis
**Date**: January 18, 2026
**Author**: Victoria Alabi
**Generated**: Auto-generated from model metadata (do not edit manually)

---

## Executive Summary

This report summarizes the training and evaluation of sentiment analysis models across multiple algorithms and domains.

### Production Model

**Algorithm**: SVM
**Training Dataset**: amazon_polarity (35999 samples)
**Model Size**: 10.7 MB

| Metric | Value |
|--------|-------|
| Test Accuracy | 89.3% |
| Test F1 | 0.893 |
| Test Precision | 0.893 |
| Test Recall | 0.893 |
| ROC-AUC | 0.893 |
| Cross-Domain Avg | N/A |

### Production Model Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | 5279 (TN) | 670 (FP) |
| **Actual Positive** | 615 (FN) | 5436 (TP) |

### Best Generalizing Model

**Model**: svm-amazon_polarity
**Cross-Domain Average Accuracy**: 84.8%

---

## Part 1: Model Comparison (All 12 Experiments)

| Algorithm | Dataset | Accuracy | F1 | Precision | Recall | Training Time |
|-----------|---------|----------|-----|-----------|--------|---------------|
| SVM | amazon_polarity | 89.8% | 0.898 | 0.898 | 0.898 | 295m 54s |
| SVM | imdb_50k | 88.9% | 0.890 | 0.883 | 0.897 | 119m 54s |
| SVM | yelp | 92.1% | 0.948 | 0.942 | 0.953 | 477m 41s |
| LOGISTIC_REGRESSION | amazon_polarity | 88.0% | 0.880 | 0.880 | 0.880 | 167m 56s |
| Logistic Regression | imdb_50k | 83.9% | 0.838 | 0.844 | 0.831 | 39m 22s |
| Logistic Regression | yelp | 87.3% | 0.914 | 0.928 | 0.900 | 64m 44s |
| Random Forest | amazon_polarity | 87.9% | 0.878 | 0.886 | 0.871 | 94m 23s |
| Random Forest | imdb_50k | 85.9% | 0.860 | 0.853 | 0.867 | 36m 7s |
| Random Forest | yelp | 89.8% | 0.936 | 0.886 | 0.992 | 124m 6s |
| Naive Bayes | amazon_polarity | 79.9% | 0.791 | 0.827 | 0.758 | 69m 6s |
| Naive Bayes | imdb_50k | 83.0% | 0.831 | 0.825 | 0.837 | 27m 41s |
| Naive Bayes | yelp | 65.4% | 0.737 | 0.853 | 0.649 | 61m 38s |

---

## Part 2: Cross-Domain Evaluation

Each model was evaluated on all three test domains. Asterisk (*) indicates in-domain evaluation.

#### SVM

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdu 50k | 87.6% * | 81.4% | 79.8% | 80.6% |
| amazon polarity | 85.0% | 89.8% * | 84.5% | 84.8% |
| yelp | 74.4% | 79.2% | 94.0% * | 76.8% |

#### LOGISTIC REGRESSION

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdu 50k | 82.4% * | 74.6% | 73.0% | 73.8% |
| amazon polarity | 83.5% | 88.0% * | 82.6% | 83.0% |
| yelp | 62.2% | 71.0% | 89.1% * | 66.6% |

#### RANDOM FOREST

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdu 50k | 85.5% * | 78.8% | 79.0% | 78.9% |
| amazon polarity | 80.5% | 87.9% * | 78.3% | 79.4% |
| yelp | 69.1% | 67.3% | 92.5% * | 68.2% |

#### NAIVE BAYES

| Train Domain | IMDB Test | Amazon Test | Yelp Test | Cross-Domain Avg |
|--------------|-----------|-------------|-----------|------------------|
| imdu 50k | 82.6% * | 61.4% | 61.1% | 61.3% |
| amazon polarity | 69.5% | 78.5% * | 67.5% | 68.5% |
| yelp | 51.3% | 62.0% | 61.9% * | 56.6% |

**Legend**: * = in-domain evaluation

---

## Part 3: Edge Case Evaluation

**Total Edge Cases**: 516 curated examples across 4 categories:
- Sarcasm
- Mixed Sentiment
- Negation Heavy
- Domain Jargon

### Production Model Edge Case Performance

| Category | Accuracy |
|----------|----------|
| sarcasm | 62.0% |
| mixed sentiment | 70.0% |
| negation heavy | 52.0% |
| domain jargon | 60.0% |

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
| amazon_polarity_logistic_regression_model_100k.ser | LOGISTIC_REGRESSION | amazon_polarity | 17.2 MB |
| imdb_50k_logistic_regression_model.ser | Logistic Regression | imdb_50k | 19.5 MB |
| yelp_logistic_regression_model.ser | Logistic Regression | yelp | 20.5 MB |
| amazon_polarity_naive_bayes_model.ser | Naive Bayes | amazon_polarity | 17.7 MB |
| imdb_50k_naive_bayes_model.ser | Naive Bayes | imdb_50k | 19.9 MB |
| yelp_naive_bayes_model.ser | Naive Bayes | yelp | 21.1 MB |
| sentiment_model.ser | SVM | amazon_polarity | 10.7 MB |
| amazon_polarity_random_forest_model.ser | Random Forest | amazon_polarity | 318.9 MB |
| imdb_50k_random_forest_model.ser | Random Forest | imdb_50k | 157.2 MB |
| yelp_random_forest_model.ser | Random Forest | yelp | 324.3 MB |
| amazon_polarity_svm_model_100k.ser | SVM | amazon_polarity | 17.0 MB |
| imdb_50k_svm_model.ser | SVM | imdb_50k | 55.2 MB |
| yelp_svm_model.ser | SVM | yelp | 64.8 MB |

---

## Metadata

- **Report Generated**: 2026-01-18T14:02:45Z
- **Git Commit**: 5d07a3d
- **Java Version**: 24.0.1

