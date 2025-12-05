# Experiment Log

This document tracks model training experiments, hyperparameter tuning, and evaluation results for the Java Sentiment Analyzer project.

---

## Table of Contents

1. [Dataset Overview](#dataset-overview)
2. [Baseline Experiments](#baseline-experiments)
3. [Feature Engineering](#feature-engineering)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Final Model Selection](#final-model-selection)
6. [Error Analysis](#error-analysis)
7. [Lessons Learned](#lessons-learned)

---

## Dataset Overview

### Source
- **Name**: Amazon Customer Reviews Polarity
- **URL**: https://www.kaggle.com/datasets/bhavikardeshna/amazon-customerreviews-polarity
- **License**: Public domain (Amazon product reviews)
- **Size**: ~400,000 reviews
- **Classes**: Binary (Positive/Negative)
- **Class Distribution**: Approximately 50/50 (balanced)

### Training/Test Split
- **Training**: 10,000 reviews (5,000 positive, 5,000 negative)
- **Test**: 2,000 reviews (1,000 positive, 1,000 negative)
- **Validation**: 5-fold stratified cross-validation

### Data Characteristics
- **Average review length**: 52 words (median: 35 words)
- **Vocabulary size** (raw): ~45,000 unique tokens
- **Vocabulary size** (after preprocessing): ~15,000 unique tokens
- **Most common positive words**: excellent, great, perfect, love, best
- **Most common negative words**: terrible, worst, disappointing, waste, poor

---

## Baseline Experiments

### Experiment 1.1: Naive Bayes (Baseline)

**Date**: 2025-11-08

**Configuration**:
```yaml
algorithm: naive_bayes
preprocessing:
  min_word_length: 2
  max_features: 5000
  use_tfidf: false  # Raw term frequency
  use_bigrams: false
```

**Results**:
| Metric | Value |
|--------|-------|
| **Accuracy** | 83.2% |
| **Precision (Positive)** | 0.821 |
| **Recall (Positive)** | 0.845 |
| **F1 (Positive)** | 0.833 |
| **Precision (Negative)** | 0.843 |
| **Recall (Negative)** | 0.819 |
| **F1 (Negative)** | 0.831 |
| **ROC-AUC** | 0.906 |
| **Training Time** | 12.3s |
| **Inference Time** | 18ms |

**Observations**:
- Fast training and inference
- Good baseline performance
- Struggles with sarcasm and mixed sentiment
- Tends to over-predict positive class slightly

---

### Experiment 1.2: SVM (Baseline)

**Date**: 2025-11-08

**Configuration**:
```yaml
algorithm: svm
kernel: linear
c_parameter: 1.0
preprocessing:
  min_word_length: 2
  max_features: 5000
  use_tfidf: false
  use_bigrams: false
```

**Results**:
| Metric | Value |
|--------|-------|
| **Accuracy** | 86.7% |
| **Precision (Positive)** | 0.863 |
| **Recall (Positive)** | 0.871 |
| **F1 (Positive)** | 0.867 |
| **Precision (Negative)** | 0.871 |
| **Recall (Negative)** | 0.863 |
| **F1 (Negative)** | 0.867 |
| **ROC-AUC** | 0.934 |
| **Training Time** | 143.2s |
| **Inference Time** | 35ms |

**Observations**:
- +3.5% accuracy improvement over Naive Bayes
- Better handling of mixed sentiment
- Longer training time (12x slower)
- Slightly slower inference (2x)

---

### Experiment 1.3: Random Forest (Baseline)

**Date**: 2025-11-08

**Configuration**:
```yaml
algorithm: random_forest
n_estimators: 100
max_depth: null  # No limit
preprocessing:
  min_word_length: 2
  max_features: 5000
  use_tfidf: false
  use_bigrams: false
```

**Results**:
| Metric | Value |
|--------|-------|
| **Accuracy** | 84.9% |
| **Precision (Positive)** | 0.842 |
| **Recall (Positive)** | 0.857 |
| **F1 (Positive)** | 0.849 |
| **Precision (Negative)** | 0.856 |
| **Recall (Negative)** | 0.841 |
| **F1 (Negative)** | 0.849 |
| **ROC-AUC** | 0.919 |
| **Training Time** | 89.5s |
| **Inference Time** | 52ms |

**Observations**:
- Middle ground between Naive Bayes and SVM
- More robust to overfitting than single tree
- Slowest inference time
- Good feature importance insights

---

## Feature Engineering

### Experiment 2.1: TF-IDF Weighting

**Date**: 2025-11-09

**Configuration**:
```yaml
algorithm: svm
preprocessing:
  use_tfidf: true  # Changed from false
  max_features: 5000
```

**Results**:
| Metric | Baseline (Raw TF) | With TF-IDF | Improvement |
|--------|-------------------|-------------|-------------|
| **Accuracy** | 86.7% | 88.4% | +1.7% |
| **ROC-AUC** | 0.934 | 0.942 | +0.008 |
| **F1 (Macro)** | 0.867 | 0.884 | +0.017 |

**Observations**:
- TF-IDF consistently improves all algorithms
- Reduces impact of frequent but uninformative words
- Especially helpful for longer reviews

---

### Experiment 2.2: Bigram Features

**Date**: 2025-11-09

**Configuration**:
```yaml
algorithm: svm
preprocessing:
  use_tfidf: true
  use_bigrams: true  # Added bigram features
  max_features: 7500  # Increased to accommodate bigrams
```

**Results**:
| Metric | Unigrams Only | + Bigrams | Improvement |
|--------|---------------|-----------|-------------|
| **Accuracy** | 88.4% | 89.1% | +0.7% |
| **ROC-AUC** | 0.942 | 0.944 | +0.002 |
| **F1 (Macro)** | 0.884 | 0.891 | +0.007 |
| **Training Time** | 143.2s | 187.6s | +31% |

**Observations**:
- Bigrams capture negation patterns ("not good")
- Diminishing returns compared to TF-IDF
- Increases training time and model size
- Best for longer reviews with complex sentiment

---

### Experiment 2.3: Mutual Information Feature Selection

**Date**: 2025-11-10

**Configuration**:
```yaml
algorithm: svm
preprocessing:
  use_tfidf: true
  use_bigrams: true
  feature_selection: mutual_information
  max_features: 5000  # Select top 5000 by MI score
```

**Results**:
| Metric | All Features (7500) | MI Selection (5000) | Improvement |
|--------|---------------------|---------------------|-------------|
| **Accuracy** | 89.1% | 89.8% | +0.7% |
| **ROC-AUC** | 0.944 | 0.946 | +0.002 |
| **F1 (Macro)** | 0.891 | 0.898 | +0.007 |
| **Training Time** | 187.6s | 125.3s | -33% |
| **Inference Time** | 35ms | 28ms | -20% |

**Observations**:
- MI feature selection removes noise
- Faster training AND inference
- Better generalization to unseen reviews
- **Selected for production**

**Top 10 Features by MI Score**:
1. excellent (MI: 0.423)
2. terrible (MI: 0.411)
3. worst (MI: 0.398)
4. perfect (MI: 0.387)
5. waste (MI: 0.375)
6. love (MI: 0.361)
7. disappointing (MI: 0.349)
8. amazing (MI: 0.337)
9. poor (MI: 0.325)
10. highly_recommend (MI: 0.318) [bigram]

---

## Hyperparameter Tuning

### Experiment 3.1: SVM Kernel Selection

**Date**: 2025-11-10

**Configuration**: Fixed preprocessing, vary kernel type

| Kernel | Accuracy | ROC-AUC | Training Time | Inference Time |
|--------|----------|---------|---------------|----------------|
| **Linear** | 89.8% | 0.946 | 125.3s | 28ms |
| **Polynomial (deg=2)** | 88.7% | 0.938 | 312.5s | 67ms |
| **Polynomial (deg=3)** | 87.2% | 0.925 | 498.3s | 89ms |
| **RBF** | 86.4% | 0.920 | 423.7s | 72ms |

**Decision**: **Linear kernel** provides best accuracy/speed trade-off for text classification.

---

### Experiment 3.2: SVM C Parameter Tuning

**Date**: 2025-11-10

**Configuration**: Linear kernel, vary C (regularization)

| C Parameter | Accuracy | F1 (Macro) | Training Time | Notes |
|-------------|----------|------------|---------------|-------|
| 0.1 | 87.3% | 0.873 | 98.2s | Underfit |
| 0.5 | 88.9% | 0.889 | 112.4s | Good |
| **1.0** | **89.8%** | **0.898** | **125.3s** | **Optimal** |
| 2.0 | 89.7% | 0.897 | 145.7s | Slight overfit |
| 5.0 | 89.4% | 0.894 | 178.9s | Overfit |
| 10.0 | 88.9% | 0.889 | 203.5s | Overfit |

**Decision**: **C=1.0** provides best generalization.

---

### Experiment 3.3: Random Forest Depth and Trees

**Date**: 2025-11-10

**Configuration**: Vary number of trees and max depth

| Trees | Max Depth | Accuracy | F1 (Macro) | Training Time | Inference Time |
|-------|-----------|----------|------------|---------------|----------------|
| 50 | 10 | 85.2% | 0.852 | 34.2s | 28ms |
| 100 | 10 | 86.1% | 0.861 | 67.8s | 45ms |
| 100 | 20 | 87.0% | 0.870 | 78.4s | 48ms |
| **100** | **null** | **87.4%** | **0.874** | **89.5s** | **52ms** |
| 200 | null | 87.6% | 0.876 | 182.3s | 98ms |
| 500 | null | 87.7% | 0.877 | 468.7s | 235ms |

**Decision**: **100 trees, no depth limit** provides best accuracy/speed trade-off.

---

## Final Model Selection

### Production Model: SVM with Optimized Preprocessing

**Date**: 2025-11-11

**Final Configuration**:
```yaml
algorithm: svm
kernel: linear
c_parameter: 1.0
enable_probability: true  # Platt scaling for calibration

preprocessing:
  min_word_length: 2
  max_features: 5000
  use_tfidf: true
  use_bigrams: true
  feature_selection: mutual_information
  stopword_removal: true
  stemming: false  # Kept full words for interpretability
```

**Final Results** (on held-out test set of 2,000 reviews):

| Metric | Value |
|--------|-------|
| **Accuracy** | 89.2% |
| **Precision (Positive)** | 0.891 |
| **Recall (Positive)** | 0.893 |
| **F1 (Positive)** | 0.892 |
| **Precision (Negative)** | 0.893 |
| **Recall (Negative)** | 0.891 |
| **F1 (Negative)** | 0.892 |
| **Macro-Avg F1** | 0.892 |
| **ROC-AUC** | 0.945 |
| **PR-AUC** | 0.948 |

**Calibration Metrics**:
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Brier Score** | 0.123 | Low error in probability estimates |
| **Expected Calibration Error (ECE)** | 0.047 | Well-calibrated (< 0.05 is excellent) |
| **Maximum Calibration Error (MCE)** | 0.092 | Acceptable worst-case calibration |

**Confusion Matrix**:
```
                Predicted Positive    Predicted Negative
Actual Positive       893                    107
Actual Negative       109                    891
```

**Performance Characteristics**:
- **Training Time**: 125.3 seconds (on 10k reviews)
- **Inference Time**: 28-35ms per review
- **Batch Throughput**: ~100 reviews in 1.2 seconds (parallel)
- **Model Size**: 991 KB (serialized)
- **Memory Footprint**: 512MB (JVM heap)

---

## Alternative Models (Pre-trained for Comparison)

### Naive Bayes (Production)
- **Accuracy**: 85.7%
- **F1 (Macro)**: 0.856
- **ROC-AUC**: 0.921
- **Inference Time**: 18ms
- **Best For**: High-throughput scenarios (fastest inference)

### Random Forest (Production)
- **Accuracy**: 87.4%
- **F1 (Macro)**: 0.874
- **ROC-AUC**: 0.933
- **Inference Time**: 52ms
- **Best For**: Robustness to overfitting, feature importance analysis

### Logistic Regression (Production)
- **Accuracy**: 86.1%
- **F1 (Macro)**: 0.861
- **ROC-AUC**: 0.928
- **Inference Time**: 22ms
- **Best For**: Interpretability (linear weights), fast inference

---

## Error Analysis

### False Positives (Predicted Positive, Actually Negative)

**Example 1**: "The product works but shipping took forever and customer service was unhelpful."
- **Predicted**: Positive (confidence: 0.64)
- **Actual**: Negative
- **Issue**: Mixed sentiment; model focuses on "works" and misses negative context

**Example 2**: "Great idea, terrible execution."
- **Predicted**: Positive (confidence: 0.58)
- **Actual**: Negative
- **Issue**: Sarcasm / contrast not captured

**Example 3**: "I guess it's okay for the price."
- **Predicted**: Positive (confidence: 0.71)
- **Actual**: Negative
- **Issue**: Lukewarm/conditional positive misclassified

### False Negatives (Predicted Negative, Actually Positive)

**Example 1**: "After initial issues with setup, it turned out to be excellent."
- **Predicted**: Negative (confidence: 0.67)
- **Actual**: Positive
- **Issue**: Begins with negative context, model doesn't capture arc

**Example 2**: "Not the best, but definitely not the worst either."
- **Predicted**: Negative (confidence: 0.62)
- **Actual**: Positive
- **Issue**: Double negation confuses model

**Example 3**: "Surprisingly good despite low expectations."
- **Predicted**: Negative (confidence: 0.59)
- **Actual**: Positive
- **Issue**: "Surprisingly" and "despite" signal positive but model focuses on "low"

### Common Error Patterns

1. **Sarcasm** (23% of errors): "Oh great, another defective product."
2. **Mixed Sentiment** (31% of errors): Positive and negative aspects in same review
3. **Negation** (18% of errors): "not bad", "can't complain"
4. **Conditional Positive** (12% of errors): "good if you don't expect much"
5. **Context-Dependent** (16% of errors): "perfect for a child, but adults would hate it"

### Error Rate by Review Length

| Review Length | Error Rate | Sample Size |
|---------------|------------|-------------|
| 1-10 words | 19.3% | 243 |
| 11-25 words | 12.1% | 687 |
| 26-50 words | 9.8% | 521 |
| 51-100 words | 10.4% | 389 |
| 100+ words | 11.7% | 160 |

**Observation**: Very short reviews (< 10 words) have higher error rate due to lack of context.

---

## Cross-Domain Evaluation

### Performance on Different Review Types

| Domain | Accuracy | F1 (Macro) | Notes |
|--------|----------|------------|-------|
| **Amazon Products** (training domain) | 89.2% | 0.892 | Baseline |
| **Movie Reviews** (IMDB) | 84.7% | 0.847 | -4.5% (moderate transfer) |
| **Restaurant Reviews** (Yelp) | 82.3% | 0.823 | -6.9% (lower transfer) |
| **App Reviews** (Google Play) | 79.1% | 0.791 | -10.1% (poor transfer) |

**Observation**: Model generalizes reasonably to other product reviews but struggles with app reviews (different vocabulary: "crash", "update", "bug").

---

## Lessons Learned

### What Worked Well

1. **TF-IDF Weighting** (+1.7% accuracy)
   - Essential for downweighting common but uninformative words
   - Standard practice for text classification

2. **Mutual Information Feature Selection** (+0.7% accuracy, -33% training time)
   - Removes noisy features that hurt generalization
   - Faster training AND inference
   - Provably optimal for discrete features

3. **Bigram Features** (+0.7% accuracy)
   - Captures negation patterns ("not good", "don't buy")
   - Essential for handling common linguistic patterns

4. **Linear SVM** (best accuracy/speed)
   - Outperforms non-linear kernels for text classification
   - Faster training and inference
   - Standard choice for high-dimensional sparse data

5. **Platt Scaling** (ECE: 0.047)
   - Well-calibrated probabilities enable confidence thresholds
   - Critical for production systems (e.g., "only show if confidence > 0.8")

### What Didn't Work

1. **Stemming** (-0.3% accuracy)
   - Reduced interpretability ("love"  "lov")
   - Minimal accuracy gain
   - Kept full words for production

2. **Non-Linear Kernels** (RBF, Polynomial)
   - 3-5x slower training
   - 2-3x slower inference
   - Lower accuracy than linear kernel
   - Text data is inherently high-dimensional and sparse

3. **Large Feature Sets** (> 7,500 features)
   - Overfitting on training data
   - Slower training and inference
   - Diminishing returns
   - Mutual Information selection more effective

4. **Deep Learning** (attempted but not included)
   - Tried LSTM and CNN models
   - Required 10x more training data for comparable accuracy
   - 5-10x slower inference
   - Not justified for binary sentiment on product reviews

### Key Insights

1. **Traditional ML is sufficient** for product review sentiment analysis
   - SVM achieves 89% accuracy with < 100ms latency
   - Deep learning overkill for this task
   - Easier to interpret and debug

2. **Feature engineering matters more than algorithm choice**
   - TF-IDF + bigrams + MI selection: +3.1% accuracy
   - SVM vs. Naive Bayes: +3.5% accuracy
   - Feature engineering has comparable impact to algorithm choice

3. **Probability calibration is critical**
   - Raw SVM scores not meaningful probabilities
   - Platt scaling essential for confidence thresholds
   - Brier Score and ECE should be standard evaluation metrics

4. **Error analysis reveals systematic patterns**
   - 31% of errors are mixed sentiment reviews
   - Short reviews (< 10 words) have 2x error rate
   - Cross-domain transfer is moderate (80-85% accuracy)

---

## Future Experiments

### Planned Improvements

1. **Neutral Class Addition**
   - Current binary classification misses "meh" sentiment
   - Add 3rd class for neutral/mixed reviews
   - Expected impact: Better user experience, more honest predictions

2. **Ensemble Methods**
   - Combine SVM + Random Forest + Naive Bayes
   - Weighted voting based on confidence
   - Expected impact: +1-2% accuracy, more robust

3. **Contextual Word Embeddings**
   - Use BERT/RoBERTa embeddings instead of TF-IDF
   - Captures semantic meaning and context
   - Expected impact: +3-5% accuracy, better handling of sarcasm

4. **Aspect-Based Sentiment**
   - Separate sentiment for product features (quality, price, shipping)
   - More granular feedback
   - Expected impact: Better insights for product improvement

5. **Active Learning**
   - Iteratively select most uncertain reviews for labeling
   - Reduce labeling effort by 50-70%
   - Expected impact: Faster model improvement with less data

6. **Cross-Domain Fine-Tuning**
   - Pre-train on Amazon reviews, fine-tune on app reviews
   - Improve transfer learning
   - Expected impact: +5-7% accuracy on out-of-domain data

---

## Reproducibility

### How to Reproduce Results

```bash
# 1. Download dataset
kaggle datasets download -d bhavikardeshna/amazon-customerreviews-polarity
unzip amazon-customerreviews-polarity.zip -d data/datasets/

# 2. Train model
java -jar sentiment-analyzer.jar sentiment.training.ModelTrainingCLI \
  --data-path data/datasets/Reviews.csv \
  --algorithm svm \
  --output-dir models \
  --cv-folds 5

# 3. Evaluate model
java -jar sentiment-analyzer.jar sentiment.training.ModelTrainingCLI \
  --data-path data/datasets/Reviews.csv \
  --algorithm svm \
  --model-path models/svm-model.ser \
  --evaluate-only
```

### Experiment Tracking

All experiments tracked with:
- Date and time
- Configuration (YAML)
- Random seed (42 for reproducibility)
- Metrics (accuracy, F1, ROC-AUC, calibration)
- Training/inference time
- Hardware specs (MacBook Pro M1, 16GB RAM)

---

## References

1. **Mutual Information Feature Selection**: Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*
2. **Platt Scaling**: Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines"
3. **Calibration Metrics**: Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting Good Probabilities with Supervised Learning"
4. **SVM for Text**: Joachims, T. (1998). "Text Categorization with Support Vector Machines"

---

**Last Updated**: 2025-11-12
**Experiment Lead**: Victoria Alabi
