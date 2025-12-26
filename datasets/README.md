# Datasets

## Primary Datasets

### Sources
- **IMDB 50K**: http://ai.stanford.edu/~amaas/data/sentiment/
  - 25K train, 10K test (movie reviews)
- **Amazon Polarity**: https://huggingface.co/datasets/amazon_polarity
  - 100K train, 20K test (product reviews)
- **Yelp**: https://www.yelp.com/dataset
  - 100K train, 20K test (restaurant reviews)

### Known Biases
- **IMDB**: Drama-heavy (38%), excludes neutral reviews (5-6 star ratings removed)
- **Amazon**: Verified purchase bias, electronics overrepresented (45%)
- **Yelp**: Restaurant-focused, potential spam contamination

## Edge Case Collection (200 Real Failures)

**STATUS**:  COMPLETE - 200 real model failures extracted

### Categories
- **sarcasm.csv**: 50 examples - Positive words with sarcastic/negative context
- **mixed_sentiment.csv**: 50 examples - Contains "but" with both positive and negative words
- **negation_heavy.csv**: 50 examples - Multiple negation words (not, never, no)
- **domain_jargon.csv**: 50 examples - Technical terminology, long reviews (>300 words)

### Collection Method
1. Extracted ~18,000 prediction errors from 3 models (SVM, Naive Bayes, Random Forest)
2. Filtered to high-confidence errors only (confidence  0.65)
3. Applied linguistic pattern detection (not random sampling)
4. Sampled 50 per category with diverse text lengths
5. Total: 200 edge cases from 3,845 candidate failures

**These are REAL failures** - actual examples where trained models got wrong.

## Download Scripts

See `scripts/download_datasets.sh` for automated dataset downloads.
