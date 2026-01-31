# Model Training Guide

## Prerequisites

- Java 21 (JDK)
- Maven 3.9+
- Datasets in `data/raw/`

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

## Training Pipeline

1. **Train base models:** `./scripts/train_all_models.sh`
2. **Cross-domain evaluation:** `./scripts/evaluate_cross_domain.sh`
3. **Promote best model:** `./scripts/promote_to_production.sh`

Output:
```
models/production/
├── sentiment_model.ser
├── sentiment_model.metadata.json
└── sentiment_model-feature-importance.json
```

## Datasets

| Dataset | Train | Test | Domain |
|---------|-------|------|--------|
| imdb_50k | ~40K | ~10K | Movie reviews |
| amazon_polarity | ~40K | ~10K | Product reviews |
| yelp | ~20K | ~5K | Restaurant reviews |

Binary classification only (positive/negative). Neutral samples filtered.

## Configuration

Key settings in `application.yml`:
```yaml
sentiment:
  training:
    random-seed: 42
  features:
    max-features: 5000
    min-term-freq: 1
    use-tfidf: true
    use-bigrams: true
    mi-selection-threshold: 50000
```

Environment overrides: `SENTIMENT_RANDOM_SEED`, `SENTIMENT_MI_THRESHOLD`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | `export MAVEN_OPTS="-Xmx4g"` |
| Low accuracy (<75%) | Check data loading, verify preprocessing config |
| 3x3 confusion matrix | Neutral samples not filtered - check data pipeline |

Reset everything: `./scripts/reset.sh`
