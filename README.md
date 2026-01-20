# Java Sentiment Analyzer

A multi-algorithm sentiment analysis system in Java, built with Spring Boot and Weka.

## Features

- **REST API:** Simple endpoints for single and batch sentiment analysis
- **Multiple Algorithms:** SVM, Naive Bayes, Random Forest, Logistic Regression
- **Cross-Domain Evaluation:** Models trained and tested across 3 domains (movies, products, restaurants)
- **Edge Case Testing:** Curated test suite for sarcasm, negation, mixed sentiment, jargon
- **Docker Support:** Production-ready containerization

## Quick Start

```bash
# Clone and run
git clone <repo-url>
cd java-sentiment-analyzer
docker-compose up
```

The API starts on http://localhost:8080.

**Test it:**
```bash
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"This product is amazing!"}'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.92,
  "processingTimeMs": 15
}
```

## API Endpoints

### Single Text Analysis
```bash
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"This product is amazing!"}'
```

### Batch Analysis
```bash
curl -X POST http://localhost:8080/api/v1/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["Great product!", "Terrible quality.", "It works."]}'
```

### Health Check
```bash
curl http://localhost:8080/api/v1/health
```

## Training Pipeline

Train all 12 models (4 algorithms x 3 datasets):
```bash
./scripts/train_all_models.sh
```

Run cross-domain evaluation:
```bash
./scripts/evaluate_cross_domain.sh
```

Evaluate edge cases:
```bash
./scripts/evaluate_edge_cases.sh all
```

Promote best model to production:
```bash
./scripts/promote_to_production.sh
```

See [docs/TRAINING.md](docs/TRAINING.md) for detailed training documentation.

## Datasets

| Dataset | Train | Test | Domain |
|---------|-------|------|--------|
| IMDB 50K | ~40K | ~10K | Movie reviews |
| Amazon Polarity | ~40K | ~10K | Product reviews |
| Yelp | ~20K | ~5K | Restaurant reviews |

All datasets are capped at 50K samples with stratified sampling to preserve class distribution. Binary classification only (positive/negative).

## Technology Stack

- **Java 21**
- **Spring Boot 3.4**
- **Weka 3.9.6** (ML algorithms)
- **Maven** (build)
- **Resilience4j** (rate limiting)
- **Docker** (deployment)

## Prerequisites

- **Docker** (for quick start)
- **Java 21** (for local development)
- **Maven 3.9+** (for building from source)

## Project Structure

```
├── src/main/java/sentiment/
│   ├── api/                 # REST controllers
│   ├── models/              # ML classifiers
│   ├── preprocessing/       # Text preprocessing
│   ├── training/            # Model training
│   └── evaluation/          # Cross-domain & edge case evaluation
├── scripts/                 # Training and evaluation scripts
├── models/production/       # Deployed model
└── docs/                    # Documentation
```

## Documentation

- [TRAINING.md](docs/TRAINING.md) - Model training and evaluation
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) - Docker deployment and monitoring
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design

## Configuration

Key settings can be customized via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTIMENT_RANDOM_SEED` | 42 | Random seed for reproducibility |
| `SENTIMENT_MI_THRESHOLD` | 50000 | MI feature selection threshold |
| `SENTIMENT_DATA_IMDB` | `data/raw/imdb_50k/IMDB Dataset.csv` | IMDB dataset path |
| `SENTIMENT_DATA_AMAZON` | `data/raw/amazon_polarity/train.csv` | Amazon dataset path |
| `SENTIMENT_DATA_YELP` | `data/raw/yelp/yelp_reviews.csv` | Yelp dataset path |

**Rate Limits** (per minute):
- Single analysis: 100 requests
- Batch analysis: 20 requests
- Model comparison: 2 requests (per 5 min)

See `application.yml` for all configuration options.

## Limitations

- **Binary classifier only**: Predicts `positive` or `negative` (no `neutral`)
- Neutral/ambiguous text will be classified with lower confidence
- For 3-class prediction, retrain with a dataset containing neutral labels

## License

MIT License. See [LICENSE](LICENSE) for details.
