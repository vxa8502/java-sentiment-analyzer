# Java Sentiment Analyzer

Sentiment classification API that handles negated expressions most models get wrong.

```bash
curl -X POST https://java-sentiment-api.onrender.com/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"I am not disappointed"}'
# → positive (83% confidence)
```

| Metric | Value |
|--------|-------|
| Cross-domain accuracy | 88.0% (Amazon → IMDB → Yelp) |
| Latency | <10ms mean, <3ms p99 |
| Negation handling | "not bad" → positive, "not recommend" → negative |

Built in Java to demonstrate ML outside the Python ecosystem. Uses Weka for interpretable models, Spring Boot for the API layer, deployed on Render.

## Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 89.6% |
| **Cross-Domain Average** | 88.0% |
| **F1 Score** | 0.896 |

The production model (SVM trained on Amazon reviews) maintains 85-91% accuracy across different domains:

| Test Domain | Accuracy |
|-------------|----------|
| Amazon (in-domain) | 89.2% |
| IMDB movies | 85.2% |
| Yelp restaurants | 90.9% |

### Model Comparison

| Algorithm | Best In-Domain | Cross-Domain Avg |
|-----------|----------------|------------------|
| **SVM** | 94.0% (Yelp) | **88.0%** |
| Random Forest | 92.0% (Yelp) | 84.6% |
| Logistic Regression | 92.6% (Yelp) | 82.6% |
| Naive Bayes | 82.9% (IMDB) | 76.3% |

### Top Features

| Positive | Weight | Negative | Weight |
|----------|--------|----------|--------|
| excellent | +0.83 | disappointing | -1.12 |
| awesome | +0.79 | worst | -1.09 |
| fantastic | +0.75 | disappointment | -0.99 |
| not_be not_disappointed | +0.64 | boring | -0.96 |
| perfect | +0.67 | disappointed | -0.93 |

The bigram `not_be not_disappointed` shows negation scope handling (Pang et al. 2002).

## Quick Start

```bash
docker build -t sentiment-api .
docker run -p 8080:8080 sentiment-api
```

Test: `curl http://localhost:8080/api/v1/health`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/sentiment/analyze` | POST | Single text classification |
| `/api/v1/sentiment/batch` | POST | Batch classification (up to 100) |
| `/api/v1/model/feature-importance` | GET | Top features by weight |
| `/api/v1/health` | GET | Health check |

## Architecture

| Concern | Solution |
|---------|----------|
| Fault tolerance | Circuit breaker (Resilience4j) |
| Rate limiting | 100 req/min single, 20 req/min batch |
| Thread safety | ReadWriteLock for concurrent inference |
| Reproducibility | SHA-256 checksums on train/test splits |

## Training

```bash
./scripts/train_all_models.sh          # Train 12 models
./scripts/evaluate_cross_domain.sh     # Cross-domain evaluation
./scripts/promote_to_production.sh     # Deploy best model
```

## Technology

| Component | Technology |
|-----------|------------|
| ML | Weka 3.9.6 |
| API | Spring Boot 3.4 |
| Resilience | Resilience4j |
| Runtime | Java 21 |
| Deploy | Docker |

## Testing

| Metric | Value |
|--------|-------|
| Test count | 480 |
| Test coverage | 50% (instruction) |
| CI | GitHub Actions |

Run tests: `mvn test`
Coverage report: `mvn jacoco:report` (output: `target/site/jacoco/index.html`)

## Limitations

- Binary classification only (positive/negative)
- English text only
- Sarcasm detection limited
- Some negation edge cases ("not terrible", "not bad at all") may misclassify

## Roadmap

- [ ] **Per-prediction explanations** - Endpoint returning which features contributed to a specific classification (e.g., `POST /api/v1/model/explain`)

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Training](docs/TRAINING.md)
- [Deployment](docs/DEPLOYMENT.md)
- [Data Cards](docs/data_cards/)

## License

MIT
