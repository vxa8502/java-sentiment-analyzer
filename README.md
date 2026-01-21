# Java Sentiment Analyzer

**88% accuracy across 3 domains** | **<50ms latency** | **Production-ready REST API**

A sentiment analysis system that classifies text as positive or negative, built in Java with Spring Boot and Weka. Trained and evaluated across movie reviews, product reviews, and restaurant reviews to ensure real-world generalization.

---

## Why Java for ML?

Most ML projects live in Python notebooks. This one ships as a production service.

I built this in Java to demonstrate:

- **Enterprise deployment patterns**: Spring Boot containerization, health checks, circuit breakers
- **Production reliability**: Type-safe pipelines catch errors at compile time, not runtime
- **Weka's maturity**: 20+ years of peer-reviewed ML research, proven accuracy on text classification
- **JVM performance**: Parallel preprocessing and inference via ReadWriteLock concurrency

The result: a sentiment classifier that runs anywhere Docker runs, handles 1,000+ requests/minute, and maintains consistent accuracy across different text domains.

---

## Results

### Production Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 88.4% |
| **Cross-Domain Average** | 88.2% |
| **F1 Score** | 0.884 |
| **Latency** | 10-50ms |
| **Throughput** | ~1,000 req/min |

The production model (SVM trained on Amazon product reviews) was selected for **best generalization**—it maintains 84-91% accuracy when applied to completely different domains:

| Test Domain | Accuracy |
|-------------|----------|
| Amazon (in-domain) | 88.4% |
| IMDB movies | 84.8% |
| Yelp restaurants | 91.5% |

### Model Comparison (12 experiments)

Trained 4 algorithms across 3 domains to find the best trade-off between accuracy and generalization:

| Algorithm | Best In-Domain | Cross-Domain Avg | Notes |
|-----------|----------------|------------------|-------|
| **SVM** | 94.5% (Yelp) | **88.2%** | Best generalizing |
| Random Forest | 92.5% (Yelp) | 85.3% | Largest model files |
| Logistic Regression | 92.3% (Yelp) | 83.4% | Fastest training |
| Naive Bayes | 83.9% (IMDB) | 76.2% | Fastest inference |

Full results: [Model Comparison Report](results/FINAL_COMPREHENSIVE_REPORT.md)

---

## Quick Start

```bash
docker-compose up
```

The API starts at http://localhost:8080. Test it:

```bash
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"This product exceeded my expectations!"}'
```

```json
{
  "sentiment": "positive",
  "confidence": 0.92,
  "processingTimeMs": 15
}
```

---

## API Endpoints

### Single Analysis
```bash
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Terrible customer service, would not recommend."}'
```

### Batch Analysis (up to 100 texts)
```bash
curl -X POST http://localhost:8080/api/v1/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["Great product!", "Disappointing quality.", "It works fine."]}'
```

### Feature Importance
```bash
curl http://localhost:8080/api/v1/sentiment/feature-importance
```

### Health Check
```bash
curl http://localhost:8080/api/v1/health
```

See [examples/](examples/) for Python client and more curl examples.

---

## Architecture Highlights

This isn't a notebook experiment—it's a deployable system with production patterns:

| Concern | Solution |
|---------|----------|
| **Fault tolerance** | Circuit breaker pattern (Resilience4j) prevents cascade failures |
| **Rate limiting** | 100 req/min single, 20 req/min batch |
| **Thread safety** | ReadWriteLock allows concurrent inference during training |
| **Reproducibility** | SHA-256 checksums lock train/test splits |
| **Observability** | Spring Actuator health checks, latency metrics |

### ML Pipeline

```
Raw Data → Stratified Sampling → TF-IDF + Bigrams → MI Feature Selection → SVM Training
                                                                              ↓
                                              Production Model ← Grid Search Tuning
```

Key preprocessing decisions:
- **Mutual Information** feature selection (statistically principled, not arbitrary top-k)
- **TF-IDF with bigrams** captures phrases like "not good" and "highly recommend"
- **Stratified splits** ensure balanced positive/negative in train and test

Deep dive: [Architecture Documentation](docs/ARCHITECTURE.md)

---

## Training Your Own Models

### Full Pipeline

```bash
# 1. Prepare immutable train/test splits (run once)
./scripts/prepare_data.sh

# 2. Train all 12 models (4 algorithms x 3 datasets)
./scripts/train_all_models.sh

# 3. Evaluate cross-domain generalization
./scripts/evaluate_cross_domain.sh --persist

# 4. Select and deploy best model
./scripts/promote_to_production.sh

# 5. Generate comparison report
./scripts/generate_report.sh
```

### Train a Single Model

```bash
mvn exec:java -Dexec.mainClass="sentiment.training.TrainModel" \
  -Dexec.args="--algorithm svm --dataset amazon_polarity"
```

Training documentation: [TRAINING.md](docs/TRAINING.md)

---

## Datasets

| Dataset | Samples | Domain | Source |
|---------|---------|--------|--------|
| IMDB 50K | 50,000 | Movie reviews | Maas et al. (2011) |
| Amazon Polarity | 50,000 | Product reviews | McAuley & Leskovec (2013) |
| Yelp | 25,000 | Restaurant reviews | Yelp Dataset Challenge |

All datasets use stratified 80/20 train/test splits with SHA-256 verification.

Data quality documentation: [Data Cards](docs/data_cards/)

---

## Technology Stack

| Layer | Technology | Why |
|-------|------------|-----|
| **ML** | Weka 3.9.6 | Mature, accurate, interpretable algorithms |
| **API** | Spring Boot 3.4 | Production-grade REST with minimal config |
| **Resilience** | Resilience4j | Circuit breaker, rate limiting |
| **Build** | Maven | Standard Java dependency management |
| **Deploy** | Docker | Consistent environments, easy scaling |
| **Runtime** | Java 21 | Latest LTS with virtual threads support |

---

## Project Structure

```
java-sentiment-analyzer/
├── src/main/java/sentiment/
│   ├── api/                  # REST controllers, request/response models
│   ├── models/               # SVM, NaiveBayes, RandomForest, LogisticRegression
│   ├── preprocessing/        # Text cleaning, tokenization, TF-IDF
│   ├── training/             # Training orchestration, metadata persistence
│   ├── evaluation/           # Cross-domain testing, edge case analysis
│   └── config/               # Spring configuration, model loading
├── scripts/                  # Training and evaluation automation
├── models/production/        # Deployed model + feature importance
├── data/processed/           # Locked train/test splits with manifests
├── results/                  # Auto-generated evaluation reports
└── docs/                     # Architecture, deployment, training guides
```

---

## Documentation

| Document | Contents |
|----------|----------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, ML decisions, thread safety |
| [TRAINING.md](docs/TRAINING.md) | Training pipeline, edge cases, troubleshooting |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Docker, monitoring, production checklist |
| [Data Cards](docs/data_cards/) | Dataset provenance, biases, limitations |
| [Results Report](results/FINAL_COMPREHENSIVE_REPORT.md) | Full model comparison (auto-generated) |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTIMENT_RANDOM_SEED` | 42 | Reproducibility seed |
| `SENTIMENT_MI_THRESHOLD` | 50000 | Feature selection threshold |

Rate limits (per minute): 100 single / 20 batch / 2 model-compare

See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for full configuration reference.

---

## Limitations

- **Binary classification only**: Positive or negative (no neutral class)
- **English text**: Not tested on other languages
- **Domain shift**: Accuracy may vary on domains unlike training data (tech support, legal, medical)

For neutral sentiment or multi-class, retrain with appropriately labeled data.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
