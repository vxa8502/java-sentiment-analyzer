# Java Sentiment Analyzer

A sentiment analysis API built in Java with Spring Boot.

Goal: Understand engineering rigor of production ML Systems (type safety, circuit breakers, health checks, and containerized deployment).

| Metric | Value |
|--------|-------|
| Cross-domain accuracy | **87.9%** (tested on 3 different text domains) |
| Latency | **10-50ms** per request |
| Throughput | **1,000+ req/min** |

**[Try the Live API](https://java-sentiment-api.onrender.com/api/v1/health)**

```bash
curl -X POST https://java-sentiment-api.onrender.com/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"This product is amazing and I love it"}'
```

```json
{"sentiment":"positive","confidence":0.95,"text":"This product is amazing and I love it","processingTimeMs":12}
```

---

## Why Java for ML?

I built this to demonstrate cross-language proficiency and enterprise deployment patterns that Python notebooks don't address:

- **Type-safe pipelines** catch errors at compile time, not in production
- **Spring Boot patterns**: circuit breakers, rate limiting, health checks
- **Weka's maturity**: 20+ years of peer-reviewed ML algorithms
- **JVM concurrency**: ReadWriteLock enables parallel inference during model updates

The result: a sentiment classifier that runs anywhere Docker runs, handles real traffic patterns, and maintains consistent accuracy across completely different text domains (movie reviews, product reviews, restaurant reviews).

---

## Results

### Production Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 88.5% |
| **Cross-Domain Average** | 87.9% |
| **F1 Score** | 0.885 |
| **Latency** | 10-50ms |
| **Throughput** | ~1,000 req/min |

The production model (SVM trained on Amazon product reviews) was selected for **best generalization**—it maintains 85-91% accuracy when applied to completely different domains:

| Test Domain | Accuracy |
|-------------|----------|
| Amazon (in-domain) | 88.5% |
| IMDB movies | 84.9% |
| Yelp restaurants | 91.0% |

### Model Comparison (12 experiments)

Trained 4 algorithms across 3 domains to find the best trade-off between accuracy and generalization:

| Algorithm | Best In-Domain | Cross-Domain Avg | Notes |
|-----------|----------------|------------------|-------|
| **SVM** | 93.7% (Yelp) | **87.9%** | Best generalizing |
| Random Forest | 91.2% (Yelp) | 85.5% | Largest model files |
| Logistic Regression | 92.2% (Yelp) | 82.8% | Fastest training |
| Naive Bayes | 84.0% (IMDB) | 76.3% | Fastest inference |

Full results: [Model Comparison Report](results/FINAL_COMPREHENSIVE_REPORT.md)

### What the Model Learned

Top features by SVM weight show the model captures negation patterns and domain-specific language:

| Positive Indicators | Weight | Negative Indicators | Weight |
|---------------------|--------|---------------------|--------|
| not disappointed | +1.32 | not worth | -1.50 |
| excellent | +0.94 | disappointing | -1.26 |
| awesome | +0.92 | not recommend | -1.27 |
| fantastic | +0.91 | worst | -1.23 |
| four stars | +0.88 | two stars | -1.03 |

The bigram "not disappointed" ranking as the top positive indicator demonstrates that TF-IDF bigrams successfully capture negation—a common failure mode for bag-of-words models.

Full feature analysis: [Feature Importance API endpoint](#feature-importance) or `models/production/sentiment_model-feature-importance.json`

---

## Quick Start

**Try the live API:**
```bash
curl -X POST https://java-sentiment-api.onrender.com/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"This product exceeded my expectations!"}'
```

**Or run locally with Docker:**
```bash
docker build -t sentiment-api .
docker run -p 8080:8080 sentiment-api
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
  "text": "This product exceeded my expectations!",
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
curl http://localhost:8080/api/v1/model/feature-importance
```

### Health Check
```bash
curl http://localhost:8080/api/v1/health
```

See [demo/](demo/) for Python client and more curl examples.

---

## Architecture Highlights

Attempt at a deployable system with production patterns:

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
./scripts/evaluate_cross_domain.sh

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
| **Runtime** | Java 24 | Latest with virtual threads support |

---

## Project Structure

```
java-sentiment-analyzer/
├── src/main/java/sentiment/
│   ├── api/                  # REST controllers, request/response models
│   ├── models/               # SVM, NaiveBayes, RandomForest, LogisticRegression
│   ├── preprocessing/        # Text cleaning, tokenization, TF-IDF
│   ├── training/             # Training orchestration, metadata persistence
│   ├── evaluation/           # Cross-domain testing, metrics
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
| [TRAINING.md](docs/TRAINING.md) | Training pipeline, troubleshooting |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Docker, monitoring, production checklist |
| [Data Cards](docs/data_cards/) | Dataset provenance, biases, limitations |
| [Results Report](results/FINAL_COMPREHENSIVE_REPORT.md) | Full 12-model comparison (auto-generated) |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTIMENT_RANDOM_SEED` | 42 | Reproducibility seed |
| `SENTIMENT_MI_THRESHOLD` | 50000 | Feature selection threshold |

Rate limits (per minute, default profile): 100 single / 20 batch / 2 model-compare

Note: Production profile uses stricter limits (60 single / 10 batch / 5 per 10min).

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
