# Architecture Overview

## Summary

| Component | Technology | Key Design |
|-----------|------------|------------|
| **API** | Spring Boot | REST endpoints, rate limiting, circuit breaker |
| **Models** | Weka (SVM, NB, LR, RF) | Strategy pattern, calibrated probabilities |
| **Preprocessing** | TF-IDF + MI selection | Thread-safe with ReadWriteLock |
| **Deployment** | Docker | Non-root container, health checks |

**Production model**: SVM trained on Amazon reviews, 88.0% cross-domain accuracy.

## Data Flow

**Training**: CSV → DataLoader → TextPreprocessor.fit() → WekaInstancesConverter.fit() → Classifier.train() → Serialized model

**Inference**: HTTP request → Rate limiter → Circuit breaker → Classifier.classify() → Response

## Package Structure

### `sentiment.data`
- `SimpleDatasetLoader`: CSV/TSV parsing with auto-detection
- `Dataset`: Immutable data container
- `DataPreparer`: Creates reproducible train/test splits with SHA-256 checksums

### `sentiment.preprocessing`
- `TextPreprocessor`: Text cleaning, tokenization, stopword removal, MI feature selection
- `WekaInstancesConverter`: TF-IDF vectorization via Weka filters
- Thread-safe via `ReadWriteLock` (concurrent reads, exclusive writes)

### `sentiment.models`
- `SentimentClassifier` interface with implementations: SVM, NaiveBayes, RandomForest, LogisticRegression
- All models return calibrated probabilities (Platt scaling for SVM)

### `sentiment.evaluation`
- `ClassifierEvaluationResult`: Accuracy, F1, precision, recall, ROC-AUC
- `CalibrationMetrics`: Brier Score, ECE
- `CrossDomainEvaluator`: Tests generalization across IMDB, Amazon, Yelp

### `sentiment.api`
- `SentimentController`: REST endpoints
- `PredictionMetrics`: Production monitoring via Micrometer

**Endpoints:**
- `POST /api/v1/sentiment/analyze` - Single classification
- `POST /api/v1/sentiment/batch` - Batch classification
- `GET /api/v1/model/feature-importance` - Top features
- `GET /api/v1/health` - Health check

### `sentiment.training`
- `ModelTrainer`: Offline training with cross-validation

## Key Implementation Details

### Thread Safety
```java
private final ReadWriteLock lock = new ReentrantReadWriteLock();

public void fit(List<Dataset> datasets) {
    lock.writeLock().lock();
    try { /* modify state */ }
    finally { lock.writeLock().unlock(); }
}

public Instances transform(List<Dataset> datasets) {
    lock.readLock().lock();
    try { /* read-only access */ }
    finally { lock.readLock().unlock(); }
}
```

### SVM Probability Calibration
Weka's SMO with `-V -1` enables Platt scaling to convert SVM outputs to calibrated probabilities.

### Circuit Breaker
```java
@CircuitBreaker(name = "modelInference", fallbackMethod = "classifyTextFallback")
private ResponseEntity<SentimentResponse> classifyText(String text, Double threshold) { ... }
```

Configuration: Opens after 50% failure rate, waits 30s before retry.

### Rate Limiting
```yaml
resilience4j.ratelimiter.instances.sentimentApi:
  limit-for-period: 100
  limit-refresh-period: 1m
```

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| ML Library | Weka | Mature, interpretable, low resource usage |
| Framework | Spring Boot | Ecosystem, monitoring, configuration |
| Classification | Binary only | Higher accuracy, clearer evaluation |
| Feature Selection | Mutual Information | No distributional assumptions, captures non-linear relationships |
| Models | Pre-trained | Fast startup for container orchestration |

## References

- Weka: https://waikato.github.io/weka-wiki/
- Spring Boot: https://docs.spring.io/spring-boot/docs/current/reference/html/
- Platt Scaling: Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines"
