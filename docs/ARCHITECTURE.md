# Architecture Overview

This document details the technical architecture, design patterns, and implementation decisions for the Java Sentiment Analyzer.

---

## TL;DR

| Component | Technology | Key Design |
|-----------|------------|------------|
| **API** | Spring Boot | REST endpoints, rate limiting (100 req/min), circuit breaker |
| **Models** | Weka (SVM, NB, LR, RF) | Strategy pattern, calibrated probabilities |
| **Preprocessing** | TF-IDF + MI selection | Thread-safe with ReadWriteLock |
| **Deployment** | Docker | Non-root container, health checks |

**Production model**: SVM trained on Amazon reviews, 88.2% cross-domain accuracy.

**Quick links**: [Training Guide](TRAINING.md) | [Deployment Guide](DEPLOYMENT.md) | [Data Cards](data_cards/)

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Package Structure](#package-structure)
3. [Component Deep Dive](#component-deep-dive)
4. [Technical Decisions](#technical-decisions)

---

## High-Level Architecture

```
+-----------------------------------------------------------------------+
|                          REST API LAYER                               |
|  - Spring Boot Controllers (Port 8080)                                |
|  - Rate Limiting: 100 req/min (Resilience4j)                          |
|  - Circuit Breaker: Fails fast on model errors                        |
|  - Production Metrics: Micrometer + Prometheus                        |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                        SERVICE LAYER                                  |
|  SentimentClassifier Interface (Strategy Pattern)                     |
|  - Thread-safe inference via ReadWriteLock                            |
|  - Probability calibration (Platt scaling for SVM)                    |
+-----------------------------------------------------------------------+
         |              |                 |                |
         v              v                 v                v
   +----------+   +----------+   +--------------+   +-------------+
   |   SVM    |   |  Naive   |   |   Random     |   |  Logistic   |
   | (Linear) |   |  Bayes   |   |   Forest     |   | Regression  |
   +----------+   +----------+   +--------------+   +-------------+
         |              |                 |                |
         +------+-------+-----------------+----------------+
                |
                v
+-----------------------------------------------------------------------+
|                    PREPROCESSING PIPELINE                             |
|  TextPreprocessor:                                                    |
|    - Text cleaning (URLs, HTML, emoticons)                            |
|    - Tokenization via AdvancedTokenizer                               |
|    - Stopword removal via IntelligentStopwordRemover                  |
|    - Vocabulary capture with MI feature selection (configurable)      |
|                                                                       |
|  WekaInstancesConverter:                                              |
|    - TF-IDF vectorization (Weka's StringToWordVector filter)          |
|    - Normalization (L2 norm)                                          |
|    - Bigram support (optional)                                        |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------+
|                          DATA LAYER                                   |
|  SimpleDatasetLoader:                                                 |
|    - CSV/TSV format support with auto-detection                       |
|    - Flexible column matching (text/review, sentiment/label)          |
|    - Binary classification (positive/negative only)                   |
+-----------------------------------------------------------------------+
```

**Data Flow:**
1. **Training**: CSV → SimpleDatasetLoader → TextPreprocessor.fit() → WekaInstancesConverter.fit() → Classifier.train() → Serialized model
2. **Inference**: HTTP request → Rate limiter → Circuit breaker → Classifier.classify() → TextPreprocessor.transform() → WekaInstancesConverter.transform() → Prediction → Metrics → Response

**Architectural Principles:**
- **Separation of Concerns**: API, business logic, data access cleanly separated
- **Interface-Driven**: All major components behind interfaces for testability
- **Immutability Where Possible**: DTOs, evaluation results immutable
- **Fail-Fast**: Validation at API boundary, propagate failures quickly

---

## Package Structure

### `sentiment.data` - Data Loading Layer

**Responsibility**: Load and parse datasets from various formats

**Key Classes:**
- `SimpleDatasetLoader`: Flexible loader with format auto-detection (CSV, TSV, JSONL, TXT)
- `Dataset`: Immutable data container with builder pattern
- `DatasetLoadResult`: Metadata wrapper (load time, format type, file path)
- `DataLoadingException`: Domain-specific exception for loading errors
- `DataQualityReport`: Validation and quality metrics for loaded datasets
- `DatasetStatistics`: Statistical summary of dataset characteristics
- `DataPreparer`: Creates immutable train/test splits with manifest-based locking
- `SplitManifest`: Tracks split metadata and SHA-256 checksums for reproducibility

**Design Pattern**: Simple factory with switch-based format detection

**Why This Design?**
- Pragmatism: Single loader handles most common formats (CSV/TSV with flexible column detection)
- Extensibility: Easy to add specialized loaders for specific datasets (e.g., IMDB)
- Automatic column detection: Handles various naming conventions (text/review/comment, sentiment/label/polarity)
- Progressive enhancement: Start simple, add complexity only when needed

**Example Usage:**
```java
SimpleDatasetLoader loader = new SimpleDatasetLoader();
List<Dataset> data = loader.load("reviews.csv");
// Automatically detects CSV format and column names

DatasetLoadResult result = loader.loadWithMetadata("reviews.csv");
System.out.println("Loaded " + result.datasets().size() + " samples in " + result.loadTimeMs() + "ms");
```

**Column Detection:**
- Text columns: `text`, `review`, `comment`, `content`, `message`, `body`
- Sentiment columns: `sentiment`, `label`, `polarity`, `class`, `rating`
- Case-insensitive matching

**Sentiment Parsing Strategies:**
```java
// Text labels
"positive" / "pos" / "1" / "4"  → POSITIVE
"negative" / "neg" / "0" / "-1" → NEGATIVE

// Numeric ratings (1-5 stars)
1.0-2.5  → NEGATIVE
3.5-5.0  → POSITIVE
// Note: Neutral samples (2.5-3.5 or "neutral" label) are filtered out
```

**Error Handling:**
- **Invalid rows**: Logged and skipped (e.g., empty text, unrecognized sentiment)
- **High error rate warning**: If >50% of rows fail, warning logged
- **Progress logging**: Every 10,000 rows during large dataset loads
- **Validation**: Fails fast if no valid data loaded or required columns missing

**Performance Considerations:**
- Uses Apache Commons CSV for robust parsing (handles quotes, escapes, edge cases)
- Streaming parser: Processes row-by-row (memory-efficient for large files)
- Builder pattern for Dataset construction (validates at build time)

**Data Preparation Workflow:**

The `DataPreparer` and `SplitManifest` classes ensure reproducible, immutable data splits:

```
raw/{domain}/*.csv
  → DataPreparer.prepare()
    → StratifiedDataSplitter (80/20 split preserving class balance)
    → processed/{domain}/
        ├── train.csv
        ├── test.csv
        └── splits.manifest.json  ← Lock file with SHA-256 checksums
```

**Key Invariant:** Once splits are created, they are immutable. The manifest acts as a "lock file":
- `prepare()` skips if manifest exists (prevents accidental overwrites)
- `forceReset()` required for explicit regeneration
- `verify()` checks SHA-256 checksums match actual files

This ensures `test_accuracy == in-domain accuracy` across all experiments.

---

### `sentiment.preprocessing` - Text Processing Pipeline

**Responsibility**: Transform raw text into feature vectors for ML models

**Key Classes:**
- `TextPreprocessor`: Orchestrates text cleaning, tokenization, stopword removal; includes embedded MI feature selection
- `WekaInstancesConverter`: Converts preprocessed text to Weka Instances with TF-IDF vectorization
- `AdvancedTokenizer`: Intelligent tokenization (handles URLs, emoticons, contractions)
- `IntelligentStopwordRemover`: Context-aware stopword filtering
- `ContractionExpander`: "don't"  "do not"

**Design Pattern**: Template Method + Strategy

**Pipeline Stages:**
```
Raw Text
   
AdvancedTokenizer.tokenize()
   
ContractionExpander.expand()
   
IntelligentStopwordRemover.removeStopwords()
   
WekaInstancesConverter.fit/transform() - TF-IDF via Weka filters
   
TextPreprocessor.fit() - MI feature selection (embedded in PipelineState)
   
Weka Instances
```

**Note on Component Separation:**
- **Mutual Information feature selection** is NOT a separate class - it's embedded in `TextPreprocessor.PipelineState.captureVocabularyStatsWithPrincipledSelection()` (lines 559-642)
- **TF-IDF vectorization** is handled by `WekaInstancesConverter`, not a standalone `TFIDFFeatureExtractor`

**Thread Safety:**
- Uses `ReadWriteLock` for concurrent inference
- Exclusive lock during `fit()` (model training)
- Shared lock during `transform()` (inference)

**Mathematical Foundation:**

**Mutual Information Feature Selection:**

Mutual information (MI) measures how much knowing whether a word appears in a review tells us about its sentiment. It's based on information theory, which quantifies "information" as reduction in uncertainty.

**Intuitive Example:**
- Word "excellent": Appears in 90% of positive reviews, 10% of negative reviews
  - **High MI**: Knowing "excellent" appears strongly predicts positive sentiment
- Word "the": Appears in 50% of positive reviews, 50% of negative reviews
  - **Low MI**: Knowing "the" appears tells us nothing about sentiment

**Formula:**
```
I(X;Y) = H(Y) - H(Y|X)

Where:
- X = feature presence/absence (word appears or not)
- Y = sentiment class (positive/negative)
- H(Y) = entropy of class distribution (uncertainty about sentiment before seeing word)
- H(Y|X) = conditional entropy (uncertainty after seeing word)
- I(X;Y) = reduction in uncertainty = information gained

Higher I(X;Y) = word X provides more information about sentiment Y
```

**Why Mutual Information?**
- **Theoretically principled**: Based on information theory, not heuristics
- **Captures non-linear relationships**: Detects complex word-sentiment patterns
- **No distributional assumptions**: Works for any data distribution (vs. chi-squared assumes independence)
- **Provably optimal**: Maximizes discriminative power for discrete features

**Key Configuration:**
```yaml
sentiment:
  features:  # Single source of truth for feature extraction
    min-word-length: 2
    max-features: 5000
    min-term-freq: 1
    use-tfidf: true
    use-bigrams: true
```

**Note:** There is no `use-mutual-information` config flag - MI selection is always applied when vocabulary exceeds the threshold (default: 50,000). The threshold is configurable via:
- `sentiment.features.mi-selection-threshold` in `application.yml`
- `SENTIMENT_MI_THRESHOLD` environment variable

---

### `sentiment.models` - Classification Layer

**Responsibility**: Train and execute sentiment classification models

**Key Classes:**
- `SentimentClassifier` (interface): Common contract for all algorithms
- `SVMClassifier`: Support Vector Machine (SMO algorithm, Weka)
- `NaiveBayesClassifier`: Probabilistic Bayes classifier
- `RandomForestClassifier`: Ensemble of decision trees
- `LogisticRegressionClassifier`: Linear log-odds model
- `ClassifierEvaluator`: Cross-validation, performance metrics
- `WekaModelPersistence`: Serialization/deserialization

**Design Pattern**: Strategy + Template Method

**Interface Contract:**
```java
public interface SentimentClassifier {
    void train(List<Dataset> trainingData) throws Exception;
    String classify(String text) throws Exception;
    double[] getClassificationProbabilities(String text) throws Exception;
    boolean isTrained();
    AlgorithmType getAlgorithmType();
    String[] getSupportedClasses();
}
```

**Why This Interface?**
- **Polymorphism**: Swap algorithms without changing client code
- **Testability**: Easy to mock for unit tests
- **Type Safety**: `AlgorithmType` enum instead of strings
- **Probability Support**: All models return calibrated probabilities

**Algorithm Comparison:**

| Algorithm | Training Time | Inference Time | Interpretability | Best For |
|-----------|--------------|----------------|------------------|----------|
| SVM | O(n²) | O(sv) | Low | High accuracy, moderate data |
| Naive Bayes | O(n) | O(c) | High | Speed, small data |
| Random Forest | O(n log n × trees) | O(trees × depth) | Medium | Robustness, overfitting prevention |
| Logistic Regression | O(n × iterations) | O(features) | High | Interpretability, linear separability |

**Persistence Strategy:**
- Pre-trained models stored as serialized Weka classifiers
- Path configurable via `application.yml`
- Fallback: Train on-demand if pre-trained unavailable
- Version compatibility: Models tied to Weka 3.9.6

---

### `sentiment.evaluation` - Model Assessment

**Responsibility**: Compute comprehensive evaluation metrics

**Key Classes:**
- `ClassifierEvaluationResult`: Immutable metrics container (includes ROC-AUC/PR-AUC via Weka's Evaluation)
- `CalibrationMetrics`: Brier Score, ECE, MCE
- `StratifiedDataSplitter`: Maintain class distribution in splits
- `FeatureImportanceAnalyzer`: Interpretability metrics via permutation importance
- `CrossDomainEvaluator`: Tests model generalization across domains (IMDB, Amazon, Yelp)

**Metrics Provided:**

**Basic Metrics:**
- Accuracy: (TP + TN) / Total
- Precision: TP / (TP + FP)  per class
- Recall: TP / (TP + FN)  per class
- F1 Score: 2 × (Precision × Recall) / (Precision + Recall)

**Advanced Metrics:**
- **ROC-AUC**: Area under Receiver Operating Characteristic curve
- **PR-AUC**: Area under Precision-Recall curve (better for imbalanced data)
- **Confusion Matrix**: Full matrix with row/column labels

**Calibration Metrics:**
- **Brier Score**: Mean squared error of probability predictions
  - Formula: `(1/n) × ∑(p_i - y_i)²`
  - Range: [0, 1], lower is better
- **Expected Calibration Error (ECE)**: Average calibration gap across probability bins
- **Maximum Calibration Error (MCE)**: Worst-case calibration gap

**Why Calibration Matters:**
- Raw SVM outputs aren't true probabilities
- Calibration ensures confidence scores are meaningful
- Critical for decision-making (e.g., "only show sentiment if confidence > 0.8")

**Example Output:**
```
Model Evaluation Results (SVM):
  Accuracy: 89.2%
  Macro-Avg F1: 0.892
  ROC-AUC: 0.945
  Brier Score: 0.123
  ECE: 0.047

Confusion Matrix:
           Predicted Pos  Predicted Neg
Actual Pos     4512           488
Actual Neg      392          4608
```

**Cross-Domain & Robustness Testing:**

The evaluation package includes specialized tools to assess model generalization and robustness:

**CrossDomainEvaluator:**
- Tests trained models across multiple domains (IMDB movie reviews, Amazon products, Yelp businesses)
- Measures domain transfer performance to detect overfitting to specific datasets
- Evaluates whether sentiment patterns learned generalize beyond training domain
- Critical for production deployment across diverse use cases

---

### `sentiment.api` - REST API Layer

**Responsibility**: HTTP interface for sentiment analysis

**Key Classes:**
- `SentimentController`: Main REST endpoints
- `SentimentRequest/Response`: DTOs with validation
- `BatchRequest/BatchResponse`: Batch processing DTOs
- `HealthResponse`: Health check response
- `FeatureImportanceResponse`: Feature importance analysis results
- `ErrorResponse`: Standardized error message format
- `RestApiExceptionHandler`: Centralized error handling
- `PredictionMetrics`: Production monitoring via Micrometer (sentiment.api.metrics package)

**Endpoints:**

```
POST /api/v1/sentiment/analyze        - Single text classification
POST /api/v1/sentiment/batch          - Batch classification (parallel)
GET  /api/v1/model/feature-importance - Model interpretability (top features by importance)
GET  /api/v1/health                   - Health check
```

**Features:**

**1. Rate Limiting (Resilience4j):**
```yaml
resilience4j:
  ratelimiter:
    instances:
      sentimentApi:
        limit-for-period: 100     # requests per minute
        limit-refresh-period: 1m
```

**2. Validation (Bean Validation):**
```java
@NotBlank(message = "Text cannot be blank")
@Size(max = 10000, message = "Text cannot exceed 10000 characters")
String text;

@DecimalMin(value = "0.0", message = "Confidence threshold must be between 0.0 and 1.0")
@DecimalMax(value = "1.0", message = "Confidence threshold must be between 0.0 and 1.0")
Double confidenceThreshold;
```

**3. Error Handling:**
- 400 Bad Request: Invalid input (validation failure)
- 404 Not Found: Endpoint does not exist
- 429 Too Many Requests: Rate limit exceeded
- 500 Internal Server Error: Model inference failure
- 503 Service Unavailable: Model not loaded

**4. Monitoring:**
- Spring Boot Actuator endpoints (`/actuator/health`, `/actuator/metrics`)
- Prometheus metrics export
- Request/response timing
- Error rate tracking

**5. Batch Processing:**
- Parallel execution via `ForkJoinPool`
- Order preservation (results match input order)
- Per-request timing

---

### `sentiment.config` - Configuration Management

**Responsibility**: Centralized application configuration

**Key Classes:**
- `SentimentConfiguration`: Main config properties
- `WebMvcConfiguration`: Spring MVC customization
- `ActuatorConfiguration`: Monitoring setup

**Configuration Hierarchy:**
```
application.yml (base config)
   
application-{profile}.yml (profile-specific: dev, prod, test)
   
Environment Variables (overrides)
   
Command-line arguments (highest priority)
```

**Example Configuration:**
```yaml
sentiment:
  model-path: /app/models/production/sentiment_model.ser
  confidence-threshold: 0.7

  preprocessing:
    min-word-length: 2
    max-features: 5000
    use-tfidf: true
    use-bigrams: true

  api:
    max-batch-size: 100
    rate-limit: 100
    validation:
      max-text-length: 10000
```

---

### `sentiment.training` - Model Training Tools

**Responsibility**: Offline model training and evaluation

**Key Classes:**
- `ModelTrainer`: Core training logic with cross-validation
- `TrainingMetadata`: Records training configuration and metrics
- `ClassifierTrainingTemplate`: Template method for algorithm-specific training

**Usage:**
```bash
# Train a single model
mvn exec:java -Dexec.mainClass="sentiment.training.ModelTrainer" \
  -Dexec.args="--algorithm SVM --dataset imdb_50k"

# Train all 12 models (4 algorithms x 3 datasets)
./scripts/train_all_models.sh

# Algorithms: SVM, NAIVE_BAYES, LOGISTIC_REGRESSION, RANDOM_FOREST
# Datasets: imdb_50k, amazon_polarity, yelp
```

---

## Component Deep Dive

### TextPreprocessor: Thread Safety

**Challenge**: TF-IDF state is mutable (vocabulary, IDF weights). How to support concurrent inference during training?

**Solution**: `ReadWriteLock`

```java
private final ReadWriteLock lock = new ReentrantReadWriteLock();

public void fit(List<Dataset> datasets) {
    lock.writeLock().lock();  // Exclusive lock for training
    try {
        // Modify vocabulary, IDF weights
        this.vocabulary = buildVocabulary(datasets);
        this.idfWeights = computeIDF(datasets);
    } finally {
        lock.writeLock().unlock();
    }
}

public Instances transform(List<Dataset> datasets) {
    lock.readLock().lock();  // Shared lock for inference
    try {
        // Read-only access to vocabulary, IDF weights
        return vectorize(datasets, vocabulary, idfWeights);
    } finally {
        lock.readLock().unlock();
    }
}
```

**Benefits:**
- Multiple threads can call `transform()` concurrently
- Training (`fit()`) blocks all inference
- No race conditions on mutable state

---

### SVM Probability Calibration

**Challenge**: SVM decision function outputs are not probabilities.

SVMs output raw decision values (signed distances from the hyperplane), not probabilities. For binary classification, a value of +2.5 means "strongly positive" but doesn't mean "85% confident." This makes it hard to:
- Set confidence thresholds ("only show predictions > 80% confident")
- Compare predictions across different models
- Make probabilistic decisions

**Solution**: Platt scaling (sigmoid function fitted on validation set)

Platt scaling transforms SVM outputs into calibrated probabilities using a sigmoid:

```
P(y=1|x) = 1 / (1 + exp(A*f(x) + B))
```

Where:
- `f(x)` = SVM decision function output
- `A, B` = parameters learned via maximum likelihood on validation data

**Intuition**: Fits a logistic regression on top of SVM outputs to convert distances → probabilities.

**Weka Implementation:**

Probability calibration is enabled via SMO's `-V` option (number of cross-validation folds for internal probability calibration):

```java
// From SVMConfig.toOptionsString() (line 116):
String options = "-C " + c +
                 " -V -1" +  // Enable CV-based probability calibration (Platt scaling)
                 " ...";
smo.setOptions(weka.core.Utils.splitOptions(options));
```

The `-V -1` option enables Weka's internal cross-validation for probability calibration, which fits logistic models to convert SVM decision values into well-calibrated probabilities.

**Result**: `getClassificationProbabilities()` returns calibrated probabilities in [0, 1].

**Validation**: Brier Score and ECE metrics confirm calibration quality (see evaluation metrics).

---

### WekaInstancesConverter: TF-IDF Pipeline

**Challenge**: Convert preprocessed text into Weka's ML-ready `Instances` format with TF-IDF features.

**What is TF-IDF?**
TF-IDF (Term Frequency-Inverse Document Frequency) transforms text into numerical features by weighing words based on:
- **TF (Term Frequency)**: How often a word appears in a document (more = higher weight)
- **IDF (Inverse Document Frequency)**: How rare a word is across all documents (rarer = higher weight)

**Why?** Common words like "the" get low weights, while distinctive words like "excellent" get high weights. This helps ML models focus on meaningful words.

**Solution**: Composition pattern that orchestrates `TextPreprocessor` + Weka filters

**Implementation:**
```java
@Component
@Scope("prototype")
public class WekaInstancesConverter extends TrainingTemplate<Instances> {
    private final TextPreprocessor textPreprocessor;
    private StringToWordVector trainedStringToWordFilter;  // TF-IDF
    private Normalize trainedNormalizationFilter;

    public Instances fit(List<Dataset> datasets) {
        // 1. Fit TextPreprocessor first (captures vocabulary + MI selection)
        textPreprocessor.fit(datasets);

        // 2. Configure Weka's StringToWordVector filter
        StringToWordVector filter = new StringToWordVector();
        filter.setWordsToKeep(maxFeatures);
        filter.setMinTermFreq(minTermFreq);
        filter.setTFTransform(true);
        filter.setIDFTransform(useTfIdf);
        filter.setNGramTokenizer(useBigrams ? new NGramTokenizer() : null);

        // 3. Train filter on preprocessed text
        filter.setInputFormat(rawInstances);
        this.trainedStringToWordFilter = filter;

        return Filter.useFilter(rawInstances, filter);
    }
}
```

**Why This Design?**
- Separation of concerns: Text preprocessing (TextPreprocessor) vs. vectorization (Weka filters)
- Leverages Weka's battle-tested TF-IDF implementation
- Prototype scope ensures isolated state per training run

---

### Production Metrics: PredictionMetrics

**Challenge**: Monitor model performance in production without ground truth labels.

**Solution**: Track proxy metrics via Micrometer

**Metrics Tracked:**
```java
@Component
public class PredictionMetrics {
    // Label distribution (binary)
    Counter positiveCounter;
    Counter negativeCounter;

    // Confidence monitoring
    DistributionSummary confidenceDistribution;  // Percentiles: p50, p95, p99
    Counter lowConfidenceCounter;  // Predictions < 0.6 confidence

    // Latency tracking
    Timer inferenceTimer;  // Latency percentiles: p50, p95, p99
}
```

**Exposed Metrics (Prometheus format):**
- `sentiment_predictions_total{label="positive|negative"}`
- `sentiment_predictions_low_confidence`
- `sentiment_inference_duration_seconds{quantile="0.5|0.95|0.99"}`
- `sentiment_prediction_confidence{quantile="0.5|0.95|0.99"}`

**Use Cases:**
- **Detect model degradation**: Sharp increase in low-confidence predictions
- **Label distribution drift**: Sudden shift in positive/negative ratio
- **Latency SLA monitoring**: P99 latency exceeds threshold

---

### Circuit Breaker: Resilience Pattern

**Challenge**: Model inference failures can cascade and overwhelm the service.

**Solution**: Resilience4j circuit breaker with fallback

**Implementation:**
```java
@CircuitBreaker(name = "modelInference", fallbackMethod = "classifyTextFallback")
private ResponseEntity<SentimentResponse> classifyText(String text, Double threshold) {
    // Normal inference path
    String sentiment = classifier.classify(text);
    return ResponseEntity.ok(SentimentResponse.success(sentiment, ...));
}

private ResponseEntity<SentimentResponse> classifyTextFallback(String text, Double threshold, Exception e) {
    // Graceful degradation when circuit is OPEN
    return ResponseEntity.status(503).body(SentimentResponse.error(
        "Sentiment analysis temporarily unavailable. Please retry in 30 seconds.", text));
}
```

**Configuration** (application.yml):
```yaml
resilience4j:
  circuitbreaker:
    instances:
      modelInference:
        failure-rate-threshold: 50       # Open if 50% fail
        slow-call-duration-threshold: 2s # Calls > 2s count as failures
        wait-duration-in-open-state: 30s # Try recovery after 30s
        sliding-window-size: 10          # Evaluate last 10 requests
```

**States:**
- **CLOSED**: Normal operation, all requests go through
- **OPEN**: Too many failures, return fallback immediately (fail-fast)
- **HALF_OPEN**: Testing if service recovered (3 test requests)

**Benefits:**
- Prevents cascading failures
- Fast failure response (no timeout waiting)
- Self-healing (automatic recovery testing)

---

### Batch Processing: Order Preservation

**Challenge**: Parallel processing with `ForkJoinPool` may complete out of order.

**Solution**: Use indexed futures and reconstruct order:

```java
List<CompletableFuture<IndexedResult>> futures = IntStream.range(0, texts.size())
    .mapToObj(i -> CompletableFuture.supplyAsync(
        () -> new IndexedResult(i, classifier.classify(texts.get(i))),
        executor
    ))
    .collect(Collectors.toList());

// Wait for all, then sort by index
List<SentimentResponse> results = futures.stream()
    .map(CompletableFuture::join)
    .sorted(Comparator.comparingInt(IndexedResult::getIndex))
    .map(IndexedResult::getResult)
    .collect(Collectors.toList());
```

## Concurrency Model

### Read-Heavy Workload Optimization

**Assumption**: Inference (read) >> Training (write)

**Strategy**:
- `ReadWriteLock` allows concurrent reads
- Writes (training) are rare and can block
- Pre-trained models eliminate training in production

**Performance Characteristics**:
- Single-threaded: ~30 inferences/second
- Multi-threaded (8 cores): ~200 inferences/second
- Scalability limited by JVM GC, not locking

**Benchmark Methodology** (reference only, not production SLA):
- **Hardware**: MacBook Pro M1 Max, 32GB RAM
- **JVM**: Java 21, -Xmx4g -Xms4g
- **Model**: Pre-trained SVM (linear kernel, 5000 features)
- **Dataset**: Amazon product reviews, avg length 50 words
- **Measurement**: Average throughput over 10,000 requests, warm JVM (10 warmup runs)
- **Concurrency**: Single-threaded = sequential loop; Multi-threaded = parallel stream with 8 threads
- **Latency**: Excludes network overhead (measures classifier.classify() only)

**Note**: Actual production throughput depends on text length, model complexity, hardware, and JVM tuning. These figures provide relative comparison, not absolute guarantees.

---

## Technical Decisions

### Why Weka Over Deeplearning4j?

| Factor | Weka | Deeplearning4j |
|--------|------|----------------|
| **Maturity** | 20+ years, stable | Newer, evolving |
| **Model Complexity** | Traditional ML | Deep learning |
| **Resource Requirements** | Low (< 512MB) | High (GPU beneficial) |
| **Interpretability** | High (tree-based, linear) | Low (black box) |
| **Training Time** | Minutes | Hours/days |
| **Inference Latency** | < 50ms | 100-500ms |

**Decision**: Weka for this project (sentiment analysis on product reviews doesn't require deep learning complexity).

---

### Why Spring Boot Over JAX-RS (Jersey)?

| Factor | Spring Boot | JAX-RS |
|--------|-------------|--------|
| **Ecosystem** | Massive (Security, Data, Cloud) | Smaller |
| **Monitoring** | Actuator + Prometheus | Manual |
| **Configuration** | YAML + profiles | XML or code |
| **Dependency Injection** | Spring DI | CDI or manual |
| **Learning Curve** | Moderate | Steeper |

**Decision**: Spring Boot for comprehensive ecosystem and monitoring.

---

### Why Pre-trained Models?

**Options:**
1. **Train on startup**: 2-5 minutes, fresh model
2. **Pre-trained models**: < 5 seconds, stale model

**Decision**: Pre-trained with fallback to training.

**Rationale**:
- Production systems need fast startup (container orchestration)
- Model staleness acceptable for product reviews (language stable)
- Config flag allows override: `prefer-pretrained: false`

---

### Why Binary Classification Only?

**Options:**
1. Binary (positive/negative)
2. Multi-class (positive/neutral/negative)
3. Multi-label (tags: angry, happy, sarcastic, etc.)

**Decision**: Binary classification.

**Rationale**:
- Most use cases are binary (recommend/don't recommend)
- Neutral class is ambiguous (indifferent vs. mixed sentiment)
- Higher accuracy and clearer evaluation metrics
- Cross-domain generalization is more reliable with binary labels
- Datasets with neutral labels have inconsistent definitions

---

### Why Mutual Information Over Chi-Squared?

**Comparison:**

| Metric | Chi-Squared | Mutual Information |
|--------|-------------|-------------------|
| **Assumptions** | Feature independence | None |
| **Non-linear relationships** | Poor | Good |
| **Computational cost** | O(n) | O(n log n) |
| **Theoretical foundation** | Hypothesis test | Information theory |

**Empirical Result**: MI improved accuracy by 5-8% on Amazon reviews dataset.

**Decision**: Mutual Information for feature selection.

---

## Future Architectural Improvements

### 1. Model Versioning

**Current State**: Single model per algorithm
**Proposed**: Model registry with versioning (model-svm-v1.0.ser, model-svm-v1.1.ser)

**Benefits**:
- A/B testing
- Rollback on regression
- Gradual rollout

---

### 2. Feature Store

**Current State**: Features computed on-demand
**Proposed**: Pre-computed feature cache (Redis)

**Benefits**:
- Faster inference (skip preprocessing)
- Consistent features across services
- Reusable for other models

---

### 3. Model Serving Layer

**Current State**: Model embedded in API service
**Proposed**: Separate model serving (gRPC service)

**Benefits**:
- Independent scaling (API vs. model)
- Multi-language clients (Python, Go, etc.)
- Model updates without API redeployment

---

### 4. Distributed Inference

**Current State**: Single-node, in-memory model
**Proposed**: Model sharding across nodes (Kubernetes StatefulSet)

**Benefits**:
- Handle higher throughput (> 10k req/sec)
- Fault tolerance (replica sets)
- Geographic distribution (latency optimization)

---

## Glossary

**Bigram**: A pair of consecutive words (e.g., "very good"). Used as features alongside single words (unigrams) to capture phrase meanings.

**Brier Score**: Measures how well predicted probabilities match actual outcomes. Lower is better. Formula: average of (predicted_prob - actual_label)².

**Calibration**: Process of ensuring model confidence scores are accurate probabilities (e.g., predictions marked "80% confident" should be correct 80% of the time).

**Circuit Breaker**: Resilience pattern that stops calling a failing service to prevent cascading failures. Has 3 states: CLOSED (working), OPEN (failing, return fallback), HALF_OPEN (testing recovery).

**ECE (Expected Calibration Error)**: Measures average difference between predicted confidence and actual accuracy across confidence bins.

**Entropy**: In information theory, measures uncertainty. High entropy = unpredictable, low entropy = predictable.

**Feature**: In ML, a measurable property used for prediction. For text, typically word presence/absence or TF-IDF weights.

**IDF (Inverse Document Frequency)**: Measures how rare a word is across documents. Rare words get higher weights.

**Mutual Information**: Measures how much knowing one variable tells you about another. Used here to find words that best predict sentiment.

**Platt Scaling**: Technique to convert SVM outputs into calibrated probabilities using logistic regression.

**Prototype Scope**: Spring bean scope where each injection gets a new instance (vs. singleton where all share one instance).

**ReadWriteLock**: Concurrency control that allows multiple readers OR one writer (but not both simultaneously).

**ROC-AUC**: Area Under Receiver Operating Characteristic curve. Measures how well a binary classifier separates classes (1.0 = perfect, 0.5 = random).

**Strategy Pattern**: Design pattern where different algorithms implement the same interface, allowing runtime swapping.

**TF (Term Frequency)**: How often a word appears in a document. Higher frequency = higher weight.

**TF-IDF**: Combines TF and IDF to weight words. Balances frequency (TF) with distinctiveness (IDF).

---

## References

- **Weka Documentation**: https://waikato.github.io/weka-wiki/
- **Spring Boot Reference**: https://docs.spring.io/spring-boot/docs/current/reference/html/
- **Mutual Information**: Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*
- **Platt Scaling**: Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines"

---

**Last Updated**: 2026-01-24
**Author**: Victoria Alabi
