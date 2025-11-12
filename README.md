# Java Sentiment Analyzer

> Enterprise-grade sentiment analysis system demonstrating production Java ML deployment patterns, multi-algorithm comparison, and scalable API design.

[![Java](https://img.shields.io/badge/Java-21-orange.svg)](https://openjdk.java.net/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-3.4.0-brightgreen.svg)](https://spring.io/projects/spring-boot)
[![Weka](https://img.shields.io/badge/Weka-3.9.6-blue.svg)](https://www.cs.waikato.ac.nz/ml/weka/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Why Java for Machine Learning?

As a Python-native ML practitioner, I built this project to **complement my toolkit** and understand how sentiment analysis deploys in enterprise Java environments. While Python dominates ML research, Java powers mission-critical production systems at scale.

**Key Motivations:**
- **Enterprise Reality**: Many companies run Java-first infrastructures where Python deployment is complex
- **Production Patterns**: Explore how strong typing, thread safety, and JVM performance characteristics benefit ML systems
- **Technical Breadth**: Demonstrate cross-language adaptability beyond Python comfort zone
- **Systems Thinking**: Compare Weka/Smile ecosystem against Python's scikit-learn for real-world trade-offs

This project showcases **production-ready ML engineering** in Java, not just model training.

---

## Features

### Core ML Capabilities
- **Multi-Algorithm Comparison**: SVM (SMO), Naive Bayes, Random Forest, Logistic Regression
- **Advanced Preprocessing**: Mutual Information feature selection, TF-IDF vectorization, n-gram support
- **Comprehensive Evaluation**: ROC-AUC, PR-AUC, calibration metrics (Brier Score, ECE), confusion matrices
- **Thread-Safe Pipeline**: Concurrent inference with ReadWriteLock protection

### Production Engineering
- **REST API**: Spring Boot endpoints with rate limiting (Resilience4j)
- **Containerization**: Multi-stage Docker build with security best practices
- **Monitoring**: Spring Boot Actuator + Prometheus metrics
- **Configuration**: Profile-based config (dev/prod) with environment variable overrides
- **Error Handling**: Comprehensive validation and structured error responses

### Systems Design Highlights
- **Strategy Pattern**: Pluggable dataset loaders with auto-detection
- **Interface Segregation**: Clean classifier abstraction for algorithm swapping
- **Model Persistence**: Serialized pre-trained models with fallback training
- **Batch Processing**: Parallel predictions with order preservation

---

## Quick Start

### Prerequisites
- Docker (recommended) OR
- Java 21 + Maven 3.9+

### Option 1: Docker (Fastest)

```bash
# Build the image
docker build -t sentiment-analyzer .

# Run the container
docker run -p 8080:8080 sentiment-analyzer

# Test the API
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"This product exceeded my expectations!"}'
```

### Option 2: Maven (Local Development)

```bash
# Build the project
mvn clean package -DskipTests

# Run the application
java -jar target/sentiment-analyzer-1.0.0.jar

# Or use Maven Spring Boot plugin
mvn spring-boot:run
```

---

## API Usage

### Single Text Analysis

```bash
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The customer service was outstanding and resolved my issue immediately!",
    "confidenceThreshold": 0.7
  }'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.92,
  "text": "The customer service was outstanding and resolved my issue immediately!",
  "processingTimeMs": 45
}
```

### Batch Analysis

```bash
curl -X POST http://localhost:8080/api/v1/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Amazing product, highly recommend!",
      "Terrible experience, complete waste of money.",
      "It works okay, nothing special."
    ]
  }'
```

**Response:**
```json
{
  "results": [
    {"sentiment": "positive", "confidence": 0.94, "processingTimeMs": 32},
    {"sentiment": "negative", "confidence": 0.89, "processingTimeMs": 28},
    {"sentiment": "neutral", "confidence": 0.76, "processingTimeMs": 35}
  ],
  "totalProcessingTimeMs": 95
}
```

### Health Check

```bash
curl http://localhost:8080/api/v1/health
```

**Response:**
```json
{
  "status": "UP",
  "modelLoaded": true,
  "algorithmType": "SVM",
  "timestamp": "2025-11-12T10:30:45Z"
}
```

See [examples/](examples/) for more API usage patterns.

---

## Architecture

### Package Structure

```
src/main/java/sentiment/
├── data/               # Dataset loading with strategy pattern
│   ├── DatasetLoader.java          # Interface for pluggable loaders
│   ├── DatasetLoaderRegistry.java  # Auto-detection via compatibility testing
│   └── [CSV/Amazon/Product loaders]
│
├── preprocessing/      # Text processing pipeline
│   ├── TextPreprocessor.java       # Thread-safe preprocessing coordinator
│   ├── TFIDFFeatureExtractor.java  # Feature vectorization
│   ├── MutualInformationSelector.java  # Information-theoretic feature selection
│   └── [Tokenizer, Stopwords, Contraction expansion]
│
├── models/             # ML classifiers
│   ├── SentimentClassifier.java    # Common interface for all algorithms
│   ├── SVMClassifier.java          # Support Vector Machine (SMO)
│   ├── NaiveBayesClassifier.java   # Probabilistic classifier
│   ├── RandomForestClassifier.java # Ensemble method
│   ├── LogisticRegressionClassifier.java
│   └── [Evaluator, CrossValidator, Persistence]
│
├── evaluation/         # Performance metrics
│   ├── ClassifierEvaluationResult.java  # Comprehensive metrics
│   ├── AUCCalculator.java              # ROC-AUC computation
│   ├── CalibrationMetrics.java         # Brier Score, ECE, MCE
│   └── StratifiedDataSplitter.java     # Proper train/test splits
│
├── api/                # REST endpoints
│   ├── SentimentController.java    # Spring Boot controllers
│   ├── [Request/Response DTOs]
│   └── RestApiExceptionHandler.java  # Centralized error handling
│
├── config/             # Application configuration
│   └── SentimentConfiguration.java  # Spring @ConfigurationProperties
│
└── training/           # Offline model training
    └── ModelTrainingCLI.java  # CLI for batch training
```

### Design Patterns

- **Strategy Pattern**: `DatasetLoader` implementations for different data formats
- **Interface Segregation**: `SentimentClassifier` abstraction for algorithm swapping
- **Dependency Injection**: Spring-managed components throughout
- **Template Method**: `TextPreprocessor` fit/transform workflow
- **Factory Pattern**: `DatasetLoaderRegistry` for loader selection

### Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Weka over Deeplearning4j** | Mature ecosystem, interpretable models, lower resource requirements |
| **Spring Boot** | Industry-standard framework for Java APIs, excellent monitoring/observability |
| **Mutual Information for feature selection** | Provably optimal for discrete features, outperforms chi-squared |
| **ReadWriteLock for thread safety** | Allows concurrent reads during inference, exclusive writes during training |
| **Multi-stage Docker build** | Separates build dependencies from runtime, reduces image size by ~60% |
| **Pre-trained models** | Fast startup for production (< 5 seconds) vs. training on-demand (2-5 minutes) |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed design documentation.

---

## Performance

### Model Comparison

Trained on **Amazon Customer Reviews Polarity** dataset (10,000 samples, 50/50 positive/negative split).

| Algorithm | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Inference Time |
|-----------|----------|-----------|--------|----------|---------|----------------|
| **SVM (SMO)** | 89.2% | 0.891 | 0.893 | 0.892 | 0.945 | 35ms |
| **Random Forest** | 87.4% | 0.870 | 0.878 | 0.874 | 0.933 | 52ms |
| **Naive Bayes** | 85.7% | 0.853 | 0.859 | 0.856 | 0.921 | 18ms |
| **Logistic Regression** | 86.1% | 0.858 | 0.864 | 0.861 | 0.928 | 22ms |

**Calibration Metrics (SVM):**
- Brier Score: 0.123 (lower is better)
- Expected Calibration Error (ECE): 0.047
- Maximum Calibration Error (MCE): 0.092

**Key Insights:**
- SVM provides best accuracy/AUC balance
- Naive Bayes fastest for high-throughput scenarios
- Random Forest most stable across different text domains
- All models show good probability calibration (ECE < 0.05)

### API Performance

- **Single prediction**: 30-50ms (including preprocessing)
- **Batch prediction (100 texts)**: 1.2s (parallel processing)
- **Throughput**: ~1,000 predictions/minute (dev rate limit: 60/min)
- **Memory footprint**: ~512MB (JVM heap)
- **Cold start**: < 5 seconds (pre-trained model loading)

Tested on: MacBook Pro M1, 16GB RAM

---

## Dataset

**Training Data**: [Amazon Customer Reviews Polarity](https://www.kaggle.com/datasets/bhavikardeshna/amazon-customerreviews-polarity)

- **Source**: Kaggle (originally from Amazon product reviews)
- **Size**: ~400k reviews (10k subset used for training)
- **Classes**: Binary (Positive/Negative)
- **Format**: CSV with `text` and `sentiment` columns
- **License**: Public domain (Amazon reviews dataset)

### Data Setup

```bash
# Download from Kaggle
kaggle datasets download -d bhavikardeshna/amazon-customerreviews-polarity

# Extract to project directory
unzip amazon-customerreviews-polarity.zip -d data/datasets/

# Update application.yml
sentiment:
  data:
    training-file: /path/to/data/datasets/Reviews.csv
```

**Note**: Dataset not included in repository due to size (~200MB). Download separately for training.

---

## Configuration

### Environment Variables

```bash
# Model configuration
SENTIMENT_MODEL_TYPE=svm                    # svm, naive_bayes, random_forest, logistic_regression
SENTIMENT_CONFIDENCE_THRESHOLD=0.7          # Minimum confidence for classification

# Model paths (optional, uses pre-trained if available)
SENTIMENT_SVM_MODEL=/app/models/svm-model.ser
SENTIMENT_PREFER_PRETRAINED=true

# API configuration
SENTIMENT_API_MAX_BATCH_SIZE=100
SENTIMENT_API_RATE_LIMIT=1000

# Preprocessing
SENTIMENT_PREPROCESSING_MAX_FEATURES=5000
SENTIMENT_PREPROCESSING_USE_BIGRAMS=true
```

### Application Profiles

```bash
# Development profile (relaxed rate limits, verbose logging)
java -jar app.jar --spring.profiles.active=dev

# Production profile (strict limits, optimized performance)
java -jar app.jar --spring.profiles.active=production

# Test profile (in-memory data, mocked dependencies)
mvn test -Dspring.profiles.active=test
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for full configuration reference.

---

## Development

### Running Tests

```bash
# Run all tests
mvn test

# Run specific test class
mvn test -Dtest=SVMClassifierTest

# Run with coverage report
mvn clean test jacoco:report
# View report at target/site/jacoco/index.html
```

### Building from Source

```bash
# Clean build
mvn clean package

# Skip tests (faster)
mvn package -DskipTests

# Build Docker image
docker build -t sentiment-analyzer:latest .
```

### Project Structure

- **Source code**: `src/main/java/sentiment/`
- **Tests**: `src/test/java/sentiment/`
- **Configuration**: `src/main/resources/application.yml`
- **Pre-trained models**: `models/` (svm-model.ser, naive_bayes-model.ser, etc.)
- **Documentation**: `docs/`

---

## What I Learned

### Java ML Ecosystem vs. Python

**Strengths:**
- **Type Safety**: Compile-time guarantees eliminate entire classes of runtime errors common in Python ML pipelines
- **Concurrency**: JVM threading model makes parallel inference straightforward (ReadWriteLock, ExecutorService)
- **Deployment**: Single executable JAR with no dependency hell (vs. pip/conda conflicts)
- **Performance**: JVM warm-up compensates initial startup cost; mature garbage collection for long-running services
- **Enterprise Integration**: Seamless integration with Spring ecosystem (security, monitoring, service discovery)

**Trade-offs:**
- **Library Maturity**: Weka is stable but less cutting-edge than scikit-learn/PyTorch
- **Iteration Speed**: Compilation step slows experimentation vs. Python REPL
- **Deep Learning**: Limited DL support (Deeplearning4j exists but ecosystem smaller)
- **Notebook Workflow**: No Jupyter equivalent (though JShell + Gradle possible)

### Technical Insights

1. **Mutual Information feature selection** outperformed chi-squared for text classification (5-8% accuracy gain)
2. **Thread-safe preprocessing** is non-trivial - mutable TF-IDF state requires careful locking
3. **Model calibration matters** - raw SVM probabilities poorly calibrated without Platt scaling
4. **Rate limiting is essential** - even demo APIs need abuse protection (learned the hard way with runaway test script)
5. **Docker multi-stage builds** reduced image size from 850MB to 320MB (Maven dependencies in build stage only)

### Career Takeaways

- **Java remains essential** for enterprise ML deployment despite Python's research dominance
- **Production ML is 80% engineering** (API design, monitoring, error handling) and 20% modeling
- **Cross-language fluency is valuable** - demonstrates adaptability beyond language wars
- **Systems thinking matters more than algorithms** - clean architecture beats fancy models for long-term maintainability

---

## Roadmap

### Future Enhancements

- [ ] Multi-label classification (positive/negative/neutral/mixed)
- [ ] Confidence interval estimation via bootstrapping
- [ ] Model A/B testing framework
- [ ] Real-time streaming with Kafka integration
- [ ] Fine-grained sentiment (1-5 stars)
- [ ] Explainability (LIME/SHAP for Java)
- [ ] Kubernetes deployment manifests
- [ ] GraphQL API alternative

### Known Limitations

- Binary classification only (positive/negative)
- English language only
- Pre-trained models optimized for product reviews (may not generalize to social media text)
- No active learning / online training support
- Single-node deployment (no distributed inference)

---

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) - Design patterns and technical decisions
- [API Reference](docs/API_REFERENCE.md) - Complete endpoint documentation
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions
- [Experiment Log](docs/EXPERIMENT_LOG.md) - Model training experiments and results

---

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| **Language** | Java | 21 (LTS) |
| **ML Framework** | Weka | 3.9.6 |
| **Web Framework** | Spring Boot | 3.4.0 |
| **Build Tool** | Maven | 3.9+ |
| **Containerization** | Docker | - |
| **Testing** | JUnit 5 | 5.11.4 |
| **Monitoring** | Actuator + Prometheus | - |
| **Resilience** | Resilience4j | - |
| **Validation** | Hibernate Validator | 8.0+ |
| **Logging** | SLF4J + Logback | - |

---

## Contributing

This is a learning project and portfolio piece, but feedback is welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Run tests (`mvn test`)
4. Submit a pull request

---

## License

[MIT License](LICENSE) - feel free to use this project for learning and portfolio purposes.

---
