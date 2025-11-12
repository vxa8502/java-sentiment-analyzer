# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-12

### Added
- **Core ML Functionality**
  - Multi-algorithm sentiment classification (SVM, Naive Bayes, Random Forest, Logistic Regression)
  - Advanced text preprocessing pipeline with TF-IDF vectorization
  - Mutual Information feature selection for optimal vocabulary
  - Thread-safe inference with ReadWriteLock
  - Comprehensive evaluation metrics (ROC-AUC, PR-AUC, calibration metrics)
  - Pre-trained models with serialization/deserialization support

- **REST API**
  - Spring Boot REST endpoints for sentiment analysis
  - Single text analysis endpoint (`POST /api/v1/sentiment/analyze`)
  - Batch analysis endpoint with parallel processing (`POST /api/v1/sentiment/batch`)
  - Health check endpoint (`GET /api/v1/health`)
  - Bean validation for request parameters
  - Centralized exception handling with structured error responses
  - Rate limiting via Resilience4j (configurable by profile)

- **Data Loading**
  - Pluggable dataset loader architecture with strategy pattern
  - Auto-detection registry for format identification
  - Support for Amazon Polarity, Product Reviews, and Service Reviews datasets
  - Compatibility testing for loader selection

- **Production Features**
  - Docker containerization with multi-stage build
  - Security-hardened Dockerfile (non-root user, minimal attack surface)
  - Spring Boot Actuator for monitoring and health checks
  - Prometheus metrics export
  - Profile-based configuration (dev, production, test)
  - Environment variable configuration overrides
  - JVM tuning for container environments

- **Developer Experience**
  - Comprehensive unit tests (12 test classes, 5,621 lines)
  - Integration tests for API endpoints
  - Maven build system with Spring Boot plugin
  - Detailed logging with SLF4J/Logback
  - Model training CLI for offline training

- **Documentation**
  - Comprehensive README with "Why Java" narrative and quick start
  - Architecture documentation with design patterns
  - Complete API reference with examples
  - Deployment guide for local, Docker, and cloud platforms
  - Usage examples in Bash, Python, JavaScript, Java, Go, Ruby
  - Experiment log template for model training results

- **Configuration & Deployment**
  - docker-compose.yml for easy local deployment
  - Configurable preprocessing parameters
  - Configurable model paths and algorithm selection
  - Configurable API rate limits and validation rules
  - Support for custom confidence thresholds

### Technical Highlights
- **Package Architecture**: Clean separation of concerns across 9 packages (data, preprocessing, models, evaluation, api, config, training, util)
- **Design Patterns**: Strategy, Template Method, Dependency Injection, Registry, Immutable Value Objects
- **Thread Safety**: ReadWriteLock for concurrent inference during model training
- **Calibration**: Platt scaling for SVM probability calibration
- **Batch Processing**: Order-preserving parallel execution with ForkJoinPool

### Performance
- Startup time: < 5 seconds (with pre-trained models)
- Single request latency: 30-50ms
- Batch (100 texts) latency: 1-2 seconds
- Throughput: ~1,000 requests/minute (single instance)
- Memory footprint: 512MB (steady state)

### Dependencies
- Java 21 (LTS)
- Spring Boot 3.4.0
- Weka 3.9.6
- Maven 3.9+
- JUnit 5.11.4
- Resilience4j (rate limiting)
- Hibernate Validator 8.0+

### Dataset
- Training data: Amazon Customer Reviews Polarity (Kaggle)
- Model accuracy: 85-89% (depending on algorithm)
- Binary classification: Positive/Negative sentiment

---

## [Unreleased]

### Planned
- Multi-label classification (positive/negative/neutral/mixed)
- Confidence interval estimation via bootstrapping
- Model A/B testing framework
- Real-time streaming with Kafka integration
- Explainability features (LIME/SHAP for Java)
- Kubernetes deployment manifests
- GraphQL API alternative
- API authentication (API key support)
- Fine-grained sentiment (1-5 star ratings)
