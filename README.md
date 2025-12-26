# Java Sentiment Analyzer

A multi-algorithm sentiment analysis system in Java, built with Spring Boot and Weka.

## Description

This project is a high-performance sentiment analysis service that provides a REST API for analyzing the sentiment of text. It supports single and batch text analysis, confidence thresholds, and provides detailed model performance metrics. The system is designed for scalability and can be easily extended with new machine learning models.

## Features

- **REST API:** Simple and intuitive API for sentiment analysis.
- **Batch Processing:** Analyze multiple texts in a single request for efficiency.
- **Multiple Algorithms:** SVM, Naive Bayes, Random Forest, Logistic Regression.
- **Confidence Scores:** Provides confidence scores for each prediction.
- **Cross-Domain Evaluation:** 12 models trained, tested across 3 domains (movies, products, restaurants).
- **Edge Case Testing:** 200 real failures tested (sarcasm, negation, mixed sentiment, jargon).
- **Error Analysis:** Built-in tools to find and categorize model failures.
- **Docker Support:** Comes with a `Dockerfile` and `docker-compose.yml` for easy deployment.

## Production Model

**Deployed Model**: Logistic Regression (trained on Amazon Product Reviews)

| Metric | Performance |
|--------|-------------|
| **Cross-domain average** | 83.0% (rank #2) |
| **Edge case accuracy** | **62.5% (rank #1)**  |
| **In-domain (Amazon)** | 87.5% |
| **Generalization** | IMDB: 83.5%, Yelp: 82.5% |
| **Model size** | 17 MB |
| **Inference speed** | ~10ms per prediction |

**Why this model?**
- Best balance of cross-domain generalization and edge case robustness
- Handles sarcasm (58%), negation (60%), and mixed sentiment (64%)
- Well-calibrated confidence scores (Brier: 0.095)
- Fast inference and production-ready

## Evaluation Results

- **12 Models Trained**: 4 algorithms Ã— 3 domains
- **Best Cross-Domain**: SVM Amazon (84.3%)
- **Best Edge Cases**: Logistic Regression Amazon (62.5%)
- **Most Robust**: Logistic Regression Amazon (deployed)
- **200 Edge Cases**: Real failures from 18,000+ prediction errors
- See [results/FINAL_COMPREHENSIVE_REPORT.md](results/FINAL_COMPREHENSIVE_REPORT.md) for complete evaluation

## Technology Stack

- **Java 21**
- **Spring Boot 3:** For the REST API and application framework.
- **Weka 3.9:** For machine learning algorithms.
- **Maven:** For project build and dependency management.
- **Jackson:** For JSON processing.
- **Resilience4j:** For rate limiting.
- **Docker:** For containerization.

## Quick Start

**Clone and run with production model (< 1 minute):**

```bash
git clone https://github.com/your-username/java-sentiment-analyzer.git
cd java-sentiment-analyzer
docker-compose up
```

The API will start on http://localhost:8080 using the **production model** (Logistic Regression trained on Amazon reviews).

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

---

## Three Ways to Use This Project

### 1ƒ£ Quick Demo (95% of users)

**Just want to see it working?**

```bash
git clone https://github.com/your-username/java-sentiment-analyzer.git
cd java-sentiment-analyzer
docker-compose up
```

 Uses production model (included in repo)
 ~50MB repo size
 Works immediately

---

### 2ƒ£ Reproduce Full Evaluation (Researchers)

**Want all 12 models to compare performance?**

```bash
git clone https://github.com/your-username/java-sentiment-analyzer.git
cd java-sentiment-analyzer

# Download all pre-trained models (~294MB)
./scripts/download_pretrained_models.sh

# Re-run cross-domain evaluation
./scripts/evaluate_cross_domain.sh

# Test on edge cases
./scripts/evaluate_edge_cases.sh all

# View results
cat results/cross_domain_matrix.json
```

 All 12 models included
 Full evaluation reproducible
 ~30 second download

---

### 3ƒ£ Retrain from Scratch (ML Engineers)

**Want to modify and retrain?**

```bash
git clone https://github.com/your-username/java-sentiment-analyzer.git
cd java-sentiment-analyzer

# Train all 12 models (~2 hours)
./scripts/train_all_models.sh

# Models saved to models/{svm,naive_bayes,random_forest,logistic_regression}/
```

 Full training pipeline
 Modify preprocessing, hyperparameters
 Train on your own data

---

## Prerequisites

- **Docker** (for quick start)
- **Java 21** (for local development)
- **Maven 3.6+** (for building from source)

## API Usage

### Single Text Analysis

```bash
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"This product is amazing!"}'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.94,
  "processingTimeMs": 42
}
```

### Batch Analysis

```bash
curl -X POST http://localhost:8080/api/v1/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["Great product!", "Terrible quality.", "It works."]}'
```

### Health Check (with Production Metrics)

```bash
curl http://localhost:8080/api/v1/health
```

**Response includes:**
- Model status and algorithm
- Production metrics: confidence, latency, label distribution
- See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for monitoring details

## Cross-Domain Evaluation

Test all models across all domains:
```bash
./scripts/evaluate_cross_domain.sh
```

Analyze prediction errors:
```bash
./scripts/analyze_errors.sh svm imdb_50k --export --top-n 50
```

Test models on edge cases:
```bash
./scripts/evaluate_edge_cases.sh all
```

## Datasets

- **IMDB 50K**: 25K train, 10K test (movie reviews)
- **Amazon Polarity**: 100K train, 20K test (product reviews)
- **Yelp**: 100K train, 20K test (restaurant reviews)

See [datasets/README.md](datasets/README.md) for sources, biases, and edge case details.

## Model Details

### All 12 Models

| Algorithm | Training Domain | Cross-Domain Avg | Edge Case Accuracy |
|-----------|----------------|------------------|-------------------|
| **Logistic Regression** | **Amazon** | **83.0%** | **62.5%**  |
| SVM | Amazon | **84.3%**  | 57.5% |
| Random Forest | Yelp | 68.2% | 61.5%  |
| SVM | IMDB | 80.6% | 46.0% |
| Naive Bayes | Amazon | 68.5% | 12.5%  |

**Download all models**: `./scripts/download_pretrained_models.sh`

### Training Your Own Models

```bash
# Train a specific model
mvn exec:java -Dexec.mainClass="sentiment.training.TrainModel" \
  -Dexec.args="./data/Reviews.csv ./models/svm-model.ser 10000 true"

# Or train all 12 models
./scripts/train_all_models.sh
```

Each training run generates:
- `model.ser` - Trained model
- `model.metadata.json` - Training metrics, hyperparameters, reproducibility info
- `model.feature-importance.json` - Feature analysis (if available)

**For detailed training documentation, see [TRAINING.md](docs/TRAINING.md)**

## Documentation

- **[TRAINING.md](docs/TRAINING.md)** - Model training, comparison, and reproducibility
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Docker deployment and production monitoring

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
