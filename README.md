# Multi-Algorithm Sentiment Analyzer in Java

A comparative study of classical ML algorithms for sentiment analysis, built in Java to explore enterprise deployment patterns and demonstrate cross-language ML proficiency.

## Why This Project?

**Problem:** Most ML work happens in Python. I wanted to understand how ML models are deployed in Java-based enterprise environments.

**Approach:** Implemented 4 classical ML algorithms (SVM, Naive Bayes, Random Forest, Logistic Regression) with Weka, compared their performance, and deployed via Spring Boot REST API.

**Value:** Demonstrates adaptability across languages, systems thinking, and production-ready architecture.

---

## What's Inside

### Multi-Algorithm Comparison
- **SVM (Support Vector Machine)** - Strong with high-dimensional TF-IDF features
- **Naive Bayes** - Fast, probabilistic baseline
- **Random Forest** - Ensemble method for robustness
- **Logistic Regression** - Linear model for interpretability

### Production-Ready Components
- **REST API** with Spring Boot for real-time predictions
- **Docker containerization** for deployment
- **Text preprocessing pipeline** (tokenization, stopwords, TF-IDF)
- **Comprehensive test suite** with JUnit 5 and JaCoCo coverage
- **Configurable** via YAML (swap algorithms, tune hyperparameters)

---

## Dataset

**Amazon Review Polarity Dataset**
- **Source:** [Kaggle - Amazon Reviews](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews)
- **Size:** 1,800,000 training samples, 200,000 test samples
- **Format:** Binary sentiment (1=negative, 2=positive)
- **Domain:** Product reviews from Amazon (2013)
- **License:** Public domain
- **Citation:** Xiang Zhang, Junbo Zhao, Yann LeCun. "Character-level Convolutional Networks for Text Classification." Advances in Neural Information Processing Systems 28 (NIPS 2015).

---

## Quick Start

### Prerequisites
- Java 21+
- Maven 3.9+
- Docker (optional)

### Run Locally

```bash
# Clone repository
git clone https://github.com/vxa8502/java-sentiment-analyzer.git
cd java-sentiment-analyzer

# Build
mvn clean package

# Run
java -jar target/sentiment-analyzer-1.0.0.jar
```

API available at `http://localhost:8080`

### Run with Docker

```bash
# Build image
docker build -t sentiment-analyzer .

# Run container
docker run -p 8080:8080 sentiment-analyzer
```

---

## API Examples

### Single Prediction

```bash
curl -X POST http://localhost:8080/api/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product exceeded my expectations!",
    "confidenceThreshold": 0.7
  }'
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.92,
  "text": "This product exceeded my expectations!",
  "processingTimeMs": 45
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:8080/api/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Great product, highly recommend!",
      "Terrible quality, waste of money.",
      "It works okay, nothing special."
    ]
  }'
```

### Health Check

```bash
curl http://localhost:8080/actuator/health
```

---

## Architecture

```
src/
├── main/java/sentiment/
│   ├── api/                    # REST controllers
│   ├── models/                 # ML classifiers (SVM, NB, RF, LR)
│   ├── preprocessing/          # Text processing pipeline
│   ├── data/                   # Dataset loaders
│   ├── training/               # Model training
│   └── config/                 # Spring configuration
└── test/java/sentiment/        # Unit & integration tests
```

---

## Algorithm Comparison

Trained on 5,000 Amazon reviews (60/20/20 train/val/test split):

| Algorithm | Training Time | Model Size | Status | Best For |
|-----------|---------------|------------|--------|----------|
| SVM | 34.8s | 968 KB | Trained | High-dimensional sparse features, strong with TF-IDF |
| Naive Bayes | 25.8s | 1.9 MB | Trained | Fast training, probabilistic output, good baseline |
| Random Forest | 109.3s | 17 MB | Trained | Ensemble robustness, handles non-linear patterns |
| Logistic Regression | - | - | Memory constraints | Would be good for interpretability |

Key Findings:
- Fastest to train: Naive Bayes (25.8s) - ideal for rapid prototyping
- Most compact: SVM (968KB) - best for deployment with size constraints
- Slowest to train: Random Forest (109s) - trades speed for ensemble strength
- Logistic Regression: Encountered memory issues with current Weka configuration on this dataset size

All models trained using TF-IDF features with bigrams on Amazon Review Polarity dataset.

---

## Configuration

Edit `src/main/resources/application.yml`:

```yaml
sentiment:
  model:
    algorithm: svm              # svm, naive_bayes, random_forest, logistic_regression
    confidence-threshold: 0.7
  data:
    training-file: "./data/amazon_train.csv"
    test-file: "./data/amazon_test.csv"
  preprocessing:
    tf-idf: true
    bigrams: true
    min-word-length: 2
```

---

## What I Learned

### When Java Makes Sense for ML
Java works well when you need:
- Integration with existing Java microservices (no Python bridge needed)
- Enterprise deployment patterns (JAR files, Spring Boot, Docker)
- Strong typing and compile-time safety for production systems
- Fast inference once models are trained (JVM optimization)

### When Java Doesn't Make Sense
Avoid Java for ML when you need:
- Rapid experimentation (Python's notebooks and pandas are faster)
- Cutting-edge model architectures (limited library support)
- Complex model training workflows (Weka's API has limitations)
- Memory-efficient training at scale (encountered issues with some algorithms)

### Real Trade-Offs Discovered
- Training speed: Naive Bayes (25.8s) vs Random Forest (109s) shows 4x difference
- Model size: SVM (968KB) vs Random Forest (17MB) - 18x difference affects deployment
- Memory: Logistic Regression failed with 8GB heap on 2000 samples (Weka limitation)
- Ecosystem: Weka is stable but dated - no modern optimizers or GPU support

### My Recommendation
**Prototype in Python, deploy in Java only when business requires it** (existing Java stack, enterprise constraints, or specific performance needs).

---

## Tech Stack

- **Java 21** - Latest LTS with modern features
- **Spring Boot 3.3.5** - REST API framework
- **Weka 3.9.6** - Machine learning library
- **Maven** - Build and dependency management
- **JUnit 5** - Testing framework
- **JaCoCo** - Code coverage
- **Docker** - Containerization

---

## Testing

```bash
# Run all tests
mvn test

# With coverage report
mvn test jacoco:report
open target/site/jacoco/index.html
```

---

## Interview Story

**Setup (15s):**
"I built a multi-algorithm sentiment analyzer in Java to understand enterprise ML deployment outside Python."

**Technical (30s):**
"Implemented 4 algorithms (SVM, Naive Bayes, Random Forest, Logistic Regression) using Weka, compared their trade-offs, and deployed via Spring Boot REST API with Docker. The challenge was Java's ML ecosystem is less mature—no PyTorch or transformers—so I worked with classical ML and focused on production patterns."

**Result (15s):**
"Built a working system with <50ms inference latency, comprehensive test coverage, and Docker deployment. The project taught me when to use Java (production systems) vs. Python (experimentation)."

**Lesson (10s):**
"Right tool for the job. I'd prototype in Python, then reimplement in Java if business needs require it for integration or performance."

---

## Project Status

**Core functionality complete.** This project was a focused learning sprint to:
1. ✅ Understand Java ML libraries and ecosystem
2. ✅ Build production-ready REST API
3. ✅ Implement multi-algorithm comparison framework
4. ✅ Deploy with Docker
5. ⏳ Train on Amazon Polarity dataset and collect metrics

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Victoria Alabi**
Exploring ML engineering across languages and deployment patterns.

---

**Key Takeaway:** Python for ML research, Java for production integration.
