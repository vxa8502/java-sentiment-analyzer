# Java Sentiment Analyzer

A multi-algorithm sentiment analysis system in Java, built with Spring Boot and Weka.

## Description

This project is a high-performance sentiment analysis service that provides a REST API for analyzing the sentiment of text. It supports single and batch text analysis, confidence thresholds, and provides detailed model performance metrics. The system is designed for scalability and can be easily extended with new machine learning models.

## Features

- **REST API:** Simple and intuitive API for sentiment analysis.
- **Batch Processing:** Analyze multiple texts in a single request for efficiency.
- **Multiple Algorithms:** Supports SVM, Naive Bayes, and other classifiers (extendable).
- **Confidence Scores:** Provides confidence scores for each prediction.
- **Model Training:** Includes a command-line tool for training new models.
- **Health Check:** Endpoint for monitoring service status and model information.
- **Feature Importance:** Analyze which words have the most impact on sentiment.
- **Docker Support:** Comes with a `Dockerfile` and `docker-compose.yml` for easy deployment.

## Technology Stack

- **Java 21**
- **Spring Boot 3:** For the REST API and application framework.
- **Weka 3.9:** For machine learning algorithms.
- **Maven:** For project build and dependency management.
- **Jackson:** For JSON processing.
- **Resilience4j:** For rate limiting.
- **Docker:** For containerization.

## Getting Started

### Prerequisites

- Java 21 or later
- Maven 3.6 or later
- Docker (optional, for containerized deployment)

### Building the Project

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/java-sentiment-analyzer.git
    cd java-sentiment-analyzer
    ```

2.  Build the project using Maven:
    ```bash
    mvn clean install
    ```

### Running the Application

You can run the application using the Spring Boot Maven plugin:

```bash
mvn spring-boot:run
```

The application will start on port `8080`.

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

## Training Models

```bash
mvn exec:java -Dexec.mainClass="sentiment.training.TrainModel" \
  -Dexec.args="./data/Reviews.csv ./models/svm-model.ser 10000 true 30 false"
```

This trains an SVM model and generates:
- `svm-model.ser` - Trained model
- `svm-model.metadata.json` - Training metrics and reproducibility info
- `svm-model.feature-importance.json` - Feature analysis

**For detailed training documentation, see [TRAINING.md](docs/TRAINING.md)**

## Documentation

- **[TRAINING.md](docs/TRAINING.md)** - Model training, comparison, and reproducibility
- **[DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Docker deployment and production monitoring

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
