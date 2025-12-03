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

## API Endpoints

### Health Check

- **GET** `/api/v1/health`

  Returns the health of the service, including model status and uptime.

### Analyze Sentiment

- **POST** `/api/v1/sentiment/analyze`

  Analyzes the sentiment of a single text.

  **Request Body:**

  ```json
  {
    "text": "This is a wonderful product!",
    "confidenceThreshold": 0.8
  }
  ```

### Batch Analyze Sentiment

- **POST** `/api/v1/sentiment/batch`

  Analyzes the sentiment of multiple texts in a batch.

  **Request Body:**

  ```json
  {
    "texts": [
      "This is a great movie.",
      "I am not happy with the service.",
      "The weather is neutral."
    ],
    "confidenceThreshold": 0.75
  }
  ```

### Get Feature Importance

- **GET** `/api/v1/model/feature-importance`

  Returns the most important features (words) for the sentiment model.

  **Query Parameters:**

  - `topFeatures` (optional, default: 30): The number of top features to return.

## Usage

Here are some examples of how to use the API with `curl`.

### Positive Sentiment

```bash
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"This product is absolutely amazing! Best purchase ever!"}'
```

### Negative Sentiment

```bash
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Terrible quality. Complete waste of money. Very disappointed."}'
```

### Batch Analysis

```bash
curl -X POST http://localhost:8080/api/v1/sentiment/batch \
  -H "Content-Type: application/json" \
  -d 
    "texts": [
      "Amazing product, highly recommend!",
      "Terrible experience, very disappointed.",
      "It works as expected, nothing special."
    ]
  }
```

## Training a New Model

You can train a new sentiment analysis model using the provided training module.

### Command

```bash
mvn exec:java -Dexec.mainClass="sentiment.training.TrainModel" \
  -Dexec.args="<dataPath> <outputPath> [maxSamples] [showFeatureImportance] [topFeaturesCount] [enableHyperparameterTuning]"
```

### Example

```bash
mvn exec:java -Dexec.mainClass="sentiment.training.TrainModel" \
  -Dexec.args="./src/main/resources/datasets/v1_raw/Reviews.csv ./models/svm-model-v2.ser 10000 true 30 false"
```

This command will train a new SVM model on the `Reviews.csv` dataset and save it to `./models/svm-model-v2.ser`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
