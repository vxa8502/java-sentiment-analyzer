# Deployment Guide

Simple deployment instructions for the Java Sentiment Analyzer.

---

## Quick Start: Docker

### Build and Run

```bash
# Build Docker image
docker build -t sentiment-analyzer .

# Run container
docker run -d \
  --name sentiment-api \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models \
  sentiment-analyzer

# Check health
curl http://localhost:8080/api/v1/health
```

### Test the API

```bash
# Analyze sentiment
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Great product, highly recommend!"}'
```

---

## Local Development

### Prerequisites

- Java 21
- Maven 3.9+

### Run Locally

```bash
# Build project
mvn clean package -DskipTests

# Run with Spring Boot
mvn spring-boot:run

# Or run JAR directly
java -jar target/sentiment-analyzer-1.0.0.jar
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPRING_PROFILES_ACTIVE` | `dev` | Application profile (`dev` or `production`) |
| `SENTIMENT_MODEL_TYPE` | `svm` | Algorithm: `svm`, `naive_bayes`, `random_forest` |
| `SENTIMENT_CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence for predictions |
| `SERVER_PORT` | `8080` | API port |
| `JAVA_OPTS` | `-Xmx512m` | JVM memory settings |

### Example with Custom Config

```bash
docker run -d \
  -p 8080:8080 \
  -e SPRING_PROFILES_ACTIVE=production \
  -e SENTIMENT_MODEL_TYPE=svm \
  -e JAVA_OPTS="-Xmx768m -Xms512m" \
  -v $(pwd)/models:/app/models \
  sentiment-analyzer
```

---

## Cloud Deployment (Optional)

### Google Cloud Run

```bash
# Build and push
gcloud builds submit --tag gcr.io/<project-id>/sentiment-analyzer

# Deploy
gcloud run deploy sentiment-analyzer \
  --image gcr.io/<project-id>/sentiment-analyzer \
  --platform managed \
  --region us-central1 \
  --memory 1Gi \
  --port 8080
```

**Note:** You don't need cloud deployment until you have actual traffic that justifies it. Start with Docker locally.

---

## Health Check & Monitoring

### Health Endpoint

```bash
curl http://localhost:8080/api/v1/health | jq .
```

**Response:**
```json
{
  "status": "UP",
  "version": "1.0.0",
  "modelLoaded": true,
  "modelType": "SVM",
  "uptimeMs": 123456,
  "productionMetrics": {
    "totalPredictions": 1523,
    "labelDistribution": {
      "positive": 892,
      "negative": 631
    },
    "averageConfidence": 0.847,
    "lowConfidenceRatePercent": 3.2,
    "latencyStats": {
      "meanMs": 12.3,
      "p95Ms": 23.0,
      "p99Ms": 45.0
    }
  }
}
```

### Key Metrics to Monitor

1. **Model Performance:**
   - `averageConfidence` - Should stay > 0.7 (model degradation if dropping)
   - `lowConfidenceRatePercent` - Alert if > 10%
   - `labelDistribution` - Check for label imbalance

2. **System Performance:**
   - `latencyStats.p99Ms` - Alert if > 100ms
   - HTTP 503 errors - Model not loaded

### Prometheus Metrics

Exposed at `/actuator/prometheus`:
- `sentiment_predictions_total{label}` - Prediction counts by label
- `sentiment_inference_duration_seconds` - Latency histogram
- `sentiment_prediction_confidence` - Confidence distribution
- `sentiment_predictions_low_confidence_total` - Low confidence counter

---

## Troubleshooting

### Issue: Container Fails to Start

**Check logs:**
```bash
docker logs sentiment-api
```

**Common causes:**
- Model file not found: Mount models directory with `-v $(pwd)/models:/app/models`
- Out of memory: Increase with `-m 1g` or set `JAVA_OPTS="-Xmx768m"`

### Issue: Predictions Are Low Quality

**Check production metrics:**
```bash
curl http://localhost:8080/api/v1/health | jq '.productionMetrics'
```

**Red flags:**
- `averageConfidence` < 0.6 → Model degradation or domain shift
- `lowConfidenceRatePercent` > 15% → Input data doesn't match training distribution

**Solution:** Retrain model on production-like data (see [TRAINING.md](TRAINING.md))

### Issue: Slow Response Times

**Check latency:**
```bash
curl http://localhost:8080/api/v1/health | jq '.productionMetrics.latencyStats'
```

**Solutions:**
1. Use faster algorithm: Switch to Naive Bayes (`SENTIMENT_MODEL_TYPE=naive_bayes`)
2. Increase CPU: `docker run --cpus 2 ...`
3. Reduce features: Retrain with fewer max features

---

## Security

### Non-Root Container

Dockerfile already runs as non-root user (UID 1001).

### Network Security

```bash
# Create isolated network
docker network create --driver bridge sentiment-net
docker run --network sentiment-net sentiment-analyzer
```

### Secrets Management

**DO NOT** hardcode API keys or secrets. Use environment variables:

```bash
docker run -e SENTIMENT_API_KEY=$(cat api-key.txt) sentiment-analyzer
```

---

## Performance

### Single Instance Benchmarks

- **Startup Time:** < 5 seconds
- **Memory:** 512MB (steady state)
- **Throughput:** ~1000 requests/minute
- **Latency:** 10-30ms (p50), 40-80ms (p99)

### Scaling

Horizontal scaling via load balancer:
- 1 instance → 1K req/min
- 3 instances → 3K req/min
- 10 instances → 9.5K req/min

---

For detailed training instructions, see [TRAINING.md](TRAINING.md).

**Last Updated:** 2024-12-12
