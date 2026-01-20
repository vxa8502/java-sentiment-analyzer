# Deployment Guide

Production deployment and configuration for the Java Sentiment Analyzer.

## Quick Start (Docker Compose)

```bash
docker-compose up
```

API available at http://localhost:8080.

## Local Development

### Prerequisites

- Java 21
- Maven 3.9+
- Production model in `models/production/` (run `./scripts/promote_to_production.sh` first)

### Run Locally

```bash
# Build
mvn clean package -DskipTests

# Run with Spring Boot
mvn spring-boot:run

# Or run JAR directly
java -jar target/sentiment-analyzer-*.jar
```

## Docker

### Build Image

```bash
docker build -t sentiment-analyzer .
```

### Run Container

```bash
docker run -d \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models \
  sentiment-analyzer
```

### With Custom Configuration

```bash
docker run -d \
  -p 8080:8080 \
  -e SPRING_PROFILES_ACTIVE=production \
  -e MAX_HEAP=1g \
  -e MIN_HEAP=512m \
  -v $(pwd)/models:/app/models \
  sentiment-analyzer
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPRING_PROFILES_ACTIVE` | `production` | Application profile |
| `SENTIMENT_MODEL_PATH` | `/app/models/production/sentiment_model.ser` | Path to model file |
| `SENTIMENT_CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence for predictions |
| `MAX_HEAP` | `512m` | JVM max heap size |
| `MIN_HEAP` | `256m` | JVM initial heap size |
| `SERVER_PORT` | `8080` | API port |

### Preprocessing Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTIMENT_PREPROCESSING_MAX_FEATURES` | `5000` | Max vocabulary size |
| `SENTIMENT_PREPROCESSING_MIN_WORD_LENGTH` | `2` | Min word length |
| `SENTIMENT_PREPROCESSING_USE_TFIDF` | `true` | Use TF-IDF weighting |
| `SENTIMENT_PREPROCESSING_USE_BIGRAMS` | `true` | Include bigrams |
| `SENTIMENT_MI_THRESHOLD` | `50000` | MI feature selection threshold |

### Training Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTIMENT_RANDOM_SEED` | `42` | Random seed for reproducibility |
| `SENTIMENT_DATA_IMDB` | `data/raw/imdb_50k/IMDB Dataset.csv` | IMDB dataset path |
| `SENTIMENT_DATA_AMAZON` | `data/raw/amazon_polarity/train.csv` | Amazon dataset path |
| `SENTIMENT_DATA_YELP` | `data/raw/yelp/yelp_reviews.csv` | Yelp dataset path |

### API Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTIMENT_API_MAX_BATCH_SIZE` | `100` | Max texts per batch request |
| `SENTIMENT_API_VALIDATION_MAX_TEXT_LENGTH` | `10000` | Max characters per text |

**Rate Limits (configured in `application.yml`):**
- Single analysis: 100 requests/min
- Batch analysis: 20 requests/min
- Model comparison: 2 requests/5min

## Health Check & Monitoring

### Health Endpoint

```bash
curl http://localhost:8080/api/v1/health
```

### Key Metrics to Monitor

1. **Model Status:**
   - `modelLoaded` - Should be `true`
   - `averageConfidence` - Alert if dropping below 0.7

2. **System Performance:**
   - Response latency - Alert if p99 > 100ms
   - HTTP 503 errors - Model not loaded

### Spring Actuator

Additional endpoints at `/actuator/`:
- `/actuator/health` - Basic health
- `/actuator/info` - App info
- `/actuator/prometheus` - Prometheus metrics (if enabled)

## Troubleshooting

### Container Fails to Start

**Check logs:**
```bash
docker logs sentiment-analyzer
```

**Common causes:**
- Model file not found: Ensure `models/production/sentiment_model.ser` exists
- Out of memory: Increase with `-e MAX_HEAP=1g`

### Model Not Found

```bash
# Verify model exists
ls -la models/production/

# If missing, promote a trained model
./scripts/promote_to_production.sh
```

### Slow Response Times

**Solutions:**
1. Increase memory: `-e MAX_HEAP=1g`
2. Increase CPU: `docker run --cpus 2 ...`
3. Check model size - smaller vocabulary = faster inference

### Out of Memory

```bash
# Increase heap
docker run -e MAX_HEAP=1g -e MIN_HEAP=512m ...

# Or in docker-compose.yml
environment:
  - MAX_HEAP=1g
  - MIN_HEAP=512m
```

## Security

### Non-Root Container

The Dockerfile runs as non-root user (UID 1001) by default.

### Network Isolation

```bash
# Use docker-compose network (automatic)
docker-compose up

# Or create isolated network manually
docker network create sentiment-net
docker run --network sentiment-net sentiment-analyzer
```

## Performance

### Expected Performance

| Metric | Value |
|--------|-------|
| Startup time | < 10 seconds |
| Memory (steady state) | 512MB - 1GB |
| Single request latency | 10-50ms |
| Batch (100 texts) | 1-2 seconds |
| Throughput | ~1000 req/min per instance |

### Resource Limits

Set in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 1G
    reservations:
      cpus: '1'
      memory: 512M
```

## Production Checklist

- [ ] Production model exists (`models/production/sentiment_model.ser`)
- [ ] Model metadata exists (`models/production/sentiment_model.metadata.json`)
- [ ] Health endpoint responds (`/api/v1/health`)
- [ ] Memory limits configured appropriately
- [ ] Logs accessible for debugging

---

For model training, see [TRAINING.md](TRAINING.md).
