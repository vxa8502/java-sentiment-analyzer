# Deployment Guide

Complete guide for deploying the Java Sentiment Analyzer in various environments.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Production Deployment](#production-deployment)
5. [Configuration](#configuration)
6. [Monitoring & Observability](#monitoring--observability)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### For Local Development

- **Java 21** (JDK)
  ```bash
  # Check Java version
  java -version
  # Should show: openjdk version "21.x.x"
  ```

- **Maven 3.9+**
  ```bash
  # Check Maven version
  mvn -version
  # Should show: Apache Maven 3.9.x or higher
  ```

- **Training Dataset** (optional, for model training)
  - Download from [Kaggle: Amazon Customer Reviews Polarity](https://www.kaggle.com/datasets/bhavikardeshna/amazon-customerreviews-polarity)
  - Place in `data/datasets/Reviews.csv`

### For Docker Deployment

- **Docker** 20.10+
  ```bash
  # Check Docker version
  docker --version
  # Should show: Docker version 20.10.x or higher
  ```

- **Docker Compose** (optional, for multi-container setups)
  ```bash
  # Check Docker Compose version
  docker compose version
  # Should show: Docker Compose version 2.x.x
  ```

---

## Local Development

### 1. Clone Repository

```bash
git clone https://github.com/victoriaalabi/java-sentiment-analyzer.git
cd java-sentiment-analyzer
```

### 2. Build Project

```bash
# Clean and build
mvn clean package

# Skip tests for faster build
mvn clean package -DskipTests

# Build output: target/sentiment-analyzer-1.0.0.jar
```

### 3. Run Application

**Option A: Maven Spring Boot Plugin**
```bash
# Run with default configuration
mvn spring-boot:run

# Run with specific profile
mvn spring-boot:run -Dspring-boot.run.profiles=dev

# Run with custom port
mvn spring-boot:run -Dspring-boot.run.arguments=--server.port=9090
```

**Option B: Java JAR**
```bash
# Run with default configuration
java -jar target/sentiment-analyzer-1.0.0.jar

# Run with specific profile
java -jar target/sentiment-analyzer-1.0.0.jar --spring.profiles.active=dev

# Run with environment variables
SENTIMENT_MODEL_TYPE=naive_bayes java -jar target/sentiment-analyzer-1.0.0.jar
```

### 4. Verify Deployment

```bash
# Check health endpoint
curl http://localhost:8080/api/v1/health

# Test sentiment analysis
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Great product!"}'
```

---

## Docker Deployment

### 1. Build Docker Image

```bash
# Build image
docker build -t sentiment-analyzer:latest .

# Build with specific tag
docker build -t sentiment-analyzer:1.0.0 .

# Build for different platform (e.g., ARM64)
docker build --platform linux/arm64 -t sentiment-analyzer:latest .
```

**Build Process:**
- **Stage 1**: Maven build (downloads dependencies, compiles code, packages JAR)
- **Stage 2**: Runtime image (copies JAR, sets up user, configures healthcheck)

**Image Size**: ~320MB (multi-stage build optimization)

### 2. Run Container

**Basic Deployment:**
```bash
docker run -d \
  --name sentiment-api \
  -p 8080:8080 \
  sentiment-analyzer:latest
```

**With Volume Mounts** (for custom models/data):
```bash
docker run -d \
  --name sentiment-api \
  -p 8080:8080 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  sentiment-analyzer:latest
```

**With Environment Variables:**
```bash
docker run -d \
  --name sentiment-api \
  -p 8080:8080 \
  -e SPRING_PROFILES_ACTIVE=production \
  -e SENTIMENT_MODEL_TYPE=svm \
  -e SENTIMENT_CONFIDENCE_THRESHOLD=0.8 \
  sentiment-analyzer:latest
```

**With Resource Limits:**
```bash
docker run -d \
  --name sentiment-api \
  -p 8080:8080 \
  -m 1g \
  --cpus 2 \
  -e JAVA_OPTS="-Xmx768m -Xms512m" \
  sentiment-analyzer:latest
```

### 3. Docker Compose

**File**: `docker-compose.yml`

```yaml
version: '3.8'

services:
  sentiment-api:
    build: .
    container_name: sentiment-analyzer
    ports:
      - "8080:8080"
    environment:
      - SPRING_PROFILES_ACTIVE=production
      - SENTIMENT_MODEL_TYPE=svm
      - SENTIMENT_CONFIDENCE_THRESHOLD=0.7
      - JAVA_OPTS=-Xmx512m -Xms256m
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/api/v1/health"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 60s
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 1G
        reservations:
          cpus: '1'
          memory: 512M
```

**Usage:**
```bash
# Start services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down

# Rebuild and restart
docker compose up -d --build
```

### 4. Verify Container

```bash
# Check container status
docker ps | grep sentiment-api

# View logs
docker logs -f sentiment-api

# Check health
docker exec sentiment-api curl -f http://localhost:8080/api/v1/health

# Monitor resource usage
docker stats sentiment-api
```

---

## Production Deployment

### Cloud Platform Deployment

#### AWS Elastic Container Service (ECS)

**1. Push Image to ECR:**
```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag sentiment-analyzer:latest \
  <account-id>.dkr.ecr.us-east-1.amazonaws.com/sentiment-analyzer:latest

# Push image
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/sentiment-analyzer:latest
```

**2. Create ECS Task Definition:**
```json
{
  "family": "sentiment-analyzer",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "sentiment-api",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/sentiment-analyzer:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "SPRING_PROFILES_ACTIVE",
          "value": "production"
        },
        {
          "name": "SENTIMENT_MODEL_TYPE",
          "value": "svm"
        }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8080/api/v1/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/sentiment-analyzer",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**3. Create ECS Service:**
```bash
aws ecs create-service \
  --cluster sentiment-cluster \
  --service-name sentiment-api \
  --task-definition sentiment-analyzer \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=sentiment-api,containerPort=8080"
```

---

#### Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/<project-id>/sentiment-analyzer

# Deploy to Cloud Run
gcloud run deploy sentiment-analyzer \
  --image gcr.io/<project-id>/sentiment-analyzer \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 2 \
  --port 8080 \
  --set-env-vars SPRING_PROFILES_ACTIVE=production,SENTIMENT_MODEL_TYPE=svm
```

---

#### Azure Container Instances

```bash
# Create resource group
az group create --name sentiment-rg --location eastus

# Deploy container
az container create \
  --resource-group sentiment-rg \
  --name sentiment-analyzer \
  --image <your-registry>/sentiment-analyzer:latest \
  --cpu 2 \
  --memory 2 \
  --ports 8080 \
  --environment-variables SPRING_PROFILES_ACTIVE=production \
  --restart-policy OnFailure
```

---

### Kubernetes Deployment

**Deployment Manifest** (`k8s/deployment.yaml`):
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-analyzer
  labels:
    app: sentiment-analyzer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sentiment-analyzer
  template:
    metadata:
      labels:
        app: sentiment-analyzer
    spec:
      containers:
      - name: sentiment-api
        image: sentiment-analyzer:1.0.0
        ports:
        - containerPort: 8080
        env:
        - name: SPRING_PROFILES_ACTIVE
          value: "production"
        - name: SENTIMENT_MODEL_TYPE
          value: "svm"
        - name: JAVA_OPTS
          value: "-Xmx768m -Xms512m"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        volumeMounts:
        - name: models
          mountPath: /app/models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: sentiment-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: sentiment-analyzer
spec:
  type: LoadBalancer
  selector:
    app: sentiment-analyzer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
```

**Deploy to Kubernetes:**
```bash
# Apply deployment
kubectl apply -f k8s/deployment.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/sentiment-analyzer

# Scale deployment
kubectl scale deployment sentiment-analyzer --replicas=5
```

---

## Configuration

### Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `SPRING_PROFILES_ACTIVE` | Application profile | `dev` | `production` |
| `SENTIMENT_MODEL_TYPE` | Algorithm to use | `svm` | `naive_bayes`, `random_forest` |
| `SENTIMENT_CONFIDENCE_THRESHOLD` | Minimum confidence | `0.7` | `0.8` |
| `SENTIMENT_SVM_MODEL` | SVM model path | `/app/models/svm-model.ser` | `/custom/path/model.ser` |
| `SENTIMENT_PREFER_PRETRAINED` | Use pre-trained models | `true` | `false` |
| `SENTIMENT_REQUIRE_PRETRAINED` | Fail if no pretrained | `false` | `true` |
| `SENTIMENT_API_MAX_BATCH_SIZE` | Max batch size | `100` | `200` |
| `SENTIMENT_PREPROCESSING_MAX_FEATURES` | Max TF-IDF features | `5000` | `10000` |
| `SERVER_PORT` | API port | `8080` | `9090` |
| `JAVA_OPTS` | JVM options | `-Xmx512m` | `-Xmx2g -XX:+UseG1GC` |

---

### Application Profiles

**Development Profile** (`application-dev.yml`):
```yaml
sentiment:
  api:
    rate-limit: 1000
resilience4j:
  ratelimiter:
    instances:
      sentimentApi:
        limit-for-period: 10
        limit-refresh-period: 1m
logging:
  level:
    sentiment: DEBUG
```

**Production Profile** (`application-production.yml`):
```yaml
sentiment:
  api:
    rate-limit: 60
resilience4j:
  ratelimiter:
    instances:
      sentimentApi:
        limit-for-period: 60
        limit-refresh-period: 1m
logging:
  level:
    sentiment: INFO
```

---

### JVM Tuning

**Recommended JVM Options:**

```bash
# For 512MB container
JAVA_OPTS="-Xmx384m -Xms256m -XX:+UseContainerSupport -XX:MaxRAMPercentage=75.0"

# For 1GB container
JAVA_OPTS="-Xmx768m -Xms512m -XX:+UseContainerSupport -XX:MaxRAMPercentage=75.0"

# For 2GB container
JAVA_OPTS="-Xmx1536m -Xms1024m -XX:+UseContainerSupport -XX:MaxRAMPercentage=75.0"

# Enable G1GC (recommended for low-latency)
JAVA_OPTS="$JAVA_OPTS -XX:+UseG1GC -XX:MaxGCPauseMillis=200"

# Enable JMX monitoring
JAVA_OPTS="$JAVA_OPTS -Dcom.sun.management.jmxremote -Dcom.sun.management.jmxremote.port=9010 -Dcom.sun.management.jmxremote.authenticate=false"
```

---

## Monitoring & Observability

### Spring Boot Actuator

**Enable Actuator Endpoints:**
```yaml
management:
  endpoints:
    web:
      exposure:
        include: health,info,metrics,prometheus
  endpoint:
    health:
      show-details: when-authorized
```

**Endpoints:**
- `/actuator/health` - Health check
- `/actuator/info` - Application info
- `/actuator/metrics` - JVM metrics
- `/actuator/prometheus` - Prometheus metrics

---

### Prometheus + Grafana

**Prometheus Scrape Config** (`prometheus.yml`):
```yaml
scrape_configs:
  - job_name: 'sentiment-analyzer'
    metrics_path: '/actuator/prometheus'
    static_configs:
      - targets: ['sentiment-api:8080']
```

**Key Metrics to Monitor:**
- `http_server_requests_seconds_count` - Request count
- `http_server_requests_seconds_sum` - Total latency
- `jvm_memory_used_bytes` - Memory usage
- `jvm_gc_pause_seconds` - GC pause time
- `resilience4j_ratelimiter_available_permissions` - Rate limiter state

---

### Logging

**Log Aggregation (ELK Stack):**
```bash
# Configure Logback to output JSON
# Edit src/main/resources/logback-spring.xml
<encoder class="net.logstash.logback.encoder.LogstashEncoder"/>
```

**Centralized Logging (Docker):**
```yaml
services:
  sentiment-api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## Troubleshooting

### Issue: Model Not Loading

**Symptoms:**
- Health check returns `"modelLoaded": false`
- API returns 503 Service Unavailable

**Diagnosis:**
```bash
# Check model file exists
docker exec sentiment-api ls -lh /app/models/

# Check application logs
docker logs sentiment-api | grep -i "model"
```

**Solutions:**
1. **Missing model files**: Ensure models are in container
   ```bash
   docker run -v $(pwd)/models:/app/models sentiment-analyzer
   ```

2. **Wrong model path**: Check environment variable
   ```bash
   docker run -e SENTIMENT_SVM_MODEL=/app/models/svm-model.ser sentiment-analyzer
   ```

3. **Corrupted model**: Retrain model
   ```bash
   java -jar sentiment-analyzer.jar sentiment.training.ModelTrainingCLI \
     --data-path /path/to/Reviews.csv \
     --output-dir ./models
   ```

---

### Issue: High Memory Usage

**Symptoms:**
- Container OOM killed
- Slow response times
- Frequent GC pauses

**Diagnosis:**
```bash
# Check JVM memory settings
docker exec sentiment-api java -XX:+PrintFlagsFinal -version | grep -i heap

# Monitor memory usage
docker stats sentiment-api
```

**Solutions:**
1. **Increase container memory**:
   ```bash
   docker run -m 1g sentiment-analyzer
   ```

2. **Tune JVM heap**:
   ```bash
   docker run -e JAVA_OPTS="-Xmx768m -Xms512m" sentiment-analyzer
   ```

3. **Enable G1GC**:
   ```bash
   docker run -e JAVA_OPTS="-XX:+UseG1GC -XX:MaxGCPauseMillis=200" sentiment-analyzer
   ```

---

### Issue: Slow Inference

**Symptoms:**
- `processingTimeMs > 500ms`
- Timeouts on batch requests

**Diagnosis:**
```bash
# Check CPU usage
docker stats sentiment-api

# Check thread count
docker exec sentiment-api jstack 1 | grep "nid" | wc -l
```

**Solutions:**
1. **Increase CPU allocation**:
   ```bash
   docker run --cpus 4 sentiment-analyzer
   ```

2. **Use faster algorithm**:
   ```bash
   docker run -e SENTIMENT_MODEL_TYPE=naive_bayes sentiment-analyzer
   ```

3. **Reduce feature count**:
   ```bash
   docker run -e SENTIMENT_PREPROCESSING_MAX_FEATURES=3000 sentiment-analyzer
   ```

---

### Issue: Rate Limit Too Restrictive

**Symptoms:**
- 429 Too Many Requests errors
- Legitimate traffic blocked

**Solutions:**
1. **Increase rate limit**:
   ```yaml
   resilience4j:
     ratelimiter:
       instances:
         sentimentApi:
           limit-for-period: 100  # Increase from 60
   ```

2. **Use production profile**:
   ```bash
   docker run -e SPRING_PROFILES_ACTIVE=production sentiment-analyzer
   ```

3. **Disable rate limiting** (dev only):
   ```yaml
   resilience4j:
     ratelimiter:
       instances:
         sentimentApi:
           limit-for-period: 1000000
   ```

---

### Issue: Container Fails Health Check

**Symptoms:**
- Container restarts repeatedly
- Health endpoint returns 503

**Diagnosis:**
```bash
# Check startup logs
docker logs sentiment-api

# Manually test health endpoint
docker exec sentiment-api curl -f http://localhost:8080/api/v1/health
```

**Solutions:**
1. **Increase startup time**:
   ```yaml
   healthcheck:
     start_period: 120s  # Give more time for model loading
   ```

2. **Check model loading**:
   ```bash
   docker logs sentiment-api | grep -i "model loaded"
   ```

3. **Verify port binding**:
   ```bash
   docker exec sentiment-api netstat -tuln | grep 8080
   ```

---

## Security Best Practices

### 1. Non-Root User

Dockerfile already configures non-root user:
```dockerfile
RUN groupadd -g 1001 appuser && \
    useradd -r -u 1001 -g appuser appuser
USER appuser
```

### 2. Secrets Management

**DO NOT** hardcode secrets. Use environment variables or secrets managers:

```bash
# AWS Secrets Manager
aws secretsmanager get-secret-value --secret-id sentiment-api-key \
  --query SecretString --output text | \
  jq -r .SENTIMENT_API_KEY | \
  docker run -e SENTIMENT_API_KEY=$(cat -) sentiment-analyzer
```

### 3. Network Security

**Restrict Container Network:**
```bash
docker network create --driver bridge sentiment-net
docker run --network sentiment-net sentiment-analyzer
```

### 4. Image Scanning

```bash
# Scan for vulnerabilities
docker scan sentiment-analyzer:latest

# Use distroless base image (future enhancement)
FROM gcr.io/distroless/java21-debian12
```

---

## Performance Benchmarks

### Single Instance Performance

| Metric | Value |
|--------|-------|
| **Startup Time** | < 5 seconds (pre-trained model) |
| **Single Request Latency** | 30-50ms |
| **Batch (100 texts) Latency** | 1-2 seconds |
| **Throughput** | ~1,000 requests/minute |
| **Memory Footprint** | 512MB (steady state) |
| **CPU Usage** | 10-20% (idle), 80-100% (load) |

### Horizontal Scaling

| Replicas | Throughput | Latency (p99) |
|----------|-----------|---------------|
| 1 | 1,000 req/min | 100ms |
| 3 | 3,000 req/min | 120ms |
| 5 | 5,000 req/min | 150ms |
| 10 | 9,500 req/min | 200ms |

---

## Backup & Disaster Recovery

### Model Backup

```bash
# Backup models
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/

# Upload to S3
aws s3 cp models-backup-$(date +%Y%m%d).tar.gz s3://sentiment-backups/
```

### Model Restore

```bash
# Download from S3
aws s3 cp s3://sentiment-backups/models-backup-20251112.tar.gz .

# Extract
tar -xzf models-backup-20251112.tar.gz

# Restart container with restored models
docker run -v $(pwd)/models:/app/models sentiment-analyzer
```

---

**Last Updated**: 2025-11-12
**Author**: Victoria Alabi
