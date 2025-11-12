# API Reference

Complete documentation for the Java Sentiment Analyzer REST API.

---

## Table of Contents

1. [Base URL](#base-url)
2. [Authentication](#authentication)
3. [Rate Limiting](#rate-limiting)
4. [Endpoints](#endpoints)
5. [Error Handling](#error-handling)
6. [Request/Response Examples](#requestresponse-examples)

---

## Base URL

```
http://localhost:8080/api/v1
```

When deployed to production, replace `localhost:8080` with your actual domain.

---

## Authentication

**Current Version**: No authentication required (suitable for internal/demo deployments).

**Future Versions**: Will support API key authentication via `X-API-Key` header.

---

## Rate Limiting

The API uses **Resilience4j** for rate limiting to prevent abuse.

### Rate Limits (by Profile)

| Profile | Endpoint | Limit | Window |
|---------|----------|-------|--------|
| **Development** | `/sentiment/analyze` | 10 requests | 1 minute |
| **Development** | `/sentiment/batch` | 5 requests | 1 minute |
| **Production** | `/sentiment/analyze` | 60 requests | 1 minute |
| **Production** | `/sentiment/batch` | 10 requests | 1 minute |

### Rate Limit Headers

Responses include rate limit information:

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 42
X-RateLimit-Reset: 2025-11-12T10:35:00Z
```

### Rate Limit Exceeded Response

**Status Code**: `429 Too Many Requests`

```json
{
  "timestamp": "2025-11-12T10:30:45Z",
  "status": 429,
  "error": "Too Many Requests",
  "message": "Rate limit exceeded. Please try again later.",
  "path": "/api/v1/sentiment/analyze"
}
```

---

## Endpoints

### 1. Single Text Analysis

Analyze sentiment for a single text input.

**Endpoint**: `POST /api/v1/sentiment/analyze`

**Request Headers**:
```http
Content-Type: application/json
```

**Request Body**:
```json
{
  "text": "string (required, 1-10000 characters)",
  "confidenceThreshold": 0.7 (optional, default: 0.7, range: 0.0-1.0)
}
```

**Response** (Success - 200 OK):
```json
{
  "sentiment": "positive | negative | uncertain",
  "confidence": 0.92,
  "text": "The original input text",
  "processingTimeMs": 45
}
```

**Field Descriptions**:
- `sentiment`: Predicted sentiment class
  - `"positive"`: Confidence >= threshold
  - `"negative"`: Confidence >= threshold
  - `"uncertain"`: Confidence < threshold for both classes
- `confidence`: Probability of predicted class (0.0-1.0)
- `text`: Original input text (echoed back)
- `processingTimeMs`: Inference time in milliseconds

---

### 2. Batch Text Analysis

Analyze sentiment for multiple texts in parallel.

**Endpoint**: `POST /api/v1/sentiment/batch`

**Request Headers**:
```http
Content-Type: application/json
```

**Request Body**:
```json
{
  "texts": [
    "string (required, 1-10000 characters per text)",
    "string",
    ...
  ],
  "confidenceThreshold": 0.7 (optional, default: 0.7, range: 0.0-1.0)
}
```

**Constraints**:
- Maximum batch size: 100 texts (configurable via `sentiment.api.max-batch-size`)
- Texts processed in parallel
- Results returned in **same order** as input

**Response** (Success - 200 OK):
```json
{
  "results": [
    {
      "sentiment": "positive",
      "confidence": 0.94,
      "text": "Amazing product, highly recommend!",
      "processingTimeMs": 32
    },
    {
      "sentiment": "negative",
      "confidence": 0.89,
      "text": "Terrible experience, complete waste of money.",
      "processingTimeMs": 28
    },
    {
      "sentiment": "uncertain",
      "confidence": 0.62,
      "text": "It works okay, nothing special.",
      "processingTimeMs": 35
    }
  ],
  "totalProcessingTimeMs": 95
}
```

**Field Descriptions**:
- `results`: Array of sentiment results (matches input order)
- `totalProcessingTimeMs`: Total wall-clock time for batch processing

**Performance Notes**:
- Batch processing uses `ForkJoinPool` for parallelism
- Typical throughput: ~100 texts in 1-2 seconds
- Individual `processingTimeMs` may overlap (parallel execution)

---

### 3. Health Check

Check API and model health status.

**Endpoint**: `GET /api/v1/health`

**Request Headers**: None required

**Response** (Success - 200 OK):
```json
{
  "status": "UP | DOWN",
  "modelLoaded": true,
  "algorithmType": "SVM",
  "timestamp": "2025-11-12T10:30:45Z",
  "version": "1.0.0"
}
```

**Field Descriptions**:
- `status`: Overall health status
  - `"UP"`: API operational, model loaded
  - `"DOWN"`: API operational, but model not loaded or failed
- `modelLoaded`: Whether classifier is ready for inference
- `algorithmType`: Active classification algorithm
- `timestamp`: Current server time (ISO 8601)
- `version`: API version

**Response** (Service Unavailable - 503):
```json
{
  "status": "DOWN",
  "modelLoaded": false,
  "algorithmType": null,
  "timestamp": "2025-11-12T10:30:45Z",
  "version": "1.0.0",
  "error": "Model failed to load: file not found"
}
```

---

## Error Handling

### Error Response Format

All errors follow a consistent structure:

```json
{
  "timestamp": "2025-11-12T10:30:45Z",
  "status": 400,
  "error": "Bad Request",
  "message": "Detailed error message",
  "path": "/api/v1/sentiment/analyze",
  "validationErrors": [
    {
      "field": "text",
      "message": "Text cannot be null"
    }
  ]
}
```

---

### HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| **200** | OK | Request succeeded |
| **400** | Bad Request | Validation failure, malformed JSON |
| **429** | Too Many Requests | Rate limit exceeded |
| **500** | Internal Server Error | Model inference failure, unexpected error |
| **503** | Service Unavailable | Model not loaded, service starting up |

---

### Common Error Scenarios

#### 1. Missing Required Field

**Request**:
```json
{
  "confidenceThreshold": 0.8
}
```

**Response** (400 Bad Request):
```json
{
  "timestamp": "2025-11-12T10:30:45Z",
  "status": 400,
  "error": "Bad Request",
  "message": "Validation failed",
  "path": "/api/v1/sentiment/analyze",
  "validationErrors": [
    {
      "field": "text",
      "message": "Text cannot be null"
    }
  ]
}
```

---

#### 2. Text Too Long

**Request**:
```json
{
  "text": "A string with 10,001 characters..."
}
```

**Response** (400 Bad Request):
```json
{
  "timestamp": "2025-11-12T10:30:45Z",
  "status": 400,
  "error": "Bad Request",
  "message": "Validation failed",
  "path": "/api/v1/sentiment/analyze",
  "validationErrors": [
    {
      "field": "text",
      "message": "Text must be 1-10000 characters"
    }
  ]
}
```

---

#### 3. Invalid Confidence Threshold

**Request**:
```json
{
  "text": "Great product!",
  "confidenceThreshold": 1.5
}
```

**Response** (400 Bad Request):
```json
{
  "timestamp": "2025-11-12T10:30:45Z",
  "status": 400,
  "error": "Bad Request",
  "message": "Validation failed",
  "path": "/api/v1/sentiment/analyze",
  "validationErrors": [
    {
      "field": "confidenceThreshold",
      "message": "Confidence threshold must be 0-1"
    }
  ]
}
```

---

#### 4. Batch Size Exceeded

**Request**:
```json
{
  "texts": ["text1", "text2", ..., "text101"]  // 101 texts
}
```

**Response** (400 Bad Request):
```json
{
  "timestamp": "2025-11-12T10:30:45Z",
  "status": 400,
  "error": "Bad Request",
  "message": "Batch size exceeds maximum allowed (100)",
  "path": "/api/v1/sentiment/batch"
}
```

---

#### 5. Empty Batch Request

**Request**:
```json
{
  "texts": []
}
```

**Response** (400 Bad Request):
```json
{
  "timestamp": "2025-11-12T10:30:45Z",
  "status": 400,
  "error": "Bad Request",
  "message": "Texts array cannot be empty",
  "path": "/api/v1/sentiment/batch"
}
```

---

#### 6. Model Not Loaded

**Request**: Any inference request when model fails to load

**Response** (503 Service Unavailable):
```json
{
  "timestamp": "2025-11-12T10:30:45Z",
  "status": 503,
  "error": "Service Unavailable",
  "message": "Sentiment classifier not loaded. Check model files.",
  "path": "/api/v1/sentiment/analyze"
}
```

---

## Request/Response Examples

### Example 1: Positive Review

**Request**:
```bash
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This product exceeded all my expectations! Fast shipping, great quality, and excellent customer service. Highly recommended!",
    "confidenceThreshold": 0.75
  }'
```

**Response**:
```json
{
  "sentiment": "positive",
  "confidence": 0.94,
  "text": "This product exceeded all my expectations! Fast shipping, great quality, and excellent customer service. Highly recommended!",
  "processingTimeMs": 42
}
```

---

### Example 2: Negative Review

**Request**:
```bash
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Absolutely terrible. Product broke after two days, customer service was unhelpful, and refund process took weeks. Do not buy!",
    "confidenceThreshold": 0.7
  }'
```

**Response**:
```json
{
  "sentiment": "negative",
  "confidence": 0.91,
  "text": "Absolutely terrible. Product broke after two days, customer service was unhelpful, and refund process took weeks. Do not buy!",
  "processingTimeMs": 38
}
```

---

### Example 3: Uncertain Sentiment

**Request**:
```bash
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "It arrived on time.",
    "confidenceThreshold": 0.8
  }'
```

**Response**:
```json
{
  "sentiment": "uncertain",
  "confidence": 0.62,
  "text": "It arrived on time.",
  "processingTimeMs": 35
}
```

**Note**: Short, neutral texts often result in `"uncertain"` sentiment due to lack of sentiment-bearing words.

---

### Example 4: Batch Processing

**Request**:
```bash
curl -X POST http://localhost:8080/api/v1/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Amazing product, will buy again!",
      "Terrible quality, very disappointed.",
      "It works as described.",
      "Best purchase this year!",
      "Complete waste of money."
    ],
    "confidenceThreshold": 0.75
  }'
```

**Response**:
```json
{
  "results": [
    {
      "sentiment": "positive",
      "confidence": 0.93,
      "text": "Amazing product, will buy again!",
      "processingTimeMs": 28
    },
    {
      "sentiment": "negative",
      "confidence": 0.89,
      "text": "Terrible quality, very disappointed.",
      "processingTimeMs": 32
    },
    {
      "sentiment": "uncertain",
      "confidence": 0.68,
      "text": "It works as described.",
      "processingTimeMs": 30
    },
    {
      "sentiment": "positive",
      "confidence": 0.96,
      "text": "Best purchase this year!",
      "processingTimeMs": 26
    },
    {
      "sentiment": "negative",
      "confidence": 0.92,
      "text": "Complete waste of money.",
      "processingTimeMs": 29
    }
  ],
  "totalProcessingTimeMs": 145
}
```

---

### Example 5: Health Check

**Request**:
```bash
curl http://localhost:8080/api/v1/health
```

**Response** (Healthy):
```json
{
  "status": "UP",
  "modelLoaded": true,
  "algorithmType": "SVM",
  "timestamp": "2025-11-12T10:30:45Z",
  "version": "1.0.0"
}
```

---

### Example 6: Using Different Confidence Thresholds

**High Threshold (Conservative)**:
```bash
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Pretty good product overall.",
    "confidenceThreshold": 0.9
  }'
```

**Response**:
```json
{
  "sentiment": "uncertain",
  "confidence": 0.82,
  "text": "Pretty good product overall.",
  "processingTimeMs": 37
}
```
*(Confidence 0.82 < 0.9, so classified as "uncertain")*

---

**Low Threshold (Permissive)**:
```bash
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Pretty good product overall.",
    "confidenceThreshold": 0.5
  }'
```

**Response**:
```json
{
  "sentiment": "positive",
  "confidence": 0.82,
  "text": "Pretty good product overall.",
  "processingTimeMs": 37
}
```
*(Confidence 0.82 >= 0.5, so classified as "positive")*

---

## Configuration

### Environment Variables

Override default API behavior via environment variables:

```bash
# Model selection
SENTIMENT_MODEL_TYPE=svm  # svm, naive_bayes, random_forest, logistic_regression

# Confidence threshold
SENTIMENT_CONFIDENCE_THRESHOLD=0.7

# API limits
SENTIMENT_API_MAX_BATCH_SIZE=100
SENTIMENT_API_RATE_LIMIT=1000

# Model paths
SENTIMENT_SVM_MODEL=/app/models/svm-model.ser
SENTIMENT_PREFER_PRETRAINED=true
```

---

### Application Profiles

**Development Profile** (relaxed limits):
```bash
java -jar sentiment-analyzer.jar --spring.profiles.active=dev
```

**Production Profile** (strict limits):
```bash
java -jar sentiment-analyzer.jar --spring.profiles.active=production
```

---

## Client Libraries

### Java (Spring RestTemplate)

```java
RestTemplate restTemplate = new RestTemplate();

SentimentRequest request = new SentimentRequest();
request.setText("Great product!");
request.setConfidenceThreshold(0.7);

SentimentResponse response = restTemplate.postForObject(
    "http://localhost:8080/api/v1/sentiment/analyze",
    request,
    SentimentResponse.class
);

System.out.println("Sentiment: " + response.getSentiment());
System.out.println("Confidence: " + response.getConfidence());
```

---

### Python (requests)

```python
import requests

response = requests.post(
    'http://localhost:8080/api/v1/sentiment/analyze',
    json={
        'text': 'Great product!',
        'confidenceThreshold': 0.7
    }
)

result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")
```

---

### JavaScript (fetch)

```javascript
fetch('http://localhost:8080/api/v1/sentiment/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    text: 'Great product!',
    confidenceThreshold: 0.7
  })
})
  .then(response => response.json())
  .then(data => {
    console.log('Sentiment:', data.sentiment);
    console.log('Confidence:', data.confidence);
  });
```

---

### cURL (shell script)

```bash
#!/bin/bash

TEXT="Amazing product, highly recommend!"
THRESHOLD=0.75

curl -s -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$TEXT\",\"confidenceThreshold\":$THRESHOLD}" \
  | jq '.sentiment, .confidence'
```

---

## Monitoring & Observability

### Spring Boot Actuator Endpoints

```bash
# Health check (basic)
curl http://localhost:8080/actuator/health

# Detailed health info
curl http://localhost:8080/actuator/health/liveness
curl http://localhost:8080/actuator/health/readiness

# Application info
curl http://localhost:8080/actuator/info

# Metrics (Prometheus format)
curl http://localhost:8080/actuator/prometheus
```

---

### Key Metrics

**Request Metrics**:
- `http_server_requests_seconds_count`: Total requests
- `http_server_requests_seconds_sum`: Total processing time
- `http_server_requests_seconds_max`: Max request time

**Classifier Metrics**:
- `sentiment_classification_duration_seconds`: Inference time histogram
- `sentiment_classification_total`: Total classifications by sentiment
- `sentiment_confidence_score`: Confidence score distribution

**Rate Limiter Metrics**:
- `resilience4j_ratelimiter_available_permissions`: Available rate limit tokens
- `resilience4j_ratelimiter_waiting_threads`: Threads waiting for rate limit

---

## Testing

### Integration Test Example

```bash
# Start the API
docker run -p 8080:8080 sentiment-analyzer

# Wait for startup
sleep 10

# Test single analysis
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Test product review"}' \
  -w "\nHTTP Status: %{http_code}\n"

# Test batch analysis
curl -X POST http://localhost:8080/api/v1/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["Good","Bad","Okay"]}' \
  -w "\nHTTP Status: %{http_code}\n"

# Test health check
curl http://localhost:8080/api/v1/health \
  -w "\nHTTP Status: %{http_code}\n"
```

---

## FAQ

### Q: What happens if I send empty text?

**A**: Validation error (400 Bad Request):
```json
{
  "message": "Text must be 1-10000 characters"
}
```

---

### Q: Can I analyze text in languages other than English?

**A**: The current model is trained on English reviews only. Non-English text will likely produce unreliable results.

---

### Q: How do I interpret the confidence score?

**A**:
- `confidence >= 0.9`: Very confident
- `confidence 0.7-0.9`: Confident (default threshold)
- `confidence 0.5-0.7`: Uncertain (consider raising threshold)
- `confidence < 0.5`: Very uncertain

---

### Q: What's the maximum request size?

**A**:
- Single request: 10,000 characters per text
- Batch request: 100 texts × 10,000 characters = ~1MB total

---

### Q: How is "uncertain" sentiment determined?

**A**: If the confidence for both "positive" and "negative" classes falls below the `confidenceThreshold`, the result is "uncertain".

---

### Q: Can I change the algorithm at runtime?

**A**: Not without restart. Set `SENTIMENT_MODEL_TYPE` environment variable and restart the service.

---

## Changelog

**v1.0.0** (2025-11-12)
- Initial API release
- Single and batch sentiment analysis
- Health check endpoint
- Rate limiting
- Comprehensive error handling

---

**Last Updated**: 2025-11-12
**API Version**: 1.0.0
**Author**: Victoria Alabi
