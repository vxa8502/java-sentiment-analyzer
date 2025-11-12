# API Usage Examples

This directory contains example code for interacting with the Java Sentiment Analyzer API in various programming languages.

---

## Prerequisites

Ensure the API is running before executing these examples:

```bash
# Option 1: Docker
docker run -p 8080:8080 sentiment-analyzer

# Option 2: Maven
mvn spring-boot:run

# Option 3: JAR
java -jar target/sentiment-analyzer-1.0.0.jar
```

Verify the API is accessible:
```bash
curl http://localhost:8080/api/v1/health
```

---

## Bash / cURL Examples

### Running the Script

```bash
# Make executable (if not already)
chmod +x curl_examples.sh

# Run all examples
./curl_examples.sh
```

The script demonstrates:
- Health check
- Positive sentiment analysis
- Negative sentiment analysis
- Neutral/uncertain sentiment
- Custom confidence thresholds
- Batch analysis
- Long text reviews
- Mixed sentiment
- Edge cases
- Validation errors

### Manual cURL Commands

**Single Text Analysis:**
```bash
curl -X POST http://localhost:8080/api/v1/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Great product!"}'
```

**Batch Analysis:**
```bash
curl -X POST http://localhost:8080/api/v1/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["Good","Bad","Okay"]}'
```

**Health Check:**
```bash
curl http://localhost:8080/api/v1/health
```

---

## Python Client

### Installation

```bash
# Install required library
pip install requests

# Or using a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install requests
```

### Running the Script

```bash
# Make executable (if not already)
chmod +x python_client.py

# Run examples
./python_client.py

# Or directly with Python
python python_client.py
```

### Using as a Library

```python
from python_client import SentimentAnalyzerClient

# Initialize client
client = SentimentAnalyzerClient(base_url="http://localhost:8080/api/v1")

# Analyze single text
result = client.analyze("This product is amazing!")
print(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']:.2%}")

# Batch analysis
results = client.analyze_batch([
    "Great product!",
    "Terrible quality.",
    "It works okay."
])
for r in results['results']:
    print(f"{r['text']} -> {r['sentiment']}")

# Health check
health = client.health_check()
print(f"API Status: {health['status']}")
```

---

## JavaScript / Node.js Example

### Using `fetch` (Node.js 18+)

```javascript
// analyze.js
async function analyzeSentiment(text) {
  const response = await fetch('http://localhost:8080/api/v1/sentiment/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, confidenceThreshold: 0.7 })
  });

  const result = await response.json();
  return result;
}

// Usage
(async () => {
  const result = await analyzeSentiment("Great product!");
  console.log(`Sentiment: ${result.sentiment}`);
  console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
})();
```

**Run:**
```bash
node analyze.js
```

### Using `axios`

```javascript
const axios = require('axios');

const API_URL = 'http://localhost:8080/api/v1';

async function analyzeSentiment(text) {
  const response = await axios.post(`${API_URL}/sentiment/analyze`, {
    text: text,
    confidenceThreshold: 0.7
  });
  return response.data;
}

async function analyzeBatch(texts) {
  const response = await axios.post(`${API_URL}/sentiment/batch`, {
    texts: texts
  });
  return response.data;
}

// Usage
(async () => {
  // Single analysis
  const result = await analyzeSentiment("Amazing product!");
  console.log(result);

  // Batch analysis
  const batchResult = await analyzeBatch([
    "Great!",
    "Terrible!",
    "Okay."
  ]);
  console.log(batchResult.results);
})();
```

**Install and run:**
```bash
npm install axios
node analyze.js
```

---

## Java Client Example

### Using Spring `RestTemplate`

```java
import org.springframework.web.client.RestTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;

public class SentimentClient {
    private final RestTemplate restTemplate = new RestTemplate();
    private final String baseUrl = "http://localhost:8080/api/v1";

    public SentimentResponse analyze(String text) {
        String url = baseUrl + "/sentiment/analyze";

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        SentimentRequest request = new SentimentRequest(text, 0.7);
        HttpEntity<SentimentRequest> entity = new HttpEntity<>(request, headers);

        return restTemplate.postForObject(url, entity, SentimentResponse.class);
    }

    public static void main(String[] args) {
        SentimentClient client = new SentimentClient();
        SentimentResponse response = client.analyze("Great product!");

        System.out.println("Sentiment: " + response.getSentiment());
        System.out.println("Confidence: " + response.getConfidence());
    }
}

// DTOs
class SentimentRequest {
    private String text;
    private double confidenceThreshold;

    public SentimentRequest(String text, double confidenceThreshold) {
        this.text = text;
        this.confidenceThreshold = confidenceThreshold;
    }

    // Getters and setters
}

class SentimentResponse {
    private String sentiment;
    private double confidence;
    private String text;
    private long processingTimeMs;

    // Getters and setters
}
```

### Using Apache HttpClient

```java
import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import com.fasterxml.jackson.databind.ObjectMapper;

public class SentimentClient {
    private final CloseableHttpClient httpClient = HttpClients.createDefault();
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final String baseUrl = "http://localhost:8080/api/v1";

    public SentimentResponse analyze(String text) throws Exception {
        HttpPost post = new HttpPost(baseUrl + "/sentiment/analyze");
        post.setHeader("Content-Type", "application/json");

        String json = String.format("{\"text\":\"%s\",\"confidenceThreshold\":0.7}", text);
        post.setEntity(new StringEntity(json));

        try (CloseableHttpResponse response = httpClient.execute(post)) {
            String responseBody = EntityUtils.toString(response.getEntity());
            return objectMapper.readValue(responseBody, SentimentResponse.class);
        }
    }
}
```

---

## Go Example

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
)

type SentimentRequest struct {
    Text               string  `json:"text"`
    ConfidenceThreshold float64 `json:"confidenceThreshold"`
}

type SentimentResponse struct {
    Sentiment       string  `json:"sentiment"`
    Confidence      float64 `json:"confidence"`
    Text            string  `json:"text"`
    ProcessingTimeMs int64   `json:"processingTimeMs"`
}

func analyzeSentiment(text string) (*SentimentResponse, error) {
    url := "http://localhost:8080/api/v1/sentiment/analyze"

    request := SentimentRequest{
        Text:               text,
        ConfidenceThreshold: 0.7,
    }

    jsonData, err := json.Marshal(request)
    if err != nil {
        return nil, err
    }

    resp, err := http.Post(url, "application/json", bytes.NewBuffer(jsonData))
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result SentimentResponse
    if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
        return nil, err
    }

    return &result, nil
}

func main() {
    result, err := analyzeSentiment("Great product!")
    if err != nil {
        panic(err)
    }

    fmt.Printf("Sentiment: %s\n", result.Sentiment)
    fmt.Printf("Confidence: %.2f%%\n", result.Confidence*100)
}
```

**Run:**
```bash
go run sentiment_client.go
```

---

## Ruby Example

```ruby
require 'net/http'
require 'json'
require 'uri'

class SentimentClient
  def initialize(base_url = 'http://localhost:8080/api/v1')
    @base_url = base_url
  end

  def analyze(text, confidence_threshold = 0.7)
    uri = URI("#{@base_url}/sentiment/analyze")
    request = Net::HTTP::Post.new(uri)
    request.content_type = 'application/json'
    request.body = JSON.dump({
      text: text,
      confidenceThreshold: confidence_threshold
    })

    response = Net::HTTP.start(uri.hostname, uri.port) do |http|
      http.request(request)
    end

    JSON.parse(response.body)
  end

  def analyze_batch(texts, confidence_threshold = 0.7)
    uri = URI("#{@base_url}/sentiment/batch")
    request = Net::HTTP::Post.new(uri)
    request.content_type = 'application/json'
    request.body = JSON.dump({
      texts: texts,
      confidenceThreshold: confidence_threshold
    })

    response = Net::HTTP.start(uri.hostname, uri.port) do |http|
      http.request(request)
    end

    JSON.parse(response.body)
  end
end

# Usage
client = SentimentClient.new
result = client.analyze("Great product!")
puts "Sentiment: #{result['sentiment']}"
puts "Confidence: #{(result['confidence'] * 100).round(1)}%"
```

---

## Environment Variables

All examples support customizing the API URL via environment variable:

```bash
export SENTIMENT_API_URL=http://your-api-domain.com/api/v1

# Run examples
./curl_examples.sh
python python_client.py
```

---

## Common Issues

### Connection Refused

**Error:** `curl: (7) Failed to connect to localhost port 8080: Connection refused`

**Solution:** Ensure the API is running:
```bash
docker ps | grep sentiment
# Or
curl http://localhost:8080/api/v1/health
```

### Invalid JSON

**Error:** `400 Bad Request` with validation errors

**Solution:** Ensure your JSON is properly formatted:
```bash
# ✅ Correct
curl -d '{"text":"Hello"}'

# ❌ Incorrect (missing quotes)
curl -d '{text:Hello}'
```

### Rate Limit Exceeded

**Error:** `429 Too Many Requests`

**Solution:** Wait for rate limit window to reset (1 minute) or increase limits in configuration.

---

## Next Steps

- Explore the [API Reference](../docs/API_REFERENCE.md) for complete endpoint documentation
- Check the [Architecture](../docs/ARCHITECTURE.md) to understand system design
- Review [Deployment Guide](../docs/DEPLOYMENT.md) for production setup

---

**Last Updated**: 2025-11-12
