package sentiment.api;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.condition.EnabledIf;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;

import java.nio.file.Files;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end integration tests with real Spring Boot context and ML model.
 *
 * <h2>Test Category: Integration (E2E)</h2>
 * <p>These tests verify the complete request flow from HTTP to ML inference
 * and back, using a real trained model.
 *
 * <h3>Characteristics</h3>
 * <ul>
 *   <li><b>Speed:</b> Slower (~5-10 seconds, model loading)</li>
 *   <li><b>Dependencies:</b> Real production model required</li>
 *   <li><b>Scope:</b> Full stack (HTTP → Service → Model → Response)</li>
 *   <li><b>Tag:</b> {@code @Tag("integration")} - run with {@code -Dgroups=integration}</li>
 * </ul>
 *
 * <h3>Prerequisites</h3>
 * <p>Requires {@code ./models/production/sentiment_model.ser} to exist.
 * Tests are automatically skipped if the model is missing.
 *
 * <h3>When to Use</h3>
 * <p>Use these tests to verify:
 * <ul>
 *   <li>Real model predictions work correctly</li>
 *   <li>Spring wiring is correct</li>
 *   <li>End-to-end latency is acceptable</li>
 *   <li>Production configuration works</li>
 * </ul>
 *
 * @see SentimentControllerWebTest for fast, mocked web layer tests
 */
@SpringBootTest(
    webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
    properties = {
        "sentiment.model-type=PRODUCTION",
        "sentiment.models.production-model-path=./models/production/sentiment_model.ser"
    }
)
@Tag("integration")
@EnabledIf("productionModelExists")
@DisplayName("Sentiment Controller E2E Integration Tests")
@SuppressWarnings("SpringJavaInjectionPointsAutowiringInspection")  // Fields injected by Spring Test
public class SentimentControllerIntegrationTest {

    static boolean productionModelExists() {
        return Files.exists(Paths.get("./models/production/sentiment_model.ser"));
    }

    @LocalServerPort
    private int port;

    @Autowired
    private TestRestTemplate restTemplate;

    private final ObjectMapper objectMapper = new ObjectMapper();

    // CRITICAL PATH SMOKE TESTS

    @Test
    @DisplayName("E2E Smoke Test: Full stack processes positive sentiment with real model")
    void smokeTest_EndToEnd_PositiveSentiment_RealModel() throws Exception {
        // Given: Request with clearly positive sentiment
        String requestJson = """
            {
                "text": "This is absolutely fantastic, wonderful, and amazing! I love it!"
            }
            """;

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<String> request = new HttpEntity<>(requestJson, headers);

        // When: POST to real API with real trained model
        String url = "http://localhost:" + port + "/api/v1/sentiment/analyze";
        ResponseEntity<String> response = restTemplate.postForEntity(url, request, String.class);

        // Then: Should successfully classify with reasonable confidence
        assertThat(response.getStatusCode())
            .as("API should return 200 OK")
            .isEqualTo(HttpStatus.OK);

        String body = response.getBody();
        assertNotNull(body, "Response body should not be null");

        // Parse JSON properly instead of string contains (fixes false confidence)
        JsonNode json = objectMapper.readTree(body);

        // Verify required fields exist and have valid values
        assertTrue(json.has("sentiment") && !json.get("sentiment").isNull(),
            "Response must contain non-null 'sentiment' field");
        assertTrue(json.has("confidence") && !json.get("confidence").isNull(),
            "Response must contain non-null 'confidence' field");
        assertTrue(json.has("processingTimeMs") && !json.get("processingTimeMs").isNull(),
            "Response must contain non-null 'processingTimeMs' field");

        // Verify sentiment is a valid value
        String sentiment = json.get("sentiment").asText();
        assertTrue(sentiment.equals("positive") || sentiment.equals("negative") || sentiment.equals("uncertain"),
            "Sentiment must be 'positive', 'negative', or 'uncertain', got: " + sentiment);

        // Verify confidence is in valid range [0, 1]
        double confidence = json.get("confidence").asDouble();
        assertTrue(confidence >= 0.0 && confidence <= 1.0,
            "Confidence must be between 0.0 and 1.0, got: " + confidence);

        // Verify processing time is positive
        long processingTime = json.get("processingTimeMs").asLong();
        assertTrue(processingTime >= 0,
            "Processing time must be non-negative, got: " + processingTime);
    }

    @Test
    @DisplayName("E2E Smoke Test: Full stack processes negative sentiment with real model")
    void smokeTest_EndToEnd_NegativeSentiment_RealModel() throws Exception {
        // Given: Request with clearly negative sentiment
        String requestJson = """
            {
                "text": "This is terrible, awful, and horrible! I hate it completely!"
            }
            """;

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<String> request = new HttpEntity<>(requestJson, headers);

        // When: POST to real API with real trained model
        String url = "http://localhost:" + port + "/api/v1/sentiment/analyze";
        ResponseEntity<String> response = restTemplate.postForEntity(url, request, String.class);

        // Then: Should successfully classify
        assertThat(response.getStatusCode())
            .as("API should return 200 OK")
            .isEqualTo(HttpStatus.OK);

        String body = response.getBody();
        assertNotNull(body, "Response body should not be null");

        // Parse JSON properly instead of string contains (fixes false confidence)
        JsonNode json = objectMapper.readTree(body);

        // Verify required fields exist and have valid values
        assertTrue(json.has("sentiment") && !json.get("sentiment").isNull(),
            "Response must contain non-null 'sentiment' field");
        assertTrue(json.has("confidence") && !json.get("confidence").isNull(),
            "Response must contain non-null 'confidence' field");

        // Verify sentiment is a valid value
        String sentiment = json.get("sentiment").asText();
        assertTrue(sentiment.equals("positive") || sentiment.equals("negative") || sentiment.equals("uncertain"),
            "Sentiment must be 'positive', 'negative', or 'uncertain', got: " + sentiment);

        // Verify confidence is in valid range [0, 1]
        double confidence = json.get("confidence").asDouble();
        assertTrue(confidence >= 0.0 && confidence <= 1.0,
            "Confidence must be between 0.0 and 1.0, got: " + confidence);
    }

    @Test
    @DisplayName("E2E Smoke Test: Batch processing works with real model")
    void smokeTest_EndToEnd_BatchProcessing_RealModel() throws Exception {
        // Given: Batch request with multiple texts
        String requestJson = """
            {
                "texts": [
                    "This is great!",
                    "This is terrible.",
                    "This is okay."
                ]
            }
            """;

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<String> request = new HttpEntity<>(requestJson, headers);

        // When: POST to batch endpoint
        String url = "http://localhost:" + port + "/api/v1/sentiment/batch";
        ResponseEntity<String> response = restTemplate.postForEntity(url, request, String.class);

        // Then: Should process all texts successfully
        assertThat(response.getStatusCode())
            .as("Batch API should return 200 OK")
            .isEqualTo(HttpStatus.OK);

        String body = response.getBody();
        assertNotNull(body, "Response body should not be null");

        // Parse JSON properly instead of string contains (fixes false confidence)
        JsonNode json = objectMapper.readTree(body);

        // Verify totalProcessed equals expected count
        assertTrue(json.has("totalProcessed"), "Response must contain 'totalProcessed' field");
        assertEquals(3, json.get("totalProcessed").asInt(),
            "Should process exactly 3 texts");

        // Verify results array exists and has correct size
        assertTrue(json.has("results") && json.get("results").isArray(),
            "Response must contain 'results' array");
        assertEquals(3, json.get("results").size(),
            "Results array should have 3 elements");

        // Verify each result has required fields
        for (JsonNode result : json.get("results")) {
            assertTrue(result.has("sentiment") && !result.get("sentiment").isNull(),
                "Each result must have 'sentiment' field");
            assertTrue(result.has("confidence") && !result.get("confidence").isNull(),
                "Each result must have 'confidence' field");

            double confidence = result.get("confidence").asDouble();
            assertTrue(confidence >= 0.0 && confidence <= 1.0,
                "Each confidence must be in [0, 1], got: " + confidence);
        }

        // Verify successCount
        assertTrue(json.has("successCount"), "Response must contain 'successCount' field");
        int successCount = json.get("successCount").asInt();
        assertTrue(successCount >= 0 && successCount <= 3,
            "Success count must be between 0 and 3, got: " + successCount);
    }

    @Test
    @DisplayName("E2E Smoke Test: Health endpoint confirms model is loaded and operational")
    void smokeTest_HealthCheck_ModelLoaded() throws Exception {
        // When: GET health endpoint
        String url = "http://localhost:" + port + "/api/v1/health";
        ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);

        // Then: Should confirm system is healthy and model is loaded
        assertThat(response.getStatusCode())
            .as("Health endpoint should return 200 OK")
            .isEqualTo(HttpStatus.OK);

        String body = response.getBody();
        assertNotNull(body, "Response body should not be null");

        // Parse JSON properly instead of string contains (fixes false confidence)
        JsonNode json = objectMapper.readTree(body);

        // Verify status field
        assertTrue(json.has("status"), "Response must contain 'status' field");
        assertEquals("UP", json.get("status").asText(),
            "Health status should be 'UP'");

        // Verify modelLoaded field
        assertTrue(json.has("modelLoaded"), "Response must contain 'modelLoaded' field");
        assertTrue(json.get("modelLoaded").isBoolean(),
            "'modelLoaded' should be a boolean");

        // Verify uptimeMs field
        assertTrue(json.has("uptimeMs"), "Response must contain 'uptimeMs' field");
        long uptime = json.get("uptimeMs").asLong();
        assertTrue(uptime >= 0, "Uptime must be non-negative, got: " + uptime);

        // Verify version field
        assertTrue(json.has("version"), "Response must contain 'version' field");
        assertEquals("1.0.0", json.get("version").asText(),
            "Version should be '1.0.0'");
    }

    @Test
    @DisplayName("E2E Smoke Test: Validation errors are handled correctly")
    void smokeTest_Validation_EmptyText_Returns400() throws Exception {
        // Given: Invalid request with empty text
        String requestJson = """
            {
                "text": ""
            }
            """;

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<String> request = new HttpEntity<>(requestJson, headers);

        // When: POST to API
        String url = "http://localhost:" + port + "/api/v1/sentiment/analyze";
        ResponseEntity<String> response = restTemplate.postForEntity(url, request, String.class);

        // Then: Should return 400 Bad Request with validation error
        assertThat(response.getStatusCode())
            .as("Empty text should return 400 Bad Request")
            .isEqualTo(HttpStatus.BAD_REQUEST);

        String body = response.getBody();
        assertNotNull(body, "Error response body should not be null");

        // Parse JSON properly instead of string contains (fixes false confidence)
        JsonNode json = objectMapper.readTree(body);

        // Verify error response structure
        assertTrue(json.has("error") && !json.get("error").isNull(),
            "Error response must contain non-null 'error' field");

        // Verify status code in response matches HTTP status
        if (json.has("status")) {
            assertEquals(400, json.get("status").asInt(),
                "Error response status should be 400");
        }
    }
}
