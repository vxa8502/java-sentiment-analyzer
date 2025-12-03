package sentiment.api;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * End-to-end integration smoke tests with real Spring Boot context and ML model.
 */
@SpringBootTest(
    webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
    properties = {
        "sentiment.models.svm-model-path=./models/svm-model.ser",
        "sentiment.models.nb-model-path=./models/nb-model.ser",
        "sentiment.models.rf-model-path=./models/rf-model.ser",
        "sentiment.models.lr-model-path=./models/lr-model.ser"
    }
)
@Tag("integration")
@DisplayName("Sentiment Controller E2E Integration Tests")
@SuppressWarnings("SpringJavaInjectionPointsAutowiringInspection")  // Fields injected by Spring Test
public class SentimentControllerIntegrationTest {

    @LocalServerPort
    private int port;

    @Autowired
    private TestRestTemplate restTemplate;

    // CRITICAL PATH SMOKE TESTS

    @Test
    @DisplayName("E2E Smoke Test: Full stack processes positive sentiment with real model")
    void smokeTest_EndToEnd_PositiveSentiment_RealModel() {
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
        assertThat(body)
            .as("Response should contain sentiment field")
            .contains("\"sentiment\"");

        assertThat(body)
            .as("Response should contain confidence field")
            .contains("\"confidence\"");

        assertThat(body)
            .as("Response should contain processingTimeMs field")
            .contains("\"processingTimeMs\"");

        // Verify confidence is a reasonable probability value
        assertThat(body)
            .as("Confidence should be a numeric value")
            .containsPattern("\"confidence\"\\s*:\\s*0\\.\\d+");
    }

    @Test
    @DisplayName("E2E Smoke Test: Full stack processes negative sentiment with real model")
    void smokeTest_EndToEnd_NegativeSentiment_RealModel() {
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
        assertThat(body)
            .as("Response should contain sentiment classification")
            .contains("\"sentiment\"");

        assertThat(body)
            .as("Response should contain confidence score")
            .contains("\"confidence\"");
    }

    @Test
    @DisplayName("E2E Smoke Test: Batch processing works with real model")
    void smokeTest_EndToEnd_BatchProcessing_RealModel() {
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
        assertThat(body)
            .as("Response should indicate 3 texts processed")
            .contains("\"totalProcessed\":3");

        assertThat(body)
            .as("Response should contain results array")
            .contains("\"results\"");

        assertThat(body)
            .as("Response should contain success count")
            .contains("\"successCount\"");
    }

    @Test
    @DisplayName("E2E Smoke Test: Health endpoint confirms model is loaded and operational")
    void smokeTest_HealthCheck_ModelLoaded() {
        // When: GET health endpoint
        String url = "http://localhost:" + port + "/api/v1/health";
        ResponseEntity<String> response = restTemplate.getForEntity(url, String.class);

        // Then: Should confirm system is healthy and model is loaded
        assertThat(response.getStatusCode())
            .as("Health endpoint should return 200 OK")
            .isEqualTo(HttpStatus.OK);

        String body = response.getBody();
        assertThat(body)
            .as("Health status should be UP")
            .contains("\"status\":\"UP\"");

        assertThat(body)
            .as("Should report model loaded status")
            .contains("\"modelLoaded\"");

        assertThat(body)
            .as("Should report uptime")
            .contains("\"uptimeMs\"");

        assertThat(body)
            .as("Should report version")
            .contains("\"version\":\"1.0.0\"");
    }

    @Test
    @DisplayName("E2E Smoke Test: Validation errors are handled correctly")
    void smokeTest_Validation_EmptyText_Returns400() {
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
        assertThat(body)
            .as("Error response should contain error field")
            .contains("error");
    }
}
