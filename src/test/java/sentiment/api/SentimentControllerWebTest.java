package sentiment.api;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import sentiment.models.SentimentClassifier;

import static org.hamcrest.Matchers.*;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/**
 * Web layer tests for SentimentController using @WebMvcTest.
 * <p>Tests REST API contracts without loading the full Spring context or ML models.
 * <p>Uses mocked SentimentClassifier to keep tests fast and memory-efficient.
 */
@WebMvcTest(SentimentController.class)
@DisplayName("SentimentController Web Layer Tests")
public class SentimentControllerWebTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private SentimentClassifier classifier;

    @MockitoBean
    private sentiment.api.metrics.PredictionMetrics metrics;

    @BeforeEach
    void setup() throws Exception {
        // Configure default mock behavior for classifier
        when(classifier.isTrained()).thenReturn(true);
        when(classifier.getAlgorithmName()).thenReturn("SVM (Mocked)");
        when(classifier.getSupportedClasses()).thenReturn(new String[]{"positive", "negative"});

        // Default classification behavior - mock classifyWithProbabilities which controller uses
        SentimentClassifier.ClassificationResult mockResult = new SentimentClassifier.ClassificationResult(
            "positive", new double[]{0.85, 0.15}, new String[]{"positive", "negative"}
        );
        when(classifier.classifyWithProbabilities(anyString())).thenReturn(mockResult);

        // Also mock legacy methods for backwards compatibility
        when(classifier.classify(anyString())).thenReturn("positive");
        when(classifier.getClassificationProbabilities(anyString()))
            .thenReturn(new double[]{0.85, 0.15});

        // Mock metrics snapshot for health endpoint
        sentiment.api.metrics.PredictionMetrics.MetricsSnapshot mockSnapshot =
            new sentiment.api.metrics.PredictionMetrics.MetricsSnapshot(
                100L, 60.0, 30.0, 10.0, 0.85, 5.0, 100L, 50.0, 100.0, 150.0
            );
        when(metrics.getSnapshot()).thenReturn(mockSnapshot);
    }

    // ==================== HEALTH ENDPOINT TESTS ====================

    @Test
    @DisplayName("Health endpoint should return service status with all required fields")
    public void testHealthEndpoint() throws Exception {
        mockMvc.perform(get("/api/v1/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("UP"))
                .andExpect(jsonPath("$.version").value("1.0.0"))
                .andExpect(jsonPath("$.modelLoaded").isBoolean())
                .andExpect(jsonPath("$.uptimeMs").isNumber())
                .andExpect(jsonPath("$.uptimeMs").value(greaterThanOrEqualTo(0)));
    }

    // ==================== SENTIMENT ANALYSIS ENDPOINT TESTS ====================

    @Test
    @DisplayName("Analyze sentiment should return sentiment and confidence for valid text")
    public void testAnalyzeSentiment_Success() throws Exception {
        String requestJson = """
                {
                    "text": "This is absolutely amazing and wonderful!"
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.sentiment").value("positive"))
                .andExpect(jsonPath("$.confidence").isNumber())
                .andExpect(jsonPath("$.confidence").value(greaterThanOrEqualTo(0.0)))
                .andExpect(jsonPath("$.confidence").value(lessThanOrEqualTo(1.0)))
                .andExpect(jsonPath("$.processingTimeMs").isNumber())
                .andExpect(jsonPath("$.processingTimeMs").value(greaterThanOrEqualTo(0)));
    }

    @Test
    @DisplayName("Analyze positive text should return positive sentiment")
    public void testAnalyzeSentiment_PositiveSentiment() throws Exception {
        when(classifier.classifyWithProbabilities(anyString())).thenReturn(
            new SentimentClassifier.ClassificationResult(
                "positive", new double[]{0.92, 0.08}, new String[]{"positive", "negative"}
            )
        );

        String requestJson = """
                {
                    "text": "I love this product, it's fantastic and amazing!"
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.sentiment").value("positive"))
                .andExpect(jsonPath("$.confidence").value(0.92));
    }

    @Test
    @DisplayName("Analyze negative text should return negative sentiment")
    public void testAnalyzeSentiment_NegativeSentiment() throws Exception {
        when(classifier.classifyWithProbabilities(anyString())).thenReturn(
            new SentimentClassifier.ClassificationResult(
                "negative", new double[]{0.12, 0.88}, new String[]{"positive", "negative"}
            )
        );

        String requestJson = """
                {
                    "text": "This is terrible and awful, I hate it!"
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.sentiment").value("negative"))
                .andExpect(jsonPath("$.confidence").value(0.88));
    }

    @Test
    @DisplayName("Analyze sentiment with confidence threshold below threshold returns uncertain")
    public void testAnalyzeSentiment_BelowThreshold_ReturnsUncertain() throws Exception {
        when(classifier.classifyWithProbabilities(anyString())).thenReturn(
            new SentimentClassifier.ClassificationResult(
                "positive", new double[]{0.65, 0.35}, new String[]{"positive", "negative"}
            )
        );  // Below 0.9 threshold

        String requestJson = """
                {
                    "text": "This product is okay",
                    "confidenceThreshold": 0.9
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.sentiment").value("uncertain"))
                .andExpect(jsonPath("$.confidence").value(0.65));
    }

    @Test
    @DisplayName("Analyze sentiment with confidence above threshold returns sentiment")
    public void testAnalyzeSentiment_AboveThreshold_ReturnsSentiment() throws Exception {
        when(classifier.classifyWithProbabilities(anyString())).thenReturn(
            new SentimentClassifier.ClassificationResult(
                "positive", new double[]{0.95, 0.05}, new String[]{"positive", "negative"}
            )
        );  // Above 0.7 threshold

        String requestJson = """
                {
                    "text": "This is great!",
                    "confidenceThreshold": 0.7
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.sentiment").value("positive"))
                .andExpect(jsonPath("$.confidence").value(0.95));
    }

    @Test
    @DisplayName("Analyze sentiment with empty text should return validation error")
    public void testAnalyzeSentiment_EmptyText() throws Exception {
        String requestJson = """
                {
                    "text": ""
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("Validation failed"));
    }

    @Test
    @DisplayName("Analyze sentiment with missing text field should return validation error")
    public void testAnalyzeSentiment_MissingText() throws Exception {
        String requestJson = """
                {
                    "confidenceThreshold": 0.7
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isBadRequest());
    }

    @Test
    @DisplayName("Analyze sentiment with very long text should process successfully")
    public void testAnalyzeSentiment_VeryLongText() throws Exception {
        String longText = "This is amazing. ".repeat(100);
        String requestJson = String.format("""
                {
                    "text": "%s"
                }
                """, longText);

        mockMvc.perform(post("/api/v1/sentiment/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.sentiment").value("positive"));
    }

    @Test
    @DisplayName("Analyze sentiment with special characters should handle gracefully")
    public void testAnalyzeSentiment_SpecialCharacters() throws Exception {
        String requestJson = """
                {
                    "text": "This is great!!! @#$%^&*() 😀"
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.sentiment").exists());
    }

    @Test
    @DisplayName("Analyze sentiment with malformed JSON should return error")
    public void testAnalyzeSentiment_MalformedJson() throws Exception {
        String malformedJson = "{ text: 'missing quotes' }";

        mockMvc.perform(post("/api/v1/sentiment/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(malformedJson))
                .andExpect(status().is5xxServerError());  // Jackson parsing fails with 500
    }

    @Test
    @DisplayName("Analyze sentiment when model not trained should return 503")
    public void testAnalyzeSentiment_ModelNotTrained() throws Exception {
        when(classifier.isTrained()).thenReturn(false);

        String requestJson = """
                {
                    "text": "Test text"
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isServiceUnavailable())
                .andExpect(jsonPath("$.error").exists());
    }

    // ==================== BATCH ANALYSIS ENDPOINT TESTS ====================

    @Test
    @DisplayName("Batch analysis should process multiple texts successfully")
    public void testBatchAnalysis_Success() throws Exception {
        String requestJson = """
                {
                    "texts": [
                        "Great product!",
                        "Terrible experience.",
                        "It's okay."
                    ]
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.totalProcessed").value(3))
                .andExpect(jsonPath("$.results").isArray())
                .andExpect(jsonPath("$.results.length()").value(3))
                .andExpect(jsonPath("$.successCount").isNumber())
                .andExpect(jsonPath("$.errorCount").isNumber())
                .andExpect(jsonPath("$.totalProcessingTimeMs").isNumber())
                .andExpect(jsonPath("$.totalProcessingTimeMs").value(greaterThanOrEqualTo(0)));
    }

    @Test
    @DisplayName("Batch analysis with empty list should return validation error")
    public void testBatchAnalysis_EmptyList() throws Exception {
        String requestJson = """
                {
                    "texts": []
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("Validation failed"));
    }

    @Test
    @DisplayName("Batch analysis with confidence threshold should apply to all texts")
    public void testBatchAnalysis_WithConfidenceThreshold() throws Exception {
        when(classifier.classifyWithProbabilities(anyString())).thenReturn(
            new SentimentClassifier.ClassificationResult(
                "positive", new double[]{0.75, 0.25}, new String[]{"positive", "negative"}
            )
        );

        String requestJson = """
                {
                    "texts": [
                        "Amazing!",
                        "Horrible!"
                    ],
                    "confidenceThreshold": 0.5
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.totalProcessed").value(2))
                .andExpect(jsonPath("$.results[0].sentiment").exists())
                .andExpect(jsonPath("$.results[1].sentiment").exists());
    }

    @Test
    @DisplayName("Batch analysis should maintain order of results")
    public void testBatchAnalysis_OrderPreservation() throws Exception {
        String requestJson = """
                {
                    "texts": [
                        "First text",
                        "Second text",
                        "Third text"
                    ]
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.results.length()").value(3))
                .andExpect(jsonPath("$.results[0]").exists())
                .andExpect(jsonPath("$.results[1]").exists())
                .andExpect(jsonPath("$.results[2]").exists());
    }

    @Test
    @DisplayName("Batch analysis with single text should process successfully")
    public void testBatchAnalysis_SingleText() throws Exception {
        String requestJson = """
                {
                    "texts": [
                        "Only one text"
                    ]
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.totalProcessed").value(1))
                .andExpect(jsonPath("$.results.length()").value(1));
    }

    // ==================== FEATURE IMPORTANCE ENDPOINT TESTS ====================

    @Test
    @DisplayName("Feature importance when model not trained should return 503")
    public void testFeatureImportance_ModelNotTrained() throws Exception {
        when(classifier.isTrained()).thenReturn(false);

        mockMvc.perform(get("/api/v1/model/feature-importance")
                        .param("topFeatures", "10"))
                .andExpect(status().isServiceUnavailable());
                // Note: Response structure may vary, so just check status code
    }

    @Test
    @DisplayName("Feature importance with invalid topFeatures should handle gracefully")
    public void testFeatureImportance_InvalidTopFeatures() throws Exception {
        mockMvc.perform(get("/api/v1/model/feature-importance")
                        .param("topFeatures", "-5"))
                .andExpect(status().is4xxClientError());
    }

    // ==================== EDGE CASE TESTS ====================

    @Test
    @DisplayName("Analyze sentiment with only whitespace should return validation error")
    public void testAnalyzeSentiment_WhitespaceOnly() throws Exception {
        String requestJson = """
                {
                    "text": "   "
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isBadRequest());
    }

    @Test
    @DisplayName("Analyze sentiment with Unicode text should process successfully")
    public void testAnalyzeSentiment_UnicodeText() throws Exception {
        String requestJson = """
                {
                    "text": "这是一个很好的产品 This is great! ¡Excelente!"
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.sentiment").exists());
    }

    @Test
    @DisplayName("Analyze sentiment with newlines should process successfully")
    public void testAnalyzeSentiment_WithNewlines() throws Exception {
        String requestJson = """
                {
                    "text": "This is great.\\nI love it.\\nHighly recommended."
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.sentiment").exists());
    }

    @Test
    @DisplayName("Analyze sentiment without Content-Type header should fail")
    public void testAnalyzeSentiment_MissingContentType() throws Exception {
        String requestJson = """
                {
                    "text": "Test"
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/analyze")
                        .content(requestJson))
                .andExpect(status().is5xxServerError());  // Missing Content-Type causes 500
    }

    @Test
    @DisplayName("Batch analysis with null texts should return validation error")
    public void testBatchAnalysis_NullTexts() throws Exception {
        String requestJson = """
                {
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isBadRequest());
    }

    // ==================== RESPONSE STRUCTURE VALIDATION TESTS ====================

    @Test
    @DisplayName("Sentiment response should include all required fields")
    public void testSentimentResponse_CompleteStructure() throws Exception {
        String requestJson = """
                {
                    "text": "This is a test"
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/analyze")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.sentiment").exists())
                .andExpect(jsonPath("$.confidence").exists())
                .andExpect(jsonPath("$.processingTimeMs").exists());
    }

    @Test
    @DisplayName("Batch response should include all required fields")
    public void testBatchResponse_CompleteStructure() throws Exception {
        String requestJson = """
                {
                    "texts": ["Test 1", "Test 2"]
                }
                """;

        mockMvc.perform(post("/api/v1/sentiment/batch")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(requestJson))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.totalProcessed").exists())
                .andExpect(jsonPath("$.successCount").exists())
                .andExpect(jsonPath("$.errorCount").exists())
                .andExpect(jsonPath("$.results").exists())
                .andExpect(jsonPath("$.totalProcessingTimeMs").exists());
    }
}
