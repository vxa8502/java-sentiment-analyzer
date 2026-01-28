package sentiment.api;

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import sentiment.api.metrics.PredictionMetrics;
import sentiment.models.SentimentClassifier;

import java.time.Duration;

/**
 * Service layer for sentiment classification operations.
 * Encapsulates model inference with circuit breaker protection.
 *
 * The circuit breaker is applied here (not in the controller) because Spring AOP
 * requires public methods called through the proxy to intercept annotations.
 * Internal method calls within the same class bypass the proxy entirely.
 */
@Service
public class SentimentService {

    private static final Logger logger = LoggerFactory.getLogger(SentimentService.class);

    private final SentimentClassifier classifier;
    private final PredictionMetrics metrics;

    public SentimentService(SentimentClassifier classifier, PredictionMetrics metrics) {
        this.classifier = classifier;
        this.metrics = metrics;
    }

    /**
     * Classifies text and applies optional confidence thresholding.
     * Protected by circuit breaker to prevent cascading failures when model fails.
     *
     * @param text the text to classify
     * @param confidenceThreshold optional threshold below which sentiment is marked "uncertain"
     * @return ResponseEntity containing the classification result
     */
    @CircuitBreaker(name = "modelInference", fallbackMethod = "classifyTextFallback")
    public ResponseEntity<SentimentResponse> classifyText(String text, Double confidenceThreshold) {
        long startTime = System.currentTimeMillis();

        try {
            if (!classifier.isTrained()) {
                logger.error("Attempted classification with untrained model");
                return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                        .body(SentimentResponse.error(
                            "Model initialization in progress. Please retry in 30 seconds. " +
                            "Check /api/v1/health for status.", text));
            }

            SentimentClassifier.ClassificationResult result = classifier.classifyWithProbabilities(text);
            String sentiment = result.label();
            double confidence = result.confidence();

            String warning = null;

            if (confidenceThreshold != null && confidence < confidenceThreshold) {
                logger.debug("Confidence {} below threshold {}, marking uncertain",
                           confidence, confidenceThreshold);
                sentiment = "uncertain";
            }

            if (confidence < 0.75) {
                warning = "Low confidence prediction. The model classifies as 'positive' or 'negative' only. " +
                         "Ambiguous text may produce unreliable results. Use confidenceThreshold parameter to get 'uncertain' for low-confidence predictions.";
            }

            long processingTime = System.currentTimeMillis() - startTime;
            logger.debug("Classification: '{}' (confidence: {}) in {}ms",
                       sentiment, confidence, processingTime);

            metrics.recordPrediction(sentiment, confidence, Duration.ofMillis(processingTime));

            if (warning != null) {
                return ResponseEntity.ok(
                    SentimentResponse.successWithWarning(sentiment, confidence, text, processingTime, warning)
                );
            } else {
                return ResponseEntity.ok(
                    SentimentResponse.success(sentiment, confidence, text, processingTime)
                );
            }

        } catch (IllegalStateException | IllegalArgumentException e) {
            logger.warn("Classification failed due to invalid input: {}", e.getMessage());
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(SentimentResponse.error(e.getMessage(), text));
        } catch (Exception e) {
            logger.error("Classification failed for text: {}",
                        sanitizeForLogging(text), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(SentimentResponse.error("Classification failed. Please try again.", text));
        }
    }

    /**
     * Fallback method for classifyText when circuit breaker is OPEN.
     * Returns a graceful error response instead of cascading failures.
     */
    public ResponseEntity<SentimentResponse> classifyTextFallback(String text, Double confidenceThreshold, Exception e) {
        logger.warn("Circuit breaker OPEN - model inference unavailable. Cause: {}", e.getMessage());

        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                .body(SentimentResponse.error(
                    "Sentiment analysis temporarily unavailable. The model is experiencing issues " +
                    "and is recovering. Please retry in 30 seconds. Check /actuator/health for " +
                    "circuit breaker status.", text));
    }

    /**
     * Returns the underlying classifier for health checks and feature extraction.
     */
    public SentimentClassifier getClassifier() {
        return classifier;
    }

    /**
     * Returns the metrics collector for health endpoint.
     */
    public PredictionMetrics getMetrics() {
        return metrics;
    }

    /**
     * Sanitizes text for logging to prevent log injection attacks.
     */
    private String sanitizeForLogging(String text) {
        if (text == null) {
            return "null";
        }
        String truncated = text.substring(0, Math.min(50, text.length()));
        return truncated
                .replaceAll("[\n\r]", " ")
                .replaceAll("[^\\x20-\\x7E]", "?");
    }
}
