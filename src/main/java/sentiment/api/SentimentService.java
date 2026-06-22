package sentiment.api;

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import sentiment.monitoring.DriftStatistics;
import sentiment.monitoring.PredictionLogRecord;
import sentiment.monitoring.PredictionLogger;
import sentiment.monitoring.PredictionMetrics;
import sentiment.models.SentimentClassifier;

import java.time.Duration;
import java.util.regex.Pattern;

/**
 * Service layer for sentiment classification operations.
 * Encapsulates model inference with circuit breaker protection.
 *
 * <p>The circuit breaker is applied here (not in the controller) because Spring AOP
 * requires public methods called through the proxy to intercept annotations.
 * Internal method calls within the same class bypass the proxy entirely.
 *
 * <h2>Logging Strategy</h2>
 * <ul>
 *   <li><b>ERROR</b>: Unexpected failures requiring investigation (model crashes, infrastructure issues)</li>
 *   <li><b>WARN</b>: Expected error conditions (invalid input, model not ready, known exceptions)</li>
 *   <li><b>INFO</b>: Important state changes (circuit breaker state, model reload)</li>
 *   <li><b>DEBUG</b>: Request details, confidence values, processing times</li>
 * </ul>
 */
@Service
public class SentimentService {

    private static final Logger logger = LoggerFactory.getLogger(SentimentService.class);

    // Pre-compiled patterns for log sanitization (performance fix - avoid regex compilation per call)
    private static final Pattern NEWLINE_PATTERN = Pattern.compile("[\n\r]");
    private static final Pattern NON_PRINTABLE_PATTERN = Pattern.compile("[^\\x20-\\x7E]");

    private final SentimentClassifier classifier;
    private final PredictionMetrics metrics;
    private final ObjectProvider<PredictionLogger> predictionLoggerProvider;
    private final ObjectProvider<DriftStatistics> driftStatisticsProvider;

    public SentimentService(
            SentimentClassifier classifier,
            PredictionMetrics metrics,
            ObjectProvider<PredictionLogger> predictionLoggerProvider,
            ObjectProvider<DriftStatistics> driftStatisticsProvider) {
        this.classifier = classifier;
        this.metrics = metrics;
        this.predictionLoggerProvider = predictionLoggerProvider;
        this.driftStatisticsProvider = driftStatisticsProvider;
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
                // WARN not ERROR: This is expected during startup while model loads
                logger.warn("Classification attempted while model not yet trained");
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

            // Log prediction for drift detection (async, non-blocking)
            logPredictionForDrift(text, sentiment, confidence, processingTime);

            if (warning != null) {
                return ResponseEntity.ok(
                    SentimentResponse.successWithWarning(sentiment, confidence, text, processingTime, warning)
                );
            } else {
                return ResponseEntity.ok(
                    SentimentResponse.success(sentiment, confidence, text, processingTime)
                );
            }

        } catch (sentiment.models.ClassificationException e) {
            // Handle domain-specific classification errors with appropriate status codes
            return handleClassificationException(e, text);
        } catch (IllegalStateException | IllegalArgumentException e) {
            logger.warn("Classification failed due to invalid input: {}", e.getMessage());
            return ResponseEntity.status(HttpStatus.BAD_REQUEST)
                    .body(SentimentResponse.error(e.getMessage(), text));
        } catch (Exception e) {
            // Unexpected errors - log full stack trace for debugging
            logger.error("Unexpected classification error for text: {}",
                        sanitizeForLogging(text), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(SentimentResponse.error("Classification failed unexpectedly. Please try again.", text));
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
     * Handles ClassificationException and maps error types to HTTP status codes.
     * This provides consistent error responses based on the specific failure type.
     */
    private ResponseEntity<SentimentResponse> handleClassificationException(
            sentiment.models.ClassificationException e, String text) {

        return switch (e.getErrorType()) {
            case NOT_TRAINED -> {
                logger.warn("Classification attempted on untrained model: {}", e.getMessage());
                yield ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                        .body(SentimentResponse.error(
                            "Model not ready. Please wait for initialization to complete.", text));
            }
            case INVALID_INPUT -> {
                logger.debug("Invalid classification input: {}", e.getMessage());
                yield ResponseEntity.status(HttpStatus.BAD_REQUEST)
                        .body(SentimentResponse.error(e.getMessage(), text));
            }
            case INFERENCE_ERROR -> {
                logger.error("Model inference failed: {}", e.getMessage(), e);
                yield ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                        .body(SentimentResponse.error(
                            "Classification failed due to model error. Please try again.", text));
            }
            case TRAINING_ERROR -> {
                logger.error("Training error during classification: {}", e.getMessage(), e);
                yield ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                        .body(SentimentResponse.error(
                            "Model is in an inconsistent state. Please contact support.", text));
            }
            case UNKNOWN -> {
                logger.error("Unknown classification error: {}", e.getMessage(), e);
                yield ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                        .body(SentimentResponse.error("Classification failed. Please try again.", text));
            }
        };
    }

    /**
     * Sanitizes text for logging to prevent log injection attacks.
     * Uses pre-compiled patterns to avoid regex compilation on each call.
     */
    private String sanitizeForLogging(String text) {
        if (text == null) {
            return "null";
        }
        String truncated = text.substring(0, Math.min(50, text.length()));
        String noNewlines = NEWLINE_PATTERN.matcher(truncated).replaceAll(" ");
        return NON_PRINTABLE_PATTERN.matcher(noNewlines).replaceAll("?");
    }

    /**
     * Logs prediction for drift detection (async, non-blocking).
     * Safe to call even if drift detection is disabled.
     */
    private void logPredictionForDrift(String text, String sentiment, double confidence, long processingTimeMs) {
        PredictionLogger predictionLogger = predictionLoggerProvider.getIfAvailable();
        DriftStatistics driftStatistics = driftStatisticsProvider.getIfAvailable();

        if (predictionLogger == null && driftStatistics == null) {
            return; // Drift detection disabled
        }

        PredictionLogRecord record = PredictionLogRecord.create(
                text,
                sentiment,
                confidence,
                processingTimeMs,
                classifier.getAlgorithmName()
        );

        if (predictionLogger != null) {
            predictionLogger.logPrediction(record);
        }

        if (driftStatistics != null) {
            driftStatistics.recordForDrift(record);
        }
    }
}
