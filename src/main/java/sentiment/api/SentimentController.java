package sentiment.api;

import io.github.resilience4j.ratelimiter.annotation.RateLimiter;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import sentiment.models.SentimentClassifier;

import java.util.ArrayList;
import java.util.List;

/**
 * REST API controller for sentiment analysis operations.
 * Endpoints:
 * - POST /api/v1/sentiment/analyze - Analyze single text
 * - POST /api/v1/sentiment/batch - Batch analysis
 * - GET /api/v1/health - Health check
 */
@RestController
@RequestMapping("/api/v1")
@org.springframework.context.annotation.Profile("!training")
public class SentimentController {

    private static final Logger logger = LoggerFactory.getLogger(SentimentController.class);
    private final SentimentClassifier classifier;
    private final long startTime;

    public SentimentController(SentimentClassifier classifier) {
        this.classifier = classifier;
        this.startTime = System.currentTimeMillis();
        logger.info("SentimentController initialized with {} classifier",
                   classifier.getAlgorithmName());
    }

    /**
     * Analyzes sentiment for a single text input.
     * Example request:
     * POST /api/v1/sentiment/analyze
     * {
     *   "text": "This movie was absolutely amazing!",
     *   "confidenceThreshold": 0.7
     * }
     * Example response:
     * {
     *   "sentiment": "positive",
     *   "confidence": 0.92,
     *   "text": "This movie was absolutely amazing!",
     *   "processingTimeMs": 45
     * }
     */
    @PostMapping("/sentiment/analyze")
    @RateLimiter(name = "sentimentApi")
    public ResponseEntity<SentimentResponse> analyzeSentiment(
            @Valid @RequestBody SentimentRequest request,
            HttpServletRequest httpRequest) {

        String clientIp = extractClientIp(httpRequest);
        logger.info("REQUEST: client={}, textLength={}", clientIp, request.getText().length());

        ResponseEntity<SentimentResponse> response = classifyText(
            request.getText(), request.getConfidenceThreshold());

        logger.info("RESPONSE: client={}, sentiment={}, time={}ms",
                    clientIp, response.getBody().getSentiment(),
                    response.getBody().getProcessingTimeMs());

        return response;
    }

    /**
     * Analyzes sentiment for multiple texts in batch.
     *
     * ✅ THREAD-SAFE: Uses parallel processing with order preservation
     * - Preprocessor is thread-safe (ReadWriteLock for concurrent reads)
     * - Classifier is thread-safe (ReadWriteLock via executeInference)
     * - Order is preserved using indexed parallel streams
     *
     * Example request:
     * POST /api/v1/sentiment/batch
     * {
     *   "texts": [
     *     "Great product!",
     *     "Terrible experience.",
     *     "It's okay, nothing special."
     *   ],
     *   "confidenceThreshold": 0.7
     * }
     */
    @PostMapping("/sentiment/batch")
    @RateLimiter(name = "batchApi")
    public ResponseEntity<BatchResponse> analyzeBatch(
            @Valid @RequestBody BatchRequest request) {

        long batchStartTime = System.currentTimeMillis();
        logger.info("Batch classification request: {} texts", request.getTexts().size());

        // ✅ PARALLEL PROCESSING: Process texts concurrently using parallel streams
        // Order preservation: Use indexed stream to maintain input order
        List<SentimentResponse> results = java.util.stream.IntStream.range(0, request.getTexts().size())
            .parallel()
            .mapToObj(i -> {
                String text = request.getTexts().get(i);
                ResponseEntity<SentimentResponse> result = classifyText(text, request.getConfidenceThreshold());
                return new IndexedResult(i, result.getBody());
            })
            .sorted(java.util.Comparator.comparingInt(IndexedResult::getIndex))
            .map(IndexedResult::getResult)
            .toList();

        long totalProcessingTime = System.currentTimeMillis() - batchStartTime;
        BatchResponse response = new BatchResponse(results);
        response.setTotalProcessingTimeMs(totalProcessingTime);

        logger.info("Batch completed: {} success, {} errors in {}ms (parallel processing)",
                   response.getSuccessCount(), response.getErrorCount(), totalProcessingTime);

        return ResponseEntity.ok(response);
    }

    /**
     * Helper class to preserve order during parallel processing.
     * Each result is paired with its original index for sorting.
     */
    private record IndexedResult(int index, SentimentResponse result) {
        int getIndex() { return index; }
        SentimentResponse getResult() { return result; }
    }

    /**
     * Health check endpoint.
     *
     * Example response:
     * {
     *   "status": "UP",
     *   "version": "1.0.0",
     *   "modelLoaded": true,
     *   "modelType": "SVM (SMO)",
     *   "uptimeMs": 45000
     * }
     */
    @GetMapping("/health")
    public ResponseEntity<HealthResponse> health() {
        long uptime = System.currentTimeMillis() - startTime;

        HealthResponse response = HealthResponse.healthy(
            "1.0.0",
            classifier.isTrained(),
            classifier.isTrained() ? classifier.getAlgorithmName() : "Not loaded",
            uptime
        );

        logger.debug("Health check: model loaded={}, uptime={}ms",
                    classifier.isTrained(), uptime);

        return ResponseEntity.ok(response);
    }

    /**
     * Internal method to classify a single text and apply confidence thresholding.
     * Extracted to avoid code duplication between single and batch endpoints.
     *
     * @param text The text to classify
     * @param confidenceThreshold Optional threshold for uncertain classification
     * @return ResponseEntity with classification result
     */
    private ResponseEntity<SentimentResponse> classifyText(String text, Double confidenceThreshold) {
        long startTime = System.currentTimeMillis();

        try {
            // Check if model is trained
            if (!classifier.isTrained()) {
                logger.error("Attempted classification with untrained model");
                return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                        .body(SentimentResponse.error(
                            "Model initialization in progress. Please retry in 30 seconds. " +
                            "Check /api/v1/health for status.", text));
            }

            // Get prediction and confidence
            String sentiment = classifier.classify(text);
            double[] probabilities = classifier.getClassificationProbabilities(text);

            // Find confidence for predicted class
            String[] classes = classifier.getSupportedClasses();
            double confidence = 0.0;
            for (int i = 0; i < classes.length; i++) {
                if (classes[i].equalsIgnoreCase(sentiment)) {
                    confidence = probabilities[i];
                    break;
                }
            }

            // Apply confidence threshold if provided
            if (confidenceThreshold != null && confidence < confidenceThreshold) {
                logger.debug("Confidence {:.3f} below threshold {:.3f}, marking uncertain",
                           confidence, confidenceThreshold);
                sentiment = "uncertain";
            }

            long processingTime = System.currentTimeMillis() - startTime;
            logger.info("Classification: '{}' (confidence: {:.3f}) in {}ms",
                       sentiment, confidence, processingTime);

            return ResponseEntity.ok(
                SentimentResponse.success(sentiment, confidence, text, processingTime)
            );

        } catch (Exception e) {
            logger.error("Classification failed for text: {}",
                        text.substring(0, Math.min(50, text.length())), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(SentimentResponse.error(e.getMessage(), text));
        }
    }

    /**
     * Extracts the client IP address from the HTTP request.
     * Checks common proxy headers (X-Forwarded-For, X-Real-IP) before falling back to remote address.
     *
     * @param request The HTTP servlet request
     * @return The client IP address
     */
    private String extractClientIp(HttpServletRequest request) {
        String ip = request.getHeader("X-Forwarded-For");
        if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getHeader("X-Real-IP");
        }
        if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getRemoteAddr();
        }
        // If X-Forwarded-For contains multiple IPs, take the first one (original client)
        if (ip != null && ip.contains(",")) {
            ip = ip.split(",")[0].trim();
        }
        return ip != null ? ip : "unknown";
    }
}
