package sentiment.api;

import io.github.resilience4j.ratelimiter.annotation.RateLimiter;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.beans.factory.annotation.Value;
import sentiment.evaluation.FeatureImportancePersistence;
import sentiment.models.SVMClassifier;
import sentiment.models.SentimentClassifier;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * REST API controller for sentiment analysis operations.
 * Endpoints:
 * - POST /api/v1/sentiment/analyze - Analyze single text
 * - POST /api/v1/sentiment/batch - Batch analysis
 * - GET /api/v1/model/feature-importance - Get feature importance (SVM only)
 * - GET /api/v1/health - Health check
 */
@RestController
@RequestMapping("/api/v1")
@org.springframework.context.annotation.Profile("!training")
public class SentimentController {

    private static final Logger logger = LoggerFactory.getLogger(SentimentController.class);
    private final SentimentClassifier classifier;
    private final String svmModelPath;
    private final long startTime;

    // Cache feature importance to avoid reloading on every request
    private FeatureImportanceResponse cachedFeatureImportance;
    private final Object featureImportanceLock = new Object();

    public SentimentController(SentimentClassifier classifier,
                               @Value("${sentiment.models.svm-model-path:./models/svm-model.ser}") String svmModelPath) {
        this.classifier = classifier;
        this.svmModelPath = svmModelPath;
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
        logger.info("REQUEST: client={}, textLength={}", clientIp, request.text().length());

        ResponseEntity<SentimentResponse> response = classifyText(
            request.text(), request.confidenceThreshold());

        SentimentResponse body = response.getBody();
        if (body != null) {
            logger.info("RESPONSE: client={}, sentiment={}, time={}ms",
                        clientIp, body.sentiment(), body.processingTimeMs());
        }

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
        logger.info("Batch classification request: {} texts", request.texts().size());

        // ✅ PARALLEL PROCESSING: Process texts concurrently using parallel streams
        // Order preservation: Use indexed stream to maintain input order
        List<SentimentResponse> results = java.util.stream.IntStream.range(0, request.texts().size())
            .parallel()
            .mapToObj(i -> {
                String text = request.texts().get(i);
                ResponseEntity<SentimentResponse> result = classifyText(text, request.confidenceThreshold());
                return new IndexedResult(i, result.getBody());
            })
            .sorted(java.util.Comparator.comparingInt(IndexedResult::getIndex))
            .map(IndexedResult::getResult)
            .toList();

        long totalProcessingTime = System.currentTimeMillis() - batchStartTime;
        BatchResponse response = BatchResponse.fromResults(results, totalProcessingTime);

        logger.info("Batch completed: {} success, {} errors in {}ms (parallel processing)",
                   response.successCount(), response.errorCount(), totalProcessingTime);

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
     * Returns feature importance analysis for the trained model.
     *
     * This endpoint is designed for notebook-based exploration and model understanding.
     * It shows which features (words/n-grams) most strongly influence predictions.
     *
     * IMPORTANT:
     * - Available for all model types (SVM, Naive Bayes, Random Forest, Logistic Regression)
     * - Results are cached after first request for performance
     * - Use topFeatures parameter to control response size
     *
     * Example request:
     * GET /api/v1/model/feature-importance?topFeatures=50
     *
     * Example response:
     * {
     *   "modelType": "SVM",
     *   "totalFeatures": 5000,
     *   "topFeatures": [
     *     {"feature": "excellent", "weight": 0.234, "significance": 0.234, "direction": "positive"},
     *     {"feature": "terrible", "weight": -0.198, "significance": 0.198, "direction": "negative"}
     *   ],
     *   "statistics": {"mean": 0.002, "stdDev": 0.004, "median": 0.001, "percentile95": 0.009},
     *   "analysisTimeMs": 234
     * }
     */
    @GetMapping("/model/feature-importance")
    public ResponseEntity<FeatureImportanceResponse> getFeatureImportance(
            @RequestParam(defaultValue = "30") int topFeatures) {

        logger.info("Feature importance request: topFeatures={}", topFeatures);

        try {
            // Check if model is trained
            if (!classifier.isTrained()) {
                return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                        .body(FeatureImportanceResponse.error(
                                "Model not trained yet. Check /api/v1/health for status."));
            }

            // Note: Feature importance now works for all model types, not just SVM

            // Use cached result if available
            synchronized (featureImportanceLock) {
                if (cachedFeatureImportance != null &&
                    cachedFeatureImportance.topFeatures().size() >= topFeatures) {
                    logger.info("Returning cached feature importance");
                    // Return subset if requested fewer features
                    List<FeatureImportanceResponse.FeatureInfo> subset =
                            cachedFeatureImportance.topFeatures().subList(0,
                                    Math.min(topFeatures, cachedFeatureImportance.topFeatures().size()));

                    return ResponseEntity.ok(new FeatureImportanceResponse(
                            cachedFeatureImportance.modelType(),
                            cachedFeatureImportance.totalFeatures(),
                            subset,
                            cachedFeatureImportance.statistics(),
                            cachedFeatureImportance.analysisTimeMs(),
                            cachedFeatureImportance.note()
                    ));
                }

                // Load pre-computed feature importance from file
                logger.info("Loading pre-computed feature importance from file...");
                Path featureImportancePath = FeatureImportancePersistence.getFeatureImportancePath(
                        Paths.get(svmModelPath));

                if (!Files.exists(featureImportancePath)) {
                    return ResponseEntity.status(HttpStatus.NOT_FOUND)
                            .body(FeatureImportanceResponse.error(
                                    "Feature importance data not found. Please re-train the model with " +
                                    "--show-feature-importance flag to generate feature importance data. " +
                                    "Expected file: " + featureImportancePath));
                }

                try {
                    sentiment.evaluation.domain.FeatureImportanceResult data =
                            FeatureImportancePersistence.load(featureImportancePath);

                    // Convert to response format
                    List<FeatureImportanceResponse.FeatureInfo> allFeatures = data.topFeatures().stream()
                            .map(FeatureImportanceResponse.FeatureInfo::fromDomain)
                            .collect(Collectors.toList());

                    FeatureImportanceResponse.Statistics stats =
                            FeatureImportanceResponse.Statistics.fromDomain(data.statistics());

                    // Cache the full result
                    cachedFeatureImportance = FeatureImportanceResponse.success(
                            classifier.getAlgorithmName(),
                            data.statistics().totalFeatures(),
                            allFeatures,
                            stats,
                            data.analysisTimeMs()
                    );

                    // Return requested subset
                    List<FeatureImportanceResponse.FeatureInfo> subset = allFeatures.stream()
                            .limit(topFeatures)
                            .collect(Collectors.toList());

                    return ResponseEntity.ok(new FeatureImportanceResponse(
                            classifier.getAlgorithmName(),
                            data.statistics().totalFeatures(),
                            subset,
                            stats,
                            data.analysisTimeMs(),
                            cachedFeatureImportance.note()
                    ));

                } catch (Exception e) {
                    logger.error("Failed to load feature importance: {}", e.getMessage(), e);
                    return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                            .body(FeatureImportanceResponse.error(
                                    "Failed to load feature importance data: " + e.getMessage()));
                }
            }

        } catch (Exception e) {
            logger.error("Feature importance analysis failed: {}", e.getMessage(), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(FeatureImportanceResponse.error(
                            "Feature importance analysis failed: " + e.getMessage()));
        }
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
                logger.debug("Confidence {} below threshold {}, marking uncertain",
                           confidence, confidenceThreshold);
                sentiment = "uncertain";
            }

            long processingTime = System.currentTimeMillis() - startTime;
            logger.info("Classification: '{}' (confidence: {}) in {}ms",
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
