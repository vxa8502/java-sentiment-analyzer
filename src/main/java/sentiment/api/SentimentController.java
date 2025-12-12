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
import sentiment.models.SentimentClassifier;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Collectors;

/**
 * REST API controller for sentiment analysis operations.
 */
@RestController
@RequestMapping("/api/v1")
@org.springframework.context.annotation.Profile("!training")
public class SentimentController {

    private static final Logger logger = LoggerFactory.getLogger(SentimentController.class);
    private final SentimentClassifier classifier;
    private final String svmModelPath;
    private final long startTime;

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
     * Analyzes sentiment for multiple texts in batch using parallel processing.
     * Results maintain input order despite concurrent execution.
     */
    @PostMapping("/sentiment/batch")
    @RateLimiter(name = "batchApi")
    public ResponseEntity<BatchResponse> analyzeBatch(
            @Valid @RequestBody BatchRequest request) {

        long batchStartTime = System.currentTimeMillis();
        logger.info("Batch classification request: {} texts", request.texts().size());

        // Process texts concurrently using indexed parallel streams for order preservation
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
     * Pairs result with original index for order-preserving parallel processing.
     */
    private record IndexedResult(int index, SentimentResponse result) {
        int getIndex() { return index; }
        SentimentResponse getResult() { return result; }
    }

    /**
     * Returns feature importance analysis from pre-computed data.
     * Results are cached after first request.
     *
     * @param topFeatures Number of top features to return (default: 30)
     */
    @GetMapping("/model/feature-importance")
    public ResponseEntity<FeatureImportanceResponse> getFeatureImportance(
            @RequestParam(defaultValue = "30") int topFeatures) {

        logger.info("Feature importance request: topFeatures={}", topFeatures);

        try {
            if (!classifier.isTrained()) {
                return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                        .body(FeatureImportanceResponse.error(
                                "Model not trained yet. Check /api/v1/health for status."));
            }

            // Use cached result if available
            synchronized (featureImportanceLock) {
                if (cachedFeatureImportance != null &&
                    cachedFeatureImportance.topFeatures().size() >= topFeatures) {
                    logger.info("Returning cached feature importance");
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

                    List<FeatureImportanceResponse.FeatureInfo> allFeatures = data.topFeatures().stream()
                            .map(FeatureImportanceResponse.FeatureInfo::fromDomain)
                            .collect(Collectors.toList());

                    FeatureImportanceResponse.Statistics stats =
                            FeatureImportanceResponse.Statistics.fromDomain(data.statistics());

                    cachedFeatureImportance = FeatureImportanceResponse.success(
                            classifier.getAlgorithmName(),
                            data.statistics().totalFeatures(),
                            allFeatures,
                            stats,
                            data.analysisTimeMs()
                    );

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
     * Health check endpoint returning service status and model information.
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
     * Classifies text and applies optional confidence thresholding.
     */
    private ResponseEntity<SentimentResponse> classifyText(String text, Double confidenceThreshold) {
        long startTime = System.currentTimeMillis();

        try {
            if (!classifier.isTrained()) {
                logger.error("Attempted classification with untrained model");
                return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                        .body(SentimentResponse.error(
                            "Model initialization in progress. Please retry in 30 seconds. " +
                            "Check /api/v1/health for status.", text));
            }

            String sentiment = classifier.classify(text);
            double[] probabilities = classifier.getClassificationProbabilities(text);

            String[] classes = classifier.getSupportedClasses();
            double confidence = 0.0;
            for (int i = 0; i < classes.length; i++) {
                if (classes[i].equalsIgnoreCase(sentiment)) {
                    confidence = probabilities[i];
                    break;
                }
            }

            if (confidenceThreshold != null && confidence < confidenceThreshold) {
                logger.debug("Confidence {} below threshold {}, marking uncertain",
                           confidence, confidenceThreshold);
                sentiment = "uncertain";
            }

            long processingTime = System.currentTimeMillis() - startTime;
            logger.debug("Classification: '{}' (confidence: {}) in {}ms",
                       sentiment, confidence, processingTime);

            return ResponseEntity.ok(
                SentimentResponse.success(sentiment, confidence, text, processingTime)
            );

        } catch (Exception e) {
            logger.error("Classification failed for text: {}",
                        sanitizeForLogging(text), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(SentimentResponse.error(e.getMessage(), text));
        }
    }

    /**
     * Extracts client IP from request, checking proxy headers first.
     */
    private String extractClientIp(HttpServletRequest request) {
        String ip = request.getHeader("X-Forwarded-For");
        if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getHeader("X-Real-IP");
        }
        if (ip == null || ip.isEmpty() || "unknown".equalsIgnoreCase(ip)) {
            ip = request.getRemoteAddr();
        }
        if (ip != null && ip.contains(",")) {
            ip = ip.split(",")[0].trim();
        }
        return ip != null ? ip : "unknown";
    }

    /**
     * Sanitizes text for logging to prevent log injection attacks.
     * Removes newlines, carriage returns, and non-printable characters.
     *
     * @param text the text to sanitize
     * @return sanitized text truncated to 50 characters
     */
    private String sanitizeForLogging(String text) {
        if (text == null) {
            return "null";
        }

        String truncated = text.substring(0, Math.min(50, text.length()));
        // Remove newlines and carriage returns to prevent log injection
        // Replace non-printable characters with '?'
        return truncated
                .replaceAll("[\n\r]", " ")
                .replaceAll("[^\\x20-\\x7E]", "?");
    }
}
