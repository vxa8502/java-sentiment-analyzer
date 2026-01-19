package sentiment.api;

import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.ratelimiter.annotation.RateLimiter;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.beans.factory.annotation.Value;
import sentiment.api.metrics.PredictionMetrics;
import sentiment.evaluation.FeatureImportancePersistence;
import sentiment.models.SentimentClassifier;

import java.util.List;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
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
    private final PredictionMetrics metrics;
    private final String loadedModelPath;
    private final long startTime;

    private FeatureImportanceResponse cachedFeatureImportance;
    private final Object featureImportanceLock = new Object();

    public SentimentController(SentimentClassifier classifier,
                               PredictionMetrics metrics,
                               String loadedModelPath) {
        this.classifier = classifier;
        this.metrics = metrics;
        this.loadedModelPath = loadedModelPath;
        this.startTime = System.currentTimeMillis();
        logger.info("SentimentController initialized with {} classifier (model: {}) and production metrics",
                   classifier.getAlgorithmName(), loadedModelPath);
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
     * Returns feature importance analysis.
     * For SVM models, extracts coefficients directly from the trained model.
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

                long startTime = System.currentTimeMillis();

                // For SVM models, extract weights directly from the model
                if (classifier instanceof sentiment.models.SVMClassifier svmClassifier) {
                    logger.info("Extracting feature importance directly from SVM coefficients...");
                    java.util.Map<String, Double> weights = svmClassifier.extractFeatureWeights();

                    if (!weights.isEmpty()) {
                        long nonZero = weights.values().stream()
                                .filter(w -> Math.abs(w) > 1e-10).count();

                        if (nonZero > 0) {
                            // Sort by absolute weight
                            List<FeatureImportanceResponse.FeatureInfo> allFeatures = weights.entrySet().stream()
                                    .sorted((a, b) -> Double.compare(Math.abs(b.getValue()), Math.abs(a.getValue())))
                                    .map(e -> new FeatureImportanceResponse.FeatureInfo(
                                            e.getKey(),
                                            e.getValue(),
                                            Math.abs(e.getValue()),
                                            e.getValue() > 0 ? "positive" : (e.getValue() < 0 ? "negative" : "neutral")
                                    ))
                                    .collect(Collectors.toList());

                            // Compute statistics
                            double[] absWeights = weights.values().stream()
                                    .mapToDouble(Math::abs).toArray();
                            double mean = java.util.Arrays.stream(absWeights).average().orElse(0);
                            double variance = java.util.Arrays.stream(absWeights)
                                    .map(w -> (w - mean) * (w - mean))
                                    .average().orElse(0);
                            double stdDev = Math.sqrt(variance);

                            java.util.Arrays.sort(absWeights);
                            double median = absWeights.length % 2 == 0
                                    ? (absWeights[absWeights.length/2 - 1] + absWeights[absWeights.length/2]) / 2
                                    : absWeights[absWeights.length/2];
                            double p95 = absWeights[(int)(0.95 * (absWeights.length - 1))];

                            FeatureImportanceResponse.Statistics stats = new FeatureImportanceResponse.Statistics(
                                    mean, stdDev, median, p95);

                            long duration = System.currentTimeMillis() - startTime;

                            cachedFeatureImportance = new FeatureImportanceResponse(
                                    classifier.getAlgorithmName(),
                                    weights.size(),
                                    allFeatures,
                                    stats,
                                    duration,
                                    "Feature importance extracted directly from SVM coefficients. " +
                                    "Positive weights indicate positive sentiment, negative weights indicate negative sentiment."
                            );

                            List<FeatureImportanceResponse.FeatureInfo> subset = allFeatures.stream()
                                    .limit(topFeatures)
                                    .collect(Collectors.toList());

                            logger.info("Extracted {} features with {} non-zero weights in {}ms",
                                    weights.size(), nonZero, duration);

                            return ResponseEntity.ok(new FeatureImportanceResponse(
                                    classifier.getAlgorithmName(),
                                    weights.size(),
                                    subset,
                                    stats,
                                    duration,
                                    cachedFeatureImportance.note()
                            ));
                        }
                    }
                    logger.warn("SVM coefficient extraction returned no non-zero weights, falling back to file");
                }

                // Fall back to loading from pre-computed file
                logger.info("Loading pre-computed feature importance from file...");
                Path featureImportancePath = FeatureImportancePersistence.getFeatureImportancePath(
                        Paths.get(loadedModelPath));

                if (!Files.exists(featureImportancePath)) {
                    return ResponseEntity.status(HttpStatus.NOT_FOUND)
                            .body(FeatureImportanceResponse.error(
                                    "Feature importance data not found and could not be extracted from model. " +
                                    "Please re-train the model with --show-feature-importance flag. " +
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
     * Health check endpoint returning service status, model information, and production metrics.
     */
    @GetMapping("/health")
    public ResponseEntity<HealthResponse> health() {
        long uptime = System.currentTimeMillis() - startTime;

        // Get production metrics snapshot
        PredictionMetrics.MetricsSnapshot snapshot = metrics.getSnapshot();

        // Get supported labels from classifier (may be subset of all possible labels)
        List<String> supportedLabels = classifier.isTrained()
                ? List.of(classifier.getSupportedClasses())
                : List.of();

        HealthResponse.ProductionMetrics productionMetrics = new HealthResponse.ProductionMetrics(
                snapshot.totalPredictions(),
                new HealthResponse.LabelDistribution(
                        snapshot.positivePredictions(),
                        snapshot.negativePredictions(),
                        snapshot.neutralPredictions()
                ),
                snapshot.averageConfidence(),
                snapshot.lowConfidenceRatePercent(),
                new HealthResponse.LatencyStats(
                        snapshot.meanLatencyMs(),
                        snapshot.p95LatencyMs(),
                        snapshot.p99LatencyMs()
                )
        );

        HealthResponse response = HealthResponse.withMetrics(
            "1.0.0",
            classifier.isTrained(),
            classifier.isTrained() ? classifier.getAlgorithmName() : "Not loaded",
            supportedLabels,
            uptime,
            productionMetrics
        );

        logger.debug("Health check: model loaded={}, uptime={}ms, predictions={}",
                    classifier.isTrained(), uptime, snapshot.totalPredictions());

        return ResponseEntity.ok(response);
    }

    /**
     * Classifies text and applies optional confidence thresholding.
     * Protected by circuit breaker to prevent cascading failures when model fails.
     */
    @CircuitBreaker(name = "modelInference", fallbackMethod = "classifyTextFallback")
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

            // Use atomic classification to avoid race conditions in concurrent batch processing
            SentimentClassifier.ClassificationResult result = classifier.classifyWithProbabilities(text);
            String sentiment = result.label();
            double confidence = result.confidence();

            String warning = null;

            if (confidenceThreshold != null && confidence < confidenceThreshold) {
                logger.debug("Confidence {} below threshold {}, marking uncertain",
                           confidence, confidenceThreshold);
                sentiment = "uncertain";
            }

            // Warn if confidence is low - may indicate neutral text misclassified
            // Production model is binary (positive/negative only)
            if (confidence < 0.75) {
                warning = "Low confidence prediction. This model only supports 'positive' and 'negative' labels. " +
                         "Neutral or ambiguous text may be misclassified. Check /api/v1/health for supportedLabels.";
            }

            long processingTime = System.currentTimeMillis() - startTime;
            logger.debug("Classification: '{}' (confidence: {}) in {}ms",
                       sentiment, confidence, processingTime);

            // Record production metrics
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

        } catch (Exception e) {
            logger.error("Classification failed for text: {}",
                        sanitizeForLogging(text), e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(SentimentResponse.error(e.getMessage(), text));
        }
    }

    /**
     * Fallback method for classifyText when circuit breaker is OPEN.
     * Returns a graceful error response instead of cascading failures.
     */
    private ResponseEntity<SentimentResponse> classifyTextFallback(String text, Double confidenceThreshold, Exception e) {
        logger.warn("Circuit breaker OPEN - model inference unavailable. Cause: {}", e.getMessage());

        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                .body(SentimentResponse.error(
                    "Sentiment analysis temporarily unavailable. The model is experiencing issues " +
                    "and is recovering. Please retry in 30 seconds. Check /actuator/health for " +
                    "circuit breaker status.", text));
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
