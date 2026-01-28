package sentiment.api;

import io.github.resilience4j.ratelimiter.annotation.RateLimiter;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.beans.factory.annotation.Qualifier;
import sentiment.api.metrics.PredictionMetrics;
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
    private final SentimentService sentimentService;
    private final String loadedModelPath;
    private final long startTime;

    private FeatureImportanceResponse cachedFeatureImportance;
    private final Object featureImportanceLock = new Object();

    public SentimentController(SentimentService sentimentService,
                               @Qualifier("loadedModelPath") String loadedModelPath) {
        this.sentimentService = sentimentService;
        this.loadedModelPath = loadedModelPath;
        this.startTime = System.currentTimeMillis();
        SentimentClassifier classifier = sentimentService.getClassifier();
        String algorithmName = classifier != null ? classifier.getAlgorithmName() : "unknown";
        logger.info("SentimentController initialized with {} classifier (model: {}) and production metrics",
                   algorithmName, loadedModelPath);
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

        ResponseEntity<SentimentResponse> response = sentimentService.classifyText(
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
                ResponseEntity<SentimentResponse> result = sentimentService.classifyText(text, request.confidenceThreshold());
                return new IndexedResult(i, result.getBody());
            })
            .sorted(java.util.Comparator.comparingInt(IndexedResult::index))
            .map(IndexedResult::result)
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
    private record IndexedResult(int index, SentimentResponse result) {}

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

        // Validate topFeatures parameter
        if (topFeatures < 1) {
            return ResponseEntity.badRequest()
                    .body(FeatureImportanceResponse.error(
                            "Invalid topFeatures value: must be at least 1"));
        }

        SentimentClassifier classifier = sentimentService.getClassifier();

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
                    return ResponseEntity.ok(cachedFeatureImportance.withTopFeatures(topFeatures));
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

                            logger.info("Extracted {} features with {} non-zero weights in {}ms",
                                    weights.size(), nonZero, duration);

                            return ResponseEntity.ok(cachedFeatureImportance.withTopFeatures(topFeatures));
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

                    return ResponseEntity.ok(cachedFeatureImportance.withTopFeatures(topFeatures));

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

        SentimentClassifier classifier = sentimentService.getClassifier();
        PredictionMetrics metrics = sentimentService.getMetrics();

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
                        snapshot.uncertainPredictions()
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

}
