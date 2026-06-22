package sentiment.api;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Cacheable;
import org.springframework.stereotype.Service;
import sentiment.config.CacheConfiguration;
import sentiment.evaluation.FeatureImportancePersistence;
import sentiment.models.SentimentClassifier;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Service for extracting and caching feature importance data.
 * <p>
 * Uses Spring's {@code @Cacheable} to automatically cache results,
 * replacing manual synchronization and caching logic.
 */
@Service
public class FeatureImportanceService {

    private static final Logger logger = LoggerFactory.getLogger(FeatureImportanceService.class);

    private final SentimentService sentimentService;
    private final String modelPath;

    public FeatureImportanceService(
            SentimentService sentimentService,
            @org.springframework.beans.factory.annotation.Qualifier("loadedModelPath") String modelPath) {
        this.sentimentService = sentimentService;
        this.modelPath = modelPath;
    }

    /**
     * Retrieves feature importance data, caching the result.
     * <p>
     * The cache key is based on the model path, so reloading a different
     * model will result in a cache miss and fresh computation.
     *
     * @return Optional containing the feature importance response, or empty if unavailable
     */
    @Cacheable(value = CacheConfiguration.FEATURE_IMPORTANCE_CACHE, key = "#root.target.modelPath")
    public Optional<FeatureImportanceResponse> getFeatureImportance() {
        logger.info("Computing feature importance (cache miss)...");

        SentimentClassifier classifier = sentimentService.getClassifier();
        if (classifier == null || !classifier.isTrained()) {
            logger.warn("Classifier not available or not trained");
            return Optional.empty();
        }

        long startTime = System.currentTimeMillis();

        // Try SVM coefficient extraction first
        Optional<FeatureImportanceResponse> svmResult = extractFromSVM(classifier, startTime);
        if (svmResult.isPresent()) {
            return svmResult;
        }

        // Fall back to pre-computed file
        return loadFromFile(classifier, startTime);
    }

    /**
     * Extracts feature importance directly from SVM coefficients.
     *
     * @param classifier the trained classifier
     * @param startTime  computation start time for duration tracking
     * @return Optional containing the response, or empty if extraction failed
     */
    private Optional<FeatureImportanceResponse> extractFromSVM(SentimentClassifier classifier, long startTime) {
        if (!(classifier instanceof sentiment.models.SVMClassifier svmClassifier)) {
            return Optional.empty();
        }

        logger.info("Extracting feature importance from SVM coefficients...");
        Map<String, Double> weights = svmClassifier.extractFeatureWeights();

        if (weights.isEmpty()) {
            logger.warn("SVM coefficient extraction returned empty weights");
            return Optional.empty();
        }

        long nonZero = weights.values().stream()
                .filter(w -> Math.abs(w) > 1e-10)
                .count();

        if (nonZero == 0) {
            logger.warn("SVM coefficient extraction returned no non-zero weights");
            return Optional.empty();
        }

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
        FeatureImportanceResponse.Statistics stats = computeStatistics(weights);
        long duration = System.currentTimeMillis() - startTime;

        logger.info("Extracted {} features with {} non-zero weights in {}ms",
                weights.size(), nonZero, duration);

        return Optional.of(new FeatureImportanceResponse(
                classifier.getAlgorithmName(),
                weights.size(),
                allFeatures,
                stats,
                duration,
                "Feature importance extracted directly from SVM coefficients. " +
                "Positive weights indicate positive sentiment, negative weights indicate negative sentiment.",
                null  // no error
        ));
    }

    /**
     * Loads feature importance from pre-computed file.
     *
     * @param classifier the trained classifier
     * @param startTime  computation start time for duration tracking
     * @return Optional containing the response, or empty if loading failed
     */
    private Optional<FeatureImportanceResponse> loadFromFile(SentimentClassifier classifier, long startTime) {
        logger.info("Loading pre-computed feature importance from file...");

        Path featureImportancePath = FeatureImportancePersistence.getFeatureImportancePath(
                Paths.get(modelPath));

        if (!Files.exists(featureImportancePath)) {
            logger.warn("Feature importance file not found: {}", featureImportancePath);
            return Optional.empty();
        }

        try {
            sentiment.evaluation.domain.FeatureImportanceResult data =
                    FeatureImportancePersistence.load(featureImportancePath);

            List<FeatureImportanceResponse.FeatureInfo> allFeatures = data.topFeatures().stream()
                    .map(FeatureImportanceResponse.FeatureInfo::fromDomain)
                    .collect(Collectors.toList());

            FeatureImportanceResponse.Statistics stats =
                    FeatureImportanceResponse.Statistics.fromDomain(data.statistics());

            long duration = System.currentTimeMillis() - startTime;
            logger.info("Loaded feature importance from file in {}ms", duration);

            return Optional.of(FeatureImportanceResponse.success(
                    classifier.getAlgorithmName(),
                    data.statistics().totalFeatures(),
                    allFeatures,
                    stats,
                    data.analysisTimeMs()
            ));

        } catch (Exception e) {
            // Log full exception for debugging, but return a meaningful error response
            // rather than silently returning empty
            logger.error("Failed to load feature importance from {}: {}", featureImportancePath, e.getMessage(), e);

            // Return an error response so the caller knows why it failed
            // This is better than Optional.empty() which gives no context
            long duration = System.currentTimeMillis() - startTime;
            return Optional.of(FeatureImportanceResponse.error(
                    String.format("Failed to load feature importance: %s", e.getMessage()),
                    duration
            ));
        }
    }

    /**
     * Computes statistics from feature weights.
     */
    private FeatureImportanceResponse.Statistics computeStatistics(Map<String, Double> weights) {
        double[] absWeights = weights.values().stream()
                .mapToDouble(Math::abs)
                .toArray();

        double mean = Arrays.stream(absWeights).average().orElse(0);
        double variance = Arrays.stream(absWeights)
                .map(w -> (w - mean) * (w - mean))
                .average().orElse(0);
        double stdDev = Math.sqrt(variance);

        Arrays.sort(absWeights);
        double median = absWeights.length % 2 == 0
                ? (absWeights[absWeights.length/2 - 1] + absWeights[absWeights.length/2]) / 2
                : absWeights[absWeights.length/2];
        double p95 = absWeights[(int)(0.95 * (absWeights.length - 1))];

        return new FeatureImportanceResponse.Statistics(mean, stdDev, median, p95);
    }

    /**
     * Clears the feature importance cache.
     * Call this after model reload.
     */
    @CacheEvict(value = CacheConfiguration.FEATURE_IMPORTANCE_CACHE, allEntries = true)
    public void clearCache() {
        logger.info("Feature importance cache cleared");
    }

    /**
     * Returns the model path used as cache key.
     */
    public String getModelPath() {
        return modelPath;
    }
}
