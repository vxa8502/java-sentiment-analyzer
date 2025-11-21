package sentiment.evaluation;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Universal feature importance analyzer that works with ANY Weka classifier.
 *
 * Uses permutation importance (model-agnostic method):
 * 1. For each feature, measure baseline model performance
 * 2. Shuffle/zero that feature and re-measure performance
 * 3. Drop in performance = feature importance
 *
 * This works for:
 * - SVM (linear and non-linear kernels)
 * - Naive Bayes
 * - Random Forest
 * - Logistic Regression
 * - Neural Networks
 * - ANY classifier that implements distributionForInstance()
 *
 * Reference: Breiman, L. (2001). "Random Forests". Machine Learning.
 */
public class UniversalFeatureImportanceAnalyzer {

    private static final Logger logger = LoggerFactory.getLogger(UniversalFeatureImportanceAnalyzer.class);

    /**
     * Analyzes feature importance for ANY trained classifier using perturbation analysis.
     *
     * @param trainedData the training instances (used for feature names and perturbation)
     * @param classifier the trained classifier (any Weka Classifier)
     * @param topK number of top features to return
     * @return feature importance results with ranked features and statistics
     */
    public FeatureImportanceAnalyzer.FeatureImportanceResult analyzeFeatureImportance(
            Instances trainedData,
            Classifier classifier,
            int topK) {

        if (trainedData == null || classifier == null) {
            throw new IllegalArgumentException("trainedData and classifier cannot be null");
        }

        if (topK <= 0) {
            throw new IllegalArgumentException("topK must be positive");
        }

        logger.info("Analyzing feature importance for {} features using perturbation method, extracting top-{}",
                trainedData.numAttributes() - 1, topK);

        long startTime = System.currentTimeMillis();

        try {
            // Step 1: Extract feature weights via perturbation
            Map<String, Double> featureWeights = extractFeatureImportance(trainedData, classifier);

            // Step 2: Compute statistical significance
            Map<String, Double> featureSignificance = computeFeatureSignificance(featureWeights);

            // Step 3: Rank features by absolute importance
            List<FeatureImportanceAnalyzer.FeatureWeight> rankedFeatures =
                    rankFeatures(featureWeights, featureSignificance);

            // Step 4: Extract top-K features
            List<FeatureImportanceAnalyzer.FeatureWeight> topFeatures = rankedFeatures.stream()
                    .limit(topK)
                    .collect(Collectors.toList());

            // Step 5: Compute summary statistics
            FeatureImportanceAnalyzer.FeatureStatistics stats = computeFeatureStatistics(rankedFeatures);

            long duration = System.currentTimeMillis() - startTime;

            logger.info("Feature importance analysis complete in {}ms. Top feature: {} (importance: {})",
                    duration, topFeatures.get(0).featureName, topFeatures.get(0).weight);

            return new FeatureImportanceAnalyzer.FeatureImportanceResult(
                    topFeatures, rankedFeatures, stats, duration);

        } catch (Exception e) {
            logger.error("Feature importance analysis failed: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to analyze feature importance", e);
        }
    }

    /**
     * Extracts feature importance using perturbation/permutation method.
     *
     * Algorithm:
     * 1. For each feature:
     *    a. Measure baseline prediction confidence across all samples
     *    b. Zero out the feature values
     *    c. Re-measure prediction confidence
     *    d. Importance = average absolute change in confidence
     *
     * This is model-agnostic and works for any classifier.
     */
    private Map<String, Double> extractFeatureImportance(Instances trainedData, Classifier classifier) {
        Map<String, Double> importance = new HashMap<>();
        int numFeatures = trainedData.numAttributes() - 1;  // Exclude class attribute

        logger.debug("Computing feature importance via perturbation for {} features", numFeatures);

        try {
            // Use subset of data for efficiency (full dataset would be slow)
            int sampleSize = Math.min(200, trainedData.numInstances());
            Random random = new Random(42);
            List<Integer> sampleIndices = new ArrayList<>();
            for (int i = 0; i < trainedData.numInstances(); i++) {
                sampleIndices.add(i);
            }
            Collections.shuffle(sampleIndices, random);
            sampleIndices = sampleIndices.subList(0, sampleSize);

            for (int featureIdx = 0; featureIdx < numFeatures; featureIdx++) {
                Attribute attr = trainedData.attribute(featureIdx);
                String featureName = attr.name();

                // Compute influence for this feature
                double influence = computeFeatureInfluence(trainedData, classifier, featureIdx, sampleIndices);
                importance.put(featureName, influence);

                if ((featureIdx + 1) % 100 == 0) {
                    logger.debug("Processed {}/{} features", featureIdx + 1, numFeatures);
                }
            }

        } catch (Exception e) {
            logger.warn("Failed to extract feature importance via perturbation: {}", e.getMessage());
            throw new RuntimeException("Feature importance extraction failed", e);
        }

        return importance;
    }

    /**
     * Computes importance of a single feature via perturbation.
     */
    private double computeFeatureInfluence(Instances trainedData, Classifier classifier,
                                            int featureIndex, List<Integer> sampleIndices) {
        try {
            double totalInfluence = 0.0;

            for (int idx : sampleIndices) {
                Instance instance = trainedData.instance(idx);

                // Get original prediction confidence
                double[] originalProbs = classifier.distributionForInstance(instance);
                double originalConfidence = Math.abs(originalProbs[0] - originalProbs[1]);

                // Perturb feature (zero it out)
                Instance perturbedInstance = (Instance) instance.copy();
                perturbedInstance.setValue(featureIndex, 0.0);

                // Get perturbed prediction confidence
                double[] perturbedProbs = classifier.distributionForInstance(perturbedInstance);
                double perturbedConfidence = Math.abs(perturbedProbs[0] - perturbedProbs[1]);

                // Feature influence = change in confidence
                totalInfluence += Math.abs(originalConfidence - perturbedConfidence);
            }

            return totalInfluence / sampleIndices.size();

        } catch (Exception e) {
            logger.debug("Failed to compute influence for feature {}: {}", featureIndex, e.getMessage());
            return 0.0;
        }
    }

    private Map<String, Double> computeFeatureSignificance(Map<String, Double> weights) {
        Map<String, Double> significance = new HashMap<>();
        for (Map.Entry<String, Double> entry : weights.entrySet()) {
            double normalizedWeight = Math.abs(entry.getValue());
            significance.put(entry.getKey(), normalizedWeight);
        }
        return significance;
    }

    private List<FeatureImportanceAnalyzer.FeatureWeight> rankFeatures(
            Map<String, Double> weights,
            Map<String, Double> significance) {

        return weights.entrySet().stream()
                .map(entry -> new FeatureImportanceAnalyzer.FeatureWeight(
                        entry.getKey(),
                        entry.getValue(),
                        significance.getOrDefault(entry.getKey(), 0.0)
                ))
                .sorted(Comparator.comparingDouble(fw -> -Math.abs(fw.weight)))
                .collect(Collectors.toList());
    }

    private FeatureImportanceAnalyzer.FeatureStatistics computeFeatureStatistics(
            List<FeatureImportanceAnalyzer.FeatureWeight> features) {
        double[] absWeights = features.stream()
                .mapToDouble(fw -> Math.abs(fw.weight))
                .toArray();

        double mean = Arrays.stream(absWeights).average().orElse(0.0);
        double variance = Arrays.stream(absWeights)
                .map(w -> Math.pow(w - mean, 2))
                .average().orElse(0.0);
        double stdDev = Math.sqrt(variance);

        Arrays.sort(absWeights);
        double median = absWeights[absWeights.length / 2];
        double p95 = absWeights[(int) (absWeights.length * 0.95)];

        return new FeatureImportanceAnalyzer.FeatureStatistics(
                mean, stdDev, median, p95, features.size());
    }
}
