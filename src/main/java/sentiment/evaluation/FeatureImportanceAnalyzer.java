package sentiment.evaluation;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sentiment.evaluation.domain.FeatureImportanceResult;
import sentiment.evaluation.domain.FeatureStatistics;
import sentiment.evaluation.domain.FeatureWeight;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Analyzes feature importance for any trained Weka classifier using permutation importance.
 * <p>
 * This analyzer uses permutation importance, a model-agnostic method that measures
 * the impact of each feature by perturbing its values and observing the change in
 * model predictions. The algorithm:
 * <ol>
 * <li>Measures baseline model performance for each feature</li>
 * <li>Zeros out the feature values and re-measures performance</li>
 * <li>Calculates importance as the drop in prediction confidence</li>
 * </ol>
 * <p>
 * This approach is compatible with all Weka classifiers that implement
 * {@code distributionForInstance()}, including SVM, Naive Bayes, Random Forest,
 * Logistic Regression, and Neural Networks.
 *
 * @see weka.classifiers.Classifier#distributionForInstance(weka.core.Instance)
 * @see sentiment.evaluation.domain.FeatureImportanceResult
 */
public class FeatureImportanceAnalyzer {

    private static final Logger logger = LoggerFactory.getLogger(FeatureImportanceAnalyzer.class);
    private static final int DEFAULT_SAMPLE_SIZE = 200;

    /**
     * Analyzes feature importance for any trained classifier using perturbation analysis.
     *
     * @param trainedData the training instances used for feature names and perturbation
     * @param classifier the trained classifier to analyze
     * @param topK the number of top features to return (must be positive)
     * @return feature importance results containing ranked features and statistics
     * @throws IllegalArgumentException if trainedData or classifier is null, or if topK is not positive
     * @throws RuntimeException if feature importance analysis fails
     */
    public FeatureImportanceResult analyzeFeatureImportance(
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
            List<FeatureWeight> rankedFeatures = rankFeatures(featureWeights, featureSignificance);

            // Step 4: Extract top-K features
            List<FeatureWeight> topFeatures = rankedFeatures.stream()
                    .limit(topK)
                    .collect(Collectors.toList());

            // Step 5: Compute summary statistics
            FeatureStatistics stats = computeFeatureStatistics(rankedFeatures);

            long duration = System.currentTimeMillis() - startTime;

            logger.info("Feature importance analysis complete in {}ms. Top feature: {} (importance: {})",
                    duration, topFeatures.get(0).featureName(), topFeatures.get(0).weight());

            return new FeatureImportanceResult(topFeatures, rankedFeatures, stats, duration);

        } catch (Exception e) {
            logger.error("Feature importance analysis failed: {}", e.getMessage(), e);
            throw new RuntimeException("Failed to analyze feature importance", e);
        }
    }

    /**
     * Extracts feature importance using the perturbation method.
     * <p>
     * For each feature, this method:
     * <ol>
     * <li>Measures baseline prediction confidence across a sample of instances</li>
     * <li>Zeros out the feature values</li>
     * <li>Re-measures prediction confidence</li>
     * <li>Computes importance as the average absolute change in confidence</li>
     * </ol>
     * <p>
     * To improve efficiency, this method uses a random sample of up to 200 instances
     * from the training data rather than the entire dataset.
     *
     * @param trainedData the training instances to analyze
     * @param classifier the trained classifier to evaluate
     * @return a map of feature names to their computed importance scores
     * @throws RuntimeException if feature importance extraction fails
     */
    private Map<String, Double> extractFeatureImportance(Instances trainedData, Classifier classifier) {
        Map<String, Double> importance = new HashMap<>();
        int numFeatures = trainedData.numAttributes() - 1;  // Exclude class attribute

        logger.debug("Computing feature importance via perturbation for {} features", numFeatures);

        try {
            // Use subset of data for efficiency (full dataset would be slow)
            int sampleSize = Math.min(DEFAULT_SAMPLE_SIZE, trainedData.numInstances());
            Random random = new Random(42);

            // Efficiently generate random sample indices without creating full list
            Set<Integer> sampleIndicesSet = new HashSet<>();
            int totalInstances = trainedData.numInstances();
            while (sampleIndicesSet.size() < sampleSize) {
                sampleIndicesSet.add(random.nextInt(totalInstances));
            }
            List<Integer> sampleIndices = new ArrayList<>(sampleIndicesSet);

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
     * Computes the importance of a single feature via perturbation.
     * <p>
     * For each instance in the sample, this method zeros out the specified feature,
     * measures the change in prediction confidence, and returns the average change
     * across all sampled instances.
     *
     * @param trainedData the training instances
     * @param classifier the trained classifier
     * @param featureIndex the index of the feature to evaluate
     * @param sampleIndices the indices of instances to use for evaluation
     * @return the average influence score for the feature (0.0 if computation fails)
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

    /**
     * Computes statistical significance scores for features based on their weights.
     * <p>
     * Currently returns the absolute value of each weight as the significance score.
     *
     * @param weights a map of feature names to their importance weights
     * @return a map of feature names to their significance scores
     */
    private Map<String, Double> computeFeatureSignificance(Map<String, Double> weights) {
        Map<String, Double> significance = new HashMap<>();
        for (Map.Entry<String, Double> entry : weights.entrySet()) {
            double normalizedWeight = Math.abs(entry.getValue());
            significance.put(entry.getKey(), normalizedWeight);
        }
        return significance;
    }

    /**
     * Ranks features by their absolute importance weights in descending order.
     *
     * @param weights a map of feature names to their importance weights
     * @param significance a map of feature names to their significance scores
     * @return a sorted list of feature weights, ordered by absolute importance (highest first)
     */
    private List<FeatureWeight> rankFeatures(
            Map<String, Double> weights,
            Map<String, Double> significance) {

        return weights.entrySet().stream()
                .map(entry -> new FeatureWeight(
                        entry.getKey(),
                        entry.getValue(),
                        significance.getOrDefault(entry.getKey(), 0.0)
                ))
                .sorted(Comparator.comparingDouble(fw -> -Math.abs(fw.weight())))
                .collect(Collectors.toList());
    }

    /**
     * Computes summary statistics for a list of ranked features.
     * <p>
     * Calculates mean, standard deviation, median, and 95th percentile of
     * the absolute importance weights.
     *
     * @param features the list of ranked features
     * @return feature statistics including mean, standard deviation, median, 95th percentile, and total count
     */
    private FeatureStatistics computeFeatureStatistics(List<FeatureWeight> features) {
        double[] absWeights = features.stream()
                .mapToDouble(fw -> Math.abs(fw.weight()))
                .toArray();

        // Compute mean and variance in single pass to avoid redundant streaming
        double sum = 0.0;
        double sumSq = 0.0;
        for (double w : absWeights) {
            sum += w;
            sumSq += w * w;
        }
        double mean = sum / absWeights.length;
        double variance = (sumSq / absWeights.length) - (mean * mean);
        double stdDev = Math.sqrt(variance);

        // Sort for median and percentiles
        Arrays.sort(absWeights);

        // Correct median calculation (average of two middle values for even-length arrays)
        double median;
        int n = absWeights.length;
        if (n % 2 == 0) {
            median = (absWeights[n / 2 - 1] + absWeights[n / 2]) / 2.0;
        } else {
            median = absWeights[n / 2];
        }

        // Correct 95th percentile with linear interpolation
        double p95Index = 0.95 * (n - 1);
        int lowerIndex = (int) Math.floor(p95Index);
        int upperIndex = (int) Math.ceil(p95Index);
        double p95;
        if (lowerIndex == upperIndex) {
            p95 = absWeights[lowerIndex];
        } else {
            double fraction = p95Index - lowerIndex;
            p95 = absWeights[lowerIndex] + fraction * (absWeights[upperIndex] - absWeights[lowerIndex]);
        }

        return new FeatureStatistics(mean, stdDev, median, p95, features.size());
    }
}
