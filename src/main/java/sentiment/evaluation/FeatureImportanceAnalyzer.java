package sentiment.evaluation;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sentiment.evaluation.domain.FeatureImportanceResult;
import sentiment.evaluation.domain.FeatureStatistics;
import sentiment.evaluation.domain.FeatureWeight;
import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Analyzes feature importance for trained Weka classifiers.
 * Uses direct weight extraction for linear SVMs and perturbation analysis for other classifiers.
 * @see weka.classifiers.Classifier#distributionForInstance(weka.core.Instance)
 * @see sentiment.evaluation.domain.FeatureImportanceResult
 */
public class FeatureImportanceAnalyzer {

    private static final Logger logger = LoggerFactory.getLogger(FeatureImportanceAnalyzer.class);
    private static final int DEFAULT_SAMPLE_SIZE = 200;

    /**
     * Analyzes feature importance for any trained classifier.
     * Uses direct weight extraction for linear SVMs, perturbation analysis for others.
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

        // Detect if this is a linear SVM
        boolean isLinearSVM = isLinearSVM(classifier);
        String method = isLinearSVM ? "direct coefficient extraction" : "perturbation method";

        logger.info("Analyzing feature importance for {} features using {}, extracting top-{}",
                trainedData.numAttributes() - 1, method, topK);

        long startTime = System.currentTimeMillis();

        try {
            // Step 1: Extract feature weights (method depends on classifier type)
            Map<String, Double> featureWeights = isLinearSVM
                ? extractLinearSVMWeights(trainedData, (SMO) classifier)
                : extractFeatureImportance(trainedData, classifier);

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
     * Falls back to perturbation-based feature importance extraction.
     * This method is called when direct coefficient extraction fails for linear SVMs.
     *
     * @param reason the reason for falling back (logged as a warning)
     * @param trainedData the training instances
     * @param classifier the classifier to analyze
     * @return feature importance map from perturbation analysis
     */
    private Map<String, Double> fallbackToPerturbation(
            String reason, Instances trainedData, Classifier classifier) {
        logger.warn("{}", reason);
        logger.warn("Falling back to perturbation method");
        return extractFeatureImportance(trainedData, classifier);
    }

    /**
     * Detects if the classifier is a linear SVM (SMO with linear kernel or normalized poly kernel).
     *
     * @param classifier the classifier to check
     * @return true if it's a linear SVM, false otherwise
     */
    private boolean isLinearSVM(Classifier classifier) {
        if (!(classifier instanceof SMO)) {
            return false;
        }

        SMO smo = (SMO) classifier;

        // Check if using normalized poly kernel (default) with exponent 1.0 (linear)
        if (smo.getKernel() instanceof PolyKernel) {
            PolyKernel polyKernel = (PolyKernel) smo.getKernel();
            // Linear kernel is poly with exponent = 1.0
            // Use epsilon comparison to handle floating point precision
            double exponent = polyKernel.getExponent();
            boolean isLinear = Math.abs(exponent - 1.0) < 1e-9;
            logger.debug("PolyKernel exponent: {}, isLinear: {}", exponent, isLinear);
            return isLinear;
        }

        // RBF kernel is definitely non-linear
        if (smo.getKernel() instanceof RBFKernel) {
            return false;
        }

        // Other polynomial kernels with degree > 1 are non-linear
        return false;
    }

    /**
     * Extracts feature weights directly from a linear SVM's coefficient vector.
     * This is much more accurate and efficient than perturbation for linear models.
     *
     * <p>For linear SVM with Weka's SMO, the weight vector is stored directly in sparse format:
     * <ul>
     *   <li>sparseIndices[classifier][class] = array of non-zero feature indices</li>
     *   <li>sparseWeights[classifier][class] = array of corresponding weight values</li>
     * </ul>
     *
     * <p>For binary classification, we use class index 0 which gives us the weights
     * for the decision boundary. Positive weights favor the positive class.
     *
     * @param trainedData the training instances (for feature names)
     * @param smo the trained linear SVM classifier
     * @return a map of feature names to their coefficient weights
     * @throws RuntimeException if weight extraction fails
     */
    private Map<String, Double> extractLinearSVMWeights(Instances trainedData, SMO smo) {
        Map<String, Double> weights = new HashMap<>();
        int classIndex = trainedData.classIndex();
        int numAttributes = trainedData.numAttributes();

        logger.info("Extracting linear SVM coefficients for {} features", numAttributes - 1);
        logger.info("SMO kernel type: {}", smo.getKernel().getClass().getSimpleName());

        try {
            // Get sparse representation from SMO
            // Structure: sparseIndices[classifier][class] = array of feature indices
            //            sparseWeights[classifier][class] = array of weight values
            int[][][] sparseIndices = smo.sparseIndices();
            double[][][] sparseWeights = smo.sparseWeights();

            if (sparseIndices == null || sparseWeights == null) {
                return fallbackToPerturbation(
                        "sparseIndices() or sparseWeights() returned null", trainedData, smo);
            }

            if (sparseIndices.length == 0 || sparseWeights.length == 0) {
                return fallbackToPerturbation(
                        "Empty sparse arrays - no classifiers found", trainedData, smo);
            }

            logger.info("Found {} classifier(s) in SMO", sparseIndices.length);

            // Initialize all features to 0 (excluding class attribute)
            for (int i = 0; i < numAttributes; i++) {
                if (i != classIndex) {
                    weights.put(trainedData.attribute(i).name(), 0.0);
                }
            }

            // For binary classification, use the first classifier (index 0)
            // and iterate over available class weight vectors
            int[][] indicesForClassifier = sparseIndices[0];
            double[][] weightsForClassifier = sparseWeights[0];

            if (indicesForClassifier == null || weightsForClassifier == null) {
                return fallbackToPerturbation(
                        "No sparse data for classifier 0", trainedData, smo);
            }

            logger.info("Classifier 0 has {} class weight vectors", indicesForClassifier.length);

            int totalWeightsExtracted = 0;

            // Process weight vectors - find the first non-null class weight vector
            // In Weka's binary SVM, one class may have null weights while the other has the data
            int[] featureIndices = null;
            double[] featureWeights = null;
            int usedClassIdx = -1;

            for (int classIdx = 0; classIdx < indicesForClassifier.length; classIdx++) {
                if (indicesForClassifier[classIdx] != null && weightsForClassifier[classIdx] != null) {
                    featureIndices = indicesForClassifier[classIdx];
                    featureWeights = weightsForClassifier[classIdx];
                    usedClassIdx = classIdx;
                    logger.info("Using class {} weights ({} non-zero entries)", classIdx, featureIndices.length);
                    break;
                } else {
                    logger.debug("Class {} has null sparse data, skipping", classIdx);
                }
            }

            if (featureIndices == null || featureWeights == null) {
                return fallbackToPerturbation(
                        "No valid weight vectors found in any class", trainedData, smo);
            }

            // Extract weights from the sparse representation
            for (int i = 0; i < featureIndices.length && i < featureWeights.length; i++) {
                int featureIdx = featureIndices[i];

                // Skip class attribute and validate index
                if (featureIdx == classIndex) continue;
                if (featureIdx < 0 || featureIdx >= numAttributes) {
                    logger.debug("Skipping invalid feature index: {}", featureIdx);
                    continue;
                }

                String featureName = trainedData.attribute(featureIdx).name();
                double weight = featureWeights[i];

                weights.put(featureName, weight);
                totalWeightsExtracted++;
            }

            logger.info("Extracted {} feature weights from sparse representation", totalWeightsExtracted);

            // Verify we got non-zero weights
            long nonZeroCount = weights.values().stream().filter(w -> Math.abs(w) > 1e-10).count();
            logger.info("Non-zero feature weights: {} out of {}", nonZeroCount, numAttributes - 1);

            if (nonZeroCount == 0) {
                return fallbackToPerturbation(
                        "All extracted weights are zero! This indicates a problem with coefficient extraction.",
                        trainedData, smo);
            }

            return weights;

        } catch (Exception e) {
            logger.error("Failed to extract linear SVM weights: {}", e.getMessage(), e);
            return fallbackToPerturbation(
                    "Exception during coefficient extraction", trainedData, smo);
        }
    }

    /**
     * Extracts feature importance using the perturbation method.
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
     * <p>
     * Confidence is measured as the maximum probability (works for binary and multi-class).
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

                // Get original prediction confidence (max probability across all classes)
                double[] originalProbs = classifier.distributionForInstance(instance);
                double originalConfidence = getMaxProbability(originalProbs);

                // Perturb feature (zero it out)
                Instance perturbedInstance = (Instance) instance.copy();
                perturbedInstance.setValue(featureIndex, 0.0);

                // Get perturbed prediction confidence
                double[] perturbedProbs = classifier.distributionForInstance(perturbedInstance);
                double perturbedConfidence = getMaxProbability(perturbedProbs);

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
     * Returns the maximum probability from a distribution.
     * Works for both binary and multi-class classification.
     *
     * @param probabilities probability distribution over classes
     * @return the maximum probability value
     */
    private double getMaxProbability(double[] probabilities) {
        double max = probabilities[0];
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > max) {
                max = probabilities[i];
            }
        }
        return max;
    }

    /**
     * Computes statistical significance scores for features based on their weights.
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
