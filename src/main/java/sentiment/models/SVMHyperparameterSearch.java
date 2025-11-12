package sentiment.models;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.core.Instances;
import sentiment.data.Dataset;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Hyperparameter search for SVM using stratified k-fold cross-validation.
 *
 * METHODOLOGY:
 * ============
 * 1. Grid Search: Tests all combinations of hyperparameters
 * 2. Stratified K-Fold CV: Preserves class distribution in each fold
 * 3. Macro-F1 as primary metric: Handles class imbalance better than accuracy
 * 4. Class weighting: Adjusts SVM cost function to prioritize minority classes
 *
 * HYPERPARAMETER GRID:
 * ===================
 * - C (Regularization): [0.01, 0.1, 1.0, 10.0, 100.0]
 * - Kernel: [Linear, Polynomial, RBF]
 * - Gamma (RBF only): [0.001, 0.01, 0.1, 1.0]
 * - Degree (Poly only): [2, 3, 4]
 *
 * USAGE:
 * ======
 * SVMHyperparameterSearch search = new SVMHyperparameterSearch(5);  // 5-fold CV
 * SVMConfig bestConfig = search.findOptimalConfig(trainData, true);  // Enable class weighting
 */
public class SVMHyperparameterSearch {

    private static final Logger logger = LoggerFactory.getLogger(SVMHyperparameterSearch.class);

    private final int numFolds;
    private final Random random;

    // Default hyperparameter grid
    private static final double[] C_VALUES = {0.01, 0.1, 1.0, 10.0, 100.0};
    private static final SVMConfig.KernelType[] KERNEL_TYPES = {
            SVMConfig.KernelType.LINEAR,
            SVMConfig.KernelType.POLYNOMIAL,
            SVMConfig.KernelType.RBF
    };
    private static final double[] GAMMA_VALUES = {0.001, 0.01, 0.1, 1.0};  // RBF only
    private static final int[] DEGREE_VALUES = {2, 3, 4};  // Polynomial only

    /**
     * Creates a hyperparameter search with specified number of CV folds.
     *
     * @param numFolds Number of folds for cross-validation (typically 5 or 10)
     */
    public SVMHyperparameterSearch(int numFolds) {
        if (numFolds < 2) {
            throw new IllegalArgumentException("numFolds must be >= 2");
        }
        this.numFolds = numFolds;
        this.random = new Random(42);  // Fixed seed for reproducibility
        logger.info("Created hyperparameter search with {}-fold CV", numFolds);
    }

    /**
     * Finds the optimal SVM configuration using grid search with stratified k-fold CV.
     *
     * ALGORITHM:
     * 1. Generate hyperparameter grid
     * 2. For each configuration:
     *    a. Perform stratified k-fold CV
     *    b. Calculate mean macro-F1 and std dev across folds
     * 3. Select configuration with highest mean macro-F1
     * 4. Return best configuration with CV performance stats
     *
     * @param trainingData Training instances (already preprocessed and vectorized)
     * @param enableClassWeighting Whether to apply class weighting for imbalanced data
     * @return Best SVMConfig with CV performance metrics
     */
    public SVMConfig findOptimalConfig(Instances trainingData, boolean enableClassWeighting) throws Exception {
        if (trainingData == null || trainingData.numInstances() < numFolds) {
            throw new IllegalArgumentException(
                    String.format("Need at least %d instances for %d-fold CV", numFolds, numFolds));
        }

        logger.info("Starting hyperparameter search on {} instances", trainingData.numInstances());
        logger.info("Class weighting: {}", enableClassWeighting ? "ENABLED" : "DISABLED");

        // Step 1: Calculate class weights if needed
        double[] classWeights = null;
        if (enableClassWeighting) {
            classWeights = calculateClassWeights(trainingData);
            logger.info("Class weights: {}", Arrays.toString(classWeights));
        }

        // Step 2: Generate hyperparameter grid
        List<SVMConfig> configGrid = generateConfigGrid(classWeights);
        logger.info("Testing {} configurations", configGrid.size());

        // Step 3: Evaluate each configuration with stratified k-fold CV
        List<SVMConfig> evaluatedConfigs = new ArrayList<>();
        int configNum = 0;

        for (SVMConfig config : configGrid) {
            configNum++;
            logger.info("Evaluating config {}/{}: {}", configNum, configGrid.size(), config);

            try {
                evaluateConfigWithCV(config, trainingData);
                evaluatedConfigs.add(config);
                logger.info("  -> Macro-F1: {} (±{}), Accuracy: {}",
                        String.format("%.4f", config.getCvMacroF1()),
                        String.format("%.4f", config.getCvStdDev()),
                        String.format("%.4f", config.getCvAccuracy()));
            } catch (Exception e) {
                logger.warn("  -> Failed to evaluate config: {}", e.getMessage());
            }
        }

        if (evaluatedConfigs.isEmpty()) {
            throw new Exception("No configurations completed successfully");
        }

        // Step 4: Select best configuration based on macro-F1
        SVMConfig bestConfig = evaluatedConfigs.stream()
                .max(Comparator.comparingDouble(c -> c.getCvMacroF1()))
                .orElseThrow(() -> new Exception("Failed to find best configuration"));

        logger.info("Best configuration found: {}", bestConfig);
        logger.info("  -> Macro-F1: {} (±{})",
                String.format("%.4f", bestConfig.getCvMacroF1()),
                String.format("%.4f", bestConfig.getCvStdDev()));
        logger.info("  -> Accuracy: {}", String.format("%.4f", bestConfig.getCvAccuracy()));

        return bestConfig;
    }

    /**
     * Calculates class weights for imbalanced datasets.
     *
     * Formula: weight_i = n_samples / (n_classes * n_samples_i)
     *
     * This ensures minority classes have higher weight in the SVM loss function.
     */
    private double[] calculateClassWeights(Instances data) {
        int numClasses = data.numClasses();
        int[] classCounts = new int[numClasses];

        // Count instances per class
        for (int i = 0; i < data.numInstances(); i++) {
            int classIndex = (int) data.instance(i).classValue();
            classCounts[classIndex]++;
        }

        // Calculate weights: n_total / (n_classes * n_class_i)
        double[] weights = new double[numClasses];
        int totalInstances = data.numInstances();

        for (int i = 0; i < numClasses; i++) {
            if (classCounts[i] > 0) {
                weights[i] = (double) totalInstances / (numClasses * classCounts[i]);
            } else {
                weights[i] = 1.0;  // Default weight for classes with no instances
            }
        }

        return weights;
    }

    /**
     * Generates the hyperparameter grid to search.
     *
     * STRATEGY:
     * - Start with linear kernel (fastest, often works well for text)
     * - Test polynomial kernels with different degrees
     * - Test RBF kernels with different gamma values
     * - For each kernel, test different C values
     */
    private List<SVMConfig> generateConfigGrid(double[] classWeights) {
        List<SVMConfig> grid = new ArrayList<>();

        for (SVMConfig.KernelType kernelType : KERNEL_TYPES) {
            for (double c : C_VALUES) {
                switch (kernelType) {
                    case LINEAR:
                        // Linear kernel: only C matters
                        grid.add(new SVMConfig(c, kernelType, 0.01, 1, classWeights, 1.0E-12));
                        break;

                    case POLYNOMIAL:
                        // Polynomial: test different degrees
                        for (int degree : DEGREE_VALUES) {
                            grid.add(new SVMConfig(c, kernelType, 0.01, degree, classWeights, 1.0E-12));
                        }
                        break;

                    case RBF:
                        // RBF: test different gamma values
                        for (double gamma : GAMMA_VALUES) {
                            grid.add(new SVMConfig(c, kernelType, gamma, 1, classWeights, 1.0E-12));
                        }
                        break;
                }
            }
        }

        return grid;
    }

    /**
     * Evaluates a single configuration using stratified k-fold cross-validation.
     *
     * STRATIFICATION:
     * - Preserves class distribution in each fold
     * - Critical for imbalanced datasets
     * - Ensures each fold has representative samples from all classes
     *
     * @param config Configuration to evaluate
     * @param data Training data
     */
    private void evaluateConfigWithCV(SVMConfig config, Instances data) throws Exception {
        // Stratify the data first
        Instances stratifiedData = new Instances(data);
        stratifiedData.randomize(random);
        stratifiedData.stratify(numFolds);

        double[] macroF1Scores = new double[numFolds];
        double[] accuracyScores = new double[numFolds];

        // Perform k-fold cross-validation
        for (int fold = 0; fold < numFolds; fold++) {
            Instances trainFold = stratifiedData.trainCV(numFolds, fold, random);
            Instances testFold = stratifiedData.testCV(numFolds, fold);

            // Train SVM with current configuration
            SMO smo = createSMOFromConfig(config);
            smo.buildClassifier(trainFold);

            // Evaluate on test fold
            Evaluation evaluation = new Evaluation(trainFold);
            evaluation.evaluateModel(smo, testFold);

            // Calculate macro-averaged F1 (average across all classes)
            double macroF1 = calculateMacroF1(evaluation, data.numClasses());
            double accuracy = evaluation.pctCorrect() / 100.0;

            macroF1Scores[fold] = macroF1;
            accuracyScores[fold] = accuracy;
        }

        // Calculate mean and standard deviation
        double meanMacroF1 = Arrays.stream(macroF1Scores).average().orElse(0.0);
        double meanAccuracy = Arrays.stream(accuracyScores).average().orElse(0.0);
        double stdDevMacroF1 = calculateStdDev(macroF1Scores, meanMacroF1);

        // Store results in config
        config.setCvMacroF1(meanMacroF1);
        config.setCvAccuracy(meanAccuracy);
        config.setCvStdDev(stdDevMacroF1);
    }

    /**
     * Creates a configured SMO classifier from an SVMConfig.
     */
    private SMO createSMOFromConfig(SVMConfig config) throws Exception {
        SMO smo = new SMO();

        // Set kernel
        Kernel kernel = config.createKernel();
        smo.setKernel(kernel);

        // Set options (C, epsilon, etc.)
        String options = config.toOptionsString();
        smo.setOptions(weka.core.Utils.splitOptions(options));

        return smo;
    }

    /**
     * Calculates macro-averaged F1 score.
     *
     * Macro-F1 = average of per-class F1 scores
     * This gives equal weight to all classes, making it better for imbalanced data.
     */
    private double calculateMacroF1(Evaluation evaluation, int numClasses) throws Exception {
        double sumF1 = 0.0;
        int validClasses = 0;

        for (int i = 0; i < numClasses; i++) {
            double f1 = evaluation.fMeasure(i);
            if (!Double.isNaN(f1)) {
                sumF1 += f1;
                validClasses++;
            }
        }

        return validClasses > 0 ? sumF1 / validClasses : 0.0;
    }

    /**
     * Calculates standard deviation of scores.
     */
    private double calculateStdDev(double[] scores, double mean) {
        double sumSquaredDiff = 0.0;
        for (double score : scores) {
            double diff = score - mean;
            sumSquaredDiff += diff * diff;
        }
        return Math.sqrt(sumSquaredDiff / scores.length);
    }

    /**
     * Quick search for fast experimentation (reduced grid).
     *
     * Tests only:
     * - C: [0.1, 1.0, 10.0]
     * - Kernels: [Linear, RBF]
     * - Gamma: [0.01, 0.1] (RBF only)
     */
    public SVMConfig findOptimalConfigQuick(Instances trainingData, boolean enableClassWeighting) throws Exception {
        logger.info("Running QUICK hyperparameter search (reduced grid)");

        double[] classWeights = enableClassWeighting ? calculateClassWeights(trainingData) : null;

        List<SVMConfig> quickGrid = new ArrayList<>();

        // Linear kernel with 3 C values
        for (double c : new double[]{0.1, 1.0, 10.0}) {
            quickGrid.add(new SVMConfig(c, SVMConfig.KernelType.LINEAR, 0.01, 1, classWeights, 1.0E-12));
        }

        // RBF kernel with 2 gamma values and 3 C values
        for (double c : new double[]{0.1, 1.0, 10.0}) {
            for (double gamma : new double[]{0.01, 0.1}) {
                quickGrid.add(new SVMConfig(c, SVMConfig.KernelType.RBF, gamma, 1, classWeights, 1.0E-12));
            }
        }

        logger.info("Testing {} configurations (quick mode)", quickGrid.size());

        // Evaluate and select best
        List<SVMConfig> evaluatedConfigs = new ArrayList<>();
        for (SVMConfig config : quickGrid) {
            try {
                evaluateConfigWithCV(config, trainingData);
                evaluatedConfigs.add(config);
            } catch (Exception e) {
                logger.warn("Failed to evaluate config: {}", e.getMessage());
            }
        }

        return evaluatedConfigs.stream()
                .max(Comparator.comparingDouble(c -> c.getCvMacroF1()))
                .orElseThrow(() -> new Exception("No valid configurations found"));
    }
}
