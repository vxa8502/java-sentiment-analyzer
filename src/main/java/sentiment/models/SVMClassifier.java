package sentiment.models;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Instance;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import sentiment.evaluation.ClassifierEvaluationResult;
import sentiment.data.Dataset;
import sentiment.util.ValidationUtils;

import java.util.*;

/**
 * SVM classifier for sentiment analysis using Weka's SMO implementation.
 *
 * <p>Supports optional hyperparameter tuning via cross-validation to optimize
 * kernel type, complexity parameter (C), and kernel-specific parameters.
 */
public class SVMClassifier extends ClassifierTrainingTemplate<ClassifierEvaluationResult>
        implements ClassifierEvaluator {

    private static final Logger logger = LoggerFactory.getLogger(SVMClassifier.class);

    private SMO smo;
    private final TextPreprocessor preprocessor;

    private boolean enableHyperparameterTuning = false;
    private int cvFolds = 5;
    private SVMConfig optimalConfig;
    private double classImbalanceThreshold = 3.0;

    /**
     * Creates an SVM classifier with default configuration.
     */
    public SVMClassifier(TextPreprocessor preprocessor, WekaInstancesConverter converter) {
        ValidationUtils.requireAllNonNull(
                new Object[]{preprocessor, converter},
                new String[]{"preprocessor", "converter"});

        this.preprocessor = preprocessor;
        this.converter = converter;
        this.smo = new SMO();
        // Enable Platt scaling for calibrated probability outputs
        // Without this, SMO returns hard 0/1 confidence values
        this.smo.setBuildCalibrationModels(true);

        logger.info("Created SVMClassifier - manages full training pipeline");
    }

    /**
     * Creates classifier with custom SMO configuration.
     */
    public SVMClassifier(TextPreprocessor preprocessor, WekaInstancesConverter converter, SMO customSMO) {
        ValidationUtils.requireAllNonNull(
                new Object[]{preprocessor, converter, customSMO},
                new String[]{"preprocessor", "converter", "customSMO"});

        this.preprocessor = preprocessor;
        this.converter = converter;
        this.smo = customSMO;

        logger.info("Created SVMClassifier with custom SMO");
    }

    /**
     * Enables hyperparameter tuning via grid search with k-fold cross-validation.
     * Training time increases 5-10x but may improve accuracy by 2-5%.
     * Enable for final production models, disable for experimentation.
     * Configurable via {@code SENTIMENT_SVM_TUNE_ENABLED} env var.
     *
     * @param enable whether to enable tuning
     * @param numFolds number of CV folds (typically 5 or 10)
     */
    public void setHyperparameterTuning(boolean enable, int numFolds) {
        this.enableHyperparameterTuning = enable;
        this.cvFolds = numFolds;
        logger.info("Hyperparameter tuning: {} ({}-fold CV)", enable ? "ENABLED" : "DISABLED", numFolds);
    }

    /**
     * Sets the class imbalance threshold for automatic class weighting.
     * When max_class_count / min_class_count exceeds threshold, class weighting is enabled.
     * Configurable via {@code SENTIMENT_SVM_CLASS_IMBALANCE_THRESHOLD} env var (default: 3.0).
     *
     * @param threshold imbalance ratio (max/min class counts) that triggers weighting (must be > 1.0)
     * @throws IllegalArgumentException if threshold <= 1.0
     */
    public void setClassImbalanceThreshold(double threshold) {
        if (threshold <= 1.0) {
            throw new IllegalArgumentException("Class imbalance threshold must be greater than 1.0");
        }
        this.classImbalanceThreshold = threshold;
        logger.debug("Class imbalance threshold set to: {}", threshold);
    }

    @Override
    public AlgorithmType getAlgorithmType() {
        return AlgorithmType.SVM;
    }

    @Override
    protected Logger getLogger() {
        return logger;
    }

    @Override
    protected weka.classifiers.Classifier getWekaClassifierInstance() {
        return smo;
    }

    @Override
    protected void setWekaClassifierInstance(weka.classifiers.Classifier classifier) {
        this.smo = (weka.classifiers.functions.SMO) classifier;
    }


    /**
     * Trains the classifier on raw datasets, fitting the complete preprocessing and feature extraction pipeline.
     */
    @Override
    protected ClassifierEvaluationResult doTrain(List<Dataset> rawDatasets) throws Exception {
        if (rawDatasets == null || rawDatasets.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }

        logger.info("Training SVM on {} raw datasets with full pipeline", rawDatasets.size());

        // Step 1: Fit vectorization pipeline (preprocessing + TF-IDF)
        // Note: converter.fit() internally calls preprocessor.fit()
        logger.info("Step 1/2: Fitting vectorization pipeline");
        Instances trainingInstances = converter.fit(rawDatasets);
        logger.info("Vectorization complete. Features: {}, Vocabulary: {}",
                trainingInstances.numAttributes() - 1,
                converter.getVocabulary().size());

        // Step 2: Train SVM on converted Instances
        logger.info("Step 2/2: Training SVM classifier");
        validateWekaTrainingData(trainingInstances);
        configureSMOForTraining(trainingInstances);
        performAlgorithmSpecificTraining(trainingInstances);
        finalizeTraining(trainingInstances);

        // Validate pipeline consistency
        validatePipelineConsistency();

        logger.info("SVM training complete. Pipeline ready for inference.");

        // Return null - no evaluation during training
        return null;
    }

    /**
     * Validates training data and checks class balance.
     */
    @Override
    protected void validateWekaTrainingData(Instances data) {
        super.validateWekaTrainingData(data); // Common validation
        checkClassBalance(data); // SVM-specific validation
    }

    /**
     * Validates class distribution and detects class imbalance.
     */
    private void checkClassBalance(Instances data) {
        int[] classCounts = new int[data.classAttribute().numValues()];
        for (int i = 0; i < data.numInstances(); i++) {
            classCounts[(int) data.instance(i).classValue()]++;
        }

        // Count only non-zero classes (e.g., binary classification may not have neutral)
        long nonZeroClasses = Arrays.stream(classCounts).filter(count -> count > 0).count();

        if (nonZeroClasses < 2) {
            throw new IllegalArgumentException(
                    "Invalid class distribution: need at least 2 classes with instances. " +
                            "Got class counts: " + Arrays.toString(classCounts));
        }

        // Calculate imbalance ratio only among classes that actually exist
        int minNonZeroCount = Arrays.stream(classCounts).filter(count -> count > 0).min().orElse(0);
        int maxNonZeroCount = Arrays.stream(classCounts).filter(count -> count > 0).max().orElse(0);

        if (maxNonZeroCount > 0 && minNonZeroCount > 0) {
            double imbalanceRatio = (double) maxNonZeroCount / minNonZeroCount;
            if (imbalanceRatio > 10.0) {
                String ratio = String.format("%.1f", imbalanceRatio);
                logger.warn("High class imbalance detected (ratio: {}:1)", ratio);
            }
        }
    }

    /**
     * Validates and logs preprocessing pipeline statistics.
     */
    private void validatePipelineConsistency() {
        Set<String> preprocessorVocabSet = preprocessor.getPipelineState().vocabularyFrequencies.keySet();
        Set<String> converterVocabSet = converter.getVocabulary();
        int numFeatures = trainingDataStructure.numAttributes() - 1;

        // Note: Converter vocabulary may include bigrams not in preprocessor unigram vocabulary
        // This is expected behavior when bigram features are enabled

        // Log pipeline statistics
        int selectedFeatures = converterVocabSet.size();
        int totalVocab = preprocessorVocabSet.size();
        double selectionRatio = (double) selectedFeatures / totalVocab;
        String selectionPct = String.format("%.1f%%", selectionRatio * 100);

        logger.info("Pipeline statistics:");
        logger.info("  - Preprocessor vocabulary: {} terms", totalVocab);
        logger.info("  - Converter vocabulary:    {} terms", selectedFeatures);
        logger.info("  - Feature selection ratio: {}", selectionPct);
        logger.info("  - Feature count:           {} features", numFeatures);
        logger.info("  - Training instances:      {}", trainingDataStructure.numInstances());
    }

    private void configureSMOForTraining(Instances trainingData) throws Exception {
        if (enableHyperparameterTuning) {
            logger.warn("********************************************************************************");
            logger.warn("*                             ATTENTION: SLOW TRAINING                           *");
            logger.warn("* Hyperparameter tuning is ENABLED. This will significantly increase training    *");
            logger.warn("* time (potentially 8-10 hours on large datasets). If this is not intended,      *");
            logger.warn("* disable it by setting the 'enableHyperparameterTuning' argument to 'false'     *");
            logger.warn("* when running TrainModel, or by ensuring the SENTIMENT_SVM_TUNE_ENABLED env     *");
            logger.warn("* variable is not set to 'true'.                                                 *");
            logger.warn("********************************************************************************");
            logger.info("=== HYPERPARAMETER TUNING ENABLED ===");
            logger.info("Performing grid search with {}-fold stratified cross-validation", cvFolds);

            // Detect class imbalance and enable class weighting if needed
            boolean useClassWeighting = shouldUseClassWeighting(trainingData);

            // Perform grid search
            optimalConfig = performGridSearch(trainingData, useClassWeighting);

            // Apply optimal configuration
            applySVMConfig(optimalConfig);

            logger.info("=== OPTIMAL CONFIGURATION SELECTED ===");
            logger.info("Config: {}", optimalConfig);
            logger.info("Expected Performance: Macro-F1={} (±{}), Accuracy={}",
                    String.format("%.4f", optimalConfig.getCvMacroF1()),
                    String.format("%.4f", optimalConfig.getCvStdDev()),
                    String.format("%.4f", optimalConfig.getCvAccuracy()));

        } else {
            logger.info("Hyperparameter tuning DISABLED - using Linear kernel");
            logger.warn("RECOMMENDATION: Enable hyperparameter tuning for production models");

            // Default: Linear kernel with C=0.1
            // Linear kernels work best for high-dimensional sparse text data (TF-IDF)
            // and allow direct coefficient extraction for feature importance
            optimalConfig = new SVMConfig(0.1, SVMConfig.KernelType.LINEAR, 0.01, 1, null, 1.0E-12);
            applySVMConfig(optimalConfig);

            logger.info("Using default config: C=0.1, Linear Kernel (enables feature importance)");
        }
    }

    /**
     * Checks if class weighting should be applied based on imbalance ratio.
     */
    private boolean shouldUseClassWeighting(Instances data) {
        int[] classCounts = new int[data.numClasses()];
        for (int i = 0; i < data.numInstances(); i++) {
            classCounts[(int) data.instance(i).classValue()]++;
        }

        int minCount = Arrays.stream(classCounts).filter(c -> c > 0).min().orElse(1);
        int maxCount = Arrays.stream(classCounts).filter(c -> c > 0).max().orElse(1);

        double imbalanceRatio = (double) maxCount / minCount;
        boolean shouldWeight = imbalanceRatio > classImbalanceThreshold;

        if (shouldWeight) {
            logger.info("Class imbalance detected (ratio: {}:1, threshold: {}) - enabling class weighting",
                    String.format("%.1f", imbalanceRatio), classImbalanceThreshold);
        }

        return shouldWeight;
    }

    private void applySVMConfig(SVMConfig config) throws Exception {
        // Set kernel
        smo.setKernel(config.createKernel());

        // Set options (C, epsilon, etc.)
        String options = config.toOptionsString();
        smo.setOptions(weka.core.Utils.splitOptions(options));

        // Ensure calibration models are enabled for probability outputs
        smo.setBuildCalibrationModels(true);

        // Log configuration (handle null kernel for mocked SMO)
        Kernel kernel = smo.getKernel();
        String kernelName = (kernel != null) ? kernel.getClass().getSimpleName() : "null";
        logger.debug("SMO configured: C={}, Epsilon={}, Kernel={}, Calibration={}",
                smo.getC(), smo.getEpsilon(), kernelName, smo.getBuildCalibrationModels());
    }

    @Override
    protected void performAlgorithmSpecificTraining(Instances trainingData) throws Exception {
        logger.info("Training SVM model on {} instances", trainingData.numInstances());
        smo.buildClassifier(trainingData);
        logger.info("SVM model training complete");
    }

    /**
     * Performs evaluation with advanced metrics including ROC-AUC, PR-AUC, and calibration.
     */
    @Override
    protected ClassifierEvaluationResult performEvaluation(Instances testData)
            throws Exception {
        long startTime = System.currentTimeMillis();

        Evaluation evaluation = new Evaluation(trainingDataStructure);

        //  NEW: Collect probabilities during evaluation
        int n = testData.numInstances();
        int numClasses = supportedClasses.length;
        double[][] probabilities = new double[n][numClasses];
        int[] actualLabels = new int[n];

        for (int i = 0; i < n; i++) {
            Instance instance = testData.instance(i);
            // Record prediction in Evaluation object
            evaluation.evaluateModelOnceAndRecordPrediction(smo, instance);
            // Extract probabilities for advanced metrics (AUC, calibration)
            probabilities[i] = smo.distributionForInstance(instance);
            actualLabels[i] = (int) instance.classValue();
        }

        long evaluationTime = System.currentTimeMillis() - startTime;

        // Now compute advanced metrics using the collected data
        ClassifierEvaluationResult result = buildEvaluationResult(
                evaluation, testData, evaluationTime,
                probabilities, actualLabels);

        String accuracy = String.format("%.3f", result.getAccuracy());
        logger.info("Evaluation complete in {}ms: accuracy={} (single-pass optimized)",
                evaluationTime, accuracy);

        return result;
    }

    /**
     * Builds evaluation result including calibration metrics (AUC metrics now in base class).
     */
    private ClassifierEvaluationResult buildEvaluationResult(
            Evaluation evaluation, Instances testData,
            long evaluationTimeMs,
            double[][] probabilities, int[] actualLabels) {

        // Get base evaluation result (includes ROC-AUC and PR-AUC from base class)
        ClassifierEvaluationResult baseResult = super.buildEvaluationResult(
                evaluation, testData, evaluationTimeMs);

        // Compute calibration metrics
        sentiment.evaluation.CalibrationMetrics calibrationMetrics = null;
        int numClasses = supportedClasses.length;

        try {
            if (numClasses == 2) {
                // Binary classification: use positive class probabilities
                double[] positiveClassProbs = new double[probabilities.length];
                for (int i = 0; i < probabilities.length; i++) {
                    positiveClassProbs[i] = probabilities[i][1];
                }
                calibrationMetrics = sentiment.evaluation.CalibrationMetrics.compute(
                        positiveClassProbs, actualLabels, 10);
            } else {
                // Multi-class: compute averaged calibration
                calibrationMetrics = sentiment.evaluation.CalibrationMetrics.computeMultiClass(
                        probabilities, actualLabels, 10);
            }

            logger.debug("Advanced metrics computed: ROC-AUC={}, PR-AUC={}, Brier={}",
                    String.format("%.4f", baseResult.getMacroAvgROCAUC()),
                    String.format("%.4f", baseResult.getMacroAvgPRAUC()),
                    String.format("%.4f", calibrationMetrics.getBrierScore()));

        } catch (Exception e) {
            logger.warn("Failed to compute calibration metrics: {}", e.getMessage());
        }

        // Return result with calibration metrics added
        return new ClassifierEvaluationResult(
                baseResult.getAlgorithmName(),
                baseResult.getAccuracy(),
                baseResult.getPrecision(),
                baseResult.getRecall(),
                baseResult.getF1Score(),
                baseResult.getMacroAvgPrecision(),
                baseResult.getMacroAvgRecall(),
                baseResult.getMacroAvgF1(),
                baseResult.getWeightedPrecision(),
                baseResult.getWeightedRecall(),
                baseResult.getWeightedF1(),
                baseResult.getConfusionMatrix(),
                baseResult.getClassLabels(),
                baseResult.getRocAUC(),
                baseResult.getMacroAvgROCAUC(),
                baseResult.getPrAUC(),
                baseResult.getMacroAvgPRAUC(),
                calibrationMetrics,
                baseResult.getAdditionalStats()
        );
    }

    /**
     * Adds SVM-specific parameters to evaluation statistics.
     */
    @Override
    protected Map<String, Object> buildAdditionalStats(
            Evaluation evaluation, Instances testData, long evaluationTimeMs) {

        // Get base stats
        Map<String, Object> stats = super.buildAdditionalStats(evaluation, testData, evaluationTimeMs);

        // Add SVM-specific parameters
        try {
            stats.put("svm_complexity_c", smo.getC());
            stats.put("svm_epsilon", smo.getEpsilon());
        } catch (Exception e) {
            logger.debug("Could not extract SVM parameters");
        }

        return stats;
    }

    @Override
    public String getModelSummary() {
        requireTrained();

        StringBuilder summary = new StringBuilder();
        summary.append("=== SVM Classifier Summary ===\n\n");
        summary.append(String.format("Algorithm: %s\n", AlgorithmType.SVM.getDisplayName()));
        summary.append(String.format("State: %s\n", getState()));

        if (optimalConfig != null) {
            summary.append(String.format("Configuration: %s\n", optimalConfig));
            if (optimalConfig.getCvMacroF1() != null) {
                summary.append(String.format("CV Performance: Macro-F1=%.4f (±%.4f)\n",
                        optimalConfig.getCvMacroF1(), optimalConfig.getCvStdDev()));
            }
        } else {
            summary.append(String.format("C: %.4f, Epsilon: %.2E\n", getSMO().getC(), getSMO().getEpsilon()));
        }

        summary.append("\n");
        summary.append(String.format("Training: %d instances, %d features\n",
                getTrainingInstanceCount(), getFeatureCount()));
        summary.append(String.format("Classes: %d (%s)\n",
                supportedClasses.length, String.join(", ", supportedClasses)));
        summary.append(String.format("Training Time: %dms\n", lastTrainingTimeMs));
        summary.append(String.format("Vocabulary: %d terms", converter.getVocabulary().size()));

        return summary.toString();
    }

    @Override
    protected String getSubclassDiagnostics() {
        return String.format("""
                === SVMClassifier Diagnostics ===
                SMO: %s
                Training: %d instances, %d features
                Supported classes: %s""",
                smo != null ? "initialized" : "null",
                getTrainingInstanceCount(),
                getFeatureCount(),
                supportedClasses != null ? String.join(", ", supportedClasses) : "none"
        );
    }

    // HYPERPARAMETER SEARCH (PRIVATE)

    // OPTIMAL CONFIGURATION
    // Selected from 16-config grid search (5-fold CV on Amazon training set):
    //   - C values tested: {0.01, 0.05, 0.1, 1.0}
    //   - Kernels tested: Linear, RBF (gamma: 0.001, 0.01), Poly (degree: 2)
    // Result: Linear kernel with C=0.05 achieved 88.91% CV accuracy (best of 16)
    // See docs/TRAINING.md for rationale.
    private static final double[] C_VALUES = {0.05};
    private static final SVMConfig.KernelType[] KERNEL_TYPES = {
            SVMConfig.KernelType.LINEAR
    };
    private static final double[] GAMMA_VALUES = {0.001};  // Not used for linear kernel
    private static final int[] DEGREE_VALUES = {2};  // Not used for linear kernel

    /**
     * Finds the optimal SVM configuration using grid search with stratified k-fold CV.
     * Evaluates all hyperparameter combinations and selects the one with highest macro-F1 score.
     */
    private SVMConfig performGridSearch(Instances trainingData, boolean enableClassWeighting) throws Exception {
        if (trainingData == null || trainingData.numInstances() < cvFolds) {
            throw new IllegalArgumentException(
                    String.format("Need at least %d instances for %d-fold CV", cvFolds, cvFolds));
        }

        logger.info("Starting hyperparameter search on {} instances", trainingData.numInstances());
        logger.info("Class weighting: {}", enableClassWeighting ? "ENABLED" : "DISABLED");

        // Step 1: Calculate class weights if needed
        double[] classWeights = null;
        if (enableClassWeighting) {
            classWeights = calculateClassWeightsForGrid(trainingData);
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
                .max(Comparator.comparingDouble(SVMConfig::getCvMacroF1))
                .orElseThrow(() -> new Exception("Failed to find best configuration"));

        logger.info("Best configuration found: {}", bestConfig);
        logger.info("  -> Macro-F1: {} (±{})",
                String.format("%.4f", bestConfig.getCvMacroF1()),
                String.format("%.4f", bestConfig.getCvStdDev()));
        logger.info("  -> Accuracy: {}", String.format("%.4f", bestConfig.getCvAccuracy()));

        return bestConfig;
    }

    /**
     * Calculates class weights inversely proportional to class frequencies.
     * Minority classes receive higher weights in the SVM loss function.
     */
    private double[] calculateClassWeightsForGrid(Instances data) {
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
     * Generates all hyperparameter combinations to evaluate.
     * Tests linear, polynomial (varying degree), and RBF (varying gamma) kernels
     * across multiple regularization values.
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
     * Evaluates a configuration using Weka's built-in stratified k-fold cross-validation.
     * Updates the config with mean accuracy and macro-F1 score.
     */
    private void evaluateConfigWithCV(SVMConfig config, Instances data) throws Exception {
        // Create SMO classifier with config settings
        SMO smoForCV = new SMO();
        smoForCV.setKernel(config.createKernel());
        smoForCV.setOptions(weka.core.Utils.splitOptions(config.toOptionsString()));

        // Use Weka's built-in stratified cross-validation (automatically stratifies if class is nominal)
        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(smoForCV, data, cvFolds, new Random(42));

        // Extract aggregated metrics
        double accuracy = evaluation.pctCorrect() / 100.0;
        double macroF1 = calculateMacroF1(evaluation, data.numClasses());

        // Estimate std dev from confusion matrix (Weka doesn't expose per-fold variance)
        double stdDev = estimateStdDevFromCV(evaluation, accuracy);

        // Store results in config
        config.setCvMacroF1(macroF1);
        config.setCvAccuracy(accuracy);
        config.setCvStdDev(stdDev);
    }

    /**
     * Calculates macro-averaged F1 score (mean of per-class F1 scores).
     * Gives equal weight to all classes regardless of frequency.
     */
    private double calculateMacroF1(Evaluation evaluation, int numClasses) {
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
     * Estimates standard deviation from cross-validation results.
     * Since Weka's crossValidateModel doesn't expose per-fold metrics, we estimate
     * variance using a bootstrap-based approximation from the confusion matrix.
     */
    private double estimateStdDevFromCV(Evaluation evaluation, double accuracy) {
        // Simple heuristic: use sqrt(p*(1-p)/n) where p is accuracy, n is sample size
        // This is the standard error for a binomial proportion
        int totalInstances = (int) evaluation.numInstances();
        if (totalInstances == 0) return 0.0;

        double variance = accuracy * (1.0 - accuracy) / totalInstances;
        return Math.sqrt(variance) * Math.sqrt(cvFolds);  // Adjust for k folds
    }

    // ==================== PUBLIC ACCESSORS ====================

    /**
     * Returns the underlying SMO classifier for testing and advanced configuration.
     */
    public SMO getSMO() {
        return smo;
    }

    /**
     * Extracts feature importance weights directly from the linear SVM coefficients.
     * This method only works for linear kernel SVMs.
     *
     * <p>For linear SVM with Weka's SMO, the weight vector is stored directly in sparse format:
     * <ul>
     *   <li>sparseIndices[classifier][class] = array of non-zero feature indices</li>
     *   <li>sparseWeights[classifier][class] = array of corresponding weight values</li>
     * </ul>
     *
     * @return Map of feature names to their coefficient weights, or empty map if extraction fails
     */
    public Map<String, Double> extractFeatureWeights() {
        if (!isTrained()) {
            logger.warn("Cannot extract feature weights: model not trained");
            return Collections.emptyMap();
        }

        try {
            // Get sparse representation from SMO
            int[][][] sparseIndices = smo.sparseIndices();
            double[][][] sparseWeights = smo.sparseWeights();

            if (sparseIndices == null || sparseWeights == null ||
                sparseIndices.length == 0 || sparseWeights.length == 0) {
                logger.warn("sparseIndices() or sparseWeights() returned null or empty");
                return Collections.emptyMap();
            }

            // Get training data structure for feature names
            Instances trainedData = getTrainingStructure();
            if (trainedData == null) {
                logger.warn("Training data structure not available");
                return Collections.emptyMap();
            }

            int classIndex = trainedData.classIndex();
            int numAttributes = trainedData.numAttributes();
            Map<String, Double> weights = new HashMap<>();

            // Initialize all features to 0 (excluding class attribute)
            for (int i = 0; i < numAttributes; i++) {
                if (i != classIndex) {
                    weights.put(trainedData.attribute(i).name(), 0.0);
                }
            }

            // For binary classification, use first classifier
            int[][] indicesForClassifier = sparseIndices[0];
            double[][] weightsForClassifier = sparseWeights[0];

            if (indicesForClassifier == null || weightsForClassifier == null ||
                indicesForClassifier.length == 0) {
                logger.warn("No sparse data for classifier 0");
                return Collections.emptyMap();
            }

            // Find the first non-null class weight vector
            // In Weka's binary SVM, one class may have null weights while the other has the data
            int[] featureIndices = null;
            double[] featureWeights = null;

            for (int classIdx = 0; classIdx < indicesForClassifier.length; classIdx++) {
                if (indicesForClassifier[classIdx] != null && weightsForClassifier[classIdx] != null) {
                    featureIndices = indicesForClassifier[classIdx];
                    featureWeights = weightsForClassifier[classIdx];
                    logger.debug("Using class {} weights ({} entries)", classIdx, featureIndices.length);
                    break;
                }
            }

            if (featureIndices == null || featureWeights == null) {
                logger.warn("No valid weight vectors found");
                return Collections.emptyMap();
            }

            int extracted = 0;
            for (int i = 0; i < featureIndices.length && i < featureWeights.length; i++) {
                int featureIdx = featureIndices[i];

                if (featureIdx == classIndex) continue;
                if (featureIdx < 0 || featureIdx >= numAttributes) continue;

                String featureName = trainedData.attribute(featureIdx).name();
                weights.put(featureName, featureWeights[i]);
                extracted++;
            }

            long nonZeroCount = weights.values().stream().filter(w -> Math.abs(w) > 1e-10).count();
            logger.info("Extracted {} non-zero feature weights ({} total)", nonZeroCount, extracted);
            return weights;

        } catch (Exception e) {
            logger.error("Failed to extract feature weights: {}", e.getMessage(), e);
            return Collections.emptyMap();
        }
    }

    /**
     * Returns the optimal configuration from hyperparameter grid search.
     * Returns null if hyperparameter tuning was disabled during training.
     * Use this to inspect selected kernel, C parameter, and CV accuracy.
     *
     * @return optimal config or null if tuning was not performed
     */
    public SVMConfig getOptimalConfig() {
        return optimalConfig;
    }
}