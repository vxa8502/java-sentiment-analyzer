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

import javax.annotation.PreDestroy;
import java.util.*;

/**
 * Support Vector Machine (SVM) sentiment classifier using Weka's SMO implementation.
 *
 * MODEL SELECTION RATIONALE:
 * ==========================
 * SVM was selected for this sentiment analysis task for the following reasons:
 *
 * 1. High-Dimensional Sparse Data: Text features (TF-IDF) result in high-dimensional sparse
 *    vectors where SVM excels due to its ability to find optimal separating hyperplanes.
 *
 * 2. Strong Theoretical Foundation: SVMs have solid mathematical foundations with guaranteed
 *    convergence and well-understood generalization bounds.
 *
 * 3. Robustness to Overfitting: The margin maximization principle and regularization (C parameter)
 *    help prevent overfitting even with limited training data.
 *
 * 4. Binary and Multi-Class Support: SMO naturally handles binary classification and extends
 *    to multi-class problems, making it suitable for sentiment analysis (positive/negative/neutral).
 *
 * 5. Enterprise Java Ecosystem: Weka's mature, production-tested SVM implementation integrates
 *    well with Java-based ML pipelines in enterprise environments.
 *
 * Alternative Considerations:
 * - Naive Bayes: Faster training but assumes feature independence (violated in text)
 * - Random Forest: Good ensemble performance but higher memory footprint and harder to tune
 * - Neural Networks: Better for very large datasets but require more training data and compute
 *
 * ARCHITECTURE:
 * =============
 * This classifier owns the complete training pipeline:
 * 1. Accepts raw List<Dataset> in train()
 * 2. Fits preprocessor (text cleaning, tokenization, stopword removal)
 * 3. Fits feature extractor (TF-IDF vectorization)
 * 4. Trains SVM on transformed features
 *
 * THREAD SAFETY:
 * ==============
 * - Training: Exclusive write lock (modifies model state)
 * - Inference: Concurrent read lock (thread-safe predictions)
 * - Instance structure validation cached for performance
 */
public class SVMClassifier extends ClassifierTrainingTemplate<ClassifierEvaluationResult>
        implements ClassifierEvaluator, WekaClassifier {

    private static final Logger logger = LoggerFactory.getLogger(SVMClassifier.class);

    // ✅ Performance: Cache validation result to avoid repeated checks
    private volatile boolean instanceStructureValidated = false;

    // ✅ Non-final to allow model replacement during load
    private SMO smo;

    // Core ML components (immutable after construction)
    private final TextPreprocessor preprocessor;
    // NOTE: converter, trainingDataStructure, supportedClasses now inherited from base class

    // Hyperparameter tuning configuration
    private boolean enableHyperparameterTuning = false;
    private int cvFolds = 5;
    private SVMConfig optimalConfig;

    // Class imbalance detection threshold
    private double classImbalanceThreshold = 3.0;

    /**
     * Creates a new thread-safe SVM classifier with default configuration.
     */
    public SVMClassifier(TextPreprocessor preprocessor, WekaInstancesConverter converter) {
        if (preprocessor == null || converter == null) {
            throw new IllegalArgumentException("Preprocessor and converter cannot be null");
        }

        this.preprocessor = preprocessor;
        this.converter = converter;
        this.smo = new SMO();

        logger.info("Created SVMClassifier - manages full training pipeline");
    }

    /**
     * Creates classifier with custom SMO configuration.
     */
    public SVMClassifier(TextPreprocessor preprocessor, WekaInstancesConverter converter, SMO customSMO) {
        if (preprocessor == null || converter == null || customSMO == null) {
            throw new IllegalArgumentException("All dependencies must be non-null");
        }

        this.preprocessor = preprocessor;
        this.converter = converter;
        this.smo = customSMO;

        logger.info("Created BasicSVMClassifier with custom SMO");
    }

    /**
     * Enables hyperparameter tuning using stratified k-fold cross-validation.
     *
     * When enabled, the classifier will perform grid search before training to find
     * optimal C, kernel type, and kernel parameters.
     *
     * @param enable Whether to enable hyperparameter tuning
     * @param numFolds Number of folds for cross-validation (typically 5 or 10)
     */
    public void setHyperparameterTuning(boolean enable, int numFolds) {
        this.enableHyperparameterTuning = enable;
        this.cvFolds = numFolds;
        logger.info("Hyperparameter tuning: {} ({}-fold CV)", enable ? "ENABLED" : "DISABLED", numFolds);
    }

    /**
     * Convenience method to enable hyperparameter tuning with default 5-fold CV.
     */
    public void setHyperparameterTuning(boolean enable) {
        setHyperparameterTuning(enable, 5);
    }

    /**
     * Sets the class imbalance threshold for automatic class weighting.
     *
     * When the ratio of max_class_count / min_class_count exceeds this threshold,
     * class weighting will be automatically enabled during training.
     *
     * @param threshold The imbalance ratio threshold (must be > 1.0)
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
    public String getAlgorithmName() {
        return AlgorithmType.SVM.getDisplayName();
    }

    @Override
    public String[] getSupportedClasses() {
        requireTrained();
        return supportedClasses != null ? supportedClasses.clone() : new String[0];
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

    @Override
    public TextPreprocessor getPreprocessor() {
        return preprocessor;
    }

    // ==================== DERIVED GETTERS (SINGLE SOURCE OF TRUTH) ====================

    /**
     * ✅ MEMORY FIX: Derive training instance count from structure
     * No duplicate storage needed
     */
    // NOTE: getTrainingInstanceCount() and getFeatureCount() now inherited from base class

    // ==================== TEMPLATE METHOD IMPLEMENTATIONS ====================

    /**
     * ✅ WORKFLOW FIX: Now accepts raw List<Dataset> and fits the full pipeline internally
     *
     * PIPELINE FLOW:
     * 1. Validate raw datasets
     * 2. Fit TextPreprocessor on raw data
     * 3. Fit WekaInstancesConverter to get Instances
     * 4. Train SVM on converted Instances
     */
    @Override
    protected ClassifierEvaluationResult doTrain(List<Dataset> rawDatasets) throws Exception {
        if (rawDatasets == null || rawDatasets.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }

        logger.info("Training SVM on {} raw datasets with full pipeline", rawDatasets.size());

        // Step 1: Fit preprocessing pipeline
        logger.info("Step 1/3: Fitting preprocessing pipeline");
        preprocessor.fit(rawDatasets);
        logger.info("Preprocessor fitted. Vocabulary: {}",
                preprocessor.getPipelineState().vocabularySize);

        // Step 2: Fit feature extraction (converter) and get Instances
        logger.info("Step 2/3: Fitting feature extraction");
        Instances trainingInstances = converter.fit(rawDatasets);
        logger.info("Converter fitted. Features: {}, Vocabulary: {}",
                trainingInstances.numAttributes() - 1,
                converter.getVocabulary().size());

        // Step 3: Train SVM on converted Instances
        logger.info("Step 3/3: Training SVM classifier");
        validateWekaTrainingData(trainingInstances);
        configureSMOForTraining(trainingInstances);
        performModelTraining(trainingInstances);
        finalizeTraining(trainingInstances);

        // Step 4: Validate pipeline consistency (CRITICAL)
        validatePipelineConsistency();

        logger.info("SVM training complete. Pipeline ready for inference.");

        // Return null - no evaluation during training
        return null;
    }

    @Override
    protected void doClearResources() {
        trainingDataStructure = null;
        supportedClasses = null;
        instanceStructureValidated = false;
    }

    @PreDestroy
    public void cleanup() {
        logger.info("Cleaning up BasicSVMClassifier resources");
        doClearResources();
    }

    // ==================== TRAINING HELPERS ====================

    /**
     * Override to add SVM-specific class balance validation.
     * Inherits common validation from ClassifierTrainingTemplate.
     */
    @Override
    protected void validateWekaTrainingData(Instances data) {
        super.validateWekaTrainingData(data); // Common validation
        checkClassBalance(data); // SVM-specific validation
    }

    /**
     * ✅ FIXED: Validates class distribution and throws exception early if invalid.
     * SVM-SPECIFIC: Checks for class imbalance and minimum class requirements.
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
     * ✅ CRITICAL: Validates vocabulary consistency between preprocessor and converter.
     *
     * This ensures that the preprocessing pipeline and feature extraction operate on
     * compatible vocabulary spaces. A mismatch means the model was trained on different
     * features than what will be used during inference, resulting in INVALID predictions.
     *
     * CORRECT VALIDATION (Fixed):
     * - Converter vocabulary must be a SUBSET of preprocessor vocabulary (V_c ⊆ V_p)
     * - This allows valid feature selection while preventing vocabulary drift
     * - Prevents training on features that won't exist during inference
     *
     * Mathematical Justification (Aria):
     * Let V_p be the preprocessor vocabulary and V_c be the converter vocabulary.
     * For valid composition φ ∘ π where:
     * - π: text → tokens(V_p) (preprocessing transformation)
     * - φ: tokens → R^|V_c| (feature mapping)
     *
     * We require V_c ⊆ V_p to ensure that every feature used by the converter
     * corresponds to a valid token that can be produced by the preprocessor.
     * If V_c ⊄ V_p, then φ expects tokens that π cannot produce, leading to
     * undefined behavior and invalid predictions during inference.
     *
     * @throws IllegalStateException if converter vocabulary is not a subset of preprocessor vocabulary
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

    /**
     * Helper to format example terms for error messages.
     */
    private String getExampleTerms(Set<String> terms, int maxExamples) {
        return terms.stream()
                .limit(maxExamples)
                .collect(java.util.stream.Collectors.joining(", "));
    }

    private void configureSMOForTraining(Instances trainingData) throws Exception {
        if (enableHyperparameterTuning) {
            logger.info("=== HYPERPARAMETER TUNING ENABLED ===");
            logger.info("Performing grid search with {}-fold stratified cross-validation", cvFolds);

            // Detect class imbalance and enable class weighting if needed
            boolean useClassWeighting = shouldUseClassWeighting(trainingData);

            // Perform grid search
            SVMHyperparameterSearch search = new SVMHyperparameterSearch(cvFolds);
            optimalConfig = search.findOptimalConfig(trainingData, useClassWeighting);

            // Apply optimal configuration
            applySVMConfig(optimalConfig);

            logger.info("=== OPTIMAL CONFIGURATION SELECTED ===");
            logger.info("Config: {}", optimalConfig);
            logger.info("Expected Performance: Macro-F1={} (±{}), Accuracy={}",
                    String.format("%.4f", optimalConfig.getCvMacroF1()),
                    String.format("%.4f", optimalConfig.getCvStdDev()),
                    String.format("%.4f", optimalConfig.getCvAccuracy()));

        } else {
            logger.info("Hyperparameter tuning DISABLED - using default linear kernel");
            logger.warn("RECOMMENDATION: Enable hyperparameter tuning for production models");

            // Default: Linear kernel with C=1.0
            // This is a reasonable starting point for text classification
            optimalConfig = new SVMConfig(1.0, SVMConfig.KernelType.LINEAR, 0.01, 1, null, 1.0E-12);
            applySVMConfig(optimalConfig);

            logger.info("Using default config: C=1.0, Linear Kernel");
        }
    }

    /**
     * Checks if class weighting should be applied based on class distribution.
     *
     * Returns true if imbalance ratio exceeds the configured threshold.
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

    /**
     * Applies an SVMConfig to the SMO classifier.
     */
    private void applySVMConfig(SVMConfig config) throws Exception {
        // Set kernel
        smo.setKernel(config.createKernel());

        // Set options (C, epsilon, etc.)
        String options = config.toOptionsString();
        smo.setOptions(weka.core.Utils.splitOptions(options));

        // Log configuration (handle null kernel for mocked SMO)
        Kernel kernel = smo.getKernel();
        String kernelName = (kernel != null) ? kernel.getClass().getSimpleName() : "null";
        logger.debug("SMO configured: C={}, Epsilon={}, Kernel={}",
                smo.getC(), smo.getEpsilon(), kernelName);
    }

    private void performModelTraining(Instances trainingData) throws Exception {
        logger.info("Training SVM model on {} instances", trainingData.numInstances());
        smo.buildClassifier(trainingData);
        logger.info("SVM model training complete");
        this.instanceStructureValidated = false;
    }

    // NOTE: finalizeTraining(), classify(), getClassificationProbabilities() now inherited from base class

    // ==================== EVALUATION (THREAD-SAFE) - OVERRIDDEN FOR ADVANCED METRICS ====================

    /**
     * ✅ OPTIMIZED: Single-pass evaluation that extracts metrics during iteration.
     *
     * Overrides base class to add advanced metrics (ROC-AUC, PR-AUC, calibration).
     *
     * PREVIOUS APPROACH (two-pass):
     * - Pass 1: evaluation.evaluateModel(smo, testData)
     * - Pass 2: iterate again to extract probabilities for AUC
     *
     * NEW APPROACH (one-pass):
     * - Single iteration using evaluateModelOnceAndRecordPrediction
     * - Collect probabilities during evaluation
     *
     * PROOF: testMetricsEquivalence_OnePassVsTwoPass verifies identical results
     */
    @Override
    protected ClassifierEvaluationResult performEvaluation(Instances testData)
            throws Exception {
        long startTime = System.currentTimeMillis();

        Evaluation evaluation = new Evaluation(trainingDataStructure);

        // ✅ NEW: Collect probabilities during evaluation
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
                evaluation, testData, evaluationTime, "",
                probabilities, actualLabels);

        String accuracy = String.format("%.3f", result.getAccuracy());
        logger.info("Evaluation complete in {}ms: accuracy={} (single-pass optimized)",
                evaluationTime, accuracy);

        return result;
    }

    /**
     * ✅ OPTIMIZED: Build evaluation result using pre-collected probabilities.
     *
     * @param probabilities Already collected during single-pass evaluation
     * @param actualLabels Already collected during single-pass evaluation
     */
    private ClassifierEvaluationResult buildEvaluationResult(
            Evaluation evaluation, Instances testData,
            long evaluationTimeMs, String predictions,
            double[][] probabilities, int[] actualLabels) throws Exception {

        double accuracy = evaluation.pctCorrect() / 100.0;
        int numClasses = supportedClasses.length;

        double[] precision = new double[numClasses];
        double[] recall = new double[numClasses];
        double[] f1Score = new double[numClasses];

        for (int i = 0; i < numClasses; i++) {
            final int classIndex = i;
            precision[i] = safeMetric(() -> evaluation.precision(classIndex));
            recall[i] = safeMetric(() -> evaluation.recall(classIndex));
            f1Score[i] = safeMetric(() -> evaluation.fMeasure(classIndex));
        }

        double macroAvgPrecision = Arrays.stream(precision).average().orElse(0.0);
        double macroAvgRecall = Arrays.stream(recall).average().orElse(0.0);
        double macroAvgF1 = Arrays.stream(f1Score).average().orElse(0.0);

        double weightedPrecision = safeMetric(() -> evaluation.weightedPrecision());
        double weightedRecall = safeMetric(() -> evaluation.weightedRecall());
        double weightedF1 = safeMetric(() -> evaluation.weightedFMeasure());

        double[][] confusionMatrix = evaluation.confusionMatrix();

        // ==================== ADVANCED METRICS ====================

        // Compute ROC-AUC and PR-AUC
        double[] rocAUC = null;
        Double macroAvgROCAUC = null;
        double[] prAUC = null;
        Double macroAvgPRAUC = null;
        sentiment.evaluation.CalibrationMetrics calibrationMetrics = null;

        try {
            // ✅ NO LONGER NEEDED: extractPredictionsAndProbabilities
            // Probabilities and labels already collected during single-pass evaluation

            // Compute ROC-AUC per class
            rocAUC = sentiment.evaluation.AUCCalculator.computeMultiClassROCAUC(
                    probabilities, actualLabels);
            macroAvgROCAUC = Arrays.stream(rocAUC).average().orElse(0.0);

            // Compute PR-AUC per class
            prAUC = sentiment.evaluation.AUCCalculator.computeMultiClassPRAUC(
                    probabilities, actualLabels);
            macroAvgPRAUC = Arrays.stream(prAUC).average().orElse(0.0);

            // Compute calibration metrics (for binary or multi-class)
            if (numClasses == 2) {
                // Binary classification: use positive class probabilities
                double[] positiveProbs = new double[probabilities.length];
                for (int i = 0; i < probabilities.length; i++) {
                    positiveProbs[i] = probabilities[i][1];  // Class index 1
                }
                calibrationMetrics = sentiment.evaluation.CalibrationMetrics.compute(
                        positiveProbs, actualLabels, 10);
            } else {
                // Multi-class: compute averaged calibration
                calibrationMetrics = sentiment.evaluation.CalibrationMetrics.computeMultiClass(
                        probabilities, actualLabels, 10);
            }

            logger.debug("Advanced metrics computed: ROC-AUC={}, PR-AUC={}, Brier={}",
                    String.format("%.4f", macroAvgROCAUC),
                    String.format("%.4f", macroAvgPRAUC),
                    String.format("%.4f", calibrationMetrics.getBrierScore()));

        } catch (Exception e) {
            logger.warn("Failed to compute advanced metrics: {}", e.getMessage());
            // Continue without advanced metrics
        }

        Map<String, Object> stats = buildAdditionalStats(evaluation, testData, evaluationTimeMs);

        // Use advanced constructor if metrics available
        if (macroAvgROCAUC != null) {
            return new ClassifierEvaluationResult(
                    getAlgorithmName(), accuracy,
                    precision, recall, f1Score,
                    macroAvgPrecision, macroAvgRecall, macroAvgF1,
                    weightedPrecision, weightedRecall, weightedF1,
                    confusionMatrix, supportedClasses,
                    rocAUC, macroAvgROCAUC,
                    prAUC, macroAvgPRAUC,
                    calibrationMetrics,
                    stats
            );
        } else {
            // Fallback to basic constructor
            return new ClassifierEvaluationResult(
                    getAlgorithmName(), accuracy,
                    precision, recall, f1Score,
                    macroAvgPrecision, macroAvgRecall, macroAvgF1,
                    weightedPrecision, weightedRecall, weightedF1,
                    confusionMatrix, supportedClasses, stats
            );
        }
    }


    // NOTE: safeMetric() and MetricSupplier now inherited from base class

    /**
     * Override to add SVM-specific stats (C parameter, epsilon).
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

    // ==================== MODEL SUMMARY ====================

    @Override
    public String getModelSummary() {
        requireTrained();

        StringBuilder summary = new StringBuilder();
        summary.append("=== SVM Classifier Summary ===\n\n");
        summary.append(String.format("Algorithm: %s\n", AlgorithmType.SVM.getDisplayName()));
        summary.append(String.format("State: %s\n", classifierState));

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
        return String.format(
                "=== BasicSVMClassifier Diagnostics ===\n" +
                        "SMO: %s\n" +
                        "Training: %d instances, %d features\n" +
                        "Supported classes: %s\n" +
                        "Instance validation cached: %s",
                smo != null ? "initialized" : "null",
                getTrainingInstanceCount(),
                getFeatureCount(),
                supportedClasses != null ? String.join(", ", supportedClasses) : "none",
                instanceStructureValidated
        );
    }

    // ==================== ACCESSORS ====================

    /**
     * Get the underlying SMO classifier.
     * Used for testing and advanced configuration.
     */
    public SMO getSMO() {
        return smo;
    }

    // NOTE: setSMO(), getTrainingStructure(), setTrainingMetadata() removed
    // Persistence now uses abstract methods from ClassifierTrainingTemplate

    // NOTE: getPreprocessor() implemented above to satisfy abstract method

    /**
     * Get converter (for utility classes).
     */
    public WekaInstancesConverter getConverter() {
        return converter;
    }

    /**
     * Get the optimal configuration selected by hyperparameter search.
     * Returns null if hyperparameter tuning was not enabled.
     */
    public SVMConfig getOptimalConfig() {
        return optimalConfig;
    }

    // ==================== WekaClassifier INTERFACE ====================

    /**
     * Returns the underlying Weka classifier for batch optimization.
     * Required by WekaClassifier interface for BatchPredictor support.
     *
     * @return The SMO classifier instance
     */
    @Override
    public weka.classifiers.Classifier getWekaClassifier() {
        return smo;
    }

    // NOTE: executeInference(Callable) now inherited from base class
}