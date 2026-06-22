package sentiment.models;

import sentiment.data.Dataset;
import sentiment.PipelineState;
import sentiment.TrainingTemplate;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import sentiment.util.ValidationUtils;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;

/**
 * Template base class implementing common classifier training/inference logic.
 * <p>Extends {@link TrainingTemplate} to provide unified state management and lifecycle control.
 *
 * <h2>Thread Safety Contract</h2>
 * <p>This class provides thread-safe inference through a centralized locking mechanism:
 *
 * <ul>
 *   <li><b>Training phase</b>: NOT thread-safe. The {@link #train} method must be called from a
 *       single thread. Do not call training while inference is in progress.</li>
 *   <li><b>Inference phase</b>: Thread-safe. After training completes, multiple threads can safely
 *       call {@link #classify}, {@link #classifyWithProbabilities}, and {@link #getClassificationProbabilities}
 *       concurrently.</li>
 *   <li><b>State queries</b>: Thread-safe. Methods like {@link #isTrained()}, {@link #getState()},
 *       {@link #getSupportedClasses()} are safe to call from any thread at any time.</li>
 * </ul>
 *
 * <h3>Synchronization Strategy</h3>
 * <p>Weka classifiers ({@link weka.classifiers.Classifier}) are NOT thread-safe internally.
 * This template uses a single object lock ({@code classifierLock}) to serialize all inference
 * operations. This ensures:
 * <ul>
 *   <li>No concurrent modifications to classifier internal state</li>
 *   <li>Consistent reads of probability distributions</li>
 *   <li>Safe instance transformation through the preprocessing pipeline</li>
 * </ul>
 *
 * <h3>Performance Implications</h3>
 * <p>Inference operations are serialized, which may limit throughput under high concurrency.
 * For high-throughput scenarios, consider:
 * <ul>
 *   <li>Running multiple classifier instances behind a load balancer</li>
 *   <li>Using request batching to amortize lock overhead</li>
 *   <li>Caching frequent predictions</li>
 * </ul>
 *
 * @param <T> type of training result (optional, can be {@link Void})
 * @see #executeInference for the synchronized inference method
 */
public abstract class ClassifierTrainingTemplate<T> extends TrainingTemplate<T> implements WekaClassifier {

    // =========================================================================
    // WEKA-SPECIFIC SHARED STATE
    // These fields are populated during training and read during inference.
    // Write access is single-threaded (training), read access is synchronized.
    // =========================================================================

    /**
     * Training data structure containing schema only (no instances).
     * Used for feature count and instance validation during inference.
     * <p>Thread safety: Written once during training, read-only thereafter.
     */
    protected Instances trainingDataStructure;

    /**
     * Supported class labels (e.g., {@code ["positive", "negative", "neutral"]}).
     * Populated during training from the Weka class attribute.
     * <p>Thread safety: Written once during training, read-only thereafter.
     */
    protected String[] supportedClasses;

    /**
     * Feature converter for Weka instances.
     * Subclasses should set this field if they use {@link WekaInstancesConverter}.
     * <p>Thread safety: Set during construction, read-only thereafter.
     */
    protected WekaInstancesConverter converter;

    /**
     * Lock for thread-safe classifier operations.
     * <p>All Weka classifier inference operations must be synchronized on this lock
     * because Weka classifiers maintain internal state that is not thread-safe.
     * <p>Usage: Use {@link #executeInference} for all inference operations.
     */
    private final Object classifierLock = new Object();

    // TEMPLATE METHOD: TRAINING PHASE

    /**
     * Trains the classifier on the provided training data.
     *
     * @param trainingData the training datasets
     * @throws ClassificationException if training fails
     */
    @Override
    public void train(List<Dataset> trainingData) throws ClassificationException {
        try {
            trainInternal(trainingData);
        } catch (IllegalArgumentException e) {
            throw e;  // Let validation errors propagate as-is
        } catch (Exception e) {
            throw ClassificationException.trainingError(e);
        }
    }

    /**
     * Default implementation of training workflow for simple classifiers.
     * <p>
     * Uses the standard pipeline: fit preprocessing → fit feature extraction → train model.
     * Subclasses with custom training logic (e.g., SVMClassifier with hyperparameter tuning)
     * should override this method.
     *
     * @param rawDatasets raw training datasets
     * @return null (no evaluation result during training)
     * @throws Exception if training fails
     */
    @Override
    protected T doTrain(List<Dataset> rawDatasets) throws Exception {
        if (rawDatasets == null || rawDatasets.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }

        // Use consolidated training pipeline from base class
        performStandardTrainingPipeline(rawDatasets, this::performAlgorithmSpecificTraining);

        return null;
    }

    /**
     * Performs algorithm-specific model training.
     * <p>
     * Subclasses must implement this to call their specific Weka classifier's
     * {@code buildClassifier()} method.
     *
     * @param trainingData prepared Weka instances
     * @throws Exception if training fails
     */
    protected abstract void performAlgorithmSpecificTraining(Instances trainingData) throws Exception;

    /**
     * Returns the underlying Weka classifier instance.
     * This is used by the consolidated inference methods in the base class.
     *
     * @return the Weka classifier (e.g., NaiveBayes, SMO, Logistic, RandomForest)
     */
    protected abstract weka.classifiers.Classifier getWekaClassifierInstance();

    /**
     * Sets the underlying Weka classifier instance.
     * Used by persistence utilities to restore a saved model.
     *
     * @param classifier the Weka classifier to set
     */
    protected abstract void setWekaClassifierInstance(weka.classifiers.Classifier classifier);

    /**
     * Returns the underlying Weka classifier for external use (e.g., feature importance analysis).
     * Implements {@link WekaClassifier#getWekaClassifier()} by delegating to the internal
     * {@link #getWekaClassifierInstance()} method.
     *
     * @return the Weka classifier
     */
    @Override
    public weka.classifiers.Classifier getWekaClassifier() {
        return getWekaClassifierInstance();
    }

    /**
     * Returns the component type for logging.
     *
     * @return "classifier"
     */
    @Override
    protected final String getComponentType() {
        return "classifier";
    }

    // READ-ONLY STATE ACCESS (inherited from base, but providing classifier-specific aliases)

    /**
     * Returns the current classifier state.
     *
     * @return the current {@link PipelineState}
     */
    @SuppressWarnings("unused") // Public API for external consumers
    public PipelineState getClassifierState() {
        return getState();
    }

    /**
     * Checks if the classifier is ready for inference.
     *
     * @return {@code true} if the classifier is trained and ready, {@code false} otherwise
     */
    @Override
    public boolean isTrained() {
        return isReady();
    }

    // WEKA TRAINING DATA VALIDATION

    /**
     * Validates Weka training data structure and statistics.
     * <p>
     * Common validation logic shared across all Weka-based classifiers.
     * Subclasses can override to add algorithm-specific validations.
     *
     * @param data Weka Instances to validate
     * @throws IllegalArgumentException if data structure is invalid
     */
    protected void validateWekaTrainingData(Instances data) {
        if (data.numInstances() < 10) {
            getLogger().warn("Small training set ({} instances)", data.numInstances());
        }

        if (data.classIndex() == -1) {
            throw new IllegalArgumentException("Training data must have class attribute set");
        }

        if (data.classAttribute().numValues() < 2) {
            throw new IllegalArgumentException("Need at least 2 classes");
        }

        logWekaDatasetStatistics(data);
    }

    /**
     * Logs Weka dataset statistics including instances, features, and class distribution.
     *
     * @param data Weka Instances to log statistics for
     */
    protected void logWekaDatasetStatistics(Instances data) {
        getLogger().info("Dataset: {} instances, {} features, {} classes",
                data.numInstances(), data.numAttributes() - 1,
                data.classAttribute().numValues());

        int[] classCounts = new int[data.classAttribute().numValues()];
        for (int i = 0; i < data.numInstances(); i++) {
            classCounts[(int) data.instance(i).classValue()]++;
        }

        for (int i = 0; i < classCounts.length; i++) {
            String percentage = String.format("%.1f", (classCounts[i] * 100.0) / data.numInstances());
            getLogger().info("  {}: {} ({}%)",
                    data.classAttribute().value(i),
                    classCounts[i],
                    percentage);
        }
    }

    // WEKA HELPER METHODS

    /**
     * Formats probability distribution as a human-readable string.
     *
     * @param probs probability array (must match {@code supportedClasses} length)
     * @return formatted string like {@code "[positive: 0.850, negative: 0.150]"}
     */
    protected String formatProbabilities(double[] probs) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < probs.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%s: %.3f", supportedClasses[i], probs[i]));
        }
        sb.append("]");
        return sb.toString();
    }

    /**
     * Logs probability distribution at DEBUG level and returns the probabilities.
     *
     * @param probs probability array to log and return
     * @return the same probability array (for convenient return statements)
     */
    protected double[] logProbabilityDistribution(double[] probs) {
        if (getLogger().isDebugEnabled()) {
            getLogger().debug("Probability distribution: {}", formatProbabilities(probs));
        }
        return probs;
    }

    /**
     * Returns the training instance count.
     *
     * @return number of training instances, or {@code 0} if not trained
     */
    protected int getTrainingInstanceCount() {
        return trainingDataStructure != null ? trainingDataStructure.numInstances() : 0;
    }

    /**
     * Returns the feature count excluding the class attribute.
     *
     * @return number of features, or {@code 0} if not trained
     */
    protected int getFeatureCount() {
        return trainingDataStructure != null ? trainingDataStructure.numAttributes() - 1 : 0;
    }

    /**
     * Validates text input for classification methods.
     *
     * @param text input text to validate
     * @throws IllegalArgumentException if {@code text} is {@code null} or empty
     */
    protected void validateTextInput(String text) {
        ValidationUtils.requireNonEmpty(text);
    }

    /**
     * Safely computes a metric, returning {@code 0.0} on NaN or exception.
     *
     * @param supplier metric computation that may throw or return NaN
     * @return metric value, or {@code 0.0} if unavailable
     */
    protected double safeMetric(MetricSupplier supplier) {
        try {
            double value = supplier.get();
            return Double.isNaN(value) ? 0.0 : value;
        } catch (Exception e) {
            return 0.0;
        }
    }

    /**
     * Functional interface for metric suppliers that may throw exceptions.
     */
    @FunctionalInterface
    protected interface MetricSupplier {
        /**
         * Computes and returns a metric value.
         *
         * @return the computed metric
         * @throws Exception if computation fails
         */
        double get() throws Exception;
    }

    /**
     * Builds additional statistics map for evaluation results.
     *
     * @param evaluation Weka evaluation object
     * @param testData test dataset
     * @param evaluationTimeMs time taken for evaluation in milliseconds
     * @return map of additional statistics
     */
    protected Map<String, Object> buildAdditionalStats(
            Evaluation evaluation, Instances testData, long evaluationTimeMs) {

        Map<String, Object> stats = new HashMap<>();
        stats.put("totalInstances", testData.numInstances());
        stats.put("correctlyClassified", (int) evaluation.correct());
        stats.put("incorrectlyClassified", (int) evaluation.incorrect());
        stats.put("evaluationTimeMs", evaluationTimeMs);
        stats.put("kappa", safeMetric(evaluation::kappa));

        if (lastTrainingTimeMs > 0) {
            stats.put("trainingTimeMs", lastTrainingTimeMs);
        }

        if (converter != null && converter.isReady()) {
            stats.put("vocabularySize", converter.getVocabulary().size());
        }

        return stats;
    }

    /**
     * Builds a {@link sentiment.evaluation.ClassifierEvaluationResult} from Weka Evaluation.
     *
     * @param evaluation Weka evaluation object
     * @param testData test dataset
     * @param evaluationTimeMs time taken for evaluation in milliseconds
     * @return complete evaluation result
     */
    protected sentiment.evaluation.ClassifierEvaluationResult buildEvaluationResult(
            Evaluation evaluation, Instances testData, long evaluationTimeMs) {

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

        double weightedPrecision = safeMetric(evaluation::weightedPrecision);
        double weightedRecall = safeMetric(evaluation::weightedRecall);
        double weightedF1 = safeMetric(evaluation::weightedFMeasure);

        double[][] confusionMatrix = evaluation.confusionMatrix();

        // Compute ROC-AUC and PR-AUC for all classes
        double[] rocAUC = new double[numClasses];
        double[] prAUC = new double[numClasses];
        for (int i = 0; i < numClasses; i++) {
            final int classIndex = i;
            rocAUC[i] = safeMetric(() -> evaluation.areaUnderROC(classIndex));
            prAUC[i] = safeMetric(() -> evaluation.areaUnderPRC(classIndex));
        }
        Double macroAvgROCAUC = Arrays.stream(rocAUC).average().orElse(0.0);
        Double macroAvgPRAUC = Arrays.stream(prAUC).average().orElse(0.0);

        Map<String, Object> stats = buildAdditionalStats(evaluation, testData, evaluationTimeMs);

        return new sentiment.evaluation.ClassifierEvaluationResult(
                getAlgorithmName(), accuracy,
                precision, recall, f1Score,
                macroAvgPrecision, macroAvgRecall, macroAvgF1,
                weightedPrecision, weightedRecall, weightedF1,
                confusionMatrix, supportedClasses,
                rocAUC, macroAvgROCAUC, prAUC, macroAvgPRAUC,
                null,  // calibrationMetrics - computed by subclasses if needed
                stats
        );
    }

    /**
     * Returns the supported sentiment classes for this classifier.
     *
     * @return Array of class labels (e.g., ["negative", "neutral", "positive"])
     * @throws IllegalStateException if classifier hasn't been trained
     */
    @Override
    public String[] getSupportedClasses() {
        requireTrained();
        return supportedClasses != null ? supportedClasses.clone() : new String[0];
    }

    /**
     * Cleans up classifier resources on container shutdown.
     * Subclasses can override to add custom cleanup, but must call super.cleanup().
     */
    @jakarta.annotation.PreDestroy
    public void cleanup() {
        getLogger().info("Cleaning up {} resources", getClass().getSimpleName());
        doClearResources();
    }

    // CONSOLIDATED WEKA TRAINING HELPERS

    /**
     * Finalizes training by storing metadata required for inference.
     *
     * @param trainingData the training Instances used for model training
     */
    protected final void finalizeTraining(Instances trainingData) {
        this.trainingDataStructure = new Instances(trainingData, 0);

        this.supportedClasses = new String[trainingData.classAttribute().numValues()];
        for (int i = 0; i < supportedClasses.length; i++) {
            supportedClasses[i] = trainingData.classAttribute().value(i);
        }

        getLogger().info("Training finalized. Model ready. Classes: {}",
                String.join(", ", supportedClasses));
    }

    /**
     * Performs the standard training pipeline orchestration.
     * <p>
     * The pipeline consists of two steps:
     * <ol>
     *   <li>Fit vectorization pipeline (preprocessing + TF-IDF feature extraction)</li>
     *   <li>Train classifier using the provided {@link ModelTrainer}</li>
     * </ol>
     * <p>
     * The return value is optional and can be ignored by subclasses.
     * Training metadata is automatically stored via {@link #finalizeTraining(Instances)}.
     *
     * @param rawDatasets raw training datasets
     * @param modelTrainer lambda that performs algorithm-specific training
     * @return trained Weka Instances (optional, can be ignored)
     * @throws Exception if any pipeline step fails
     */
    @SuppressWarnings("UnusedReturnValue") // Return value is optional for subclasses
    protected final Instances performStandardTrainingPipeline(
            List<Dataset> rawDatasets,
            ModelTrainer modelTrainer) throws Exception {

        getLogger().info("Training {} on {} raw datasets with full pipeline",
                getAlgorithmName(), rawDatasets.size());

        // Step 1: Fit FULL vectorization pipeline (preprocessing + TF-IDF)
        // WekaInstancesConverter now owns the complete text→features transformation
        getLogger().info("Step 1/2: Fitting vectorization pipeline (preprocessing + TF-IDF)");
        Instances trainingInstances = converter.fit(rawDatasets);
        getLogger().info(" Vectorization complete. Features: {}, TF-IDF vocabulary: {}",
                trainingInstances.numAttributes() - 1,
                converter.getVocabulary().size());

        // Step 2: Train classifier (algorithm-specific)
        getLogger().info("Step 2/2: Training {} classifier", getAlgorithmName());
        validateWekaTrainingData(trainingInstances);
        modelTrainer.train(trainingInstances);
        finalizeTraining(trainingInstances);

        getLogger().info(" {} training complete. Pipeline ready for inference.", getAlgorithmName());

        return trainingInstances;
    }

    /**
     * Functional interface for algorithm-specific model training.
     * <p>
     * Implementations train the classifier on prepared Weka instances.
     */
    @FunctionalInterface
    protected interface ModelTrainer {
        /**
         * Trains the model on the provided Weka instances.
         *
         * @param data prepared Weka instances for training
         * @throws Exception if training fails
         */
        void train(Instances data) throws Exception;
    }

    // CONSOLIDATED INFERENCE METHODS

    /**
     * Classifies text and returns the predicted sentiment label.
     * This method provides thread-safe inference via read lock protection.
     *
     * @param text input text to classify
     * @return predicted sentiment label
     * @throws ClassificationException if classification fails
     * @throws IllegalStateException if classifier is not trained
     * @throws IllegalArgumentException if text is null or empty
     */
    @Override
    public String classify(String text) throws ClassificationException {
        requireTrained();
        validateTextInput(text);

        try {
            return executeInference((InferenceTask<String>) () -> {
                getLogger().debug("INFERENCE: Classifying text: '{}'",
                        text.substring(0, Math.min(50, text.length())));

                Instance instance = converter.transform(text, "unknown");
                instance.setDataset(trainingDataStructure);

                // Synchronize classifier call - Weka classifiers are not thread-safe internally
                double classIndex;
                synchronized (classifierLock) {
                    classIndex = getWekaClassifierInstance().classifyInstance(instance);
                }
                String predicted = supportedClasses[(int) classIndex];

                getLogger().debug("Classification result: {}", predicted);
                return predicted;
            });
        } catch (Exception e) {
            throw ClassificationException.inferenceError(e);
        }
    }

    /**
     * Returns RAW classification probabilities for all classes.
     * This method provides thread-safe inference via read lock protection.
     * <p>
     * <b>Note:</b> Returns unsmoothed probabilities directly from the classifier.
     * Use this for evaluation and calibration analysis where exact values matter.
     * For user-facing applications, use {@link #classifyWithProbabilities(String)}
     * which applies probability smoothing to prevent exact 0/1 values.
     *
     * @param text input text to classify
     * @return raw probability distribution over all classes (may contain exact 0.0 or 1.0)
     * @throws ClassificationException if classification fails
     * @throws IllegalStateException if classifier is not trained
     * @throws IllegalArgumentException if text is null or empty
     * @see #classifyWithProbabilities(String) for smoothed probabilities
     */
    @Override
    public double[] getClassificationProbabilities(String text) throws ClassificationException {
        requireTrained();
        validateTextInput(text);

        try {
            return executeInference((InferenceTask<double[]>) () -> {
                getLogger().debug("INFERENCE: Getting probabilities for: '{}'",
                        text.substring(0, Math.min(50, text.length())));

                Instance instance = converter.transform(text, "unknown");
                instance.setDataset(trainingDataStructure);

                // Synchronize classifier call - Weka classifiers are not thread-safe internally
                double[] probs;
                synchronized (classifierLock) {
                    probs = getWekaClassifierInstance().distributionForInstance(instance);
                }
                return logProbabilityDistribution(probs);
            });
        } catch (Exception e) {
            throw ClassificationException.inferenceError(e);
        }
    }

    /**
     * Minimum probability bound to prevent exact 0.0/1.0 values.
     * This improves calibration and allows confidence thresholding to work properly.
     */
    private static final double PROB_EPSILON = 0.001;

    /**
     * Classifies text and returns both label and probabilities in a single atomic operation.
     * This avoids race conditions that can occur when calling classify() and
     * getClassificationProbabilities() separately under concurrent load.
     *
     * @param text input text to classify
     * @return ClassificationResult containing label, probabilities, and class names
     * @throws ClassificationException if classification fails
     * @throws IllegalStateException if classifier is not trained
     * @throws IllegalArgumentException if text is null or empty
     */
    @Override
    public SentimentClassifier.ClassificationResult classifyWithProbabilities(String text) throws ClassificationException {
        requireTrained();
        validateTextInput(text);

        try {
            return executeInference((InferenceTask<SentimentClassifier.ClassificationResult>) () -> {
                getLogger().debug("INFERENCE: Classifying with probabilities: '{}'",
                        text.substring(0, Math.min(50, text.length())));

                // Single instance creation and transformation
                Instance instance = converter.transform(text, "unknown");
                instance.setDataset(trainingDataStructure);

                // Synchronize classifier call - Weka classifiers are not thread-safe internally
                double[] rawProbs;
                synchronized (classifierLock) {
                    rawProbs = getWekaClassifierInstance().distributionForInstance(instance);
                }

                // Apply probability smoothing to prevent exact 0/1 values
                // This ensures confidence thresholding works and expresses inherent uncertainty
                double[] probs = smoothProbabilities(rawProbs);

                // Find the predicted class (highest probability)
                int predictedIndex = 0;
                double maxProb = probs[0];
                for (int i = 1; i < probs.length; i++) {
                    if (probs[i] > maxProb) {
                        maxProb = probs[i];
                        predictedIndex = i;
                    }
                }

                String predicted = supportedClasses[predictedIndex];
                getLogger().debug("Classification result: {} (confidence: {}, raw: {})",
                        predicted, maxProb, rawProbs[predictedIndex]);

                return new SentimentClassifier.ClassificationResult(predicted, probs, supportedClasses.clone());
            });
        } catch (Exception e) {
            throw ClassificationException.inferenceError(e);
        }
    }

    /**
     * Applies probability smoothing to prevent exact 0/1 values.
     * Clips probabilities to [epsilon, 1-epsilon] and renormalizes.
     *
     * @param probs raw probability distribution
     * @return smoothed probabilities that sum to 1.0
     */
    private double[] smoothProbabilities(double[] probs) {
        double[] smoothed = new double[probs.length];
        double sum = 0.0;

        for (int i = 0; i < probs.length; i++) {
            // Clip to [epsilon, 1-epsilon] range
            smoothed[i] = Math.max(PROB_EPSILON, Math.min(1.0 - PROB_EPSILON, probs[i]));
            sum += smoothed[i];
        }

        // Renormalize to ensure probabilities sum to 1
        if (sum > 0) {
            for (int i = 0; i < smoothed.length; i++) {
                smoothed[i] /= sum;
            }
        }

        return smoothed;
    }

    /**
     * Evaluates the classifier on test data.

     * @param testData Weka Instances to evaluate on
     * @return evaluation results with all metrics
     * @throws Exception if evaluation fails or classifier is not trained
     * @throws IllegalArgumentException if {@code testData} is {@code null} or empty
     */
    public sentiment.evaluation.ClassifierEvaluationResult evaluate(Instances testData) throws Exception {
        requireTrained();

        if (testData == null || testData.numInstances() == 0) {
            throw new IllegalArgumentException("Test data cannot be null or empty");
        }

        getLogger().info("Evaluating on {} test instances", testData.numInstances());

        return executeInference((InferenceTask<sentiment.evaluation.ClassifierEvaluationResult>) () -> {
            validateTestDataStructure(testData);
            return performEvaluation(testData);
        });
    }

    /**
     * Validates that test data structure matches training data.
     *
     * @param testData test Instances to validate
     * @throws Exception if structure mismatch is detected
     */
    protected void validateTestDataStructure(Instances testData) throws Exception {
        if (testData.numAttributes() != trainingDataStructure.numAttributes()) {
            throw new Exception(String.format(
                    "Attribute mismatch: training=%d, test=%d",
                    trainingDataStructure.numAttributes(), testData.numAttributes()));
        }

        getLogger().info("Test data validated: {} instances", testData.numInstances());
    }

    /**
     * Performs evaluation and builds the result object.
     * Common evaluation logic extracted from all classifier implementations.
     *
     * @param testData test Instances
     * @return evaluation result with all computed metrics
     * @throws Exception if evaluation fails
     */
    protected sentiment.evaluation.ClassifierEvaluationResult performEvaluation(Instances testData) throws Exception {
        long startTime = System.currentTimeMillis();

        Evaluation evaluation = new Evaluation(trainingDataStructure);
        evaluation.evaluateModel(getWekaClassifierInstance(), testData);

        long evaluationTime = System.currentTimeMillis() - startTime;

        sentiment.evaluation.ClassifierEvaluationResult result = buildEvaluationResult(
                evaluation, testData, evaluationTime);

        String accuracy = String.format("%.3f", result.getAccuracy());
        getLogger().info("Evaluation complete in {}ms: accuracy={}",
                evaluationTime, accuracy);

        return result;
    }

    /**
     * Executes an inference task using the {@link java.util.concurrent.Callable} interface.
     *
     * @param <R> return type
     * @param task the inference task to execute
     * @return task result
     * @throws Exception if task fails
     */
    public <R> R executeInference(java.util.concurrent.Callable<R> task) throws Exception {
        return executeInference((InferenceTask<R>) task::call);
    }

    // RESOURCE CLEANUP

    /**
     * Clears classifier-specific resources during reset.
     */
    @Override
    protected final void doClearResources() {
        trainingDataStructure = null;
        supportedClasses = null;
    }

    // PERSISTENCE SUPPORT

    /**
     * Returns the training data structure for persistence operations.
     *
     * @return training data structure containing schema only (no instances)
     */
    Instances getTrainingStructure() {
        return trainingDataStructure;
    }

    /**
     * Sets training metadata after loading from persistence.
     *
     * @param structure training data structure
     * @param classes supported class labels
     */
    void setTrainingMetadata(Instances structure, String[] classes) {
        this.trainingDataStructure = structure;
        this.supportedClasses = classes.clone();

        // Also set converter to READY state if it exists
        // This is critical for model loading - the converter needs to be ready for inference
        if (converter != null) {
            converter.setState(PipelineState.READY);
            // Also set the filter training structure for inference
            converter.setFilterTrainingStructure(structure);
        }
    }

    /**
     * Gets the WekaInstancesConverter for state management during model loading and feature analysis.
     *
     * @return the converter instance, or null if not set
     */
    public WekaInstancesConverter getConverter() {
        return converter;
    }

    // Lock access methods (acquireWriteLock, releaseWriteLock) inherited from TrainingTemplate base class
}