package sentiment.models;

import org.slf4j.Logger;
import sentiment.data.Dataset;
import sentiment.PipelineState;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import sentiment.util.ValidationUtils;
import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;

import javax.annotation.PreDestroy;
import java.util.*;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Template base class implementing common classifier training/inference logic.
 * Uses Template Method pattern - subclasses implement {@link #doTrain(List)} and {@link #doClearResources()}.
 * <p>
 * Provides thread-safe state management via ReadWriteLock:
 * <ul>
 *   <li>Training: write lock (exclusive, call ONCE at startup)</li>
 *   <li>Inference: read lock (concurrent, thread-safe after training)</li>
 *   <li>State machine: UNINITIALIZED → TRAINING → READY (or ERROR on failure)</li>
 * </ul>
 *
 * @param <T> type of training result (optional, can be {@link Void})
 */
public abstract class ClassifierTrainingTemplate<T> implements SentimentClassifier {

    // Thread-safe state management
    protected final ReadWriteLock stateLock = new ReentrantReadWriteLock();
    protected volatile PipelineState classifierState = PipelineState.UNINITIALIZED;

    // Performance tracking
    protected volatile long lastTrainingTimeMs = 0;

    // WEKA-SPECIFIC SHARED STATE
    // Subclasses that use Weka should populate these fields during training

    /**
     * Training data structure containing schema only (no instances).
     * <p>
     * Used for feature count and instance validation during inference.
     */
    protected Instances trainingDataStructure;

    /**
     * Supported class labels (e.g., {@code ["positive", "negative", "neutral"]}).
     * <p>
     * Populated during training from the Weka class attribute.
     */
    protected String[] supportedClasses;

    /**
     * Feature converter for Weka instances.
     * <p>
     * Subclasses should set this field if they use {@link WekaInstancesConverter}.
     */
    protected WekaInstancesConverter converter;

    // TEMPLATE METHOD: TRAINING PHASE

    /**
     * Trains the classifier on the provided training data.
     * NOT DESIGNED for concurrent training calls - call ONCE during initialization.
     * After successful completion, inference methods become thread-safe for concurrent access.
     * @param trainingData the training datasets
     * @throws IllegalStateException if already training or in error state
     * @throws IllegalArgumentException if {@code trainingData} is {@code null} or empty
     * @throws Exception if training fails
     */
    @Override
    public final void train(List<Dataset> trainingData) throws Exception {
        validateTrainInput(trainingData);

        stateLock.writeLock().lock(); // EXCLUSIVE ACCESS
        try {
            validateStateBeforeTraining();

            getLogger().info("Starting TRAINING phase on {} samples. State: {} -> TRAINING",
                    trainingData.size(), classifierState);
            long startTime = System.currentTimeMillis();

            // ENFORCED transition: current -> TRAINING
            transitionToState(PipelineState.TRAINING);

            try {
                // ★ SUBCLASS HOOK: Implement specific classifier training logic
                doTrain(trainingData);

                // ENFORCED transition: TRAINING -> READY (success)
                transitionToState(PipelineState.READY);

                this.lastTrainingTimeMs = System.currentTimeMillis() - startTime;
                getLogger().info("TRAINING completed successfully in {}ms. State: TRAINING -> READY. " +
                        "NOW THREAD-SAFE for concurrent inference.", lastTrainingTimeMs);

            } catch (Exception e) {
                // ENFORCED transition: TRAINING -> ERROR (failure)
                transitionToState(PipelineState.ERROR);
                getLogger().error("TRAINING failed. State: TRAINING -> ERROR", e);
                throw new Exception("Failed to train classifier: " + e.getMessage(), e);
            }

        } finally {
            stateLock.writeLock().unlock(); // Always release!
        }
    }

    // TEMPLATE METHOD: INFERENCE PHASE

    /**
     * Executes an inference task with thread-safe read lock protection.
     * <p>
     * This method allows multiple concurrent reads while ensuring no state modifications
     * occur during inference. The read lock permits concurrent execution by multiple threads.
     * @param <R> the return type of the inference task
     * @param task lambda containing inference logic
     * @return result of the inference task
     * @throws Exception if inference fails or classifier is not ready
     */
    protected final <R> R executeInference(InferenceTask<R> task) throws Exception {
        stateLock.readLock().lock(); // CONCURRENT READS ALLOWED
        try {
            validateReadyForInference();
            return task.execute();

        } catch (IllegalStateException e) {
            throw e; // Re-throw state errors
        } catch (Exception e) {
            getLogger().error("Inference failed in state {}", classifierState, e);
            throw new Exception("Failed to execute inference: " + e.getMessage(), e);
        } finally {
            stateLock.readLock().unlock(); // Always release!
        }
    }

    /**
     * Functional interface for inference tasks that may throw checked exceptions.
     * <p>
     * This interface allows lambdas to throw checked exceptions, which are then
     * propagated by {@link #executeInference(InferenceTask)}.
     *
     * @param <R> the return type of the inference task
     */
    @FunctionalInterface
    protected interface InferenceTask<R> {
        /**
         * Executes the inference task.
         *
         * @return the result of the inference
         * @throws Exception if the inference fails
         */
        R execute() throws Exception;
    }

    // ABSTRACT METHODS FOR SUBCLASSES

    /**
     * Implements algorithm-specific classifier training logic.
     * <p>
     * This method is called by {@link #train(List)} and is protected by a write lock,
     * so thread safety is not required.
     *
     * @param trainingData the training datasets
     * @return training result (may be {@code null} if no result is needed)
     * @throws Exception if training fails (will transition to ERROR state)
     */
    protected abstract T doTrain(List<Dataset> trainingData) throws Exception;

    /**
     * Implements algorithm-specific resource cleanup logic.
     * <p>
     * This method is called by {@link #reset()} and is protected by a write lock,
     * so thread safety is not required.
     */
    protected abstract void doClearResources();

    /**
     * Provides the logger instance for this classifier.
     * <p>
     * Required for consistent logging across template and subclass implementations.
     *
     * @return the logger instance
     */
    protected abstract Logger getLogger();

    /**
     * Returns the underlying Weka classifier instance.
     * <p>
     * This is used by the consolidated inference methods in the base class.
     *
     * @return the Weka classifier (e.g., NaiveBayes, SMO, Logistic, RandomForest)
     */
    protected abstract weka.classifiers.Classifier getWekaClassifierInstance();

    /**
     * Sets the underlying Weka classifier instance.
     * <p>
     * Used by persistence utilities to restore a saved model.
     *
     * @param classifier the Weka classifier to set
     */
    protected abstract void setWekaClassifierInstance(weka.classifiers.Classifier classifier);

    /**
     * Returns the text preprocessor instance.
     * <p>
     * Required for pipeline operations and {@code WekaClassifier} interface.
     *
     * @return the text preprocessor
     */
    public abstract TextPreprocessor getPreprocessor();

    // STATE MANAGEMENT

    /**
     * Enforces validated state transitions.
     * <p>
     * All state changes must go through this method to ensure valid transitions
     * according to the state machine.
     *
     * @param newState the target state
     * @throws IllegalStateException if the transition is invalid
     */
    private void transitionToState(PipelineState newState) {
        PipelineState oldState = classifierState;
        oldState.validateTransition(newState); // Throws if invalid
        classifierState = newState;
        getLogger().debug("State transition: {} -> {}", oldState, newState);
    }

    /**
     * Validates that the classifier is ready for inference operations.
     *
     * @throws IllegalStateException if not in READY state
     */
    protected void validateReadyForInference() {
        if (classifierState != PipelineState.READY) {
            throw new IllegalStateException(
                    "Cannot perform inference - not trained. Current state: " + classifierState +
                            ". Call train() with training data first.");
        }
    }

    /**
     * Validates that the current state allows training to start.
     *
     * @throws IllegalStateException if already training or trained
     */
    private void validateStateBeforeTraining() {
        if (classifierState == PipelineState.TRAINING) {
            throw new IllegalStateException(
                    "Training already in progress. Current state: " + classifierState);
        }

        if (classifierState == PipelineState.READY) {
            getLogger().warn("Classifier already trained. Call reset() before retraining.");
            throw new IllegalStateException(
                    "Already trained. Current state: " + classifierState + ". Call reset() first.");
        }
    }

    /**
     * Validates training input parameters.
     *
     * @param trainingData the training data to validate
     * @throws IllegalArgumentException if {@code trainingData} is {@code null} or empty
     */
    private void validateTrainInput(List<Dataset> trainingData) {
        if (trainingData == null || trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
    }

    // RESET AND CLEANUP

    /**
     * Resets the classifier to UNINITIALIZED state.
     * <p>
     * This method is <b>NOT thread-safe</b> and should only be called during
     * application shutdown,
     * controlled retraining scenarios,
     * error recovery.
     */
    public void reset() {
        stateLock.writeLock().lock(); // EXCLUSIVE ACCESS
        try {
            getLogger().info("Resetting classifier from state: {}", classifierState);

            // Clear subclass resources
            doClearResources();

            // Reset timing
            lastTrainingTimeMs = 0;

            // Transition to UNINITIALIZED
            if (classifierState != PipelineState.UNINITIALIZED) {
                if (classifierState == PipelineState.TRAINING) {
                    transitionToState(PipelineState.ERROR);
                }
                transitionToState(PipelineState.UNINITIALIZED);
            }

            getLogger().info("Reset complete. New state: {}", classifierState);

        } finally {
            stateLock.writeLock().unlock(); // Always release!
        }
    }

    /**
     * Performs automatic cleanup when the bean is destroyed.
     * <p>
     * Spring calls this method via {@link PreDestroy} annotation.
     */
    @SuppressWarnings("unused") // Called by Spring framework via @PreDestroy
    @PreDestroy
    public void cleanup() {
        getLogger().info("Cleaning up classifier resources");
        reset();
    }

    // READ-ONLY STATE ACCESS

    /**
     * Returns the current classifier state.
     * <p>
     * Thread-safe via volatile read.
     *
     * @return the current {@link PipelineState}
     */
    @SuppressWarnings("unused") // Public API for external consumers
    public PipelineState getClassifierState() {
        return classifierState;
    }

    /**
     * Checks if the classifier is ready for inference.
     * <p>
     * Thread-safe via volatile read.
     *
     * @return {@code true} if the classifier is trained and ready, {@code false} otherwise
     */
    @Override
    public boolean isTrained() {
        return classifierState == PipelineState.READY;
    }

    /**
     * Returns the time taken for the last training operation.
     * <p>
     * Thread-safe via volatile read.
     *
     * @return training time in milliseconds
     */
    public long getLastTrainingTimeMs() {
        return lastTrainingTimeMs;
    }

    /**
     * Returns comprehensive diagnostics information about the classifier state.
     * <p>
     * Includes current state, state description, capabilities, and training time.
     * Subclass-specific diagnostics can be added via {@link #getSubclassDiagnostics()}.
     *
     * @return formatted diagnostics string
     */
    @SuppressWarnings("unused") // Public API for monitoring and debugging
    public String getDiagnostics() {
        stateLock.readLock().lock();
        try {
            StringBuilder diag = new StringBuilder();
            diag.append("Classifier Training Template Diagnostics \n");
            diag.append(String.format("Current state: %s\n", classifierState));
            diag.append(String.format("State description: %s\n", classifierState.getDescription()));
            diag.append(String.format("Is ready: %s\n", classifierState.isReady()));
            diag.append(String.format("Can start training: %s\n", classifierState.canStartTraining()));
            diag.append(String.format("Is error: %s\n", classifierState.isError()));
            diag.append(String.format("Last training time: %d ms\n", lastTrainingTimeMs));

            // Subclass diagnostics
            String subclassDiag = getSubclassDiagnostics();
            if (subclassDiag != null && !subclassDiag.isEmpty()) {
                diag.append("\n").append(subclassDiag);
            }

            return diag.toString();
        } finally {
            stateLock.readLock().unlock();
        }
    }

    /**
     * Provides additional subclass-specific diagnostics.
     * <p>
     * Subclasses can override this method to add their own diagnostic information.
     *
     * @return subclass diagnostics string, or empty string if none
     */
    protected String getSubclassDiagnostics() {
        return "";
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
     * <p>
     * Common logging logic shared across all Weka-based classifiers.
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

    // ==================== WEKA HELPER METHODS ====================

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
     * <p>
     * This is a convenience method that centralizes the common pattern of
     * logging probability distributions in classification methods across all
     * classifier implementations.
     * <p>
     * <b>Usage in subclasses:</b>
     * <pre>
     * double[] probs = classifier.distributionForInstance(instance);
     * return logProbabilityDistribution(probs);
     * </pre>
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
     * <p>
     * Subclasses can override to add algorithm-specific statistics.
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
     * <p>
     * Subclasses can override {@link #getAlgorithmName()} to customize the algorithm name.
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

        Map<String, Object> stats = buildAdditionalStats(evaluation, testData, evaluationTimeMs);

        return new sentiment.evaluation.ClassifierEvaluationResult(
                getAlgorithmName(), accuracy,
                precision, recall, f1Score,
                macroAvgPrecision, macroAvgRecall, macroAvgF1,
                weightedPrecision, weightedRecall, weightedF1,
                confusionMatrix, supportedClasses, stats
        );
    }

    /**
     * Returns the algorithm name for evaluation results.
     * <p>
     * Subclasses should override to provide a specific algorithm name.
     *
     * @return algorithm name (default: class simple name)
     */
    public String getAlgorithmName() {
        return this.getClass().getSimpleName();
    }

    // ==================== CONSOLIDATED WEKA TRAINING HELPERS ====================

    /**
     * Finalizes training by storing metadata required for inference.
     * <p>
     * Called after successful model training to prepare the classifier for inference operations.
     * This method extracts the training data structure (schema only) and supported class labels.
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
     * Subclasses can call this method from {@link #doTrain(List)} to avoid code duplication.
     * The pipeline consists of three steps:
     * <ol>
     *   <li>Fit preprocessing pipeline (text cleaning, tokenization, etc.)</li>
     *   <li>Fit feature extraction (convert text to Weka instances)</li>
     *   <li>Train classifier using the provided {@link ModelTrainer}</li>
     * </ol>
     * <p>
     * <b>Note:</b> The return value is optional and can be ignored by subclasses.
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

        // Step 1: Fit preprocessing pipeline
        getLogger().info("Step 1/3: Fitting preprocessing pipeline");
        getPreprocessor().fit(rawDatasets);
        getLogger().info("Preprocessor fitted. Vocabulary: {}",
                getPreprocessor().getPipelineState().vocabularySize);

        // Step 2: Fit feature extraction
        getLogger().info("Step 2/3: Fitting feature extraction");
        Instances trainingInstances = converter.fit(rawDatasets);
        getLogger().info("Converter fitted. Features: {}, Vocabulary: {}",
                trainingInstances.numAttributes() - 1,
                converter.getVocabulary().size());

        // Step 3: Train classifier (algorithm-specific)
        getLogger().info("Step 3/3: Training {} classifier", getAlgorithmName());
        validateWekaTrainingData(trainingInstances);
        modelTrainer.train(trainingInstances);
        finalizeTraining(trainingInstances);

        getLogger().info("{} training complete. Pipeline ready for inference.", getAlgorithmName());

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

    // ==================== CONSOLIDATED INFERENCE METHODS ====================

    /**
     * Classifies text and returns the predicted sentiment label.
     * <p>
     * This method provides thread-safe inference via read lock protection.
     *
     * @param text input text to classify
     * @return predicted sentiment label
     * @throws Exception if classification fails or classifier is not trained
     */
    public String classify(String text) throws Exception {
        requireTrained();
        validateTextInput(text);

        return executeInference((InferenceTask<String>) () -> {
            getLogger().debug("INFERENCE: Classifying text: '{}'",
                    text.substring(0, Math.min(50, text.length())));

            Instance instance = converter.transform(text, "unknown");
            instance.setDataset(trainingDataStructure);

            double classIndex = getWekaClassifierInstance().classifyInstance(instance);
            String predicted = supportedClasses[(int) classIndex];

            getLogger().debug("Classification result: {}", predicted);
            return predicted;
        });
    }

    /**
     * Returns classification probabilities for all classes.
     * <p>
     * This method provides thread-safe inference via read lock protection.
     *
     * @param text input text to classify
     * @return probability distribution over all classes
     * @throws Exception if classification fails or classifier is not trained
     */
    public double[] getClassificationProbabilities(String text) throws Exception {
        requireTrained();
        validateTextInput(text);

        return executeInference((InferenceTask<double[]>) () -> {
            getLogger().debug("INFERENCE: Getting probabilities for: '{}'",
                    text.substring(0, Math.min(50, text.length())));

            Instance instance = converter.transform(text, "unknown");
            instance.setDataset(trainingDataStructure);

            double[] probs = getWekaClassifierInstance().distributionForInstance(instance);
            return logProbabilityDistribution(probs);
        });
    }

    /**
     * Evaluates the classifier on test data.
     * <p>
     * This method provides thread-safe evaluation via read lock protection.
     *
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
     * <p>
     * Ensures the number of attributes in the test data matches the training data.
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
     * <p>
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
     * <p>
     * This adapter method allows compatibility with code that uses {@code Callable}
     * instead of {@link InferenceTask}.
     *
     * @param <R> return type
     * @param task the inference task to execute
     * @return task result
     * @throws Exception if task fails
     */
    public <R> R executeInference(java.util.concurrent.Callable<R> task) throws Exception {
        return executeInference((InferenceTask<R>) task::call);
    }

    // ==================== PERSISTENCE SUPPORT ====================

    /**
     * Returns the training data structure for persistence operations.
     * <p>
     * Package-private to allow {@code WekaModelPersistence} access.
     *
     * @return training data structure containing schema only (no instances)
     */
    Instances getTrainingStructure() {
        return trainingDataStructure;
    }

    /**
     * Sets training metadata after loading from persistence.
     * <p>
     * Package-private to allow {@code WekaModelPersistence} access.
     *
     * @param structure training data structure
     * @param classes supported class labels
     */
    void setTrainingMetadata(Instances structure, String[] classes) {
        this.trainingDataStructure = structure;
        this.supportedClasses = classes.clone();
    }
}