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
 * Template base class for classifier training with explicit train/predict workflow.
 *
 * ✅ UPDATED: Now uses unified PipelineState instead of ClassifierState
 *
 * DESIGN PATTERN: Template Method + Strategy
 * ==========================================
 * This class implements the INVARIANT parts of classifier training:
 * - Thread-safe state management (ReadWriteLock)
 * - State machine enforcement (UNINITIALIZED -> TRAINING -> READY)
 * - Training phase protection (write lock)
 * - Inference phase concurrency (read lock)
 *
 * Subclasses implement the VARIANT parts:
 * - Specific algorithm configuration (SVM, Naive Bayes, etc.)
 * - Classifier training logic (doTrain)
 * - Classification methods (classify, getClassificationProbabilities)
 * - Resource cleanup (doClearResources)
 *
 * Optional interfaces for subclasses:
 * - Implement ClassifierEvaluator if supporting evaluation
 * - Implement ClassifierPersistence if supporting save/load
 *
 * THREAD SAFETY GUARANTEE:
 * ========================
 * After train() completes successfully:
 * ✅ Multiple threads can call inference methods concurrently
 * ✅ No synchronization overhead during inference (ReadWriteLock.readLock)
 * ✅ State mutations only during training (ReadWriteLock.writeLock)
 *
 * USAGE PATTERN:
 * ==============
 * 1. Application startup: classifier.train(trainingData) - ONCE
 * 2. Request handling: classifier.classify(text) - MANY times, concurrently
 * 3. Application shutdown: classifier.cleanup() - automatic via @PreDestroy
 *
 * STATE MACHINE:
 * ==============
 * UNINITIALIZED --train()--> TRAINING --success--> READY
 *      ^                        |                    |
 *      |                     failure              reset()
 *      |                        |                    |
 *      +--------<------------ERROR---------<---------+
 *
 * @param <T> Type of training result (optional, can be Void)
 */
public abstract class ClassifierTrainingTemplate<T> implements SentimentClassifier {

    // Thread-safe state management
    protected final ReadWriteLock stateLock = new ReentrantReadWriteLock();
    protected volatile PipelineState classifierState = PipelineState.UNINITIALIZED;

    // Performance tracking
    protected long lastTrainingTimeMs = 0;

    // ==================== WEKA-SPECIFIC SHARED STATE ====================
    // Subclasses that use Weka should populate these fields during training

    /**
     * Training data structure (empty instances with schema only).
     * Used for feature count and instance validation.
     */
    protected Instances trainingDataStructure;

    /**
     * Supported class labels (e.g., ["positive", "negative", "neutral"]).
     * Populated during training from Weka class attribute.
     */
    protected String[] supportedClasses;

    /**
     * Feature converter (Weka-specific).
     * Subclasses should set this if they use WekaInstancesConverter.
     */
    protected WekaInstancesConverter converter;

    // ==================== TEMPLATE METHOD: TRAINING PHASE ====================

    /**
     * TRAINING PHASE: Train the classifier on training data.
     *
     * This method is NOT thread-safe and should be called ONCE during initialization.
     * After successful completion, inference methods become thread-safe.
     *
     * TEMPLATE METHOD PATTERN:
     * 1. Validate input (common)
     * 2. Acquire write lock (common)
     * 3. Validate state machine (common)
     * 4. Transition to TRAINING (common)
     * 5. **Call doTrain() - SUBCLASS IMPLEMENTS THIS**
     * 6. Transition to READY (common)
     * 7. Release write lock (common)
     *
     * @param trainingData Training datasets
     * @throws IllegalStateException if already training or in error state
     * @throws IllegalArgumentException if trainingData is null/empty
     */
    @Override
    public final void train(List<Dataset> trainingData) throws Exception {
        validateTrainInput(trainingData);

        stateLock.writeLock().lock(); // ⚠️ EXCLUSIVE ACCESS
        try {
            validateStateBeforeTraining();

            getLogger().info("Starting TRAINING phase on {} samples. State: {} -> TRAINING",
                    trainingData.size(), classifierState);
            long startTime = System.currentTimeMillis();

            // ENFORCED transition: current -> TRAINING
            transitionToState(PipelineState.TRAINING);

            try {
                // ★ SUBCLASS HOOK: Implement specific classifier training logic
                T result = doTrain(trainingData);

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
            stateLock.writeLock().unlock(); // ✅ Always release
        }
    }

    // ==================== TEMPLATE METHOD: INFERENCE PHASE ====================

    /**
     * Execute inference task with thread-safe read lock.
     *
     * USAGE IN SUBCLASS:
     * <pre>
     * public String classify(String text) {
     *     return executeInference(() -> {
     *         // Your classification logic here
     *         Instance instance = convertTextToInstance(text);
     *         double result = classifier.classifyInstance(instance);
     *         return mapToLabel(result);
     *     });
     * }
     * </pre>
     *
     * @param task Lambda containing inference logic
     * @return Result of inference task
     */
    protected final <R> R executeInference(InferenceTask<R> task) throws Exception {
        stateLock.readLock().lock(); // ⚠️ CONCURRENT READS ALLOWED
        try {
            validateReadyForInference();
            return task.execute();

        } catch (IllegalStateException e) {
            throw e; // Re-throw state errors
        } catch (Exception e) {
            getLogger().error("Inference failed in state {}", classifierState, e);
            throw new Exception("Failed to execute inference: " + e.getMessage(), e);
        } finally {
            stateLock.readLock().unlock(); // ✅ Always release
        }
    }

    /**
     * Functional interface for inference tasks.
     * Allows lambdas that throw checked exceptions.
     */
    @FunctionalInterface
    protected interface InferenceTask<R> {
        R execute() throws Exception;
    }

    // ==================== ABSTRACT METHODS FOR SUBCLASSES ====================

    /**
     * Subclass implements specific classifier training logic.
     *
     * CALLED WITHIN: train() method, protected by write lock
     * THREAD SAFETY: NOT required - exclusive access guaranteed
     *
     * IMPLEMENTATION GUIDELINES:
     * 1. Configure algorithm parameters
     * 2. Build/train the classifier
     * 3. Store trained model state
     * 4. Generate training statistics
     *
     * @param trainingData Training datasets
     * @return Training result (can be null if no result needed)
     * @throws Exception if training fails (will transition to ERROR state)
     */
    protected abstract T doTrain(List<Dataset> trainingData) throws Exception;

    /**
     * Subclass implements resource cleanup logic.
     *
     * CALLED WITHIN: reset() method, protected by write lock
     * THREAD SAFETY: NOT required - exclusive access guaranteed
     *
     * IMPLEMENTATION GUIDELINES:
     * 1. Set classifier references to null
     * 2. Clear cached data structures
     * 3. Release any held resources
     */
    protected abstract void doClearResources();

    /**
     * Subclass provides its logger instance.
     * Required for consistent logging across template and subclass.
     */
    protected abstract Logger getLogger();

    /**
     * Subclass returns the underlying Weka classifier instance.
     * This is used by the consolidated inference methods in the base class.
     *
     * @return The Weka classifier (e.g., NaiveBayes, SMO, Logistic, RandomForest)
     */
    protected abstract weka.classifiers.Classifier getWekaClassifierInstance();

    /**
     * Subclass sets the underlying Weka classifier instance.
     * Used by persistence utilities to restore a saved model.
     *
     * @param classifier The Weka classifier to set
     */
    protected abstract void setWekaClassifierInstance(weka.classifiers.Classifier classifier);

    /**
     * Subclass returns the text preprocessor instance.
     * Required for pipeline operations and WekaClassifier interface.
     *
     * @return The text preprocessor
     */
    public abstract TextPreprocessor getPreprocessor();

    // ==================== STATE MANAGEMENT ====================

    /**
     * ENFORCED state transition with validation.
     * All state changes MUST go through this method.
     */
    private void transitionToState(PipelineState newState) {
        PipelineState oldState = classifierState;
        oldState.validateTransition(newState); // Throws if invalid
        classifierState = newState;
        getLogger().debug("State transition: {} -> {}", oldState, newState);
    }

    /**
     * Validate classifier is ready for inference.
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
     * Validate state allows training to start.
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
     * Validate train() input parameters.
     */
    private void validateTrainInput(List<Dataset> trainingData) {
        if (trainingData == null || trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
    }

    // ==================== RESET AND CLEANUP ====================

    /**
     * Reset classifier to UNINITIALIZED state.
     *
     * NOT thread-safe - should only be called during:
     * - Application shutdown
     * - Controlled retraining scenarios
     * - Error recovery
     */
    public void reset() {
        stateLock.writeLock().lock(); // ⚠️ EXCLUSIVE ACCESS
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
            stateLock.writeLock().unlock(); // ✅ Always release
        }
    }

    /**
     * Automatic cleanup on bean destruction.
     * Spring calls this via @PreDestroy.
     */
    @PreDestroy
    public void cleanup() {
        getLogger().info("Cleaning up classifier resources");
        reset();
    }

    // ==================== READ-ONLY STATE ACCESS ====================

    /**
     * Get current classifier state.
     * Thread-safe: volatile read
     */
    public PipelineState getClassifierState() {
        return classifierState;
    }

    /**
     * Check if classifier is ready for inference.
     * Thread-safe: volatile read
     */
    @Override
    public boolean isTrained() {
        return classifierState == PipelineState.READY;
    }

    /**
     * Get training time in milliseconds.
     * Thread-safe: volatile read
     */
    public long getLastTrainingTimeMs() {
        return lastTrainingTimeMs;
    }

    /**
     * Get comprehensive diagnostics string.
     */
    public String getDiagnostics() {
        stateLock.readLock().lock();
        try {
            StringBuilder diag = new StringBuilder();
            diag.append("=== Classifier Training Template Diagnostics ===\n");
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
     * Subclass can override to provide additional diagnostics.
     */
    protected String getSubclassDiagnostics() {
        return "";
    }

    // ==================== WEKA TRAINING DATA VALIDATION ====================

    /**
     * Validate Weka training data structure and statistics.
     *
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
     * Log Weka dataset statistics (instances, features, class distribution).
     *
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
     * Format probability distribution as human-readable string.
     *
     * @param probs Probability array (must match supportedClasses length)
     * @return Formatted string like "[positive: 0.850, negative: 0.150]"
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
     * Log probability distribution at DEBUG level and return the probabilities.
     *
     * <p>This is a convenience method that centralizes the common pattern of
     * logging probability distributions in classification methods across all
     * classifier implementations.
     *
     * <p><b>Usage in subclasses:</b>
     * <pre>
     * double[] probs = classifier.distributionForInstance(instance);
     * return logProbabilityDistribution(probs);
     * </pre>
     *
     * @param probs Probability array to log and return
     * @return The same probability array (for convenient return statements)
     */
    protected double[] logProbabilityDistribution(double[] probs) {
        if (getLogger().isDebugEnabled()) {
            getLogger().debug("Probability distribution: {}", formatProbabilities(probs));
        }
        return probs;
    }

    /**
     * Get training instance count (0 if not trained).
     */
    protected int getTrainingInstanceCount() {
        return trainingDataStructure != null ? trainingDataStructure.numInstances() : 0;
    }

    /**
     * Get feature count (0 if not trained).
     * Excludes class attribute.
     */
    protected int getFeatureCount() {
        return trainingDataStructure != null ? trainingDataStructure.numAttributes() - 1 : 0;
    }

    /**
     * Validate text input for classification methods.
     *
     * @param text Input text to validate
     * @throws IllegalArgumentException if text is null or empty
     */
    protected void validateTextInput(String text) {
        ValidationUtils.requireNonEmpty(text);
    }

    /**
     * Safely compute metric, returning 0.0 on NaN or exception.
     *
     * @param supplier Metric computation that may throw or return NaN
     * @return Metric value or 0.0 if unavailable
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
        double get() throws Exception;
    }

    /**
     * Build additional statistics map for evaluation results.
     * Subclasses can override to add algorithm-specific stats.
     *
     * @param evaluation Weka evaluation object
     * @param testData Test dataset
     * @param evaluationTimeMs Time taken for evaluation
     * @return Map of additional statistics
     */
    protected Map<String, Object> buildAdditionalStats(
            Evaluation evaluation, Instances testData, long evaluationTimeMs) {

        Map<String, Object> stats = new HashMap<>();
        stats.put("totalInstances", testData.numInstances());
        stats.put("correctlyClassified", (int) evaluation.correct());
        stats.put("incorrectlyClassified", (int) evaluation.incorrect());
        stats.put("evaluationTimeMs", evaluationTimeMs);
        stats.put("kappa", safeMetric(() -> evaluation.kappa()));

        if (lastTrainingTimeMs > 0) {
            stats.put("trainingTimeMs", lastTrainingTimeMs);
        }

        if (converter != null && converter.isReady()) {
            stats.put("vocabularySize", converter.getVocabulary().size());
        }

        return stats;
    }

    /**
     * Build ClassifierEvaluationResult from Weka Evaluation.
     * Subclasses can override getAlgorithmName() to customize the algorithm name.
     *
     * @param evaluation Weka evaluation object
     * @param testData Test dataset
     * @param evaluationTimeMs Time taken for evaluation
     * @return Complete evaluation result
     */
    protected sentiment.evaluation.ClassifierEvaluationResult buildEvaluationResult(
            Evaluation evaluation, Instances testData, long evaluationTimeMs) throws Exception {

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
     * Get algorithm name for evaluation results.
     * Subclasses should override to provide specific algorithm name.
     *
     * @return Algorithm name (default: class simple name)
     */
    public String getAlgorithmName() {
        return this.getClass().getSimpleName();
    }

    // ==================== CONSOLIDATED WEKA TRAINING HELPERS ====================

    /**
     * Finalize training by storing metadata.
     * Called after successful model training to prepare for inference.
     *
     * CONSOLIDATED: This was duplicated across all 4 classifiers.
     *
     * @param trainingData The training Instances used for model training
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
     * Perform the common training pipeline orchestration.
     * Subclasses can call this from doTrain() to avoid duplication.
     *
     * CONSOLIDATED: Extracted from duplicate code in all classifiers.
     *
     * @param rawDatasets Raw training datasets
     * @param modelTrainer Lambda that performs the algorithm-specific training
     * @return Trained Weka Instances
     */
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
     */
    @FunctionalInterface
    protected interface ModelTrainer {
        void train(Instances data) throws Exception;
    }

    // ==================== CONSOLIDATED INFERENCE METHODS ====================

    /**
     * Classify text and return predicted sentiment label.
     *
     * CONSOLIDATED: This was duplicated identically across all 4 classifiers.
     * The only difference is the Weka classifier instance, now accessed via getWekaClassifierInstance().
     *
     * @param text Input text to classify
     * @return Predicted sentiment label
     * @throws Exception if classification fails
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
     * Get classification probabilities for all classes.
     *
     * CONSOLIDATED: This was duplicated identically across all 4 classifiers.
     *
     * @param text Input text to classify
     * @return Probability distribution over classes
     * @throws Exception if classification fails
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
     * Evaluate classifier on test data.
     *
     * CONSOLIDATED: This was duplicated identically across all 4 classifiers.
     *
     * @param testData Weka Instances to evaluate on
     * @return Evaluation results
     * @throws Exception if evaluation fails
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
     * Validate test data structure matches training data.
     *
     * CONSOLIDATED: This was duplicated identically across all 4 classifiers.
     *
     * @param testData Test Instances to validate
     * @throws Exception if structure mismatch detected
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
     * Perform evaluation and build result.
     *
     * CONSOLIDATED: Common evaluation logic extracted from classifiers.
     *
     * @param testData Test Instances
     * @return Evaluation result
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
     * Execute inference task with Callable interface (for WekaClassifier).
     *
     * CONSOLIDATED: This adapter was duplicated across all 4 classifiers.
     *
     * @param task The inference task to execute
     * @param <R> Return type
     * @return Task result
     * @throws Exception if task fails
     */
    public <R> R executeInference(java.util.concurrent.Callable<R> task) throws Exception {
        return executeInference((InferenceTask<R>) () -> task.call());
    }

    // ==================== PERSISTENCE SUPPORT ====================

    /**
     * Get training data structure for persistence.
     * Package-private to allow WekaModelPersistence access.
     *
     * @return Training data structure (schema only, no instances)
     */
    Instances getTrainingStructure() {
        return trainingDataStructure;
    }

    /**
     * Set training metadata after loading from persistence.
     * Package-private to allow WekaModelPersistence access.
     *
     * @param structure Training data structure
     * @param classes Supported class labels
     */
    void setTrainingMetadata(Instances structure, String[] classes) {
        this.trainingDataStructure = structure;
        this.supportedClasses = classes.clone();
    }
}