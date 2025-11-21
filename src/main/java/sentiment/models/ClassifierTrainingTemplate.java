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
 * <p>
 * Extends {@link TrainingTemplate} to provide unified state management and lifecycle control.
 * Uses Template Method pattern - subclasses implement {@link #doTrain(List)} and {@link #doClearResources()}.
 *
 * <p>Adds classifier-specific features:
 * <ul>
 *   <li>Weka integration ({@link weka.classifiers.Classifier})</li>
 *   <li>Training data structure management</li>
 *   <li>Class label support</li>
 *   <li>Feature extraction pipeline</li>
 *   <li>Evaluation and metrics</li>
 *   <li>Model persistence support</li>
 * </ul>
 *
 * @param <T> type of training result (optional, can be {@link Void})
 */
public abstract class ClassifierTrainingTemplate<T> extends TrainingTemplate<T> implements SentimentClassifier {

    // WEKA-SPECIFIC SHARED STATE
    // Subclasses that use Weka should populate these fields during training

    /**
     * Training data structure containing schema only (no instances).
     * Used for feature count and instance validation during inference.
     */
    protected Instances trainingDataStructure;

    /**
     * Supported class labels (e.g., {@code ["positive", "negative", "neutral"]}).
     * Populated during training from the Weka class attribute.
     */
    protected String[] supportedClasses;

    /**
     * Feature converter for Weka instances.
     * Subclasses should set this field if they use {@link WekaInstancesConverter}.
     */
    protected WekaInstancesConverter converter;

    // TEMPLATE METHOD: TRAINING PHASE

    /**
     * Trains the classifier on the provided training data.
     * <p>
     * Implements {@link SentimentClassifier#train(List)} by delegating to
     * {@link TrainingTemplate#trainInternal(List)}.
     *
     * @param trainingData the training datasets
     * @throws Exception if training fails
     */
    @Override
    public void train(List<Dataset> trainingData) throws Exception {
        trainInternal(trainingData);
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
     * <p>
     * Convenience method that delegates to {@link #getState()}.
     * Thread-safe via volatile read.
     *
     * @return the current {@link PipelineState}
     */
    @SuppressWarnings("unused") // Public API for external consumers
    public PipelineState getClassifierState() {
        return getState();
    }

    /**
     * Checks if the classifier is ready for inference.
     * <p>
     * Implements {@link SentimentClassifier#isTrained()}.
     * Thread-safe via volatile read.
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
     * <p>
     * This is a convenience method that centralizes the common pattern of
     * logging probability distributions in classification methods across all
     * classifier implementations.
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
     * Returns the algorithm name for evaluation results.
     * <p>
     * Subclasses should override to provide a specific algorithm name.
     *
     * @return algorithm name (default: class simple name)
     */
    public String getAlgorithmName() {
        return this.getClass().getSimpleName();
    }

    // CONSOLIDATED WEKA TRAINING HELPERS

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

    // CONSOLIDATED INFERENCE METHODS

    /**
     * Classifies text and returns the predicted sentiment label.
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

    // RESOURCE CLEANUP

    /**
     * Clears classifier-specific resources during reset.
     * <p>
     * Consolidated implementation that clears Weka-specific state managed by this template.
     * Final to ensure consistent cleanup behavior across all classifier implementations.
     */
    @Override
    protected final void doClearResources() {
        trainingDataStructure = null;
        supportedClasses = null;
    }

    // PERSISTENCE SUPPORT

    /**
     * Returns the training data structure for persistence operations.
     * Package-private to allow {@code WekaModelPersistence} access.
     *
     * @return training data structure containing schema only (no instances)
     */
    Instances getTrainingStructure() {
        return trainingDataStructure;
    }

    /**
     * Sets training metadata after loading from persistence.
     * Package-private to allow {@code WekaModelPersistence} access.
     *
     * @param structure training data structure
     * @param classes supported class labels
     */
    void setTrainingMetadata(Instances structure, String[] classes) {
        this.trainingDataStructure = structure;
        this.supportedClasses = classes.clone();
    }

    // Lock access methods (acquireWriteLock, releaseWriteLock) inherited from TrainingTemplate base class
}