package sentiment.models;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.trees.RandomForest;
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
 * Random Forest sentiment classifier using Weka's RandomForest implementation.
 *
 * MODEL SELECTION RATIONALE:
 * ==========================
 * Random Forest is a powerful ensemble method that combines multiple decision trees:
 *
 * 1. Ensemble Strength: Aggregates predictions from multiple decision trees, reducing
 *    overfitting and variance compared to single decision trees.
 *
 * 2. Feature Robustness: Handles both relevant and irrelevant features well due to
 *    random feature selection at each split, making it robust to noise.
 *
 * 3. Non-Linear Patterns: Captures complex non-linear relationships and feature
 *    interactions that linear models (SVM, Logistic Regression) might miss.
 *
 * 4. Implicit Feature Selection: Built-in feature importance ranking helps identify
 *    which words/phrases are most predictive for sentiment.
 *
 * 5. Minimal Hyperparameter Tuning: Works well with default settings, though can be
 *    tuned for optimal performance (number of trees, max depth, features per split).
 *
 * 6. Handles Imbalanced Data: Performs well even when class distributions are skewed,
 *    common in real-world sentiment datasets.
 *
 * Limitations:
 * - Higher Memory Usage: Stores multiple trees, requiring more memory than single models
 * - Slower Inference: Must aggregate predictions from all trees (typically 100+)
 * - Less Interpretable: Black-box ensemble makes it harder to explain predictions
 * - Training Time: Slower than Naive Bayes, though parallelizable
 *
 * Trade-offs vs. SVM:
 * + Better at capturing feature interactions
 * + More robust to irrelevant features
 * + Provides feature importance scores
 * - Higher memory footprint (stores full trees)
 * - Slower inference (must evaluate multiple trees)
 * - Less effective on very high-dimensional sparse data
 *
 * Trade-offs vs. Naive Bayes:
 * + Higher accuracy on complex patterns (5-10% improvement)
 * + No independence assumption violations
 * - 10-50x slower training
 * - 5-10x higher memory usage
 *
 * ARCHITECTURE:
 * =============
 * Follows the same pipeline as BasicSVMClassifier:
 * 1. Accepts raw List<Dataset> in train()
 * 2. Fits preprocessor (text cleaning, tokenization)
 * 3. Fits feature extractor (TF-IDF vectorization)
 * 4. Trains Random Forest on transformed features
 *
 * CONFIGURATION:
 * ==============
 * Default parameters optimized for text classification:
 * - Number of trees: 100 (balances accuracy and speed)
 * - Features per split: sqrt(num_features) (standard for classification)
 * - Max depth: Unlimited (trees grown until pure leaves)
 *
 * THREAD SAFETY:
 * ==============
 * - Training: Exclusive write lock (modifies model state)
 * - Inference: Concurrent read lock (thread-safe predictions)
 */
public class RandomForestClassifier extends ClassifierTrainingTemplate<ClassifierEvaluationResult>
        implements ClassifierEvaluator {

    private static final Logger logger = LoggerFactory.getLogger(RandomForestClassifier.class);

    // Default configuration optimized for text classification
    private static final int DEFAULT_NUM_TREES = 100;
    private static final int DEFAULT_MAX_DEPTH = 0; // Unlimited
    private static final int DEFAULT_NUM_FEATURES = 0; // Auto (sqrt of total features)

    private volatile boolean instanceStructureValidated = false;

    private RandomForest randomForest;

    private final TextPreprocessor preprocessor;
    private final WekaInstancesConverter converter;

    private Instances trainingDataStructure;
    private String[] supportedClasses;

    /**
     * Creates a new thread-safe Random Forest classifier with default configuration.
     * Default: 100 trees, unlimited depth, sqrt(features) per split
     */
    public RandomForestClassifier(TextPreprocessor preprocessor, WekaInstancesConverter converter) {
        if (preprocessor == null || converter == null) {
            throw new IllegalArgumentException("Preprocessor and converter cannot be null");
        }

        this.preprocessor = preprocessor;
        this.converter = converter;
        this.randomForest = new RandomForest();

        // Configure with defaults optimized for text classification
        configureDefaultParameters();

        logger.info("Created RandomForestClassifier - ensemble method with {} trees", DEFAULT_NUM_TREES);
    }

    /**
     * Creates classifier with custom RandomForest configuration.
     */
    public RandomForestClassifier(TextPreprocessor preprocessor, WekaInstancesConverter converter,
                                   RandomForest customRandomForest) {
        if (preprocessor == null || converter == null || customRandomForest == null) {
            throw new IllegalArgumentException("All dependencies must be non-null");
        }

        this.preprocessor = preprocessor;
        this.converter = converter;
        this.randomForest = customRandomForest;

        logger.info("Created RandomForestClassifier with custom configuration");
    }

    private void configureDefaultParameters() {
        randomForest.setNumIterations(DEFAULT_NUM_TREES);
        randomForest.setMaxDepth(DEFAULT_MAX_DEPTH);
        randomForest.setNumFeatures(DEFAULT_NUM_FEATURES);
    }

    @Override
    public AlgorithmType getAlgorithmType() {
        return AlgorithmType.RANDOM_FOREST;
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

    private int getTrainingInstanceCount() {
        return trainingDataStructure != null ? trainingDataStructure.numInstances() : 0;
    }

    private int getFeatureCount() {
        return trainingDataStructure != null ? trainingDataStructure.numAttributes() - 1 : 0;
    }

    // ==================== TEMPLATE METHOD IMPLEMENTATIONS ====================

    @Override
    protected ClassifierEvaluationResult doTrain(List<Dataset> rawDatasets) throws Exception {
        if (rawDatasets == null || rawDatasets.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }

        logger.info("Training Random Forest on {} raw datasets with full pipeline", rawDatasets.size());

        // Step 1: Fit preprocessing pipeline
        logger.info("Step 1/3: Fitting preprocessing pipeline");
        preprocessor.fit(rawDatasets);
        logger.info("Preprocessor fitted. Vocabulary: {}",
                preprocessor.getPipelineState().vocabularySize);

        // Step 2: Fit feature extraction
        logger.info("Step 2/3: Fitting feature extraction");
        Instances trainingInstances = converter.fit(rawDatasets);
        logger.info("Converter fitted. Features: {}, Vocabulary: {}",
                trainingInstances.numAttributes() - 1,
                converter.getVocabulary().size());

        // Step 3: Train Random Forest
        logger.info("Step 3/3: Training Random Forest classifier ({} trees)", randomForest.getNumIterations());
        validateTrainingData(trainingInstances);
        performModelTraining(trainingInstances);
        finalizeTraining(trainingInstances);

        logger.info("Random Forest training complete. Pipeline ready for inference.");

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
        logger.info("Cleaning up RandomForestClassifier resources");
        doClearResources();
    }

    // ==================== TRAINING HELPERS ====================

    private void validateTrainingData(Instances data) {
        if (data.numInstances() < 10) {
            logger.warn("Small training set ({} instances)", data.numInstances());
        }

        if (data.classIndex() == -1) {
            throw new IllegalArgumentException("Training data must have class attribute set");
        }

        if (data.classAttribute().numValues() < 2) {
            throw new IllegalArgumentException("Need at least 2 classes");
        }

        logDatasetStatistics(data);
    }

    private void logDatasetStatistics(Instances data) {
        logger.info("Dataset: {} instances, {} features, {} classes",
                data.numInstances(), data.numAttributes() - 1,
                data.classAttribute().numValues());

        int[] classCounts = new int[data.classAttribute().numValues()];
        for (int i = 0; i < data.numInstances(); i++) {
            classCounts[(int) data.instance(i).classValue()]++;
        }

        for (int i = 0; i < classCounts.length; i++) {
            String percentage = String.format("%.1f", (classCounts[i] * 100.0) / data.numInstances());
            logger.info("  {}: {} ({}%)",
                    data.classAttribute().value(i),
                    classCounts[i],
                    percentage);
        }
    }

    private void performModelTraining(Instances trainingData) throws Exception {
        logger.info("Training Random Forest model on {} instances ({} trees)",
                trainingData.numInstances(), randomForest.getNumIterations());
        randomForest.buildClassifier(trainingData);
        logger.info("Random Forest model training complete");
    }

    private void finalizeTraining(Instances trainingData) {
        this.trainingDataStructure = new Instances(trainingData, 0);

        this.supportedClasses = new String[trainingData.classAttribute().numValues()];
        for (int i = 0; i < supportedClasses.length; i++) {
            supportedClasses[i] = trainingData.classAttribute().value(i);
        }

        this.instanceStructureValidated = false;

        logger.info("Training finalized. Model ready. Classes: {}",
                String.join(", ", supportedClasses));
    }

    // ==================== CLASSIFICATION (THREAD-SAFE) ====================

    @Override
    public String classify(String text) throws Exception {
        requireTrained();

        if (text == null || text.trim().isEmpty()) {
            throw new IllegalArgumentException("Text cannot be null or empty");
        }

        return executeInference(() -> {
            logger.debug("INFERENCE: Classifying text: '{}'",
                    text.substring(0, Math.min(50, text.length())));

            Instance instance = converter.transform(text, "unknown");
            instance.setDataset(trainingDataStructure);

            double classIndex = randomForest.classifyInstance(instance);
            String predicted = supportedClasses[(int) classIndex];

            logger.debug("Classification result: {}", predicted);
            return predicted;
        });
    }

    @Override
    public double[] getClassificationProbabilities(String text) throws Exception {
        requireTrained();

        if (text == null || text.trim().isEmpty()) {
            throw new IllegalArgumentException("Text cannot be null or empty");
        }

        return executeInference(() -> {
            logger.debug("INFERENCE: Getting probabilities for: '{}'",
                    text.substring(0, Math.min(50, text.length())));

            Instance instance = converter.transform(text, "unknown");
            instance.setDataset(trainingDataStructure);

            double[] probs = randomForest.distributionForInstance(instance);
            logger.debug("Probability distribution: {}", formatProbabilities(probs));

            return probs;
        });
    }

    private String formatProbabilities(double[] probs) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < probs.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%s: %.3f", supportedClasses[i], probs[i]));
        }
        sb.append("]");
        return sb.toString();
    }

    // ==================== EVALUATION (THREAD-SAFE) ====================

    @Override
    public ClassifierEvaluationResult evaluate(Instances testData) throws Exception {
        requireTrained();

        if (testData == null || testData.numInstances() == 0) {
            throw new IllegalArgumentException("Test data cannot be null or empty");
        }

        logger.info("Evaluating on {} test instances", testData.numInstances());

        return executeInference(() -> {
            validateTestDataStructure(testData);
            return performEvaluation(testData);
        });
    }

    private void validateTestDataStructure(Instances testData) throws Exception {
        if (testData.numAttributes() != trainingDataStructure.numAttributes()) {
            throw new Exception(String.format(
                    "Attribute mismatch: training=%d, test=%d",
                    trainingDataStructure.numAttributes(), testData.numAttributes()));
        }

        logger.info("Test data validated: {} instances", testData.numInstances());
    }

    private ClassifierEvaluationResult performEvaluation(Instances testData) throws Exception {
        long startTime = System.currentTimeMillis();

        Evaluation evaluation = new Evaluation(trainingDataStructure);
        evaluation.evaluateModel(randomForest, testData);

        long evaluationTime = System.currentTimeMillis() - startTime;

        ClassifierEvaluationResult result = buildEvaluationResult(
                evaluation, testData, evaluationTime);

        String accuracy = String.format("%.3f", result.getAccuracy());
        logger.info("Evaluation complete in {}ms: accuracy={}",
                evaluationTime, accuracy);

        return result;
    }

    private ClassifierEvaluationResult buildEvaluationResult(
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

        return new ClassifierEvaluationResult(
                getAlgorithmName(), accuracy,
                precision, recall, f1Score,
                macroAvgPrecision, macroAvgRecall, macroAvgF1,
                weightedPrecision, weightedRecall, weightedF1,
                confusionMatrix, supportedClasses, stats
        );
    }

    private double safeMetric(MetricSupplier supplier) {
        try {
            double value = supplier.get();
            return Double.isNaN(value) ? 0.0 : value;
        } catch (Exception e) {
            return 0.0;
        }
    }

    @FunctionalInterface
    private interface MetricSupplier {
        double get() throws Exception;
    }

    private Map<String, Object> buildAdditionalStats(
            Evaluation evaluation, Instances testData, long evaluationTimeMs) {

        Map<String, Object> stats = new HashMap<>();
        stats.put("totalInstances", testData.numInstances());
        stats.put("correctlyClassified", (int) evaluation.correct());
        stats.put("incorrectlyClassified", (int) evaluation.incorrect());
        stats.put("evaluationTimeMs", evaluationTimeMs);
        stats.put("kappa", safeMetric(() -> evaluation.kappa()));

        // Random Forest specific parameters
        stats.put("numTrees", randomForest.getNumIterations());
        stats.put("maxDepth", randomForest.getMaxDepth());

        if (lastTrainingTimeMs > 0) {
            stats.put("trainingTimeMs", lastTrainingTimeMs);
        }

        if (converter.isReady()) {
            stats.put("vocabularySize", converter.getVocabulary().size());
        }

        return stats;
    }

    // ==================== MODEL SUMMARY ====================

    @Override
    public String getModelSummary() {
        requireTrained();

        return String.format(
                "=== Random Forest Classifier Summary ===\n\n" +
                        "Algorithm: %s\n" +
                        "State: %s\n" +
                        "Trees: %d, Max Depth: %s\n\n" +
                        "Training: %d instances, %d features\n" +
                        "Classes: %d (%s)\n" +
                        "Training Time: %dms\n" +
                        "Vocabulary: %d terms\n\n" +
                        "Performance Characteristics:\n" +
                        "  - High accuracy (ensemble strength)\n" +
                        "  - Robust to irrelevant features\n" +
                        "  - Captures non-linear patterns",
                AlgorithmType.RANDOM_FOREST.getDisplayName(),
                classifierState,
                randomForest.getNumIterations(),
                randomForest.getMaxDepth() == 0 ? "Unlimited" : String.valueOf(randomForest.getMaxDepth()),
                getTrainingInstanceCount(),
                getFeatureCount(),
                supportedClasses.length,
                String.join(", ", supportedClasses),
                lastTrainingTimeMs,
                converter.getVocabulary().size()
        );
    }

    @Override
    protected String getSubclassDiagnostics() {
        return String.format(
                "=== RandomForestClassifier Diagnostics ===\n" +
                        "RandomForest: %s\n" +
                        "Trees: %d\n" +
                        "Training: %d instances, %d features\n" +
                        "Supported classes: %s\n" +
                        "Instance validation cached: %s",
                randomForest != null ? "initialized" : "null",
                randomForest != null ? randomForest.getNumIterations() : 0,
                getTrainingInstanceCount(),
                getFeatureCount(),
                supportedClasses != null ? String.join(", ", supportedClasses) : "none",
                instanceStructureValidated
        );
    }

    // ==================== ACCESSORS ====================

    public RandomForest getRandomForest() {
        return randomForest;
    }

    void setRandomForest(RandomForest randomForest) {
        this.randomForest = randomForest;
    }

    Instances getTrainingStructure() {
        return trainingDataStructure;
    }

    void setTrainingMetadata(Instances structure, String[] classes) {
        this.trainingDataStructure = structure;
        this.supportedClasses = classes.clone();
        this.instanceStructureValidated = false;
    }

    public TextPreprocessor getPreprocessor() {
        return preprocessor;
    }

    public WekaInstancesConverter getConverter() {
        return converter;
    }
}
