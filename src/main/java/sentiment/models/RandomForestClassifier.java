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
        implements ClassifierEvaluator, WekaClassifier {

    private static final Logger logger = LoggerFactory.getLogger(RandomForestClassifier.class);

    // Default configuration optimized for text classification
    private static final int DEFAULT_NUM_TREES = 100;
    private static final int DEFAULT_MAX_DEPTH = 0; // Unlimited
    private static final int DEFAULT_NUM_FEATURES = 0; // Auto (sqrt of total features)

    private volatile boolean instanceStructureValidated = false;

    private RandomForest randomForest;

    private final TextPreprocessor preprocessor;
    // NOTE: converter, trainingDataStructure, supportedClasses now inherited from base class

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
    public String getAlgorithmName() {
        return AlgorithmType.RANDOM_FOREST.getDisplayName();
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
        return randomForest;
    }

    @Override
    protected void setWekaClassifierInstance(weka.classifiers.Classifier classifier) {
        this.randomForest = (weka.classifiers.trees.RandomForest) classifier;
    }

    @Override
    public TextPreprocessor getPreprocessor() {
        return preprocessor;
    }

    // NOTE: getTrainingInstanceCount() and getFeatureCount() now inherited from base class

    // ==================== TEMPLATE METHOD IMPLEMENTATIONS ====================

    @Override
    protected ClassifierEvaluationResult doTrain(List<Dataset> rawDatasets) throws Exception {
        if (rawDatasets == null || rawDatasets.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }

        // Use consolidated training pipeline from base class
        performStandardTrainingPipeline(rawDatasets, this::performModelTraining);

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

    private void performModelTraining(Instances trainingData) throws Exception {
        logger.info("Training Random Forest model on {} instances ({} trees)",
                trainingData.numInstances(), randomForest.getNumIterations());
        randomForest.buildClassifier(trainingData);
        logger.info("Random Forest model training complete");
        this.instanceStructureValidated = false;
    }

    // NOTE: finalizeTraining(), classify(), getClassificationProbabilities(), evaluate() now inherited from base class

    /**
     * Override to add Random Forest-specific stats (numTrees, maxDepth).
     */
    @Override
    protected Map<String, Object> buildAdditionalStats(
            Evaluation evaluation, Instances testData, long evaluationTimeMs) {

        // Get base stats
        Map<String, Object> stats = super.buildAdditionalStats(evaluation, testData, evaluationTimeMs);

        // Add Random Forest specific parameters
        stats.put("numTrees", randomForest.getNumIterations());
        stats.put("maxDepth", randomForest.getMaxDepth());

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
                getState(),
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

    /**
     * Get the underlying Random Forest classifier.
     * Used for testing and advanced configuration.
     */
    public RandomForest getRandomForest() {
        return randomForest;
    }

    // NOTE: setRandomForest(), getTrainingStructure(), setTrainingMetadata() removed
    // Persistence now uses abstract methods from ClassifierTrainingTemplate

    // NOTE: getPreprocessor() implemented above to satisfy abstract method

    public WekaInstancesConverter getConverter() {
        return converter;
    }

    // ==================== WEKA CLASSIFIER INTERFACE ====================

    /**
     * Gets the underlying Weka classifier for batch operations.
     *
     * @return The RandomForest classifier as base Classifier type
     */
    @Override
    public weka.classifiers.Classifier getWekaClassifier() {
        return randomForest;
    }

    // NOTE: executeInference(Callable) now inherited from base class
}
