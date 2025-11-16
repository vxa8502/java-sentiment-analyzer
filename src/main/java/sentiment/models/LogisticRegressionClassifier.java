package sentiment.models;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.functions.Logistic;
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
 * Logistic Regression sentiment classifier using Weka's Logistic implementation.
 *
 * MODEL SELECTION RATIONALE:
 * ==========================
 * Logistic Regression is a simple yet effective linear classifier:
 *
 * 1. Simplicity and Interpretability: Linear model with clear feature weights, making it
 *    easy to understand which words contribute to positive vs. negative sentiment.
 *
 * 2. Probabilistic Output: Produces well-calibrated probabilities via sigmoid function,
 *    ideal for confidence thresholding and risk-based decision making.
 *
 * 3. Fast Training and Inference: Efficient gradient-based optimization converges quickly,
 *    even on large datasets. Inference is a simple dot product + sigmoid.
 *
 * 4. Regularization Built-in: Ridge regression (L2 penalty) prevents overfitting and
 *    handles correlated features well, common in text data.
 *
 * 5. Baseline for Neural Networks: Logistic regression is essentially a single-layer
 *    neural network, making it a good baseline before exploring deeper architectures.
 *
 * 6. Multi-Class Support: Naturally extends to multi-class via softmax (one-vs-rest or
 *    multinomial logistic regression).
 *
 * Limitations:
 * - Linear Decision Boundary: Cannot capture complex non-linear patterns or feature
 *   interactions without manual feature engineering.
 * - Feature Independence: Like Naive Bayes, struggles with correlated features unless
 *   properly regularized.
 * - Sensitive to Feature Scaling: Requires normalized features for optimal performance
 *   (TF-IDF naturally provides this).
 *
 * Trade-offs vs. SVM:
 * + Faster training (especially on large datasets)
 * + More interpretable (direct feature weights)
 * + Better probability calibration
 * - Lower accuracy on complex patterns (2-4% typical gap)
 * - Requires feature scaling
 *
 * Trade-offs vs. Naive Bayes:
 * + Better accuracy (3-7% improvement)
 * + Handles feature correlations via regularization
 * + More reliable probability estimates
 * - 2-5x slower training
 * - Requires more careful hyperparameter tuning (ridge parameter)
 *
 * Trade-offs vs. Random Forest:
 * + Faster training and inference
 * + More interpretable (feature weights)
 * + Lower memory usage
 * - Lower accuracy on non-linear patterns
 * - Cannot capture feature interactions automatically
 *
 * ARCHITECTURE:
 * =============
 * Follows the same pipeline as BasicSVMClassifier:
 * 1. Accepts raw List<Dataset> in train()
 * 2. Fits preprocessor (text cleaning, tokenization)
 * 3. Fits feature extractor (TF-IDF vectorization)
 * 4. Trains Logistic Regression on transformed features
 *
 * CONFIGURATION:
 * ==============
 * Default parameters:
 * - Ridge parameter: Auto-selected via cross-validation
 * - Max iterations: 100 (sufficient for most text datasets)
 *
 * THREAD SAFETY:
 * ==============
 * - Training: Exclusive write lock (modifies model state)
 * - Inference: Concurrent read lock (thread-safe predictions)
 */
public class LogisticRegressionClassifier extends ClassifierTrainingTemplate<ClassifierEvaluationResult>
        implements ClassifierEvaluator, WekaClassifier {

    private static final Logger logger = LoggerFactory.getLogger(LogisticRegressionClassifier.class);

    private volatile boolean instanceStructureValidated = false;

    private Logistic logistic;

    private final TextPreprocessor preprocessor;
    // NOTE: converter, trainingDataStructure, supportedClasses now inherited from base class

    /**
     * Creates a new thread-safe Logistic Regression classifier with default configuration.
     */
    public LogisticRegressionClassifier(TextPreprocessor preprocessor, WekaInstancesConverter converter) {
        if (preprocessor == null || converter == null) {
            throw new IllegalArgumentException("Preprocessor and converter cannot be null");
        }

        this.preprocessor = preprocessor;
        this.converter = converter;
        this.logistic = new Logistic();

        logger.info("Created LogisticRegressionClassifier - interpretable linear baseline");
    }

    /**
     * Creates classifier with custom Logistic configuration.
     */
    public LogisticRegressionClassifier(TextPreprocessor preprocessor, WekaInstancesConverter converter,
                                         Logistic customLogistic) {
        if (preprocessor == null || converter == null || customLogistic == null) {
            throw new IllegalArgumentException("All dependencies must be non-null");
        }

        this.preprocessor = preprocessor;
        this.converter = converter;
        this.logistic = customLogistic;

        logger.info("Created LogisticRegressionClassifier with custom configuration");
    }

    @Override
    public AlgorithmType getAlgorithmType() {
        return AlgorithmType.LOGISTIC_REGRESSION;
    }

    @Override
    public String getAlgorithmName() {
        return AlgorithmType.LOGISTIC_REGRESSION.getDisplayName();
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
        return logistic;
    }

    @Override
    protected void setWekaClassifierInstance(weka.classifiers.Classifier classifier) {
        this.logistic = (weka.classifiers.functions.Logistic) classifier;
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
        logger.info("Cleaning up LogisticRegressionClassifier resources");
        doClearResources();
    }

    // ==================== TRAINING HELPERS ====================

    private void performModelTraining(Instances trainingData) throws Exception {
        logger.info("Training Logistic Regression model on {} instances", trainingData.numInstances());
        logistic.buildClassifier(trainingData);
        logger.info("Logistic Regression model training complete");
        this.instanceStructureValidated = false;
    }

    // NOTE: finalizeTraining(), classify(), getClassificationProbabilities(), evaluate() now inherited from base class

    // ==================== MODEL SUMMARY ====================

    @Override
    public String getModelSummary() {
        requireTrained();

        return String.format(
                "=== Logistic Regression Classifier Summary ===\n\n" +
                        "Algorithm: %s\n" +
                        "State: %s\n\n" +
                        "Training: %d instances, %d features\n" +
                        "Classes: %d (%s)\n" +
                        "Training Time: %dms\n" +
                        "Vocabulary: %d terms\n\n" +
                        "Performance Characteristics:\n" +
                        "  - Fast training and inference\n" +
                        "  - Interpretable feature weights\n" +
                        "  - Well-calibrated probabilities",
                AlgorithmType.LOGISTIC_REGRESSION.getDisplayName(),
                getState(),
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
                "=== LogisticRegressionClassifier Diagnostics ===\n" +
                        "Logistic: %s\n" +
                        "Training: %d instances, %d features\n" +
                        "Supported classes: %s\n" +
                        "Instance validation cached: %s",
                logistic != null ? "initialized" : "null",
                getTrainingInstanceCount(),
                getFeatureCount(),
                supportedClasses != null ? String.join(", ", supportedClasses) : "none",
                instanceStructureValidated
        );
    }

    // ==================== ACCESSORS ====================

    /**
     * Get the underlying Logistic Regression classifier.
     * Used for testing and advanced configuration.
     */
    public Logistic getLogistic() {
        return logistic;
    }

    // NOTE: setLogistic(), getTrainingStructure(), setTrainingMetadata() removed
    // Persistence now uses abstract methods from ClassifierTrainingTemplate

    // NOTE: getPreprocessor() implemented above to satisfy abstract method

    public WekaInstancesConverter getConverter() {
        return converter;
    }

    // ==================== WEKA CLASSIFIER INTERFACE ====================

    /**
     * Gets the underlying Weka classifier for batch operations.
     *
     * @return The Logistic classifier as base Classifier type
     */
    @Override
    public weka.classifiers.Classifier getWekaClassifier() {
        return logistic;
    }

    // NOTE: executeInference(Callable) now inherited from base class
}
