package sentiment.models;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.bayes.NaiveBayes;
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
 * Naive Bayes sentiment classifier using Weka's NaiveBayes implementation.
 * <br>
 * <br>
 * Naive Bayes is a fast, probabilistic classifier ideal for baseline comparisons:
 * <br>
 * <br>
 * 1. Computational Efficiency: Extremely fast training and inference, making it excellent
 *    for real-time applications and large-scale text classification.
 * <br>
 * 2. Probabilistic Interpretation: Provides well-calibrated probability estimates that
 *    are interpretable and useful for confidence thresholding.
 * <br>
 * 3. Small Data Performance: Works well even with limited training data due to its
 *    strong independence assumptions, which reduce parameter space.
 * <br>
 * 4. Baseline Benchmark: Industry-standard baseline for text classification, making it
 *    valuable for comparing against more complex models.
 * <br>
 * 5. Low Memory Footprint: Requires minimal memory as it only stores feature probabilities,
 *    not entire training instances.
 * <br>
 * <br>
 * Limitations:
 * <br>
 * - Assumes features are conditionally independent given the class, which is often violated in text (e.g., "not good" vs "good").
 *  <br>
 * - May underperform on datasets where feature interactions are critical for correct classification.
 * <br>
 */
public class NaiveBayesClassifier extends ClassifierTrainingTemplate<ClassifierEvaluationResult>
        implements ClassifierEvaluator, WekaClassifier {

    private static final Logger logger = LoggerFactory.getLogger(NaiveBayesClassifier.class);

    private volatile boolean instanceStructureValidated = false;

    private NaiveBayes naiveBayes;

    private final TextPreprocessor preprocessor;
    // NOTE: converter, trainingDataStructure, supportedClasses now inherited from base class

    /**
     * Creates a new thread-safe Naive Bayes classifier with default configuration.
     */
    public NaiveBayesClassifier(TextPreprocessor preprocessor, WekaInstancesConverter converter) {
        if (preprocessor == null || converter == null) {
            throw new IllegalArgumentException("Preprocessor and converter cannot be null");
        }

        this.preprocessor = preprocessor;
        this.converter = converter;
        this.naiveBayes = new NaiveBayes();

        logger.info("Created NaiveBayesClassifier - fast probabilistic baseline");
    }

    /**
     * Creates classifier with custom NaiveBayes configuration.
     */
    public NaiveBayesClassifier(TextPreprocessor preprocessor, WekaInstancesConverter converter,
                                 NaiveBayes customNaiveBayes) {
        if (preprocessor == null || converter == null || customNaiveBayes == null) {
            throw new IllegalArgumentException("All dependencies must be non-null");
        }

        this.preprocessor = preprocessor;
        this.converter = converter;
        this.naiveBayes = customNaiveBayes;

        logger.info("Created NaiveBayesClassifier with custom configuration");
    }

    @Override
    public AlgorithmType getAlgorithmType() {
        return AlgorithmType.NAIVE_BAYES;
    }

    @Override
    public String getAlgorithmName() {
        return AlgorithmType.NAIVE_BAYES.getDisplayName();
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
        return naiveBayes;
    }

    @Override
    protected void setWekaClassifierInstance(weka.classifiers.Classifier classifier) {
        this.naiveBayes = (weka.classifiers.bayes.NaiveBayes) classifier;
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
        logger.info("Cleaning up NaiveBayesClassifier resources");
        doClearResources();
    }

    // ==================== TRAINING HELPERS ====================

    private void performModelTraining(Instances trainingData) throws Exception {
        logger.info("Training Naive Bayes model on {} instances", trainingData.numInstances());
        naiveBayes.buildClassifier(trainingData);
        logger.info("Naive Bayes model training complete");
        this.instanceStructureValidated = false;
    }

    // NOTE: classify(), getClassificationProbabilities(), evaluate() now inherited from base class

    // ==================== MODEL SUMMARY ====================

    @Override
    public String getModelSummary() {
        requireTrained();

        return String.format(
                "=== Naive Bayes Classifier Summary ===\n\n" +
                        "Algorithm: %s\n" +
                        "State: %s\n\n" +
                        "Training: %d instances, %d features\n" +
                        "Classes: %d (%s)\n" +
                        "Training Time: %dms\n" +
                        "Vocabulary: %d terms\n\n" +
                        "Performance Characteristics:\n" +
                        "  - Fast training and inference\n" +
                        "  - Probabilistic predictions\n" +
                        "  - Low memory footprint",
                AlgorithmType.NAIVE_BAYES.getDisplayName(),
                classifierState,
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
                "=== NaiveBayesClassifier Diagnostics ===\n" +
                        "NaiveBayes: %s\n" +
                        "Training: %d instances, %d features\n" +
                        "Supported classes: %s\n" +
                        "Instance validation cached: %s",
                naiveBayes != null ? "initialized" : "null",
                getTrainingInstanceCount(),
                getFeatureCount(),
                supportedClasses != null ? String.join(", ", supportedClasses) : "none",
                instanceStructureValidated
        );
    }

    // ==================== ACCESSORS ====================

    /**
     * Get the underlying Naive Bayes classifier.
     * Used for testing and advanced configuration.
     */
    public NaiveBayes getNaiveBayes() {
        return naiveBayes;
    }

    // NOTE: setNaiveBayes(), getTrainingStructure(), setTrainingMetadata() removed
    // Persistence now uses abstract methods from ClassifierTrainingTemplate

    // NOTE: getPreprocessor() implemented above to satisfy abstract method

    public WekaInstancesConverter getConverter() {
        return converter;
    }

    // ==================== WEKA CLASSIFIER INTERFACE ====================

    /**
     * Gets the underlying Weka classifier for batch operations.
     *
     * @return The NaiveBayes classifier as base Classifier type
     */
    @Override
    public weka.classifiers.Classifier getWekaClassifier() {
        return naiveBayes;
    }

    // NOTE: executeInference(Callable) now inherited from base class
}
