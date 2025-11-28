package sentiment.models;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import sentiment.evaluation.ClassifierEvaluationResult;

import javax.annotation.PreDestroy;
import java.util.*;

/**
 * Random Forest sentiment classifier using Weka's ensemble decision tree implementation.
 *
 * <p>Random Forest combines multiple decision trees to reduce overfitting and improve
 * accuracy over single models. Well-suited for capturing non-linear patterns and feature
 * interactions in text data.
 */
public class RandomForestClassifier extends ClassifierTrainingTemplate<ClassifierEvaluationResult>
        implements ClassifierEvaluator, WekaClassifier {

    private static final Logger logger = LoggerFactory.getLogger(RandomForestClassifier.class);

    // Default configuration optimized for text classification
    private static final int DEFAULT_NUM_TREES = 100;
    private static final int DEFAULT_MAX_DEPTH = 0; // Unlimited
    private static final int DEFAULT_NUM_FEATURES = 0; // Auto (sqrt of total features)

    private RandomForest randomForest;

    private final TextPreprocessor preprocessor;
    // NOTE: converter, trainingDataStructure, supportedClasses now inherited from base class

    /**
     * Creates Random Forest classifier with default configuration.
     * Uses 100 trees, unlimited depth, and sqrt(features) per split.
     *
     * @param preprocessor text preprocessor for cleaning and tokenization
     * @param converter feature extraction and Weka conversion
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
     * Creates classifier with custom Random Forest configuration.
     *
     * @param preprocessor text preprocessor for cleaning and tokenization
     * @param converter feature extraction and Weka conversion
     * @param customRandomForest pre-configured Random Forest instance
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

    // TEMPLATE METHOD IMPLEMENTATIONS

    @Override
    protected void performAlgorithmSpecificTraining(Instances trainingData) throws Exception {
        logger.info("Training Random Forest model on {} instances ({} trees)",
                trainingData.numInstances(), randomForest.getNumIterations());
        randomForest.buildClassifier(trainingData);
        logger.info("Random Forest model training complete");
    }

    @PreDestroy
    public void cleanup() {
        logger.info("Cleaning up RandomForestClassifier resources");
        doClearResources();
    }

    // NOTE: finalizeTraining(), classify(), getClassificationProbabilities(), evaluate() now inherited from base class

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

    // MODEL SUMMARY

    @Override
    public String getModelSummary() {
        requireTrained();

        return """
                === Random Forest Classifier Summary ===

                Algorithm: %s
                State: %s
                Trees: %d, Max Depth: %s

                Training: %d instances, %d features
                Classes: %d (%s)
                Training Time: %dms
                Vocabulary: %d terms

                Performance Characteristics:
                  - High accuracy (ensemble strength)
                  - Robust to irrelevant features
                  - Captures non-linear patterns
                """.formatted(
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
        return """
                === RandomForestClassifier Diagnostics ===
                RandomForest: %s
                Trees: %d
                Training: %d instances, %d features
                Supported classes: %s
                """.formatted(
                randomForest != null ? "initialized" : "null",
                randomForest != null ? randomForest.getNumIterations() : 0,
                getTrainingInstanceCount(),
                getFeatureCount(),
                supportedClasses != null ? String.join(", ", supportedClasses) : "none"
        );
    }

    // ACCESSORS

    // NOTE: setRandomForest(), getTrainingStructure(), setTrainingMetadata() removed
    // Persistence now uses abstract methods from ClassifierTrainingTemplate

    // NOTE: getPreprocessor() implemented above to satisfy abstract method

    /**
     * Returns the feature extractor and Weka converter.
     *
     * @return the converter instance used for feature extraction
     */
    public WekaInstancesConverter getConverter() {
        return converter;
    }

    // WEKA CLASSIFIER INTERFACE

    /**
     * Returns the underlying Weka classifier for batch operations.
     *
     * @return the Random Forest classifier as base Classifier type
     */
    @Override
    public weka.classifiers.Classifier getWekaClassifier() {
        return randomForest;
    }

    // NOTE: executeInference(Callable) now inherited from base class
}
