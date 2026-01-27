package sentiment.models;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import sentiment.evaluation.ClassifierEvaluationResult;
import sentiment.util.ValidationUtils;

import java.util.*;

/**
 * Random Forest sentiment classifier using Weka's ensemble decision tree implementation.
 *
 * <p>Random Forest combines multiple decision trees to reduce overfitting and improve
 * accuracy over single models. Well-suited for capturing non-linear patterns and feature
 * interactions in text data.
 */
public class RandomForestClassifier extends ClassifierTrainingTemplate<ClassifierEvaluationResult>
        implements ClassifierEvaluator {

    private static final Logger logger = LoggerFactory.getLogger(RandomForestClassifier.class);

    // Default configuration optimized for text classification
    private static final int DEFAULT_NUM_TREES = 100;
    private static final int DEFAULT_MAX_DEPTH = 0; // Unlimited
    private static final int DEFAULT_NUM_FEATURES = 0; // Auto (sqrt of total features)

    private RandomForest randomForest;

    /**
     * Creates Random Forest classifier with default configuration.
     * Uses 100 trees, unlimited depth, and sqrt(features) per split.
     *
     * @param preprocessor text preprocessor for cleaning and tokenization
     * @param converter feature extraction and Weka conversion
     */
    public RandomForestClassifier(TextPreprocessor preprocessor, WekaInstancesConverter converter) {
        ValidationUtils.requireAllNonNull(
                new Object[]{preprocessor, converter},
                new String[]{"preprocessor", "converter"});

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
        ValidationUtils.requireAllNonNull(
                new Object[]{preprocessor, converter, customRandomForest},
                new String[]{"preprocessor", "converter", "customRandomForest"});

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


    // NOTE: getTrainingInstanceCount() and getFeatureCount() now inherited from base class

    // TEMPLATE METHOD IMPLEMENTATIONS

    @Override
    protected void performAlgorithmSpecificTraining(Instances trainingData) throws Exception {
        logger.info("Training Random Forest model on {} instances ({} trees)",
                trainingData.numInstances(), randomForest.getNumIterations());
        randomForest.buildClassifier(trainingData);
        logger.info("Random Forest model training complete");
    }

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

}
