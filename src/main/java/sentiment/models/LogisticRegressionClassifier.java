package sentiment.models;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.functions.Logistic;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import sentiment.evaluation.ClassifierEvaluationResult;

import javax.annotation.PreDestroy;
import java.util.*;

/**
 * Logistic Regression sentiment classifier using Weka's Logistic implementation.
 *
 * <p>A fast, interpretable linear classifier that produces well-calibrated probabilities.
 * Effective as a baseline model and for applications requiring feature interpretability.
 */
public class LogisticRegressionClassifier extends ClassifierTrainingTemplate<ClassifierEvaluationResult>
        implements ClassifierEvaluator, WekaClassifier {

    private static final Logger logger = LoggerFactory.getLogger(LogisticRegressionClassifier.class);

    private Logistic logistic;

    private final TextPreprocessor preprocessor;
    // NOTE: converter, trainingDataStructure, supportedClasses now inherited from base class

    /**
     * Creates a Logistic Regression classifier with default configuration.
     *
     * @param preprocessor text preprocessor for feature extraction
     * @param converter instances converter for Weka format transformation
     * @throws IllegalArgumentException if any parameter is null
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
     *
     * @param preprocessor text preprocessor for feature extraction
     * @param converter instances converter for Weka format transformation
     * @param customLogistic pre-configured Logistic instance
     * @throws IllegalArgumentException if any parameter is null
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


    // NOTE: getTrainingInstanceCount() and getFeatureCount() now inherited from base class

    // TEMPLATE METHOD IMPLEMENTATIONS

    @Override
    protected void performAlgorithmSpecificTraining(Instances trainingData) throws Exception {
        logger.info("Training Logistic Regression model on {} instances", trainingData.numInstances());
        logistic.buildClassifier(trainingData);
        logger.info("Logistic Regression model training complete");
    }

    @PreDestroy
    public void cleanup() {
        logger.info("Cleaning up LogisticRegressionClassifier resources");
        doClearResources();
    }

    @Override
    protected Map<String, Object> buildAdditionalStats(
            Evaluation evaluation, Instances testData, long evaluationTimeMs) {

        // Get base stats
        Map<String, Object> stats = super.buildAdditionalStats(evaluation, testData, evaluationTimeMs);

        // Add Logistic Regression specific parameters
        try {
            stats.put("ridge", logistic.getRidge());
            stats.put("maxIterations", logistic.getMaxIts());
        } catch (Exception e) {
            logger.debug("Could not extract Logistic Regression parameters");
        }

        return stats;
    }

    // NOTE: finalizeTraining(), classify(), getClassificationProbabilities(), evaluate() now inherited from base class

    // MODEL SUMMARY

    @Override
    public String getModelSummary() {
        requireTrained();

        return """
                === Logistic Regression Classifier Summary ===

                Algorithm: %s
                State: %s

                Training: %d instances, %d features
                Classes: %d (%s)
                Training Time: %dms
                Vocabulary: %d terms

                Performance Characteristics:
                  - Fast training and inference
                  - Interpretable feature weights
                  - Well-calibrated probabilities""".formatted(
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
        return """
                === LogisticRegressionClassifier Diagnostics ===
                Logistic: %s
                Training: %d instances, %d features
                Supported classes: %s""".formatted(
                logistic != null ? "initialized" : "null",
                getTrainingInstanceCount(),
                getFeatureCount(),
                supportedClasses != null ? String.join(", ", supportedClasses) : "none"
        );
    }

    // ACCESSORS

    /**
     * Gets the underlying Logistic classifier.
     *
     * @return the Logistic instance
     */
    public Logistic getLogistic() {
        return logistic;
    }

    // NOTE: setLogistic(), getTrainingStructure(), setTrainingMetadata() removed
    // Persistence now uses abstract methods from ClassifierTrainingTemplate

    // WEKA CLASSIFIER INTERFACE

    /**
     * Gets the underlying Weka classifier.
     *
     * @return the Logistic classifier as base Classifier type
     */
    @Override
    public weka.classifiers.Classifier getWekaClassifier() {
        return logistic;
    }

    // NOTE: executeInference(Callable) now inherited from base class
}
