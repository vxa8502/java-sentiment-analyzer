package sentiment.models;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import sentiment.evaluation.ClassifierEvaluationResult;
import sentiment.data.Dataset;

import javax.annotation.PreDestroy;
import java.util.*;

/**
 * Naive Bayes sentiment classifier using Weka's NaiveBayes implementation.
 * <p>
 * A fast, probabilistic classifier ideal for baseline comparisons and real-time applications.
 * Assumes conditional independence between features (often violated in text),
 * therefore may underperform when feature interactions are critical.
 */
public class NaiveBayesClassifier extends ClassifierTrainingTemplate<ClassifierEvaluationResult>
        implements ClassifierEvaluator, WekaClassifier {

    private static final Logger logger = LoggerFactory.getLogger(NaiveBayesClassifier.class);

    private NaiveBayes naiveBayes;

    private final TextPreprocessor preprocessor;

    /**
     * Creates a Naive Bayes classifier with default configuration.
     *
     * @param preprocessor text preprocessor for feature extraction
     * @param converter instances converter for Weka format transformation
     * @throws IllegalArgumentException if any parameter is null
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
     *
     * @param preprocessor text preprocessor for feature extraction
     * @param converter instances converter for Weka format transformation
     * @param customNaiveBayes pre-configured NaiveBayes instance
     * @throws IllegalArgumentException if any parameter is null
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


    // TEMPLATE METHOD IMPLEMENTATIONS

    @Override
    protected void performAlgorithmSpecificTraining(Instances trainingData) throws Exception {
        logger.info("Training Naive Bayes model on {} instances", trainingData.numInstances());
        naiveBayes.buildClassifier(trainingData);
        logger.info("Naive Bayes model training complete");
    }

    @PreDestroy
    public void cleanup() {
        logger.info("Cleaning up NaiveBayesClassifier resources");
        doClearResources();
    }

    @Override
    protected Map<String, Object> buildAdditionalStats(
            Evaluation evaluation, Instances testData, long evaluationTimeMs) {

        // Get base stats
        Map<String, Object> stats = super.buildAdditionalStats(evaluation, testData, evaluationTimeMs);

        // Add Naive Bayes specific parameters
        stats.put("useKernelEstimator", naiveBayes.getUseKernelEstimator());
        stats.put("useSupervisedDiscretization", naiveBayes.getUseSupervisedDiscretization());

        return stats;
    }

    // MODEL SUMMARY

    @Override
    public String getModelSummary() {
        requireTrained();

        return String.format(
                """
                        Naive Bayes Classifier Summary
                        
                        Algorithm: %s
                        State: %s
                        
                        Training: %d instances, %d features
                        Classes: %d (%s)
                        Training Time: %dms
                        Vocabulary: %d terms
                        
                        Performance Characteristics:
                          - Fast training and inference
                          - Probabilistic predictions
                          - Low memory footprint""",
                AlgorithmType.NAIVE_BAYES.getDisplayName(),
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
                """
                        NaiveBayesClassifier Diagnostics
                        NaiveBayes: %s
                        Training: %d instances, %d features
                        Supported classes: %s""",
                naiveBayes != null ? "initialized" : "null",
                getTrainingInstanceCount(),
                getFeatureCount(),
                supportedClasses != null ? String.join(", ", supportedClasses) : "none"
        );
    }

    // ACCESSORS


    public WekaInstancesConverter getConverter() {
        return converter;
    }

    // WEKA CLASSIFIER INTERFACE

    /**
     * Returns the underlying Weka classifier for batch operations.
     *
     * @return the NaiveBayes classifier as base Classifier type
     */
    @Override
    public weka.classifiers.Classifier getWekaClassifier() {
        return naiveBayes;
    }

}
