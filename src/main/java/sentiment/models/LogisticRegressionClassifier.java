package sentiment.models;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.functions.Logistic;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import sentiment.evaluation.ClassifierEvaluationResult;
import sentiment.util.ValidationUtils;

import java.util.*;

/**
 * Logistic Regression sentiment classifier using Weka's Logistic implementation.
 *
 * <p>A fast, interpretable linear classifier that produces well-calibrated probabilities.
 * Effective as a baseline model and for applications requiring feature interpretability.
 */
public class LogisticRegressionClassifier extends ClassifierTrainingTemplate<ClassifierEvaluationResult>
        implements ClassifierEvaluator {

    private static final Logger logger = LoggerFactory.getLogger(LogisticRegressionClassifier.class);

    private Logistic logistic;

    /**
     * Creates a Logistic Regression classifier with default configuration.
     *
     * @param preprocessor text preprocessor for feature extraction
     * @param converter instances converter for Weka format transformation
     * @throws IllegalArgumentException if any parameter is null
     */
    public LogisticRegressionClassifier(TextPreprocessor preprocessor, WekaInstancesConverter converter) {
        ValidationUtils.requireAllNonNull(
                new Object[]{preprocessor, converter},
                new String[]{"preprocessor", "converter"});

        this.converter = converter;
        this.logistic = new Logistic();

        // FIX: Weka's default maxIts=-1 (unlimited!) causes 13+ hour hangs
        // Set reasonable max iterations for production use
        this.logistic.setMaxIts(100);  // 100 iterations is sufficient for convergence

        // OPTIMIZATION: Use Conjugate Gradient Descent (10-50x faster than default BFGS)
        this.logistic.setUseConjugateGradientDescent(true);

        // OPTIMIZATION: Increase ridge for faster convergence (default 1.0E-8 is too small)
        this.logistic.setRidge(1.0E-6);

        logger.info("Created LogisticRegressionClassifier - optimized for large datasets (maxIts=100, CGD=true, ridge=1e-6)");
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
        ValidationUtils.requireAllNonNull(
                new Object[]{preprocessor, converter, customLogistic},
                new String[]{"preprocessor", "converter", "customLogistic"});

        this.converter = converter;
        this.logistic = customLogistic;

        logger.info("Created LogisticRegressionClassifier with custom configuration");
    }

    @Override
    public AlgorithmType getAlgorithmType() {
        return AlgorithmType.LOGISTIC_REGRESSION;
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

    @Override
    protected Map<String, Object> buildAdditionalStats(
            Evaluation evaluation, Instances testData, long evaluationTimeMs) {

        // Get base stats
        Map<String, Object> stats = super.buildAdditionalStats(evaluation, testData, evaluationTimeMs);

        // Add Logistic Regression specific parameters
        try {
            stats.put("ridge", logistic.getRidge());
            stats.put("maxIterations", logistic.getMaxIts());
            stats.put("useConjugateGradientDescent", logistic.getUseConjugateGradientDescent());
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
}
