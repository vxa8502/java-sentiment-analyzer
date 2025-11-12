package sentiment.models;

import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import weka.classifiers.Classifier;
import java.util.concurrent.Callable;

/**
 * Extended interface for sentiment classifiers based on Weka.
 *
 * WHY THIS EXISTS:
 * ================
 * Batch prediction operations need access to:
 * 1. Preprocessing components (for batch preprocessing)
 * 2. Feature conversion (for batch transformation)
 * 3. Underlying Weka classifier (for optimized batch inference)
 * 4. Thread-safe execution context
 *
 * This interface exposes these internals for performance-critical
 * batch operations while keeping them separate from the core
 * SentimentClassifier contract.
 *
 * IMPLEMENTING CLASSIFIERS:
 * =========================
 * All Weka-based classifiers implement this interface to support
 * optimized batch operations via BatchPredictor:
 *
 * - SVMClassifier (Support Vector Machine)
 * - NaiveBayesClassifier (Naive Bayes)
 * - LogisticRegressionClassifier (Logistic Regression)
 * - RandomForestClassifier (Random Forest)
 *
 * @see BatchPredictor for batch optimization using this interface
 */
public interface WekaClassifier extends SentimentClassifier {

    /**
     * Gets the text preprocessor used by this classifier.
     * Required for batch preprocessing operations.
     *
     * @return The text preprocessor instance
     */
    TextPreprocessor getPreprocessor();

    /**
     * Gets the Weka instances converter used by this classifier.
     * Required for batch feature transformation operations.
     *
     * @return The converter instance
     */
    WekaInstancesConverter getConverter();

    /**
     * Gets the underlying Weka classifier for direct instance classification.
     * Required for optimized batch inference.
     *
     * @return The Weka classifier (e.g., SMO, NaiveBayes, Logistic, RandomForest)
     */
    Classifier getWekaClassifier();

    /**
     * Executes an inference task in a thread-safe context.
     *
     * This method handles proper read-locking for concurrent inference
     * while ensuring no training operations are in progress.
     *
     * Uses standard Callable interface to avoid conflicts with internal
     * InferenceTask definitions in base classes.
     *
     * @param task The inference task to execute
     * @param <R> Return type of the task
     * @return The result of the inference task
     * @throws Exception if the task execution fails
     */
    <R> R executeInference(Callable<R> task) throws Exception;
}
