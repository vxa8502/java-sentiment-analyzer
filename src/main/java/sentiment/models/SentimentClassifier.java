package sentiment.models;

import sentiment.data.Dataset;
import java.util.List;

/**
 * Core contract for sentiment classification algorithms.
 * <br>
 * This interface focuses ONLY on essential training and prediction operations.
 * <br>
 * Evaluation → ClassifierEvaluator
 * <br>
 * Persistence → WekaModelPersistence
 * <br>
 * This follows Interface Segregation Principle (ISP):
 * <br>
 * - Smaller, specific interfaces instead of a large comprehensive one
 * <br>
 * - Clients depend only on methods they use
 */
public interface SentimentClassifier {

    /**
     * Trains the classifier on raw training data.
     * <p>
     * The classifier handles the complete pipeline:
     * <ol>
     *   <li>Fits preprocessing pipeline (TextPreprocessor)</li>
     *   <li>Fits feature extraction (WekaInstancesConverter)</li>
     *   <li>Trains classification model (SVM, Naive Bayes, etc.)</li>
     * </ol>
     *
     * @param trainingData Raw Dataset objects to train on
     * @throws Exception if training fails due to data issues or algorithm problems
     * @throws IllegalArgumentException if trainingData is null or empty
     */
    void train(List<Dataset> trainingData) throws Exception;

    /**
     * Classifies a single text string and returns the predicted sentiment label.
     * <br>
     * The input text will be automatically preprocessed using the same pipeline
     * as training data.
     *
     * @param text Raw text to classify (will be preprocessed automatically)
     * @return Predicted sentiment label (e.g., "positive", "negative", "neutral")
     * @throws Exception if classification fails or preprocessing errors occur
     * @throws IllegalStateException if classifier hasn't been trained
     * @throws IllegalArgumentException if text is null or empty
     */
    String classify(String text) throws Exception;

    /**
     * Returns classification probabilities for all possible sentiment classes.
     * <br>
     * Provides confidence scores for each sentiment category, enabling
     * threshold-based filtering and uncertainty quantification.
     * <br>
     * Array format: [negative_prob, positive_prob] for binary classification
     *              [negative_prob, neutral_prob, positive_prob] for 3-class
     *
     * @param text Raw text to analyze (will be preprocessed automatically)
     * @return Array of probabilities, one per class, in class attribute order
     * @throws Exception if classification fails or preprocessing errors occur
     * @throws IllegalStateException if classifier hasn't been trained
     * @throws IllegalArgumentException if text is null or empty
     */
    double[] getClassificationProbabilities(String text) throws Exception;

    /**
     * Checks if the classifier has been trained and is ready for predictions.
     *
     * @return true if classifier is trained and ready, false otherwise
     */
    boolean isTrained();

    /**
     * Returns the algorithm type for this classifier (type-safe enum).
     * <br>
     * PRIMARY METHOD for algorithm identification with compile-time type safety.
     * Used internally by exception handling, logging, and model comparison.
     * <br>
     * Examples: AlgorithmType.SVM, AlgorithmType.NAIVE_BAYES, AlgorithmType.RANDOM_FOREST
     *
     * @return Type-safe algorithm identifier
     */
    AlgorithmType getAlgorithmType();

    /**
     * Returns the algorithm name/type for this classifier (convenience method).
     * <br>
     * CONVENIENCE WRAPPER around getAlgorithmType() for backward compatibility
     * and user-facing displays. Equivalent to getAlgorithmType().getDisplayName().
     *<br>
     * Examples: "SVM (SMO)", "Naive Bayes", "Random Forest"
     * <br>
     * @return Human-readable algorithm name
     */
    default String getAlgorithmName() {
        return getAlgorithmType().getDisplayName();
    }

    /**
     * Returns the supported sentiment classes for this classifier.
     * <br>
     * Useful for validation and UI generation. Order should match
     * the probability array returned by getClassificationProbabilities().
     *
     * @return Array of class labels (e.g., ["negative", "positive"])
     * @throws IllegalStateException if classifier hasn't been trained
     */
    String[] getSupportedClasses();

    /**
     * Helper method to ensure classifier has been trained before use.
     * <p>
     * Should be called by all methods that require a trained model.
     * Provides consistent error messaging across implementations.
     *
     * @throws IllegalStateException if classifier hasn't been trained yet
     */
    default void requireTrained() {
        if (!isTrained()) {
            throw new IllegalStateException(
                    "Classifier must be trained before use. Call train() first."
            );
        }
    }
}