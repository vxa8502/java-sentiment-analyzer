package sentiment.models;

import sentiment.data.Dataset;
import java.util.List;

/**
 * Core contract for sentiment classification algorithms.
 * Defines essential training, prediction, and introspection operations.
 */
public interface SentimentClassifier {

    /**
     * Trains the classifier on raw training data.
     *
     * @param trainingData Raw Dataset objects to train on
     * @throws Exception if training fails due to data issues or algorithm problems
     * @throws IllegalArgumentException if trainingData is null or empty
     */
    void train(List<Dataset> trainingData) throws Exception;

    /**
     * Classifies a single text string and returns the predicted sentiment label.
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
     * Provides confidence scores for each sentiment category, enabling
     * threshold-based filtering and uncertainty quantification.
     *
     * @param text Raw text to analyze (will be preprocessed automatically)
     * @return Array of probabilities, one per class, in the order returned by {@code getSupportedClasses()}
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
     *
     * @return Type-safe algorithm identifier
     */
    AlgorithmType getAlgorithmType();

    /**
     * Returns the algorithm name/type for this classifier (convenience method).
     *
     * @return Human-readable algorithm name
     */
    default String getAlgorithmName() {
        return getAlgorithmType().getDisplayName();
    }

    /**
     * Returns the supported sentiment classes for this classifier.
     *
     * @return Array of class labels (e.g., ["negative", "neutral", "positive"])
     * @throws IllegalStateException if classifier hasn't been trained
     */
    String[] getSupportedClasses();

    /**
     * Helper method to ensure classifier has been trained before use.
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