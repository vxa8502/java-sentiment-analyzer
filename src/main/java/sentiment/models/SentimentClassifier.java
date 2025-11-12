package sentiment.models;

import sentiment.data.Dataset;
import java.util.List;

/**
 * Core contract for sentiment classification algorithms.
 *
 * DESIGN PRINCIPLE: Single Responsibility
 * ========================================
 * This interface focuses ONLY on essential training and prediction operations.
 *
 * Evaluation → ClassifierEvaluator
 * Persistence → ClassifierPersistence
 *
 * This follows Interface Segregation Principle (ISP):
 * - Clients depend only on methods they use
 * - Not all classifiers need evaluation/persistence
 * - Clear separation of concerns
 *
 * @author Sentiment Analysis Team
 * @since 2.0
 */
public interface SentimentClassifier {

    /**
     * Trains the classifier on raw training data.
     *
     * The classifier handles the complete pipeline:
     * 1. Fits preprocessing pipeline (TextPreprocessor)
     * 2. Fits feature extraction (WekaInstancesConverter)
     * 3. Trains classification model (SVM, Naive Bayes, etc.)
     *
     * @param trainingData Raw Dataset objects to train on
     * @throws Exception if training fails due to data issues or algorithm problems
     * @throws IllegalArgumentException if trainingData is null or empty
     */
    void train(List<Dataset> trainingData) throws Exception;

    /**
     * Classifies a single text string and returns the predicted sentiment label.
     *
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
     *
     * Provides confidence scores for each sentiment category, enabling
     * threshold-based filtering and uncertainty quantification.
     *
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
     *
     * PRIMARY METHOD for algorithm identification with compile-time type safety.
     * Used internally by exception handling, logging, and model comparison.
     *
     * Examples: AlgorithmType.SVM, AlgorithmType.NAIVE_BAYES, AlgorithmType.RANDOM_FOREST
     *
     * @return Type-safe algorithm identifier
     */
    AlgorithmType getAlgorithmType();

    /**
     * Returns the algorithm name/type for this classifier (convenience method).
     *
     * CONVENIENCE WRAPPER around getAlgorithmType() for backward compatibility
     * and user-facing displays. Equivalent to getAlgorithmType().getDisplayName().
     *
     * Examples: "SVM (SMO)", "Naive Bayes", "Random Forest"
     *
     * @return Human-readable algorithm name
     */
    default String getAlgorithmName() {
        return getAlgorithmType().getDisplayName();
    }

    /**
     * Returns the supported sentiment classes for this classifier.
     *
     * Useful for validation and UI generation. Order should match
     * the probability array returned by getClassificationProbabilities().
     *
     * @return Array of class labels (e.g., ["negative", "positive"])
     * @throws IllegalStateException if classifier hasn't been trained
     */
    String[] getSupportedClasses();

    /**
     * Helper method to ensure classifier has been trained before use.
     *
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