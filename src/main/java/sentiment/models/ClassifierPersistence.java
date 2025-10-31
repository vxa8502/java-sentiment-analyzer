package sentiment.models;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Interface for model persistence operations (save/load).
 *
 * DESIGN PRINCIPLE: Interface Segregation
 * ========================================
 * Separated from core SentimentClassifier to follow ISP:
 * - Not all classifiers need persistence (e.g., demo/testing classifiers)
 * - Some deployment scenarios use in-memory models only
 * - Allows stateless classifier implementations
 *
 * Implement this interface if your classifier supports:
 * - Saving trained models to disk
 * - Loading pre-trained models
 * - Model versioning and serialization
 *
 * @author Sentiment Analysis Team
 * @since 2.0
 */
public interface ClassifierPersistence {

    /**
     * Saves trained model to disk for later reuse.
     *
     * Serializes the trained classifier, training structure, and metadata
     * to the specified file path. Enables model reuse without retraining.
     *
     * Implementation Notes:
     * - Should save all state needed to restore the model
     * - Include version information for compatibility checking
     * - Consider compression for large models
     * - Validate file path and permissions before writing
     *
     * Typical Usage:
     * <pre>
     * classifier.train(trainingData);
     * classifier.saveModel(Path.of("models/sentiment_svm_v1.model"));
     * </pre>
     *
     * @param modelPath Path where model should be saved
     * @throws IOException if file cannot be written or serialization fails
     * @throws IllegalStateException if classifier hasn't been trained
     */
    void saveModel(Path modelPath) throws IOException;

    /**
     * Loads previously trained model from disk.
     *
     * Deserializes a saved model and restores the classifier to trained state.
     * After loading, the classifier can immediately perform predictions without
     * retraining.
     *
     * Implementation Notes:
     * - Should validate model version compatibility
     * - Verify all required dependencies are available
     * - Transition classifier to READY state after successful load
     * - Consider backward compatibility with older model versions
     *
     * Typical Usage:
     * <pre>
     * SentimentClassifier classifier = new SVMClassifier(...);
     * classifier.loadModel(Path.of("models/sentiment_svm_v1.model"));
     * String result = classifier.classify("This movie is great!");
     * </pre>
     *
     * @param modelPath Path to saved model file
     * @throws IOException if file cannot be read or deserialization fails
     * @throws ClassNotFoundException if model contains unknown classes
     * @throws IllegalArgumentException if model file doesn't exist
     */
    void loadModel(Path modelPath) throws IOException, ClassNotFoundException;
}