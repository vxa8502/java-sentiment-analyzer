package sentiment.preprocessing;

import sentiment.data.Dataset;
import java.nio.file.Path;
import java.util.List;

/**
 * Enhanced preprocessing pipeline interface for sentiment analysis.
 * Defines the contract for preprocessing text with stateful training and persistence.
 */
public interface PreprocessingPipeline {

    /**
     * Get the version of this preprocessing pipeline
     * @return Version string (e.g., "1.0.0")
     */
    String getVersion();

    /**
     * Fit the preprocessing pipeline on training data.
     * This trains any stateful components (e.g., vocabulary, feature extractors).
     *
     * @param data Training dataset to fit the pipeline on
     * @throws IllegalArgumentException if data is null or empty
     * @throws IllegalStateException if pipeline is already fitted
     */
    void fit(List<Dataset> data);

    /**
     * Transform a single text through the preprocessing pipeline.
     * Pipeline must be fitted before calling this method.
     *
     * @param text The input text to transform
     * @return Preprocessed text ready for model input
     * @throws IllegalStateException if pipeline not fitted
     * @throws IllegalArgumentException if text is null or empty
     */
    String transform(String text);

    /**
     * Save the current pipeline state to disk for later reuse.
     * Persists fitted vocabularies, filters, and configuration.
     *
     * @param path File path to save the pipeline state
     * @throws IllegalStateException if pipeline not fitted
     * @throws java.io.IOException if saving fails
     */
    void saveState(Path path) throws java.io.IOException;

    /**
     * Load a previously saved pipeline state from disk.
     * Restores fitted vocabularies, filters, and configuration.
     *
     * @param path File path to load the pipeline state from
     * @throws java.io.IOException if loading fails
     * @throws IllegalArgumentException if path is invalid
     */
    void loadState(Path path) throws java.io.IOException;

    // Legacy methods for backward compatibility

    /**
     * Preprocess raw text through the complete pipeline.
     * For compatibility with existing implementations.
     *
     * @param rawText The input text to preprocess
     * @return Preprocessed text ready for feature extraction
     */
    default String preprocessText(String rawText) {
        return transform(rawText);
    }

    /**
     * Clean text by removing noise, HTML, URLs, etc.
     * For compatibility with existing implementations.
     *
     * @param rawText The input text to clean
     * @return Cleaned text
     */
    default String cleanText(String rawText) {
        // Default implementation delegates to transform
        return transform(rawText);
    }
}