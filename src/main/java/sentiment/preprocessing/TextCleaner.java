package sentiment.preprocessing;

import java.util.List;

/**
 * Interface for text cleaning and basic preprocessing operations.
 *
 * This interface defines the minimal contract needed for text preprocessing
 * without coupling to the full PreprocessingPipeline lifecycle (fit/transform/save/load).
 *
 * DESIGN PATTERN: Interface Segregation Principle
 * ================================================
 * Classes that only need basic text cleaning should depend on TextCleaner,
 * not the full PreprocessingPipeline interface.
 *
 * CIRCULAR DEPENDENCY SOLUTION:
 * ==============================
 * Before:
 *   TextPreprocessor → TFIDFFeatureExtractor → PreprocessingPipeline (TextPreprocessor)
 *   ❌ CIRCULAR!
 *
 * After:
 *   TextPreprocessor implements TextCleaner
 *   TFIDFFeatureExtractor → TextCleaner (no circular dependency)
 *   ✅ CLEAN!
 *
 * @see TextPreprocessor
 * @see TFIDFFeatureExtractor
 */
public interface TextCleaner {

    /**
     * Clean raw text by removing noise, URLs, HTML tags, etc.
     *
     * This method performs basic text cleaning operations like:
     * - Removing URLs and emails
     * - Removing HTML tags
     * - Handling social media mentions and hashtags
     * - Expanding contractions
     * - Normalizing whitespace
     *
     * @param rawText The raw input text to clean
     * @return Cleaned text ready for tokenization
     */
    String cleanText(String rawText);

    /**
     * Tokenize cleaned text into individual words/tokens.
     *
     * This method splits the cleaned text into meaningful tokens,
     * handling edge cases like:
     * - Compound words
     * - Emoticons
     * - Special tokens (url_token, email_token, etc.)
     *
     * @param text The cleaned text to tokenize
     * @return List of tokens extracted from the text
     */
    List<String> tokenize(String text);

    /**
     * Remove stopwords from a list of tokens.
     *
     * Stopwords are common words that typically don't carry much
     * sentiment information (e.g., "the", "a", "is", "are").
     *
     * @param tokens The list of tokens to filter
     * @return Filtered list with stopwords removed
     */
    List<String> removeStopwords(List<String> tokens);

    /**
     * Complete text preprocessing pipeline: clean → tokenize → remove stopwords → join.
     *
     * This is a convenience method that chains all preprocessing steps together.
     * Default implementation:
     * 1. Clean the raw text
     * 2. Tokenize into words
     * 3. Remove stopwords
     * 4. Join tokens back into a single string
     *
     * Implementations can override this for custom behavior.
     *
     * @param rawText The raw input text to preprocess
     * @return Fully preprocessed text ready for feature extraction
     */
    default String preprocessText(String rawText) {
        if (rawText == null || rawText.trim().isEmpty()) {
            return "";
        }

        String cleaned = cleanText(rawText);
        List<String> tokens = tokenize(cleaned);
        List<String> filtered = removeStopwords(tokens);
        return String.join(" ", filtered);
    }
}
