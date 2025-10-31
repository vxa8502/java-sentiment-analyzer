package sentiment.api;

/**
 * Centralized validation constants for API input validation.
 *
 * These constants ensure consistent validation rules across all endpoints:
 * - Single text analysis
 * - Batch text analysis
 * - Streaming events
 *
 * Engineering rationale:
 * - MAX_TEXT_LENGTH prevents memory exhaustion and excessive processing time
 * - MAX_BATCH_SIZE prevents overwhelming the system with too many requests
 * - MIN_TEXT_LENGTH ensures meaningful input (though more lenient)
 *
 * Production considerations:
 * - These limits can be overridden via application.properties
 * - Adjust based on your infrastructure capacity and use case
 * - Monitor processing times and adjust if needed
 */
public final class ValidationConstants {

    /**
     * Maximum length for a single text input.
     *
     * Rationale: 10,000 characters is sufficient for most use cases:
     * - Movie reviews: ~500-2000 chars
     * - Product reviews: ~200-1000 chars
     * - Social media posts: ~280-500 chars
     * - Long-form content: up to 10,000 chars
     *
     * This limit prevents:
     * - Memory exhaustion from extremely large inputs
     * - Excessive TF-IDF vectorization time
     * - Model inference timeouts
     */
    public static final int MAX_TEXT_LENGTH = 10000;

    /**
     * Minimum length for a single text input.
     *
     * Rationale: Very short texts (1-2 chars) are unlikely to be meaningful,
     * but we allow short inputs like "Good" or "Bad" which can be valid reviews.
     */
    public static final int MIN_TEXT_LENGTH = 1;

    /**
     * Maximum number of texts in a batch request.
     *
     * Rationale: 1000 texts per batch balances:
     * - Throughput: Process multiple texts efficiently
     * - Resource usage: Don't overwhelm the system
     * - Response time: Keep batch processing under reasonable limits
     */
    public static final int MAX_BATCH_SIZE = 1000;

    /**
     * Minimum number of texts in a batch request.
     */
    public static final int MIN_BATCH_SIZE = 1;

    /**
     * Error message templates for consistent error responses.
     */
    public static final String TEXT_TOO_LONG_MESSAGE =
        "Text cannot exceed " + MAX_TEXT_LENGTH + " characters";

    public static final String TEXT_TOO_SHORT_MESSAGE =
        "Text must be at least " + MIN_TEXT_LENGTH + " character";

    public static final String BATCH_TOO_LARGE_MESSAGE =
        "Cannot process more than " + MAX_BATCH_SIZE + " texts at once";

    public static final String BATCH_TOO_SMALL_MESSAGE =
        "Batch must contain at least " + MIN_BATCH_SIZE + " text";

    public static final String TEXT_BLANK_MESSAGE =
        "Text cannot be blank";

    // Private constructor to prevent instantiation
    private ValidationConstants() {
        throw new UnsupportedOperationException("Utility class cannot be instantiated");
    }
}
