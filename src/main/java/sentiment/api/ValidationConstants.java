package sentiment.api;

/**
 * Centralized validation constants for API input validation.
 * These constants ensure consistent validation rules across all endpoints:
 */
public final class ValidationConstants {

    /**
     * Maximum length for a single text input.
     */
    public static final int MAX_TEXT_LENGTH = 10000;

    /**
     * Minimum length for a single text input.
     */
    public static final int MIN_TEXT_LENGTH = 1;

    /**
     * Maximum number of texts in a batch request.
     */
    public static final int MAX_BATCH_SIZE = 100;

    /**
     * Minimum number of texts in a batch request.
     */
    public static final int MIN_BATCH_SIZE = 1;

    /**
     * Error message templates for consistent error responses.
     */
    public static final String TEXT_TOO_LONG_MESSAGE =
        "Text cannot exceed " + MAX_TEXT_LENGTH + " characters";

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
