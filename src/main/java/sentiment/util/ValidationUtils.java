package sentiment.util;

import java.util.Collection;

/**
 * Centralized validation utilities.
 *
 * <p>This class provides consistent validation methods for null or empty values
 * (strings and collections) across the entire application. Using these utilities ensures:
 * <ul>
 *   <li>Consistent validation logic (no code duplication)</li>
 *   <li>Uniform error messages</li>
 *   <li>Single source of truth for text validation rules</li>
 * </ul>
 *
 * <p><b>Design Pattern:</b> Static utility class (non-instantiable)
 *
 * <p><b>Usage Examples:</b>
 * <pre>
 * // Strict validation (throws exception)
 * ValidationUtils.requireNonEmpty(text, "Input text");
 * ValidationUtils.requireNonEmpty(datasets, "Training datasets");
 *
 * // Lenient validation (returns boolean)
 * if (ValidationUtils.isNullOrEmpty(text)) {
 *     return defaultValue;
 * }
 * if (ValidationUtils.isNullOrEmpty(results)) {
 *     return Collections.emptyList();
 * }
 * </pre>
 */
public final class ValidationUtils {

    /**
     * Private constructor to prevent instantiation.
     * This is a utility class with only static methods.
     */
    private ValidationUtils() {
        throw new AssertionError("ValidationUtils is a utility class and cannot be instantiated");
    }

    /**
     * Checks if a string is null or empty (after trimming whitespace).
     *
     * <p>This is the <b>lenient validation</b> pattern - returns a boolean
     * allowing callers to handle empty text gracefully.
     *
     * <p><b>Use case:</b> Preprocessing utilities that can return defaults
     * or skip processing for empty inputs.
     *
     * @param text The text to check
     * @return {@code true} if text is null or empty after trimming, {@code false} otherwise
     */
    public static boolean isNullOrEmpty(String text) {
        return text == null || text.trim().isEmpty();
    }

    /**
     * Validates that text is non-null and non-empty, throwing an exception if invalid.
     *
     * <p>This is the <b>strict validation</b> pattern - throws an exception
     * ensuring that callers must provide valid input.
     *
     * <p><b>Use case:</b> Classifiers and transformers that require valid text
     * to function correctly.
     *
     * @param text The text to validate
     * @param fieldName The name of the field being validated (for error message)
     * @throws IllegalArgumentException if text is null or empty
     */
    public static void requireNonEmpty(String text, String fieldName) {
        if (isNullOrEmpty(text)) {
            throw new IllegalArgumentException(fieldName + " cannot be null or empty");
        }
    }

    /**
     * Validates that text is non-null and non-empty, throwing an exception if invalid.
     *
     * <p>Convenience overload that uses "Text" as the default field name.
     *
     * @param text The text to validate
     * @throws IllegalArgumentException if text is null or empty
     */
    public static void requireNonEmpty(String text) {
        requireNonEmpty(text, "Text");
    }

    /**
     * Returns the trimmed text if non-empty, otherwise returns the default value.
     *
     * <p>This is a convenience method for the common pattern of providing
     * a default value when text is empty.
     *
     * @param text The text to check
     * @param defaultValue The value to return if text is null or empty
     * @return The trimmed text if non-empty, otherwise the default value
     */
    public static String orDefault(String text, String defaultValue) {
        return isNullOrEmpty(text) ? defaultValue : text.trim();
    }

    /**
     * Validates that all provided objects are non-null, throwing an exception if any is null.
     *
     * <p>This is the <b>strict validation</b> pattern for constructor dependencies.
     *
     * <p><b>Use case:</b> Validating constructor parameters in a single statement.
     *
     * <p><b>Usage Example:</b>
     * <pre>
     * public MyClass(Foo foo, Bar bar) {
     *     ValidationUtils.requireAllNonNull(
     *         new Object[]{foo, bar},
     *         new String[]{"foo", "bar"}
     *     );
     * }
     * </pre>
     *
     * @param objects Array of objects to validate
     * @param names Array of parameter names (for error messages), must match objects length
     * @throws IllegalArgumentException if any object is null or arrays have mismatched lengths
     */
    public static void requireAllNonNull(Object[] objects, String[] names) {
        if (objects.length != names.length) {
            throw new IllegalArgumentException("Objects and names arrays must have same length");
        }

        for (int i = 0; i < objects.length; i++) {
            if (objects[i] == null) {
                throw new IllegalArgumentException(names[i] + " cannot be null");
            }
        }
    }

    // ========== Collection Validation ==========

    /**
     * Checks if a collection is null or empty.
     *
     * <p>This is the <b>lenient validation</b> pattern - returns a boolean
     * allowing callers to handle empty collections gracefully.
     *
     * <p><b>Use case:</b> Data processing methods that can return early
     * or provide defaults for empty inputs.
     *
     * @param collection The collection to check
     * @return {@code true} if collection is null or empty, {@code false} otherwise
     */
    public static boolean isNullOrEmpty(Collection<?> collection) {
        return collection == null || collection.isEmpty();
    }

    /**
     * Validates that a collection is non-null and non-empty, throwing an exception if invalid.
     *
     * <p>This is the <b>strict validation</b> pattern - throws an exception
     * ensuring that callers must provide valid input.
     *
     * <p><b>Use case:</b> Training methods and data loaders that require
     * at least one element to function correctly.
     *
     * @param collection The collection to validate
     * @param fieldName The name of the field being validated (for error message)
     * @throws IllegalArgumentException if collection is null or empty
     */
    public static void requireNonEmpty(Collection<?> collection, String fieldName) {
        if (isNullOrEmpty(collection)) {
            throw new IllegalArgumentException(fieldName + " cannot be null or empty");
        }
    }
}
