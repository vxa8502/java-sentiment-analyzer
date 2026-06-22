package sentiment.api;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.Map;

/**
 * Standardized error response for all REST API endpoints.
 * <p>
 * Provides consistent error formatting across the application with support
 * for simple error messages and detailed validation errors.
 * <p>
 * Uses record pattern with static factory methods for consistency with
 * other response types (SentimentResponse, HealthResponse, etc.).
 *
 * @param error     Error type/title (e.g., "Validation failed", "Invalid input")
 * @param message   Detailed error message (may be null)
 * @param status    HTTP status code
 * @param timestamp Unix timestamp in milliseconds
 * @param details   Field-level validation errors (may be null)
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record ErrorResponse(
        String error,
        String message,
        Integer status,
        Long timestamp,
        Map<String, String> details
) {

    /**
     * Creates an error response with error title, message, and status.
     *
     * @param error   the error type/title
     * @param message detailed error message
     * @param status  the HTTP status code
     * @return ErrorResponse instance
     */
    public static ErrorResponse of(String error, String message, int status) {
        return new ErrorResponse(error, message, status, System.currentTimeMillis(), null);
    }

    /**
     * Creates an error response with just error title and status.
     *
     * @param error  the error type/title
     * @param status the HTTP status code
     * @return ErrorResponse instance
     */
    public static ErrorResponse of(String error, int status) {
        return new ErrorResponse(error, null, status, System.currentTimeMillis(), null);
    }

    /**
     * Creates an error response with validation details.
     *
     * @param error   the error type/title
     * @param details map of field-level validation errors
     * @param status  the HTTP status code
     * @return ErrorResponse instance
     */
    public static ErrorResponse withDetails(String error, Map<String, String> details, int status) {
        return new ErrorResponse(error, null, status, System.currentTimeMillis(), details);
    }

    /**
     * Creates a simple error response with just the error message.
     *
     * @param error the error message
     * @return ErrorResponse instance with 500 status
     */
    public static ErrorResponse simple(String error) {
        return new ErrorResponse(error, null, 500, System.currentTimeMillis(), null);
    }
}
