package sentiment.api;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.Map;

/**
 * Standardized error response for all REST API endpoints.
 *
 * Provides consistent error formatting across the application with support
 * for simple error messages and detailed validation errors.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class ErrorResponse {

    private String error;
    private String message;
    private Integer status;
    private Long timestamp;
    private Map<String, String> details;

    /**
     * Default constructor - sets timestamp automatically.
     */
    public ErrorResponse() {
        this.timestamp = System.currentTimeMillis();
    }

    /**
     * Simple error constructor for backward compatibility.
     *
     * @param error the error message
     */
    public ErrorResponse(String error) {
        this.error = error;
        this.timestamp = System.currentTimeMillis();
    }

    /**
     * Constructor with error and status code.
     *
     * @param error the error type/title
     * @param status the HTTP status code
     */
    public ErrorResponse(String error, int status) {
        this.error = error;
        this.status = status;
        this.timestamp = System.currentTimeMillis();
    }

    /**
     * Full constructor with error, message, and status.
     *
     * @param error the error type/title
     * @param message detailed error message
     * @param status the HTTP status code
     */
    public ErrorResponse(String error, String message, int status) {
        this.error = error;
        this.message = message;
        this.status = status;
        this.timestamp = System.currentTimeMillis();
    }

    /**
     * Constructor with validation details.
     *
     * @param error the error type/title
     * @param details map of field-level validation errors
     * @param status the HTTP status code
     */
    public ErrorResponse(String error, Map<String, String> details, int status) {
        this.error = error;
        this.details = details;
        this.status = status;
        this.timestamp = System.currentTimeMillis();
    }

    // Getters and setters

    public String getError() {
        return error;
    }

    public void setError(String error) {
        this.error = error;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public Integer getStatus() {
        return status;
    }

    public void setStatus(Integer status) {
        this.status = status;
    }

    public Long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(Long timestamp) {
        this.timestamp = timestamp;
    }

    public Map<String, String> getDetails() {
        return details;
    }

    public void setDetails(Map<String, String> details) {
        this.details = details;
    }
}
