package sentiment.api;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.validation.FieldError;
import org.springframework.web.HttpMediaTypeNotSupportedException;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.servlet.NoHandlerFoundException;
import org.springframework.web.servlet.resource.NoResourceFoundException;

import java.util.HashMap;
import java.util.Map;

/**
 * Global exception handler for REST API endpoints.
 * Catches exceptions thrown by controllers and converts them to
 * appropriate HTTP responses with proper status codes and error messages.
 */
@ControllerAdvice
public class RestApiExceptionHandler {

    private static final Logger logger = LoggerFactory.getLogger(RestApiExceptionHandler.class);

    // HELPER METHODS

    /**
     * Create a standardized ErrorResponse with automatic timestamp.
     *
     * @param error Error title/type
     * @param message Detailed error message
     * @param status HTTP status
     * @return ErrorResponse object
     */
    private ErrorResponse buildErrorResponse(String error, String message, HttpStatus status) {
        return new ErrorResponse(error, message, status.value());
    }

    /**
     * Log error at appropriate level and create ResponseEntity.
     *
     * @param logPrefix Log message prefix (e.g., "Model not trained")
     * @param ex The exception being handled
     * @param errorTitle Error title for response
     * @param errorMessage Error message for response
     * @param status HTTP status
     * @return ResponseEntity with ErrorResponse body
     */
    private ResponseEntity<ErrorResponse> logAndBuildResponse(
            String logPrefix,
            Exception ex,
            String errorTitle,
            String errorMessage,
            HttpStatus status) {

        // Log at appropriate level based on status code
        if (status.is4xxClientError()) {
            logger.info("{}: {}", logPrefix, ex.getMessage());
        } else {
            logger.error("{}: {}", logPrefix, ex.getMessage(), ex);
        }

        ErrorResponse response = buildErrorResponse(errorTitle, errorMessage, status);
        return ResponseEntity.status(status).body(response);
    }

    /**
     * Convenience overload that uses exception message as error message.
     */
    private ResponseEntity<ErrorResponse> logAndBuildResponse(
            String logPrefix,
            Exception ex,
            String errorTitle,
            HttpStatus status) {
        return logAndBuildResponse(logPrefix, ex, errorTitle, ex.getMessage(), status);
    }

    // EXCEPTION HANDLERS

    /**
     * Handles malformed JSON in request body.
     * Returns 400 Bad Request instead of 500 Internal Server Error.
     */
    @ExceptionHandler(HttpMessageNotReadableException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public ResponseEntity<ErrorResponse> handleMalformedJson(
            HttpMessageNotReadableException ex) {

        String message = "Malformed JSON request";
        Throwable cause = ex.getCause();
        if (cause != null) {
            String causeMessage = cause.getMessage();
            if (causeMessage != null && causeMessage.length() < 200) {
                message = "Invalid JSON: " + causeMessage.split("\n")[0];
            }
        }

        logger.info("Malformed JSON request: {}", ex.getMessage());

        ErrorResponse response = buildErrorResponse(
                "Invalid request body",
                message,
                HttpStatus.BAD_REQUEST
        );
        return ResponseEntity.badRequest().body(response);
    }

    /**
     * Handles missing or unsupported Content-Type header.
     * Returns 415 Unsupported Media Type.
     */
    @ExceptionHandler(HttpMediaTypeNotSupportedException.class)
    @ResponseStatus(HttpStatus.UNSUPPORTED_MEDIA_TYPE)
    public ResponseEntity<ErrorResponse> handleUnsupportedMediaType(
            HttpMediaTypeNotSupportedException ex) {

        logger.info("Unsupported media type: {}", ex.getMessage());

        ErrorResponse response = buildErrorResponse(
                "Unsupported media type",
                "Content-Type must be application/json",
                HttpStatus.UNSUPPORTED_MEDIA_TYPE
        );
        return ResponseEntity.status(HttpStatus.UNSUPPORTED_MEDIA_TYPE).body(response);
    }

    /**
     * Handles wrong HTTP method (e.g., GET on a POST-only endpoint).
     * Returns 405 Method Not Allowed.
     */
    @ExceptionHandler(HttpRequestMethodNotSupportedException.class)
    @ResponseStatus(HttpStatus.METHOD_NOT_ALLOWED)
    public ResponseEntity<ErrorResponse> handleMethodNotAllowed(
            HttpRequestMethodNotSupportedException ex) {

        String supportedMethods = ex.getSupportedHttpMethods() != null
                ? ex.getSupportedHttpMethods().toString()
                : "unknown";

        logger.info("Method not allowed: {} (supported: {})", ex.getMethod(), supportedMethods);

        ErrorResponse response = buildErrorResponse(
                "Method not allowed",
                String.format("HTTP method %s is not supported for this endpoint. Supported: %s",
                        ex.getMethod(), supportedMethods),
                HttpStatus.METHOD_NOT_ALLOWED
        );
        return ResponseEntity.status(HttpStatus.METHOD_NOT_ALLOWED).body(response);
    }

    /**
     * Handles validation errors from @Valid annotations.
     */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public ResponseEntity<ErrorResponse> handleValidationExceptions(
            MethodArgumentNotValidException ex) {

        Map<String, String> errors = new HashMap<>();
        ex.getBindingResult().getAllErrors().forEach(error -> {
            String fieldName = ((FieldError) error).getField();
            String errorMessage = error.getDefaultMessage();
            errors.put(fieldName, errorMessage);
        });

        logger.info("Validation failed: {}", errors);

        ErrorResponse response = new ErrorResponse(
                "Validation failed",
                errors,
                HttpStatus.BAD_REQUEST.value()
        );

        return ResponseEntity.badRequest().body(response);
    }

    /**
     * Handles illegal argument exceptions.
     */
    @ExceptionHandler(IllegalArgumentException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public ResponseEntity<ErrorResponse> handleIllegalArgument(
            IllegalArgumentException ex) {
        return logAndBuildResponse(
                "Invalid argument",
                ex,
                "Invalid input",
                HttpStatus.BAD_REQUEST
        );
    }

    /**
     * Handles illegal state exceptions.
     */
    @ExceptionHandler(IllegalStateException.class)
    @ResponseStatus(HttpStatus.SERVICE_UNAVAILABLE)
    public ResponseEntity<ErrorResponse> handleIllegalState(
            IllegalStateException ex) {
        return logAndBuildResponse(
                "Invalid state",
                ex,
                "Service unavailable",
                HttpStatus.SERVICE_UNAVAILABLE
        );
    }

    /**
     * Handles requests to non-existent static resources.
     * Returns 404 Not Found instead of 500 Internal Server Error.
     */
    @ExceptionHandler(NoResourceFoundException.class)
    @ResponseStatus(HttpStatus.NOT_FOUND)
    public ResponseEntity<ErrorResponse> handleNoResourceFound(NoResourceFoundException ex) {
        return logAndBuildResponse(
                "Resource not found",
                ex,
                "Not found",
                "The requested resource does not exist",
                HttpStatus.NOT_FOUND
        );
    }

    /**
     * Handles requests to non-existent controller endpoints.
     * Returns 404 Not Found instead of 500 Internal Server Error.
     */
    @ExceptionHandler(NoHandlerFoundException.class)
    @ResponseStatus(HttpStatus.NOT_FOUND)
    public ResponseEntity<ErrorResponse> handleNoHandlerFound(NoHandlerFoundException ex) {
        return logAndBuildResponse(
                "Endpoint not found",
                ex,
                "Not found",
                "The requested endpoint does not exist: " + ex.getRequestURL(),
                HttpStatus.NOT_FOUND
        );
    }

    /**
     * Catches all other unhandled exceptions.
     */
    @ExceptionHandler(Exception.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public ResponseEntity<ErrorResponse> handleGenericException(Exception ex) {
        return logAndBuildResponse(
                "Unexpected error occurred",
                ex,
                "Internal server error",
                "An unexpected error occurred. Please try again later.",
                HttpStatus.INTERNAL_SERVER_ERROR
        );
    }
}
