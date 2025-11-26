package sentiment.api;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.FieldError;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import sentiment.models.exceptions.SentimentClassifierException;

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
     * Handles general sentiment classifier exceptions.
     */
    @ExceptionHandler(SentimentClassifierException.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public ResponseEntity<ErrorResponse> handleSentimentClassifierException(
            SentimentClassifierException ex) {
        return logAndBuildResponse(
                "Sentiment classifier error",
                ex,
                "Classifier error",
                HttpStatus.INTERNAL_SERVER_ERROR
        );
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
