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
import sentiment.models.exceptions.ClassificationException;
import sentiment.models.exceptions.ModelNotTrainedException;
import sentiment.models.exceptions.SentimentClassifierException;

import java.util.HashMap;
import java.util.Map;

/**
 * Global exception handler for REST API endpoints.
 *
 * Catches exceptions thrown by controllers and converts them to
 * appropriate HTTP responses with proper status codes and error messages.
 */
@ControllerAdvice
public class RestApiExceptionHandler {

    private static final Logger logger = LoggerFactory.getLogger(RestApiExceptionHandler.class);

    /**
     * Handles validation errors from @Valid annotations.
     */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public ResponseEntity<Map<String, Object>> handleValidationExceptions(
            MethodArgumentNotValidException ex) {

        Map<String, String> errors = new HashMap<>();
        ex.getBindingResult().getAllErrors().forEach(error -> {
            String fieldName = ((FieldError) error).getField();
            String errorMessage = error.getDefaultMessage();
            errors.put(fieldName, errorMessage);
        });

        logger.info("Validation failed: {}", errors);

        Map<String, Object> response = new HashMap<>();
        response.put("error", "Validation failed");
        response.put("details", errors);
        response.put("status", HttpStatus.BAD_REQUEST.value());

        return ResponseEntity.badRequest().body(response);
    }

    /**
     * Handles model not trained exceptions.
     */
    @ExceptionHandler(ModelNotTrainedException.class)
    @ResponseStatus(HttpStatus.SERVICE_UNAVAILABLE)
    public ResponseEntity<Map<String, Object>> handleModelNotTrained(
            ModelNotTrainedException ex) {

        logger.error("Model not trained: {}", ex.getMessage());

        Map<String, Object> response = new HashMap<>();
        response.put("error", "Model not available");
        response.put("message", ex.getMessage());
        response.put("status", HttpStatus.SERVICE_UNAVAILABLE.value());

        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(response);
    }

    /**
     * Handles classification exceptions.
     */
    @ExceptionHandler(ClassificationException.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public ResponseEntity<Map<String, Object>> handleClassificationException(
            ClassificationException ex) {

        logger.error("Classification error: {}", ex.getMessage(), ex);

        Map<String, Object> response = new HashMap<>();
        response.put("error", "Classification failed");
        response.put("message", ex.getMessage());
        response.put("status", HttpStatus.INTERNAL_SERVER_ERROR.value());

        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
    }

    /**
     * Handles general sentiment classifier exceptions.
     */
    @ExceptionHandler(SentimentClassifierException.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public ResponseEntity<Map<String, Object>> handleSentimentClassifierException(
            SentimentClassifierException ex) {

        logger.error("Sentiment classifier error: {}", ex.getMessage(), ex);

        Map<String, Object> response = new HashMap<>();
        response.put("error", "Classifier error");
        response.put("message", ex.getMessage());
        response.put("status", HttpStatus.INTERNAL_SERVER_ERROR.value());

        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
    }

    /**
     * Handles illegal argument exceptions.
     */
    @ExceptionHandler(IllegalArgumentException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public ResponseEntity<Map<String, Object>> handleIllegalArgument(
            IllegalArgumentException ex) {

        logger.info("Invalid argument: {}", ex.getMessage());

        Map<String, Object> response = new HashMap<>();
        response.put("error", "Invalid input");
        response.put("message", ex.getMessage());
        response.put("status", HttpStatus.BAD_REQUEST.value());

        return ResponseEntity.badRequest().body(response);
    }

    /**
     * Handles illegal state exceptions.
     */
    @ExceptionHandler(IllegalStateException.class)
    @ResponseStatus(HttpStatus.SERVICE_UNAVAILABLE)
    public ResponseEntity<Map<String, Object>> handleIllegalState(
            IllegalStateException ex) {

        logger.error("Invalid state: {}", ex.getMessage());

        Map<String, Object> response = new HashMap<>();
        response.put("error", "Service unavailable");
        response.put("message", ex.getMessage());
        response.put("status", HttpStatus.SERVICE_UNAVAILABLE.value());

        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE).body(response);
    }

    /**
     * Catches all other unhandled exceptions.
     */
    @ExceptionHandler(Exception.class)
    @ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
    public ResponseEntity<Map<String, Object>> handleGenericException(Exception ex) {

        logger.error("Unexpected error occurred", ex);

        Map<String, Object> response = new HashMap<>();
        response.put("error", "Internal server error");
        response.put("message", "An unexpected error occurred. Please try again later.");
        response.put("status", HttpStatus.INTERNAL_SERVER_ERROR.value());

        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
    }
}
