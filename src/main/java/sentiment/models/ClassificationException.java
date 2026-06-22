package sentiment.models;

/**
 * Exception thrown when sentiment classification fails.
 * <p>
 * This exception provides a specific type for classification errors, replacing
 * the generic {@code throws Exception} declaration in {@link SentimentClassifier}.
 * <p>
 * Common causes:
 * <ul>
 *   <li>Classifier not trained (use {@link #notTrained()})</li>
 *   <li>Invalid input text (use {@link #invalidInput(String)})</li>
 *   <li>Model inference failure (use {@link #inferenceError(Throwable)})</li>
 *   <li>Training failure (use {@link #trainingError(Throwable)})</li>
 * </ul>
 */
public class ClassificationException extends Exception {

    private final ErrorType errorType;

    /**
     * Error type categories for classification exceptions.
     */
    public enum ErrorType {
        /** Classifier has not been trained */
        NOT_TRAINED,
        /** Input text is invalid (null, empty, or malformed) */
        INVALID_INPUT,
        /** Model inference failed */
        INFERENCE_ERROR,
        /** Model training failed */
        TRAINING_ERROR,
        /** Other/unknown error */
        UNKNOWN
    }

    private ClassificationException(String message, ErrorType errorType) {
        super(message);
        this.errorType = errorType;
    }

    private ClassificationException(String message, Throwable cause, ErrorType errorType) {
        super(message, cause);
        this.errorType = errorType;
    }

    /**
     * Returns the error type category.
     */
    public ErrorType getErrorType() {
        return errorType;
    }

    /**
     * Creates exception for untrained classifier.
     */
    public static ClassificationException notTrained() {
        return new ClassificationException(
                "Classifier must be trained before use. Call train() first.",
                ErrorType.NOT_TRAINED
        );
    }

    /**
     * Creates exception for invalid input text.
     *
     * @param reason why the input is invalid
     */
    public static ClassificationException invalidInput(String reason) {
        return new ClassificationException(
                "Invalid input: " + reason,
                ErrorType.INVALID_INPUT
        );
    }

    /**
     * Creates exception for model inference errors.
     *
     * @param cause the underlying exception
     */
    public static ClassificationException inferenceError(Throwable cause) {
        return new ClassificationException(
                "Classification failed: " + cause.getMessage(),
                cause,
                ErrorType.INFERENCE_ERROR
        );
    }

    /**
     * Creates exception for training errors.
     *
     * @param cause the underlying exception
     */
    public static ClassificationException trainingError(Throwable cause) {
        return new ClassificationException(
                "Training failed: " + cause.getMessage(),
                cause,
                ErrorType.TRAINING_ERROR
        );
    }

    /**
     * Creates exception with custom message.
     *
     * @param message error description
     */
    public static ClassificationException of(String message) {
        return new ClassificationException(message, ErrorType.UNKNOWN);
    }

    /**
     * Creates exception with custom message and cause.
     *
     * @param message error description
     * @param cause   the underlying exception
     */
    public static ClassificationException of(String message, Throwable cause) {
        return new ClassificationException(message, cause, ErrorType.UNKNOWN);
    }
}
