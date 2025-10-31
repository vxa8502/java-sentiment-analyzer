package sentiment.models.exceptions;

import sentiment.models.AlgorithmType;

/**
 * Exception thrown when text classification fails.
 * Includes information about the input text and classification context.
 *
 * ✅ ENHANCED: Type-safe algorithm identification
 */
public class ClassificationException extends SentimentClassifierException {

    private final String inputText;
    private final ClassificationFailureReason failureReason;

    public enum ClassificationFailureReason {
        TEXT_PREPROCESSING_FAILED("Text preprocessing failed"),
        FEATURE_EXTRACTION_FAILED("Feature extraction failed"),
        MODEL_PREDICTION_FAILED("Model prediction failed"),
        INVALID_INPUT("Invalid input text"),
        PROBABILITY_CALCULATION_FAILED("Probability calculation failed"),
        RESULT_VALIDATION_FAILED("Result validation failed");

        private final String description;

        ClassificationFailureReason(String description) {
            this.description = description;
        }

        public String getDescription() { return description; }
    }

    public ClassificationException(AlgorithmType algorithmType, String inputText,
                                   ClassificationFailureReason reason, String technicalDetails,
                                   Throwable cause) {
        super(algorithmType, "text classification", true,
                createUserMessage(reason, inputText), technicalDetails, cause);
        this.inputText = inputText;
        this.failureReason = reason;
    }

    private static String createUserMessage(ClassificationFailureReason reason, String inputText) {
        String truncatedText = inputText != null && inputText.length() > 50
                ? inputText.substring(0, 50) + "..."
                : inputText;
        return String.format("Classification failed: %s (Input: '%s')",
                reason.getDescription(), truncatedText);
    }

    public String getInputText() { return inputText; }
    public ClassificationFailureReason getFailureReason() { return failureReason; }
}