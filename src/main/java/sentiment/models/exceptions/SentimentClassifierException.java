package sentiment.models.exceptions;

import sentiment.models.AlgorithmType;

/**
 * Base exception for all sentiment classifier-related errors.
 * Provides common functionality for error context, recovery hints, and debugging information.
 *
 * ✅ ENHANCED: Type-safe algorithm identification using AlgorithmType enum
 */
public abstract class SentimentClassifierException extends Exception {

    private final AlgorithmType algorithmType;
    private final String context;
    private final boolean recoverable;
    private final String userMessage;
    private final String technicalDetails;

    protected SentimentClassifierException(AlgorithmType algorithmType, String context,
                                           boolean recoverable, String userMessage,
                                           String technicalDetails, Throwable cause) {
        super(createMessage(algorithmType, userMessage, technicalDetails), cause);
        this.algorithmType = algorithmType != null ? algorithmType : AlgorithmType.UNKNOWN;
        this.context = context;
        this.recoverable = recoverable;
        this.userMessage = userMessage;
        this.technicalDetails = technicalDetails;
    }

    private static String createMessage(AlgorithmType algorithmType,
                                        String userMessage,
                                        String technicalDetails) {
        String algo = algorithmType != null ? algorithmType.getDisplayName() : "Classifier";
        String msg = userMessage != null ? userMessage : "Unknown error";
        String details = technicalDetails != null ? technicalDetails : "No details available";

        return String.format("[%s] %s. Technical details: %s", algo, msg, details);
    }

    public AlgorithmType getAlgorithmType() { return algorithmType; }
    public String getAlgorithmName() { return algorithmType.getDisplayName(); }
    public String getContext() { return context; }
    public boolean isRecoverable() { return recoverable; }
    public String getUserMessage() { return userMessage; }
    public String getTechnicalDetails() { return technicalDetails; }

    /**
     * Returns user-friendly error message without technical details
     */
    public String getUserFriendlyMessage() {
        return userMessage;
    }

    /**
     * Returns comprehensive diagnostic information for debugging
     */
    public String getDiagnosticInfo() {
        StringBuilder info = new StringBuilder();
        info.append("Algorithm: ").append(algorithmType.getDisplayName()).append("\n");
        info.append("Algorithm Description: ").append(algorithmType.getDescription()).append("\n");
        info.append("Context: ").append(context).append("\n");
        info.append("Recoverable: ").append(recoverable).append("\n");
        info.append("User Message: ").append(userMessage).append("\n");
        info.append("Technical Details: ").append(technicalDetails).append("\n");
        if (getCause() != null) {
            info.append("Root Cause: ").append(getCause().getClass().getSimpleName())
                    .append(" - ").append(getCause().getMessage()).append("\n");
        }
        return info.toString();
    }
}