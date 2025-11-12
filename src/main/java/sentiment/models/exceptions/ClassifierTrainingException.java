package sentiment.models.exceptions;

import sentiment.models.AlgorithmType;

/**
 * Exception thrown when classifier training fails.
 * Includes detailed information about training data, parameters, and failure reasons.
 *
 * ✅ ENHANCED: Type-safe algorithm identification
 */
public class ClassifierTrainingException extends SentimentClassifierException {

    private final int trainingInstances;
    private final int features;
    private final String[] classLabels;
    private final TrainingFailureReason failureReason;

    public enum TrainingFailureReason {
        INSUFFICIENT_DATA("Not enough training data"),
        INVALID_DATA_FORMAT("Training data format is invalid"),
        MEMORY_EXHAUSTED("Insufficient memory for training"),
        ALGORITHM_CONVERGENCE("Algorithm failed to converge"),
        PARAMETER_CONFIGURATION("Invalid algorithm parameters"),
        DATA_PREPROCESSING("Data preprocessing failed"),
        FEATURE_EXTRACTION("Feature extraction failed"),
        UNKNOWN("Unknown training failure");

        private final String description;

        TrainingFailureReason(String description) {
            this.description = description;
        }

        public String getDescription() { return description; }
    }

    public ClassifierTrainingException(AlgorithmType algorithmType, String context,
                                       int trainingInstances, int features,
                                       String[] classLabels, TrainingFailureReason reason,
                                       String technicalDetails, Throwable cause) {
        super(algorithmType, context, determineRecoverability(reason),
                createUserMessage(reason, trainingInstances, features),
                technicalDetails, cause);
        this.trainingInstances = trainingInstances;
        this.features = features;
        this.classLabels = classLabels != null ? classLabels.clone() : new String[0];
        this.failureReason = reason;
    }

    private static boolean determineRecoverability(TrainingFailureReason reason) {
        switch (reason) {
            case INSUFFICIENT_DATA:
            case INVALID_DATA_FORMAT:
            case PARAMETER_CONFIGURATION:
                return true; // User can fix these
            case MEMORY_EXHAUSTED:
            case ALGORITHM_CONVERGENCE:
                return false; // May require system changes
            default:
                return false;
        }
    }

    private static String createUserMessage(TrainingFailureReason reason, int instances, int features) {
        return String.format("Training failed: %s (Dataset: %d instances, %d features)",
                reason.getDescription(), instances, features);
    }

    public int getTrainingInstances() { return trainingInstances; }
    public int getFeatures() { return features; }
    public String[] getClassLabels() { return classLabels.clone(); }
    public TrainingFailureReason getFailureReason() { return failureReason; }

    /**
     * Provides specific recovery suggestions based on failure reason
     */
    public String getRecoverySuggestion() {
        switch (failureReason) {
            case INSUFFICIENT_DATA:
                return "Increase training dataset size. Recommended minimum: 100 instances per class.";
            case INVALID_DATA_FORMAT:
                return "Verify training data format and ensure proper preprocessing.";
            case PARAMETER_CONFIGURATION:
                return "Review algorithm parameters. Try default settings or parameter tuning.";
            case MEMORY_EXHAUSTED:
                return "Reduce dataset size, increase JVM heap size, or use feature selection.";
            case ALGORITHM_CONVERGENCE:
                return "Try different algorithm parameters or alternative algorithms.";
            case FEATURE_EXTRACTION:
                return "Check feature extraction pipeline and input data quality.";
            default:
                return "Check logs for detailed error information and contact support.";
        }
    }
}