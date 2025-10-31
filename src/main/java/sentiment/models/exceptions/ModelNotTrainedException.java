package sentiment.models.exceptions;

import sentiment.models.AlgorithmType;

/**
 * Exception thrown when attempting to use an untrained classifier.
 * Provides clear guidance on required training steps.
 *
 * ✅ ENHANCED: Type-safe algorithm identification
 */
public class ModelNotTrainedException extends SentimentClassifierException {

    private final String attemptedOperation;
    private final String[] requiredSteps;

    public ModelNotTrainedException(AlgorithmType algorithmType, String attemptedOperation) {
        super(algorithmType, "model state validation", true,
                String.format("Cannot perform '%s' - model has not been trained", attemptedOperation),
                "Classifier requires training before use", null);
        this.attemptedOperation = attemptedOperation;
        this.requiredSteps = new String[]{
                "1. Prepare training data using WekaInstancesConverter.convertToInstances()",
                "2. Call classifier.train(trainingData)",
                "3. Verify training completed successfully",
                "4. Then perform " + attemptedOperation
        };
    }

    public String getAttemptedOperation() { return attemptedOperation; }
    public String[] getRequiredSteps() { return requiredSteps.clone(); }


    public String getTrainingGuidance() {
        String nl = System.lineSeparator();
        StringBuilder guidance = new StringBuilder();
        guidance.append("Required steps to train the classifier:").append(nl);
        for (String step : requiredSteps) {
            guidance.append("  ").append(step).append(nl);
        }
        return guidance.toString();
    }
}