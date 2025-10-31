package sentiment.models;

import java.util.Arrays;

/**
 * Comprehensive performance metrics for a single model.
 *
 * This class encapsulates all relevant performance characteristics for model comparison:
 * - Accuracy metrics (overall, per-class precision/recall/F1)
 * - Training and inference timing
 * - Confusion matrix for error analysis
 * - Model metadata (algorithm type, parameters)
 *
 * Used by AlgorithmBenchmark to systematically compare models and identify trade-offs.
 */
public class ClassificationMetrics {

    private final AlgorithmType algorithmType;
    private final String modelName;

    // Accuracy metrics
    private final double accuracy;
    private final double precision;
    private final double recall;
    private final double f1Score;

    // Per-class metrics
    private final double[] perClassPrecision;
    private final double[] perClassRecall;
    private final double[] perClassF1;

    // Timing metrics (milliseconds)
    private final long trainingTimeMs;
    private final long inferenceTimeMs;  // Average per instance
    private final long evaluationTimeMs; // Total evaluation time

    // Error analysis
    private final double[][] confusionMatrix;
    private final int correctlyClassified;
    private final int incorrectlyClassified;

    // Model-specific metadata
    private final String[] classLabels;
    private final int trainingInstances;
    private final int featureCount;

    private ClassificationMetrics(Builder builder) {
        this.algorithmType = builder.algorithmType;
        this.modelName = builder.modelName;
        this.accuracy = builder.accuracy;
        this.precision = builder.precision;
        this.recall = builder.recall;
        this.f1Score = builder.f1Score;
        this.perClassPrecision = builder.perClassPrecision;
        this.perClassRecall = builder.perClassRecall;
        this.perClassF1 = builder.perClassF1;
        this.trainingTimeMs = builder.trainingTimeMs;
        this.inferenceTimeMs = builder.inferenceTimeMs;
        this.evaluationTimeMs = builder.evaluationTimeMs;
        this.confusionMatrix = builder.confusionMatrix;
        this.correctlyClassified = builder.correctlyClassified;
        this.incorrectlyClassified = builder.incorrectlyClassified;
        this.classLabels = builder.classLabels;
        this.trainingInstances = builder.trainingInstances;
        this.featureCount = builder.featureCount;
    }

    // ==================== GETTERS ====================

    public AlgorithmType getAlgorithmType() {
        return algorithmType;
    }

    public String getModelName() {
        return modelName;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public double getPrecision() {
        return precision;
    }

    public double getRecall() {
        return recall;
    }

    public double getF1Score() {
        return f1Score;
    }

    public double[] getPerClassPrecision() {
        return perClassPrecision != null ? perClassPrecision.clone() : new double[0];
    }

    public double[] getPerClassRecall() {
        return perClassRecall != null ? perClassRecall.clone() : new double[0];
    }

    public double[] getPerClassF1() {
        return perClassF1 != null ? perClassF1.clone() : new double[0];
    }

    public long getTrainingTimeMs() {
        return trainingTimeMs;
    }

    public long getInferenceTimeMs() {
        return inferenceTimeMs;
    }

    public long getEvaluationTimeMs() {
        return evaluationTimeMs;
    }

    public double[][] getConfusionMatrix() {
        if (confusionMatrix == null) return new double[0][0];
        double[][] copy = new double[confusionMatrix.length][];
        for (int i = 0; i < confusionMatrix.length; i++) {
            copy[i] = confusionMatrix[i].clone();
        }
        return copy;
    }

    public int getCorrectlyClassified() {
        return correctlyClassified;
    }

    public int getIncorrectlyClassified() {
        return incorrectlyClassified;
    }

    public String[] getClassLabels() {
        return classLabels != null ? classLabels.clone() : new String[0];
    }

    public int getTrainingInstances() {
        return trainingInstances;
    }

    public int getFeatureCount() {
        return featureCount;
    }

    // ==================== DERIVED METRICS ====================

    /**
     * Returns training speed in instances per second.
     */
    public double getTrainingSpeed() {
        if (trainingTimeMs == 0) return 0.0;
        return (trainingInstances * 1000.0) / trainingTimeMs;
    }

    /**
     * Returns inference speed in instances per second.
     */
    public double getInferenceSpeed() {
        if (inferenceTimeMs == 0) return 0.0;
        return 1000.0 / inferenceTimeMs;
    }

    /**
     * Returns error rate (1 - accuracy).
     */
    public double getErrorRate() {
        return 1.0 - accuracy;
    }

    // ==================== FORMATTING ====================

    @Override
    public String toString() {
        return String.format(
                "%s: Accuracy=%.3f, Precision=%.3f, Recall=%.3f, F1=%.3f, " +
                        "TrainingTime=%dms, InferenceTime=%dms",
                modelName, accuracy, precision, recall, f1Score,
                trainingTimeMs, inferenceTimeMs
        );
    }

    /**
     * Returns a detailed multi-line summary of all metrics.
     */
    public String toDetailedString() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== ").append(modelName).append(" ===\n");
        sb.append("Algorithm: ").append(algorithmType.getDisplayName()).append("\n\n");

        sb.append("Performance Metrics:\n");
        sb.append(String.format("  Accuracy:  %.4f (%.2f%%)\n", accuracy, accuracy * 100));
        sb.append(String.format("  Precision: %.4f\n", precision));
        sb.append(String.format("  Recall:    %.4f\n", recall));
        sb.append(String.format("  F1-Score:  %.4f\n\n", f1Score));

        sb.append("Timing Metrics:\n");
        sb.append(String.format("  Training:   %d ms (%.1f inst/sec)\n",
                trainingTimeMs, getTrainingSpeed()));
        sb.append(String.format("  Inference:  %d ms/inst (%.1f inst/sec)\n",
                inferenceTimeMs, getInferenceSpeed()));
        sb.append(String.format("  Evaluation: %d ms\n\n", evaluationTimeMs));

        sb.append("Classification Results:\n");
        sb.append(String.format("  Correct:   %d\n", correctlyClassified));
        sb.append(String.format("  Incorrect: %d\n", incorrectlyClassified));
        sb.append(String.format("  Total:     %d\n\n", correctlyClassified + incorrectlyClassified));

        sb.append("Model Configuration:\n");
        sb.append(String.format("  Training instances: %d\n", trainingInstances));
        sb.append(String.format("  Features: %d\n", featureCount));
        sb.append(String.format("  Classes: %s\n", String.join(", ", classLabels)));

        return sb.toString();
    }

    // ==================== BUILDER ====================

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private AlgorithmType algorithmType;
        private String modelName;
        private double accuracy;
        private double precision;
        private double recall;
        private double f1Score;
        private double[] perClassPrecision;
        private double[] perClassRecall;
        private double[] perClassF1;
        private long trainingTimeMs;
        private long inferenceTimeMs;
        private long evaluationTimeMs;
        private double[][] confusionMatrix;
        private int correctlyClassified;
        private int incorrectlyClassified;
        private String[] classLabels;
        private int trainingInstances;
        private int featureCount;

        public Builder algorithmType(AlgorithmType algorithmType) {
            this.algorithmType = algorithmType;
            return this;
        }

        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        public Builder accuracy(double accuracy) {
            this.accuracy = accuracy;
            return this;
        }

        public Builder precision(double precision) {
            this.precision = precision;
            return this;
        }

        public Builder recall(double recall) {
            this.recall = recall;
            return this;
        }

        public Builder f1Score(double f1Score) {
            this.f1Score = f1Score;
            return this;
        }

        public Builder perClassPrecision(double[] perClassPrecision) {
            this.perClassPrecision = perClassPrecision;
            return this;
        }

        public Builder perClassRecall(double[] perClassRecall) {
            this.perClassRecall = perClassRecall;
            return this;
        }

        public Builder perClassF1(double[] perClassF1) {
            this.perClassF1 = perClassF1;
            return this;
        }

        public Builder trainingTimeMs(long trainingTimeMs) {
            this.trainingTimeMs = trainingTimeMs;
            return this;
        }

        public Builder inferenceTimeMs(long inferenceTimeMs) {
            this.inferenceTimeMs = inferenceTimeMs;
            return this;
        }

        public Builder evaluationTimeMs(long evaluationTimeMs) {
            this.evaluationTimeMs = evaluationTimeMs;
            return this;
        }

        public Builder confusionMatrix(double[][] confusionMatrix) {
            this.confusionMatrix = confusionMatrix;
            return this;
        }

        public Builder correctlyClassified(int correctlyClassified) {
            this.correctlyClassified = correctlyClassified;
            return this;
        }

        public Builder incorrectlyClassified(int incorrectlyClassified) {
            this.incorrectlyClassified = incorrectlyClassified;
            return this;
        }

        public Builder classLabels(String[] classLabels) {
            this.classLabels = classLabels;
            return this;
        }

        public Builder trainingInstances(int trainingInstances) {
            this.trainingInstances = trainingInstances;
            return this;
        }

        public Builder featureCount(int featureCount) {
            this.featureCount = featureCount;
            return this;
        }

        public ClassificationMetrics build() {
            if (algorithmType == null) {
                throw new IllegalStateException("Algorithm type is required");
            }
            if (modelName == null || modelName.isEmpty()) {
                throw new IllegalStateException("Model name is required");
            }
            return new ClassificationMetrics(this);
        }
    }
}
