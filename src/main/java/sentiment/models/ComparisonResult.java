package sentiment.models;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Results from comparing multiple sentiment analysis models.
 *
 * This class aggregates performance metrics across different algorithms to enable:
 * - Side-by-side performance comparison
 * - Trade-off analysis (accuracy vs. speed vs. memory)
 * - Best model selection for specific use cases
 * - Interview-ready insights and talking points
 *
 * Designed to answer questions like:
 * - "Which model is most accurate?"
 * - "Which model is fastest for real-time inference?"
 * - "What's the accuracy/speed trade-off?"
 * - "Which model is best for my production constraints?"
 */
public class ComparisonResult {

    private final List<ClassificationMetrics> modelMetrics;
    private final String datasetName;
    private final int testInstances;
    private final long comparisonTimestamp;

    public ComparisonResult(List<ClassificationMetrics> modelMetrics, String datasetName, int testInstances) {
        if (modelMetrics == null || modelMetrics.isEmpty()) {
            throw new IllegalArgumentException("Model metrics cannot be null or empty");
        }

        this.modelMetrics = new ArrayList<>(modelMetrics);
        this.datasetName = datasetName != null ? datasetName : "Unknown";
        this.testInstances = testInstances;
        this.comparisonTimestamp = System.currentTimeMillis();
    }

    // ==================== GETTERS ====================

    public List<ClassificationMetrics> getClassificationMetrics() {
        return new ArrayList<>(modelMetrics);
    }

    public String getDatasetName() {
        return datasetName;
    }

    public int getTestInstances() {
        return testInstances;
    }

    public long getComparisonTimestamp() {
        return comparisonTimestamp;
    }

    // ==================== RANKING & ANALYSIS ====================

    /**
     * Returns the model with the highest accuracy.
     */
    public ClassificationMetrics getBestByAccuracy() {
        return modelMetrics.stream()
                .max(Comparator.comparingDouble(ClassificationMetrics::getAccuracy))
                .orElse(null);
    }

    /**
     * Returns the model with the highest F1 score.
     */
    public ClassificationMetrics getBestByF1() {
        return modelMetrics.stream()
                .max(Comparator.comparingDouble(ClassificationMetrics::getF1Score))
                .orElse(null);
    }

    /**
     * Returns the model with the fastest training time.
     */
    public ClassificationMetrics getFastestTraining() {
        return modelMetrics.stream()
                .min(Comparator.comparingLong(ClassificationMetrics::getTrainingTimeMs))
                .orElse(null);
    }

    /**
     * Returns the model with the fastest inference time.
     */
    public ClassificationMetrics getFastestInference() {
        return modelMetrics.stream()
                .min(Comparator.comparingLong(ClassificationMetrics::getInferenceTimeMs))
                .orElse(null);
    }

    /**
     * Returns models ranked by accuracy (highest to lowest).
     */
    public List<ClassificationMetrics> getRankedByAccuracy() {
        return modelMetrics.stream()
                .sorted(Comparator.comparingDouble(ClassificationMetrics::getAccuracy).reversed())
                .collect(Collectors.toList());
    }

    /**
     * Returns models ranked by training speed (fastest to slowest).
     */
    public List<ClassificationMetrics> getRankedByTrainingSpeed() {
        return modelMetrics.stream()
                .sorted(Comparator.comparingLong(ClassificationMetrics::getTrainingTimeMs))
                .collect(Collectors.toList());
    }

    /**
     * Returns models ranked by inference speed (fastest to slowest).
     */
    public List<ClassificationMetrics> getRankedByInferenceSpeed() {
        return modelMetrics.stream()
                .sorted(Comparator.comparingLong(ClassificationMetrics::getInferenceTimeMs))
                .collect(Collectors.toList());
    }

    /**
     * Calculates the accuracy gap between best and worst models.
     */
    public double getAccuracySpread() {
        double max = modelMetrics.stream()
                .mapToDouble(ClassificationMetrics::getAccuracy)
                .max()
                .orElse(0.0);
        double min = modelMetrics.stream()
                .mapToDouble(ClassificationMetrics::getAccuracy)
                .min()
                .orElse(0.0);
        return max - min;
    }

    /**
     * Calculates the training time ratio (slowest / fastest).
     */
    public double getTrainingTimeRatio() {
        long max = modelMetrics.stream()
                .mapToLong(ClassificationMetrics::getTrainingTimeMs)
                .max()
                .orElse(1);
        long min = modelMetrics.stream()
                .mapToLong(ClassificationMetrics::getTrainingTimeMs)
                .min()
                .orElse(1);
        return (double) max / Math.max(min, 1);
    }

    /**
     * Calculates the inference time ratio (slowest / fastest).
     */
    public double getInferenceTimeRatio() {
        long max = modelMetrics.stream()
                .mapToLong(ClassificationMetrics::getInferenceTimeMs)
                .max()
                .orElse(1);
        long min = modelMetrics.stream()
                .mapToLong(ClassificationMetrics::getInferenceTimeMs)
                .min()
                .orElse(1);
        return (double) max / Math.max(min, 1);
    }

    // ==================== RECOMMENDATION ENGINE ====================

    /**
     * Recommends the best model based on use case requirements.
     *
     * @param prioritizeAccuracy If true, prioritize accuracy over speed
     * @param prioritizeInferenceSpeed If true, prioritize inference speed over training speed
     * @return Recommended model with justification
     */
    public ModelRecommendation getRecommendation(boolean prioritizeAccuracy, boolean prioritizeInferenceSpeed) {
        ClassificationMetrics recommended;
        String justification;

        if (prioritizeAccuracy) {
            recommended = getBestByAccuracy();
            justification = String.format(
                    "Recommended for highest accuracy (%.2f%%). Best for offline analysis where accuracy matters most.",
                    recommended.getAccuracy() * 100
            );
        } else if (prioritizeInferenceSpeed) {
            recommended = getFastestInference();
            justification = String.format(
                    "Recommended for fastest inference (%d ms/instance). Best for real-time applications.",
                    recommended.getInferenceTimeMs()
            );
        } else {
            // Balance accuracy and training speed
            recommended = modelMetrics.stream()
                    .max(Comparator.comparingDouble(m ->
                            m.getAccuracy() * 0.7 + (1.0 / (m.getTrainingTimeMs() + 1)) * 0.3))
                    .orElse(getBestByAccuracy());
            justification = String.format(
                    "Recommended for balanced accuracy (%.2f%%) and training speed (%d ms). Best overall choice.",
                    recommended.getAccuracy() * 100,
                    recommended.getTrainingTimeMs()
            );
        }

        return new ModelRecommendation(recommended, justification);
    }

    // ==================== FORMATTING ====================

    @Override
    public String toString() {
        return String.format(
                "ComparisonResult[dataset=%s, models=%d, testInstances=%d]",
                datasetName, modelMetrics.size(), testInstances
        );
    }

    /**
     * Returns a comprehensive comparison table showing all models side-by-side.
     */
    public String toComparisonTable() {
        StringBuilder sb = new StringBuilder();
        sb.append("\n=== MODEL COMPARISON RESULTS ===\n");
        sb.append("Dataset: ").append(datasetName).append("\n");
        sb.append("Test Instances: ").append(testInstances).append("\n\n");

        // Header
        sb.append(String.format("%-25s %10s %10s %10s %10s %12s %12s\n",
                "Model", "Accuracy", "Precision", "Recall", "F1-Score", "Train(ms)", "Infer(ms)"));
        sb.append("-".repeat(100)).append("\n");

        // Rows (sorted by accuracy)
        getRankedByAccuracy().forEach(metrics -> {
            sb.append(String.format("%-25s %9.2f%% %10.4f %10.4f %10.4f %12d %12d\n",
                    metrics.getModelName(),
                    metrics.getAccuracy() * 100,
                    metrics.getPrecision(),
                    metrics.getRecall(),
                    metrics.getF1Score(),
                    metrics.getTrainingTimeMs(),
                    metrics.getInferenceTimeMs()
            ));
        });

        sb.append("-".repeat(100)).append("\n\n");

        // Key insights
        sb.append("=== KEY INSIGHTS ===\n");
        ClassificationMetrics bestAccuracy = getBestByAccuracy();
        ClassificationMetrics fastestTrain = getFastestTraining();
        ClassificationMetrics fastestInfer = getFastestInference();

        sb.append(String.format("Highest Accuracy:      %s (%.2f%%)\n",
                bestAccuracy.getModelName(), bestAccuracy.getAccuracy() * 100));
        sb.append(String.format("Fastest Training:      %s (%d ms)\n",
                fastestTrain.getModelName(), fastestTrain.getTrainingTimeMs()));
        sb.append(String.format("Fastest Inference:     %s (%d ms/instance)\n",
                fastestInfer.getModelName(), fastestInfer.getInferenceTimeMs()));
        sb.append(String.format("Accuracy Spread:       %.2f%%\n", getAccuracySpread() * 100));
        sb.append(String.format("Training Speed Ratio:  %.1fx\n", getTrainingTimeRatio()));
        sb.append(String.format("Inference Speed Ratio: %.1fx\n\n", getInferenceTimeRatio()));

        return sb.toString();
    }

    /**
     * Returns a detailed analysis with trade-offs and recommendations.
     */
    public String toDetailedAnalysis() {
        StringBuilder sb = new StringBuilder();
        sb.append(toComparisonTable());

        sb.append("=== TRADE-OFF ANALYSIS ===\n\n");

        // Accuracy vs. Speed analysis
        ClassificationMetrics bestAccuracy = getBestByAccuracy();
        ClassificationMetrics fastestInfer = getFastestInference();

        if (!bestAccuracy.equals(fastestInfer)) {
            double accuracyGap = bestAccuracy.getAccuracy() - fastestInfer.getAccuracy();
            double speedGain = (double) bestAccuracy.getInferenceTimeMs() / fastestInfer.getInferenceTimeMs();

            sb.append("Accuracy vs. Inference Speed:\n");
            sb.append(String.format("  %s is %.2f%% more accurate than %s\n",
                    bestAccuracy.getModelName(),
                    accuracyGap * 100,
                    fastestInfer.getModelName()));
            sb.append(String.format("  But %s is %.1fx faster for inference\n",
                    fastestInfer.getModelName(), speedGain));
            sb.append(String.format("  Trade-off: Accept %.2f%% accuracy loss for %.1fx speedup\n\n",
                    accuracyGap * 100, speedGain));
        }

        // Recommendations
        sb.append("=== RECOMMENDATIONS ===\n\n");

        ModelRecommendation accuracyRec = getRecommendation(true, false);
        sb.append("For Maximum Accuracy:\n");
        sb.append("  ").append(accuracyRec.getClassificationMetrics().getModelName()).append("\n");
        sb.append("  ").append(accuracyRec.getJustification()).append("\n\n");

        ModelRecommendation speedRec = getRecommendation(false, true);
        sb.append("For Real-Time Applications:\n");
        sb.append("  ").append(speedRec.getClassificationMetrics().getModelName()).append("\n");
        sb.append("  ").append(speedRec.getJustification()).append("\n\n");

        ModelRecommendation balancedRec = getRecommendation(false, false);
        sb.append("For Balanced Performance:\n");
        sb.append("  ").append(balancedRec.getClassificationMetrics().getModelName()).append("\n");
        sb.append("  ").append(balancedRec.getJustification()).append("\n\n");

        return sb.toString();
    }

    // ==================== NESTED CLASSES ====================

    /**
     * Model recommendation with justification.
     */
    public static class ModelRecommendation {
        private final ClassificationMetrics modelMetrics;
        private final String justification;

        public ModelRecommendation(ClassificationMetrics modelMetrics, String justification) {
            this.modelMetrics = modelMetrics;
            this.justification = justification;
        }

        public ClassificationMetrics getClassificationMetrics() {
            return modelMetrics;
        }

        public String getJustification() {
            return justification;
        }

        @Override
        public String toString() {
            return String.format("%s: %s", modelMetrics.getModelName(), justification);
        }
    }
}
