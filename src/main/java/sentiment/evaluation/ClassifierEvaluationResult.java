package sentiment.evaluation;

import java.util.Map;

/**
 * Comprehensive evaluation result for sentiment classifiers.
 *
 * Contains all standard ML evaluation metrics:
 * - Accuracy
 * - Per-class Precision, Recall, F1-Score
 * - Macro and Weighted averages
 * - Confusion matrix
 * - Additional statistics (training time, vocabulary size, etc.)
 *
 * Thread-safe immutable record.
 *
 * USAGE:
 * - Returned by BasicSVMClassifier.evaluate()
 * - Built by BasicSVMClassifier.buildEvaluationResult() (lines 386-422)
 */
public class ClassifierEvaluationResult {

    private final String algorithmName;
    private final double accuracy;

    // Per-class metrics (indexed by class)
    private final double[] precision;
    private final double[] recall;
    private final double[] f1Score;

    // Macro averages (unweighted)
    private final double macroAvgPrecision;
    private final double macroAvgRecall;
    private final double macroAvgF1;

    // Weighted averages (by class support)
    private final double weightedPrecision;
    private final double weightedRecall;
    private final double weightedF1;

    // Confusion matrix [actual][predicted]
    private final double[][] confusionMatrix;

    // Class labels for interpretation
    private final String[] classLabels;

    // ==================== ADVANCED METRICS ====================

    // ROC-AUC: Area Under ROC Curve (per-class and macro-average)
    private final double[] rocAUC;          // Per-class ROC-AUC
    private final Double macroAvgROCAUC;    // Macro-average ROC-AUC (null if not computed)

    // PR-AUC: Area Under Precision-Recall Curve (per-class and macro-average)
    private final double[] prAUC;           // Per-class PR-AUC
    private final Double macroAvgPRAUC;     // Macro-average PR-AUC (null if not computed)

    // Calibration metrics
    private final CalibrationMetrics calibrationMetrics;  // null if not computed

    // Additional statistics
    private final Map<String, Object> additionalStats;

    /**
     * Full constructor with basic metrics (backward compatible).
     */
    public ClassifierEvaluationResult(
            String algorithmName,
            double accuracy,
            double[] precision,
            double[] recall,
            double[] f1Score,
            double macroAvgPrecision,
            double macroAvgRecall,
            double macroAvgF1,
            double weightedPrecision,
            double weightedRecall,
            double weightedF1,
            double[][] confusionMatrix,
            String[] classLabels,
            Map<String, Object> additionalStats) {

        this(algorithmName, accuracy, precision, recall, f1Score,
             macroAvgPrecision, macroAvgRecall, macroAvgF1,
             weightedPrecision, weightedRecall, weightedF1,
             confusionMatrix, classLabels,
             null, null, null, null,  // Advanced metrics set to null
             null, additionalStats);
    }

    /**
     * Full constructor with advanced metrics (ROC-AUC, PR-AUC, calibration).
     */
    public ClassifierEvaluationResult(
            String algorithmName,
            double accuracy,
            double[] precision,
            double[] recall,
            double[] f1Score,
            double macroAvgPrecision,
            double macroAvgRecall,
            double macroAvgF1,
            double weightedPrecision,
            double weightedRecall,
            double weightedF1,
            double[][] confusionMatrix,
            String[] classLabels,
            double[] rocAUC,
            Double macroAvgROCAUC,
            double[] prAUC,
            Double macroAvgPRAUC,
            CalibrationMetrics calibrationMetrics,
            Map<String, Object> additionalStats) {

        this.algorithmName = algorithmName;
        this.accuracy = accuracy;
        this.precision = precision != null ? precision.clone() : new double[0];
        this.recall = recall != null ? recall.clone() : new double[0];
        this.f1Score = f1Score != null ? f1Score.clone() : new double[0];
        this.macroAvgPrecision = macroAvgPrecision;
        this.macroAvgRecall = macroAvgRecall;
        this.macroAvgF1 = macroAvgF1;
        this.weightedPrecision = weightedPrecision;
        this.weightedRecall = weightedRecall;
        this.weightedF1 = weightedF1;
        this.confusionMatrix = deepCopyMatrix(confusionMatrix);
        this.classLabels = classLabels != null ? classLabels.clone() : new String[0];
        this.rocAUC = rocAUC != null ? rocAUC.clone() : new double[0];
        this.macroAvgROCAUC = macroAvgROCAUC;
        this.prAUC = prAUC != null ? prAUC.clone() : new double[0];
        this.macroAvgPRAUC = macroAvgPRAUC;
        this.calibrationMetrics = calibrationMetrics;
        this.additionalStats = additionalStats != null ? Map.copyOf(additionalStats) : Map.of();
    }

    // ==================== GETTERS ====================

    public String getAlgorithmName() {
        return algorithmName;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public double[] getPrecision() {
        return precision.clone();
    }

    public double[] getRecall() {
        return recall.clone();
    }

    public double[] getF1Score() {
        return f1Score.clone();
    }

    public double getMacroAvgPrecision() {
        return macroAvgPrecision;
    }

    public double getMacroAvgRecall() {
        return macroAvgRecall;
    }

    public double getMacroAvgF1() {
        return macroAvgF1;
    }

    public double getWeightedPrecision() {
        return weightedPrecision;
    }

    public double getWeightedRecall() {
        return weightedRecall;
    }

    public double getWeightedF1() {
        return weightedF1;
    }

    public double[][] getConfusionMatrix() {
        return deepCopyMatrix(confusionMatrix);
    }

    public String[] getClassLabels() {
        return classLabels.clone();
    }

    public Map<String, Object> getAdditionalStats() {
        return additionalStats;
    }

    public double[] getRocAUC() {
        return rocAUC.clone();
    }

    public Double getMacroAvgROCAUC() {
        return macroAvgROCAUC;
    }

    public double[] getPrAUC() {
        return prAUC.clone();
    }

    public Double getMacroAvgPRAUC() {
        return macroAvgPRAUC;
    }

    public CalibrationMetrics getCalibrationMetrics() {
        return calibrationMetrics;
    }

    public boolean hasAdvancedMetrics() {
        return macroAvgROCAUC != null || macroAvgPRAUC != null || calibrationMetrics != null;
    }

    // ==================== UTILITY METHODS ====================

    /**
     * Get precision for a specific class
     */
    public double getPrecisionForClass(int classIndex) {
        if (classIndex < 0 || classIndex >= precision.length) {
            throw new IllegalArgumentException("Invalid class index: " + classIndex);
        }
        return precision[classIndex];
    }

    /**
     * Get recall for a specific class
     */
    public double getRecallForClass(int classIndex) {
        if (classIndex < 0 || classIndex >= recall.length) {
            throw new IllegalArgumentException("Invalid class index: " + classIndex);
        }
        return recall[classIndex];
    }

    /**
     * Get F1 score for a specific class
     */
    public double getF1ForClass(int classIndex) {
        if (classIndex < 0 || classIndex >= f1Score.length) {
            throw new IllegalArgumentException("Invalid class index: " + classIndex);
        }
        return f1Score[classIndex];
    }

    /**
     * Get statistics for a specific class by name
     */
    public ClassMetrics getMetricsForClass(String className) {
        for (int i = 0; i < classLabels.length; i++) {
            if (classLabels[i].equals(className)) {
                return new ClassMetrics(className, precision[i], recall[i], f1Score[i]);
            }
        }
        throw new IllegalArgumentException("Class not found: " + className);
    }

    /**
     * Format evaluation result as a readable string
     */
    public String toDetailedString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("=== %s Evaluation Results ===\n\n", algorithmName));
        sb.append(String.format("Overall Accuracy: %.4f (%.2f%%)\n\n", accuracy, accuracy * 100));

        sb.append("Per-Class Metrics:\n");
        for (int i = 0; i < classLabels.length; i++) {
            sb.append(String.format("  %s:\n", classLabels[i]));
            sb.append(String.format("    Precision: %.4f\n", precision[i]));
            sb.append(String.format("    Recall:    %.4f\n", recall[i]));
            sb.append(String.format("    F1-Score:  %.4f\n", f1Score[i]));

            // Add ROC-AUC and PR-AUC if available
            if (rocAUC.length > i) {
                sb.append(String.format("    ROC-AUC:   %.4f\n", rocAUC[i]));
            }
            if (prAUC.length > i) {
                sb.append(String.format("    PR-AUC:    %.4f\n", prAUC[i]));
            }
        }

        sb.append("\nMacro Averages (unweighted):\n");
        sb.append(String.format("  Precision: %.4f\n", macroAvgPrecision));
        sb.append(String.format("  Recall:    %.4f\n", macroAvgRecall));
        sb.append(String.format("  F1-Score:  %.4f\n", macroAvgF1));

        if (macroAvgROCAUC != null) {
            sb.append(String.format("  ROC-AUC:   %.4f\n", macroAvgROCAUC));
        }
        if (macroAvgPRAUC != null) {
            sb.append(String.format("  PR-AUC:    %.4f\n", macroAvgPRAUC));
        }

        sb.append("\nWeighted Averages:\n");
        sb.append(String.format("  Precision: %.4f\n", weightedPrecision));
        sb.append(String.format("  Recall:    %.4f\n", weightedRecall));
        sb.append(String.format("  F1-Score:  %.4f\n", weightedF1));

        sb.append("\nConfusion Matrix:\n");
        sb.append(formatConfusionMatrix());

        // Add calibration metrics if available
        if (calibrationMetrics != null) {
            sb.append("\nCalibration Metrics:\n");
            sb.append(String.format("  Brier Score: %.4f (lower is better)\n",
                    calibrationMetrics.getBrierScore()));
            sb.append(String.format("  ECE:         %.4f (expected calibration error)\n",
                    calibrationMetrics.getExpectedCalibrationError()));
            sb.append(String.format("  MCE:         %.4f (maximum calibration error)\n",
                    calibrationMetrics.getMaximumCalibrationError()));
        }

        if (!additionalStats.isEmpty()) {
            sb.append("\nAdditional Statistics:\n");
            additionalStats.forEach((key, value) ->
                sb.append(String.format("  %s: %s\n", key, value))
            );
        }

        return sb.toString();
    }

    /**
     * Format confusion matrix for display
     */
    private String formatConfusionMatrix() {
        StringBuilder sb = new StringBuilder();

        // Header row
        sb.append("           ");
        for (String label : classLabels) {
            sb.append(String.format("%-12s", label));
        }
        sb.append("\n");

        // Matrix rows
        for (int i = 0; i < confusionMatrix.length; i++) {
            sb.append(String.format("%-10s ", classLabels[i]));
            for (int j = 0; j < confusionMatrix[i].length; j++) {
                sb.append(String.format("%-12.0f", confusionMatrix[i][j]));
            }
            sb.append("\n");
        }

        return sb.toString();
    }

    /**
     * Deep copy a 2D array
     */
    private double[][] deepCopyMatrix(double[][] matrix) {
        if (matrix == null) return new double[0][0];

        double[][] copy = new double[matrix.length][];
        for (int i = 0; i < matrix.length; i++) {
            copy[i] = matrix[i] != null ? matrix[i].clone() : new double[0];
        }
        return copy;
    }

    @Override
    public String toString() {
        return String.format(
            "ClassifierEvaluationResult{algorithm='%s', accuracy=%.4f, macroF1=%.4f, weightedF1=%.4f}",
            algorithmName, accuracy, macroAvgF1, weightedF1
        );
    }

    // ==================== NESTED CLASSES ====================

    /**
     * Metrics for a single class
     */
    public static class ClassMetrics {
        private final String className;
        private final double precision;
        private final double recall;
        private final double f1Score;

        public ClassMetrics(String className, double precision, double recall, double f1Score) {
            this.className = className;
            this.precision = precision;
            this.recall = recall;
            this.f1Score = f1Score;
        }

        public String getClassName() {
            return className;
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

        @Override
        public String toString() {
            return String.format("ClassMetrics{%s: P=%.4f, R=%.4f, F1=%.4f}",
                className, precision, recall, f1Score);
        }
    }
}
