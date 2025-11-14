package sentiment.evaluation;

import java.util.Map;

/**
 * Comprehensive evaluation result for sentiment classifiers.
 * <p>
 * Contains all standard machine learning evaluation metrics including:
 * <ul>
 *   <li>Overall accuracy</li>
 *   <li>Per-class precision, recall, and F1-score</li>
 *   <li>Macro and weighted averages</li>
 *   <li>Confusion matrix</li>
 *   <li>Advanced metrics: ROC-AUC, PR-AUC, calibration metrics</li>
 *   <li>Additional statistics (e.g., training time, vocabulary size)</li>
 * </ul>
 * <p>
 * This class is thread-safe and immutable. All array fields are defensively copied
 * on construction and return to ensure immutability.
 *
 * @see sentiment.models.ClassifierEvaluator
 * @see CalibrationMetrics
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

    // ADVANCED METRICS

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
     * Constructs an evaluation result with basic metrics (backward compatible).
     * <p>
     * Advanced metrics (ROC-AUC, PR-AUC, calibration) are set to {@code null}.
     *
     * @param algorithmName the name of the classification algorithm
     * @param accuracy the overall classification accuracy (0.0 to 1.0)
     * @param precision per-class precision values
     * @param recall per-class recall values
     * @param f1Score per-class F1-score values
     * @param macroAvgPrecision macro-averaged precision (unweighted mean)
     * @param macroAvgRecall macro-averaged recall (unweighted mean)
     * @param macroAvgF1 macro-averaged F1-score (unweighted mean)
     * @param weightedPrecision weighted average precision (by class support)
     * @param weightedRecall weighted average recall (by class support)
     * @param weightedF1 weighted average F1-score (by class support)
     * @param confusionMatrix confusion matrix where {@code [i][j]} represents actual class {@code i} predicted as class {@code j}
     * @param classLabels labels for each class in order
     * @param additionalStats additional statistics as key-value pairs (may be {@code null})
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
     * Constructs an evaluation result with all available metrics including advanced metrics.
     * <p>
     * This is the most comprehensive constructor that accepts all basic and advanced
     * evaluation metrics.
     *
     * @param algorithmName the name of the classification algorithm
     * @param accuracy the overall classification accuracy (0.0 to 1.0)
     * @param precision per-class precision values
     * @param recall per-class recall values
     * @param f1Score per-class F1-score values
     * @param macroAvgPrecision macro-averaged precision (unweighted mean)
     * @param macroAvgRecall macro-averaged recall (unweighted mean)
     * @param macroAvgF1 macro-averaged F1-score (unweighted mean)
     * @param weightedPrecision weighted average precision (by class support)
     * @param weightedRecall weighted average recall (by class support)
     * @param weightedF1 weighted average F1-score (by class support)
     * @param confusionMatrix confusion matrix where {@code [i][j]} represents actual class {@code i} predicted as class {@code j}
     * @param classLabels labels for each class in order
     * @param rocAUC per-class ROC-AUC (Receiver Operating Characteristic - Area Under Curve) values (may be {@code null})
     * @param macroAvgROCAUC macro-averaged ROC-AUC (may be {@code null})
     * @param prAUC per-class PR-AUC (Precision-Recall - Area Under Curve) values (may be {@code null})
     * @param macroAvgPRAUC macro-averaged PR-AUC (may be {@code null})
     * @param calibrationMetrics calibration metrics including Brier score and ECE (may be {@code null})
     * @param additionalStats additional statistics as key-value pairs (may be {@code null})
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

    // GETTERS

    /**
     * Returns the name of the classification algorithm.
     *
     * @return the algorithm name
     */
    public String getAlgorithmName() {
        return algorithmName;
    }

    /**
     * Returns the overall classification accuracy.
     *
     * @return the accuracy value between 0.0 and 1.0
     */
    public double getAccuracy() {
        return accuracy;
    }

    /**
     * Returns a copy of the per-class precision values.
     *
     * @return array of precision values, one per class
     */
    public double[] getPrecision() {
        return precision.clone();
    }

    /**
     * Returns a copy of the per-class recall values.
     *
     * @return array of recall values, one per class
     */
    public double[] getRecall() {
        return recall.clone();
    }

    /**
     * Returns a copy of the per-class F1-score values.
     *
     * @return array of F1-score values, one per class
     */
    public double[] getF1Score() {
        return f1Score.clone();
    }

    /**
     * Returns the macro-averaged precision (unweighted mean across classes).
     *
     * @return the macro-averaged precision
     */
    public double getMacroAvgPrecision() {
        return macroAvgPrecision;
    }

    /**
     * Returns the macro-averaged recall (unweighted mean across classes).
     *
     * @return the macro-averaged recall
     */
    public double getMacroAvgRecall() {
        return macroAvgRecall;
    }

    /**
     * Returns the macro-averaged F1-score (unweighted mean across classes).
     *
     * @return the macro-averaged F1-score
     */
    public double getMacroAvgF1() {
        return macroAvgF1;
    }

    /**
     * Returns the weighted average precision (weighted by class support).
     *
     * @return the weighted average precision
     */
    public double getWeightedPrecision() {
        return weightedPrecision;
    }

    /**
     * Returns the weighted average recall (weighted by class support).
     *
     * @return the weighted average recall
     */
    public double getWeightedRecall() {
        return weightedRecall;
    }

    /**
     * Returns the weighted average F1-score (weighted by class support).
     *
     * @return the weighted average F1-score
     */
    public double getWeightedF1() {
        return weightedF1;
    }

    /**
     * Returns a deep copy of the confusion matrix.
     * <p>
     * The confusion matrix is indexed as {@code [actual][predicted]}, where element
     * {@code [i][j]} represents the number of instances of actual class {@code i}
     * that were predicted as class {@code j}.
     *
     * @return a deep copy of the confusion matrix
     */
    public double[][] getConfusionMatrix() {
        return deepCopyMatrix(confusionMatrix);
    }

    /**
     * Returns a copy of the class labels array.
     *
     * @return array of class labels in order
     */
    public String[] getClassLabels() {
        return classLabels.clone();
    }

    /**
     * Returns an unmodifiable view of the additional statistics.
     *
     * @return map of additional statistics
     */
    public Map<String, Object> getAdditionalStats() {
        return additionalStats;
    }

    /**
     * Returns a copy of the per-class ROC-AUC values.
     *
     * @return array of ROC-AUC values, one per class (empty array if not computed)
     */
    public double[] getRocAUC() {
        return rocAUC.clone();
    }

    /**
     * Returns the macro-averaged ROC-AUC.
     *
     * @return the macro-averaged ROC-AUC, or {@code null} if not computed
     */
    public Double getMacroAvgROCAUC() {
        return macroAvgROCAUC;
    }

    /**
     * Returns a copy of the per-class PR-AUC values.
     *
     * @return array of PR-AUC values, one per class (empty array if not computed)
     */
    public double[] getPrAUC() {
        return prAUC.clone();
    }

    /**
     * Returns the macro-averaged PR-AUC.
     *
     * @return the macro-averaged PR-AUC, or {@code null} if not computed
     */
    public Double getMacroAvgPRAUC() {
        return macroAvgPRAUC;
    }

    /**
     * Returns the calibration metrics.
     *
     * @return the calibration metrics, or {@code null} if not computed
     */
    public CalibrationMetrics getCalibrationMetrics() {
        return calibrationMetrics;
    }

    /**
     * Checks whether advanced metrics (ROC-AUC, PR-AUC, or calibration) are available.
     *
     * @return {@code true} if any advanced metrics are present, {@code false} otherwise
     */
    public boolean hasAdvancedMetrics() {
        return macroAvgROCAUC != null || macroAvgPRAUC != null || calibrationMetrics != null;
    }

    // UTILITY METHODS

    /**
     * Returns the precision for a specific class by index.
     *
     * @param classIndex the zero-based index of the class
     * @return the precision value for the specified class
     * @throws IllegalArgumentException if the class index is invalid
     */
    public double getPrecisionForClass(int classIndex) {
        if (classIndex < 0 || classIndex >= precision.length) {
            throw new IllegalArgumentException("Invalid class index: " + classIndex);
        }
        return precision[classIndex];
    }

    /**
     * Returns the recall for a specific class by index.
     *
     * @param classIndex the zero-based index of the class
     * @return the recall value for the specified class
     * @throws IllegalArgumentException if the class index is invalid
     */
    public double getRecallForClass(int classIndex) {
        if (classIndex < 0 || classIndex >= recall.length) {
            throw new IllegalArgumentException("Invalid class index: " + classIndex);
        }
        return recall[classIndex];
    }

    /**
     * Returns the F1-score for a specific class by index.
     *
     * @param classIndex the zero-based index of the class
     * @return the F1-score value for the specified class
     * @throws IllegalArgumentException if the class index is invalid
     */
    public double getF1ForClass(int classIndex) {
        if (classIndex < 0 || classIndex >= f1Score.length) {
            throw new IllegalArgumentException("Invalid class index: " + classIndex);
        }
        return f1Score[classIndex];
    }

    /**
     * Returns all metrics for a specific class by name.
     *
     * @param className the name of the class
     * @return a {@link ClassMetrics} object containing precision, recall, and F1-score
     * @throws IllegalArgumentException if the class name is not found
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
     * Formats the evaluation result as a detailed, human-readable string.
     * <p>
     * The output includes:
     * <ul>
     *   <li>Overall accuracy</li>
     *   <li>Per-class metrics (precision, recall, F1-score, ROC-AUC, PR-AUC)</li>
     *   <li>Macro and weighted averages</li>
     *   <li>Confusion matrix</li>
     *   <li>Calibration metrics (if available)</li>
     *   <li>Additional statistics (if any)</li>
     * </ul>
     *
     * @return a formatted string representation of all evaluation metrics
     */
    public String toDetailedString() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("%s Evaluation Results\n\n", algorithmName));
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
     * Formats the confusion matrix for display with row and column labels.
     *
     * @return formatted confusion matrix as a string
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
     * Creates a deep copy of a 2D array.
     * <p>
     * This method ensures defensive copying by cloning both the outer array
     * and each inner array.
     *
     * @param matrix the 2D array to copy (may be {@code null})
     * @return a deep copy of the matrix, or an empty 2D array if {@code null}
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

    // NESTED CLASSES

    /**
     * Encapsulates classification metrics for a single class.
     * <p>
     * This immutable class provides precision, recall, and F1-score values
     * for a specific class label.
     */
    public static class ClassMetrics {
        private final String className;
        private final double precision;
        private final double recall;
        private final double f1Score;

        /**
         * Constructs metrics for a single class.
         *
         * @param className the name of the class
         * @param precision the precision value
         * @param recall the recall value
         * @param f1Score the F1-score value
         */
        public ClassMetrics(String className, double precision, double recall, double f1Score) {
            this.className = className;
            this.precision = precision;
            this.recall = recall;
            this.f1Score = f1Score;
        }

        /**
         * Returns the class name.
         *
         * @return the class name
         */
        public String getClassName() {
            return className;
        }

        /**
         * Returns the precision value.
         *
         * @return the precision
         */
        public double getPrecision() {
            return precision;
        }

        /**
         * Returns the recall value.
         *
         * @return the recall
         */
        public double getRecall() {
            return recall;
        }

        /**
         * Returns the F1-score value.
         *
         * @return the F1-score
         */
        public double getF1Score() {
            return f1Score;
        }

        /**
         * Returns a string representation of the class metrics.
         *
         * @return formatted string showing class name and metric values
         */
        @Override
        public String toString() {
            return String.format("ClassMetrics{%s: P=%.4f, R=%.4f, F1=%.4f}",
                className, precision, recall, f1Score);
        }
    }
}
