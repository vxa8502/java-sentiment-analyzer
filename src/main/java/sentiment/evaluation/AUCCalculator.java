package sentiment.evaluation;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.IntStream;

/**
 * ARIA'S MATHEMATICAL RIGOR: ROC-AUC and PR-AUC Computation
 * ===========================================================
 *
 * "Accuracy alone is misleading, especially with imbalanced data.
 * ROC-AUC and PR-AUC reveal the true discrimination power."
 *
 * KEY CONCEPTS:
 * =============
 *
 * 1. ROC-AUC (Receiver Operating Characteristic - Area Under Curve):
 *    - Plots True Positive Rate (TPR) vs. False Positive Rate (FPR)
 *    - Range: [0, 1], higher is better
 *    - 0.5 = random classifier, 1.0 = perfect classifier
 *    - Robust to class imbalance
 *    - Interpretation: Probability that a random positive is ranked higher than a random negative
 *
 * 2. PR-AUC (Precision-Recall - Area Under Curve):
 *    - Plots Precision vs. Recall
 *    - Range: [0, 1], higher is better
 *    - More informative for imbalanced datasets than ROC-AUC
 *    - Focuses on positive class performance
 *
 * WHEN TO USE:
 * ============
 * - ROC-AUC: General classification performance, balanced datasets
 * - PR-AUC: Imbalanced datasets where positive class is rare (e.g., fraud detection)
 *
 * COMPUTATION:
 * ============
 * Both use trapezoidal rule for numerical integration:
 * AUC = Σ (x[i+1] - x[i]) * (y[i+1] + y[i]) / 2
 *
 * EXAMPLE:
 * ========
 * ```java
 * double[] predictedProbs = {0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1};
 * int[] actualLabels = {1, 1, 1, 0, 1, 0, 0, 0};
 *
 * double rocAUC = AUCCalculator.computeROCAUC(predictedProbs, actualLabels);
 * double prAUC = AUCCalculator.computePRAUC(predictedProbs, actualLabels);
 *
 * System.out.printf("ROC-AUC: %.3f\n", rocAUC);  // Higher is better
 * System.out.printf("PR-AUC:  %.3f\n", prAUC);   // Higher is better
 * ```
 *
 * @author Aria (Mathematical Foundations Mentor)
 */
public class AUCCalculator {

    private static final Logger logger = LoggerFactory.getLogger(AUCCalculator.class);

    /**
     * Compute ROC-AUC (Area Under ROC Curve) for binary classification.
     *
     * ALGORITHM:
     * 1. Sort predictions by descending confidence
     * 2. For each threshold, compute TPR and FPR
     * 3. Integrate area under ROC curve using trapezoidal rule
     *
     * @param predictedProbs Predicted probabilities for positive class [0, 1]
     * @param actualLabels Actual labels (0 or 1)
     * @return ROC-AUC score [0, 1]
     */
    public static double computeROCAUC(double[] predictedProbs, int[] actualLabels) {
        if (predictedProbs.length != actualLabels.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }

        if (predictedProbs.length == 0) {
            throw new IllegalArgumentException("Empty arrays");
        }

        // Create (prediction, label) pairs
        List<PredictionLabel> pairs = createPredictionLabelPairs(predictedProbs, actualLabels);

        // Sort by prediction descending (highest confidence first)
        pairs.sort((a, b) -> Double.compare(b.prediction, a.prediction));

        // Count positives and negatives
        int totalPositives = 0;
        int totalNegatives = 0;
        for (int label : actualLabels) {
            if (label == 1) totalPositives++;
            else totalNegatives++;
        }

        if (totalPositives == 0 || totalNegatives == 0) {
            logger.warn("Only one class present in labels, ROC-AUC undefined");
            return 0.5;  // Undefined, return random classifier score
        }

        // Compute ROC curve points
        List<ROCPoint> rocCurve = new ArrayList<>();
        rocCurve.add(new ROCPoint(0.0, 0.0));  // Start at origin

        int truePositives = 0;
        int falsePositives = 0;

        for (PredictionLabel pair : pairs) {
            if (pair.label == 1) {
                truePositives++;
            } else {
                falsePositives++;
            }

            double tpr = (double) truePositives / totalPositives;
            double fpr = (double) falsePositives / totalNegatives;

            rocCurve.add(new ROCPoint(fpr, tpr));
        }

        // Compute AUC using trapezoidal rule
        double auc = computeAUCTrapezoidal(rocCurve);

        logger.debug("ROC-AUC computed: {:.4f}", auc);

        return auc;
    }

    /**
     * Compute PR-AUC (Area Under Precision-Recall Curve) for binary classification.
     *
     * ALGORITHM:
     * 1. Sort predictions by descending confidence
     * 2. For each threshold, compute Precision and Recall
     * 3. Integrate area under PR curve using trapezoidal rule
     *
     * NOTE: PR-AUC is more sensitive to class imbalance than ROC-AUC.
     *
     * @param predictedProbs Predicted probabilities for positive class [0, 1]
     * @param actualLabels Actual labels (0 or 1)
     * @return PR-AUC score [0, 1]
     */
    public static double computePRAUC(double[] predictedProbs, int[] actualLabels) {
        if (predictedProbs.length != actualLabels.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }

        if (predictedProbs.length == 0) {
            throw new IllegalArgumentException("Empty arrays");
        }

        // Create (prediction, label) pairs
        List<PredictionLabel> pairs = createPredictionLabelPairs(predictedProbs, actualLabels);

        // Sort by prediction descending (highest confidence first)
        pairs.sort((a, b) -> Double.compare(b.prediction, a.prediction));

        // Count total positives
        int totalPositives = 0;
        for (int label : actualLabels) {
            if (label == 1) totalPositives++;
        }

        if (totalPositives == 0) {
            logger.warn("No positive samples, PR-AUC undefined");
            return 0.0;
        }

        // Compute PR curve points
        List<PRPoint> prCurve = new ArrayList<>();

        int truePositives = 0;
        int predictedPositives = 0;

        for (PredictionLabel pair : pairs) {
            predictedPositives++;
            if (pair.label == 1) {
                truePositives++;
            }

            double precision = (double) truePositives / predictedPositives;
            double recall = (double) truePositives / totalPositives;

            prCurve.add(new PRPoint(recall, precision));
        }

        // Compute AUC using trapezoidal rule
        double auc = computePRAUCTrapezoidal(prCurve);

        logger.debug("PR-AUC computed: {:.4f}", auc);

        return auc;
    }

    /**
     * Compute multi-class ROC-AUC using one-vs-rest approach.
     *
     * @param predictedProbs Predicted probabilities [n_samples x n_classes]
     * @param actualLabels Actual class indices [0, n_classes-1]
     * @return Per-class ROC-AUC scores
     */
    public static double[] computeMultiClassROCAUC(
            double[][] predictedProbs,
            int[] actualLabels) {

        int nSamples = predictedProbs.length;
        int nClasses = predictedProbs[0].length;

        double[] perClassAUC = new double[nClasses];

        for (int classIdx = 0; classIdx < nClasses; classIdx++) {
            // Extract probabilities for this class
            double[] classProbs = new double[nSamples];
            int[] binaryLabels = new int[nSamples];

            for (int i = 0; i < nSamples; i++) {
                classProbs[i] = predictedProbs[i][classIdx];
                binaryLabels[i] = (actualLabels[i] == classIdx) ? 1 : 0;
            }

            perClassAUC[classIdx] = computeROCAUC(classProbs, binaryLabels);
        }

        return perClassAUC;
    }

    /**
     * Compute multi-class PR-AUC using one-vs-rest approach.
     */
    public static double[] computeMultiClassPRAUC(
            double[][] predictedProbs,
            int[] actualLabels) {

        int nSamples = predictedProbs.length;
        int nClasses = predictedProbs[0].length;

        double[] perClassAUC = new double[nClasses];

        for (int classIdx = 0; classIdx < nClasses; classIdx++) {
            // Extract probabilities for this class
            double[] classProbs = new double[nSamples];
            int[] binaryLabels = new int[nSamples];

            for (int i = 0; i < nSamples; i++) {
                classProbs[i] = predictedProbs[i][classIdx];
                binaryLabels[i] = (actualLabels[i] == classIdx) ? 1 : 0;
            }

            perClassAUC[classIdx] = computePRAUC(classProbs, binaryLabels);
        }

        return perClassAUC;
    }

    // ==================== HELPER METHODS ====================

    private static List<PredictionLabel> createPredictionLabelPairs(
            double[] predictions, int[] labels) {

        List<PredictionLabel> pairs = new ArrayList<>();
        for (int i = 0; i < predictions.length; i++) {
            pairs.add(new PredictionLabel(predictions[i], labels[i]));
        }
        return pairs;
    }

    /**
     * Compute AUC using trapezoidal rule.
     *
     * AUC = Σ (x[i+1] - x[i]) * (y[i+1] + y[i]) / 2
     */
    private static double computeAUCTrapezoidal(List<ROCPoint> curve) {
        double auc = 0.0;

        for (int i = 0; i < curve.size() - 1; i++) {
            ROCPoint p1 = curve.get(i);
            ROCPoint p2 = curve.get(i + 1);

            double width = p2.fpr - p1.fpr;
            double height = (p1.tpr + p2.tpr) / 2.0;

            auc += width * height;
        }

        return auc;
    }

    /**
     * Compute PR-AUC using trapezoidal rule.
     */
    private static double computePRAUCTrapezoidal(List<PRPoint> curve) {
        double auc = 0.0;

        for (int i = 0; i < curve.size() - 1; i++) {
            PRPoint p1 = curve.get(i);
            PRPoint p2 = curve.get(i + 1);

            double width = Math.abs(p2.recall - p1.recall);
            double height = (p1.precision + p2.precision) / 2.0;

            auc += width * height;
        }

        return auc;
    }

    // ==================== DATA CLASSES ====================

    private static class PredictionLabel {
        final double prediction;
        final int label;

        PredictionLabel(double prediction, int label) {
            this.prediction = prediction;
            this.label = label;
        }
    }

    private static class ROCPoint {
        final double fpr;  // False Positive Rate (x-axis)
        final double tpr;  // True Positive Rate (y-axis)

        ROCPoint(double fpr, double tpr) {
            this.fpr = fpr;
            this.tpr = tpr;
        }
    }

    private static class PRPoint {
        final double recall;     // x-axis
        final double precision;  // y-axis

        PRPoint(double recall, double precision) {
            this.recall = recall;
            this.precision = precision;
        }
    }
}
