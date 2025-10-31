package sentiment.evaluation;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * ARIA'S MATHEMATICAL RIGOR: Probability Calibration Assessment
 * ==============================================================
 *
 * "Accuracy is not enough. Your model must know WHEN it is certain and WHEN it is not."
 *
 * CALIBRATION PROBLEM:
 * ====================
 * A model is well-calibrated if:
 * - When it predicts 70% probability, it's correct ~70% of the time
 * - When it predicts 90% probability, it's correct ~90% of the time
 *
 * Many classifiers (especially SVMs) are NOT well-calibrated out-of-the-box.
 * High accuracy ≠ good calibration.
 *
 * METRICS:
 * ========
 *
 * 1. Brier Score:
 *    BS = (1/n) Σ(p_predicted - y_actual)²
 *    - Range: [0, 1], lower is better
 *    - 0 = perfect calibration
 *    - Measures both calibration and refinement (discrimination)
 *
 * 2. Expected Calibration Error (ECE):
 *    ECE = Σ (n_b / n) |acc(b) - conf(b)|
 *    - Bins predictions by confidence
 *    - Measures gap between confidence and accuracy per bin
 *    - Range: [0, 1], lower is better
 *
 * 3. Maximum Calibration Error (MCE):
 *    MCE = max_b |acc(b) - conf(b)|
 *    - Worst-case calibration error across bins
 *
 * 4. Reliability Diagram:
 *    Visual: Plot predicted probability vs. actual frequency
 *    Well-calibrated models lie on diagonal line
 *
 * USAGE:
 * ======
 * ```java
 * // After classification
 * double[] predictedProbs = {...};  // Predicted probabilities for positive class
 * int[] actualLabels = {...};       // 0 or 1
 *
 * CalibrationMetrics metrics = CalibrationMetrics.compute(
 *     predictedProbs, actualLabels, 10
 * );
 *
 * System.out.printf("Brier Score: %.4f (lower is better)\n", metrics.brierScore);
 * System.out.printf("ECE: %.4f\n", metrics.expectedCalibrationError);
 *
 * metrics.printReliabilityDiagram();
 * ```
 *
 * @author Aria (Mathematical Foundations Mentor)
 */
public class CalibrationMetrics {

    private static final Logger logger = LoggerFactory.getLogger(CalibrationMetrics.class);

    // Core calibration metrics
    private final double brierScore;
    private final double expectedCalibrationError;  // ECE
    private final double maximumCalibrationError;   // MCE
    private final double averageConfidence;
    private final double averageAccuracy;

    // Reliability diagram data
    private final int numBins;
    private final List<CalibrationBin> bins;

    private CalibrationMetrics(
            double brierScore,
            double expectedCalibrationError,
            double maximumCalibrationError,
            double averageConfidence,
            double averageAccuracy,
            int numBins,
            List<CalibrationBin> bins) {

        this.brierScore = brierScore;
        this.expectedCalibrationError = expectedCalibrationError;
        this.maximumCalibrationError = maximumCalibrationError;
        this.averageConfidence = averageConfidence;
        this.averageAccuracy = averageAccuracy;
        this.numBins = numBins;
        this.bins = bins;
    }

    /**
     * Compute calibration metrics for binary classification.
     *
     * @param predictedProbs Predicted probabilities for positive class [0, 1]
     * @param actualLabels Actual labels (0 or 1)
     * @param numBins Number of bins for ECE and reliability diagram (typically 10)
     * @return CalibrationMetrics
     */
    public static CalibrationMetrics compute(
            double[] predictedProbs,
            int[] actualLabels,
            int numBins) {

        if (predictedProbs.length != actualLabels.length) {
            throw new IllegalArgumentException("Arrays must have same length");
        }

        if (numBins < 2) {
            throw new IllegalArgumentException("numBins must be >= 2");
        }

        int n = predictedProbs.length;

        // Validate inputs
        for (int i = 0; i < n; i++) {
            if (predictedProbs[i] < 0.0 || predictedProbs[i] > 1.0) {
                throw new IllegalArgumentException(
                        String.format("Predicted probability out of range: %.3f", predictedProbs[i]));
            }
            if (actualLabels[i] != 0 && actualLabels[i] != 1) {
                throw new IllegalArgumentException(
                        String.format("Label must be 0 or 1, got: %d", actualLabels[i]));
            }
        }

        // 1. Compute Brier Score
        double brierScore = computeBrierScore(predictedProbs, actualLabels);

        // 2. Bin predictions and compute ECE, MCE
        List<CalibrationBin> bins = createCalibrationBins(predictedProbs, actualLabels, numBins);
        double ece = computeECE(bins, n);
        double mce = computeMCE(bins);

        // 3. Overall statistics
        double avgConfidence = Arrays.stream(predictedProbs).average().orElse(0.0);
        double avgAccuracy = Arrays.stream(actualLabels).average().orElse(0.0);

        CalibrationMetrics metrics = new CalibrationMetrics(
                brierScore, ece, mce, avgConfidence, avgAccuracy, numBins, bins
        );

        logger.debug("Calibration computed: Brier={:.4f}, ECE={:.4f}, MCE={:.4f}",
                brierScore, ece, mce);

        return metrics;
    }

    /**
     * Compute calibration metrics for multi-class classification.
     *
     * APPROACH:
     * - Compute one-vs-rest calibration for each class
     * - Average Brier score and ECE across classes
     *
     * @param predictedProbs Predicted probabilities [n_samples x n_classes]
     * @param actualLabels Actual class indices [0, n_classes-1]
     * @param numBins Number of bins for ECE
     * @return CalibrationMetrics (averaged across classes)
     */
    public static CalibrationMetrics computeMultiClass(
            double[][] predictedProbs,
            int[] actualLabels,
            int numBins) {

        int nSamples = predictedProbs.length;
        int nClasses = predictedProbs[0].length;

        if (actualLabels.length != nSamples) {
            throw new IllegalArgumentException("Sample count mismatch");
        }

        // Compute calibration for each class (one-vs-rest)
        List<CalibrationMetrics> perClassMetrics = new ArrayList<>();

        for (int classIdx = 0; classIdx < nClasses; classIdx++) {
            // Extract probabilities for this class
            double[] classProbs = new double[nSamples];
            int[] binaryLabels = new int[nSamples];

            for (int i = 0; i < nSamples; i++) {
                classProbs[i] = predictedProbs[i][classIdx];
                binaryLabels[i] = (actualLabels[i] == classIdx) ? 1 : 0;
            }

            CalibrationMetrics classMetrics = compute(classProbs, binaryLabels, numBins);
            perClassMetrics.add(classMetrics);
        }

        // Average metrics across classes
        double avgBrier = perClassMetrics.stream()
                .mapToDouble(m -> m.brierScore).average().orElse(0.0);

        double avgECE = perClassMetrics.stream()
                .mapToDouble(m -> m.expectedCalibrationError).average().orElse(0.0);

        double avgMCE = perClassMetrics.stream()
                .mapToDouble(m -> m.maximumCalibrationError).average().orElse(0.0);

        double avgConf = perClassMetrics.stream()
                .mapToDouble(m -> m.averageConfidence).average().orElse(0.0);

        double avgAcc = perClassMetrics.stream()
                .mapToDouble(m -> m.averageAccuracy).average().orElse(0.0);

        logger.debug("Multi-class calibration: Brier={:.4f}, ECE={:.4f}, MCE={:.4f}",
                avgBrier, avgECE, avgMCE);

        // Return averaged metrics (bins are from first class for visualization)
        return new CalibrationMetrics(
                avgBrier, avgECE, avgMCE, avgConf, avgAcc,
                numBins, perClassMetrics.get(0).bins
        );
    }

    /**
     * Compute Brier Score.
     *
     * BS = (1/n) Σ(p_predicted - y_actual)²
     */
    private static double computeBrierScore(double[] predictedProbs, int[] actualLabels) {
        double sum = 0.0;
        for (int i = 0; i < predictedProbs.length; i++) {
            double error = predictedProbs[i] - actualLabels[i];
            sum += error * error;
        }
        return sum / predictedProbs.length;
    }

    /**
     * Create calibration bins.
     *
     * Bins: [0.0-0.1), [0.1-0.2), ..., [0.9-1.0]
     */
    private static List<CalibrationBin> createCalibrationBins(
            double[] predictedProbs,
            int[] actualLabels,
            int numBins) {

        // Initialize bins
        List<CalibrationBin> bins = new ArrayList<>();
        for (int i = 0; i < numBins; i++) {
            double lowerBound = i / (double) numBins;
            double upperBound = (i + 1) / (double) numBins;
            bins.add(new CalibrationBin(lowerBound, upperBound));
        }

        // Assign each prediction to a bin
        for (int i = 0; i < predictedProbs.length; i++) {
            double prob = predictedProbs[i];
            int label = actualLabels[i];

            // Find bin (handle edge case: prob=1.0 goes in last bin)
            int binIdx = Math.min((int) (prob * numBins), numBins - 1);

            bins.get(binIdx).addSample(prob, label);
        }

        return bins;
    }

    /**
     * Compute Expected Calibration Error (ECE).
     *
     * ECE = Σ (n_b / n) |acc(b) - conf(b)|
     */
    private static double computeECE(List<CalibrationBin> bins, int totalSamples) {
        double ece = 0.0;

        for (CalibrationBin bin : bins) {
            if (bin.count > 0) {
                double weight = (double) bin.count / totalSamples;
                double calibrationError = Math.abs(bin.getAccuracy() - bin.getAverageConfidence());
                ece += weight * calibrationError;
            }
        }

        return ece;
    }

    /**
     * Compute Maximum Calibration Error (MCE).
     *
     * MCE = max_b |acc(b) - conf(b)|
     */
    private static double computeMCE(List<CalibrationBin> bins) {
        return bins.stream()
                .filter(bin -> bin.count > 0)
                .mapToDouble(bin -> Math.abs(bin.getAccuracy() - bin.getAverageConfidence()))
                .max()
                .orElse(0.0);
    }

    // ==================== GETTERS ====================

    public double getBrierScore() {
        return brierScore;
    }

    public double getExpectedCalibrationError() {
        return expectedCalibrationError;
    }

    public double getMaximumCalibrationError() {
        return maximumCalibrationError;
    }

    public double getAverageConfidence() {
        return averageConfidence;
    }

    public double getAverageAccuracy() {
        return averageAccuracy;
    }

    public List<CalibrationBin> getBins() {
        return bins;
    }

    // ==================== VISUALIZATION ====================

    /**
     * Print reliability diagram to console.
     *
     * Shows predicted confidence vs. actual accuracy per bin.
     * Well-calibrated models have points near the diagonal.
     */
    public void printReliabilityDiagram() {
        System.out.println("\n=== Reliability Diagram ===");
        System.out.println("Bin          Confidence  Accuracy  Gap      Samples");
        System.out.println("-".repeat(60));

        for (int i = 0; i < bins.size(); i++) {
            CalibrationBin bin = bins.get(i);

            if (bin.count == 0) {
                System.out.printf("[%.1f-%.1f]   (no samples)%n",
                        bin.lowerBound, bin.upperBound);
            } else {
                double gap = bin.getAccuracy() - bin.getAverageConfidence();
                String gapSign = gap >= 0 ? "+" : "";

                System.out.printf("[%.1f-%.1f]   %.4f      %.4f    %s%.4f   %d%n",
                        bin.lowerBound, bin.upperBound,
                        bin.getAverageConfidence(),
                        bin.getAccuracy(),
                        gapSign, gap,
                        bin.count);
            }
        }

        System.out.println("-".repeat(60));
        System.out.printf("Brier Score: %.4f%n", brierScore);
        System.out.printf("ECE:         %.4f%n", expectedCalibrationError);
        System.out.printf("MCE:         %.4f%n", maximumCalibrationError);
        System.out.println("=".repeat(60) + "\n");
    }

    @Override
    public String toString() {
        return String.format(
                "CalibrationMetrics{Brier=%.4f, ECE=%.4f, MCE=%.4f, AvgConf=%.4f, AvgAcc=%.4f}",
                brierScore, expectedCalibrationError, maximumCalibrationError,
                averageConfidence, averageAccuracy
        );
    }

    // ==================== NESTED CLASSES ====================

    /**
     * A single bin in the reliability diagram.
     */
    public static class CalibrationBin {
        private final double lowerBound;
        private final double upperBound;

        private int count = 0;
        private double sumConfidence = 0.0;
        private int sumCorrect = 0;

        public CalibrationBin(double lowerBound, double upperBound) {
            this.lowerBound = lowerBound;
            this.upperBound = upperBound;
        }

        public void addSample(double predictedProb, int actualLabel) {
            count++;
            sumConfidence += predictedProb;
            sumCorrect += actualLabel;
        }

        public double getAverageConfidence() {
            return count > 0 ? sumConfidence / count : 0.0;
        }

        public double getAccuracy() {
            return count > 0 ? (double) sumCorrect / count : 0.0;
        }

        public int getCount() {
            return count;
        }

        public double getLowerBound() {
            return lowerBound;
        }

        public double getUpperBound() {
            return upperBound;
        }

        @Override
        public String toString() {
            if (count == 0) {
                return String.format("Bin[%.1f-%.1f]: empty", lowerBound, upperBound);
            }
            return String.format("Bin[%.1f-%.1f]: conf=%.3f, acc=%.3f, n=%d",
                    lowerBound, upperBound, getAverageConfidence(), getAccuracy(), count);
        }
    }
}
