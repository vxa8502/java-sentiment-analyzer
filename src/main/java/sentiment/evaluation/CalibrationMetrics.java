package sentiment.evaluation;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Computes probability calibration metrics for classification models.
 * <p>
 * A well-calibrated model produces predicted probabilities that match observed frequencies.
 * For example, among predictions with 70% confidence, approximately 70% should be correct.
 * High accuracy does not guarantee good calibration, particularly for models like SVMs.
 * </p>
 *
 * <b>Calibration Metrics</b>
 * <ul>
 *   <li><b>Brier Score:</b> Mean squared error between predicted probabilities and actual outcomes.
 *       Formula: BS = (1/n) Σ(p<sub>predicted</sub> - y<sub>actual</sub>)². Range: [0, 1], lower is better.</li>
 *   <li><b>Expected Calibration Error (ECE):</b> Weighted average of calibration errors across confidence bins.
 *       Formula: ECE = Σ (n<sub>b</sub> / n) |acc(b) - conf(b)|. Range: [0, 1], lower is better.</li>
 *   <li><b>Maximum Calibration Error (MCE):</b> Worst-case calibration error across all bins.
 *       Formula: MCE = max<sub>b</sub> |acc(b) - conf(b)|.</li>
 *   <li><b>Reliability Diagram:</b> Visual representation plotting predicted probability vs. actual frequency.
 *       Well-calibrated models align with the diagonal.</li>
 * </ul>
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
     * Computes calibration metrics for binary classification.
     *
     * @param predictedProbs predicted probabilities for the positive class, values in [0, 1]
     * @param actualLabels actual binary labels (0 or 1)
     * @param numBins number of bins for ECE and reliability diagram (typically 10)
     * @return computed calibration metrics
     * @throws IllegalArgumentException if arrays have different lengths, numBins &lt; 2,
     *         probabilities are out of range, or labels are not binary
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
        validateInputs(predictedProbs, actualLabels, n);

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

        logger.debug("Calibration computed: Brier={}, ECE={}, MCE={}",
                brierScore, ece, mce);

        return metrics;
    }

    /**
     * Computes calibration metrics for multi-class classification using one-vs-rest approach.
     * <p>
     * Calibration is computed separately for each class in a one-vs-rest fashion, then
     * Brier score, ECE, and MCE are averaged across all classes.
     * </p>
     *
     * @param predictedProbs predicted probability matrix [n_samples × n_classes]
     * @param actualLabels actual class indices in range [0, n_classes-1]
     * @param numBins number of bins for ECE computation
     * @return calibration metrics averaged across all classes
     * @throws IllegalArgumentException if sample counts do not match
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

        logger.debug("Multi-class calibration: Brier={}, ECE={}, MCE={}",
                avgBrier, avgECE, avgMCE);

        // Return averaged metrics (bins are from first class for visualization)
        return new CalibrationMetrics(
                avgBrier, avgECE, avgMCE, avgConf, avgAcc,
                numBins, perClassMetrics.get(0).bins
        );
    }

    /**
     * Validates input arrays for calibration computation.
     *
     * @param predictedProbs predicted probabilities
     * @param actualLabels actual binary labels
     * @param n number of samples
     * @throws IllegalArgumentException if probabilities are out of range or labels are not binary
     */
    private static void validateInputs(double[] predictedProbs, int[] actualLabels, int n) {
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
    }

    /**
     * Computes the Brier score:
     * <br>
     * BS = (1/n) Σ(p<sub>predicted</sub> - y<sub>actual</sub>)².
     *
     * @param predictedProbs predicted probabilities
     * @param actualLabels actual binary labels
     * @return Brier score in range [0, 1]
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
     * Creates calibration bins and assigns predictions to them.
     * <p>
     * Bins partition the probability space: [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0].
     * </p>
     *
     * @param predictedProbs predicted probabilities
     * @param actualLabels actual binary labels
     * @param numBins number of bins to create
     * @return list of calibration bins with assigned samples
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
     * Computes Expected Calibration Error (ECE): Σ (n<sub>b</sub> / n) |acc(b) - conf(b)|.
     *
     * @param bins calibration bins
     * @param totalSamples total number of samples
     * @return ECE in range [0, 1]
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
     * Computes Maximum Calibration Error (MCE):
     * <br>
     * max<sub>b</sub> |acc(b) - conf(b)|.
     *
     * @param bins calibration bins
     * @return MCE, the maximum calibration error across all bins
     */
    private static double computeMCE(List<CalibrationBin> bins) {
        return bins.stream()
                .filter(bin -> bin.count > 0)
                .mapToDouble(bin -> Math.abs(bin.getAccuracy() - bin.getAverageConfidence()))
                .max()
                .orElse(0.0);
    }

    // GETTERS

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

    public int getNumBins() {
        return numBins;
    }

    // VISUALIZATION

    /**
     * Prints a text-based reliability diagram to the console.
     * <p>
     * Displays predicted confidence vs. actual accuracy for each bin, along with
     * the calibration gap. Well-calibrated models show small gaps (points near diagonal).
     * Also prints overall Brier score, ECE, and MCE.
     * </p>
     * @deprecated Use {@link #logReliabilityDiagram()} instead
     */
    @Deprecated
    @SuppressWarnings("unused")
    public void printReliabilityDiagram() {
        System.out.println("\n Reliability Diagram ");
        System.out.println("Bin          Confidence  Accuracy  Gap      Samples");
        System.out.println("-".repeat(60));

        for (CalibrationBin bin : bins) {
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

    /**
     * Logs a text-based reliability diagram using the logger.
     * <p>
     * Displays predicted confidence vs. actual accuracy for each bin, along with
     * the calibration gap. Well-calibrated models show small gaps (points near diagonal).
     * Also logs overall Brier score, ECE, and MCE.
     * </p>
     */
    @SuppressWarnings("unused")
    public void logReliabilityDiagram() {
        logger.info("");
        logger.info("Reliability Diagram");
        logger.info("Bin          Confidence  Accuracy  Gap      Samples");
        logger.info("-".repeat(60));

        for (CalibrationBin bin : bins) {
            if (bin.count == 0) {
                logger.info("[{}-{}]   (no samples)",
                        String.format("%.1f", bin.lowerBound),
                        String.format("%.1f", bin.upperBound));
            } else {
                double gap = bin.getAccuracy() - bin.getAverageConfidence();
                String gapSign = gap >= 0 ? "+" : "";

                logger.info("[{}-{}]   {}      {}    {}{}   {}",
                        String.format("%.1f", bin.lowerBound),
                        String.format("%.1f", bin.upperBound),
                        String.format("%.4f", bin.getAverageConfidence()),
                        String.format("%.4f", bin.getAccuracy()),
                        gapSign,
                        String.format("%.4f", gap),
                        bin.count);
            }
        }

        logger.info("-".repeat(60));
        logger.info("Brier Score: {}", String.format("%.4f", brierScore));
        logger.info("ECE:         {}", String.format("%.4f", expectedCalibrationError));
        logger.info("MCE:         {}", String.format("%.4f", maximumCalibrationError));
        logger.info("=".repeat(60));
    }

    @Override
    public String toString() {
        return String.format(
                "CalibrationMetrics{Brier=%.4f, ECE=%.4f, MCE=%.4f, AvgConf=%.4f, AvgAcc=%.4f}",
                brierScore, expectedCalibrationError, maximumCalibrationError,
                averageConfidence, averageAccuracy
        );
    }

    // NESTED CLASSES

    /**
     * Represents a single confidence bin in the reliability diagram.
     * <p>
     * Each bin spans a probability range and accumulates predictions falling within
     * that range, tracking average confidence and accuracy for calibration assessment.
     * </p>
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

        @SuppressWarnings("unused")
        public double getLowerBound() {
            return lowerBound;
        }

        @SuppressWarnings("unused")
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
