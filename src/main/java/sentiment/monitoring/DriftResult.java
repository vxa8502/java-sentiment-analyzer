package sentiment.monitoring;

import java.time.Instant;

/**
 * Result of a drift detection computation.
 *
 * <p>Contains PSI (Population Stability Index) values for both
 * prediction label distribution and confidence score distribution.
 *
 * @param predictionPsi PSI for prediction label distribution (positive/negative)
 * @param confidencePsi PSI for confidence score histogram
 * @param status        drift status based on threshold comparison
 * @param computedAt    timestamp when drift was computed
 * @param referenceSize number of predictions in reference window
 * @param comparisonSize number of predictions in comparison window
 */
public record DriftResult(
        double predictionPsi,
        double confidencePsi,
        DriftStatus status,
        Instant computedAt,
        int referenceSize,
        int comparisonSize
) {
    /**
     * Creates a DriftResult indicating the system is not ready for drift detection.
     */
    public static DriftResult notReady() {
        return new DriftResult(0.0, 0.0, DriftStatus.NOT_READY, Instant.now(), 0, 0);
    }

    /**
     * Returns the maximum PSI across both distributions.
     */
    public double maxPsi() {
        return Math.max(predictionPsi, confidencePsi);
    }

    /**
     * Drift detection status levels.
     *
     * <p>Based on standard PSI interpretation thresholds:
     * <ul>
     *   <li>PSI &lt; 0.1: No significant change (HEALTHY)</li>
     *   <li>0.1 &le; PSI &lt; 0.25: Moderate change, investigation recommended (WARNING)</li>
     *   <li>PSI &ge; 0.25: Significant change, action required (CRITICAL)</li>
     * </ul>
     */
    public enum DriftStatus {
        /**
         * System is collecting reference window, not yet ready for drift detection.
         */
        NOT_READY(-1),

        /**
         * No significant drift detected. PSI below warning threshold.
         */
        HEALTHY(0),

        /**
         * Moderate drift detected. PSI between warning and critical thresholds.
         */
        WARNING(1),

        /**
         * Significant drift detected. PSI above critical threshold.
         */
        CRITICAL(2);

        private final int code;

        DriftStatus(int code) {
            this.code = code;
        }

        /**
         * Returns numeric code for Prometheus gauge.
         */
        public int getCode() {
            return code;
        }

        /**
         * Determines status based on PSI value and thresholds.
         *
         * @param psi               the PSI value to evaluate
         * @param warningThreshold  threshold for WARNING status
         * @param criticalThreshold threshold for CRITICAL status
         * @return the appropriate DriftStatus
         */
        public static DriftStatus fromPsi(double psi, double warningThreshold, double criticalThreshold) {
            if (psi >= criticalThreshold) {
                return CRITICAL;
            } else if (psi >= warningThreshold) {
                return WARNING;
            } else {
                return HEALTHY;
            }
        }
    }
}
