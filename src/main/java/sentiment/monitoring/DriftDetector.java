package sentiment.monitoring;

import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import sentiment.config.DriftDetectionProperties;

import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Scheduled service that computes drift metrics using Population Stability Index (PSI).
 *
 * <p>PSI measures the shift between two distributions:
 * <pre>
 * PSI = Σ (actual% - expected%) × ln(actual% / expected%)
 * </pre>
 *
 * <p>Standard interpretation:
 * <ul>
 *   <li>PSI &lt; 0.1: No significant change</li>
 *   <li>0.1 &le; PSI &lt; 0.25: Moderate change, investigation recommended</li>
 *   <li>PSI &ge; 0.25: Significant change, action required</li>
 * </ul>
 *
 * <p>Drift is computed for both:
 * <ul>
 *   <li>Prediction label distribution (positive/negative)</li>
 *   <li>Confidence score distribution (histogram)</li>
 * </ul>
 */
@Component
@ConditionalOnProperty(name = "sentiment.drift.enabled", havingValue = "true", matchIfMissing = true)
public class DriftDetector {

    private static final Logger logger = LoggerFactory.getLogger(DriftDetector.class);

    /**
     * Small value to prevent division by zero and log(0) in PSI calculation.
     */
    private static final double EPSILON = 0.0001;

    private final DriftStatistics statistics;
    private final DriftDetectionProperties config;
    private final AtomicReference<DriftResult> latestResult = new AtomicReference<>(DriftResult.notReady());

    // Atomic values for Prometheus gauges
    private final AtomicReference<Double> predictionPsiValue = new AtomicReference<>(0.0);
    private final AtomicReference<Double> confidencePsiValue = new AtomicReference<>(0.0);
    private final AtomicReference<Integer> driftStatusValue = new AtomicReference<>(-1);

    public DriftDetector(
            DriftStatistics statistics,
            DriftDetectionProperties config,
            MeterRegistry meterRegistry) {
        this.statistics = statistics;
        this.config = config;

        // Register Prometheus gauges
        Gauge.builder("sentiment.drift.psi.prediction", predictionPsiValue, AtomicReference::get)
                .description("Population Stability Index for prediction label distribution")
                .register(meterRegistry);

        Gauge.builder("sentiment.drift.psi.confidence", confidencePsiValue, AtomicReference::get)
                .description("Population Stability Index for confidence score distribution")
                .register(meterRegistry);

        Gauge.builder("sentiment.drift.status", driftStatusValue, ref -> ref.get().doubleValue())
                .description("Drift status: -1=not_ready, 0=healthy, 1=warning, 2=critical")
                .register(meterRegistry);

        Gauge.builder("sentiment.drift.reference.size", statistics, DriftStatistics::getReferenceWindowSize)
                .description("Number of predictions in reference window")
                .register(meterRegistry);

        Gauge.builder("sentiment.drift.comparison.size", statistics, DriftStatistics::getComparisonWindowSize)
                .description("Number of predictions in comparison window")
                .register(meterRegistry);

        logger.info("DriftDetector initialized with warning threshold {} and critical threshold {}",
                config.getWarningThreshold(), config.getCriticalThreshold());
    }

    /**
     * Scheduled drift detection.
     *
     * <p>Runs at configured interval (default: every 5 minutes).
     * Uses fixedDelay to ensure consistent spacing between runs.
     */
    @Scheduled(fixedDelayString = "${sentiment.drift.detection-interval:PT5M}")
    public void detectDrift() {
        // Drain pending records from lock-free queue into windowed structures
        int drained = statistics.drainPendingRecords();

        if (!statistics.isReadyForComparison()) {
            logger.debug("Insufficient data for drift detection. State: {}, Reference: {}, Comparison: {}, Drained: {}",
                    statistics.getState(),
                    statistics.getReferenceWindowSize(),
                    statistics.getComparisonWindowSize(),
                    drained);
            return;
        }

        try {
            DriftResult result = computeDrift();
            latestResult.set(result);
            updateMetrics(result);

            logDriftResult(result);
        } catch (Exception e) {
            logger.error("Error during drift detection", e);
        }
    }

    /**
     * Manually triggers drift detection (for testing or on-demand checks).
     *
     * @return the computed drift result
     */
    public DriftResult detectDriftNow() {
        // Drain pending records before checking
        statistics.drainPendingRecords();

        if (!statistics.isReadyForComparison()) {
            return DriftResult.notReady();
        }

        DriftResult result = computeDrift();
        latestResult.set(result);
        updateMetrics(result);
        return result;
    }

    /**
     * Returns the most recent drift detection result.
     */
    public DriftResult getLatestResult() {
        return latestResult.get();
    }

    /**
     * Computes PSI for both prediction distribution and confidence distribution.
     */
    private DriftResult computeDrift() {
        // Get distributions
        Map<String, Double> refLabelDist = statistics.getReferenceLabelDistribution();
        Map<String, Double> compLabelDist = statistics.getComparisonLabelDistribution();

        double[] refConfHist = statistics.getReferenceConfidenceHistogram();
        double[] compConfHist = statistics.getComparisonConfidenceHistogram();

        // Compute PSI
        double predictionPsi = computePSI(refLabelDist, compLabelDist);
        double confidencePsi = computePSI(refConfHist, compConfHist);

        // Determine status based on max PSI
        double maxPsi = Math.max(predictionPsi, confidencePsi);
        DriftResult.DriftStatus status = DriftResult.DriftStatus.fromPsi(
                maxPsi,
                config.getWarningThreshold(),
                config.getCriticalThreshold()
        );

        return new DriftResult(
                predictionPsi,
                confidencePsi,
                status,
                java.time.Instant.now(),
                statistics.getReferenceWindowSize(),
                statistics.getComparisonWindowSize()
        );
    }

    /**
     * Computes Population Stability Index for discrete distributions.
     *
     * <p>PSI = Σ (actual% - expected%) × ln(actual% / expected%)
     *
     * @param expected reference distribution (map of category to proportion)
     * @param actual   comparison distribution (map of category to proportion)
     * @return PSI value (0 = identical distributions)
     */
    private double computePSI(Map<String, Double> expected, Map<String, Double> actual) {
        double psi = 0.0;

        // Include all keys from both distributions
        java.util.Set<String> allKeys = new java.util.HashSet<>(expected.keySet());
        allKeys.addAll(actual.keySet());

        for (String key : allKeys) {
            double exp = Math.max(expected.getOrDefault(key, 0.0), EPSILON);
            double act = Math.max(actual.getOrDefault(key, 0.0), EPSILON);
            psi += (act - exp) * Math.log(act / exp);
        }

        return Math.max(0.0, psi); // PSI should be non-negative
    }

    /**
     * Computes Population Stability Index for histogram distributions.
     *
     * @param expected reference histogram (array of bucket proportions)
     * @param actual   comparison histogram (array of bucket proportions)
     * @return PSI value
     */
    private double computePSI(double[] expected, double[] actual) {
        if (expected.length != actual.length) {
            throw new IllegalArgumentException("Histogram lengths must match");
        }

        double psi = 0.0;
        for (int i = 0; i < expected.length; i++) {
            double exp = Math.max(expected[i], EPSILON);
            double act = Math.max(actual[i], EPSILON);
            psi += (act - exp) * Math.log(act / exp);
        }

        return Math.max(0.0, psi);
    }

    private void updateMetrics(DriftResult result) {
        predictionPsiValue.set(result.predictionPsi());
        confidencePsiValue.set(result.confidencePsi());
        driftStatusValue.set(result.status().getCode());
    }

    private void logDriftResult(DriftResult result) {
        switch (result.status()) {
            case CRITICAL -> logger.warn(
                    "CRITICAL DRIFT DETECTED: prediction PSI={}, confidence PSI={}, " +
                            "reference={}, comparison={}",
                    String.format("%.4f", result.predictionPsi()),
                    String.format("%.4f", result.confidencePsi()),
                    result.referenceSize(),
                    result.comparisonSize()
            );
            case WARNING -> logger.warn(
                    "DRIFT WARNING: prediction PSI={}, confidence PSI={}, " +
                            "reference={}, comparison={}",
                    String.format("%.4f", result.predictionPsi()),
                    String.format("%.4f", result.confidencePsi()),
                    result.referenceSize(),
                    result.comparisonSize()
            );
            case HEALTHY -> logger.debug(
                    "Drift check: HEALTHY. prediction PSI={}, confidence PSI={}",
                    String.format("%.4f", result.predictionPsi()),
                    String.format("%.4f", result.confidencePsi())
            );
            case NOT_READY -> logger.debug("Drift detection not ready");
        }
    }
}
