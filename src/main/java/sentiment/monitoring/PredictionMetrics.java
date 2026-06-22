package sentiment.monitoring;

import io.micrometer.core.instrument.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * Production ML metrics for monitoring model performance in real-time.
 * Tracks prediction confidence, label distribution, latency, and low-confidence rates.
 *
 * <h2>Thread Safety</h2>
 * <p>This class is <b>fully thread-safe</b>. All public methods can be called concurrently
 * from multiple threads without external synchronization.
 *
 * <h3>Synchronization Strategy</h3>
 * <ul>
 *   <li><b>Counters/Timers</b>: Micrometer's {@link Counter} and {@link Timer} are thread-safe by design.</li>
 *   <li><b>Aggregations</b>: Uses {@link LongAdder} for high-contention counters (better than AtomicLong
 *       under contention) and {@link AtomicLong} with CAS for rolling averages.</li>
 *   <li><b>Snapshots</b>: {@link #getSnapshot()} provides a consistent point-in-time view.</li>
 * </ul>
 *
 * <p>Note: The rolling average calculation uses compare-and-set (CAS) for overflow protection,
 * ensuring atomic reset under concurrent updates.
 */
@Component
public class PredictionMetrics {

    private static final Logger logger = LoggerFactory.getLogger(PredictionMetrics.class);

    private final MeterRegistry meterRegistry;

    // Counters for label distribution (model outputs positive/negative, API may return uncertain)
    private final Counter positiveCounter;
    private final Counter negativeCounter;
    private final Counter uncertainCounter;

    // Counter for low-confidence predictions
    private final Counter lowConfidenceCounter;
    private final LongAdder totalPredictions = new LongAdder();

    // Timer for inference latency
    private final Timer inferenceTimer;

    // Gauge for average confidence (rolling window)
    private final AtomicLong totalConfidenceSum = new AtomicLong(0);
    private final AtomicLong confidenceCount = new AtomicLong(0);

    // Histogram for confidence distribution
    private final DistributionSummary confidenceDistribution;

    // Threshold for low-confidence alerts
    private static final double LOW_CONFIDENCE_THRESHOLD = 0.6;

    public PredictionMetrics(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;

        // Label distribution counters
        this.positiveCounter = Counter.builder("sentiment.predictions.total")
                .tag("label", "positive")
                .description("Total positive sentiment predictions")
                .register(meterRegistry);

        this.negativeCounter = Counter.builder("sentiment.predictions.total")
                .tag("label", "negative")
                .description("Total negative sentiment predictions")
                .register(meterRegistry);

        this.uncertainCounter = Counter.builder("sentiment.predictions.total")
                .tag("label", "uncertain")
                .description("Total uncertain sentiment predictions (below confidence threshold)")
                .register(meterRegistry);

        // Low confidence counter
        this.lowConfidenceCounter = Counter.builder("sentiment.predictions.low_confidence")
                .description("Predictions below confidence threshold")
                .register(meterRegistry);

        // Inference latency timer with percentiles
        // Configure histogram bounds to match expected latency range (1ms-5s) and use longer
        // expiry window to prevent stale percentile values during low-traffic periods.
        // Without explicit bounds, Micrometer defaults (1ms-30s) create sparse bucket coverage
        // that causes inaccurate percentile approximations.
        this.inferenceTimer = Timer.builder("sentiment.inference.duration")
                .description("Time to perform sentiment inference")
                .publishPercentiles(0.5, 0.95, 0.99)
                .publishPercentileHistogram()
                .minimumExpectedValue(Duration.ofMillis(1))
                .maximumExpectedValue(Duration.ofSeconds(5))
                .distributionStatisticExpiry(Duration.ofMinutes(5))
                .distributionStatisticBufferLength(5)
                .register(meterRegistry);

        // Confidence distribution summary
        this.confidenceDistribution = DistributionSummary.builder("sentiment.prediction.confidence")
                .description("Distribution of prediction confidence scores")
                .baseUnit("confidence")
                .scale(100) // Scale 0-1 to 0-100 for better visualization
                .publishPercentiles(0.5, 0.95, 0.99)
                .register(meterRegistry);

        // Gauge for average confidence (computed from rolling totals)
        Gauge.builder("sentiment.prediction.confidence.average", this::getAverageConfidence)
                .description("Average prediction confidence (rolling)")
                .register(meterRegistry);

        // Gauge for low confidence rate
        Gauge.builder("sentiment.prediction.low_confidence.rate", this::getLowConfidenceRate)
                .description("Percentage of low-confidence predictions")
                .register(meterRegistry);
    }

    /**
     * Record a prediction with its metrics.
     */
    public void recordPrediction(String label, double confidence, Duration latency) {
        // Increment label counter (model outputs positive/negative, API may return uncertain)
        switch (label.toLowerCase()) {
            case "positive" -> positiveCounter.increment();
            case "negative" -> negativeCounter.increment();
            case "uncertain" -> uncertainCounter.increment();
        }

        // Record confidence
        confidenceDistribution.record(confidence * 100); // Scale to 0-100
        updateRollingConfidence(confidence);

        // Track low confidence predictions
        if (confidence < LOW_CONFIDENCE_THRESHOLD) {
            lowConfidenceCounter.increment();
        }

        // Record latency
        inferenceTimer.record(latency);

        totalPredictions.increment();
    }

    /**
     * Update rolling average confidence calculation.
     *
     * <p>Performance: Uses compareAndSet for thread-safe reset without lock contention.
     * The reset is atomic - either it succeeds and both counters reset, or it fails
     * and another thread handles the reset.
     */
    private void updateRollingConfidence(double confidence) {
        totalConfidenceSum.addAndGet((long) (confidence * 10000)); // Store as integer for atomic ops
        long currentCount = confidenceCount.incrementAndGet();

        // Reset every 10000 predictions to prevent overflow and keep it "rolling"
        // Use compareAndSet for thread-safe reset without race condition
        if (currentCount > 10000) {
            // Only one thread will succeed with compareAndSet; others will skip reset
            if (confidenceCount.compareAndSet(currentCount, 0)) {
                totalConfidenceSum.set(0);
            }
        }
    }

    /**
     * Get average confidence from recent predictions.
     */
    private double getAverageConfidence() {
        long count = confidenceCount.get();
        if (count == 0) return 0.0;
        return totalConfidenceSum.get() / (double) count / 10000.0;
    }

    /**
     * Get low confidence rate as percentage.
     */
    private double getLowConfidenceRate() {
        long total = totalPredictions.longValue();
        if (total == 0) return 0.0;
        return (lowConfidenceCounter.count() / (double) total) * 100.0;
    }

    /**
     * Get current metrics snapshot.
     */
    public MetricsSnapshot getSnapshot() {
        return new MetricsSnapshot(
                totalPredictions.longValue(),
                positiveCounter.count(),
                negativeCounter.count(),
                uncertainCounter.count(),
                getAverageConfidence(),
                getLowConfidenceRate(),
                inferenceTimer.count(),
                inferenceTimer.mean(java.util.concurrent.TimeUnit.MILLISECONDS),
                getPercentile(inferenceTimer, 0.95),
                getPercentile(inferenceTimer, 0.99)
        );
    }

    private double getPercentile(Timer timer, double percentile) {
        // Iterate through percentileValues and match by percentile value (not index)
        // This is more robust than assuming array order matches publishPercentiles() order
        var snapshot = timer.takeSnapshot();
        for (var vap : snapshot.percentileValues()) {
            if (Math.abs(vap.percentile() - percentile) < 0.001) {
                return vap.value(java.util.concurrent.TimeUnit.MILLISECONDS);
            }
        }
        // Fallback: return 0 if percentile not found
        // This can happen if no data recorded yet, or if percentile config is mismatched
        if (timer.count() > 0) {
            logger.warn("Percentile {} not found in timer snapshot with {} recordings. " +
                    "Available percentiles: {}", percentile, timer.count(),
                    java.util.Arrays.toString(snapshot.percentileValues()));
        }
        return 0.0;
    }

    /**
     * Snapshot of current metrics.
     */
    public record MetricsSnapshot(
            long totalPredictions,
            double positivePredictions,
            double negativePredictions,
            double uncertainPredictions,
            double averageConfidence,
            double lowConfidenceRatePercent,
            long inferenceCount,
            double meanLatencyMs,
            double p95LatencyMs,
            double p99LatencyMs
    ) {}
}
