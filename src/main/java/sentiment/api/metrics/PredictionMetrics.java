package sentiment.api.metrics;

import io.micrometer.core.instrument.*;
import org.springframework.stereotype.Component;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * Production ML metrics for monitoring model performance in real-time.
 * Tracks prediction confidence, label distribution, latency, and low-confidence rates.
 */
@Component
public class PredictionMetrics {

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
        this.inferenceTimer = Timer.builder("sentiment.inference.duration")
                .description("Time to perform sentiment inference")
                .publishPercentiles(0.5, 0.95, 0.99)
                .publishPercentileHistogram()
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
     */
    private void updateRollingConfidence(double confidence) {
        totalConfidenceSum.addAndGet((long) (confidence * 10000)); // Store as integer for atomic ops
        confidenceCount.incrementAndGet();

        // Reset every 10000 predictions to prevent overflow and keep it "rolling"
        if (confidenceCount.get() > 10000) {
            totalConfidenceSum.set(0);
            confidenceCount.set(0);
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
        // percentileValues() array indices based on publishPercentiles(0.5, 0.95, 0.99):
        // [0] = p50, [1] = p95, [2] = p99
        int percentileInt = (int) Math.round(percentile * 100);
        int index = switch (percentileInt) {
            case 50 -> 0;
            case 95 -> 1;
            case 99 -> 2;
            default -> throw new IllegalArgumentException("Unknown percentile: " + percentile);
        };
        return timer.takeSnapshot().percentileValues()[index].value(java.util.concurrent.TimeUnit.MILLISECONDS);
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
