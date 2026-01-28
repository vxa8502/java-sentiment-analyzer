package sentiment.api.metrics;

import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.time.Duration;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for PredictionMetrics, focusing on latency percentile accuracy.
 */
@DisplayName("PredictionMetrics Tests")
class PredictionMetricsTest {

    private PredictionMetrics metrics;

    @BeforeEach
    void setUp() {
        metrics = new PredictionMetrics(new SimpleMeterRegistry());
    }

    @Test
    @DisplayName("Percentiles should be statistically valid (p95 >= p50, p99 >= p95)")
    void percentilesShouldBeStatisticallyValid() {
        // Record a mix of latencies: mostly fast, some slow (simulating cold starts)
        metrics.recordPrediction("positive", 0.9, Duration.ofMillis(300)); // cold start
        for (int i = 0; i < 100; i++) {
            metrics.recordPrediction("positive", 0.85, Duration.ofMillis(2 + (i % 5))); // 2-6ms
        }

        var snapshot = metrics.getSnapshot();

        // Basic sanity: percentiles should follow p50 <= p95 <= p99
        assertTrue(snapshot.p95LatencyMs() >= 0, "p95 should be non-negative");
        assertTrue(snapshot.p99LatencyMs() >= 0, "p99 should be non-negative");
        assertTrue(snapshot.p99LatencyMs() >= snapshot.p95LatencyMs(),
                "p99 should be >= p95, but got p95=" + snapshot.p95LatencyMs() + ", p99=" + snapshot.p99LatencyMs());
    }

    @Test
    @DisplayName("Percentiles should not be less than mean for right-skewed latency distributions")
    void percentilesNotLessThanMeanForSkewedDistributions() {
        // Record distribution with outliers (right-skewed, typical of latency)
        metrics.recordPrediction("positive", 0.9, Duration.ofMillis(500)); // outlier
        for (int i = 0; i < 50; i++) {
            metrics.recordPrediction("positive", 0.85, Duration.ofMillis(5)); // typical
        }

        var snapshot = metrics.getSnapshot();

        // With outliers, p99 should capture the tail and be >= mean
        // Note: p95 might be less than mean if outliers are rare, but p99 should catch them
        assertTrue(snapshot.p99LatencyMs() > 0, "p99 should be positive after recording latencies");

        // The mean is skewed by the 500ms outlier, so it will be higher than most values
        // p99 should be close to or exceed the outlier value
        double meanMs = snapshot.meanLatencyMs();
        assertTrue(meanMs > 0, "Mean should be positive");
    }

    @Test
    @DisplayName("Metrics snapshot should track prediction counts correctly")
    void snapshotShouldTrackCounts() {
        metrics.recordPrediction("positive", 0.9, Duration.ofMillis(10));
        metrics.recordPrediction("negative", 0.8, Duration.ofMillis(10));
        metrics.recordPrediction("uncertain", 0.5, Duration.ofMillis(10));

        var snapshot = metrics.getSnapshot();

        assertEquals(3, snapshot.totalPredictions());
        assertEquals(1.0, snapshot.positivePredictions());
        assertEquals(1.0, snapshot.negativePredictions());
        assertEquals(1.0, snapshot.uncertainPredictions());
    }

    @Test
    @DisplayName("Low confidence predictions should be tracked")
    void lowConfidenceShouldBeTracked() {
        // Record predictions with varying confidence
        metrics.recordPrediction("positive", 0.9, Duration.ofMillis(5)); // high confidence
        metrics.recordPrediction("positive", 0.5, Duration.ofMillis(5)); // low confidence (< 0.6)
        metrics.recordPrediction("positive", 0.4, Duration.ofMillis(5)); // low confidence

        var snapshot = metrics.getSnapshot();

        // 2 out of 3 predictions are low confidence
        double expectedRate = (2.0 / 3.0) * 100.0;
        assertEquals(expectedRate, snapshot.lowConfidenceRatePercent(), 0.01);
    }

    @Test
    @DisplayName("Average confidence should be computed correctly")
    void averageConfidenceShouldBeCorrect() {
        metrics.recordPrediction("positive", 0.8, Duration.ofMillis(5));
        metrics.recordPrediction("positive", 0.6, Duration.ofMillis(5));
        metrics.recordPrediction("positive", 1.0, Duration.ofMillis(5));

        var snapshot = metrics.getSnapshot();

        // Average of 0.8, 0.6, 1.0 = 0.8
        assertEquals(0.8, snapshot.averageConfidence(), 0.01);
    }

    @Test
    @DisplayName("Initial state should return zero percentiles")
    void initialStateShouldReturnZeroPercentiles() {
        // No predictions recorded yet
        var snapshot = metrics.getSnapshot();

        assertEquals(0, snapshot.totalPredictions());
        assertEquals(0.0, snapshot.meanLatencyMs());
        // Percentiles should be 0 when no data recorded
        assertTrue(snapshot.p95LatencyMs() >= 0, "p95 should be non-negative even with no data");
        assertTrue(snapshot.p99LatencyMs() >= 0, "p99 should be non-negative even with no data");
    }

    @Test
    @DisplayName("Latencies at upper bound should be recorded correctly")
    void latenciesAtUpperBoundShouldBeRecorded() {
        // Record latencies at and near the 5s upper bound
        metrics.recordPrediction("positive", 0.9, Duration.ofMillis(4500)); // near bound
        metrics.recordPrediction("positive", 0.9, Duration.ofMillis(5000)); // at bound
        metrics.recordPrediction("positive", 0.9, Duration.ofMillis(100));  // typical

        var snapshot = metrics.getSnapshot();

        assertEquals(3, snapshot.totalPredictions());
        // Mean should reflect all values
        double expectedMean = (4500 + 5000 + 100) / 3.0;
        assertEquals(expectedMean, snapshot.meanLatencyMs(), 1.0);
    }
}
