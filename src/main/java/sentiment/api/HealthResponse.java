package sentiment.api;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * Response DTO for health check endpoint with production ML metrics.
 */
public record HealthResponse(
    String status,
    String version,
    Boolean modelLoaded,
    String modelType,
    Long uptimeMs,
    @JsonInclude(JsonInclude.Include.NON_NULL)
    ProductionMetrics productionMetrics
) {
    public static HealthResponse healthy(String version, Boolean modelLoaded, String modelType, Long uptimeMs) {
        return new HealthResponse("UP", version, modelLoaded, modelType, uptimeMs, null);
    }

    public static HealthResponse withMetrics(String version, Boolean modelLoaded, String modelType,
                                              Long uptimeMs, ProductionMetrics metrics) {
        return new HealthResponse("UP", version, modelLoaded, modelType, uptimeMs, metrics);
    }

    /**
     * Production ML metrics for monitoring model performance.
     */
    public record ProductionMetrics(
        long totalPredictions,
        LabelDistribution labelDistribution,
        double averageConfidence,
        double lowConfidenceRatePercent,
        LatencyStats latencyStats
    ) {}

    public record LabelDistribution(
        double positive,
        double negative,
        double neutral
    ) {}

    public record LatencyStats(
        double meanMs,
        double p95Ms,
        double p99Ms
    ) {}
}
