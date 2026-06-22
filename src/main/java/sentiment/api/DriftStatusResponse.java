package sentiment.api;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * Response DTO for drift detection status endpoint.
 *
 * <p>Contains PSI (Population Stability Index) values for monitoring
 * distribution drift between reference and comparison windows.
 *
 * @param status Current drift status (HEALTHY, WARNING, CRITICAL)
 * @param predictionPsi PSI for prediction label distribution
 * @param confidencePsi PSI for confidence score distribution
 * @param maxPsi Maximum PSI across all monitored distributions
 * @param referenceWindowSize Number of predictions in reference window
 * @param comparisonWindowSize Number of predictions in comparison window
 * @param pendingRecords Number of records awaiting processing
 * @param state Current detector state (COLLECTING_REFERENCE, MONITORING, etc.)
 * @param ready Whether enough data has been collected for drift detection
 * @param computedAt ISO-8601 timestamp of last drift computation
 * @param logging Status of the prediction logging subsystem
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record DriftStatusResponse(
        String status,
        double predictionPsi,
        double confidencePsi,
        double maxPsi,
        int referenceWindowSize,
        int comparisonWindowSize,
        int pendingRecords,
        String state,
        boolean ready,
        String computedAt,
        LoggingStatus logging
) {

    /**
     * Status of the prediction logging subsystem.
     *
     * @param queueSize Current size of the async logging queue
     * @param logged Total predictions logged since startup
     * @param dropped Predictions dropped due to queue overflow
     */
    public record LoggingStatus(
            int queueSize,
            long logged,
            long dropped
    ) {}
}
