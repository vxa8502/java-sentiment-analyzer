package sentiment.api;

/**
 * Response DTO for health check endpoint.
 */
public record HealthResponse(
    String status,
    String version,
    Boolean modelLoaded,
    String modelType,
    Long uptimeMs
) {
    public static HealthResponse healthy(String version, Boolean modelLoaded, String modelType, Long uptimeMs) {
        return new HealthResponse("UP", version, modelLoaded, modelType, uptimeMs);
    }
}
