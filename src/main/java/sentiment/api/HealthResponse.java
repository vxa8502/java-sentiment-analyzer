package sentiment.api;

/**
 * Response DTO for health check endpoint.
 */
public class HealthResponse {

    private String status;
    private String version;
    private Boolean modelLoaded;
    private String modelType;
    private Long uptimeMs;

    public HealthResponse() {
    }

    public HealthResponse(String status) {
        this.status = status;
    }

    public static HealthResponse healthy(String version, Boolean modelLoaded, String modelType, Long uptimeMs) {
        HealthResponse response = new HealthResponse("UP");
        response.setVersion(version);
        response.setModelLoaded(modelLoaded);
        response.setModelType(modelType);
        response.setUptimeMs(uptimeMs);
        return response;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getVersion() {
        return version;
    }

    public void setVersion(String version) {
        this.version = version;
    }

    public Boolean getModelLoaded() {
        return modelLoaded;
    }

    public void setModelLoaded(Boolean modelLoaded) {
        this.modelLoaded = modelLoaded;
    }

    public String getModelType() {
        return modelType;
    }

    public void setModelType(String modelType) {
        this.modelType = modelType;
    }

    public Long getUptimeMs() {
        return uptimeMs;
    }

    public void setUptimeMs(Long uptimeMs) {
        this.uptimeMs = uptimeMs;
    }
}
