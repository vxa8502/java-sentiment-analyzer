package sentiment.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Configuration properties for the Sentiment Analysis API.
 *
 * <p>These properties externalize values that were previously hardcoded,
 * allowing runtime configuration without code changes.
 *
 * <p>Properties are bound from the {@code sentiment.api.*} namespace in
 * application.yml or environment variables.
 */
@ConfigurationProperties(prefix = "sentiment.api")
public class ApiProperties {

    /**
     * Minimum batch size to trigger parallel processing.
     * Batches smaller than this will be processed sequentially.
     * Default: 5
     */
    private int parallelBatchThreshold = 5;

    /**
     * Maximum number of features to return from feature importance endpoint.
     * Requests for more than this will be rejected with 400 Bad Request.
     * Default: 1000
     */
    private int maxTopFeatures = 1000;

    /**
     * Whether to trust X-Forwarded-For and similar proxy headers.
     * Only enable when behind a trusted reverse proxy.
     * Default: false
     */
    private boolean trustProxyHeaders = false;

    /**
     * Comma-separated list of trusted proxy IP addresses.
     * Only used when trustProxyHeaders is true.
     */
    private String trustedProxies = "";

    /**
     * Confidence threshold below which to show low-confidence warning.
     * Default: 0.75
     */
    private double lowConfidenceThreshold = 0.75;

    // Getters and setters

    public int getParallelBatchThreshold() {
        return parallelBatchThreshold;
    }

    public void setParallelBatchThreshold(int parallelBatchThreshold) {
        this.parallelBatchThreshold = parallelBatchThreshold;
    }

    public int getMaxTopFeatures() {
        return maxTopFeatures;
    }

    public void setMaxTopFeatures(int maxTopFeatures) {
        this.maxTopFeatures = maxTopFeatures;
    }

    public boolean isTrustProxyHeaders() {
        return trustProxyHeaders;
    }

    public void setTrustProxyHeaders(boolean trustProxyHeaders) {
        this.trustProxyHeaders = trustProxyHeaders;
    }

    public String getTrustedProxies() {
        return trustedProxies;
    }

    public void setTrustedProxies(String trustedProxies) {
        this.trustedProxies = trustedProxies;
    }

    public double getLowConfidenceThreshold() {
        return lowConfidenceThreshold;
    }

    public void setLowConfidenceThreshold(double lowConfidenceThreshold) {
        this.lowConfidenceThreshold = lowConfidenceThreshold;
    }
}
