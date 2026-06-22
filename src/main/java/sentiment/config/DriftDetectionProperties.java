package sentiment.config;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

import java.time.Duration;

/**
 * Configuration properties for prediction logging and drift detection.
 *
 * <p>Drift detection uses Population Stability Index (PSI) to compare
 * a reference window (baseline) against recent predictions:
 * <ul>
 *   <li>PSI &lt; 0.1: No significant drift (HEALTHY)</li>
 *   <li>0.1 &le; PSI &lt; 0.25: Moderate drift (WARNING)</li>
 *   <li>PSI &ge; 0.25: Significant drift (CRITICAL)</li>
 * </ul>
 */
@ConfigurationProperties(prefix = "sentiment.drift")
@Validated
public class DriftDetectionProperties {

    /**
     * Whether drift detection is enabled.
     */
    private boolean enabled = true;

    /**
     * Reference window size (number of predictions).
     * The first N predictions after startup form the baseline distribution.
     */
    @Min(100)
    @Max(100000)
    private int referenceWindowSize = 1000;

    /**
     * Comparison window size for drift calculation.
     * The most recent M predictions are compared against the reference.
     */
    @Min(50)
    @Max(50000)
    private int comparisonWindowSize = 500;

    /**
     * PSI threshold for warning alert.
     * Indicates moderate distribution shift.
     */
    @Min(0)
    private double warningThreshold = 0.1;

    /**
     * PSI threshold for critical alert.
     * Indicates significant distribution shift requiring attention.
     */
    @Min(0)
    private double criticalThreshold = 0.25;

    /**
     * How often to run drift detection (ISO-8601 duration).
     * Default: PT5M (every 5 minutes).
     */
    private Duration detectionInterval = Duration.ofMinutes(5);

    /**
     * Path for prediction log files.
     * Supports environment variable substitution.
     */
    @NotBlank
    private String logPath = "data/predictions";

    /**
     * Retention period for raw prediction logs.
     * Files older than this are automatically deleted.
     */
    private Duration logRetention = Duration.ofDays(7);

    /**
     * Maximum queue size for async logging.
     * If queue is full, predictions are dropped (logged and counted).
     */
    @Min(100)
    @Max(100000)
    private int maxQueueSize = 10000;

    /**
     * Number of buckets for confidence histogram in drift detection.
     */
    @Min(5)
    @Max(20)
    private int confidenceHistogramBuckets = 10;

    // Getters and setters

    public boolean isEnabled() {
        return enabled;
    }

    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
    }

    public int getReferenceWindowSize() {
        return referenceWindowSize;
    }

    public void setReferenceWindowSize(int referenceWindowSize) {
        this.referenceWindowSize = referenceWindowSize;
    }

    public int getComparisonWindowSize() {
        return comparisonWindowSize;
    }

    public void setComparisonWindowSize(int comparisonWindowSize) {
        this.comparisonWindowSize = comparisonWindowSize;
    }

    public double getWarningThreshold() {
        return warningThreshold;
    }

    public void setWarningThreshold(double warningThreshold) {
        this.warningThreshold = warningThreshold;
    }

    public double getCriticalThreshold() {
        return criticalThreshold;
    }

    public void setCriticalThreshold(double criticalThreshold) {
        this.criticalThreshold = criticalThreshold;
    }

    public Duration getDetectionInterval() {
        return detectionInterval;
    }

    public void setDetectionInterval(Duration detectionInterval) {
        this.detectionInterval = detectionInterval;
    }

    public String getLogPath() {
        return logPath;
    }

    public void setLogPath(String logPath) {
        this.logPath = logPath;
    }

    public Duration getLogRetention() {
        return logRetention;
    }

    public void setLogRetention(Duration logRetention) {
        this.logRetention = logRetention;
    }

    public int getMaxQueueSize() {
        return maxQueueSize;
    }

    public void setMaxQueueSize(int maxQueueSize) {
        this.maxQueueSize = maxQueueSize;
    }

    public int getConfidenceHistogramBuckets() {
        return confidenceHistogramBuckets;
    }

    public void setConfidenceHistogramBuckets(int confidenceHistogramBuckets) {
        this.confidenceHistogramBuckets = confidenceHistogramBuckets;
    }
}
