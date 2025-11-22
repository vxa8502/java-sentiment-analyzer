package sentiment.evaluation.domain;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Summary statistics for the feature importance distribution.
 *
 * <p>Provides statistical insights into the overall feature importance landscape:
 * <ul>
 * <li>mean - Average absolute weight across all features</li>
 * <li>stdDev - Standard deviation indicating weight variance</li>
 * <li>median - Median absolute weight (50th percentile)</li>
 * <li>percentile95 - 95th percentile weight (threshold for highly important features)</li>
 * <li>totalFeatures - Total number of features analyzed</li>
 * </ul>
 *
 * @param mean Average absolute importance weight
 * @param stdDev Standard deviation of absolute weights
 * @param median Median absolute weight (50th percentile)
 * @param percentile95 95th percentile absolute weight
 * @param totalFeatures Total number of features in the analysis
 */
public record FeatureStatistics(
        @JsonProperty("mean") double mean,
        @JsonProperty("stdDev") double stdDev,
        @JsonProperty("median") double median,
        @JsonProperty("percentile95") double percentile95,
        @JsonProperty("totalFeatures") int totalFeatures
) {
    @Override
    public String toString() {
        return String.format("FeatureStats{mean=%.6f, std=%.6f, median=%.6f, p95=%.6f, n=%d}",
                mean, stdDev, median, percentile95, totalFeatures);
    }
}
