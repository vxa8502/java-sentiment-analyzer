package sentiment.evaluation.domain;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Summary statistics for the feature importance distribution.
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
