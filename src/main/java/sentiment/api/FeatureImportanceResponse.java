package sentiment.api;

import com.fasterxml.jackson.annotation.JsonProperty;
import sentiment.evaluation.domain.FeatureWeight;
import sentiment.evaluation.domain.FeatureStatistics;

import java.util.List;

/**
 * Response object for feature importance analysis endpoint.
 */
public record FeatureImportanceResponse(
        @JsonProperty("modelType") String modelType,
        @JsonProperty("totalFeatures") int totalFeatures,
        @JsonProperty("topFeatures") List<FeatureInfo> topFeatures,
        @JsonProperty("statistics") Statistics statistics,
        @JsonProperty("analysisTimeMs") long analysisTimeMs,
        @JsonProperty("note") String note
) {

    /**
     * Individual feature with its importance metrics.
     */
    public record FeatureInfo(
            @JsonProperty("feature") String feature,
            @JsonProperty("weight") double weight,
            @JsonProperty("significance") double significance,
            @JsonProperty("direction") String direction
    ) {
        public static FeatureInfo fromDomain(FeatureWeight fw) {
            return new FeatureInfo(fw.featureName(), fw.weight(), fw.significance(), fw.direction());
        }
    }

    /**
     * Statistical summary of feature importance distribution.
     */
    public record Statistics(
            @JsonProperty("mean") double mean,
            @JsonProperty("stdDev") double stdDev,
            @JsonProperty("median") double median,
            @JsonProperty("percentile95") double percentile95
    ) {
        public static Statistics fromDomain(FeatureStatistics stats) {
            return new Statistics(stats.mean(), stats.stdDev(), stats.median(), stats.percentile95());
        }
    }

    /**
     * Creates a successful feature importance response.
     */
    public static FeatureImportanceResponse success(
            String modelType,
            int totalFeatures,
            List<FeatureInfo> topFeatures,
            Statistics statistics,
            long analysisTimeMs) {
        return new FeatureImportanceResponse(
                modelType,
                totalFeatures,
                topFeatures,
                statistics,
                analysisTimeMs,
                "Feature importance shows which words/n-grams most strongly influence predictions. " +
                "Positive weights indicate positive sentiment, negative weights indicate negative sentiment."
        );
    }

    /**
     * Creates an error response when feature importance cannot be computed.
     */
    public static FeatureImportanceResponse error(String message) {
        return new FeatureImportanceResponse(
                "unknown",
                0,
                List.of(),
                new Statistics(0, 0, 0, 0),
                0,
                message
        );
    }

    /**
     * Creates a response when feature importance analysis is unavailable.
     */
    public static FeatureImportanceResponse unavailable(String modelType, String reason) {
        return new FeatureImportanceResponse(
                modelType,
                0,
                List.of(),
                new Statistics(0, 0, 0, 0),
                0,
                reason
        );
    }
}
