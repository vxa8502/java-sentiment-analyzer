package sentiment.api;

import com.fasterxml.jackson.annotation.JsonProperty;
import sentiment.evaluation.domain.FeatureWeight;
import sentiment.evaluation.domain.FeatureStatistics;

import java.util.List;

/**
 * Response object for feature importance analysis endpoint.
 *
 * <h2>Error Handling Pattern</h2>
 * <p>This response follows the standard error pattern used across the API:
 * <ul>
 *   <li>{@code error} field is non-null when the request failed</li>
 *   <li>{@code note} field is for informational messages on success</li>
 * </ul>
 *
 * <p>Clients should check {@code error != null} to determine if the request failed.
 */
@com.fasterxml.jackson.annotation.JsonInclude(com.fasterxml.jackson.annotation.JsonInclude.Include.NON_NULL)
public record FeatureImportanceResponse(
        @JsonProperty("modelType") String modelType,
        @JsonProperty("totalFeatures") int totalFeatures,
        @JsonProperty("topFeatures") List<FeatureInfo> topFeatures,
        @JsonProperty("statistics") Statistics statistics,
        @JsonProperty("analysisTimeMs") long analysisTimeMs,
        @JsonProperty("note") String note,
        @JsonProperty("error") String error
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
        public static Statistics empty() {
            return new Statistics(0, 0, 0, 0);
        }

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
                "Positive weights indicate positive sentiment, negative weights indicate negative sentiment.",
                null  // no error
        );
    }

    /**
     * Creates an empty response with the given model type and informational message.
     * Use this for non-error conditions like "no features available yet".
     */
    public static FeatureImportanceResponse empty(String modelType, String note) {
        return new FeatureImportanceResponse(modelType, 0, List.of(), Statistics.empty(), 0, note, null);
    }

    /**
     * Creates an error response when feature importance cannot be computed.
     * The error field will be populated; note will be null.
     */
    public static FeatureImportanceResponse error(String errorMessage) {
        return new FeatureImportanceResponse("unknown", 0, List.of(), Statistics.empty(), 0, null, errorMessage);
    }

    /**
     * Creates an error response with timing information.
     * Use this when an error occurs during computation to preserve timing context.
     */
    public static FeatureImportanceResponse error(String errorMessage, long durationMs) {
        return new FeatureImportanceResponse("unknown", 0, List.of(), Statistics.empty(), durationMs, null, errorMessage);
    }

    /**
     * Creates a response when feature importance analysis is unavailable (not an error).
     * For example, when the model type doesn't support feature extraction.
     */
    public static FeatureImportanceResponse unavailable(String modelType, String reason) {
        return empty(modelType, reason);
    }

    /**
     * Checks if this response represents an error.
     */
    public boolean isError() {
        return error != null;
    }

    /**
     * Returns a new response with features limited to the specified count.
     * If limit >= current size, returns this instance unchanged.
     */
    public FeatureImportanceResponse withTopFeatures(int limit) {
        if (limit >= topFeatures.size()) {
            return this;
        }
        return new FeatureImportanceResponse(
                modelType,
                totalFeatures,
                List.copyOf(topFeatures.subList(0, limit)),
                statistics,
                analysisTimeMs,
                note,
                error
        );
    }
}
