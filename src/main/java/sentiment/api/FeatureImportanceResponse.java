package sentiment.api;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * Response object for feature importance analysis endpoint.
 *
 * Used for notebook-based model exploration and understanding which features
 * (words/n-grams) most strongly influence sentiment predictions.
 *
 * Example response:
 * {
 *   "modelType": "SVM",
 *   "totalFeatures": 5000,
 *   "topFeatures": [
 *     {
 *       "feature": "excellent",
 *       "weight": 0.234567,
 *       "significance": 0.2346,
 *       "direction": "positive"
 *     },
 *     {
 *       "feature": "terrible",
 *       "weight": -0.198234,
 *       "significance": 0.1982,
 *       "direction": "negative"
 *     }
 *   ],
 *   "statistics": {
 *     "mean": 0.002341,
 *     "stdDev": 0.004123,
 *     "median": 0.001234,
 *     "percentile95": 0.009876
 *     },
 *   "analysisTimeMs": 234
 * }
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
        public static FeatureInfo fromAnalyzerResult(
                sentiment.evaluation.FeatureImportanceAnalyzer.FeatureWeight fw) {
            String direction = fw.weight > 0 ? "positive" : (fw.weight < 0 ? "negative" : "neutral");
            return new FeatureInfo(fw.featureName, fw.weight, fw.significance, direction);
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
        public static Statistics fromAnalyzerResult(
                sentiment.evaluation.FeatureImportanceAnalyzer.FeatureStatistics stats) {
            return new Statistics(stats.mean, stats.stdDev, stats.median, stats.percentile95);
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
     * Creates a response for non-SVM models.
     */
    public static FeatureImportanceResponse unsupported(String modelType) {
        return new FeatureImportanceResponse(
                modelType,
                0,
                List.of(),
                new Statistics(0, 0, 0, 0),
                0,
                "Feature importance analysis is only supported for SVM models. " +
                "Current model type: " + modelType
        );
    }
}
