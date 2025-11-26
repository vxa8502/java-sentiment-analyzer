package sentiment.evaluation.domain;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

/**
 * Contains the complete results of feature importance analysis.
 *
 * @param topFeatures The top-K most important features (subset of allFeatures)
 * @param allFeatures All features ranked by absolute importance (descending)
 * @param statistics Summary statistics for the importance distribution
 * @param analysisTimeMs Time taken to perform the analysis (milliseconds)
 */
public record FeatureImportanceResult(
        @JsonProperty("topFeatures") List<FeatureWeight> topFeatures,
        @JsonProperty("allFeatures") List<FeatureWeight> allFeatures,
        @JsonProperty("statistics") FeatureStatistics statistics,
        @JsonProperty("analysisTimeMs") long analysisTimeMs
) {
    /**
     * Prints the top N features to console.
     *
     * @param limit Maximum number of features to print
     */
    public void printTopFeatures(int limit) {
        System.out.println("\n=== TOP DISCRIMINATIVE FEATURES ===");
        topFeatures.stream()
                .limit(limit)
                .forEach(fw -> System.out.printf("%40s: weight=%+.6f, significance=%.4f\n",
                        fw.featureName(), fw.weight(), fw.significance()));
        System.out.println("===================================\n");
    }

    @Override
    public String toString() {
        return String.format("FeatureImportanceResult{top=%d/%d features, mean=%.4f, std=%.4f, time=%dms}",
                topFeatures.size(), allFeatures.size(), statistics.mean(), statistics.stdDev(), analysisTimeMs);
    }
}
