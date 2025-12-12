package sentiment.evaluation.domain;

import com.fasterxml.jackson.annotation.JsonProperty;
import org.slf4j.Logger;

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
     * @deprecated Use {@link #logTopFeatures(int, Logger)} instead
     *
     * @param limit Maximum number of features to print
     */
    @Deprecated
    public void printTopFeatures(int limit) {
        System.out.println("\n=== TOP DISCRIMINATIVE FEATURES ===");
        topFeatures.stream()
                .limit(limit)
                .forEach(fw -> System.out.printf("%40s: weight=%+.6f, significance=%.4f\n",
                        fw.featureName(), fw.weight(), fw.significance()));
        System.out.println("===================================\n");
    }

    /**
     * Logs the top N features using the provided logger.
     *
     * @param limit Maximum number of features to log
     * @param logger Logger instance to use for output
     */
    public void logTopFeatures(int limit, Logger logger) {
        logger.info("");
        logger.info("=== TOP DISCRIMINATIVE FEATURES ===");
        topFeatures.stream()
                .limit(limit)
                .forEach(fw -> logger.info("{}: weight={}, significance={}",
                        String.format("%40s", fw.featureName()),
                        String.format("%+.6f", fw.weight()),
                        String.format("%.4f", fw.significance())));
        logger.info("===================================");
    }

    @Override
    public String toString() {
        return String.format("FeatureImportanceResult{top=%d/%d features, mean=%.4f, std=%.4f, time=%dms}",
                topFeatures.size(), allFeatures.size(), statistics.mean(), statistics.stdDev(), analysisTimeMs);
    }
}
