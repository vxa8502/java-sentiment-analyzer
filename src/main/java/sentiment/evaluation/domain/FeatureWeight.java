package sentiment.evaluation.domain;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Represents a single feature with its importance weight and statistical significance.
 *
 * @param featureName The name of the feature (word, n-gram, or other text feature)
 * @param weight The importance weight (positive = positive sentiment, negative = negative sentiment)
 * @param significance Statistical significance score (absolute value of normalized weight)
 */
public record FeatureWeight(
        @JsonProperty("featureName") String featureName,
        @JsonProperty("weight") double weight,
        @JsonProperty("significance") double significance
) {
    /**
     * Returns the sentiment direction this feature indicates.
     *
     * @return "positive", "negative", or "neutral"
     */
    public String direction() {
        return weight > 0 ? "positive" : (weight < 0 ? "negative" : "neutral");
    }

    @Override
    public String toString() {
        return String.format("FeatureWeight{%s: %.6f (sig=%.4f)}",
                featureName, weight, significance);
    }
}
