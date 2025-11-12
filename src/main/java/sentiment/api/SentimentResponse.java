package sentiment.api;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * Response DTO for sentiment analysis results.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record SentimentResponse(
    String sentiment,
    Double confidence,
    String text,
    Long processingTimeMs,
    String error
) {
    public static SentimentResponse success(String sentiment, Double confidence, String text, Long processingTimeMs) {
        return new SentimentResponse(sentiment, confidence, text, processingTimeMs, null);
    }

    public static SentimentResponse error(String error, String text) {
        return new SentimentResponse(null, null, text, null, error);
    }
}
