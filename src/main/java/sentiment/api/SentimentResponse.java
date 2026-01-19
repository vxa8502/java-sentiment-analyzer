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
    String error,
    String warning
) {
    public static SentimentResponse success(String sentiment, Double confidence, String text, Long processingTimeMs) {
        return new SentimentResponse(sentiment, confidence, text, processingTimeMs, null, null);
    }

    public static SentimentResponse successWithWarning(String sentiment, Double confidence, String text,
                                                        Long processingTimeMs, String warning) {
        return new SentimentResponse(sentiment, confidence, text, processingTimeMs, null, warning);
    }

    public static SentimentResponse error(String error, String text) {
        return new SentimentResponse(null, null, text, null, error, null);
    }
}
