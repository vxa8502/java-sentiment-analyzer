package sentiment.api;

import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * Response DTO for sentiment analysis results.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class SentimentResponse {

    private String sentiment;
    private Double confidence;
    private String text;
    private Long processingTimeMs;
    private String error;

    public SentimentResponse() {
    }

    public SentimentResponse(String sentiment, Double confidence) {
        this.sentiment = sentiment;
        this.confidence = confidence;
    }

    public static SentimentResponse success(String sentiment, Double confidence, String text, Long processingTimeMs) {
        SentimentResponse response = new SentimentResponse(sentiment, confidence);
        response.setText(text);
        response.setProcessingTimeMs(processingTimeMs);
        return response;
    }

    public static SentimentResponse error(String error, String text) {
        SentimentResponse response = new SentimentResponse();
        response.setError(error);
        response.setText(text);
        return response;
    }

    public String getSentiment() {
        return sentiment;
    }

    public void setSentiment(String sentiment) {
        this.sentiment = sentiment;
    }

    public Double getConfidence() {
        return confidence;
    }

    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public Long getProcessingTimeMs() {
        return processingTimeMs;
    }

    public void setProcessingTimeMs(Long processingTimeMs) {
        this.processingTimeMs = processingTimeMs;
    }

    public String getError() {
        return error;
    }

    public void setError(String error) {
        this.error = error;
    }
}
