package sentiment.api;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

import static sentiment.api.ValidationConstants.*;

/**
 * Request DTO for single text sentiment analysis.
 */
public class SentimentRequest {

    @NotBlank(message = TEXT_BLANK_MESSAGE)
    @Size(min = MIN_TEXT_LENGTH, max = MAX_TEXT_LENGTH, message = TEXT_TOO_LONG_MESSAGE)
    private String text;

    private Double confidenceThreshold;

    public SentimentRequest() {
    }

    public SentimentRequest(String text) {
        this.text = text;
    }

    public SentimentRequest(String text, Double confidenceThreshold) {
        this.text = text;
        this.confidenceThreshold = confidenceThreshold;
    }

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public Double getConfidenceThreshold() {
        return confidenceThreshold;
    }

    public void setConfidenceThreshold(Double confidenceThreshold) {
        this.confidenceThreshold = confidenceThreshold;
    }
}
