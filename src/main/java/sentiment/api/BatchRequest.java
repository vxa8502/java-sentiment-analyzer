package sentiment.api;

import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Size;

import java.util.List;

import static sentiment.api.ValidationConstants.*;

/**
 * Request DTO for batch sentiment analysis.
 *
 * Validates both the batch size and individual text lengths to prevent:
 * - Overwhelming the system with too many requests
 * - Bypassing single-request validation with oversized texts
 */
public class BatchRequest {

    @NotEmpty(message = BATCH_TOO_SMALL_MESSAGE)
    @Size(min = MIN_BATCH_SIZE, max = MAX_BATCH_SIZE, message = BATCH_TOO_LARGE_MESSAGE)
    @ValidTextList
    private List<String> texts;

    private Double confidenceThreshold;

    public BatchRequest() {
    }

    public BatchRequest(List<String> texts) {
        this.texts = texts;
    }

    public BatchRequest(List<String> texts, Double confidenceThreshold) {
        this.texts = texts;
        this.confidenceThreshold = confidenceThreshold;
    }

    public List<String> getTexts() {
        return texts;
    }

    public void setTexts(List<String> texts) {
        this.texts = texts;
    }

    public Double getConfidenceThreshold() {
        return confidenceThreshold;
    }

    public void setConfidenceThreshold(Double confidenceThreshold) {
        this.confidenceThreshold = confidenceThreshold;
    }
}
