package sentiment.api;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

import static sentiment.api.ValidationConstants.*;

/**
 * Request DTO for single text sentiment analysis.
 */
public record SentimentRequest(
    @NotBlank(message = TEXT_BLANK_MESSAGE)
    @Size(max = MAX_TEXT_LENGTH, message = TEXT_TOO_LONG_MESSAGE)
    String text,

    @DecimalMin(value = "0.0", message = CONFIDENCE_THRESHOLD_RANGE_MESSAGE)
    @DecimalMax(value = "1.0", message = CONFIDENCE_THRESHOLD_RANGE_MESSAGE)
    Double confidenceThreshold
) {
    // Compact constructor for default null handling
    public SentimentRequest(String text) {
        this(text, null);
    }
}
