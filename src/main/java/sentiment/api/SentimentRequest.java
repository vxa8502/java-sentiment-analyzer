package sentiment.api;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

import static sentiment.api.ValidationConstants.*;

/**
 * Request DTO for single text sentiment analysis.
 */
public record SentimentRequest(
    @NotBlank(message = TEXT_BLANK_MESSAGE)
    @Size(min = MIN_TEXT_LENGTH, max = MAX_TEXT_LENGTH, message = TEXT_TOO_LONG_MESSAGE)
    String text,

    Double confidenceThreshold
) {
    // Compact constructor for default null handling
    public SentimentRequest(String text) {
        this(text, null);
    }
}
