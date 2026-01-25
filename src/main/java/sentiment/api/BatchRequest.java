package sentiment.api;

import jakarta.validation.constraints.DecimalMax;
import jakarta.validation.constraints.DecimalMin;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Size;

import java.util.List;

import static sentiment.api.ValidationConstants.*;

/**
 * Request DTO for batch sentiment analysis.
 */
public record BatchRequest(
    @NotEmpty(message = BATCH_TOO_SMALL_MESSAGE)
    @Size(max = MAX_BATCH_SIZE, message = BATCH_TOO_LARGE_MESSAGE)
    List<@NotBlank(message = TEXT_BLANK_MESSAGE)
        @Size(max = MAX_TEXT_LENGTH, message = TEXT_TOO_LONG_MESSAGE) String> texts,

    @DecimalMin(value = "0.0", message = CONFIDENCE_THRESHOLD_RANGE_MESSAGE)
    @DecimalMax(value = "1.0", message = CONFIDENCE_THRESHOLD_RANGE_MESSAGE)
    Double confidenceThreshold
) {
    public BatchRequest(List<String> texts) {
        this(texts, null);
    }
}
