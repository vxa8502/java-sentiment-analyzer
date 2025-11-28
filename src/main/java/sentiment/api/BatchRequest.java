package sentiment.api;

import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Size;

import java.util.List;

import static sentiment.api.ValidationConstants.*;

/**
 * Request DTO for batch sentiment analysis.
 */
public record BatchRequest(
    @NotEmpty(message = BATCH_TOO_SMALL_MESSAGE)
    @Size(min = MIN_BATCH_SIZE, max = MAX_BATCH_SIZE, message = BATCH_TOO_LARGE_MESSAGE)
    @ValidTextList
    List<String> texts,

    Double confidenceThreshold
) {
    public BatchRequest(List<String> texts) {
        this(texts, null);
    }
}
