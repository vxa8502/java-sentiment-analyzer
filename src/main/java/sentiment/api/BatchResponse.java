package sentiment.api;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.List;

/**
 * Response DTO for batch sentiment analysis.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public record BatchResponse(
    List<SentimentResponse> results,
    Integer totalProcessed,
    Integer successCount,
    Integer errorCount,
    Long totalProcessingTimeMs
) {
    public static BatchResponse fromResults(List<SentimentResponse> results, Long totalProcessingTimeMs) {
        int totalProcessed = results.size();
        int successCount = (int) results.stream().filter(r -> r.error() == null).count();
        int errorCount = (int) results.stream().filter(r -> r.error() != null).count();

        return new BatchResponse(results, totalProcessed, successCount, errorCount, totalProcessingTimeMs);
    }
}
