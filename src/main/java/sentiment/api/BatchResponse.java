package sentiment.api;

import com.fasterxml.jackson.annotation.JsonInclude;

import java.util.List;

/**
 * Response DTO for batch sentiment analysis.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public class BatchResponse {

    private List<SentimentResponse> results;
    private Integer totalProcessed;
    private Integer successCount;
    private Integer errorCount;
    private Long totalProcessingTimeMs;

    public BatchResponse() {
    }

    public BatchResponse(List<SentimentResponse> results) {
        this.results = results;
        this.totalProcessed = results.size();
        this.successCount = (int) results.stream().filter(r -> r.getError() == null).count();
        this.errorCount = (int) results.stream().filter(r -> r.getError() != null).count();
    }

    public List<SentimentResponse> getResults() {
        return results;
    }

    public void setResults(List<SentimentResponse> results) {
        this.results = results;
    }

    public Integer getTotalProcessed() {
        return totalProcessed;
    }

    public void setTotalProcessed(Integer totalProcessed) {
        this.totalProcessed = totalProcessed;
    }

    public Integer getSuccessCount() {
        return successCount;
    }

    public void setSuccessCount(Integer successCount) {
        this.successCount = successCount;
    }

    public Integer getErrorCount() {
        return errorCount;
    }

    public void setErrorCount(Integer errorCount) {
        this.errorCount = errorCount;
    }

    public Long getTotalProcessingTimeMs() {
        return totalProcessingTimeMs;
    }

    public void setTotalProcessingTimeMs(Long totalProcessingTimeMs) {
        this.totalProcessingTimeMs = totalProcessingTimeMs;
    }
}
