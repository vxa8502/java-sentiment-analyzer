package sentiment.data;

import org.slf4j.Logger;

/**
 * Simplified statistics class for tracking basic data loading issues only.
 * Tracks only fundamental parsing and validation errors.
 */
public class DatasetLoadingStats {

    // Core statistics
    private int totalRecords = 0;
    private int successfulRecords = 0;

    // Basic validation errors
    private int nullOrEmptyText = 0;
    private int invalidSentimentFormat = 0;
    private int parseErrors = 0;

    // Core Statistics
    public void incrementTotal() { totalRecords++; }
    public void incrementSuccessful() { successfulRecords++; }

    // Basic Validation Errors
    public void incrementNullOrEmptyText() { nullOrEmptyText++; }
    public void incrementInvalidSentimentFormat() { invalidSentimentFormat++; }
    public void incrementParseErrors() { parseErrors++; }

    // Getters
    public int getTotalRecords() { return totalRecords; }
    public int getSuccessfulRecords() { return successfulRecords; }
    public int getNullOrEmptyText() { return nullOrEmptyText; }
    public int getInvalidSentimentFormat() { return invalidSentimentFormat; }
    public int getParseErrors() { return parseErrors; }

    // Calculated Metrics
    public double getSuccessRate() {
        return totalRecords > 0 ? (double) successfulRecords / totalRecords : 0.0;
    }

    public int getTotalErrors() {
        return nullOrEmptyText + invalidSentimentFormat + parseErrors;
    }

    public double getErrorRate() {
        return totalRecords > 0 ? (double) getTotalErrors() / totalRecords : 0.0;
    }

    // Logging Helpers
    public void logSummary(Logger logger, String filePath, String datasetType) {
        logger.info("=== {} Loading Summary: {} ===", datasetType, getFileName(filePath));
        logger.info("Total records attempted: {}", totalRecords);
        String successRatePercent = String.format("%.2f%%", getSuccessRate() * 100);
        logger.info("Successfully loaded: {} {}",
                successfulRecords, successRatePercent);

        if (getTotalErrors() > 0) {
            logger.info("--- Loading Issues ({}) ---", getTotalErrors());
            logNonZero(logger, "  Null/Empty text", nullOrEmptyText);
            logNonZero(logger, "  Invalid sentiment format", invalidSentimentFormat);
            logNonZero(logger, "  Parse errors", parseErrors);
        }
    }

    private void logNonZero(Logger logger, String label, int count) {
        if (count > 0) {
            logger.info("{}: {}", label, count);
        }
    }

    private String getFileName(String filePath) {
        if (filePath == null || filePath.isEmpty()) {
            return "unknown file";
        }

        int lastSeparator = Math.max(filePath.lastIndexOf('/'), filePath.lastIndexOf('\\'));
        if (lastSeparator >= 0 && lastSeparator < filePath.length() - 1) {
            return filePath.substring(lastSeparator + 1);
        }

        return filePath;
    }

    @Override
    public String toString() {
        return String.format("DatasetLoadingStats{total=%d, successful=%d, successRate=%.1f%%, errors=%d}",
                totalRecords, successfulRecords, getSuccessRate() * 100, getTotalErrors());
    }
}