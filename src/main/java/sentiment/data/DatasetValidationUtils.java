package sentiment.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Simplified validation utility for data loading - structural validation only.
 * Validates that data can be read and parsed, nothing more.
 */
public class DatasetValidationUtils {
    private static final Logger logger = LoggerFactory.getLogger(DatasetValidationUtils.class);

    // Safety limits for memory protection
    public static final int MAX_TEXT_LENGTH = 50000; // Safety limit for memory

    // File Validation

    public static void validateFilePath(String filePath, String datasetType) throws DataLoadingException {
        if (filePath == null || filePath.trim().isEmpty()) {
            throw new DataLoadingException("File path cannot be null or empty", filePath, datasetType);
        }

        Path path = Paths.get(filePath);
        if (!Files.exists(path)) {
            throw new DataLoadingException("File does not exist: " + filePath, filePath, datasetType);
        }

        if (!Files.isReadable(path)) {
            throw new DataLoadingException("File is not readable: " + filePath, filePath, datasetType);
        }

        try {
            long fileSize = Files.size(path);
            if (fileSize > 500 * 1024 * 1024) { // 500MB warning
                logger.warn("Large file detected: {} MB", fileSize / (1024 * 1024));
            }
            if (fileSize == 0) {
                throw new DataLoadingException("File is empty", filePath, datasetType);
            }
        } catch (Exception e) {
            if (e instanceof DataLoadingException) throw (DataLoadingException) e;
            logger.debug("Could not check file size: {}", e.getMessage());
        }
    }

    /**
     * Validates that text field exists and can be read.
     * Does NOT transform or judge content quality.
     */
    public static ValidationResult validateRawText(String text, String datasetType,
                                                   int recordNumber, DatasetLoadingStats stats) {
        if (text == null || text.trim().isEmpty()) {
            stats.incrementNullOrEmptyText();
            return ValidationResult.invalid("NULL_OR_EMPTY_TEXT", recordNumber);
        }

        String trimmedText = text.trim();

        // Check for safety limits
        if (trimmedText.length() > MAX_TEXT_LENGTH) {
            stats.incrementParseErrors();
            logger.warn("Text exceeds safety limit in {} record {}: {} chars",
                    datasetType, recordNumber, trimmedText.length());
            return ValidationResult.invalid("EXCEEDS_SAFETY_LIMIT", recordNumber);
        }

        // Check for encoding issues that would prevent processing
        if (!StandardCharsets.UTF_8.newEncoder().canEncode(trimmedText)) {
            stats.incrementParseErrors();
            logger.debug("Invalid encoding in {} record {}", datasetType, recordNumber);
            return ValidationResult.invalid("INVALID_ENCODING", recordNumber);
        }

        // Return the trimmed text - minimal cleaning for data integrity only
        return ValidationResult.valid(trimmedText);
    }

    /**
     * Validates that sentiment field exists and contains a parseable value.
     */
    public static ValidationResult validateRawSentiment(String sentimentStr, String datasetType,
                                                        int recordNumber, DatasetLoadingStats stats) {
        if (sentimentStr == null || sentimentStr.trim().isEmpty()) {
            stats.incrementInvalidSentimentFormat();
            return ValidationResult.invalid("NULL_SENTIMENT", recordNumber);
        }

        try {
            // Just verify it can be parsed
            Dataset.SentimentLabel sentiment = Dataset.SentimentLabel.fromString(sentimentStr);
            return ValidationResult.valid(sentiment);
        } catch (IllegalArgumentException e) {
            logger.debug("Cannot parse sentiment label '{}' in {} record {}",
                    sentimentStr, datasetType, recordNumber);
            stats.incrementInvalidSentimentFormat();
            return ValidationResult.invalid("UNPARSEABLE_SENTIMENT", recordNumber);
        }
    }

    /**
     * Validates confidence value is within valid range [0.0, 1.0].
     */
    public static ValidationResult validateConfidence(Double confidence, String datasetType,
                                                      int recordNumber, DatasetLoadingStats stats) {
        if (confidence == null) {
            // Null confidence is acceptable - will use default
            return ValidationResult.valid(null);
        }

        if (confidence < 0.0 || confidence > 1.0) {
            stats.incrementParseErrors();
            logger.debug("Invalid confidence value {} in {} record {}",
                    confidence, datasetType, recordNumber);
            return ValidationResult.invalid("INVALID_CONFIDENCE_RANGE", recordNumber);
        }

        return ValidationResult.valid(confidence);
    }

    /**
     * Validates that we successfully loaded some data.
     */
    public static void validateFinalDataset(List<Dataset> datasets, DatasetLoadingStats stats,
                                            String filePath, String datasetType) throws DataLoadingException {
        if (datasets.isEmpty()) {
            throw new DataLoadingException(
                    "No valid data found. Processed " + stats.getTotalRecords() + " records, 0 successful",
                    filePath,
                    datasetType);
        }

        // Log basic statistics
        logBasicDistribution(datasets, datasetType);
    }

    /**
     * Log sentiment distribution for informational purposes only.
     */
    private static void logBasicDistribution(List<Dataset> datasets, String datasetType) {
        Map<Dataset.SentimentLabel, Long> distribution = datasets.stream()
                .collect(Collectors.groupingBy(Dataset::getSentiment, Collectors.counting()));

        logger.info("{} loaded: {} total samples with distribution: {}",
                datasetType, datasets.size(), distribution);
    }

    /**
     * Simple truncation for logging.
     */
    public static String truncateForLogging(String text, int maxLength) {
        if (text == null) return "null";
        if (text.length() <= maxLength) return text;
        return text.substring(0, maxLength) + "...";
    }

    // Validation Result Class

    public static class ValidationResult {
        private final boolean valid;
        private final String errorType;
        private final int recordNumber;
        private final Object validatedData;

        private ValidationResult(boolean valid, String errorType, int recordNumber, Object validatedData) {
            this.valid = valid;
            this.errorType = errorType;
            this.recordNumber = recordNumber;
            this.validatedData = validatedData;
        }

        public static ValidationResult valid(Object data) {
            return new ValidationResult(true, null, -1, data);
        }

        public static ValidationResult invalid(String errorType, int recordNumber) {
            return new ValidationResult(false, errorType, recordNumber, null);
        }

        public boolean isValid() { return valid; }
        public String getErrorType() { return errorType; }
        public int getRecordNumber() { return recordNumber; }

        @SuppressWarnings("unchecked")
        public <T> T getData() { return (T) validatedData; }

        public String getText() { return (String) validatedData; }
        public Dataset.SentimentLabel getSentiment() { return (Dataset.SentimentLabel) validatedData; }
        public Double getConfidence() { return (Double) validatedData; }
    }
}