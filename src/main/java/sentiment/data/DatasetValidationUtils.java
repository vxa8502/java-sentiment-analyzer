package sentiment.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;
import java.util.stream.Collectors;

/**
 * Three-layer validation utility following clean architecture principles:
 * Layer 1: Raw Input Validation (data quality)
 * Layer 2: Processing Validation (transformation quality)
 * Layer 3: Model Input Validation (feature quality)
 */
public class DatasetValidationUtils {
    private static final Logger logger = LoggerFactory.getLogger(DatasetValidationUtils.class);

    // Raw input validation thresholds
    public static final int MIN_RAW_TEXT_LENGTH = 5;
    public static final int MAX_RAW_TEXT_LENGTH = 10000;

    // Processing validation thresholds
    public static final int MIN_PROCESSED_TEXT_LENGTH = 10;
    public static final int MAX_PROCESSED_TEXT_LENGTH = 5000;
    public static final double MIN_TEXT_RETENTION_RATIO = 0.1; // After cleaning, keep at least 10%

    // Model input validation thresholds
    public static final int MIN_FEATURE_COUNT = 3;

    // Quality control thresholds
    public static final double MAX_DUPLICATE_RATIO = 0.1;
    public static final double MIN_SUCCESS_RATE = 0.3;
    public static final double SEVERE_IMBALANCE_THRESHOLD = 0.9;

    // Thread-local duplicate tracker per loading session
    private static final ThreadLocal<Set<String>> DUPLICATE_TRACKER =
            ThreadLocal.withInitial(HashSet::new);

    public static void validateFilePath(String filePath, String datasetType) throws DataLoadingException {
        if (filePath == null || filePath.trim().isEmpty()) {
            throw new DataLoadingException("File path cannot be null or empty", filePath, datasetType);
        }

        Path path = Paths.get(filePath);
        if (!Files.exists(path)) {
            throw DataLoadingException.fileNotFound(filePath, datasetType);
        }

        if (!Files.isReadable(path)) {
            throw DataLoadingException.fileNotReadable(filePath, datasetType);
        }

        // Initialize duplicate tracking for this loading session
        initializeDuplicateTracking();

        try {
            long fileSize = Files.size(path);
            if (fileSize > 100 * 1024 * 1024) {
                logger.warn("Large file detected: {} MB", fileSize / (1024 * 1024));
            }
        } catch (Exception e) {
            logger.debug("Could not check file size: {}", e.getMessage());
        }
    }

    /**
     * Validates raw text data before any processing.
     * Checks basic data quality issues like null values, encoding, length.
     */
    public static ValidationResult validateRawText(String text, String datasetType,
                                                   int recordNumber, DatasetLoadingStats stats) {
        if (text == null) {
            stats.incrementNullOrEmpty();
            return ValidationResult.invalid("NULL_TEXT", recordNumber);
        }

        text = text.trim();
        if (text.isEmpty()) {
            stats.incrementNullOrEmpty();
            return ValidationResult.invalid("EMPTY_TEXT", recordNumber);
        }

        // Check for encoding issues
        if (!StandardCharsets.UTF_8.newEncoder().canEncode(text)) {
            stats.incrementInvalidEncoding();
            logger.debug("Invalid encoding in {} record {}", datasetType, recordNumber);
            return ValidationResult.invalid("INVALID_ENCODING", recordNumber);
        }

        // Raw length validation (more permissive than processed text)
        if (text.length() < MIN_RAW_TEXT_LENGTH) {
            stats.incrementTooShortRaw();
            logger.debug("Raw text too short in {} record {}: {} chars",
                    datasetType, recordNumber, text.length());
            return ValidationResult.invalid("TOO_SHORT_RAW", recordNumber);
        }

        if (text.length() > MAX_RAW_TEXT_LENGTH) {
            stats.incrementTooLongRaw();
            logger.debug("Raw text too long in {} record {}: {} chars",
                    datasetType, recordNumber, text.length());
            // Don't reject, just truncate and warn
            text = text.substring(0, MAX_RAW_TEXT_LENGTH) + "...";
        }

        return ValidationResult.valid(text);
    }

    /**
     * Validates raw sentiment labels before processing.
     */
    public static ValidationResult validateRawSentiment(String sentimentStr, String datasetType,
                                                        int recordNumber, DatasetLoadingStats stats) {
        if (sentimentStr == null || sentimentStr.trim().isEmpty()) {
            stats.incrementMissingRequiredFields();
            return ValidationResult.invalid("MISSING_SENTIMENT", recordNumber);
        }

        try {
            Dataset.SentimentLabel sentiment = Dataset.SentimentLabel.fromString(sentimentStr.trim());
            return ValidationResult.valid(sentiment);
        } catch (IllegalArgumentException e) {
            logger.warn("Invalid sentiment label '{}' in {} record {}",
                    sentimentStr, datasetType, recordNumber);
            stats.incrementMissingRequiredFields();
            return ValidationResult.invalid("INVALID_SENTIMENT_FORMAT", recordNumber);
        }
    }

    /**
     * Validates text after preprocessing/cleaning operations.
     * Checks if cleaning was too aggressive or failed.
     */
    public static ValidationResult validateProcessedText(String cleanedText, String originalText,
                                                         String datasetType, int recordNumber,
                                                         DatasetLoadingStats stats) {
        if (cleanedText == null) {
            stats.incrementProcessingFailure();
            return ValidationResult.invalid("PROCESSING_FAILURE", recordNumber);
        }

        cleanedText = cleanedText.trim();
        if (cleanedText.isEmpty()) {
            stats.incrementOverCleaned();
            logger.debug("Text completely cleaned away in {} record {}", datasetType, recordNumber);
            return ValidationResult.invalid("OVER_CLEANED_EMPTY", recordNumber);
        }

        // Check if text was over-cleaned
        if (isOverCleaned(originalText, cleanedText, MIN_TEXT_RETENTION_RATIO)) {
            stats.incrementOverCleaned();
            logger.debug("Text over-cleaned in {} record {} (was {} chars, now {} chars)",
                    datasetType, recordNumber, originalText.length(), cleanedText.length());
            return ValidationResult.invalid("OVER_CLEANED_RATIO", recordNumber);
        }

        // Check processed length requirements (stricter than raw)
        if (cleanedText.length() < MIN_PROCESSED_TEXT_LENGTH) {
            stats.incrementInvalidAfterCleaning();
            logger.debug("Cleaned text too short in {} record {}: {} chars",
                    datasetType, recordNumber, cleanedText.length());
            return ValidationResult.invalid("TOO_SHORT_AFTER_CLEANING", recordNumber);
        }

        if (cleanedText.length() > MAX_PROCESSED_TEXT_LENGTH) {
            logger.debug("Truncating processed text in {} record {} from {} to {} chars",
                    datasetType, recordNumber, cleanedText.length(), MAX_PROCESSED_TEXT_LENGTH);
            cleanedText = cleanedText.substring(0, MAX_PROCESSED_TEXT_LENGTH) + "...";
        }

        // Check for duplicates using thread-local tracker
        String normalizedText = cleanedText.toLowerCase();
        if (!DUPLICATE_TRACKER.get().add(normalizedText)) {
            stats.incrementDuplicate();
            logger.debug("Duplicate text found in {} record {}", datasetType, recordNumber);
            return ValidationResult.invalid("DUPLICATE_TEXT", recordNumber);
        }

        return ValidationResult.valid(cleanedText);
    }

    /**
     * Validates that processed text is suitable for model input.
     * Checks feature extraction requirements.
     */
    public static ValidationResult validateModelInput(String processedText, Dataset.SentimentLabel sentiment,
                                                      String datasetType, int recordNumber,
                                                      DatasetLoadingStats stats) {
        // Estimate feature count (simple word count for now)
        int estimatedFeatureCount = processedText.split("\\s+").length;

        if (estimatedFeatureCount < MIN_FEATURE_COUNT) {
            stats.incrementInsufficientFeatures();
            logger.debug("Insufficient features in {} record {}: {} words",
                    datasetType, recordNumber, estimatedFeatureCount);
            return ValidationResult.invalid("INSUFFICIENT_FEATURES", recordNumber);
        }

        // Validate sentiment label is properly parsed
        if (sentiment == null) {
            stats.incrementInvalidLabel();
            return ValidationResult.invalid("NULL_SENTIMENT", recordNumber);
        }

        // Create Dataset using the Builder pattern to handle SentimentLabel properly
        Dataset dataset = new Dataset.Builder(processedText, sentiment).build();
        return ValidationResult.valid(dataset);
    }

    /**
     * Performs quality control checks that span multiple layers.
     */
    public static ValidationResult performQualityControl(String text, String datasetType,
                                                         int recordNumber, DatasetLoadingStats stats) {
        // Simple spam detection
        if (isSpam(text)) {
            stats.incrementSpam();
            return ValidationResult.invalid("SPAM_DETECTED", recordNumber);
        }

        // Simple fake review detection
        if (isFakeReview(text)) {
            stats.incrementFakeReview();
            return ValidationResult.invalid("FAKE_REVIEW", recordNumber);
        }

        return ValidationResult.valid(text);
    }

    public static void validateFinalDataset(List<Dataset> datasets, DatasetLoadingStats stats,
                                            String filePath, String datasetType) throws DataLoadingException {
        // Clean up duplicate tracking
        clearDuplicateTracking();

        if (datasets.isEmpty()) {
            throw DataLoadingException.insufficientData(filePath, datasetType, 0);
        }

        double successRate = stats.getSuccessRate();
        if (successRate < MIN_SUCCESS_RATE) {
            throw new DataLoadingException(
                    String.format("Very low success rate: %.1f%% - file may be corrupted",
                            successRate * 100), filePath, datasetType
            );
        }

        double duplicateRatio = (double) stats.getDuplicateRecords() / stats.getTotalRecords();
        if (duplicateRatio > MAX_DUPLICATE_RATIO) {
            String percentage = String.format("%.1f", duplicateRatio * 100);
            logger.warn("High duplicate rate: {}% ({} duplicates out of {} records)",
                    percentage, stats.getDuplicateRecords(), stats.getTotalRecords());
        }

        analyzeSentimentDistribution(datasets, datasetType);
    }

    // Helper Methods

    public static void initializeDuplicateTracking() {
        DUPLICATE_TRACKER.get().clear();
    }

    public static void clearDuplicateTracking() {
        DUPLICATE_TRACKER.remove();
    }

    public static boolean isOverCleaned(String originalText, String cleanedText, double threshold) {
        if (originalText == null || cleanedText == null) return false;
        return cleanedText.length() < originalText.length() * threshold;
    }

    public static void analyzeSentimentDistribution(List<Dataset> datasets, String datasetType) {
        Map<Dataset.SentimentLabel, Long> distribution = datasets.stream()
                .collect(Collectors.groupingBy(Dataset::getSentiment, Collectors.counting()));

        logger.info("{} sentiment distribution: {}", datasetType, distribution);

        if (distribution.size() == 1) {
            logger.warn("All samples have the same sentiment - this may indicate a problem");
            return;
        }

        long totalSentiments = distribution.values().stream().mapToLong(Long::longValue).sum();
        if (totalSentiments > 0) {
            long maxCount = distribution.values().stream().mapToLong(Long::longValue).max().orElse(0);
            double maxRatio = (double) maxCount / totalSentiments;
            String percentage = String.format("%.1f", maxRatio * 100);
            if (maxRatio > SEVERE_IMBALANCE_THRESHOLD) {
                logger.warn("Severely imbalanced dataset: {} have the same sentiment",
                        percentage);
            }
        }
    }

    public static String truncateForLogging(String text, int maxLength) {
        if (text == null) return "null";
        return text.length() > maxLength ? text.substring(0, maxLength) + "..." : text;
    }

    // Simple spam detection (placeholder - could be enhanced)
    private static boolean isSpam(String text) {
        String lowerText = text.toLowerCase();
        return lowerText.contains("buy now") || lowerText.contains("click here") ||
                lowerText.contains("amazing deal") || lowerText.matches(".*!{5,}.*");
    }

    // Simple fake review detection (placeholder - could be enhanced)
    private static boolean isFakeReview(String text) {
        String lowerText = text.toLowerCase();
        return lowerText.matches(".*(perfect|amazing|incredible|outstanding|excellent).{0,10}(product|service|item).{0,10}(perfect|amazing|incredible|outstanding|excellent).*") ||
                text.matches(".*[A-Z]{10,}.*"); // All caps sections might indicate fake reviews
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
        public Dataset getDataset() { return (Dataset) validatedData; }
    }
}