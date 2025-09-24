package sentiment.data;

import org.slf4j.Logger;

import java.util.HashMap;
import java.util.Map;

/**
 * Enhanced statistics class with distinct validation layers for tracking dataset loading progress.
 * Implements the three-layer validation pattern: Raw Input -> Processing -> Model Input
 */
public class DatasetLoadingStats {

    // Core statistics
    private int totalRecords = 0;
    private int successfulRecords = 0;

    // Layer 1: Raw Input Validation Errors
    private int nullOrEmptyRecords = 0;
    private int tooShortRawRecords = 0;
    private int tooLongRawRecords = 0;
    private int invalidEncodingRecords = 0;
    private int missingRequiredFieldsRecords = 0;

    // Layer 2: Processing Validation Errors
    private int overCleanedRecords = 0;
    private int processingFailureRecords = 0;
    private int invalidAfterCleaningRecords = 0;

    // Layer 3: Model Input Validation Errors
    private int insufficientFeaturesRecords = 0;
    private int invalidLabelRecords = 0;
    private int featureExtractionFailureRecords = 0;

    // Quality Control (can span multiple layers)
    private int duplicateRecords = 0;
    private int spamRecords = 0;
    private int fakeReviewRecords = 0;

    // Custom counters for loader-specific issues
    private final Map<String, Integer> customCounters = new HashMap<>();

    public void incrementTotal() { totalRecords++; }
    public void incrementSuccessful() { successfulRecords++; }

    public void incrementNullOrEmpty() { nullOrEmptyRecords++; }
    public void incrementTooShortRaw() { tooShortRawRecords++; }
    public void incrementTooLongRaw() { tooLongRawRecords++; }
    public void incrementInvalidEncoding() { invalidEncodingRecords++; }
    public void incrementMissingRequiredFields() { missingRequiredFieldsRecords++; }

    public void incrementOverCleaned() { overCleanedRecords++; }
    public void incrementProcessingFailure() { processingFailureRecords++; }
    public void incrementInvalidAfterCleaning() { invalidAfterCleaningRecords++; }

    public void incrementInsufficientFeatures() { insufficientFeaturesRecords++; }
    public void incrementInvalidLabel() { invalidLabelRecords++; }
    public void incrementFeatureExtractionFailure() { featureExtractionFailureRecords++; }

    public void incrementDuplicate() { duplicateRecords++; }
    public void incrementSpam() { spamRecords++; }
    public void incrementFakeReview() { fakeReviewRecords++; }

    public void incrementCustom(String key) {
        customCounters.merge(key, 1, Integer::sum);
    }

    public void setCustom(String key, int value) {
        customCounters.put(key, value);
    }

    public int getTotalRecords() { return totalRecords; }
    public int getSuccessfulRecords() { return successfulRecords; }

    // Layer 1 getters
    public int getNullOrEmptyRecords() { return nullOrEmptyRecords; }
    public int getTooShortRawRecords() { return tooShortRawRecords; }
    public int getTooLongRawRecords() { return tooLongRawRecords; }
    public int getInvalidEncodingRecords() { return invalidEncodingRecords; }
    public int getMissingRequiredFieldsRecords() { return missingRequiredFieldsRecords; }

    // Layer 2 getters
    public int getOverCleanedRecords() { return overCleanedRecords; }
    public int getProcessingFailureRecords() { return processingFailureRecords; }
    public int getInvalidAfterCleaningRecords() { return invalidAfterCleaningRecords; }

    // Layer 3 getters
    public int getInsufficientFeaturesRecords() { return insufficientFeaturesRecords; }
    public int getInvalidLabelRecords() { return invalidLabelRecords; }
    public int getFeatureExtractionFailureRecords() { return featureExtractionFailureRecords; }

    // Quality control getters
    public int getDuplicateRecords() { return duplicateRecords; }
    public int getSpamRecords() { return spamRecords; }
    public int getFakeReviewRecords() { return fakeReviewRecords; }

    public int getCustom(String key) { return customCounters.getOrDefault(key, 0); }

    public double getSuccessRate() {
        return totalRecords > 0 ? (double) successfulRecords / totalRecords : 0.0;
    }

    public int getRawInputErrors() {
        return nullOrEmptyRecords + tooShortRawRecords + tooLongRawRecords +
                invalidEncodingRecords + missingRequiredFieldsRecords;
    }

    public int getProcessingErrors() {
        return overCleanedRecords + processingFailureRecords + invalidAfterCleaningRecords;
    }

    public int getModelInputErrors() {
        return insufficientFeaturesRecords + invalidLabelRecords + featureExtractionFailureRecords;
    }

    public int getQualityControlRejections() {
        return duplicateRecords + spamRecords + fakeReviewRecords;
    }

    public int getTotalErrors() {
        return getRawInputErrors() + getProcessingErrors() + getModelInputErrors();
    }

    public int getTotalSkipped() {
        return getTotalErrors() + getQualityControlRejections();
    }

    public double getRawInputErrorRate() {
        return totalRecords > 0 ? (double) getRawInputErrors() / totalRecords : 0.0;
    }

    public double getProcessingErrorRate() {
        return totalRecords > 0 ? (double) getProcessingErrors() / totalRecords : 0.0;
    }

    public double getModelInputErrorRate() {
        return totalRecords > 0 ? (double) getModelInputErrors() / totalRecords : 0.0;
    }

    public void logLayeredSummary(Logger logger, String filePath, String datasetType) {
        logger.info("=== {} Loading Summary: {} ===", datasetType, filePath);
        logger.info("Total records processed: {}", totalRecords);
        String percentage = String.format("%.1f", getSuccessRate() * 100);
        logger.info("Successfully loaded: {} {}", successfulRecords, percentage);

        // Layer 1: Raw Input Validation
        if (getRawInputErrors() > 0) {
            logger.info("--- Layer 1: Raw Input Validation Errors ({}) ---", getRawInputErrors());
            logNonZero(logger, "  Null/Empty records", nullOrEmptyRecords);
            logNonZero(logger, "  Too short (raw)", tooShortRawRecords);
            logNonZero(logger, "  Too long (raw)", tooLongRawRecords);
            logNonZero(logger, "  Invalid encoding", invalidEncodingRecords);
            logNonZero(logger, "  Missing required fields", missingRequiredFieldsRecords);
        }

        // Layer 2: Processing Validation
        if (getProcessingErrors() > 0) {
            logger.info("--- Layer 2: Processing Validation Errors ({}) ---", getProcessingErrors());
            logNonZero(logger, "  Over-cleaned records", overCleanedRecords);
            logNonZero(logger, "  Processing failures", processingFailureRecords);
            logNonZero(logger, "  Invalid after cleaning", invalidAfterCleaningRecords);
        }

        // Layer 3: Model Input Validation
        if (getModelInputErrors() > 0) {
            logger.info("--- Layer 3: Model Input Validation Errors ({}) ---", getModelInputErrors());
            logNonZero(logger, "  Insufficient features", insufficientFeaturesRecords);
            logNonZero(logger, "  Invalid labels", invalidLabelRecords);
            logNonZero(logger, "  Feature extraction failures", featureExtractionFailureRecords);
        }

        // Quality Control
        if (getQualityControlRejections() > 0) {
            logger.info("--- Quality Control Rejections ({}) ---", getQualityControlRejections());
            logNonZero(logger, "  Duplicates removed", duplicateRecords);
            logNonZero(logger, "  Spam filtered", spamRecords);
            logNonZero(logger, "  Fake reviews filtered", fakeReviewRecords);
        }

        // Custom counters
        if (!customCounters.isEmpty()) {
            logger.info("--- Custom Metrics ---");
            for (Map.Entry<String, Integer> entry : customCounters.entrySet()) {
                if (entry.getValue() > 0) {
                    logger.info("  {}: {}", entry.getKey(), entry.getValue());
                }
            }
        }

        // Summary rates
        logger.info("=== Error Rate Breakdown ===");
        logRateIfNonZero(logger, "Raw Input Error Rate", getRawInputErrorRate());
        logRateIfNonZero(logger, "Processing Error Rate", getProcessingErrorRate());
        logRateIfNonZero(logger, "Model Input Error Rate", getModelInputErrorRate());
    }

    private void logNonZero(Logger logger, String label, int count) {
        if (count > 0) {
            logger.info("{}: {}", label, count);
        }
    }

    private void logRateIfNonZero(Logger logger, String label, double rate) {
        if (rate > 0) {
            String percentage = String.format("%.2f", rate * 100);
            logger.info("{}: {}", label, percentage);
        }
    }

    @Override
    public String toString() {
        return String.format("DatasetLoadingStats{total=%d, successful=%d, successRate=%.1f%%, " +
                        "rawErrors=%d, processingErrors=%d, modelErrors=%d}",
                totalRecords, successfulRecords, getSuccessRate() * 100,
                getRawInputErrors(), getProcessingErrors(), getModelInputErrors());
    }
}