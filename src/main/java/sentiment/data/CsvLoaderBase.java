package sentiment.data;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Simplified CSV loader base class - only parsing and basic structural validation.
 * No text cleaning or content validation - just load raw data from files.
 *
 * This class combines the template method pattern from the former BaseDatasetLoader
 * with CSV-specific parsing logic.
 */
public abstract class CsvLoaderBase implements DatasetLoader {

    protected final Logger logger = LoggerFactory.getLogger(getClass());

    // ==================== PUBLIC INTERFACE METHODS ====================

    @Override
    public final List<Dataset> loadDataset(String filePath) throws DataLoadingException {
        logger.info("Loading {} from: {}", getDatasetTypeName(), filePath);

        // Validate file can be read
        DatasetValidationUtils.validateFilePath(filePath, getDatasetTypeName());

        // Create statistics tracker
        DatasetLoadingStats stats = new DatasetLoadingStats();

        // Perform actual loading (implemented by this class)
        List<Dataset> datasets = doLoadDataset(filePath, stats);

        // Validate final result
        DatasetValidationUtils.validateFinalDataset(datasets, stats, filePath, getDatasetTypeName());

        // Log summary
        stats.logSummary(logger, filePath, getDatasetTypeName());

        return datasets;
    }

    @Override
    public CompatibilityTestResult testCompatibility(String filePath, int sampleSize) {
        // Validate basic file access
        try {
            DatasetValidationUtils.validateFilePath(filePath, getDatasetTypeName());
        } catch (DataLoadingException e) {
            return CompatibilityTestResult.failure(getDatasetTypeName(),
                "Cannot access file: " + e.getMessage());
        }

        // Check file extension
        if (!canHandle(filePath)) {
            return CompatibilityTestResult.failure(getDatasetTypeName(),
                "Unsupported file extension");
        }

        // Create statistics tracker for sampling
        DatasetLoadingStats stats = new DatasetLoadingStats();

        try {
            // Attempt to load a sample of the file
            List<Dataset> sample = doLoadDatasetSample(filePath, sampleSize, stats);

            // Calculate results
            int sampledRows = stats.getTotalRecords();
            int successfulRows = stats.getSuccessfulRecords();
            int failedRows = sampledRows - successfulRows;

            // If we couldn't parse anything, it's a complete failure
            if (sampledRows == 0) {
                return CompatibilityTestResult.failure(getDatasetTypeName(),
                    "Could not parse any records from file");
            }

            return CompatibilityTestResult.success(
                getDatasetTypeName(),
                sampledRows,
                successfulRows,
                failedRows
            );

        } catch (Exception e) {
            return CompatibilityTestResult.failure(getDatasetTypeName(),
                "Error during sampling: " + e.getMessage());
        }
    }

    // ==================== INTERNAL LOADING METHODS ====================

    protected List<Dataset> doLoadDataset(String filePath, DatasetLoadingStats stats) throws DataLoadingException {
        return doLoadDatasetInternal(filePath, stats, -1);
    }

    protected List<Dataset> doLoadDatasetSample(String filePath, int sampleSize,
                                                 DatasetLoadingStats stats) throws DataLoadingException {
        return doLoadDatasetInternal(filePath, stats, sampleSize);
    }

    /**
     * Internal method that handles both full loading and sampling
     * @param maxRows Maximum number of rows to process (-1 for unlimited)
     */
    private List<Dataset> doLoadDatasetInternal(String filePath, DatasetLoadingStats stats,
                                                 int maxRows) throws DataLoadingException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath, StandardCharsets.UTF_8))) {
            CSVFormat format = createCsvFormat();

            try (CSVParser parser = format.parse(reader)) {
                List<String> headers = parser.getHeaderNames();
                logger.debug("Processing CSV with headers: {}", headers);

                return processRecordsWithErrorRecovery(parser, headers, stats, filePath, maxRows);
            }

        } catch (IOException e) {
            throw new DataLoadingException("Failed to read CSV file", filePath, getDatasetTypeName(), e);
        }
    }

    /**
     * Process CSV records with error recovery - integrated error handling logic
     * @param maxRows Maximum number of rows to process (-1 for unlimited)
     */
    private List<Dataset> processRecordsWithErrorRecovery(CSVParser parser, List<String> headers,
                                                          DatasetLoadingStats stats, String filePath,
                                                          int maxRows) throws DataLoadingException {
        List<Dataset> datasets = new ArrayList<>();
        int consecutiveErrors = 0;
        int maxConsecutiveErrors = 100;
        int recordCount = 0;
        boolean isSampling = maxRows > 0;

        Iterator<CSVRecord> iterator = parser.iterator();

        while (true) {
            // Check if we've reached the sample limit
            if (isSampling && recordCount >= maxRows) {
                logger.debug("Reached sample limit of {} rows", maxRows);
                break;
            }

            CSVRecord record;

            try {
                if (!iterator.hasNext()) {
                    break;
                }

                record = iterator.next();
                recordCount++;
                consecutiveErrors = 0;

            } catch (RuntimeException e) {
                consecutiveErrors++;
                stats.incrementParseErrors();

                logger.warn("Skipping malformed CSV line around record {}: {}", recordCount, e.getMessage());

                if (consecutiveErrors >= maxConsecutiveErrors) {
                    throw new DataLoadingException(
                            String.format("Hit %d consecutive parsing errors - file may be corrupted", consecutiveErrors),
                            filePath, getDatasetTypeName());
                }
                continue;
            }

            // Process the record
            stats.incrementTotal();

            try {
                Dataset dataset = processRecord(record, stats, headers);

                if (dataset != null) {
                    datasets.add(dataset);
                    stats.incrementSuccessful();

                    // Log first successful record
                    if (datasets.size() == 1) {
                        logger.info("First successful record: {}", dataset);
                    }
                }

            } catch (Exception e) {
                logger.debug("Error processing record {}: {}", recordCount, e.getMessage());
                // Error already tracked by processRecord implementation
            }

            // Progress logging (skip during sampling)
            if (!isSampling && recordCount % 50000 == 0) {
                logger.info("Processed {} records, {} successful so far", recordCount, stats.getSuccessfulRecords());
            }

            // Check error rate periodically (skip during sampling to avoid false failures)
            if (!isSampling && recordCount > 100 && recordCount % 1000 == 0) {
                checkErrorRate(stats, filePath);
            }
        }

        logger.debug("Final parsing: {} total records, {} successful", recordCount, stats.getSuccessfulRecords());
        return datasets;
    }

    /**
     * Check if error rate exceeds acceptable threshold
     */
    private void checkErrorRate(DatasetLoadingStats stats, String filePath) throws DataLoadingException {
        double currentErrorRate = 1.0 - stats.getSuccessRate();
        if (currentErrorRate > getMaxErrorRate()) {
            throw new DataLoadingException(
                    String.format("Error rate %.1f%% exceeds maximum %.1f%% after %d records",
                            currentErrorRate * 100, getMaxErrorRate() * 100, stats.getTotalRecords()),
                    filePath, getDatasetTypeName());
        }
    }

    /**
     * Abstract method for processing individual CSV records.
     * Should only do basic structural validation and create Dataset with original text.
     */
    protected abstract Dataset processRecord(CSVRecord record, DatasetLoadingStats stats,
                                             List<String> headers) throws Exception;

    /**
     * Create CSV format for parsing - can be overridden by subclasses
     */
    protected CSVFormat createCsvFormat() {
        return CSVFormat.Builder.create()
                .setDelimiter(',')
                .setHeader()
                .setSkipHeaderRecord(true)
                .setIgnoreEmptyLines(true)
                .setTrim(true)
                .setQuote('"')
                .setEscape('\\')
                .setAllowMissingColumnNames(true)
                .setIgnoreHeaderCase(true)
                .build();
    }

    /**
     * Simple field extraction helper
     */
    protected String extractField(CSVRecord record, String... fieldNames) {
        for (String fieldName : fieldNames) {
            if (record.isMapped(fieldName)) {
                String value = record.get(fieldName);
                if (value != null && !value.trim().isEmpty()) {
                    return value.trim();
                }
            }
        }
        return null;
    }

    /**
     * Extract field with positional fallback
     */
    protected String extractFieldWithFallback(CSVRecord record, int fallbackPosition, String... fieldNames) {
        String value = extractField(record, fieldNames);
        if (value == null && fallbackPosition >= 0 && fallbackPosition < record.size()) {
            value = record.get(fallbackPosition);
            if (value != null && !value.trim().isEmpty()) {
                return value.trim();
            }
        }
        return value;
    }

    // ==================== HELPER METHODS ====================

    /**
     * Common pattern: validate text + sentiment, then build Dataset.
     * Returns null if validation fails (with stats automatically updated).
     *
     * This eliminates the repetitive validate → build pattern across loaders.
     */
    protected Dataset validateAndBuild(
            String rawText,
            String rawSentiment,
            int recordNumber,
            DatasetLoadingStats stats,
            String source,
            String datasetType) {

        // Validate text
        DatasetValidationUtils.ValidationResult textValidation =
                DatasetValidationUtils.validateRawText(rawText, datasetType, recordNumber, stats);
        if (!textValidation.isValid()) {
            return null;
        }

        // Validate sentiment
        DatasetValidationUtils.ValidationResult sentimentValidation =
                DatasetValidationUtils.validateRawSentiment(rawSentiment, datasetType, recordNumber, stats);
        if (!sentimentValidation.isValid()) {
            return null;
        }

        // Build and return Dataset
        String validatedText = textValidation.getText();
        Dataset.SentimentLabel validatedSentiment = sentimentValidation.getSentiment();

        return new Dataset.Builder(validatedText, validatedSentiment)
                .source(source)
                .originalLabel(rawSentiment)
                .timestamp(java.time.LocalDateTime.now())
                .build();
    }

    /**
     * Common helper to build Dataset from validated fields.
     * Use this when you've already validated the fields separately.
     */
    protected Dataset buildStandardDataset(
            String validatedText,
            Dataset.SentimentLabel validatedSentiment,
            String source,
            String originalLabel) {
        return new Dataset.Builder(validatedText, validatedSentiment)
                .source(source)
                .originalLabel(originalLabel)
                .timestamp(java.time.LocalDateTime.now())
                .build();
    }

    /**
     * Get maximum acceptable error rate for this loader type
     */
    protected abstract double getMaxErrorRate();
}