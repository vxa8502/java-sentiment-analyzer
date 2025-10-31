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
 */
public abstract class CsvLoaderBase extends BaseDatasetLoader {

    private static final Logger logger = LoggerFactory.getLogger(CsvLoaderBase.class);

    @Override
    protected List<Dataset> doLoadDataset(String filePath, DatasetLoadingStats stats) throws DataLoadingException {
        return doLoadDatasetInternal(filePath, stats, -1);
    }

    @Override
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

    /**
     * Get maximum acceptable error rate for this loader type
     */
    protected abstract double getMaxErrorRate();
}