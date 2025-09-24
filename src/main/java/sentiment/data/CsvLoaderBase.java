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
 * CsvLoaderBase using RecordProcessor template method
 */
public abstract class CsvLoaderBase extends BaseDatasetLoader {

    private static final Logger logger = LoggerFactory.getLogger(CsvLoaderBase.class);

    public enum CsvFormatHint {
        AUTO_DETECT, COMMA_DELIMITED, TAB_DELIMITED
    }

    @Override
    protected List<Dataset> doLoadDataset(String filePath, DatasetLoadingStats stats) throws DataLoadingException {

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath, StandardCharsets.UTF_8))) {

            CSVFormat format = createCommaDelimitedFormat();

            try (CSVParser parser = format.parse(reader)) {

                List<String> headers = parser.getHeaderNames();
                if (shouldValidateHeaders() && !headers.isEmpty()) {
                    validateCsvHeaders(headers, filePath);
                }

                logger.debug("Processing CSV with headers: {}", headers);

                return processRecordsWithErrorRecovery(parser, headers, stats, filePath);
            }

        } catch (IOException e) {
            throw new DataLoadingException("Failed to read CSV file", filePath, getDatasetTypeName(), e);
        }
    }

    /**
     * Process records with bulletproof error recovery - catches parsing errors and continues
     */
    private List<Dataset> processRecordsWithErrorRecovery(CSVParser parser, List<String> headers,
                                                          DatasetLoadingStats stats, String filePath) throws DataLoadingException {
        List<Dataset> datasets = new ArrayList<>();
        int consecutiveErrors = 0;
        int maxConsecutiveErrors = 100;
        int recordCount = 0;

        Iterator<CSVRecord> iterator = parser.iterator();

        while (true) {
            CSVRecord record = null;

            try {
                if (!iterator.hasNext()) {
                    break;
                }

                record = iterator.next();
                recordCount++;
                consecutiveErrors = 0;


            } catch (RuntimeException e) {
                consecutiveErrors++;
                stats.incrementCustom("malformed_csv_lines");

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

                    // Log first successful record to see what's working
                    if (datasets.size() == 1) {
                        logger.info("First successful record: {}", dataset);
                    }
                }

            } catch (Exception e) {
                stats.incrementProcessingFailure();
                if (recordCount <= 10) {
                    logger.warn("Error processing early record {}: {}", recordCount, e.getMessage());
                }
            }

            // Early progress check
            if (recordCount % 100000 == 0) {
                logger.info("Processed {} records, {} successful so far", recordCount, stats.getSuccessfulRecords());
                if (recordCount == 10000 && stats.getSuccessfulRecords() < 100) {
                    logger.warn("Very low success rate after 10k records - may need to adjust validation");
                }
            }
        }

        int malformedLines = stats.getCustom("malformed_csv_lines");
        if (malformedLines > 0) {
            logger.info("Skipped {} malformed CSV lines during parsing", malformedLines);
        }

        logger.info("Final parsing stats: {} total records parsed, {} malformed lines skipped",
                recordCount, malformedLines);

        return datasets;
    }

    private void checkErrorRate(DatasetLoadingStats stats, String filePath) throws DataLoadingException {
        if (stats.getTotalRecords() > 100) {
            double currentErrorRate = 1.0 - stats.getSuccessRate();
            if (currentErrorRate > getMaxErrorRate()) {
                throw new DataLoadingException(
                        String.format("Error rate %.1f%% exceeds maximum %.1f%% after %d records",
                                currentErrorRate * 100, getMaxErrorRate() * 100, stats.getTotalRecords()),
                        filePath, getDatasetTypeName());
            }
        }
    }

    /**
     * Abstract method for processing individual CSV records.
     */
    protected abstract Dataset processRecord(CSVRecord record, DatasetLoadingStats stats,
                                             List<String> headers) throws Exception;


    protected CsvFormatHint getCsvFormatHint() {
        return CsvFormatHint.AUTO_DETECT;
    }

    protected boolean shouldValidateHeaders() {
        return false;
    }

    protected void validateCsvHeaders(List<String> headers, String filePath) throws DataLoadingException {
        // Default implementation - no validation
    }

    protected double getMaxErrorRate() {
        return 0.15;
    }

    private CSVFormat createCsvFormat(String filePath, BufferedReader reader) throws IOException {
        return createCommaDelimitedFormat(); // Simplified for example
    }

    protected CSVFormat createCommaDelimitedFormat() {
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
}