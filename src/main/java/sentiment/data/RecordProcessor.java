package sentiment.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Template method processor for handling record-by-record dataset loading.
 * Eliminates duplication between CSV and JSON processing loops by providing
 * a unified framework for record processing with error handling, statistics,
 * and quality validation.
 *
 * @param <T> The raw record type (CSVRecord, JsonNode, etc.)
 */
public abstract class RecordProcessor<T> {

    private static final Logger logger = LoggerFactory.getLogger(RecordProcessor.class);

    private final String datasetType;
    private final String filePath;
    private final double maxErrorRate;

    // Configurable hooks for different processing behaviors
    private Consumer<DatasetLoadingStats> onProcessingStart = stats -> {};
    private Consumer<DatasetLoadingStats> onProcessingComplete = stats -> {};
    private Predicate<T> recordFilter = record -> true;
    private Function<Exception, String> errorMessageExtractor = Exception::getMessage;

    protected RecordProcessor(String datasetType, String filePath, double maxErrorRate) {
        this.datasetType = datasetType;
        this.filePath = filePath;
        this.maxErrorRate = maxErrorRate;
    }

    /**
     * Main template method that processes all records using the defined strategy.
     * Handles error tracking, validation, and statistics collection uniformly.
     */
    public final List<Dataset> processRecords(Iterable<T> records, DatasetLoadingStats stats) throws DataLoadingException {
        List<Dataset> datasets = new ArrayList<>();

        onProcessingStart.accept(stats);

        for (T record : records) {
            stats.incrementTotal();

            try {
                // Apply optional record filter
                if (!recordFilter.test(record)) {
                    stats.incrementCustom("filtered_records");
                    continue;
                }

                // Process individual record (implemented by subclasses)
                Dataset dataset = processRecord(record, stats, getRecordNumber(record));

                if (dataset != null) {
                    datasets.add(dataset);
                    stats.incrementSuccessful();
                } else {
                    // processRecord returned null - record was skipped for quality reasons
                    // Stats should already be updated by processRecord implementation
                }

            } catch (Exception e) {
                stats.incrementProcessingFailure();
                handleProcessingError(record, e, stats);

                // Check error rate
                checkErrorRate(stats);
            }
        }

        onProcessingComplete.accept(stats);

        return datasets;
    }

    /**
     * Process a single record into a Dataset object.
     * This is the main method that subclasses must implement.
     *
     * @param record The raw record to process
     * @param stats Statistics tracker to update
     * @param recordNumber The record number for logging/debugging
     * @return Dataset object if successful, null if record should be skipped
     * @throws Exception if processing fails
     */
    protected abstract Dataset processRecord(T record, DatasetLoadingStats stats, int recordNumber) throws Exception;

    /**
     * Get the record number for logging purposes.
     * Default implementation returns -1, subclasses should override if they have record numbers.
     */
    protected int getRecordNumber(T record) {
        return -1;
    }

    /**
     * Handle processing errors with logging and optional recovery.
     * Can be overridden by subclasses for custom error handling.
     */
    protected void handleProcessingError(T record, Exception e, DatasetLoadingStats stats) {
        int recordNumber = getRecordNumber(record);
        String recordInfo = truncateRecordForLogging(record);
        String errorMessage = errorMessageExtractor.apply(e);

        if (recordNumber > 0) {
            logger.warn("Error processing {} record {}: {} - Record: {}",
                    datasetType, recordNumber, errorMessage, recordInfo);
        } else {
            logger.warn("Error processing {} record: {} - Record: {}",
                    datasetType, errorMessage, recordInfo);
        }
    }

    /**
     * Truncate record content for logging to avoid excessive output.
     * Can be overridden by subclasses for record-type-specific formatting.
     */
    protected String truncateRecordForLogging(T record) {
        if (record == null) {
            return "null";
        }

        String recordStr = record.toString();
        return recordStr.length() > 100 ? recordStr.substring(0, 97) + "..." : recordStr;
    }

    private void checkErrorRate(DatasetLoadingStats stats) throws DataLoadingException {
        if (stats.getTotalRecords() > 100) { // Only check after reasonable sample
            double currentErrorRate = 1.0 - stats.getSuccessRate();
            if (currentErrorRate > maxErrorRate) {
                throw new DataLoadingException(
                        String.format("Error rate %.1f%% exceeds maximum %.1f%% after %d records",
                                currentErrorRate * 100, maxErrorRate * 100, stats.getTotalRecords()),
                        filePath, datasetType);
            }
        }
    }

    // Configuration methods for customizing behavior

    public RecordProcessor<T> onProcessingStart(Consumer<DatasetLoadingStats> callback) {
        this.onProcessingStart = callback;
        return this;
    }

    public RecordProcessor<T> onProcessingComplete(Consumer<DatasetLoadingStats> callback) {
        this.onProcessingComplete = callback;
        return this;
    }

    public RecordProcessor<T> withRecordFilter(Predicate<T> filter) {
        this.recordFilter = filter;
        return this;
    }

    public RecordProcessor<T> withErrorMessageExtractor(Function<Exception, String> extractor) {
        this.errorMessageExtractor = extractor;
        return this;
    }

    // Getters for subclasses
    protected String getDatasetType() { return datasetType; }
    protected String getFilePath() { return filePath; }
    protected double getMaxErrorRate() { return maxErrorRate; }
}

/**
 * CSV-specific implementation of RecordProcessor
 */
class CsvRecordProcessor extends RecordProcessor<org.apache.commons.csv.CSVRecord> {

    private final List<String> headers;
    private final CsvRecordHandler recordHandler;

    @FunctionalInterface
    public interface CsvRecordHandler {
        Dataset handleRecord(org.apache.commons.csv.CSVRecord record, DatasetLoadingStats stats, List<String> headers) throws Exception;
    }

    public CsvRecordProcessor(String datasetType, String filePath, double maxErrorRate,
                              List<String> headers, CsvRecordHandler recordHandler) {
        super(datasetType, filePath, maxErrorRate);
        this.headers = headers;
        this.recordHandler = recordHandler;
    }

    @Override
    protected Dataset processRecord(org.apache.commons.csv.CSVRecord record, DatasetLoadingStats stats, int recordNumber) throws Exception {
        return recordHandler.handleRecord(record, stats, headers);
    }

    @Override
    protected int getRecordNumber(org.apache.commons.csv.CSVRecord record) {
        return (int) record.getRecordNumber();
    }

    @Override
    protected String truncateRecordForLogging(org.apache.commons.csv.CSVRecord record) {
        if (record == null || record.size() == 0) {
            return "[]";
        }

        StringBuilder sb = new StringBuilder();
        sb.append("[");

        for (int i = 0; i < Math.min(record.size(), 3); i++) {
            if (i > 0) sb.append(", ");

            String value = record.get(i);
            if (value == null) {
                sb.append("null");
            } else if (value.length() > 50) {
                sb.append("\"").append(value, 0, 47).append("...\"");
            } else {
                sb.append("\"").append(value).append("\"");
            }
        }

        if (record.size() > 3) {
            sb.append(", ... (").append(record.size() - 3).append(" more)");
        }

        sb.append("]");
        return sb.toString();
    }
}

/**
 * JSON-specific implementation of RecordProcessor
 */
class JsonRecordProcessor extends RecordProcessor<com.fasterxml.jackson.databind.JsonNode> {

    private final JsonRecordHandler recordHandler;
    private int currentRecordNumber = 0;

    @FunctionalInterface
    public interface JsonRecordHandler {
        Dataset handleRecord(com.fasterxml.jackson.databind.JsonNode node, DatasetLoadingStats stats, int recordNumber) throws Exception;
    }

    public JsonRecordProcessor(String datasetType, String filePath, double maxErrorRate,
                               JsonRecordHandler recordHandler) {
        super(datasetType, filePath, maxErrorRate);
        this.recordHandler = recordHandler;
    }

    @Override
    protected Dataset processRecord(com.fasterxml.jackson.databind.JsonNode record, DatasetLoadingStats stats, int recordNumber) throws Exception {
        return recordHandler.handleRecord(record, stats, ++currentRecordNumber);
    }

    @Override
    protected int getRecordNumber(com.fasterxml.jackson.databind.JsonNode record) {
        return currentRecordNumber;
    }

    @Override
    protected String truncateRecordForLogging(com.fasterxml.jackson.databind.JsonNode record) {
        if (record == null) {
            return "null";
        }

        String jsonStr = record.toString();
        return jsonStr.length() > 100 ? jsonStr.substring(0, 97) + "..." : jsonStr;
    }
}

/**
 * String-based record processor for JSON Lines format
 */
class StringRecordProcessor extends RecordProcessor<String> {

    private final StringRecordHandler recordHandler;
    private int currentLineNumber = 0;

    @FunctionalInterface
    public interface StringRecordHandler {
        Dataset handleRecord(String line, DatasetLoadingStats stats, int lineNumber) throws Exception;
    }

    public StringRecordProcessor(String datasetType, String filePath, double maxErrorRate,
                                 StringRecordHandler recordHandler) {
        super(datasetType, filePath, maxErrorRate);
        this.recordHandler = recordHandler;

        // Configure to skip empty lines
        withRecordFilter(line -> line != null && !line.trim().isEmpty());
    }

    @Override
    protected Dataset processRecord(String record, DatasetLoadingStats stats, int recordNumber) throws Exception {
        return recordHandler.handleRecord(record, stats, ++currentLineNumber);
    }

    @Override
    protected int getRecordNumber(String record) {
        return currentLineNumber;
    }

    @Override
    protected String truncateRecordForLogging(String record) {
        if (record == null) {
            return "null";
        }
        return record.length() > 100 ? record.substring(0, 97) + "..." : record;
    }

    @Override
    protected void handleProcessingError(String record, Exception e, DatasetLoadingStats stats) {
        // For empty lines that pass through filter, increment empty counter
        if (record != null && record.trim().isEmpty()) {
            stats.incrementNullOrEmpty();
        }

        super.handleProcessingError(record, e, stats);
    }
}