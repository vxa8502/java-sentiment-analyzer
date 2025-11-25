package sentiment.data;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Simple, extensible dataset loader with format auto-detection.
 * Supports incremental addition of new formats as needed.
 */
@Component
public class SimpleDatasetLoader {

    private static final Logger logger = LoggerFactory.getLogger(SimpleDatasetLoader.class);

    // Common column name variations for flexible CSV parsing
    private static final String[] TEXT_COLUMNS = {"text", "review", "comment", "content", "message", "body"};
    private static final String[] SENTIMENT_COLUMNS = {"sentiment", "label", "polarity", "class", "rating"};

    /**
     * Load dataset from file with automatic format detection.
     *
     * @param filePath path to dataset file
     * @return list of parsed Dataset objects
     * @throws DataLoadingException if file cannot be loaded or parsed
     */
    public List<Dataset> load(String filePath) throws DataLoadingException {
        DatasetLoadResult result = loadWithMetadata(filePath);
        return result.datasets();
    }

    /**
     * Load dataset with metadata (load time, format type, etc.).
     *
     * @param filePath path to dataset file
     * @return DatasetLoadResult containing datasets and metadata
     * @throws DataLoadingException if file cannot be loaded or parsed
     */
    public DatasetLoadResult loadWithMetadata(String filePath) throws DataLoadingException {
        validateFile(filePath);

        String extension = getExtension(filePath).toLowerCase();
        String formatType = extension.substring(1).toUpperCase(); // .csv -> CSV

        logger.info("Loading dataset from: {} (format: {})", filePath, extension);

        long startTime = System.currentTimeMillis();

        List<Dataset> datasets = switch (extension) {
            case ".csv", ".tsv" -> loadCsv(filePath, extension.equals(".tsv") ? '\t' : ',');
            case ".jsonl" -> loadJsonl(filePath);
            case ".txt" -> loadPlaintext(filePath);
            default -> throw new DataLoadingException(
                "Unsupported file format: " + extension + ". Supported: .csv, .tsv, .jsonl, .txt",
                filePath,
                "Unknown"
            );
        };

        long loadTime = System.currentTimeMillis() - startTime;

        return new DatasetLoadResult(
            datasets,
            formatType + " Dataset",
            filePath,
            loadTime
        );
    }

    // CSV LOADING

    /**
     * Load CSV/TSV format with flexible column detection.
     * Handles various column naming conventions automatically.
     */
    private List<Dataset> loadCsv(String filePath, char delimiter) throws DataLoadingException {
        List<Dataset> datasets = new ArrayList<>();
        int totalRows = 0;
        int successfulRows = 0;
        int skippedRows = 0;

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath, StandardCharsets.UTF_8))) {

            CSVFormat format = CSVFormat.Builder.create()
                .setDelimiter(delimiter)
                .setHeader()
                .setSkipHeaderRecord(true)
                .setIgnoreEmptyLines(true)
                .setTrim(true)
                .setQuote('"')
                .setIgnoreHeaderCase(true)
                .build();

            try (CSVParser parser = format.parse(reader)) {
                List<String> headers = parser.getHeaderNames();
                logger.info("CSV headers detected: {}", headers);

                // Detect which columns to use
                String textColumn = detectColumn(headers, TEXT_COLUMNS);
                String sentimentColumn = detectColumn(headers, SENTIMENT_COLUMNS);

                if (textColumn == null || sentimentColumn == null) {
                    throw new DataLoadingException(
                        String.format("Could not detect required columns. Found headers: %s. Expected columns like %s and %s",
                            headers,
                            String.join("/", TEXT_COLUMNS),
                            String.join("/", SENTIMENT_COLUMNS)),
                        filePath,
                        "CSV"
                    );
                }

                logger.info("Using columns: text='{}', sentiment='{}'", textColumn, sentimentColumn);

                for (CSVRecord record : parser) {
                    totalRows++;

                    try {
                        String text = record.get(textColumn);
                        String sentimentStr = record.get(sentimentColumn);

                        // Basic validation
                        if (text == null || text.trim().isEmpty()) {
                            skippedRows++;
                            continue;
                        }

                        if (sentimentStr == null || sentimentStr.trim().isEmpty()) {
                            skippedRows++;
                            continue;
                        }

                        // Parse sentiment
                        Dataset.SentimentLabel sentiment = parseSentiment(sentimentStr.trim());
                        if (sentiment == null) {
                            skippedRows++;
                            logger.debug("Skipping row {} - invalid sentiment: {}", record.getRecordNumber(), sentimentStr);
                            continue;
                        }

                        // Build dataset
                        Dataset dataset = new Dataset.Builder(text.trim(), sentiment)
                            .source("csv")
                            .originalLabel(sentimentStr.trim())
                            .timestamp(java.time.LocalDateTime.now())
                            .build();

                        datasets.add(dataset);
                        successfulRows++;

                        // Log first successful record
                        if (datasets.size() == 1) {
                            logger.info("First record loaded: {} -> {}",
                                text.substring(0, Math.min(50, text.length())) + "...",
                                sentiment);
                        }

                    } catch (Exception e) {
                        skippedRows++;
                        logger.debug("Error parsing row {}: {}", record.getRecordNumber(), e.getMessage());
                    }

                    // Progress logging
                    if (totalRows % 10000 == 0) {
                        logger.info("Progress: {} rows processed, {} loaded", totalRows, successfulRows);
                    }
                }
            }

        } catch (IOException e) {
            throw new DataLoadingException("Failed to read CSV file", filePath, "CSV", e);
        }

        // Validate results
        if (datasets.isEmpty()) {
            throw new DataLoadingException(
                String.format("No valid data loaded. Processed %d rows, all were invalid or empty.", totalRows),
                filePath,
                "CSV"
            );
        }

        double errorRate = (double) skippedRows / totalRows;
        if (errorRate > 0.5) {
            logger.warn("High error rate: {}/{} rows skipped ({}%)",
                skippedRows, totalRows, String.format("%.1f", errorRate * 100));
        }

        logger.info("CSV loading complete: {}/{} rows loaded successfully", successfulRows, totalRows);
        return datasets;
    }

    /**
     * Detect which column to use from a list of possible names.
     */
    private String detectColumn(List<String> headers, String[] possibleNames) {
        for (String candidate : possibleNames) {
            for (String header : headers) {
                if (header.equalsIgnoreCase(candidate)) {
                    return header;
                }
            }
        }
        return null;
    }

    /**
     * Parse sentiment string into SentimentLabel enum.
     * Handles various formats: positive/negative/neutral, 1/0, pos/neg, etc.
     */
    private Dataset.SentimentLabel parseSentiment(String sentiment) {
        String lower = sentiment.toLowerCase();

        // Handle text labels
        switch (lower) {
            case "positive", "pos", "1", "4" -> {
                return Dataset.SentimentLabel.POSITIVE;
            }
            case "negative", "neg", "0", "-1" -> {
                return Dataset.SentimentLabel.NEGATIVE;
            }
            case "neutral", "2" -> {
                return Dataset.SentimentLabel.NEUTRAL;
            }
        }

        // Handle numeric ratings (1-5 stars)
        try {
            double rating = Double.parseDouble(sentiment);
            if (rating <= 2.0) {
                return Dataset.SentimentLabel.NEGATIVE;
            } else if (rating >= 4.0) {
                return Dataset.SentimentLabel.POSITIVE;
            } else {
                return Dataset.SentimentLabel.NEUTRAL;
            }
        } catch (NumberFormatException e) {
            // Not a number, continue
        }

        return null; // Unknown format
    }

    // JSONL LOADING (STUB)

    /**
     * Load JSON Lines format.
     * TODO: Implement when needed for specific dataset.
     */
    private List<Dataset> loadJsonl(String filePath) throws DataLoadingException {
        throw new DataLoadingException(
            "JSONL format not yet implemented. Add Jackson/Gson parsing when needed.",
            filePath,
            "JSONL"
        );
    }

    // PLAINTEXT LOADING (STUB)

    /**
     * Load plaintext format (e.g., FastText: __label__positive This is great).
     * TODO: Implement when needed for specific dataset.
     */
    private List<Dataset> loadPlaintext(String filePath) throws DataLoadingException {
        throw new DataLoadingException(
            "Plaintext format not yet implemented. Add regex parsing when needed.",
            filePath,
            "TXT"
        );
    }

    //  UTILITIES

    private void validateFile(String filePath) throws DataLoadingException {
        if (filePath == null || filePath.trim().isEmpty()) {
            throw new DataLoadingException("File path cannot be null or empty", filePath, "Unknown");
        }

        Path path = Paths.get(filePath);

        if (!Files.exists(path)) {
            throw new DataLoadingException("File does not exist", filePath, "Unknown");
        }

        if (!Files.isReadable(path)) {
            throw new DataLoadingException("File is not readable", filePath, "Unknown");
        }

        try {
            if (Files.size(path) == 0) {
                throw new DataLoadingException("File is empty", filePath, "Unknown");
            }
        } catch (IOException e) {
            logger.debug("Could not check file size: {}", e.getMessage());
        }
    }

    private String getExtension(String filePath) {
        int lastDot = filePath.lastIndexOf('.');
        return lastDot > 0 ? filePath.substring(lastDot) : "";
    }
}
