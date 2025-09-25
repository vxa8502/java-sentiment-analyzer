package sentiment.data;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.csv.QuoteMode;
import java.time.LocalDateTime;
import java.util.List;

/**
 * Twitter/Social Media Dataset Loader - returns completely raw, unprocessed data.
 * Only performs basic structural validation to ensure data can be loaded.
 * Handles both standard Twitter sentiment format (sentiment140) and flexible CSV layouts.
 */
class TwitterDataLoader extends CsvLoaderBase {

    private static final String DATASET_TYPE = "Twitter Data";
    private static final String[] SUPPORTED_EXTENSIONS = {".tsv", ".csv"};

    @Override
    protected Dataset processRecord(CSVRecord record, DatasetLoadingStats stats, List<String> headers) {

        int recordNumber = (int) record.getRecordNumber();

        String tweetText = null;
        String sentimentLabel = null;
        String tweetId = null;

        // Handle standard Twitter sentiment format (6 columns: sentiment, id, date, query, user, text)
        if (record.size() >= 6 && (headers.isEmpty() || !record.isMapped("text"))) {
            // Standard Twitter format without headers: sentiment, id, date, query, user, text
            sentimentLabel = record.get(0);  // polarity (0 or 4)
            tweetId = record.get(1);         // tweet id
            tweetText = record.get(5);       // tweet text

            // Convert Twitter sentiment format (0=negative, 4=positive)
            if ("0".equals(sentimentLabel)) {
                sentimentLabel = "negative";
            } else if ("4".equals(sentimentLabel)) {
                sentimentLabel = "positive";
            }
        } else {
            // Use FieldExtractor for flexible field extraction
            FieldExtractor.ExtractionResult<String> idResult =
                    FieldExtractor.extractString(record, FieldExtractor.ID_FIELDS);
            FieldExtractor.ExtractionResult<String> textResult =
                    FieldExtractor.extractString(record,
                            FieldExtractor.TEXT_FIELDS.withAdditional("tweet_text", "tweet"), 0);
            FieldExtractor.ExtractionResult<String> sentimentResult =
                    FieldExtractor.extractString(record,
                            FieldExtractor.SENTIMENT_FIELDS.withAdditional("target"), 1);

            tweetId = idResult.isPresent() ? idResult.getValue() : null;
            tweetText = textResult.isPresent() ? textResult.getValue() : null;
            sentimentLabel = sentimentResult.isPresent() ? sentimentResult.getValue() : null;
        }

        // Structural validation - check if fields exist and are parseable
        DatasetValidationUtils.ValidationResult textValidation =
                DatasetValidationUtils.validateRawText(tweetText, DATASET_TYPE, recordNumber, stats);
        if (!textValidation.isValid()) {
            return null; // Field missing or empty
        }

        DatasetValidationUtils.ValidationResult sentimentValidation =
                DatasetValidationUtils.validateRawSentiment(sentimentLabel, DATASET_TYPE, recordNumber, stats);
        if (!sentimentValidation.isValid()) {
            return null; // Sentiment not parseable
        }

        // Get the raw data (only whitespace trimmed for parsing safety)
        String rawText = textValidation.getText(); // Only trimmed, no other changes
        Dataset.SentimentLabel parsedSentiment = sentimentValidation.getSentiment();

        // Create Dataset with completely original text - no transformation whatsoever
        Dataset.Builder builder = new Dataset.Builder(rawText, parsedSentiment)
                .source("twitter")
                .originalLabel(sentimentLabel)
                .timestamp(LocalDateTime.now());

        // Add tweet ID if available
        if (tweetId != null && !tweetId.trim().isEmpty()) {
            builder.id(tweetId.trim());
        }

        return builder.build();
    }

    @Override
    protected CSVFormat createCsvFormat() {
        // Configure CSV format to handle Twitter data quirks
        return CSVFormat.Builder.create()
                .setDelimiter(',')
                .setHeader()
                .setSkipHeaderRecord(false)  // Twitter data might not have headers
                .setIgnoreEmptyLines(true)
                .setTrim(true)
                .setQuote('"')
                .setQuoteMode(QuoteMode.MINIMAL)
                .setIgnoreSurroundingSpaces(true)
                .setAllowMissingColumnNames(true)
                .setIgnoreHeaderCase(true)
                .build();
    }

    @Override
    protected double getMaxErrorRate() {
        return 0.5; // Allow up to 50% error rate for social media data (noisy format)
    }

    @Override
    public String[] getSupportedExtensions() {
        return SUPPORTED_EXTENSIONS.clone();
    }

    @Override
    public String getDatasetTypeName() {
        return DATASET_TYPE;
    }
}