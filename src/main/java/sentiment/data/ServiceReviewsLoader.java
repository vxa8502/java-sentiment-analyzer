package sentiment.data;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.csv.QuoteMode;
import java.time.LocalDateTime;
import java.util.List;

/**
 * Service Reviews Dataset Loader - returns completely raw, unprocessed data.
 * Only performs basic structural validation to ensure data can be loaded.
 *
 * Handles service-related feedback including:
 * - Customer service interactions
 * - App store reviews (Google Play, Apple App Store)
 * - Help desk/support ticket feedback
 * - Social media customer complaints
 * - Service quality surveys
 *
 * Supports flexible CSV formats with various column structures.
 */
class ServiceReviewsLoader extends CsvLoaderBase {

    private static final String DATASET_TYPE = "Service Reviews";
    private static final String[] SUPPORTED_EXTENSIONS = {".tsv", ".csv", ".jsonl"};

    @Override
    protected Dataset processRecord(CSVRecord record, DatasetLoadingStats stats, List<String> headers) {

        int recordNumber = (int) record.getRecordNumber();

        String reviewText = null;
        String sentimentLabel = null;
        String reviewId = null;

        // Handle legacy Twitter/social media format (6 columns: sentiment, id, date, query, user, text)
        // Many service feedback datasets use this format from social media sources
        if (record.size() >= 6 && (headers.isEmpty() || !record.isMapped("text"))) {
            // Format: sentiment, id, date, query, user, text
            sentimentLabel = record.get(0);  // polarity (0 or 4 in Twitter format)
            reviewId = record.get(1);        // review/ticket id
            reviewText = record.get(5);      // feedback text

            // Convert numeric sentiment format (0=negative, 4=positive) used in some datasets
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
                            FieldExtractor.TEXT_FIELDS.withAdditional("feedback", "comment", "message"), 0);
            FieldExtractor.ExtractionResult<String> sentimentResult =
                    FieldExtractor.extractString(record,
                            FieldExtractor.SENTIMENT_FIELDS.withAdditional("target"), 1);

            reviewId = idResult.isPresent() ? idResult.getValue() : null;
            reviewText = textResult.isPresent() ? textResult.getValue() : null;
            sentimentLabel = sentimentResult.isPresent() ? sentimentResult.getValue() : null;
        }

        // ONLY structural validation - check if fields exist and are parseable
        DatasetValidationUtils.ValidationResult textValidation =
                DatasetValidationUtils.validateRawText(reviewText, DATASET_TYPE, recordNumber, stats);
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

        // Create Dataset with completely ORIGINAL text - no transformation whatsoever
        Dataset.Builder builder = new Dataset.Builder(rawText, parsedSentiment)
                .source("service")
                .originalLabel(sentimentLabel)
                .timestamp(LocalDateTime.now());

        // Add review/ticket ID if available
        if (reviewId != null && !reviewId.trim().isEmpty()) {
            builder.id(reviewId.trim());
        }

        return builder.build();
    }

    @Override
    protected CSVFormat createCsvFormat() {
        // Configure CSV format to handle service feedback data quirks
        // Service data often comes from various sources (social, tickets, surveys)
        // so we need flexible parsing
        return CSVFormat.Builder.create()
                .setDelimiter(',')
                .setHeader()
                .setSkipHeaderRecord(false)  // Service data might not have headers
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
        return 0.5; // Allow up to 50% error rate for service data (often noisy, informal)
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
