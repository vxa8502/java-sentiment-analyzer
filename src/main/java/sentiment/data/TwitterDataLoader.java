package sentiment.data;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.csv.QuoteMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.LocalDateTime;
import java.util.*;

/**
 * Twitter/Social Media Dataset Loader implementing proper three-layer validation
 * Layer 1: Raw Input Validation
 * Layer 2: Processing Validation
 * Layer 3: Model Input Validation
 */
class TwitterDataLoader extends CsvLoaderBase {

    private static final Logger logger = LoggerFactory.getLogger(TwitterDataLoader.class);

    private static final String DATASET_TYPE = "Twitter Data";
    private static final String[] SUPPORTED_EXTENSIONS = {".tsv", ".csv"};

    @Override
    protected Dataset processRecord(CSVRecord record, DatasetLoadingStats stats,
                                    List<String> headers) {

        int recordNumber = (int) record.getRecordNumber();

        String tweetText = null;
        String sentimentLabel = null;
        String tweetId = null;

        if (record.size() >= 6) {
            // Standard Twitter format: sentiment, id, date, query, user, text
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
            // Fall back to your existing field extraction
            tweetId = extractField(record, "tweet_id", "id", "post_id", "message_id");
            tweetText = extractFieldWithFallback(record,
                    tweetId != null ? 1 : 0, "tweet_text", "text", "content", "message", "post", "tweet");
            sentimentLabel = extractFieldWithFallback(record,
                    tweetId != null ? 2 : 1, "sentiment", "label", "polarity", "class", "emotion", "target");
        }

        DatasetValidationUtils.ValidationResult textValidation =
                DatasetValidationUtils.validateRawText(tweetText, DATASET_TYPE, recordNumber, stats);
        if (!textValidation.isValid()) {
            return null; // Stats already incremented by validation
        }

        DatasetValidationUtils.ValidationResult sentimentValidation =
                DatasetValidationUtils.validateRawSentiment(sentimentLabel, DATASET_TYPE, recordNumber, stats);
        if (!sentimentValidation.isValid()) {
            return null; // Stats already incremented by validation
        }

        String validatedText = textValidation.getText();
        Dataset.SentimentLabel validatedSentiment = sentimentValidation.getSentiment();

        String cleanedText = TextCleaningUtils.cleanTweetText(validatedText);

        DatasetValidationUtils.ValidationResult processingValidation =
                DatasetValidationUtils.validateProcessedText(
                        cleanedText, validatedText, DATASET_TYPE, recordNumber, stats);
        if (!processingValidation.isValid()) {
            return null; // Processing destroyed too much content or failed
        }

        String processedText = processingValidation.getText();

        DatasetValidationUtils.ValidationResult qualityResult =
                DatasetValidationUtils.performQualityControl(
                        processedText, DATASET_TYPE, recordNumber, stats);
        if (!qualityResult.isValid()) {
            return null; // Failed quality control (spam, fake review, etc.)
        }

        DatasetValidationUtils.ValidationResult modelValidation =
                DatasetValidationUtils.validateModelInput(
                        processedText, validatedSentiment, DATASET_TYPE, recordNumber, stats);
        if (!modelValidation.isValid()) {
            return null; // Insufficient features or invalid for model
        }

        // If we get here, all validation layers passed
        Dataset finalDataset = modelValidation.getDataset();

        // Enhance with Twitter-specific metadata
        return enhanceWithTwitterMetadata(finalDataset, tweetId, sentimentLabel);
    }

    /**
     * Add Twitter-specific metadata to the dataset
     */
    private Dataset enhanceWithTwitterMetadata(Dataset dataset, String tweetId, String originalLabel) {
        // Create new dataset with enhanced metadata using builder pattern
        Dataset.Builder builder = new Dataset.Builder(dataset.getText(), dataset.getSentiment())
                .source("twitter")
                .originalLabel(originalLabel)
                .timestamp(LocalDateTime.now());

        if (tweetId != null && !tweetId.isEmpty()) {
            builder.id(tweetId);
        }

        return builder.build();
    }

    /**
     * Extract field with fallback to positional access
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

    @Override
    protected CSVFormat createCommaDelimitedFormat() {
        return CSVFormat.Builder.create()
                .setDelimiter(',')
                .setHeader()
                .setSkipHeaderRecord(false)
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
    protected boolean shouldValidateHeaders() {
        return true;
    }

    @Override
    protected double getMaxErrorRate() {
        return 0.5; // Allow up to 50% error rate for social media data
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