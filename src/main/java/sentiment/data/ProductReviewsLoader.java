package sentiment.data;

import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Product Reviews Dataset Loader - simplified to handle CSV only.
 * Returns completely raw, unprocessed data with only basic structural validation.
 * Supports both rating-based (1-5 stars) and sentiment-based labels.
 */
class ProductReviewsLoader extends CsvLoaderBase {

    private static final Logger logger = LoggerFactory.getLogger(ProductReviewsLoader.class);
    private static final String DATASET_TYPE = "Product Reviews";
    private static final String[] SUPPORTED_EXTENSIONS = {".csv"};

    // Field mappings for product review data
    private static final FieldExtractor.FieldMapping PRODUCT_TEXT_FIELDS = FieldExtractor.TEXT_FIELDS;
    private static final FieldExtractor.FieldMapping PRODUCT_RATING_FIELDS =
            FieldExtractor.SENTIMENT_FIELDS.withAdditional("stars", "star_rating");

    @Override
    protected Dataset processRecord(CSVRecord record, DatasetLoadingStats stats, List<String> headers) {

        int recordNumber = (int) record.getRecordNumber();

        // Extract text field
        FieldExtractor.ExtractionResult<String> textResult =
                FieldExtractor.extractString(record, PRODUCT_TEXT_FIELDS, 0);

        // Try to extract numeric rating first (1-5 stars)
        FieldExtractor.ExtractionResult<Double> ratingResult =
                FieldExtractor.extractDouble(record, PRODUCT_RATING_FIELDS, 1);

        String reviewText = textResult.isPresent() ? textResult.getValue() : null;
        Double rating = ratingResult.isPresent() ? ratingResult.getValue() : null;
        String sentimentLabel = null;

        // If no numeric rating found, try sentiment labels
        if (!ratingResult.isPresent()) {
            FieldExtractor.ExtractionResult<String> sentimentResult =
                    FieldExtractor.extractString(record, FieldExtractor.SENTIMENT_FIELDS, 1);
            sentimentLabel = sentimentResult != null && sentimentResult.isPresent() ?
                    sentimentResult.getValue() : null;
        }

        //  Structural validation - check if fields exist and are parseable
        DatasetValidationUtils.ValidationResult textValidation =
                DatasetValidationUtils.validateRawText(reviewText, DATASET_TYPE, recordNumber, stats);
        if (!textValidation.isValid()) {
            return null; // Field missing or empty
        }

        // Handle both rating-based and sentiment-based labeling
        Dataset.SentimentLabel validatedSentiment;
        String originalLabel;

        if (rating != null) {
            // Rating-based sentiment (1-5 stars)
            if (rating < 1.0 || rating > 5.0) {
                stats.incrementInvalidSentimentFormat();
                logger.debug("Invalid rating {} for record {} in {}", rating, recordNumber, DATASET_TYPE);
                return null;
            }
            validatedSentiment = convertRatingToSentiment(rating);
            originalLabel = String.valueOf(rating);
        } else if (sentimentLabel != null) {
            // Direct sentiment labeling
            DatasetValidationUtils.ValidationResult sentimentValidation =
                    DatasetValidationUtils.validateRawSentiment(sentimentLabel, DATASET_TYPE, recordNumber, stats);
            if (!sentimentValidation.isValid()) {
                return null; // Sentiment not parseable
            }
            validatedSentiment = sentimentValidation.getSentiment();
            originalLabel = sentimentLabel;
        } else {
            stats.incrementInvalidSentimentFormat();
            logger.debug("Missing both rating and sentiment for record {} in {}", recordNumber, DATASET_TYPE);
            return null;
        }

        // Get the raw data (only whitespace trimmed for parsing safety)
        String rawText = textValidation.getText(); // Only trimmed, no other changes

        // Create Dataset with completely original text - no transformation whatsoever
        Dataset.Builder builder = new Dataset.Builder(rawText, validatedSentiment)
                .source("product_reviews")
                .originalLabel(originalLabel)
                .timestamp(LocalDateTime.now());

        // Add confidence based on rating if available
        if (rating != null) {
            builder.confidence(calculateConfidenceFromRating(rating));
        }

        return builder.build();
    }

    /**
     * Convert star rating to sentiment label
     */
    private Dataset.SentimentLabel convertRatingToSentiment(double rating) {
        if (rating <= 2.0) return Dataset.SentimentLabel.NEGATIVE;
        if (rating >= 4.0) return Dataset.SentimentLabel.POSITIVE;
        return Dataset.SentimentLabel.NEUTRAL; // 3-star ratings
    }

    /**
     * Calculate confidence based on rating extremity
     */
    private double calculateConfidenceFromRating(double rating) {
        if (rating == 1.0 || rating == 5.0) return 0.9; // Very confident on extremes
        if (rating == 2.0 || rating == 4.0) return 0.7; // Moderately confident
        return 0.5; // 3-star ratings are neutral, less confident
    }

    @Override
    protected double getMaxErrorRate() {
        return 0.4; // Allow up to 40% error rate for product reviews
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