package sentiment.data;

import org.apache.commons.csv.CSVRecord;
import java.time.LocalDateTime;
import java.util.List;

/**
 * Movie Reviews Dataset Loader - returns completely raw, unprocessed data.
 * Only performs basic structural validation to ensure data can be loaded.
 */
class MovieReviewsLoader extends CsvLoaderBase {

    private static final String DATASET_TYPE = "Movie Reviews";
    private static final String[] SUPPORTED_EXTENSIONS = {".csv"};

    // Field mappings for flexible extraction
    private static final FieldExtractor.FieldMapping MOVIE_TEXT_FIELDS =
            FieldExtractor.TEXT_FIELDS.withAdditional("movie_review", "film_review");
    private static final FieldExtractor.FieldMapping MOVIE_SENTIMENT_FIELDS =
            FieldExtractor.SENTIMENT_FIELDS;

    @Override
    protected Dataset processRecord(CSVRecord record, DatasetLoadingStats stats, List<String> headers) {

        int recordNumber = (int) record.getRecordNumber();

        // Extract fields using FieldExtractor
        FieldExtractor.ExtractionResult<String> textResult =
                FieldExtractor.extractString(record, MOVIE_TEXT_FIELDS, 0);
        FieldExtractor.ExtractionResult<String> sentimentResult =
                FieldExtractor.extractString(record, MOVIE_SENTIMENT_FIELDS, 1);

        String reviewText = textResult.isPresent() ? textResult.getValue() : null;
        String sentimentLabel = sentimentResult.isPresent() ? sentimentResult.getValue() : null;

        // Structural validation - check if fields exist and are parseable
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

        // Create Dataset with completely original text - no transformation whatsoever
        return new Dataset.Builder(rawText, parsedSentiment)
                .source("movie_reviews")
                .originalLabel(sentimentLabel)
                .timestamp(LocalDateTime.now())
                .build();
    }

    @Override
    protected double getMaxErrorRate() {
        return 0.3; // Allow up to 30% error rate for movie reviews
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