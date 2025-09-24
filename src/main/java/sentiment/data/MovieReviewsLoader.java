package sentiment.data;

import org.apache.commons.csv.CSVRecord;

import java.time.LocalDateTime;
import java.util.*;

/**
 * Movie Reviews Dataset Loader implementing proper three-layer validation
 * while using FieldExtractor for flexible field mapping.
 * Layer 1: Raw Input Validation
 * Layer 2: Processing Validation
 * Layer 3: Model Input Validation
 */
class MovieReviewsLoader extends CsvLoaderBase {

    private static final String DATASET_TYPE = "Movie Reviews";
    private static final String[] SUPPORTED_EXTENSIONS = {".csv"};

    private static final FieldExtractor.FieldMapping MOVIE_TEXT_FIELDS =
            FieldExtractor.TEXT_FIELDS.withAdditional( "movie_review", "film_review");
    private static final FieldExtractor.FieldMapping MOVIE_SENTIMENT_FIELDS =
            FieldExtractor.SENTIMENT_FIELDS;

    @Override
    protected Dataset processRecord(CSVRecord record, DatasetLoadingStats stats,
                                    List<String> headers) {

        int recordNumber = (int) record.getRecordNumber();

        // Extract fields using FieldExtractor (flexible mapping)
        FieldExtractor.ExtractionResult<String> textResult =
                FieldExtractor.extractString(record, MOVIE_TEXT_FIELDS, 0);
        FieldExtractor.ExtractionResult<String> sentimentResult =
                FieldExtractor.extractString(record, MOVIE_SENTIMENT_FIELDS, 1);

        // Extract raw values for validation
        String reviewText = textResult.isPresent() ? textResult.getValue() : null;
        String sentimentLabel = sentimentResult.isPresent() ? sentimentResult.getValue() : null;

        // LAYER 1: Raw Input Validation
        DatasetValidationUtils.ValidationResult textValidation =
                DatasetValidationUtils.validateRawText(reviewText, DATASET_TYPE, recordNumber, stats);
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

        // LAYER 2: Processing Validation (Text Cleaning)
        String cleanedText = TextCleaningUtils.cleanMovieReviewText(validatedText);

        DatasetValidationUtils.ValidationResult processingValidation =
                DatasetValidationUtils.validateProcessedText(
                        cleanedText, validatedText, DATASET_TYPE, recordNumber, stats);
        if (!processingValidation.isValid()) {
            return null; // Processing destroyed too much content or failed
        }

        String processedText = processingValidation.getText();

        // Quality Control Check (spans multiple layers)
        DatasetValidationUtils.ValidationResult qualityResult =
                DatasetValidationUtils.performQualityControl(
                        processedText, DATASET_TYPE, recordNumber, stats);
        if (!qualityResult.isValid()) {
            return null; // Failed quality control (spam, fake review, etc.)
        }

        // LAYER 3: Model Input Validation
        DatasetValidationUtils.ValidationResult modelValidation =
                DatasetValidationUtils.validateModelInput(
                        processedText, validatedSentiment, DATASET_TYPE, recordNumber, stats);
        if (!modelValidation.isValid()) {
            return null; // Insufficient features or invalid for model
        }

        // If we get here, all validation layers passed
        Dataset finalDataset = modelValidation.getDataset();

        // Enhance with Movie Reviews-specific metadata
        return enhanceWithMovieMetadata(finalDataset, sentimentLabel, textResult, sentimentResult);
    }

    /**
     * Add Movie Reviews-specific metadata
     */
    private Dataset enhanceWithMovieMetadata(Dataset dataset, String originalLabel,
                                             FieldExtractor.ExtractionResult<String> textResult,
                                             FieldExtractor.ExtractionResult<String> sentimentResult) {
        return new Dataset.Builder(dataset.getText(), dataset.getSentiment())
                .source("movie_reviews")
                .originalLabel(originalLabel)
                .timestamp(LocalDateTime.now())
                .build();
    }

    @Override
    protected boolean shouldValidateHeaders() {
        return true;
    }

    @Override
    protected double getMaxErrorRate() {
        return 0.3; // Allow up to 30% error rate for movie reviews (stricter than social media)
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