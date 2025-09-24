package sentiment.data;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.*;

/**
 * Enhanced Product Reviews Dataset Loader supporting CSV, JSON, and JSONL formats
 * with proper three-layer validation and flexible field mapping.
 *
 * For CSV files: Extends CsvLoaderBase following the established architecture pattern
 * For JSON/JSONL files: Uses direct JSON processing
 */
class ProductReviewsLoader extends CsvLoaderBase {

    private static final Logger logger = LoggerFactory.getLogger(ProductReviewsLoader.class);
    private static final String DATASET_TYPE = "Product Reviews";
    private static final String[] SUPPORTED_EXTENSIONS = {".csv", ".json", ".jsonl"};

    private static final FieldExtractor.FieldMapping PRODUCT_TEXT_FIELDS = FieldExtractor.TEXT_FIELDS;
    private static final FieldExtractor.FieldMapping PRODUCT_RATING_FIELDS =
            FieldExtractor.SENTIMENT_FIELDS.withAdditional("stars", "star_rating");

    private final ObjectMapper objectMapper;

    public ProductReviewsLoader() {
        this.objectMapper = new ObjectMapper();
    }

    @Override
    protected List<Dataset> doLoadDataset(String filePath, DatasetLoadingStats stats) throws DataLoadingException {
        String extension = getFileExtension(filePath).toLowerCase();

        if (extension.equals(".csv")) {
            // Use CsvLoaderBase for CSV files (calls processRecord method)
            return super.doLoadDataset(filePath, stats);
        } else {
            // Handle JSON/JSONL directly
            try {
                return switch (extension) {
                    case ".jsonl" -> loadJsonLinesFormat(filePath, stats);
                    case ".json" -> loadSingleJsonFormat(filePath, stats);
                    default -> throw new DataLoadingException(
                            "Unsupported file format: " + extension,
                            filePath,
                            DATASET_TYPE
                    );
                };
            } catch (IOException e) {
                throw new DataLoadingException("Failed to read JSON file", filePath, DATASET_TYPE, e);
            }
        }
    }

    /**
     * Process CSV records using CsvLoaderBase template method pattern
     * This method is called by the parent class for each CSV record
     */
    @Override
    protected Dataset processRecord(CSVRecord record, DatasetLoadingStats stats, List<String> headers) {

        int recordNumber = (int) record.getRecordNumber();

        // Extract fields using FieldExtractor with flexible mapping
        FieldExtractor.ExtractionResult<String> textResult =
                FieldExtractor.extractString(record, PRODUCT_TEXT_FIELDS, 0);

        // Try to extract rating first, then sentiment
        FieldExtractor.ExtractionResult<Double> ratingResult =
                FieldExtractor.extractDouble(record, PRODUCT_RATING_FIELDS, 1);

        FieldExtractor.ExtractionResult<String> sentimentResult = null;
        Double rating = ratingResult.isPresent() ? ratingResult.getValue() : null;
        String sentimentLabel = null;

        if (!ratingResult.isPresent()) {
            // If no numeric rating, try sentiment labels
            sentimentResult = FieldExtractor.extractString(record,
                    FieldExtractor.SENTIMENT_FIELDS, 1);
            sentimentLabel = sentimentResult != null && sentimentResult.isPresent() ?
                    sentimentResult.getValue() : null;
        } else {
            // Check if the "sentiment" field actually contains a numeric rating
            FieldExtractor.ExtractionResult<String> rawSentimentResult =
                    FieldExtractor.extractString(record, FieldExtractor.SENTIMENT_FIELDS, 1);

            if (rawSentimentResult.isPresent()) {
                String rawSentiment = rawSentimentResult.getValue();
                // Try to parse as number - if it succeeds, treat as rating
                try {
                    double parsedRating = Double.parseDouble(rawSentiment);
                    if (parsedRating >= 1.0 && parsedRating <= 5.0) {
                        // This is actually a rating, not a sentiment label
                        rating = parsedRating;
                        logger.debug("Detected numeric rating '{}' in sentiment field for record {}",
                                rawSentiment, recordNumber);
                    } else {
                        // Not a valid rating, treat as sentiment label
                        sentimentLabel = rawSentiment;
                    }
                } catch (NumberFormatException e) {
                    // Not a number, treat as sentiment label
                    sentimentLabel = rawSentiment;
                }
            }
        }

        String reviewText = textResult.isPresent() ? textResult.getValue() : null;

        // Log first few records for debugging
        if (recordNumber <= 5) {
            logger.info("CSV Record {}: text='{}...', rating={}, sentiment='{}', columns={}",
                    recordNumber,
                    reviewText != null ? reviewText.substring(0, Math.min(30, reviewText.length())) : "null",
                    rating,
                    sentimentLabel,
                    record.size());
        }

        return processProductReview(reviewText, rating, sentimentLabel, recordNumber, stats,
                textResult, ratingResult);
    }

    /**
     * Load JSON Lines format
     */
    private List<Dataset> loadJsonLinesFormat(String filePath, DatasetLoadingStats stats)
            throws IOException, DataLoadingException {

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath, StandardCharsets.UTF_8))) {

            StringRecordProcessor processor = new StringRecordProcessor(
                    DATASET_TYPE,
                    filePath,
                    getMaxErrorRate(),
                    this::processJsonLine
            );

            List<String> lines = reader.lines().collect(java.util.stream.Collectors.toList());
            return processor.processRecords(lines, stats);
        }
    }

    /**
     * Load single JSON format
     */
    private List<Dataset> loadSingleJsonFormat(String filePath, DatasetLoadingStats stats)
            throws IOException, DataLoadingException {

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath, StandardCharsets.UTF_8))) {
            JsonNode rootNode = objectMapper.readTree(reader);

            JsonRecordProcessor processor = new JsonRecordProcessor(
                    DATASET_TYPE,
                    filePath,
                    getMaxErrorRate(),
                    this::processJsonReview
            );

            if (rootNode.isArray()) {
                List<JsonNode> nodes = new ArrayList<>();
                rootNode.forEach(nodes::add);
                return processor.processRecords(nodes, stats);
            } else {
                return processor.processRecords(List.of(rootNode), stats);
            }
        }
    }

    /**
     * Process a single JSON line
     */
    private Dataset processJsonLine(String line, DatasetLoadingStats stats, int lineNumber) throws Exception {
        JsonNode reviewNode = objectMapper.readTree(line);
        return processJsonReview(reviewNode, stats, lineNumber);
    }

    /**
     * Process a single JSON review with three-layer validation
     */
    private Dataset processJsonReview(JsonNode reviewNode, DatasetLoadingStats stats, int recordNumber) {

        FieldExtractor.ExtractionResult<String> textResult =
                FieldExtractor.extractString(reviewNode, PRODUCT_TEXT_FIELDS);
        FieldExtractor.ExtractionResult<Double> ratingResult =
                FieldExtractor.extractDouble(reviewNode, PRODUCT_RATING_FIELDS);

        String reviewText = textResult.isPresent() ? textResult.getValue() : null;
        Double rating = ratingResult.isPresent() ? ratingResult.getValue() : null;

        return processProductReview(reviewText, rating, null, recordNumber, stats,
                textResult, ratingResult);
    }

    /**
     * Unified processing method for CSV and JSON product reviews implementing three-layer validation
     */
    private Dataset processProductReview(String reviewText, Double rating, String sentimentLabel,
                                         int recordNumber, DatasetLoadingStats stats,
                                         FieldExtractor.ExtractionResult<String> textResult,
                                         Object ratingResult) {

        DatasetValidationUtils.ValidationResult textValidation =
                DatasetValidationUtils.validateRawText(reviewText, DATASET_TYPE, recordNumber, stats);
        if (!textValidation.isValid()) {
            return null;
        }

        Dataset.SentimentLabel validatedSentiment;
        String originalLabel;

        // Handle both rating-based and sentiment-based labeling
        if (rating != null) {
            // Rating-based sentiment (1-5 stars)
            if (rating < 1.0 || rating > 5.0) {
                stats.incrementInvalidLabel();
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
                return null;
            }
            validatedSentiment = sentimentValidation.getSentiment();
            originalLabel = sentimentLabel;
        } else {
            stats.incrementMissingRequiredFields();
            logger.debug("Missing both rating and sentiment for record {} in {}", recordNumber, DATASET_TYPE);
            return null;
        }

        String validatedText = textValidation.getText();

        String cleanedText = TextCleaningUtils.cleanProductReviewText(validatedText);

        DatasetValidationUtils.ValidationResult processingValidation =
                DatasetValidationUtils.validateProcessedText(
                        cleanedText, validatedText, DATASET_TYPE, recordNumber, stats);
        if (!processingValidation.isValid()) {
            return null;
        }

        String processedText = processingValidation.getText();

        // Quality Control Check
        DatasetValidationUtils.ValidationResult qualityResult =
                DatasetValidationUtils.performQualityControl(
                        processedText, DATASET_TYPE, recordNumber, stats);
        if (!qualityResult.isValid()) {
            return null;
        }

        DatasetValidationUtils.ValidationResult modelValidation =
                DatasetValidationUtils.validateModelInput(
                        processedText, validatedSentiment, DATASET_TYPE, recordNumber, stats);
        if (!modelValidation.isValid()) {
            return null;
        }

        Dataset finalDataset = modelValidation.getDataset();

        // Enhance with Product Reviews-specific metadata
        return enhanceWithProductMetadata(finalDataset, originalLabel, rating, textResult, ratingResult);
    }

    /**
     * Add Product Reviews-specific metadata
     */
    private Dataset enhanceWithProductMetadata(Dataset dataset, String originalLabel, Double rating,
                                               FieldExtractor.ExtractionResult<String> textResult,
                                               Object ratingResult) {
        Dataset.Builder builder = new Dataset.Builder(dataset.getText(), dataset.getSentiment())
                .source("product_reviews")
                .originalLabel(originalLabel)
                .timestamp(LocalDateTime.now());

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
        return Dataset.SentimentLabel.NEUTRAL;
    }

    /**
     * Calculate confidence based on rating extremity
     */
    private double calculateConfidenceFromRating(double rating) {
        if (rating == 1.0 || rating == 5.0) return 0.9;
        if (rating == 2.0 || rating == 4.0) return 0.7;
        return 0.5; // 3-star ratings are less confident
    }

    /**
     * Get file extension
     */
    private String getFileExtension(String filePath) {
        int lastDot = filePath.lastIndexOf('.');
        return lastDot > 0 ? filePath.substring(lastDot) : "";
    }

    @Override
    protected boolean shouldValidateHeaders() {
        return true;
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