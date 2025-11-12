package sentiment.data;

import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Simplified field extraction utility for CSV records only.
 * Provides flexible field mapping with fallback strategies for data loading.
 * Removed all JSON processing - focuses purely on CSV field extraction.
 */
public class FieldExtractor {

    private static final Logger logger = LoggerFactory.getLogger(FieldExtractor.class);

    // Pre-defined field name mappings for common data types
    public static final FieldMapping TEXT_FIELDS = new FieldMapping(
            "text", "content", "review_text", "review", "comment", "description",
            "tweet_text", "tweet", "message", "post", "body", "summary", "feedback"
    );

    public static final FieldMapping SENTIMENT_FIELDS = new FieldMapping(
            "sentiment", "label", "polarity", "class", "rating", "score",
            "emotion", "grade", "overall", "reviews.rating", "target"
    );

    public static final FieldMapping ID_FIELDS = new FieldMapping(
            "id", "tweet_id", "review_id", "post_id", "message_id",
            "reviewerID", "reviewer_id", "asin"
    );

    // Domain-specific field mappings (extend base mappings for specialized use cases)
    public static final FieldMapping PRODUCT_RATING_FIELDS = SENTIMENT_FIELDS.withAdditional(
            "stars", "star_rating"
    );

    /**
     * Immutable field mapping configuration
     */
    public static class FieldMapping {
        private final Set<String> fieldNames;
        private final List<String> orderedFieldNames;

        public FieldMapping(String... fieldNames) {
            this.orderedFieldNames = List.of(fieldNames);
            // Create lowercase set for case-insensitive matching
            Set<String> lowerCaseNames = new HashSet<>();
            for (String name : fieldNames) {
                lowerCaseNames.add(name.toLowerCase());
            }
            this.fieldNames = Collections.unmodifiableSet(lowerCaseNames);
        }

        public FieldMapping(Collection<String> fieldNames) {
            this.orderedFieldNames = new ArrayList<>(fieldNames);
            Set<String> lowerCaseNames = new HashSet<>();
            for (String name : fieldNames) {
                lowerCaseNames.add(name.toLowerCase());
            }
            this.fieldNames = Collections.unmodifiableSet(lowerCaseNames);
        }

        public Set<String> getFieldNames() {
            return fieldNames;
        }

        public List<String> getOrderedFieldNames() {
            return orderedFieldNames;
        }

        public boolean contains(String fieldName) {
            return fieldNames.contains(fieldName.toLowerCase());
        }

        /**
         * Create a new mapping with additional field names
         */
        public FieldMapping withAdditional(String... additionalFields) {
            List<String> combined = new ArrayList<>(orderedFieldNames);
            combined.addAll(List.of(additionalFields));
            return new FieldMapping(combined);
        }
    }

    /**
     * Result of field extraction with metadata
     */
    public static class ExtractionResult<T> {
        private final T value;
        private final String fieldName;
        private final boolean foundByName;
        private final int position;

        private ExtractionResult(T value, String fieldName, boolean foundByName, int position) {
            this.value = value;
            this.fieldName = fieldName;
            this.foundByName = foundByName;
            this.position = position;
        }

        public T getValue() { return value; }
        public String getFieldName() { return fieldName; }
        public boolean wasFoundByName() { return foundByName; }
        public int getPosition() { return position; }
        public boolean isPresent() { return value != null; }

        public static <T> ExtractionResult<T> empty() {
            return new ExtractionResult<>(null, null, false, -1);
        }

        public static <T> ExtractionResult<T> byName(T value, String fieldName) {
            return new ExtractionResult<>(value, fieldName, true, -1);
        }

        public static <T> ExtractionResult<T> byPosition(T value, int position) {
            return new ExtractionResult<>(value, null, false, position);
        }
    }

    /**
     * Extract string field from CSV record using field mapping with positional fallback
     */
    public static ExtractionResult<String> extractString(CSVRecord record, FieldMapping mapping,
                                                         int fallbackPosition) {
        // Try by field name first (preferred method)
        for (String fieldName : mapping.getOrderedFieldNames()) {
            if (record.isMapped(fieldName)) {
                String value = record.get(fieldName);
                if (isValidStringValue(value)) {
                    return ExtractionResult.byName(value.trim(), fieldName);
                }
            }
        }

        // Fall back to position if name-based extraction failed
        if (fallbackPosition >= 0 && fallbackPosition < record.size()) {
            String value = record.get(fallbackPosition);
            if (isValidStringValue(value)) {
                return ExtractionResult.byPosition(value.trim(), fallbackPosition);
            }
        }

        return ExtractionResult.empty();
    }

    /**
     * Extract string field from CSV record using only field mapping (no fallback)
     */
    public static ExtractionResult<String> extractString(CSVRecord record, FieldMapping mapping) {
        return extractString(record, mapping, -1);
    }

    /**
     * Extract numeric field (Double) from CSV record
     */
    public static ExtractionResult<Double> extractDouble(CSVRecord record, FieldMapping mapping,
                                                         int fallbackPosition) {
        // Try by field name first
        for (String fieldName : mapping.getOrderedFieldNames()) {
            if (record.isMapped(fieldName)) {
                String value = record.get(fieldName);
                if (isValidStringValue(value)) {
                    try {
                        double doubleValue = Double.parseDouble(value.trim());
                        return ExtractionResult.byName(doubleValue, fieldName);
                    } catch (NumberFormatException e) {
                        logger.debug("Failed to parse double from field '{}': {}", fieldName, value);
                    }
                }
            }
        }

        // Fall back to position
        if (fallbackPosition >= 0 && fallbackPosition < record.size()) {
            String value = record.get(fallbackPosition);
            if (isValidStringValue(value)) {
                try {
                    double doubleValue = Double.parseDouble(value.trim());
                    return ExtractionResult.byPosition(doubleValue, fallbackPosition);
                } catch (NumberFormatException e) {
                    logger.debug("Failed to parse double from position {}: {}", fallbackPosition, value);
                }
            }
        }

        return ExtractionResult.empty();
    }

    /**
     * Extract numeric field (Double) using only field mapping (no fallback)
     */
    public static ExtractionResult<Double> extractDouble(CSVRecord record, FieldMapping mapping) {
        return extractDouble(record, mapping, -1);
    }

    /**
     * Check if any of the specified fields exist in the record
     */
    public static boolean hasAnyField(CSVRecord record, FieldMapping mapping) {
        return mapping.getFieldNames().stream()
                .anyMatch(record::isMapped);
    }

    /**
     * Get all field names that exist in the record from the mapping
     */
    public static List<String> getExistingFields(CSVRecord record, FieldMapping mapping) {
        return mapping.getOrderedFieldNames().stream()
                .filter(record::isMapped)
                .collect(java.util.stream.Collectors.toList());
    }

    // Utility methods

    private static boolean isValidStringValue(String value) {
        return value != null && !value.trim().isEmpty();
    }

    /**
     * Create field mapping from header analysis
     */
    public static FieldMapping createMappingFromHeaders(List<String> headers, FieldMapping templateMapping) {
        List<String> matchingFields = headers.stream()
                .filter(header -> templateMapping.contains(header.toLowerCase()))
                .collect(java.util.stream.Collectors.toList());

        return new FieldMapping(matchingFields);
    }

    /**
     * Analyze field coverage in CSV headers
     */
    public static Map<String, Boolean> analyzeFieldCoverage(List<String> headers,
                                                            Map<String, FieldMapping> requiredMappings) {
        Map<String, Boolean> coverage = new HashMap<>();

        for (Map.Entry<String, FieldMapping> entry : requiredMappings.entrySet()) {
            boolean hasCoverage = headers.stream()
                    .anyMatch(header -> entry.getValue().contains(header.toLowerCase()));
            coverage.put(entry.getKey(), hasCoverage);
        }

        return coverage;
    }
}