package sentiment.data;

import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Stream;

/**
 * Utility class for flexible field extraction from CSV records and JSON nodes.
 * Provides robust field mapping with fallback strategies, validation, and type conversion.
 */
public class FieldExtractor {

    private static final Logger logger = LoggerFactory.getLogger(FieldExtractor.class);

    // Pre-defined field name mappings for common data types
    public static final FieldMapping TEXT_FIELDS = new FieldMapping(
            "text", "content", "review_text", "review", "comment", "description",
            "tweet_text", "tweet", "message", "post", "body", "summary"
    );

    public static final FieldMapping SENTIMENT_FIELDS = new FieldMapping(
            "sentiment", "label", "polarity", "class", "rating", "score",
            "emotion", "grade", "overall", "reviews.rating"
    );

    public static final FieldMapping ID_FIELDS = new FieldMapping(
            "id", "tweet_id", "review_id", "post_id", "message_id",
            "reviewerID", "reviewer_id", "asin"
    );

    public static final FieldMapping TIMESTAMP_FIELDS = new FieldMapping(
            "timestamp", "date", "time", "created_at", "reviewTime",
            "review_time", "unixReviewTime"
    );

    /**
     * Immutable field mapping configuration
     */
    public static class FieldMapping {
        private final Set<String> fieldNames;
        private final List<String> orderedFieldNames;

        public FieldMapping(String... fieldNames) {
            this.orderedFieldNames = List.of(fieldNames);
            this.fieldNames = Set.of(fieldNames);
        }

        public FieldMapping(Collection<String> fieldNames) {
            this.orderedFieldNames = new ArrayList<>(fieldNames);
            this.fieldNames = new HashSet<>(fieldNames);
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
     * Extract string field with custom validation
     */
    public static ExtractionResult<String> extractString(CSVRecord record, FieldMapping mapping,
                                                         int fallbackPosition, Predicate<String> validator) {
        ExtractionResult<String> result = extractString(record, mapping, fallbackPosition);

        if (result.isPresent() && validator != null && !validator.test(result.getValue())) {
            return ExtractionResult.empty();
        }

        return result;
    }

    /**
     * Extract and convert field to specific type
     */
    public static <T> ExtractionResult<T> extractAndConvert(CSVRecord record, FieldMapping mapping,
                                                            int fallbackPosition, Function<String, T> converter) {
        ExtractionResult<String> stringResult = extractString(record, mapping, fallbackPosition);

        if (!stringResult.isPresent()) {
            return ExtractionResult.empty();
        }

        try {
            T convertedValue = converter.apply(stringResult.getValue());
            if (convertedValue != null) {
                return stringResult.wasFoundByName()
                        ? ExtractionResult.byName(convertedValue, stringResult.getFieldName())
                        : ExtractionResult.byPosition(convertedValue, stringResult.getPosition());
            }
        } catch (Exception e) {
            logger.debug("Failed to convert field value '{}': {}", stringResult.getValue(), e.getMessage());
        }

        return ExtractionResult.empty();
    }

    /**
     * Extract numeric field (Double)
     */
    public static ExtractionResult<Double> extractDouble(CSVRecord record, FieldMapping mapping,
                                                         int fallbackPosition) {
        return extractAndConvert(record, mapping, fallbackPosition, value -> {
            try {
                return Double.parseDouble(value);
            } catch (NumberFormatException e) {
                return null;
            }
        });
    }

    /**
     * Extract numeric field (Integer)
     */
    public static ExtractionResult<Integer> extractInteger(CSVRecord record, FieldMapping mapping,
                                                           int fallbackPosition) {
        return extractAndConvert(record, mapping, fallbackPosition, value -> {
            try {
                return Integer.parseInt(value);
            } catch (NumberFormatException e) {
                return null;
            }
        });
    }

    /**
     * Extract boolean field with flexible parsing
     */
    public static ExtractionResult<Boolean> extractBoolean(CSVRecord record, FieldMapping mapping,
                                                           int fallbackPosition) {
        return extractAndConvert(record, mapping, fallbackPosition, value -> {
            String normalized = value.toLowerCase().trim();
            return switch (normalized) {
                case "true", "1", "yes", "y" -> true;
                case "false", "0", "no", "n" -> false;
                default -> null;
            };
        });
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

    /**
     * Extract multiple fields using different mappings
     */
    public static class MultiFieldExtractor {
        private final CSVRecord record;
        private final Map<String, ExtractionResult<?>> results = new HashMap<>();

        public MultiFieldExtractor(CSVRecord record) {
            this.record = record;
        }

        public MultiFieldExtractor extract(String key, FieldMapping mapping, int fallbackPosition) {
            results.put(key, extractString(record, mapping, fallbackPosition));
            return this;
        }

        public MultiFieldExtractor extract(String key, FieldMapping mapping) {
            results.put(key, extractString(record, mapping));
            return this;
        }

        public MultiFieldExtractor extractDouble(String key, FieldMapping mapping, int fallbackPosition) {
            results.put(key, FieldExtractor.extractDouble(record, mapping, fallbackPosition));
            return this;
        }

        @SuppressWarnings("unchecked")
        public <T> ExtractionResult<T> get(String key) {
            return (ExtractionResult<T>) results.get(key);
        }

        public String getString(String key) {
            ExtractionResult<String> result = get(key);
            return result != null ? result.getValue() : null;
        }

        public Double getDouble(String key) {
            ExtractionResult<Double> result = get(key);
            return result != null ? result.getValue() : null;
        }

        public boolean hasAll(String... keys) {
            return Stream.of(keys).allMatch(key -> {
                ExtractionResult<?> result = results.get(key);
                return result != null && result.isPresent();
            });
        }

        public Map<String, ExtractionResult<?>> getAllResults() {
            return new HashMap<>(results);
        }
    }

    /**
     * Create a multi-field extractor for complex extraction scenarios
     */
    public static MultiFieldExtractor forRecord(CSVRecord record) {
        return new MultiFieldExtractor(record);
    }

    // JSON extraction methods (for ProductReviewsLoader)

    /**
     * Extract string field from Jackson JsonNode using field mapping
     */
    public static ExtractionResult<String> extractString(com.fasterxml.jackson.databind.JsonNode node,
                                                         FieldMapping mapping) {
        for (String fieldName : mapping.getOrderedFieldNames()) {
            com.fasterxml.jackson.databind.JsonNode fieldNode = node.get(fieldName);
            if (fieldNode != null && !fieldNode.isNull() && fieldNode.isTextual()) {
                String value = fieldNode.asText();
                if (isValidStringValue(value)) {
                    return ExtractionResult.byName(value.trim(), fieldName);
                }
            }
        }
        return ExtractionResult.empty();
    }

    /**
     * Extract numeric field from Jackson JsonNode
     */
    public static ExtractionResult<Double> extractDouble(com.fasterxml.jackson.databind.JsonNode node,
                                                         FieldMapping mapping) {
        for (String fieldName : mapping.getOrderedFieldNames()) {
            com.fasterxml.jackson.databind.JsonNode fieldNode = node.get(fieldName);
            if (fieldNode != null && !fieldNode.isNull()) {
                try {
                    if (fieldNode.isNumber()) {
                        return ExtractionResult.byName(fieldNode.asDouble(), fieldName);
                    } else if (fieldNode.isTextual()) {
                        double value = Double.parseDouble(fieldNode.asText());
                        return ExtractionResult.byName(value, fieldName);
                    }
                } catch (NumberFormatException e) {
                    logger.debug("Invalid numeric format in field '{}': {}", fieldName, fieldNode.asText());
                }
            }
        }
        return ExtractionResult.empty();
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

    /**
     * Generate field extraction report for debugging
     */
    public static String generateExtractionReport(CSVRecord record,
                                                  Map<String, FieldMapping> mappings) {
        StringBuilder report = new StringBuilder();
        report.append("Field Extraction Report for Record ").append(record.getRecordNumber()).append(":\n");

        for (Map.Entry<String, FieldMapping> entry : mappings.entrySet()) {
            String category = entry.getKey();
            FieldMapping mapping = entry.getValue();

            ExtractionResult<String> result = extractString(record, mapping);

            report.append("  ").append(category).append(": ");
            if (result.isPresent()) {
                report.append("Found '").append(result.getValue()).append("'");
                if (result.wasFoundByName()) {
                    report.append(" (by name: ").append(result.getFieldName()).append(")");
                } else {
                    report.append(" (by position: ").append(result.getPosition()).append(")");
                }
            } else {
                report.append("Not found");
            }
            report.append("\n");
        }

        return report.toString();
    }
}