package sentiment.data;

import sentiment.util.ValidationUtils;

import java.time.LocalDateTime;
import java.util.Objects;
import java.util.UUID;

/**
 * Data model representing a text sample with sentiment label and metadata.
 * Supports both binary (positive/negative) and ternary (positive/neutral/negative) sentiment schemes.
 */
public class Dataset {

    // Core data fields
    private final String id;
    private final String text;
    private final SentimentLabel sentiment;
    private final double confidence;

    // Metadata fields
    private final String source;
    private final LocalDateTime timestamp;
    private final String originalLabel; // Store original string label for reference

    /**
     * Enum for standardized sentiment labels with support for different schemes
     */
    public enum SentimentLabel {
        POSITIVE("positive", 1),
        NEGATIVE("negative", -1),
        NEUTRAL("neutral", 0);

        private final String displayName;
        private final int numericValue;

        SentimentLabel(String displayName, int numericValue) {
            this.displayName = displayName;
            this.numericValue = numericValue;
        }

        public String getDisplayName() {
            return displayName;
        }

        public int getNumericValue() {
            return numericValue;
        }

        /**
         * Parse sentiment from various string representations.
         * Throws IllegalArgumentException if the label cannot be parsed.
         * For lenient parsing that returns null on failure, use fromStringOrNull().
         */
        public static SentimentLabel fromString(String label) {
            ValidationUtils.requireNonEmpty(label, "Sentiment label");
            SentimentLabel result = fromStringOrNull(label);
            if (result == null) {
                throw new IllegalArgumentException("Unknown sentiment label: " + label);
            }
            return result;
        }

        /**
         * Parse sentiment from various string representations, returning null if unparseable.
         * Handles: text labels (positive/negative/neutral, pos/neg, good/bad, liked/disliked),
         * numeric labels (0/1/-1/2/4), and star ratings (1.0-5.0 scale).
         */
        public static SentimentLabel fromStringOrNull(String label) {
            if (label == null || label.isBlank()) {
                return null;
            }

            String normalized = label.trim().toLowerCase();

            // Direct string matches (includes common dataset conventions)
            return switch (normalized) {
                case "positive", "pos", "1", "4", "good", "liked" -> POSITIVE;
                case "negative", "neg", "0", "-1", "bad", "disliked" -> NEGATIVE;
                case "neutral", "neut", "2", "mixed" -> NEUTRAL;
                default -> parseNumericRating(normalized);
            };
        }

        /**
         * Parse numeric ratings (e.g., 1-5 star ratings) into sentiment labels.
         * Rating <= 2.0 -> NEGATIVE, >= 4.0 -> POSITIVE, else NEUTRAL.
         */
        private static SentimentLabel parseNumericRating(String value) {
            try {
                double rating = Double.parseDouble(value);
                if (rating <= 2.0) return NEGATIVE;
                if (rating >= 4.0) return POSITIVE;
                return NEUTRAL;
            } catch (NumberFormatException e) {
                return null;
            }
        }
    }

    /**
     * Private constructor - use Builder for object creation
     */
    private Dataset(Builder builder) {
        this.id = builder.id != null ? builder.id : UUID.randomUUID().toString();
        this.text = validateAndCleanText(builder.text);
        this.sentiment = Objects.requireNonNull(builder.sentiment, "Sentiment cannot be null");
        this.confidence = validateConfidence(builder.confidence);
        this.source = builder.source;
        this.timestamp = builder.timestamp != null ? builder.timestamp : LocalDateTime.now();
        this.originalLabel = builder.originalLabel;
    }

    /**
     * Convenience constructor for basic usage
     */
    public Dataset(String text, String sentimentLabel) {
        this(new Builder(text, SentimentLabel.fromString(sentimentLabel)));
    }

    /**
     * Convenience constructor with confidence
     */
    public Dataset(String text, SentimentLabel sentiment, double confidence) {
        this(new Builder(text, sentiment).confidence(confidence));
    }

    // Validation methods
    private String validateAndCleanText(String text) {
        ValidationUtils.requireNonEmpty(text);

        String cleaned = text.trim();
        if (cleaned.length() > 10000) {
            throw new IllegalArgumentException("Text exceeds maximum length of 10000 characters");
        }

        return cleaned;
    }

    private double validateConfidence(double confidence) {
        if (confidence < 0.0 || confidence > 1.0) {
            throw new IllegalArgumentException("Confidence must be between 0.0 and 1.0, got: " + confidence);
        }
        return confidence;
    }

    // Getters
    public String getId() { return id; }
    public String getText() { return text; }
    public SentimentLabel getSentiment() { return sentiment; }
    public double getConfidence() { return confidence; }
    public String getSource() { return source; }
    public LocalDateTime getTimestamp() { return timestamp; }
    public String getOriginalLabel() { return originalLabel; }

    // Utility methods
    public boolean hasHighConfidence() {
        return confidence >= 0.8;
    }

    public boolean isFromSource(String sourceName) {
        return source != null && source.equalsIgnoreCase(sourceName);
    }

    public int getTextLength() {
        return text.length();
    }

    public String getSentimentAsString() {
        return sentiment.getDisplayName();
    }

    public int getSentimentAsNumeric() {
        return sentiment.getNumericValue();
    }

    // Builder Pattern Implementation
    public static class Builder {
        // Required fields
        private final String text;
        private final SentimentLabel sentiment;

        // Optional fields with defaults
        private String id;
        private double confidence = 0.5; // Default neutral confidence
        private String source;
        private LocalDateTime timestamp;
        private String originalLabel;

        public Builder(String text, SentimentLabel sentiment) {
            this.text = text;
            this.sentiment = sentiment;
        }

        public Builder(String text, String sentimentLabel) {
            this.text = text;
            this.sentiment = SentimentLabel.fromString(sentimentLabel);
            this.originalLabel = sentimentLabel; // Store original for reference
        }

        public Builder id(String id) {
            this.id = id;
            return this;
        }

        public Builder confidence(double confidence) {
            this.confidence = confidence;
            return this;
        }

        public Builder source(String source) {
            this.source = source;
            return this;
        }

        public Builder timestamp(LocalDateTime timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder originalLabel(String originalLabel) {
            this.originalLabel = originalLabel;
            return this;
        }

        public Dataset build() {
            return new Dataset(this);
        }
    }

    // Static factory methods for common use cases
    public static Dataset createPositive(String text) {
        return new Builder(text, SentimentLabel.POSITIVE).build();
    }

    public static Dataset createNegative(String text) {
        return new Builder(text, SentimentLabel.NEGATIVE).build();
    }

    public static Dataset createNeutral(String text) {
        return new Builder(text, SentimentLabel.NEUTRAL).build();
    }

    public static Dataset fromCsvRow(String text, String label, String source) {
        return new Builder(text, label)
                .source(source)
                .build();
    }

    // Object contract methods
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;

        Dataset dataset = (Dataset) obj;
        return Double.compare(dataset.confidence, confidence) == 0 &&
                Objects.equals(id, dataset.id) &&
                Objects.equals(text, dataset.text) &&
                sentiment == dataset.sentiment &&
                Objects.equals(source, dataset.source) &&
                Objects.equals(timestamp, dataset.timestamp);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id, text, sentiment, confidence, source, timestamp);
    }

    @Override
    public String toString() {
        return String.format("Dataset{id='%s', text='%.50s%s', sentiment=%s, confidence=%.3f, source='%s'}",
                id,
                text,
                text.length() > 50 ? "..." : "",
                sentiment.getDisplayName(),
                confidence,
                source != null ? source : "unknown");
    }

    /**
     * Create a JSON-like representation for debugging
     */
    public String toDetailedString() {
        return String.format("""
                Dataset {
                    id: %s
                    text: "%s"
                    sentiment: %s (%d)
                    confidence: %.3f
                    source: %s
                    timestamp: %s
                    originalLabel: %s
                    textLength: %d
                }""",
                id, text, sentiment.getDisplayName(), sentiment.getNumericValue(),
                confidence, source, timestamp, originalLabel, text.length());
    }
}