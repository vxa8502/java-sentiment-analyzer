package sentiment.data;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.time.LocalDateTime;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive unit tests for Dataset class.
 *
 * Tests cover:
 * - Constructors and builder pattern
 * - Validation logic
 * - Sentiment label parsing
 * - Factory methods
 * - Utility methods
 * - Object contract (equals, hashCode, toString)
 */
@DisplayName("Dataset Unit Tests")
class DatasetTest {

    // ==================== SENTIMENT LABEL TESTS ====================

    @Test
    @DisplayName("SentimentLabel.fromString should parse positive labels")
    void testSentimentLabel_ParsePositive() {
        assertEquals(Dataset.SentimentLabel.POSITIVE, Dataset.SentimentLabel.fromString("positive"));
        assertEquals(Dataset.SentimentLabel.POSITIVE, Dataset.SentimentLabel.fromString("pos"));
        assertEquals(Dataset.SentimentLabel.POSITIVE, Dataset.SentimentLabel.fromString("1"));
        assertEquals(Dataset.SentimentLabel.POSITIVE, Dataset.SentimentLabel.fromString("good"));
        assertEquals(Dataset.SentimentLabel.POSITIVE, Dataset.SentimentLabel.fromString("liked"));
    }

    @Test
    @DisplayName("SentimentLabel.fromString should parse negative labels")
    void testSentimentLabel_ParseNegative() {
        assertEquals(Dataset.SentimentLabel.NEGATIVE, Dataset.SentimentLabel.fromString("negative"));
        assertEquals(Dataset.SentimentLabel.NEGATIVE, Dataset.SentimentLabel.fromString("neg"));
        assertEquals(Dataset.SentimentLabel.NEGATIVE, Dataset.SentimentLabel.fromString("0"));
        assertEquals(Dataset.SentimentLabel.NEGATIVE, Dataset.SentimentLabel.fromString("-1"));
        assertEquals(Dataset.SentimentLabel.NEGATIVE, Dataset.SentimentLabel.fromString("bad"));
        assertEquals(Dataset.SentimentLabel.NEGATIVE, Dataset.SentimentLabel.fromString("disliked"));
    }

    @Test
    @DisplayName("SentimentLabel.fromString should parse neutral labels")
    void testSentimentLabel_ParseNeutral() {
        assertEquals(Dataset.SentimentLabel.NEUTRAL, Dataset.SentimentLabel.fromString("neutral"));
        assertEquals(Dataset.SentimentLabel.NEUTRAL, Dataset.SentimentLabel.fromString("neut"));
        assertEquals(Dataset.SentimentLabel.NEUTRAL, Dataset.SentimentLabel.fromString("2"));
        assertEquals(Dataset.SentimentLabel.NEUTRAL, Dataset.SentimentLabel.fromString("mixed"));
    }

    @Test
    @DisplayName("SentimentLabel.fromString should be case-insensitive")
    void testSentimentLabel_CaseInsensitive() {
        assertEquals(Dataset.SentimentLabel.POSITIVE, Dataset.SentimentLabel.fromString("POSITIVE"));
        assertEquals(Dataset.SentimentLabel.POSITIVE, Dataset.SentimentLabel.fromString("Positive"));
        assertEquals(Dataset.SentimentLabel.NEGATIVE, Dataset.SentimentLabel.fromString("NEGATIVE"));
    }

    @Test
    @DisplayName("SentimentLabel.fromString should trim whitespace")
    void testSentimentLabel_TrimsWhitespace() {
        assertEquals(Dataset.SentimentLabel.POSITIVE, Dataset.SentimentLabel.fromString("  positive  "));
        assertEquals(Dataset.SentimentLabel.NEGATIVE, Dataset.SentimentLabel.fromString(" negative "));
    }

    @Test
    @DisplayName("SentimentLabel.fromString should throw for null")
    void testSentimentLabel_NullThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> Dataset.SentimentLabel.fromString(null));
    }

    @Test
    @DisplayName("SentimentLabel.fromString should throw for empty string")
    void testSentimentLabel_EmptyThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> Dataset.SentimentLabel.fromString(""));
        assertThrows(IllegalArgumentException.class,
                () -> Dataset.SentimentLabel.fromString("   "));
    }

    @Test
    @DisplayName("SentimentLabel.fromString should throw for unknown label")
    void testSentimentLabel_UnknownThrows() {
        assertThrows(IllegalArgumentException.class,
                () -> Dataset.SentimentLabel.fromString("unknown"));
        assertThrows(IllegalArgumentException.class,
                () -> Dataset.SentimentLabel.fromString("invalid"));
    }

    @Test
    @DisplayName("SentimentLabel should have correct numeric values")
    void testSentimentLabel_NumericValues() {
        assertEquals(1, Dataset.SentimentLabel.POSITIVE.getNumericValue());
        assertEquals(-1, Dataset.SentimentLabel.NEGATIVE.getNumericValue());
        assertEquals(0, Dataset.SentimentLabel.NEUTRAL.getNumericValue());
    }

    @Test
    @DisplayName("SentimentLabel should have correct display names")
    void testSentimentLabel_DisplayNames() {
        assertEquals("positive", Dataset.SentimentLabel.POSITIVE.getDisplayName());
        assertEquals("negative", Dataset.SentimentLabel.NEGATIVE.getDisplayName());
        assertEquals("neutral", Dataset.SentimentLabel.NEUTRAL.getDisplayName());
    }

    // ==================== CONSTRUCTOR TESTS ====================

    @Test
    @DisplayName("Builder constructor should create valid dataset")
    void testBuilder_CreatesValidDataset() {
        Dataset dataset = new Dataset.Builder("Great product!", Dataset.SentimentLabel.POSITIVE)
                .confidence(0.9)
                .source("test-source")
                .build();

        assertNotNull(dataset);
        assertEquals("Great product!", dataset.getText());
        assertEquals(Dataset.SentimentLabel.POSITIVE, dataset.getSentiment());
        assertEquals(0.9, dataset.getConfidence(), 0.001);
        assertEquals("test-source", dataset.getSource());
    }

    @Test
    @DisplayName("Builder with string label should parse sentiment")
    void testBuilder_WithStringLabel() {
        Dataset dataset = new Dataset.Builder("Bad product", "negative").build();

        assertEquals(Dataset.SentimentLabel.NEGATIVE, dataset.getSentiment());
        assertEquals("negative", dataset.getOriginalLabel());
    }

    @Test
    @DisplayName("Builder should generate ID if not provided")
    void testBuilder_GeneratesId() {
        Dataset dataset = new Dataset.Builder("Test text", Dataset.SentimentLabel.POSITIVE).build();

        assertNotNull(dataset.getId());
        assertFalse(dataset.getId().isEmpty());
    }

    @Test
    @DisplayName("Builder should use provided ID")
    void testBuilder_UsesProvidedId() {
        String customId = "custom-id-123";
        Dataset dataset = new Dataset.Builder("Test text", Dataset.SentimentLabel.POSITIVE)
                .id(customId)
                .build();

        assertEquals(customId, dataset.getId());
    }

    @Test
    @DisplayName("Builder should set timestamp if not provided")
    void testBuilder_GeneratesTimestamp() {
        Dataset dataset = new Dataset.Builder("Test text", Dataset.SentimentLabel.POSITIVE).build();

        assertNotNull(dataset.getTimestamp());
    }

    @Test
    @DisplayName("Builder should use provided timestamp")
    void testBuilder_UsesProvidedTimestamp() {
        LocalDateTime customTime = LocalDateTime.of(2024, 1, 1, 12, 0);
        Dataset dataset = new Dataset.Builder("Test text", Dataset.SentimentLabel.POSITIVE)
                .timestamp(customTime)
                .build();

        assertEquals(customTime, dataset.getTimestamp());
    }

    @Test
    @DisplayName("Builder should default confidence to 0.5")
    void testBuilder_DefaultConfidence() {
        Dataset dataset = new Dataset.Builder("Test text", Dataset.SentimentLabel.POSITIVE).build();

        assertEquals(0.5, dataset.getConfidence(), 0.001);
    }

    @Test
    @DisplayName("Convenience constructor should work")
    void testConvenienceConstructor() {
        Dataset dataset = new Dataset("Great movie!", "positive");

        assertEquals("Great movie!", dataset.getText());
        assertEquals(Dataset.SentimentLabel.POSITIVE, dataset.getSentiment());
    }

    @Test
    @DisplayName("Convenience constructor with confidence should work")
    void testConvenienceConstructorWithConfidence() {
        Dataset dataset = new Dataset("Great movie!", Dataset.SentimentLabel.POSITIVE, 0.95);

        assertEquals("Great movie!", dataset.getText());
        assertEquals(Dataset.SentimentLabel.POSITIVE, dataset.getSentiment());
        assertEquals(0.95, dataset.getConfidence(), 0.001);
    }

    // ==================== VALIDATION TESTS ====================

    @Test
    @DisplayName("Builder should throw when text is null")
    void testBuilder_NullText_Throws() {
        assertThrows(IllegalArgumentException.class,
                () -> new Dataset.Builder(null, Dataset.SentimentLabel.POSITIVE).build());
    }

    @Test
    @DisplayName("Builder should throw when text is empty")
    void testBuilder_EmptyText_Throws() {
        assertThrows(IllegalArgumentException.class,
                () -> new Dataset.Builder("", Dataset.SentimentLabel.POSITIVE).build());
        assertThrows(IllegalArgumentException.class,
                () -> new Dataset.Builder("   ", Dataset.SentimentLabel.POSITIVE).build());
    }

    @Test
    @DisplayName("Builder should throw when text exceeds max length")
    void testBuilder_TextTooLong_Throws() {
        String longText = "a".repeat(10001);
        assertThrows(IllegalArgumentException.class,
                () -> new Dataset.Builder(longText, Dataset.SentimentLabel.POSITIVE).build());
    }

    @Test
    @DisplayName("Builder should throw when sentiment is null")
    void testBuilder_NullSentiment_Throws() {
        assertThrows(NullPointerException.class,
                () -> new Dataset.Builder("Test", (Dataset.SentimentLabel) null).build());
    }

    @Test
    @DisplayName("Builder should throw when confidence is negative")
    void testBuilder_NegativeConfidence_Throws() {
        assertThrows(IllegalArgumentException.class,
                () -> new Dataset.Builder("Test", Dataset.SentimentLabel.POSITIVE)
                        .confidence(-0.1)
                        .build());
    }

    @Test
    @DisplayName("Builder should throw when confidence exceeds 1.0")
    void testBuilder_ConfidenceTooHigh_Throws() {
        assertThrows(IllegalArgumentException.class,
                () -> new Dataset.Builder("Test", Dataset.SentimentLabel.POSITIVE)
                        .confidence(1.1)
                        .build());
    }

    @Test
    @DisplayName("Builder should trim text")
    void testBuilder_TrimsText() {
        Dataset dataset = new Dataset.Builder("  Test text  ", Dataset.SentimentLabel.POSITIVE).build();

        assertEquals("Test text", dataset.getText());
    }

    // ==================== FACTORY METHOD TESTS ====================

    @Test
    @DisplayName("createPositive should create positive dataset")
    void testCreatePositive() {
        Dataset dataset = Dataset.createPositive("Great!");

        assertEquals(Dataset.SentimentLabel.POSITIVE, dataset.getSentiment());
        assertEquals("Great!", dataset.getText());
    }

    @Test
    @DisplayName("createNegative should create negative dataset")
    void testCreateNegative() {
        Dataset dataset = Dataset.createNegative("Terrible!");

        assertEquals(Dataset.SentimentLabel.NEGATIVE, dataset.getSentiment());
        assertEquals("Terrible!", dataset.getText());
    }

    @Test
    @DisplayName("createNeutral should create neutral dataset")
    void testCreateNeutral() {
        Dataset dataset = Dataset.createNeutral("It's okay");

        assertEquals(Dataset.SentimentLabel.NEUTRAL, dataset.getSentiment());
        assertEquals("It's okay", dataset.getText());
    }

    @Test
    @DisplayName("fromCsvRow should create dataset with source")
    void testFromCsvRow() {
        Dataset dataset = Dataset.fromCsvRow("Review text", "positive", "csv-loader");

        assertEquals("Review text", dataset.getText());
        assertEquals(Dataset.SentimentLabel.POSITIVE, dataset.getSentiment());
        assertEquals("csv-loader", dataset.getSource());
    }

    // ==================== UTILITY METHOD TESTS ====================

    @Test
    @DisplayName("hasHighConfidence should return true for confidence >= 0.8")
    void testHasHighConfidence_True() {
        Dataset dataset1 = new Dataset("Test", Dataset.SentimentLabel.POSITIVE, 0.8);
        Dataset dataset2 = new Dataset("Test", Dataset.SentimentLabel.POSITIVE, 0.95);

        assertTrue(dataset1.hasHighConfidence());
        assertTrue(dataset2.hasHighConfidence());
    }

    @Test
    @DisplayName("hasHighConfidence should return false for confidence < 0.8")
    void testHasHighConfidence_False() {
        Dataset dataset = new Dataset("Test", Dataset.SentimentLabel.POSITIVE, 0.7);

        assertFalse(dataset.hasHighConfidence());
    }

    @Test
    @DisplayName("isFromSource should match source case-insensitively")
    void testIsFromSource() {
        Dataset dataset = new Dataset.Builder("Test", Dataset.SentimentLabel.POSITIVE)
                .source("IMDB")
                .build();

        assertTrue(dataset.isFromSource("IMDB"));
        assertTrue(dataset.isFromSource("imdb"));
        assertTrue(dataset.isFromSource("ImDb"));
        assertFalse(dataset.isFromSource("Twitter"));
    }

    @Test
    @DisplayName("isFromSource should handle null source")
    void testIsFromSource_NullSource() {
        Dataset dataset = new Dataset.Builder("Test", Dataset.SentimentLabel.POSITIVE).build();

        assertFalse(dataset.isFromSource("any-source"));
    }

    @Test
    @DisplayName("getTextLength should return correct length")
    void testGetTextLength() {
        Dataset dataset = new Dataset("12345", "positive");

        assertEquals(5, dataset.getTextLength());
    }

    @Test
    @DisplayName("getSentimentAsString should return display name")
    void testGetSentimentAsString() {
        Dataset dataset = new Dataset("Test", "positive");

        assertEquals("positive", dataset.getSentimentAsString());
    }

    @Test
    @DisplayName("getSentimentAsNumeric should return numeric value")
    void testGetSentimentAsNumeric() {
        Dataset positive = new Dataset("Test", "positive");
        Dataset negative = new Dataset("Test", "negative");
        Dataset neutral = new Dataset("Test", "neutral");

        assertEquals(1, positive.getSentimentAsNumeric());
        assertEquals(-1, negative.getSentimentAsNumeric());
        assertEquals(0, neutral.getSentimentAsNumeric());
    }

    // ==================== OBJECT CONTRACT TESTS ====================

    @Test
    @DisplayName("equals should return true for same object")
    void testEquals_SameObject() {
        Dataset dataset = Dataset.createPositive("Test");

        assertEquals(dataset, dataset);
    }

    @Test
    @DisplayName("equals should return true for equal datasets")
    void testEquals_EqualDatasets() {
        String id = "same-id";
        LocalDateTime timestamp = LocalDateTime.now();

        Dataset dataset1 = new Dataset.Builder("Test", Dataset.SentimentLabel.POSITIVE)
                .id(id)
                .timestamp(timestamp)
                .confidence(0.9)
                .source("test")
                .build();

        Dataset dataset2 = new Dataset.Builder("Test", Dataset.SentimentLabel.POSITIVE)
                .id(id)
                .timestamp(timestamp)
                .confidence(0.9)
                .source("test")
                .build();

        assertEquals(dataset1, dataset2);
    }

    @Test
    @DisplayName("equals should return false for null")
    void testEquals_Null() {
        Dataset dataset = Dataset.createPositive("Test");

        assertNotEquals(null, dataset);
    }

    @Test
    @DisplayName("equals should return false for different class")
    void testEquals_DifferentClass() {
        Dataset dataset = Dataset.createPositive("Test");

        assertNotEquals(dataset, "String object");
    }

    @Test
    @DisplayName("hashCode should be consistent")
    void testHashCode_Consistent() {
        Dataset dataset = Dataset.createPositive("Test");

        int hash1 = dataset.hashCode();
        int hash2 = dataset.hashCode();

        assertEquals(hash1, hash2);
    }

    @Test
    @DisplayName("hashCode should be equal for equal objects")
    void testHashCode_EqualObjects() {
        String id = "same-id";
        LocalDateTime timestamp = LocalDateTime.now();

        Dataset dataset1 = new Dataset.Builder("Test", Dataset.SentimentLabel.POSITIVE)
                .id(id)
                .timestamp(timestamp)
                .build();

        Dataset dataset2 = new Dataset.Builder("Test", Dataset.SentimentLabel.POSITIVE)
                .id(id)
                .timestamp(timestamp)
                .build();

        assertEquals(dataset1.hashCode(), dataset2.hashCode());
    }

    @Test
    @DisplayName("toString should contain key information")
    void testToString() {
        Dataset dataset = new Dataset.Builder("Test text for toString", Dataset.SentimentLabel.POSITIVE)
                .source("test-source")
                .confidence(0.85)
                .build();

        String str = dataset.toString();

        assertTrue(str.contains("Test text"));
        assertTrue(str.contains("positive"));
        assertTrue(str.contains("0.850"));
        assertTrue(str.contains("test-source"));
    }

    @Test
    @DisplayName("toString should truncate long text")
    void testToString_TruncatesLongText() {
        String longText = "a".repeat(100);
        Dataset dataset = Dataset.createPositive(longText);

        String str = dataset.toString();

        assertTrue(str.contains("..."));
        assertTrue(str.length() < longText.length() + 100);
    }

    @Test
    @DisplayName("toDetailedString should contain all fields")
    void testToDetailedString() {
        Dataset dataset = new Dataset.Builder("Test text", Dataset.SentimentLabel.POSITIVE)
                .source("test-source")
                .confidence(0.85)
                .originalLabel("pos")
                .build();

        String detailed = dataset.toDetailedString();

        assertTrue(detailed.contains("Test text"));
        assertTrue(detailed.contains("positive"));
        assertTrue(detailed.contains("1"));
        assertTrue(detailed.contains("0.850"));
        assertTrue(detailed.contains("test-source"));
        assertTrue(detailed.contains("pos"));
        assertTrue(detailed.contains("9"));
    }
}
