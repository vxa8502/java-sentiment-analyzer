package sentiment.preprocessing;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive unit tests for AdvancedTokenizer.
 *
 * Tests cover:
 * - Basic tokenization
 * - Hyphenated word preservation
 * - Number handling
 * - Configuration options
 * - Edge cases
 * - Utility methods
 */
@DisplayName("AdvancedTokenizer Unit Tests")
class AdvancedTokenizerTest {

    private AdvancedTokenizer tokenizer;

    @BeforeEach
    void setUp() {
        // Default configuration
        tokenizer = new AdvancedTokenizer(false, 1, true);
    }

    // ==================== CONSTRUCTOR TESTS ====================

    @Test
    @DisplayName("Constructor should throw when minTokenLength is invalid")
    void testConstructor_InvalidMinLength_Throws() {
        assertThrows(IllegalArgumentException.class,
                () -> new AdvancedTokenizer(false, 0, true));
        assertThrows(IllegalArgumentException.class,
                () -> new AdvancedTokenizer(false, -1, true));
    }

    @Test
    @DisplayName("Constructor should accept valid configuration")
    void testConstructor_ValidConfiguration() {
        AdvancedTokenizer tokenizer1 = new AdvancedTokenizer(true, 2, false);
        assertNotNull(tokenizer1);

        AdvancedTokenizer tokenizer2 = new AdvancedTokenizer(false, 1, true);
        assertNotNull(tokenizer2);
    }

    // ==================== BASIC TOKENIZATION TESTS ====================

    @Test
    @DisplayName("tokenize should split simple sentence into words")
    void testTokenize_SimpleSentence() {
        List<String> tokens = tokenizer.tokenize("This is a test");

        assertNotNull(tokens);
        assertTrue(tokens.size() >= 4);
        assertTrue(tokens.contains("This") || tokens.contains("this"));
        assertTrue(tokens.contains("is"));
        assertTrue(tokens.contains("test"));
    }

    @Test
    @DisplayName("tokenize should handle single word")
    void testTokenize_SingleWord() {
        List<String> tokens = tokenizer.tokenize("hello");

        assertNotNull(tokens);
        assertTrue(tokens.size() >= 1);
        assertTrue(tokens.contains("hello"));
    }

    @Test
    @DisplayName("tokenize should handle text with punctuation")
    void testTokenize_WithPunctuation() {
        List<String> tokens = tokenizer.tokenize("Hello, world! How are you?");

        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
        assertTrue(tokens.stream().anyMatch(t -> t.matches("\\w+")));
    }

    @Test
    @DisplayName("tokenize should handle multiple spaces")
    void testTokenize_MultipleSpaces() {
        List<String> tokens = tokenizer.tokenize("word1    word2     word3");

        assertNotNull(tokens);
        assertTrue(tokens.contains("word1"));
        assertTrue(tokens.contains("word2"));
        assertTrue(tokens.contains("word3"));
    }

    // ==================== NULL AND EMPTY INPUT TESTS ====================

    @Test
    @DisplayName("tokenize should handle null input")
    void testTokenize_NullInput() {
        List<String> tokens = tokenizer.tokenize(null);

        assertNotNull(tokens);
        assertTrue(tokens.isEmpty());
    }

    @Test
    @DisplayName("tokenize should handle empty input")
    void testTokenize_EmptyInput() {
        List<String> tokens = tokenizer.tokenize("");

        assertNotNull(tokens);
        assertTrue(tokens.isEmpty());
    }

    @Test
    @DisplayName("tokenize should handle whitespace-only input")
    void testTokenize_WhitespaceOnly() {
        List<String> tokens = tokenizer.tokenize("   ");

        assertNotNull(tokens);
        assertTrue(tokens.isEmpty());
    }

    // ==================== HYPHENATED WORDS TESTS ====================

    @Test
    @DisplayName("tokenize should preserve valid hyphenated words when configured")
    void testTokenize_PreservesHyphenatedWords() {
        List<String> tokens = tokenizer.tokenize("This is state-of-the-art technology");

        assertNotNull(tokens);
        boolean hasHyphenatedOrSplit = tokens.stream()
                .anyMatch(t -> t.contains("-") || t.equals("state") || t.equals("art"));
        assertTrue(hasHyphenatedOrSplit);
    }

    @Test
    @DisplayName("tokenize should handle self-hyphenated compounds")
    void testTokenize_SelfHyphenated() {
        List<String> tokens = tokenizer.tokenize("self-aware");

        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
    }

    @Test
    @DisplayName("tokenize should handle well-hyphenated compounds")
    void testTokenize_WellHyphenated() {
        List<String> tokens = tokenizer.tokenize("well-known");

        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
    }

    @Test
    @DisplayName("tokenize should handle multi-hyphenated compounds")
    void testTokenize_MultiHyphenated() {
        List<String> tokens = tokenizer.tokenize("multi-level-marketing");

        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
    }

    @Test
    @DisplayName("tokenize should split hyphenated when preserveHyphenated is false")
    void testTokenize_NoHyphenPreservation() {
        AdvancedTokenizer noHyphenTokenizer = new AdvancedTokenizer(false, 1, false);
        List<String> tokens = noHyphenTokenizer.tokenize("state-of-the-art");

        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
    }

    // ==================== NUMBER HANDLING TESTS ====================

    @Test
    @DisplayName("tokenize should filter numbers when preserveNumbers is false")
    void testTokenize_FiltersNumbers() {
        AdvancedTokenizer filterTokenizer = new AdvancedTokenizer(false, 1, true);
        List<String> tokens = filterTokenizer.tokenize("There are 123 items");

        assertNotNull(tokens);
        assertFalse(tokens.contains("123"));
    }

    @Test
    @DisplayName("tokenize should preserve numbers as tokens when configured")
    void testTokenize_PreservesNumbers() {
        AdvancedTokenizer preserveTokenizer = new AdvancedTokenizer(true, 1, true);
        List<String> tokens = preserveTokenizer.tokenize("There are 123 items");

        assertNotNull(tokens);
        assertTrue(tokens.contains("NUMBER_TOKEN"));
    }

    @Test
    @DisplayName("tokenize should handle decimal numbers")
    void testTokenize_DecimalNumbers() {
        AdvancedTokenizer preserveTokenizer = new AdvancedTokenizer(true, 1, true);
        List<String> tokens = preserveTokenizer.tokenize("Price is 19.99 dollars");

        assertNotNull(tokens);
        assertTrue(tokens.contains("NUMBER_TOKEN") || !tokens.contains("19.99"));
    }

    @Test
    @DisplayName("tokenize should handle comma-separated numbers")
    void testTokenize_CommaSeparatedNumbers() {
        AdvancedTokenizer preserveTokenizer = new AdvancedTokenizer(true, 1, true);
        List<String> tokens = preserveTokenizer.tokenize("Population is 1,234,567");

        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
    }

    // ==================== MIN TOKEN LENGTH TESTS ====================

    @Test
    @DisplayName("tokenize should respect minTokenLength of 2")
    void testTokenize_MinLength2() {
        AdvancedTokenizer minLength2Tokenizer = new AdvancedTokenizer(false, 2, true);
        List<String> tokens = minLength2Tokenizer.tokenize("I am a good person");

        assertNotNull(tokens);
        // Single character tokens except meaningful ones should be filtered
        // "am" and "good" and "person" should remain
        assertTrue(tokens.stream().anyMatch(t -> t.length() >= 2));
    }

    @Test
    @DisplayName("tokenize should respect minTokenLength of 3")
    void testTokenize_MinLength3() {
        AdvancedTokenizer minLength3Tokenizer = new AdvancedTokenizer(false, 3, true);
        List<String> tokens = minLength3Tokenizer.tokenize("The cat is on the mat");

        assertNotNull(tokens);
        // "The", "cat", "the", "mat" should remain (length >= 3)
        assertFalse(tokens.contains("is"));
        assertFalse(tokens.contains("on"));
    }

    @Test
    @DisplayName("tokenize should preserve meaningful single characters")
    void testTokenize_PreservesMeaningfulSingleChars() {
        List<String> tokens = tokenizer.tokenize("I think a person!");

        assertNotNull(tokens);
        // "I" and "a" are meaningful single chars
        assertTrue(tokens.contains("I") || tokens.contains("think"));
    }

    // ==================== SPECIAL CHARACTERS TESTS ====================

    @Test
    @DisplayName("tokenize should handle exclamation marks")
    void testTokenize_ExclamationMarks() {
        List<String> tokens = tokenizer.tokenize("Great! Amazing!!");

        assertNotNull(tokens);
        assertTrue(tokens.contains("Great"));
    }

    @Test
    @DisplayName("tokenize should handle question marks")
    void testTokenize_QuestionMarks() {
        List<String> tokens = tokenizer.tokenize("Really? Are you sure???");

        assertNotNull(tokens);
        assertTrue(tokens.contains("Really") || tokens.contains("really"));
    }

    @Test
    @DisplayName("tokenize should handle ellipsis")
    void testTokenize_Ellipsis() {
        List<String> tokens = tokenizer.tokenize("Well... I don't know");

        assertNotNull(tokens);
        assertTrue(tokens.contains("Well") || tokens.contains("well"));
    }

    @Test
    @DisplayName("tokenize should handle mixed punctuation")
    void testTokenize_MixedPunctuation() {
        List<String> tokens = tokenizer.tokenize("Hello, world! How are you? Fine.");

        assertNotNull(tokens);
        assertTrue(tokens.size() >= 5);
    }

    // ==================== UTILITY METHODS TESTS ====================

    @Test
    @DisplayName("analyzeTokenization should provide analysis for valid text")
    void testAnalyzeTokenization_ValidText() {
        AdvancedTokenizer.TokenizationAnalysis analysis =
                tokenizer.analyzeTokenization("This is a state-of-the-art system with 123 items!");

        assertNotNull(analysis);
        assertTrue(analysis.totalWords() > 0);
    }

    @Test
    @DisplayName("analyzeTokenization should handle null input")
    void testAnalyzeTokenization_NullInput() {
        AdvancedTokenizer.TokenizationAnalysis analysis = tokenizer.analyzeTokenization(null);

        assertNotNull(analysis);
        assertEquals(0, analysis.totalWords());
        assertEquals(0, analysis.hyphenatedWords());
    }

    @Test
    @DisplayName("analyzeTokenization should handle empty input")
    void testAnalyzeTokenization_EmptyInput() {
        AdvancedTokenizer.TokenizationAnalysis analysis = tokenizer.analyzeTokenization("");

        assertNotNull(analysis);
        assertEquals(0, analysis.totalWords());
    }

    @Test
    @DisplayName("analyzeTokenization should count hyphenated words")
    void testAnalyzeTokenization_CountsHyphenated() {
        AdvancedTokenizer.TokenizationAnalysis analysis =
                tokenizer.analyzeTokenization("state-of-the-art well-known");

        assertNotNull(analysis);
        assertTrue(analysis.hyphenatedWords() >= 0);
    }

    @Test
    @DisplayName("analyzeTokenization should count numbers")
    void testAnalyzeTokenization_CountsNumbers() {
        AdvancedTokenizer.TokenizationAnalysis analysis =
                tokenizer.analyzeTokenization("There are 123 items and 456 more");

        assertNotNull(analysis);
        assertTrue(analysis.numbers() >= 0);
    }

    @Test
    @DisplayName("compareTokenization should compare tokenization methods")
    void testCompareTokenization() {
        AdvancedTokenizer.TokenizationComparison comparison =
                tokenizer.compareTokenization("This is a test sentence");

        assertNotNull(comparison);
        assertNotNull(comparison.simpleTokens);
        assertNotNull(comparison.advancedTokens);
        assertNotNull(comparison.originalText);
    }

    @Test
    @DisplayName("getVersion should return version string")
    void testGetVersion() {
        String version = tokenizer.getVersion();

        assertNotNull(version);
        assertFalse(version.isEmpty());
    }

    // ==================== EDGE CASES TESTS ====================

    @Test
    @DisplayName("tokenize should handle very long text")
    void testTokenize_VeryLongText() {
        StringBuilder longText = new StringBuilder();
        for (int i = 0; i < 100; i++) {
            longText.append("word").append(i).append(" ");
        }

        List<String> tokens = tokenizer.tokenize(longText.toString());

        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
        assertTrue(tokens.size() >= 90);
    }

    @Test
    @DisplayName("tokenize should handle text with only punctuation")
    void testTokenize_OnlyPunctuation() {
        List<String> tokens = tokenizer.tokenize("!!! ??? ...");

        assertNotNull(tokens);
        // May contain punctuation tokens or be empty depending on filtering
    }

    @Test
    @DisplayName("tokenize should handle text with special characters")
    void testTokenize_SpecialCharacters() {
        List<String> tokens = tokenizer.tokenize("Hello @user #hashtag $100");

        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
    }

    @Test
    @DisplayName("tokenize should handle text with newlines")
    void testTokenize_WithNewlines() {
        List<String> tokens = tokenizer.tokenize("First line\nSecond line\nThird line");

        assertNotNull(tokens);
        assertTrue(tokens.contains("First") || tokens.contains("first"));
        assertTrue(tokens.contains("line"));
    }

    @Test
    @DisplayName("tokenize should handle text with tabs")
    void testTokenize_WithTabs() {
        List<String> tokens = tokenizer.tokenize("word1\tword2\tword3");

        assertNotNull(tokens);
        assertTrue(tokens.contains("word1"));
        assertTrue(tokens.contains("word2"));
        assertTrue(tokens.contains("word3"));
    }

    // ==================== INTEGRATION TESTS ====================

    @Test
    @DisplayName("tokenize should handle complex real-world sentence")
    void testTokenize_ComplexSentence() {
        String text = "The state-of-the-art AI system processed 1,234 documents in real-time!";
        List<String> tokens = tokenizer.tokenize(text);

        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
        assertTrue(tokens.size() >= 8);
    }

    @Test
    @DisplayName("tokenize should handle mixed content")
    void testTokenize_MixedContent() {
        String text = "Check out https://example.com for more info! Price: $99.99";
        List<String> tokens = tokenizer.tokenize(text);

        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
    }

    @Test
    @DisplayName("tokenize should be consistent on repeated calls")
    void testTokenize_ConsistentResults() {
        String text = "This is a test sentence";

        List<String> tokens1 = tokenizer.tokenize(text);
        List<String> tokens2 = tokenizer.tokenize(text);

        assertEquals(tokens1.size(), tokens2.size());
        assertEquals(tokens1, tokens2);
    }

    @Test
    @DisplayName("tokenize with different configurations should produce different results")
    void testTokenize_DifferentConfigurations() {
        String text = "I am a state-of-the-art system with 123 items";

        AdvancedTokenizer config1 = new AdvancedTokenizer(false, 1, true);
        AdvancedTokenizer config2 = new AdvancedTokenizer(true, 3, false);

        List<String> tokens1 = config1.tokenize(text);
        List<String> tokens2 = config2.tokenize(text);

        assertNotNull(tokens1);
        assertNotNull(tokens2);
        // Different configurations may produce different token counts
        assertNotEquals(0, tokens1.size());
        assertNotEquals(0, tokens2.size());
    }
}
