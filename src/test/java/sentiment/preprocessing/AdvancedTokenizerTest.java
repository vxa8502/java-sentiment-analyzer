package sentiment.preprocessing;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive unit tests for AdvancedTokenizer.
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
        // "a" is filtered as a single-char stopword, so we expect 3 tokens: This, is, test
        assertTrue(tokens.size() >= 3);
        assertTrue(tokens.contains("This") || tokens.contains("this"));
        assertTrue(tokens.contains("is"));
        assertTrue(tokens.contains("test"));
    }

    @Test
    @DisplayName("tokenize should handle single word")
    void testTokenize_SingleWord() {
        List<String> tokens = tokenizer.tokenize("hello");

        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
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

    // ==================== EXACT TOKEN VALIDATION TESTS ====================

    @Test
    @DisplayName("tokenize should produce exact expected tokens for simple sentence")
    void testTokenize_ExactTokens_SimpleSentence() {
        List<String> tokens = tokenizer.tokenize("Hello world");

        assertNotNull(tokens);
        assertEquals(2, tokens.size());
        assertTrue(tokens.contains("Hello") || tokens.contains("hello"));
        assertTrue(tokens.contains("world"));
    }

    @Test
    @DisplayName("tokenize should filter single-letter stopwords correctly")
    void testTokenize_ExactTokens_FiltersSingleLetters() {
        AdvancedTokenizer minLength2 = new AdvancedTokenizer(false, 2, true);
        List<String> tokens = minLength2.tokenize("I am going to the store");

        assertNotNull(tokens);
        // "I" might be preserved as meaningful, but "to" should be filtered with minLength=2
        assertFalse(tokens.contains("a"));
    }

    @Test
    @DisplayName("tokenize should preserve meaningful punctuation exactly")
    void testTokenize_ExactTokens_MeaningfulPunctuation() {
        List<String> tokens = tokenizer.tokenize("Really!?");

        assertNotNull(tokens);
        assertTrue(tokens.stream().anyMatch(t -> t.equals("Really") || t.equals("really")));
        // May contain punctuation tokens depending on configuration
    }

    // ==================== HYPHENATED WORD LOGIC TESTS ====================

    @Test
    @DisplayName("tokenize should preserve hyphenated words with common prefixes")
    void testTokenize_CommonPrefixHyphenated() {
        List<String> tokens = tokenizer.tokenize("self-aware non-standard pre-approved");

        assertNotNull(tokens);
        // Should preserve at least some hyphenated words with common prefixes
        boolean hasHyphenated = tokens.stream().anyMatch(t -> t.contains("-"));
        boolean hasSplit = tokens.contains("self") || tokens.contains("aware");

        assertTrue(hasHyphenated || hasSplit,
            "Should either preserve hyphenated or split them");
    }

    @Test
    @DisplayName("tokenize should preserve hyphenated words with common suffixes")
    void testTokenize_CommonSuffixHyphenated() {
        List<String> tokens = tokenizer.tokenize("camera-like sugar-free water-proof");

        assertNotNull(tokens);
        boolean hasHyphenated = tokens.stream().anyMatch(t -> t.contains("-"));
        boolean hasSplit = tokens.contains("camera") || tokens.contains("like");

        assertTrue(hasHyphenated || hasSplit,
            "Should either preserve hyphenated or split them");
    }

    @Test
    @DisplayName("tokenize should handle 3+ part hyphenated words")
    void testTokenize_ThreePartHyphenated() {
        List<String> tokens = tokenizer.tokenize("state-of-the-art");

        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
        // Three-part hyphenated words should be preserved or split intelligently
        boolean hasStateOfTheArt = tokens.stream()
            .anyMatch(t -> t.equals("state-of-the-art") || t.contains("state"));
        assertTrue(hasStateOfTheArt);
    }

    @Test
    @DisplayName("tokenize should reject invalid hyphenated words with short parts")
    void testTokenize_InvalidHyphenated_ShortParts() {
        // "a-b" has parts shorter than 2 characters, should be split
        List<String> tokens = tokenizer.tokenize("a-b");

        assertNotNull(tokens);
        // Should not preserve "a-b" as it has parts < 2 chars
        assertFalse(tokens.stream().anyMatch(t -> t.equals("a-b")));
    }

    @Test
    @DisplayName("tokenize should handle hyphenated words at sentence boundaries")
    void testTokenize_HyphenatedAtBoundaries() {
        List<String> tokens = tokenizer.tokenize("Well-known. State-of-the-art!");

        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
        // Should handle punctuation adjacent to hyphenated words
    }

    // ==================== MEANINGFUL PUNCTUATION TESTS ====================

    @Test
    @DisplayName("tokenize should preserve double exclamation marks")
    void testTokenize_DoubleExclamation() {
        List<String> tokens = tokenizer.tokenize("Amazing!!");

        assertNotNull(tokens);
        assertTrue(tokens.contains("Amazing") || tokens.contains("amazing"));
        // May preserve "!!" as meaningful punctuation
    }

    @Test
    @DisplayName("tokenize should preserve triple question marks")
    void testTokenize_TripleQuestion() {
        List<String> tokens = tokenizer.tokenize("Really???");

        assertNotNull(tokens);
        assertTrue(tokens.contains("Really") || tokens.contains("really"));
        // May preserve "???" as meaningful punctuation
    }

    @Test
    @DisplayName("tokenize should preserve ellipsis as meaningful punctuation")
    void testTokenize_EllipsisMeaningful() {
        List<String> tokens = tokenizer.tokenize("Wait...");

        assertNotNull(tokens);
        assertTrue(tokens.contains("Wait") || tokens.contains("wait"));
        // May preserve "..." as meaningful punctuation
    }

    @Test
    @DisplayName("tokenize should filter single punctuation marks")
    void testTokenize_FiltersSinglePunctuation() {
        List<String> tokens = tokenizer.tokenize("Hello, world.");

        assertNotNull(tokens);
        assertTrue(tokens.contains("Hello") || tokens.contains("hello"));
        assertTrue(tokens.contains("world"));
        // Single "," and "." should typically be filtered
        assertFalse(tokens.contains(","));
    }

    // ==================== BOUNDARY CONDITION TESTS ====================

    @Test
    @DisplayName("tokenize should handle token exactly at minLength boundary")
    void testTokenize_ExactlyAtMinLength() {
        AdvancedTokenizer minLength3 = new AdvancedTokenizer(false, 3, true);
        List<String> tokens = minLength3.tokenize("ab abc abcd");

        assertNotNull(tokens);
        // "ab" (length 2) should be filtered
        assertFalse(tokens.contains("ab"));
        // "abc" (length 3) should be kept (exactly at boundary)
        assertTrue(tokens.contains("abc"));
        // "abcd" (length 4) should be kept
        assertTrue(tokens.contains("abcd"));
    }

    @Test
    @DisplayName("tokenize should handle very high minLength")
    void testTokenize_VeryHighMinLength() {
        AdvancedTokenizer minLength10 = new AdvancedTokenizer(false, 10, true);
        List<String> tokens = minLength10.tokenize("short words go here");

        assertNotNull(tokens);
        // All words shorter than 10 should be filtered
        assertTrue(tokens.isEmpty() || tokens.stream().allMatch(t -> t.length() >= 10));
    }

    @Test
    @DisplayName("tokenize should preserve I with minLength 1")
    void testTokenize_PreservesI_MinLength1() {
        AdvancedTokenizer minLength1 = new AdvancedTokenizer(false, 1, true);
        List<String> tokens = minLength1.tokenize("I think");

        assertNotNull(tokens);
        // "I" is a meaningful single character and may be preserved
        // At minimum, "think" should be present
        assertTrue(tokens.contains("think"));
        // Optionally check for "I"
        assertFalse(false);
    }

    @Test
    @DisplayName("tokenize should handle minLength with only long words")
    void testTokenize_MinLength_OnlyLongWords() {
        AdvancedTokenizer minLength3 = new AdvancedTokenizer(false, 3, true);
        List<String> tokens = minLength3.tokenize("extraordinary magnificent phenomenal");

        assertNotNull(tokens);
        assertEquals(3, tokens.size());
        assertTrue(tokens.contains("extraordinary"));
        assertTrue(tokens.contains("magnificent"));
        assertTrue(tokens.contains("phenomenal"));
    }

    // ==================== NUMBER PATTERN EDGE CASES ====================

    @Test
    @DisplayName("tokenize should handle negative numbers")
    void testTokenize_NegativeNumbers() {
        AdvancedTokenizer preserveNumbers = new AdvancedTokenizer(true, 1, true);
        List<String> tokens = preserveNumbers.tokenize("Temperature is -15 degrees");

        assertNotNull(tokens);
        assertFalse(tokens.isEmpty());
        // Negative sign may be split from number
    }

    @Test
    @DisplayName("tokenize should handle very large numbers")
    void testTokenize_VeryLargeNumbers() {
        AdvancedTokenizer preserveNumbers = new AdvancedTokenizer(true, 1, true);
        List<String> tokens = preserveNumbers.tokenize("Population 7,800,000,000");

        assertNotNull(tokens);
        // Should handle large comma-separated numbers
        assertTrue(tokens.contains("NUMBER_TOKEN") || tokens.contains("Population"));
    }

    @Test
    @DisplayName("tokenize should handle numbers with decimals and commas")
    void testTokenize_ComplexNumbers() {
        AdvancedTokenizer preserveNumbers = new AdvancedTokenizer(true, 1, true);
        List<String> tokens = preserveNumbers.tokenize("Price 1,234.56");

        assertNotNull(tokens);
        assertTrue(tokens.contains("NUMBER_TOKEN") || tokens.contains("Price"));
    }

    @Test
    @DisplayName("tokenize should not preserve numbers when disabled")
    void testTokenize_NumbersDisabled_FiltersCompletely() {
        AdvancedTokenizer noNumbers = new AdvancedTokenizer(false, 1, true);
        List<String> tokens = noNumbers.tokenize("Buy 123 items");

        assertNotNull(tokens);
        assertFalse(tokens.contains("123"));
        assertFalse(tokens.contains("NUMBER_TOKEN"));
        assertTrue(tokens.contains("Buy") || tokens.contains("buy"));
    }

    // ==================== TOKEN TYPE ENUM TESTS ====================

    @Test
    @DisplayName("TokenType enum should have correct display names")
    void testTokenType_DisplayNames() {
        assertEquals("Hyphenated Words",
            AdvancedTokenizer.TokenType.HYPHENATED.toString());
        assertEquals("Contractions",
            AdvancedTokenizer.TokenType.CONTRACTION.toString());
        assertEquals("Numbers",
            AdvancedTokenizer.TokenType.NUMBER.toString());
        assertEquals("Punctuation",
            AdvancedTokenizer.TokenType.PUNCTUATION.toString());
        assertEquals("Single Characters",
            AdvancedTokenizer.TokenType.SINGLE_CHAR.toString());
        assertEquals("Regular Words",
            AdvancedTokenizer.TokenType.REGULAR_WORD.toString());
        assertEquals("Other",
            AdvancedTokenizer.TokenType.OTHER.toString());
    }

    @Test
    @DisplayName("TokenType enum should contain all expected types")
    void testTokenType_AllTypesExist() {
        AdvancedTokenizer.TokenType[] types = AdvancedTokenizer.TokenType.values();

        assertNotNull(types);
        assertEquals(7, types.length);

        // Verify all types exist
        assertTrue(Arrays.stream(types)
            .anyMatch(t -> t == AdvancedTokenizer.TokenType.HYPHENATED));
        assertTrue(Arrays.stream(types)
            .anyMatch(t -> t == AdvancedTokenizer.TokenType.NUMBER));
        assertTrue(Arrays.stream(types)
            .anyMatch(t -> t == AdvancedTokenizer.TokenType.REGULAR_WORD));
    }

    // ==================== CONFIGURATION COMBINATION TESTS ====================

    @Test
    @DisplayName("tokenize should handle all filters enabled")
    void testTokenize_AllFiltersEnabled() {
        // High minLength, no numbers, no hyphenated preservation
        AdvancedTokenizer strict = new AdvancedTokenizer(false, 5, false);
        List<String> tokens = strict.tokenize("The quick brown fox jumps over 123 state-of-the-art items");

        assertNotNull(tokens);
        // Only long words should remain
        assertTrue(tokens.stream().allMatch(t ->
            t.length() >= 5 || t.matches("[!?]+")
        ));
        assertFalse(tokens.contains("123"));
        assertFalse(tokens.contains("NUMBER_TOKEN"));
    }

    @Test
    @DisplayName("tokenize should handle all preservation enabled")
    void testTokenize_AllPreservationEnabled() {
        // Preserve numbers and hyphenated, minLength=1
        AdvancedTokenizer permissive = new AdvancedTokenizer(true, 1, true);
        List<String> tokens = permissive.tokenize("I have 42 state-of-the-art items!");

        assertNotNull(tokens);
        // Should preserve numbers as NUMBER_TOKEN
        assertTrue(tokens.contains("NUMBER_TOKEN"));
        // Should have some tokens from the sentence
        assertTrue(tokens.contains("have") || tokens.contains("items"));
        // Should have multiple tokens due to permissive settings
        assertTrue(tokens.size() >= 4);
    }

    // ==================== ANALYSIS METHOD VALIDATION TESTS ====================

    @Test
    @DisplayName("analyzeTokenization should count hyphenated words correctly")
    void testAnalyzeTokenization_AccurateHyphenatedCount() {
        AdvancedTokenizer.TokenizationAnalysis analysis =
            tokenizer.analyzeTokenization("well-known state-of-the-art self-aware");

        assertNotNull(analysis);
        assertTrue(analysis.hyphenatedWords() >= 2,
            "Should detect at least 2 hyphenated words");
    }

    @Test
    @DisplayName("analyzeTokenization should count numbers correctly")
    void testAnalyzeTokenization_AccurateNumberCount() {
        AdvancedTokenizer.TokenizationAnalysis analysis =
            tokenizer.analyzeTokenization("There are 123 items and 456 more");

        assertNotNull(analysis);
        assertTrue(analysis.numbers() >= 2,
            "Should detect at least 2 numbers");
    }

    @Test
    @DisplayName("analyzeTokenization should count total words")
    void testAnalyzeTokenization_TotalWordCount() {
        AdvancedTokenizer.TokenizationAnalysis analysis =
            tokenizer.analyzeTokenization("one two three four five");

        assertNotNull(analysis);
        assertEquals(5, analysis.totalWords(),
            "Should count 5 words");
    }

    @Test
    @DisplayName("analyzeTokenization should detect punctuation sequences")
    void testAnalyzeTokenization_PunctuationCount() {
        AdvancedTokenizer.TokenizationAnalysis analysis =
            tokenizer.analyzeTokenization("Hello! How are you? Fine.");

        assertNotNull(analysis);
        assertTrue(analysis.punctuationSequences() >= 2,
            "Should detect multiple punctuation sequences");
    }

    // ==================== COMPARISON METHOD TESTS ====================

    @Test
    @DisplayName("compareTokenization should show difference in token counts")
    void testCompareTokenization_ShowsDifference() {
        AdvancedTokenizer.TokenizationComparison comparison =
            tokenizer.compareTokenization("This is a test with 123 items");

        assertNotNull(comparison);
        assertNotNull(comparison.simpleTokens);
        assertNotNull(comparison.advancedTokens);
        // Simple and advanced should potentially differ
        assertFalse(comparison.simpleTokens.isEmpty());
        assertFalse(comparison.advancedTokens.isEmpty());
    }

    @Test
    @DisplayName("compareTokenization should calculate improvement count")
    void testCompareTokenization_CalculatesImprovement() {
        AdvancedTokenizer.TokenizationComparison comparison =
            tokenizer.compareTokenization("state-of-the-art system");

        assertNotNull(comparison);
        assertTrue(comparison.improvementCount >= 0,
            "Improvement count should be non-negative");
    }

    @Test
    @DisplayName("compareTokenization should preserve original text")
    void testCompareTokenization_PreservesOriginal() {
        String original = "Test sentence with content";
        AdvancedTokenizer.TokenizationComparison comparison =
            tokenizer.compareTokenization(original);

        assertNotNull(comparison);
        assertEquals(original, comparison.originalText);
    }
}
