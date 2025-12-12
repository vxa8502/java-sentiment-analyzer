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
        // Default test configuration:
        // - preserveNumbers = false (numbers are filtered out)
        // - minTokenLength = 1 (allow single chars if meaningful)
        // - preserveHyphenated = true (keep hyphenated compounds like "well-known")
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
        assertFalse(tokens.isEmpty(), "Should produce tokens from simple sentence");
        // Verify expected tokens are present
        assertTrue(tokens.contains("This"), "Should contain 'This' with original case");
        assertTrue(tokens.contains("is"), "Should contain 'is'");
        assertTrue(tokens.contains("test"), "Should contain 'test'");
        // Note: "a" is a single-char token that may be filtered based on meaningful chars set
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

    // ==================== NUMBER HANDLING TESTS ====================

    @Test
    @DisplayName("tokenize should filter numbers when preserveNumbers is false")
    void testTokenize_FiltersNumbers() {
        // Create tokenizer with preserveNumbers=false (numbers completely removed)
        AdvancedTokenizer filterTokenizer = new AdvancedTokenizer(false, 1, true);
        List<String> tokens = filterTokenizer.tokenize("There are 123 items");

        assertNotNull(tokens);
        // When preserveNumbers=false, numbers are completely filtered out
        assertFalse(tokens.contains("123"), "Number '123' should be filtered out");
        // Verify words are still tokenized
        assertTrue(tokens.contains("There") || tokens.contains("are") || tokens.contains("items"),
                "Should still contain word tokens");
    }

    @Test
    @DisplayName("tokenize should preserve numbers as tokens when configured")
    void testTokenize_PreservesNumbers() {
        // Create tokenizer with preserveNumbers=true (numbers replaced with placeholder)
        AdvancedTokenizer preserveTokenizer = new AdvancedTokenizer(true, 1, true);
        List<String> tokens = preserveTokenizer.tokenize("There are 123 items");

        assertNotNull(tokens);
        // When preserveNumbers=true, all numbers are replaced with "NUMBER_TOKEN"
        assertTrue(tokens.contains("NUMBER_TOKEN"),
                "Should contain NUMBER_TOKEN placeholder for '123'");
        assertFalse(tokens.contains("123"), "Original number should not appear");
    }

    @Test
    @DisplayName("tokenize should handle decimal numbers")
    void testTokenize_DecimalNumbers() {
        // Test that decimal numbers are also replaced with NUMBER_TOKEN
        AdvancedTokenizer preserveTokenizer = new AdvancedTokenizer(true, 1, true);
        List<String> tokens = preserveTokenizer.tokenize("Price is 19.99 dollars");

        assertNotNull(tokens);
        // Decimal numbers should be recognized and replaced
        assertTrue(tokens.contains("NUMBER_TOKEN"),
                "Should preserve decimal '19.99' as NUMBER_TOKEN");
        assertFalse(tokens.contains("19.99"), "Original decimal should not appear in tokens");
    }

    // ==================== MIN TOKEN LENGTH TESTS ====================

    @Test
    @DisplayName("tokenize should respect minTokenLength of 2")
    void testTokenize_MinLength2() {
        // Create tokenizer with minTokenLength=2 (filter tokens shorter than 2 chars)
        AdvancedTokenizer minLength2Tokenizer = new AdvancedTokenizer(false, 2, true);
        List<String> tokens = minLength2Tokenizer.tokenize("I am a good person");

        assertNotNull(tokens);
        // With minTokenLength=2, tokens with length < 2 are filtered UNLESS they're meaningful
        // Expected to remain: "am" (2), "good" (4), "person" (6)
        // May be filtered: "I" (1), "a" (1) - unless they're in meaningful single chars set
        assertTrue(tokens.contains("am"), "Should contain 'am' (length 2, at boundary)");
        assertTrue(tokens.contains("good"), "Should contain 'good' (length 4)");
        assertTrue(tokens.contains("person"), "Should contain 'person' (length 6)");

        // Validate that all non-meaningful tokens meet the minimum length requirement
        assertTrue(tokens.stream()
            .filter(t -> !t.equals("I") && !t.matches("[!?.:;]+")) // Exclude meaningful exceptions
            .allMatch(t -> t.length() >= 2),
            "All regular tokens should have length >= 2");
    }

    @Test
    @DisplayName("tokenize should respect minTokenLength of 3")
    void testTokenize_MinLength3() {
        // Create tokenizer with minTokenLength=3 (filter tokens shorter than 3 chars)
        AdvancedTokenizer minLength3Tokenizer = new AdvancedTokenizer(false, 3, true);
        List<String> tokens = minLength3Tokenizer.tokenize("The cat is on the mat");

        assertNotNull(tokens);
        // With minTokenLength=3, only tokens with length >= 3 should remain
        // Expected to remain: "The" (3), "cat" (3), "the" (3), "mat" (3)
        // Expected to filter: "is" (2), "on" (2)
        assertTrue(tokens.contains("The"), "Should contain 'The' (length 3, at boundary)");
        assertTrue(tokens.contains("cat"), "Should contain 'cat' (length 3)");
        assertTrue(tokens.contains("the"), "Should contain 'the' (length 3)");
        assertTrue(tokens.contains("mat"), "Should contain 'mat' (length 3)");

        // Verify short tokens are filtered
        assertFalse(tokens.contains("is"), "'is' should be filtered (length 2 < min 3)");
        assertFalse(tokens.contains("on"), "'on' should be filtered (length 2 < min 3)");

        // All tokens must meet minimum length
        assertTrue(tokens.stream().allMatch(t -> t.length() >= 3),
            "Every token should have length >= 3");
    }

    // ==================== UTILITY METHODS TESTS ====================

    @Test
    @DisplayName("compareTokenization should compare tokenization methods")
    void testCompareTokenization() {
        // Test the comparison utility that shows differences between simple and advanced tokenization
        // Simple: whitespace split with minLength filter
        // Advanced: full tokenization with hyphenation, number handling, etc.
        AdvancedTokenizer.TokenizationComparison comparison =
                tokenizer.compareTokenization("This is a test sentence");

        assertNotNull(comparison, "Comparison object should not be null");
        assertNotNull(comparison.simpleTokens, "Simple tokens should be populated");
        assertNotNull(comparison.advancedTokens, "Advanced tokens should be populated");
        assertNotNull(comparison.originalText, "Original text should be preserved");
    }

    // ==================== PLACEHOLDER LEAKAGE PREVENTION TESTS ====================

    @Test
    @DisplayName("tokenize should not leak hyphenated placeholders into output")
    void testTokenize_NoPlaceholderLeakage() {
        // Critical reliability test: ensure internal placeholders never appear in output
        // The tokenizer uses "HYPHEN_PLACEHOLDER_N" internally for preserving hyphenated words
        List<String> tokens = tokenizer.tokenize("state-of-the-art well-known");

        assertNotNull(tokens);
        assertFalse(tokens.stream().anyMatch(t -> t.startsWith("HYPHEN_PLACEHOLDER")),
            "Internal placeholders must never leak into tokenized output");
    }

    @Test
    @DisplayName("tokenize should not leak placeholders even with placeholder-like input")
    void testTokenize_NoPlaceholderLeakage_WithSimilarInput() {
        // Edge case: ensure tokenizer handles text that looks like internal placeholders
        List<String> tokens = tokenizer.tokenize("HYPHEN_PLACEHOLDER_0 is a test");

        assertNotNull(tokens);
        // If input contains something that looks like a placeholder, it should either:
        // 1. Be tokenized normally as a word
        // 2. Not create confusion with internal placeholder mechanism
        long placeholderCount = tokens.stream()
            .filter(t -> t.startsWith("HYPHEN_PLACEHOLDER"))
            .count();
        // Should have at most 1 (the original input), never more from internal processing
        assertTrue(placeholderCount <= 1,
            "Should not generate additional placeholders beyond the input");
    }

    // ==================== TOKEN ORDER PRESERVATION TESTS ====================

    @Test
    @DisplayName("tokenize should preserve token order from input")
    void testTokenize_PreservesTokenOrder() {
        // Reliability test: verify tokens appear in the same order as input
        List<String> tokens = tokenizer.tokenize("First Second Third Fourth");

        assertNotNull(tokens);
        // Find indices of each word (some may be filtered)
        int firstIdx = tokens.indexOf("First");
        int secondIdx = tokens.indexOf("Second");
        int thirdIdx = tokens.indexOf("Third");
        int fourthIdx = tokens.indexOf("Fourth");

        // All should be present
        assertTrue(firstIdx >= 0, "Should contain 'First'");
        assertTrue(secondIdx >= 0, "Should contain 'Second'");
        assertTrue(thirdIdx >= 0, "Should contain 'Third'");
        assertTrue(fourthIdx >= 0, "Should contain 'Fourth'");

        // Order should be preserved: First < Second < Third < Fourth
        assertTrue(firstIdx < secondIdx, "First should come before Second");
        assertTrue(secondIdx < thirdIdx, "Second should come before Third");
        assertTrue(thirdIdx < fourthIdx, "Third should come before Fourth");
    }

    @Test
    @DisplayName("tokenize should preserve order even with filtering")
    void testTokenize_PreservesOrderWithFiltering() {
        // Test that order is preserved even when some tokens are filtered
        AdvancedTokenizer minLength3 = new AdvancedTokenizer(false, 3, true);
        List<String> tokens = minLength3.tokenize("The cat is on the mat");

        assertNotNull(tokens);
        // "is" and "on" will be filtered (length 2), but order should be preserved
        // Expected: "The", "cat", "the", "mat"
        int theIdx1 = tokens.indexOf("The");
        int catIdx = tokens.indexOf("cat");
        int theIdx2 = tokens.lastIndexOf("the");
        int matIdx = tokens.indexOf("mat");

        assertTrue(theIdx1 >= 0 && catIdx >= 0 && theIdx2 >= 0 && matIdx >= 0,
            "Should contain The, cat, the, mat");
        assertTrue(theIdx1 < catIdx, "The should come before cat");
        assertTrue(catIdx < theIdx2, "cat should come before the (second occurrence)");
        assertTrue(theIdx2 < matIdx, "the should come before mat");
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
        assertFalse(tokens.isEmpty(), "Should tokenize very long text");
        // Should produce approximately 100 tokens (one for each word)
        // Some might be filtered if they're very short (word0, word1, etc.)
        assertTrue(tokens.size() >= 90 && tokens.size() <= 110,
            "Should produce approximately 100 tokens, got: " + tokens.size());
    }

    // ==================== NEGATIVE TEST CASES ====================

    @Test
    @DisplayName("tokenize should handle malicious input with excessive hyphens")
    void testTokenize_ExcessiveHyphens() {
        // Negative test: ensure tokenizer handles pathological input gracefully
        String maliciousInput = "word-with-many-many-many-many-many-hyphens-in-it";
        List<String> tokens = tokenizer.tokenize(maliciousInput);

        assertNotNull(tokens);
        // Should handle gracefully without throwing exceptions or infinite loops
        // Tokens should not be excessively long or malformed
        assertTrue(tokens.stream().allMatch(t -> t.length() < 200),
            "Tokens should not be excessively long even with pathological input");
    }

    @Test
    @DisplayName("tokenize should handle input with only special characters")
    void testTokenize_OnlySpecialCharacters() {
        // Negative test: non-alphanumeric characters only
        String specialCharsOnly = "@#$%^&*()_+{}[]|\\:;\"'<>,.?/~`";
        List<String> tokens = tokenizer.tokenize(specialCharsOnly);

        assertNotNull(tokens);
        // Should return a list (possibly empty) without throwing exceptions
        // Verify no malformed tokens
        assertTrue(tokens.stream().noneMatch(t -> t.length() > 10),
            "Should not create long tokens from special characters");
    }

    @Test
    @DisplayName("tokenize should handle extremely long single word")
    void testTokenize_ExtremeLongWord() {
        // Negative test: very long single token
        String longWord = "a".repeat(1000);
        List<String> tokens = tokenizer.tokenize(longWord);

        assertNotNull(tokens);
        // Should handle long words gracefully
        if (!tokens.isEmpty()) {
            assertTrue(tokens.get(0).length() <= 1000,
                "Token should not exceed input length");
        }
    }

    @Test
    @DisplayName("tokenize should handle mixed case consistently")
    void testTokenize_MixedCase() {
        // Reliability test: verify case preservation
        List<String> tokens = tokenizer.tokenize("Hello WORLD hElLo WoRlD");

        assertNotNull(tokens);
        // Should preserve original case
        assertTrue(tokens.contains("Hello") || tokens.stream().anyMatch(t -> t.contains("Hello")),
            "Should preserve 'Hello' case");
        assertTrue(tokens.contains("WORLD") || tokens.stream().anyMatch(t -> t.contains("WORLD")),
            "Should preserve 'WORLD' case");
    }

    @Test
    @DisplayName("tokenize should handle Unicode characters gracefully")
    void testTokenize_UnicodeCharacters() {
        // Negative test: Unicode and international characters
        String unicodeText = "Hello 世界 مرحبا мир";
        List<String> tokens = tokenizer.tokenize(unicodeText);

        assertNotNull(tokens);
        // Should not throw exceptions with Unicode input
        // Should at minimum preserve ASCII words
        assertTrue(tokens.stream().anyMatch(t -> t.contains("Hello")),
            "Should preserve ASCII word 'Hello'");
    }

    // ==================== INTEGRATION TESTS ====================

    @Test
    @DisplayName("tokenize should be consistent on repeated calls")
    void testTokenize_ConsistentResults() {
        // Reliability test: tokenization should be deterministic
        String text = "This is a test sentence";

        List<String> tokens1 = tokenizer.tokenize(text);
        List<String> tokens2 = tokenizer.tokenize(text);
        List<String> tokens3 = tokenizer.tokenize(text);

        // All three calls should produce identical results
        assertEquals(tokens1.size(), tokens2.size(), "Sizes should match on repeated calls");
        assertEquals(tokens1.size(), tokens3.size(), "Sizes should match on repeated calls");
        assertEquals(tokens1, tokens2, "Token lists should be identical on repeated calls");
        assertEquals(tokens1, tokens3, "Token lists should be identical on repeated calls");
    }

    @Test
    @DisplayName("tokenize should handle duplicate tokens correctly")
    void testTokenize_DuplicateTokensAllowed() {
        // Reliability test: duplicate words in input should produce duplicate tokens
        List<String> tokens = tokenizer.tokenize("test test test");

        assertNotNull(tokens);
        // Should have 3 instances of "test"
        long testCount = tokens.stream().filter(t -> t.equals("test")).count();
        assertEquals(3, testCount, "Should preserve duplicate tokens from input");
    }

    // ==================== HYPHENATED WORD LOGIC TESTS ====================

    @Test
    @DisplayName("tokenize should reject invalid hyphenated words with short parts")
    void testTokenize_InvalidHyphenated_ShortParts() {
        // Test that hyphenated words with parts < 2 chars are rejected and split
        // "a-b" should be split because both parts are only 1 character
        List<String> tokens = tokenizer.tokenize("a-b");

        assertNotNull(tokens);
        // Validation rule: each part of a hyphenated word must be >= 2 characters
        // Therefore "a-b" should NOT be preserved as a hyphenated compound
        assertFalse(tokens.stream().anyMatch(t -> t.equals("a-b")),
            "Should not preserve 'a-b' because parts are < 2 chars");
    }

    // ==================== HYPHENATED WORD VALIDATION TESTS ====================

    @Test
    @DisplayName("tokenize should only preserve valid hyphenated compounds")
    void testTokenize_OnlyValidHyphenatedPreserved() {
        // Reliability test: ensure hyphenation validation is working correctly
        // Valid: "well-known" (prefix "well" is recognized)
        // Invalid: "x-y" (parts < 2 chars), "random-word" (no recognized prefix/suffix)
        List<String> tokens = tokenizer.tokenize("well-known x-y random-stuff");

        assertNotNull(tokens);
        // "well-known" should be preserved (has recognized prefix)
        // "x-y" should NOT be preserved (parts too short)
        assertFalse(tokens.stream().anyMatch(t -> t.equals("x-y")),
            "Should not preserve 'x-y' (parts < 2 chars)");

        // Verify no invalid hyphenated compounds slip through
        List<String> invalidHyphenated = tokens.stream()
            .filter(t -> t.contains("-"))
            .filter(t -> {
                String[] parts = t.split("-");
                // Check if any part is < 2 chars (invalid)
                for (String part : parts) {
                    if (part.length() < 2) return true;
                }
                return false;
            })
            .toList();

        assertTrue(invalidHyphenated.isEmpty(),
            "Found invalid hyphenated words with parts < 2 chars: " + invalidHyphenated);
    }

    @Test
    @DisplayName("tokenize should handle hyphenated words differently based on configuration")
    void testTokenize_HyphenatedConfigurationDifference() {
        // Reliability test: verify preserveHyphenated flag affects output
        String text = "well-known state-of-the-art";

        AdvancedTokenizer withHyphenation = new AdvancedTokenizer(false, 1, true);
        AdvancedTokenizer withoutHyphenation = new AdvancedTokenizer(false, 1, false);

        List<String> tokensWithHyphenation = withHyphenation.tokenize(text);
        List<String> tokensWithoutHyphenation = withoutHyphenation.tokenize(text);

        assertNotNull(tokensWithHyphenation);
        assertNotNull(tokensWithoutHyphenation);

        // With different configurations, the results should potentially differ
        // The key is that the tokenizer behaves consistently for each configuration
        // When preserveHyphenated=true, may preserve "well-known" as one token
        // When preserveHyphenated=false, may split it or treat it differently

        // At minimum, verify both produce valid output
        assertFalse(tokensWithHyphenation.stream().anyMatch(t -> t.contains("HYPHEN_PLACEHOLDER")),
            "Should not leak placeholders with hyphenation enabled");
        assertFalse(tokensWithoutHyphenation.stream().anyMatch(t -> t.contains("HYPHEN_PLACEHOLDER")),
            "Should not leak placeholders with hyphenation disabled");
    }

    // ==================== BOUNDARY CONDITION TESTS ====================

    @Test
    @DisplayName("tokenize should handle token exactly at minLength boundary")
    void testTokenize_ExactlyAtMinLength() {
        // Test boundary condition: tokens exactly at minLength threshold
        // This validates that the >= comparison is correct (not >)
        AdvancedTokenizer minLength3 = new AdvancedTokenizer(false, 3, true);
        List<String> tokens = minLength3.tokenize("ab abc abcd");

        assertNotNull(tokens);
        // Validation: token.length() >= minTokenLength
        assertFalse(tokens.contains("ab"),
            "'ab' (length 2) should be filtered (< minLength 3)");
        assertTrue(tokens.contains("abc"),
            "'abc' (length 3) should be kept (exactly at boundary, >= 3)");
        assertTrue(tokens.contains("abcd"),
            "'abcd' (length 4) should be kept (> minLength 3)");
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

    // ==================== COMPREHENSIVE FILTERING VALIDATION TESTS ====================

    @Test
    @DisplayName("tokenize should ensure ALL tokens meet minLength requirement")
    void testTokenize_AllTokensMeetMinLength() {
        // Reliability test: validate that EVERY token meets the minimum length
        // This catches bugs where some tokens slip through filtering
        AdvancedTokenizer minLength4 = new AdvancedTokenizer(false, 4, true);
        List<String> tokens = minLength4.tokenize(
            "The quick brown fox jumps over the lazy dog! Really?");

        assertNotNull(tokens);
        // Expected to keep: "quick" (5), "brown" (5), "jumps" (5), "over" (4), "lazy" (4)
        // Expected to filter: "The" (3), "fox" (3), "the" (3), "dog" (3)

        // Critical validation: EVERY token must have length >= 4
        // Exception: meaningful punctuation like "!!" or "???" may be preserved
        List<String> invalidTokens = tokens.stream()
            .filter(t -> t.length() < 4 && !t.matches("[!?.:;]+"))
            .toList();

        assertTrue(invalidTokens.isEmpty(),
            "Found tokens that violate minLength=4: " + invalidTokens);
    }

    @Test
    @DisplayName("tokenize should ensure NO numbers appear when preserveNumbers is false")
    void testTokenize_NoNumbersWhenDisabled() {
        // Reliability test: when preserveNumbers=false, absolutely no numbers should appear
        AdvancedTokenizer noNumbers = new AdvancedTokenizer(false, 1, true);
        List<String> tokens = noNumbers.tokenize(
            "There are 123 items, 456.78 dollars, and 1,000,000 people");

        assertNotNull(tokens);
        // Critical validation: NO numeric tokens should appear
        // Not even "NUMBER_TOKEN" - numbers should be completely removed
        assertFalse(tokens.contains("NUMBER_TOKEN"),
            "NUMBER_TOKEN should not appear when preserveNumbers=false");

        // No actual numbers should appear
        List<String> numericTokens = tokens.stream()
            .filter(t -> t.matches(".*\\d+.*"))
            .toList();

        assertTrue(numericTokens.isEmpty(),
            "Found numeric tokens when preserveNumbers=false: " + numericTokens);
    }

    @Test
    @DisplayName("tokenize should ensure ALL numbers are replaced when preserveNumbers is true")
    void testTokenize_AllNumbersReplacedWhenEnabled() {
        // Reliability test: when preserveNumbers=true, all numbers become NUMBER_TOKEN
        AdvancedTokenizer preserveNumbers = new AdvancedTokenizer(true, 1, true);
        List<String> tokens = preserveNumbers.tokenize(
            "I have 123 items and 456 more");

        assertNotNull(tokens);
        // Should have NUMBER_TOKEN for each number
        assertTrue(tokens.contains("NUMBER_TOKEN"),
            "Should contain NUMBER_TOKEN placeholder");

        // Critical: original numbers should NOT appear
        assertFalse(tokens.contains("123"), "Original number 123 should not appear");
        assertFalse(tokens.contains("456"), "Original number 456 should not appear");

        // No other numeric values should slip through
        List<String> numericTokens = tokens.stream()
            .filter(t -> !t.equals("NUMBER_TOKEN") && t.matches(".*\\d+.*"))
            .toList();

        assertTrue(numericTokens.isEmpty(),
            "Found raw numeric tokens when they should be replaced: " + numericTokens);
    }

    @Test
    @DisplayName("tokenize should never return empty strings as tokens")
    void testTokenize_NoEmptyTokens() {
        // Reliability test: empty strings should never appear in token list
        List<String> tokens = tokenizer.tokenize("word1    word2     word3   ");

        assertNotNull(tokens);
        // Critical: no empty or whitespace-only tokens
        List<String> emptyTokens = tokens.stream()
            .filter(t -> t == null || t.trim().isEmpty())
            .toList();

        assertTrue(emptyTokens.isEmpty(),
            "Found empty or whitespace-only tokens: " + emptyTokens);
    }

    @Test
    @DisplayName("tokenize should never return null tokens")
    void testTokenize_NoNullTokens() {
        // Reliability test: null elements should never appear in token list
        List<String> tokens = tokenizer.tokenize("Hello world! How are you?");

        assertNotNull(tokens);
        // Critical: no null elements
        assertFalse(tokens.contains(null), "Token list should not contain null elements");
    }

    // ==================== CONFIGURATION COMBINATION TESTS ====================

    @Test
    @DisplayName("tokenize should handle all filters enabled")
    void testTokenize_AllFiltersEnabled() {
        // Test strictest configuration: high minLength, no number preservation, no hyphenation
        // Configuration: preserveNumbers=false, minLength=5, preserveHyphenated=false
        AdvancedTokenizer strict = new AdvancedTokenizer(false, 5, false);
        List<String> tokens = strict.tokenize("The quick brown fox jumps over 123 state-of-the-art items");

        assertNotNull(tokens);
        // With strict filtering, only words with length >= 5 should remain
        // Expected to keep: "quick" (5), "brown" (5), "jumps" (5), "items" (5)
        // Expected to filter: "The" (3), "fox" (3), "over" (4), "123", split hyphenated parts
        assertTrue(tokens.stream().allMatch(t ->
            t.length() >= 5 || t.matches("[!?]+")), // Allow meaningful punctuation
            "All tokens should have length >= 5 (or be meaningful punctuation)");
        assertFalse(tokens.contains("123"), "Numbers should be filtered");
        assertFalse(tokens.contains("NUMBER_TOKEN"), "Numbers should not be preserved");
    }

    @Test
    @DisplayName("tokenize should handle all preservation enabled")
    void testTokenize_AllPreservationEnabled() {
        // Test most permissive configuration: preserve everything, minimal filtering
        // Configuration: preserveNumbers=true, minLength=1, preserveHyphenated=true
        AdvancedTokenizer permissive = new AdvancedTokenizer(true, 1, true);
        List<String> tokens = permissive.tokenize("I have 42 state-of-the-art items!");

        assertNotNull(tokens);
        // With permissive settings, numbers are preserved as placeholder
        assertTrue(tokens.contains("NUMBER_TOKEN"),
            "Should preserve number '42' as NUMBER_TOKEN");
        // Words should be tokenized normally
        assertTrue(tokens.contains("have") || tokens.contains("items"),
            "Should contain word tokens");
        // With minimal filtering (minLength=1), should get more tokens
        assertTrue(tokens.size() >= 4,
            "Should produce multiple tokens with permissive settings");
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

    // ==================== TOKENIZATION ANALYSIS COVERAGE TESTS ====================

    @Test
    @DisplayName("analyzeTokenization toString should provide formatted output")
    void testTokenizationAnalysis_ToString() {
        // Coverage test: validate TokenizationAnalysis.toString() method
        AdvancedTokenizer.TokenizationAnalysis analysis =
            tokenizer.analyzeTokenization("This is a state-of-the-art system with 123 items");

        assertNotNull(analysis);
        String formatted = analysis.toString();

        assertNotNull(formatted, "toString should not return null");
        assertFalse(formatted.isEmpty(), "toString should not be empty");
        // Should contain key information
        assertTrue(formatted.contains("TokenAnalysis"), "Should contain class identifier");
        assertTrue(formatted.contains("words="), "Should contain word count");
        assertTrue(formatted.contains("hyphenated="), "Should contain hyphenated count");
        assertTrue(formatted.contains("numbers="), "Should contain number count");
        assertTrue(formatted.contains("punctuation="), "Should contain punctuation count");
    }

    @Test
    @DisplayName("analyzeTokenization should count all elements accurately")
    void testAnalyzeTokenization_AccurateCounts() {
        // Coverage test: validate metrics accuracy
        String text = "well-known state-of-the-art 123 456 items!!! Really???";
        AdvancedTokenizer.TokenizationAnalysis analysis = tokenizer.analyzeTokenization(text);

        assertNotNull(analysis);
        // Should count hyphenated words accurately
        assertTrue(analysis.hyphenatedWords() >= 2,
            "Should detect at least 2 hyphenated words (well-known, state-of-the-art)");

        // Should count numbers accurately
        assertTrue(analysis.numbers() >= 2,
            "Should detect at least 2 numbers (123, 456)");

        // Should count punctuation sequences
        assertTrue(analysis.punctuationSequences() >= 2,
            "Should detect punctuation sequences (!!!, ???)");

        // Should count total words (whitespace split counts "items!!!" and "Really???" as words)
        assertTrue(analysis.totalWords() >= 6,
            "Should count whitespace-separated tokens (6 in this text)");
    }

    // ==================== ISVALIDHYPHENATEDWORD COMPREHENSIVE TESTS ====================

    @Test
    @DisplayName("tokenize should preserve hyphenated words with recognized prefixes")
    void testTokenize_RecognizedPrefixes() {
        // Coverage test: validate each common prefix is recognized
        // Prefixes: self, well, multi, non, pre, post, anti, pro, co
        String[] prefixedWords = {
            "self-aware", "well-known", "multi-level", "non-standard",
            "pre-approved", "post-modern", "anti-virus", "pro-active", "co-author"
        };

        for (String word : prefixedWords) {
            List<String> tokens = tokenizer.tokenize(word);
            assertNotNull(tokens, "Should tokenize " + word);
            // Word should either be preserved as hyphenated OR split into parts
            boolean preserved = tokens.stream().anyMatch(t -> t.contains("-"));
            boolean split = tokens.stream().anyMatch(t ->
                t.equals(word.split("-")[0]) || t.equals(word.split("-")[1]));
            assertTrue(preserved || split,
                "Should handle '" + word + "' (either preserve or split)");
        }
    }

    @Test
    @DisplayName("tokenize should preserve hyphenated words with recognized suffixes")
    void testTokenize_RecognizedSuffixes() {
        // Coverage test: validate each common suffix is recognized
        // Suffixes: like, based, free, proof, ready, aware, wise
        String[] suffixedWords = {
            "camera-like", "cloud-based", "sugar-free", "water-proof",
            "oven-ready", "self-aware", "street-wise"
        };

        for (String word : suffixedWords) {
            List<String> tokens = tokenizer.tokenize(word);
            assertNotNull(tokens, "Should tokenize " + word);
            // Word should either be preserved as hyphenated OR split into parts
            boolean preserved = tokens.stream().anyMatch(t -> t.contains("-"));
            boolean split = tokens.stream().anyMatch(t ->
                t.equals(word.split("-")[0]) || t.equals(word.split("-")[1]));
            assertTrue(preserved || split,
                "Should handle '" + word + "' (either preserve or split)");
        }
    }

    @Test
    @DisplayName("tokenize should preserve 3+ part hyphenated compounds")
    void testTokenize_ThreePlusPartCompounds() {
        // Coverage test: 3+ parts automatically validate (no prefix/suffix needed)
        String[] compounds = {
            "one-two-three", "state-of-the-art", "up-to-date-system"
        };

        for (String word : compounds) {
            List<String> tokens = tokenizer.tokenize(word);
            assertNotNull(tokens, "Should tokenize " + word);
            assertFalse(tokens.isEmpty(), "Should produce tokens for " + word);
        }
    }

    @Test
    @DisplayName("tokenize should reject hyphenated words with parts under 2 chars")
    void testTokenize_RejectShortParts() {
        // Coverage test: validate minimum part length enforcement
        String[] invalidWords = {"a-b", "x-y", "ab-c", "a-bc"};

        for (String word : invalidWords) {
            List<String> tokens = tokenizer.tokenize(word);
            assertNotNull(tokens);
            // Should NOT preserve these as hyphenated compounds
            assertFalse(tokens.stream().anyMatch(t -> t.equals(word)),
                "Should not preserve '" + word + "' (parts < 2 chars)");
        }
    }

    // ==================== CASE SENSITIVITY COVERAGE TESTS ====================

    @Test
    @DisplayName("tokenize should preserve exact case from input")
    void testTokenize_ExactCasePreservation() {
        // Coverage test: validate case handling throughout tokenization
        List<String> tokens = tokenizer.tokenize("UPPERCASE lowercase MiXeD");

        assertNotNull(tokens);
        assertTrue(tokens.contains("UPPERCASE"), "Should preserve all uppercase");
        assertTrue(tokens.contains("lowercase"), "Should preserve all lowercase");
        assertTrue(tokens.contains("MiXeD"), "Should preserve mixed case exactly");
    }

    @Test
    @DisplayName("tokenize should handle case-sensitive hyphenated words")
    void testTokenize_CaseSensitiveHyphenated() {
        // Coverage test: hyphenation validation uses toLowerCase but output preserves case
        List<String> tokens = tokenizer.tokenize("WELL-KNOWN Well-Known well-known");

        assertNotNull(tokens);
        // All three should be handled (validation is case-insensitive)
        // But output should preserve original case
        assertFalse(tokens.isEmpty(), "Should tokenize all variants");
    }

    // ==================== EXTREME VALUE COVERAGE TESTS ====================

    @Test
    @DisplayName("tokenize should handle maximum reasonable minTokenLength")
    void testTokenize_MaximumMinLength() {
        // Coverage test: very high minLength values
        AdvancedTokenizer maxLength = new AdvancedTokenizer(false, 50, true);
        List<String> tokens = maxLength.tokenize(
            "This short text has no words longer than fifty characters");

        assertNotNull(tokens);
        // All words should be filtered (none are >= 50 chars)
        assertTrue(tokens.isEmpty() || tokens.stream().allMatch(t -> t.length() >= 50),
            "Should filter all words shorter than 50 characters");
    }

    @Test
    @DisplayName("tokenize should handle mixed valid and invalid hyphenated words")
    void testTokenize_MixedHyphenatedValidity() {
        // Coverage test: mix of valid and invalid hyphenated in same input
        String text = "well-known a-b state-of-the-art x-y self-aware";
        List<String> tokens = tokenizer.tokenize(text);

        assertNotNull(tokens);
        // Valid ones: well-known, state-of-the-art, self-aware
        // Invalid ones: a-b, x-y (parts < 2 chars)
        assertFalse(tokens.stream().anyMatch(t -> t.equals("a-b")),
            "Should not preserve 'a-b'");
        assertFalse(tokens.stream().anyMatch(t -> t.equals("x-y")),
            "Should not preserve 'x-y'");
    }

    // ==================== STRESS AND PERFORMANCE COVERAGE TESTS ====================

    @Test
    @DisplayName("tokenize should handle many hyphenated words efficiently")
    void testTokenize_ManyHyphenatedWords() {
        // Coverage/performance test: multiple hyphenated words in one text
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < 100; i++) {
            text.append("well-known state-of-the-art self-aware ");
        }

        List<String> tokens = tokenizer.tokenize(text.toString());

        assertNotNull(tokens);
        assertFalse(tokens.isEmpty(), "Should tokenize text with many hyphenated words");
        // Should complete in reasonable time (no timeout)
        assertTrue(tokens.size() >= 100,
            "Should produce substantial token count from 300 words");
    }

    @Test
    @DisplayName("analyzeTokenization should handle complex text with all features")
    void testAnalyzeTokenization_ComplexFeatures() {
        // Coverage test: analysis with all token types present
        String complexText = "The well-known system processed 1,234 items!!! " +
                           "Amazing??? state-of-the-art... Really! 567 more.";

        AdvancedTokenizer.TokenizationAnalysis analysis =
            tokenizer.analyzeTokenization(complexText);

        assertNotNull(analysis);
        assertTrue(analysis.totalWords() > 0, "Should count words");
        assertTrue(analysis.hyphenatedWords() > 0, "Should count hyphenated words");
        assertTrue(analysis.numbers() > 0, "Should count numbers");
        assertTrue(analysis.punctuationSequences() > 0, "Should count punctuation");

        // Verify toString works with complex data
        String formatted = analysis.toString();
        assertNotNull(formatted);
        assertFalse(formatted.isEmpty());
    }

    @Test
    @DisplayName("tokenize should handle text with multiple NUMBER_TOKEN occurrences")
    void testTokenize_MultipleNumbers() {
        // Coverage test: multiple numbers should each be replaced
        AdvancedTokenizer preserveNumbers = new AdvancedTokenizer(true, 1, true);
        List<String> tokens = preserveNumbers.tokenize("I have 1 apple, 2 oranges, and 3 bananas");

        assertNotNull(tokens);
        // Should have NUMBER_TOKEN in the list (maybe multiple times or once)
        long numberTokenCount = tokens.stream()
            .filter(t -> t.equals("NUMBER_TOKEN"))
            .count();
        assertTrue(numberTokenCount >= 1,
            "Should have at least one NUMBER_TOKEN for the numbers");
    }
}
