package sentiment.preprocessing;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import sentiment.config.TestFeatureConfig;
import sentiment.data.Dataset;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * Comprehensive unit tests for TextPreprocessor.
 *
 * Tests cover:
 * - Text cleaning (URLs, emails, mentions, emoticons)
 * - Tokenization
 * - Stopword removal
 * - Fit/transform workflow
 * - Thread safety
 * - State management
 */
@DisplayName("TextPreprocessor Unit Tests")
class TextPreprocessorTest {

    @Mock
    private ContractionExpander mockContractionExpander;

    @Mock
    private AdvancedTokenizer mockTokenizer;

    @Mock
    private IntelligentStopwordRemover mockStopwordRemover;

    private TextPreprocessor preprocessor;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);

        // Setup mock behaviors
        when(mockContractionExpander.expand(anyString())).thenAnswer(invocation -> {
            String input = invocation.getArgument(0);
            return input.replace("don't", "do not")
                       .replace("can't", "cannot")
                       .replace("won't", "will not");
        });

        when(mockTokenizer.tokenize(anyString())).thenAnswer(invocation -> {
            String input = invocation.getArgument(0);
            return List.of(input.split("\\s+"));
        });

        when(mockStopwordRemover.removeStopwords(any())).thenAnswer(invocation -> {
            List<String> tokens = invocation.getArgument(0);
            // Remove common stopwords for testing
            List<String> stopwords = List.of("the", "a", "an", "is", "are", "was", "were");
            return tokens.stream()
                        .filter(token -> !stopwords.contains(token.toLowerCase()))
                        .toList();
        });

        // Create preprocessor with mocked dependencies
        preprocessor = new TextPreprocessor(
            mockContractionExpander,
            mockTokenizer,
            mockStopwordRemover,
            TestFeatureConfig.createDefault()
        );
    }

    // ==================== CONSTRUCTOR TESTS ====================

    @Test
    @DisplayName("Constructor should throw when minWordLength is invalid")
    void testConstructor_InvalidMinWordLength_ThrowsException() {
        assertThrows(IllegalArgumentException.class,
            () -> new TextPreprocessor(
                mockContractionExpander,
                mockTokenizer,
                mockStopwordRemover,
                TestFeatureConfig.create(5000, 1, true, true, 0, 1000)  // Invalid: minWordLength must be >= 1
            ),
            "Should throw IllegalArgumentException for invalid minWordLength");
    }

    // ==================== TEXT CLEANING TESTS ====================

    @Test
    @DisplayName("cleanText should remove URLs")
    void testCleanText_RemovesUrls() {
        String input = "Check this link https://example.com for more info";
        String result = preprocessor.cleanText(input);

        assertTrue(result.contains("url_token"),
            "URLs should be replaced with URL_TOKEN");
        assertFalse(result.contains("https://example.com"),
            "Original URL should be removed");
    }

    @Test
    @DisplayName("cleanText should remove emails")
    void testCleanText_RemovesEmails() {
        String input = "Contact us at support@example.com for help";
        String result = preprocessor.cleanText(input);

        assertTrue(result.contains("email_token"),
            "Emails should be replaced with EMAIL_TOKEN");
        assertFalse(result.contains("support@example.com"),
            "Original email should be removed");
    }

    @Test
    @DisplayName("cleanText should handle mentions")
    void testCleanText_HandlesMentions() {
        String input = "Great post @username! Really enjoyed it";
        String result = preprocessor.cleanText(input);

        assertTrue(result.contains("mention_token"),
            "Mentions should be replaced with MENTION_TOKEN");
        assertFalse(result.contains("@username"),
            "Original mention should be removed");
    }

    @Test
    @DisplayName("cleanText should handle hashtags")
    void testCleanText_HandlesHashtags() {
        String input = "Check out #MachineLearning trends";
        String result = preprocessor.cleanText(input);

        assertTrue(result.contains("hashtag_"),
            "Hashtags should be replaced with HASHTAG_ prefix");
        assertTrue(result.toLowerCase().contains("machinelearning"),
            "Hashtag content should be preserved");
    }

    @Test
    @DisplayName("cleanText should preserve positive emoticons when configured")
    void testCleanText_PreservesPositiveEmoticons() {
        String input = "This is great :) I'm so happy :D";
        String result = preprocessor.cleanText(input);

        assertTrue(result.contains("positive_emoticon"),
            "Positive emoticons should be replaced with POSITIVE_EMOTICON token");
    }

    @Test
    @DisplayName("cleanText should preserve negative emoticons when configured")
    void testCleanText_PreservesNegativeEmoticons() {
        String input = "This is terrible :( So disappointing :(";
        String result = preprocessor.cleanText(input);

        assertTrue(result.contains("negative_emoticon"),
            "Negative emoticons should be replaced with NEGATIVE_EMOTICON token");
    }

    @Test
    @DisplayName("cleanText should expand contractions")
    void testCleanText_ExpandsContractions() {
        String input = "I don't think this won't work";
        String result = preprocessor.cleanText(input);

        verify(mockContractionExpander, atLeastOnce()).expand(anyString());
        // After cleaning and lowercasing
        assertTrue(result.contains("do not") || result.contains("will not"),
            "Contractions should be expanded");
    }

    @Test
    @DisplayName("cleanText should remove HTML tags")
    void testCleanText_RemovesHtmlTags() {
        String input = "This is <b>bold</b> and <a href='test'>linked</a> text";
        String result = preprocessor.cleanText(input);

        assertFalse(result.contains("<b>"),
            "HTML tags should be removed");
        assertFalse(result.contains("</b>"),
            "HTML closing tags should be removed");
        assertFalse(result.contains("<a"),
            "HTML anchor tags should be removed");
    }

    @Test
    @DisplayName("cleanText should handle repeated characters")
    void testCleanText_HandlesRepeatedCharacters() {
        String input = "Sooooo goooood!!!!";
        String result = preprocessor.cleanText(input);

        // Should reduce repeated chars (more than 3) to 2
        assertFalse(result.contains("oooo"),
            "Repeated characters should be reduced");
    }

    @Test
    @DisplayName("cleanText should normalize whitespace")
    void testCleanText_NormalizesWhitespace() {
        String input = "Multiple    spaces   between    words";
        String result = preprocessor.cleanText(input);

        assertFalse(result.contains("    "),
            "Multiple spaces should be normalized to single space");
        assertFalse(result.matches("\\s{2,}"),
            "Should not contain multiple consecutive whitespaces");
    }

    @Test
    @DisplayName("cleanText should convert to lowercase")
    void testCleanText_ConvertsToLowercase() {
        String input = "UPPERCASE And MixedCase Text";
        String result = preprocessor.cleanText(input);

        assertEquals(result, result.toLowerCase(),
            "All text should be converted to lowercase");
    }

    @Test
    @DisplayName("cleanText should handle null or empty input gracefully")
    void testCleanText_HandlesNullOrEmpty() {
        assertEquals("", preprocessor.cleanText(null),
            "Null input should return empty string");
        assertEquals("", preprocessor.cleanText(""),
            "Empty input should return empty string");
        assertEquals("", preprocessor.cleanText("   "),
            "Whitespace-only input should return empty string");
    }

    // ==================== TOKENIZATION TESTS ====================

    @Test
    @DisplayName("tokenize should call AdvancedTokenizer")
    void testTokenize_CallsAdvancedTokenizer() {
        String cleanedText = "this is a test";
        List<String> result = preprocessor.tokenize(cleanedText);

        verify(mockTokenizer, times(1)).tokenize(cleanedText);
        assertNotNull(result);
    }

    @Test
    @DisplayName("tokenize should handle empty input")
    void testTokenize_HandlesEmptyInput() {
        List<String> result = preprocessor.tokenize("");

        assertNotNull(result);
        assertTrue(result.isEmpty() || result.size() == 1,
            "Empty input should return empty list or single empty token");
    }

    // ==================== STOPWORD REMOVAL TESTS ====================

    @Test
    @DisplayName("removeStopwords should call IntelligentStopwordRemover")
    void testRemoveStopwords_CallsStopwordRemover() {
        List<String> tokens = List.of("this", "is", "good", "product");
        List<String> result = preprocessor.removeStopwords(tokens);

        verify(mockStopwordRemover, times(1)).removeStopwords(tokens);
        assertNotNull(result);
    }

    @Test
    @DisplayName("removeStopwords should handle empty token list")
    void testRemoveStopwords_HandlesEmptyList() {
        List<String> emptyTokens = new ArrayList<>();
        List<String> result = preprocessor.removeStopwords(emptyTokens);

        assertNotNull(result);
        assertTrue(result.isEmpty());
    }

    // ==================== FIT/TRANSFORM WORKFLOW TESTS ====================

    @Test
    @DisplayName("fit should throw when data is null")
    void testFit_NullData_ThrowsException() {
        assertThrows(IllegalArgumentException.class,
            () -> preprocessor.fit(null),
            "Fitting with null data should throw IllegalArgumentException");
    }

    @Test
    @DisplayName("fit should throw when data is empty")
    void testFit_EmptyData_ThrowsException() {
        List<Dataset> emptyData = new ArrayList<>();

        assertThrows(IllegalArgumentException.class,
            () -> preprocessor.fit(emptyData),
            "Fitting with empty data should throw IllegalArgumentException");
    }

    @Test
    @DisplayName("fit should update fitted state")
    void testFit_UpdatesFittedState() {
        List<Dataset> trainingData = createMockTrainingData(20);

        assertFalse(preprocessor.isFitted(),
            "Preprocessor should not be fitted initially");

        preprocessor.fit(trainingData);

        assertTrue(preprocessor.isFitted(),
            "Preprocessor should be fitted after calling fit()");
    }

    // REMOVED: TextPreprocessor no longer injects TFIDFFeatureExtractor
    // Feature extraction happens in service layer with pre-cleaned datasets

    @Test
    @DisplayName("transform should throw when not fitted")
    void testTransform_BeforeFit_ThrowsException() {
        assertThrows(IllegalStateException.class,
            () -> preprocessor.transform("Test text"),
            "Transform before fit should throw IllegalStateException");
    }

    @Test
    @DisplayName("transform should process text after fit")
    void testTransform_AfterFit_ProcessesText() {
        // Arrange
        List<Dataset> trainingData = createMockTrainingData(20);
        preprocessor.fit(trainingData);

        // Act
        String result = preprocessor.transform("This is a test text");

        // Assert
        assertNotNull(result);
        verify(mockContractionExpander, atLeastOnce()).expand(anyString());
        verify(mockTokenizer, atLeastOnce()).tokenize(anyString());
        verify(mockStopwordRemover, atLeastOnce()).removeStopwords(any());
    }

    @Test
    @DisplayName("transform should handle null or empty text gracefully")
    void testTransform_HandlesNullOrEmpty() {
        List<Dataset> trainingData = createMockTrainingData(20);
        preprocessor.fit(trainingData);

        assertEquals("", preprocessor.transform(null),
            "Transform with null should return empty string");
        assertEquals("", preprocessor.transform(""),
            "Transform with empty string should return empty string");
    }

    @Test
    @DisplayName("fit and transform should use consistent preprocessing")
    void testFitTransform_ConsistentPreprocessing() {
        // This test ensures that fit() and transform() use the same preprocessing logic
        String testText = "This is a test message";

        // Create two preprocessors with identical configuration
        TextPreprocessor preprocessor1 = new TextPreprocessor(
            mockContractionExpander,
            mockTokenizer,
            mockStopwordRemover,
            TestFeatureConfig.createDefault()
        );

        TextPreprocessor preprocessor2 = new TextPreprocessor(
            mockContractionExpander,
            mockTokenizer,
            mockStopwordRemover,
            TestFeatureConfig.createDefault()
        );

        List<Dataset> trainingData = createMockTrainingData(20);

        // Preprocessor 1: Use fit() -> transform() workflow
        preprocessor1.fit(trainingData);
        String fitTransformResult = preprocessor1.transform(testText);

        // Preprocessor 2: Manually apply preprocessing steps (what fit() should do internally)
        String cleaned = preprocessor2.cleanText(testText);
        List<String> tokens = preprocessor2.tokenize(cleaned);
        List<String> filtered = preprocessor2.removeStopwords(tokens);
        String manualResult = String.join(" ", filtered);

        // Both approaches should produce identical results
        assertEquals(manualResult, fitTransformResult,
            "fit() and transform() must use consistent preprocessing logic");
    }

    // ==================== FULL PIPELINE TESTS ====================

    @Test
    @DisplayName("transform should apply full pipeline after fit")
    void testTransform_AppliesFullPipeline() {
        List<Dataset> trainingData = createMockTrainingData(20);
        preprocessor.fit(trainingData);

        String input = "Don't visit https://spam.com for bad deals!";
        String result = preprocessor.transform(input);

        // Verify all components were called
        verify(mockContractionExpander, atLeastOnce()).expand(anyString());
        verify(mockTokenizer, atLeastOnce()).tokenize(anyString());
        verify(mockStopwordRemover, atLeastOnce()).removeStopwords(any());

        assertNotNull(result);
    }

    @Test
    @DisplayName("transform should handle complex input after fit")
    void testTransform_HandlesComplexInput() {
        List<Dataset> trainingData = createMockTrainingData(20);
        preprocessor.fit(trainingData);

        String complexInput = """
            Hey @user! Check out https://example.com :)
            This is #amazing and I can't believe it!
            Contact: test@email.com for more info.
            """;

        String result = preprocessor.transform(complexInput);

        assertNotNull(result);
        assertFalse(result.isEmpty(),
            "Complex input should produce non-empty output");
    }

    // ==================== STATE MANAGEMENT TESTS ====================

    @Test
    @DisplayName("isFitted should return false initially")
    void testIsFitted_InitiallyFalse() {
        assertFalse(preprocessor.isFitted());
    }

    @Test
    @DisplayName("getPipelineState should return state information")
    void testGetPipelineState_ReturnsState() {
        List<Dataset> trainingData = createMockTrainingData(20);
        preprocessor.fit(trainingData);

        TextPreprocessor.PipelineState state = preprocessor.getPipelineState();

        assertNotNull(state);
        assertTrue(state.vocabularySize >= 0,
            "Vocabulary size should be non-negative");
        assertEquals(20, state.trainingSampleCount,
            "Training sample count should match input");
    }

    @Test
    @DisplayName("reset should clear fitted state")
    void testReset_ClearsFittedState() {
        List<Dataset> trainingData = createMockTrainingData(20);
        preprocessor.fit(trainingData);

        assertTrue(preprocessor.isFitted(),
            "Should be fitted before reset");

        preprocessor.reset();

        assertFalse(preprocessor.isFitted(),
            "Should not be fitted after reset");
        // REMOVED: No longer calls featureExtractor.reset()
    }

    // ==================== THREAD SAFETY TESTS ====================

    @Test
    @DisplayName("Preprocessing should be deterministic across parallel threads")
    void testPreprocessing_DeterministicInParallel() {
        // Arrange
        List<Dataset> trainingData = createMockTrainingData(50);
        preprocessor.fit(trainingData);

        String testInput = "This is a test input with URLs https://example.com and emoticons :)";
        final int ITERATIONS = 1000;

        // Act - Run preprocessing 1000 times in parallel
        java.util.Set<String> uniqueResults = java.util.stream.IntStream.range(0, ITERATIONS)
            .parallel()
            .mapToObj(i -> preprocessor.transform(testInput))
            .collect(java.util.stream.Collectors.toSet());

        // Assert - MUST produce exactly one unique output
        assertEquals(1, uniqueResults.size(),
            "Preprocessing must be deterministic across parallel threads. " +
            "Got " + uniqueResults.size() + " different outputs: " + uniqueResults);
    }

    @Test
    @DisplayName("Batch transform should be thread-safe")
    void testBatchTransform_ThreadSafe() throws InterruptedException {
        // Arrange
        List<Dataset> trainingData = createMockTrainingData(50);
        preprocessor.fit(trainingData);

        List<String> testTexts = List.of(
            "Great product!",
            "Terrible experience.",
            "It's okay, nothing special."
        );

        // Act - Transform in parallel using multiple threads
        List<String> results = testTexts.parallelStream()
            .map(preprocessor::transform)
            .toList();

        // Assert
        assertNotNull(results);
        assertEquals(testTexts.size(), results.size(),
            "Should process all texts");

        // Verify idempotency - running again should produce same results
        List<String> results2 = testTexts.parallelStream()
            .map(preprocessor::transform)
            .toList();

        assertEquals(results, results2,
            "Repeated parallel transforms should produce identical results");
    }

    // ==================== INTEGRATION TESTS ====================

    @Test
    @DisplayName("End-to-end preprocessing workflow should work")
    void testEndToEndWorkflow() {
        // Arrange
        List<Dataset> trainingData = createMockTrainingData(50);
        String testText = "This product is amazing! :) https://buy.now";

        // Act - Fit phase
        preprocessor.fit(trainingData);

        // Act - Transform phase
        String result = preprocessor.transform(testText);

        // Assert
        assertNotNull(result);
        assertTrue(preprocessor.isFitted());

        // Verify pipeline components were used
        verify(mockContractionExpander, atLeastOnce()).expand(anyString());
        verify(mockTokenizer, atLeastOnce()).tokenize(anyString());
        verify(mockStopwordRemover, atLeastOnce()).removeStopwords(any());
    }

    // ==================== HELPER METHODS ====================

    /**
     * Creates mock training data for testing
     */
    private List<Dataset> createMockTrainingData(int size) {
        List<Dataset> datasets = new ArrayList<>();
        for (int i = 0; i < size; i++) {
            Dataset.SentimentLabel label = (i % 2 == 0)
                ? Dataset.SentimentLabel.POSITIVE
                : Dataset.SentimentLabel.NEGATIVE;

            String text = (label == Dataset.SentimentLabel.POSITIVE)
                ? "This is great and wonderful!"
                : "This is terrible and awful!";

            datasets.add(new Dataset.Builder(text, label).build());
        }
        return datasets;
    }
}
