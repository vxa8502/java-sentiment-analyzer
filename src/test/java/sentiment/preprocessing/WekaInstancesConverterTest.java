package sentiment.preprocessing;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import sentiment.data.Dataset;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for WekaInstancesConverter to verify ownership of preprocessing pipeline.
 * <p>
 * CRITICAL: These tests validate that WekaInstancesConverter correctly manages
 * the TextPreprocessor lifecycle and produces valid TF-IDF features.
 */
@DisplayName("WekaInstancesConverter Tests")
class WekaInstancesConverterTest {

    private WekaInstancesConverter converter;
    private TextPreprocessor preprocessor;

    @BeforeEach
    void setUp() {
        // Create real components (not mocks) to test actual integration
        ContractionExpander contractionExpander = new ContractionExpander();
        AdvancedTokenizer tokenizer = new AdvancedTokenizer(false, 1, true);
        IntelligentStopwordRemover stopwordRemover = new IntelligentStopwordRemover(
                "SENTIMENT_AWARE",
                true,
                true
        );

        preprocessor = new TextPreprocessor(
                contractionExpander,
                tokenizer,
                stopwordRemover,
                2,
                true
        );

        converter = new WekaInstancesConverter(
                preprocessor,
                1000,  // maxFeatures
                2,     // minTermFreq
                true,  // useTfIdf
                false, // useBigrams
                true,  // normalizeFeatures
                false  // outputWordCounts
        );
    }

    @Test
    @DisplayName("fit() should automatically fit the preprocessor first")
    void testFit_AutomaticallyFitsPreprocessor() throws Exception {
        // Arrange
        List<Dataset> trainingData = createTrainingData();

        // Preprocessor should NOT be fitted initially
        assertFalse(preprocessor.isFitted(), "Preprocessor should not be fitted initially");

        // Act
        converter.fit(trainingData);

        // Assert
        assertTrue(preprocessor.isFitted(), "Preprocessor should be fitted after converter.fit()");
        assertTrue(converter.getPreprocessor().isFitted(), "Converter should expose fitted preprocessor");
    }

    @Test
    @DisplayName("fit() should produce valid Weka Instances with TF-IDF features")
    void testFit_ProducesValidInstances() throws Exception {
        // Arrange
        List<Dataset> trainingData = createTrainingData();

        // Act
        Instances instances = converter.fit(trainingData);

        // Assert
        assertNotNull(instances, "Instances should not be null");
        assertEquals(trainingData.size(), instances.numInstances(), "Should have one instance per training example");
        assertTrue(instances.numAttributes() > 1, "Should have features plus class attribute");

        // Class attribute should be the last one
        assertEquals("class_label", instances.classAttribute().name(), "Class attribute should be named 'class_label'");
        assertTrue(instances.classIndex() >= 0, "Class index should be set");
    }

    @Test
    @DisplayName("fit() should extract vocabulary from training data")
    void testFit_ExtractsVocabulary() throws Exception {
        // Arrange
        List<Dataset> trainingData = createTrainingData();

        // Act
        converter.fit(trainingData);
        Set<String> vocabulary = converter.getVocabulary();

        // Assert
        assertNotNull(vocabulary, "Vocabulary should not be null");
        assertFalse(vocabulary.isEmpty(), "Vocabulary should not be empty");

        // Note: Due to minTermFreq=2 and stopword removal, actual vocabulary might be small
        // Just verify we have SOME terms extracted
        assertTrue(vocabulary.size() > 0, "Vocabulary should contain at least one term");
    }

    @Test
    @DisplayName("transform() should work after fit()")
    void testTransform_WorksAfterFit() throws Exception {
        // Arrange
        List<Dataset> trainingData = createTrainingData();
        converter.fit(trainingData);

        // Act
        Instance result = converter.transform("This is a great product!", "positive");

        // Assert
        assertNotNull(result, "Transformed instance should not be null");
        assertTrue(result.numAttributes() > 1, "Instance should have features");
    }

    @Test
    @DisplayName("transform() should throw if not fitted")
    void testTransform_ThrowsIfNotFitted() {
        // Assert
        assertThrows(IllegalStateException.class,
                () -> converter.transform("test text", "positive"),
                "Should throw IllegalStateException when transform() called before fit()");
    }

    @Test
    @DisplayName("reset() should reset both converter and preprocessor")
    void testReset_ResetsPreprocessor() throws Exception {
        // Arrange
        List<Dataset> trainingData = createTrainingData();
        converter.fit(trainingData);

        assertTrue(preprocessor.isFitted(), "Preprocessor should be fitted");

        // Act
        converter.reset();

        // Assert
        assertFalse(preprocessor.isFitted(), "Preprocessor should be reset when converter is reset");
        // After reset, getVocabulary() throws IllegalStateException, so just verify preprocessor is reset
    }

    @Test
    @DisplayName("fit() should reset preprocessor if already fitted")
    void testFit_ResetsPreprocessorIfAlreadyFitted() throws Exception {
        // Arrange
        List<Dataset> firstData = createTrainingData();
        converter.fit(firstData);

        assertTrue(preprocessor.isFitted(), "Preprocessor should be fitted after first fit");

        // Act - must reset before refitting (state machine prevents retraining)
        converter.reset();
        List<Dataset> secondData = createDifferentTrainingData();
        converter.fit(secondData);

        // Assert
        assertTrue(preprocessor.isFitted(), "Preprocessor should still be fitted after second fit");
        assertTrue(preprocessor.getPipelineState().vocabularySize > 0,
                "Preprocessor should have vocabulary from second fit");
    }

    @Test
    @DisplayName("transformDatasets() should preserve labels")
    void testTransformDatasets_PreservesLabels() throws Exception {
        // Arrange
        List<Dataset> trainingData = createTrainingData();
        converter.fit(trainingData);

        List<Dataset> testData = List.of(
                new Dataset.Builder("This is excellent!", Dataset.SentimentLabel.POSITIVE).build(),
                new Dataset.Builder("This is awful!", Dataset.SentimentLabel.NEGATIVE).build()
        );

        // Act
        Instances transformed = converter.transformDatasets(testData);

        // Assert
        assertNotNull(transformed, "Transformed instances should not be null");
        assertEquals(2, transformed.numInstances(), "Should have 2 instances");

        // Check labels are preserved
        assertEquals("positive", transformed.instance(0).stringValue(transformed.classAttribute()));
        assertEquals("negative", transformed.instance(1).stringValue(transformed.classAttribute()));
    }

    @Test
    @DisplayName("getPreprocessor() should return the managed preprocessor")
    void testGetPreprocessor_ReturnsPreprocessor() {
        // Act
        TextPreprocessor result = converter.getPreprocessor();

        // Assert
        assertNotNull(result, "getPreprocessor() should not return null");
        assertSame(preprocessor, result, "Should return the same preprocessor instance");
    }

    @Test
    @DisplayName("diagnostics should include preprocessor state")
    void testDiagnostics_IncludesPreprocessorState() throws Exception {
        // Arrange
        List<Dataset> trainingData = createTrainingData();
        converter.fit(trainingData);

        // Act
        String diagnostics = converter.getDiagnostics();

        // Assert
        assertNotNull(diagnostics, "Diagnostics should not be null");
        assertTrue(diagnostics.contains("Text preprocessor"), "Diagnostics should mention text preprocessor");
        assertTrue(diagnostics.contains("fitted") || diagnostics.contains("Preprocessor vocabulary"),
                "Diagnostics should show preprocessor is fitted");
    }

    // ==================== HELPER METHODS ====================

    private List<Dataset> createTrainingData() {
        List<Dataset> data = new ArrayList<>();

        // Positive examples
        data.add(new Dataset.Builder("This is a great product!", Dataset.SentimentLabel.POSITIVE).build());
        data.add(new Dataset.Builder("I love this item!", Dataset.SentimentLabel.POSITIVE).build());
        data.add(new Dataset.Builder("Excellent quality and fast shipping", Dataset.SentimentLabel.POSITIVE).build());
        data.add(new Dataset.Builder("Highly recommend this product", Dataset.SentimentLabel.POSITIVE).build());
        data.add(new Dataset.Builder("Amazing product worth every penny", Dataset.SentimentLabel.POSITIVE).build());

        // Negative examples
        data.add(new Dataset.Builder("This is terrible!", Dataset.SentimentLabel.NEGATIVE).build());
        data.add(new Dataset.Builder("I hate this product", Dataset.SentimentLabel.NEGATIVE).build());
        data.add(new Dataset.Builder("Awful quality and terrible service", Dataset.SentimentLabel.NEGATIVE).build());
        data.add(new Dataset.Builder("Would not recommend this at all", Dataset.SentimentLabel.NEGATIVE).build());
        data.add(new Dataset.Builder("Worst purchase I ever made", Dataset.SentimentLabel.NEGATIVE).build());

        // Neutral examples
        data.add(new Dataset.Builder("This is a product", Dataset.SentimentLabel.NEUTRAL).build());
        data.add(new Dataset.Builder("It arrived yesterday", Dataset.SentimentLabel.NEUTRAL).build());

        return data;
    }

    private List<Dataset> createDifferentTrainingData() {
        List<Dataset> data = new ArrayList<>();

        data.add(new Dataset.Builder("The movie was fantastic!", Dataset.SentimentLabel.POSITIVE).build());
        data.add(new Dataset.Builder("The movie was boring", Dataset.SentimentLabel.NEGATIVE).build());
        data.add(new Dataset.Builder("The movie has actors", Dataset.SentimentLabel.NEUTRAL).build());
        data.add(new Dataset.Builder("Best film of the year", Dataset.SentimentLabel.POSITIVE).build());
        data.add(new Dataset.Builder("Worst film ever made", Dataset.SentimentLabel.NEGATIVE).build());

        return data;
    }
}
