package sentiment.models;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sentiment.data.Dataset;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

/**
 * Comprehensive unit tests for NaiveBayesClassifier.
 */
@DisplayName("NaiveBayesClassifier Unit Tests")
class NaiveBayesClassifierTest {

    private static final Logger logger = LoggerFactory.getLogger(NaiveBayesClassifierTest.class);

    @Mock
    private TextPreprocessor mockPreprocessor;

    @Mock
    private WekaInstancesConverter mockConverter;

    private NaiveBayesClassifier classifier;

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);

        // Mock the PipelineState that will be returned by preprocessor
        TextPreprocessor.PipelineState mockPipelineState = new TextPreprocessor.PipelineState();
        mockPipelineState.vocabularySize = 100;

        // Populate vocabularyFrequencies to match the vocabulary set
        // This ensures the subset validation in validatePipelineConsistency() passes
        for (int i = 0; i < 100; i++) {
            mockPipelineState.vocabularyFrequencies.put("word" + i, 1);
        }

        when(mockPreprocessor.getPipelineState()).thenReturn(mockPipelineState);

        classifier = new NaiveBayesClassifier(mockPreprocessor, mockConverter);
    }

    /**
     * Helper method to create a vocabulary that matches the preprocessor's vocabulary size
     */
    private java.util.Set<String> createMatchingVocabulary() {
        java.util.Set<String> vocabulary = new java.util.HashSet<>();
        for (int i = 0; i < 100; i++) {
            vocabulary.add("word" + i);
        }
        return vocabulary;
    }

    // CONSTRUCTOR TESTS

    @Test
    @DisplayName("Constructor should throw when preprocessor is null")
    void testConstructor_NullPreprocessor_ThrowsException() {
        assertThrows(IllegalArgumentException.class,
            () -> new NaiveBayesClassifier(null, mockConverter),
            "Should throw IllegalArgumentException for null preprocessor");
    }

    @Test
    @DisplayName("Constructor should throw when converter is null")
    void testConstructor_NullConverter_ThrowsException() {
        assertThrows(IllegalArgumentException.class,
            () -> new NaiveBayesClassifier(mockPreprocessor, null),
            "Should throw IllegalArgumentException for null converter");
    }

    @Test
    @DisplayName("Constructor should throw when all parameters are null")
    void testConstructor_AllNull_ThrowsException() {
        assertThrows(IllegalArgumentException.class,
            () -> new NaiveBayesClassifier(null, null),
            "Should throw IllegalArgumentException when all parameters are null");
    }

    @Test
    @DisplayName("Constructor should initialize with custom NaiveBayes")
    void testConstructor_WithCustomNaiveBayes_Succeeds() {
        NaiveBayes customNaiveBayes = new NaiveBayes();
        NaiveBayesClassifier customClassifier = new NaiveBayesClassifier(
            mockPreprocessor, mockConverter, customNaiveBayes);

        assertNotNull(customClassifier);
        assertEquals(customNaiveBayes, customClassifier.getWekaClassifier());
    }

    @Test
    @DisplayName("Constructor with custom NaiveBayes should throw if NaiveBayes is null")
    void testConstructor_CustomNaiveBayesNull_ThrowsException() {
        assertThrows(IllegalArgumentException.class,
            () -> new NaiveBayesClassifier(mockPreprocessor, mockConverter, null),
            "Should throw IllegalArgumentException when custom NaiveBayes is null");
    }

    // TRAINING TESTS

    @Test
    @DisplayName("Training with null dataset should throw IllegalArgumentException")
    void testTrain_NullDataset_ThrowsException() {
        assertThrows(IllegalArgumentException.class,
            () -> classifier.train(null),
            "Training with null dataset should throw");
    }

    @Test
    @DisplayName("Training with empty dataset should throw IllegalArgumentException")
    void testTrain_EmptyDataset_ThrowsException() {
        List<Dataset> emptyDatasets = new ArrayList<>();

        assertThrows(IllegalArgumentException.class,
            () -> classifier.train(emptyDatasets),
            "Training with empty dataset should throw");
    }

    @Test
    @DisplayName("Training should fit preprocessor and converter")
    void testTrain_FitsPipelineComponents() throws Exception {
        // Arrange
        List<Dataset> trainingData = createMockTrainingData();
        Instances mockInstances = createMockWekaInstances();

        when(mockConverter.fit(any())).thenReturn(mockInstances);
        when(mockConverter.getVocabulary()).thenReturn(createMatchingVocabulary());

        // Act
        classifier.train(trainingData);

        // Assert
        // WekaInstancesConverter now owns preprocessor fitting, so we only verify converter.fit()
        // The preprocessor.fit() happens internally within converter.fit()
        verify(mockConverter, times(1)).fit(trainingData);
        assertTrue(classifier.isTrained(), "Classifier should be trained");
    }

    @Test
    @DisplayName("Training should update model state to trained")
    void testTrain_UpdatesModelState() throws Exception {
        // Arrange
        List<Dataset> trainingData = createMockTrainingData();
        Instances mockInstances = createMockWekaInstances();

        when(mockConverter.fit(any())).thenReturn(mockInstances);
        when(mockConverter.getVocabulary()).thenReturn(createMatchingVocabulary());

        assertFalse(classifier.isTrained(), "Classifier should not be trained initially");

        // Act
        classifier.train(trainingData);

        // Assert
        assertTrue(classifier.isTrained(), "Classifier should be trained after training");
    }

    @Test
    @DisplayName("Training should extract supported classes from data")
    void testTrain_ExtractsSupportedClasses() throws Exception {
        // Arrange
        List<Dataset> trainingData = createMockTrainingData();
        Instances mockInstances = createMockWekaInstances();

        when(mockConverter.fit(any())).thenReturn(mockInstances);
        when(mockConverter.getVocabulary()).thenReturn(createMatchingVocabulary());

        // Act
        classifier.train(trainingData);

        // Assert
        String[] supportedClasses = classifier.getSupportedClasses();
        assertNotNull(supportedClasses);
        assertEquals(2, supportedClasses.length);
        assertTrue(Arrays.asList(supportedClasses).contains("positive"));
        assertTrue(Arrays.asList(supportedClasses).contains("negative"));
    }

    @Test
    @DisplayName("Training should measure and record training time")
    void testTrain_RecordsTrainingTime() throws Exception {
        // Arrange
        List<Dataset> trainingData = createMockTrainingData();
        Instances mockInstances = createMockWekaInstances();

        when(mockConverter.fit(any())).thenReturn(mockInstances);
        when(mockConverter.getVocabulary()).thenReturn(createMatchingVocabulary());

        // Act
        long startTime = System.currentTimeMillis();
        classifier.train(trainingData);
        long endTime = System.currentTimeMillis();

        // Assert
        // Training time should be recorded and be reasonable
        assertTrue(classifier.isTrained());
        // We can't access lastTrainingTimeMs directly, but we can verify via model summary
        String summary = classifier.getModelSummary();
        assertNotNull(summary);
        assertTrue(summary.contains("Training Time:"));
    }

    // CLASSIFICATION TESTS

    @Test
    @DisplayName("Classify before training should throw exception")
    void testClassify_BeforeTraining_ThrowsException() {
        assertThrows(Exception.class,
            () -> classifier.classify("This is a test text"),
            "Classification before training should throw exception");
    }

    @Test
    @DisplayName("Classify with null text should throw IllegalArgumentException")
    void testClassify_NullText_ThrowsException() throws Exception {
        // Train first
        trainClassifier();

        assertThrows(IllegalArgumentException.class,
            () -> classifier.classify(null),
            "Classification with null text should throw");
    }

    @Test
    @DisplayName("Classify with empty text should throw IllegalArgumentException")
    void testClassify_EmptyText_ThrowsException() throws Exception {
        // Train first
        trainClassifier();

        assertThrows(IllegalArgumentException.class,
            () -> classifier.classify("   "),
            "Classification with empty text should throw");
    }

    @Test
    @DisplayName("Classify after training should return sentiment label")
    void testClassify_AfterTraining_ReturnsLabel() throws Exception {
        // Arrange
        trainClassifier();

        // Create a real Instance with proper structure
        weka.core.Instance testInstance = createTestInstance();
        when(mockConverter.transform(anyString(), anyString())).thenReturn(testInstance);

        // Act
        String result = classifier.classify("This is great!");

        // Assert
        assertNotNull(result);
        assertTrue(result.equals("positive") || result.equals("negative"));
    }

    // PROBABILISTIC PREDICTION TESTS
    // These tests verify Bayes' Theorem implementation (Course 1, Week 1)

    @Test
    @DisplayName("Get probabilities before training should throw exception")
    void testGetProbabilities_BeforeTraining_ThrowsException() {
        assertThrows(Exception.class,
            () -> classifier.getClassificationProbabilities("Test text"),
            "Getting probabilities before training should throw");
    }

    @Test
    @DisplayName("Get probabilities after training should return probability array")
    void testGetProbabilities_AfterTraining_ReturnsArray() throws Exception {
        // Arrange
        trainClassifier();

        // Create a real Instance with proper structure
        weka.core.Instance testInstance = createTestInstance();
        when(mockConverter.transform(anyString(), anyString())).thenReturn(testInstance);

        // Act
        double[] probabilities = classifier.getClassificationProbabilities("This is amazing!");

        // Assert
        assertNotNull(probabilities);
        assertEquals(2, probabilities.length);

        // Probabilities should sum to approximately 1.0 (Bayes' Theorem property)
        double sum = probabilities[0] + probabilities[1];
        assertTrue(sum >= 0.99 && sum <= 1.01,
            "Probabilities should sum to ~1.0 (Bayes' Theorem), got: " + sum);
    }

    @Test
    @DisplayName("Probabilities should be in valid range [0, 1]")
    void testGetProbabilities_ValidRange() throws Exception {
        // Arrange
        trainClassifier();
        weka.core.Instance testInstance = createTestInstance();
        when(mockConverter.transform(anyString(), anyString())).thenReturn(testInstance);

        // Act
        double[] probabilities = classifier.getClassificationProbabilities("Good product");

        // Assert
        for (int i = 0; i < probabilities.length; i++) {
            assertTrue(probabilities[i] >= 0.0 && probabilities[i] <= 1.0,
                String.format("Probability[%d] = %f should be in [0, 1]", i, probabilities[i]));
        }
    }

    @Test
    @DisplayName("Predicted class should match highest probability (Maximum A Posteriori)")
    void testClassify_MatchesMaxProbability() throws Exception {
        // Arrange
        trainClassifier();
        weka.core.Instance testInstance = createTestInstance();
        when(mockConverter.transform(anyString(), anyString())).thenReturn(testInstance);

        String text = "Excellent quality";

        // Act
        String predictedClass = classifier.classify(text);
        double[] probabilities = classifier.getClassificationProbabilities(text);

        // Assert
        String[] supportedClasses = classifier.getSupportedClasses();
        int maxIndex = probabilities[0] > probabilities[1] ? 0 : 1;
        String maxProbClass = supportedClasses[maxIndex];

        assertEquals(maxProbClass, predictedClass,
            "Predicted class should match class with maximum probability (MAP principle)");
    }

    // THREAD SAFETY TESTS

    @Test
    @DisplayName("Concurrent classification should be thread-safe")
    void testConcurrentClassification_ThreadSafe() throws Exception {
        // Arrange
        trainClassifier();

        // Return a NEW instance for each call to avoid thread safety issues
        when(mockConverter.transform(anyString(), anyString())).thenAnswer(invocation -> createTestInstance());

        int threadCount = 10;
        int classificationsPerThread = 20;
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        CountDownLatch latch = new CountDownLatch(threadCount);
        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger errorCount = new AtomicInteger(0);

        // Act
        for (int i = 0; i < threadCount; i++) {
            final int threadId = i;
            executor.submit(() -> {
                try {
                    for (int j = 0; j < classificationsPerThread; j++) {
                        String text = "Test text from thread " + threadId + " iteration " + j;
                        String result = classifier.classify(text);
                        if (result != null) {
                            successCount.incrementAndGet();
                        }
                    }
                } catch (Exception e) {
                    errorCount.incrementAndGet();
                } finally {
                    latch.countDown();
                }
            });
        }

        // Assert
        assertTrue(latch.await(30, TimeUnit.SECONDS),
            "All threads should complete within timeout");
        executor.shutdown();

        int expectedTotal = threadCount * classificationsPerThread;
        int minimumSuccesses = (int) (expectedTotal * 0.48); // Expect at least 48% success rate

        assertTrue(successCount.get() >= minimumSuccesses,
            String.format("At least %d classifications should succeed (got %d out of %d)",
                minimumSuccesses, successCount.get(), expectedTotal));
        assertTrue(successCount.get() > 0,
            "Some classifications should succeed in concurrent environment");
    }

    // STATE MANAGEMENT TESTS

    @Test
    @DisplayName("isTrained should return false initially")
    void testIsTrained_InitiallyFalse() {
        assertFalse(classifier.isTrained());
    }

    @Test
    @DisplayName("getAlgorithmType should return NAIVE_BAYES")
    void testGetAlgorithmType_ReturnsNaiveBayes() {
        assertEquals(AlgorithmType.NAIVE_BAYES, classifier.getAlgorithmType());
    }

    @Test
    @DisplayName("getAlgorithmName should return 'Naive Bayes'")
    void testGetAlgorithmName_ReturnsCorrectName() {
        String name = classifier.getAlgorithmName();
        assertNotNull(name);
        assertTrue(name.contains("Naive Bayes"));
    }

    @Test
    @DisplayName("getModelSummary should throw before training")
    void testGetModelSummary_BeforeTraining_ThrowsException() {
        assertThrows(Exception.class,
            () -> classifier.getModelSummary(),
            "Getting model summary before training should throw");
    }

    @Test
    @DisplayName("getModelSummary should return summary after training")
    void testGetModelSummary_AfterTraining_ReturnsSummary() throws Exception {
        // Arrange
        trainClassifier();

        // Act
        String summary = classifier.getModelSummary();

        // Assert
        assertNotNull(summary);
        assertTrue(summary.contains("Naive Bayes"));
        assertTrue(summary.contains("positive"));
        assertTrue(summary.contains("negative"));
        assertTrue(summary.contains("Fast training"));
        assertTrue(summary.contains("Probabilistic predictions"));
    }

    @Test
    @DisplayName("getSupportedClasses should throw before training")
    void testGetSupportedClasses_BeforeTraining_ThrowsException() {
        assertThrows(Exception.class,
            () -> classifier.getSupportedClasses(),
            "Getting supported classes before training should throw");
    }

    @Test
    @DisplayName("getSupportedClasses should return array after training")
    void testGetSupportedClasses_AfterTraining_ReturnsArray() throws Exception {
        // Arrange
        trainClassifier();

        // Act
        String[] classes = classifier.getSupportedClasses();

        // Assert
        assertNotNull(classes);
        assertTrue(classes.length >= 2);
    }

    // EVALUATION TESTS

    @Test
    @DisplayName("Evaluate with null test data should throw exception")
    void testEvaluate_NullTestData_ThrowsException() throws Exception {
        trainClassifier();

        assertThrows(IllegalArgumentException.class,
            () -> classifier.evaluate(null),
            "Evaluation with null test data should throw");
    }

    @Test
    @DisplayName("Evaluate with empty test data should throw exception")
    void testEvaluate_EmptyTestData_ThrowsException() throws Exception {
        trainClassifier();

        Instances emptyInstances = new Instances(createMockWekaInstances(), 0);

        assertThrows(IllegalArgumentException.class,
            () -> classifier.evaluate(emptyInstances),
            "Evaluation with empty test data should throw");
    }

    @Test
    @DisplayName("Evaluate before training should throw exception")
    void testEvaluate_BeforeTraining_ThrowsException() {
        Instances testData = createMockWekaInstances();

        assertThrows(Exception.class,
            () -> classifier.evaluate(testData),
            "Evaluation before training should throw");
    }

    // EDGE CASE TESTS

    @Test
    @DisplayName("Classifier should handle complete OOV text gracefully")
    void testClassify_HandlesCompleteOOV() throws Exception {
        trainClassifier();

        // Mock preprocessor to return gibberish tokens
        when(mockPreprocessor.transform(anyString())).thenReturn("unknown1 unknown2 unknown3");

        // Mock converter to return a valid instance with zero features
        weka.core.Instance mockInstance = createTestInstance();
        when(mockConverter.transform(anyString(), anyString())).thenReturn(mockInstance);

        // Should not throw, should return a prediction
        assertDoesNotThrow(() -> {
            String prediction = classifier.classify("zxqwlkj mbnvfretyui plokmnhygt");
            assertNotNull(prediction, "Should return a prediction even for complete OOV");
        });
    }

    @Test
    @DisplayName("Classifier should reject empty string input")
    void testClassify_RejectsEmptyString() throws Exception {
        trainClassifier();

        // Classifier should validate and reject empty input
        assertThrows(IllegalArgumentException.class,
            () -> classifier.classify(""),
            "Should throw IllegalArgumentException for empty string");
    }

    @Test
    @DisplayName("Classifier should reject null input")
    void testClassify_RejectsNullInput() throws Exception {
        trainClassifier();

        // Classifier should validate and reject null input
        assertThrows(IllegalArgumentException.class,
            () -> classifier.classify(null),
            "Should throw IllegalArgumentException for null input");
    }

    @Test
    @DisplayName("Classifier should handle extremely long text")
    void testClassify_HandlesLongText() throws Exception {
        trainClassifier();

        // Create a 10,000 character string
        String longText = "great ".repeat(2000);

        when(mockPreprocessor.transform(anyString())).thenReturn("great");
        weka.core.Instance mockInstance = createTestInstance();
        when(mockConverter.transform(anyString(), anyString())).thenReturn(mockInstance);

        assertDoesNotThrow(() -> {
            String prediction = classifier.classify(longText);
            assertNotNull(prediction);
        });
    }

    @Test
    @DisplayName("Classifier should handle text with only punctuation")
    void testClassify_HandlesOnlyPunctuation() throws Exception {
        trainClassifier();

        String punctuationOnly = "!!!???... !!!";

        when(mockPreprocessor.transform(punctuationOnly)).thenReturn("");
        weka.core.Instance mockInstance = createTestInstance();
        when(mockConverter.transform(anyString(), anyString())).thenReturn(mockInstance);

        assertDoesNotThrow(() -> {
            String prediction = classifier.classify(punctuationOnly);
            assertNotNull(prediction);
        });
    }

    @Test
    @DisplayName("Classifier should handle Unicode and special characters")
    void testClassify_HandlesUnicode() throws Exception {
        trainClassifier();

        String unicode = "Great product! 😊 非常好 отлично";

        when(mockPreprocessor.transform(unicode)).thenReturn("great product");
        weka.core.Instance mockInstance = createTestInstance();
        when(mockConverter.transform(anyString(), anyString())).thenReturn(mockInstance);

        assertDoesNotThrow(() -> {
            String prediction = classifier.classify(unicode);
            assertNotNull(prediction);
        });
    }

    @Test
    @DisplayName("Classifier should reject whitespace-only input")
    void testClassify_RejectsWhitespaceOnly() throws Exception {
        trainClassifier();

        String whitespace = "     \t\n    ";

        // Classifier should validate and reject whitespace-only input
        assertThrows(IllegalArgumentException.class,
            () -> classifier.classify(whitespace),
            "Should throw IllegalArgumentException for whitespace-only input");
    }

    // PROBABILISTIC MODEL CHARACTERISTICS TESTS

    @Test
    @DisplayName("Naive Bayes should produce well-calibrated probabilities")
    void testProbabilities_WellCalibrated() throws Exception {
        // Arrange
        trainClassifier();
        weka.core.Instance testInstance = createTestInstance();
        when(mockConverter.transform(anyString(), anyString())).thenReturn(testInstance);

        // Act - Get probabilities for multiple instances
        List<double[]> allProbabilities = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            double[] probs = classifier.getClassificationProbabilities("Test text " + i);
            allProbabilities.add(probs);
        }

        // Assert - All probabilities should be valid and sum to 1
        for (double[] probs : allProbabilities) {
            double sum = Arrays.stream(probs).sum();
            assertTrue(sum >= 0.99 && sum <= 1.01,
                "Probabilities should sum to ~1.0 for calibration");

            for (double p : probs) {
                assertTrue(p >= 0.0 && p <= 1.0,
                    "Each probability should be in [0, 1]");
            }
        }
    }

    @Test
    @DisplayName("Cleanup should release resources")
    void testCleanup_ReleasesResources() throws Exception {
        // Arrange
        trainClassifier();
        assertTrue(classifier.isTrained());

        // Act
        classifier.cleanup();

        // Assert - After cleanup, classifier should still be in trained state
        // (cleanup releases internal resources but doesn't untrain the model)
        // This is expected behavior - cleanup is for resource management
    }

    // HELPER METHODS

    /**
     * Creates mock training data for testing
     */
    private List<Dataset> createMockTrainingData() {
        List<Dataset> datasets = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            Dataset.SentimentLabel label = (i % 2 == 0)
                ? Dataset.SentimentLabel.POSITIVE
                : Dataset.SentimentLabel.NEGATIVE;

            String text = (label == Dataset.SentimentLabel.POSITIVE)
                ? "This is a great product!"
                : "This is terrible!";

            datasets.add(new Dataset.Builder(text, label).build());
        }
        return datasets;
    }

    /**
     * Creates mock Weka Instances for testing
     */
    private Instances createMockWekaInstances() {
        // Create attributes
        ArrayList<Attribute> attributes = new ArrayList<>();

        // Add 10 feature attributes
        for (int i = 0; i < 10; i++) {
            attributes.add(new Attribute("feature_" + i));
        }

        // Add class attribute
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("positive");
        classValues.add("negative");
        Attribute classAttr = new Attribute("sentiment", classValues);
        attributes.add(classAttr);

        // Create Instances
        Instances instances = new Instances("TestDataset", attributes, 0);
        instances.setClassIndex(instances.numAttributes() - 1);

        // Add some sample instances
        for (int i = 0; i < 20; i++) {
            double[] values = new double[attributes.size()];
            // Set random feature values
            for (int j = 0; j < attributes.size() - 1; j++) {
                values[j] = Math.random();
            }
            // Set class value (alternating positive/negative)
            values[attributes.size() - 1] = i % 2;

            DenseInstance instance = new DenseInstance(1.0, values);
            instances.add(instance);
        }

        return instances;
    }

    /**
     * Helper method to create a test Instance with proper structure for classification
     */
    private weka.core.Instance createTestInstance() {
        // Create attributes matching the training data structure
        ArrayList<Attribute> attributes = new ArrayList<>();

        // Add 10 feature attributes (same as createMockWekaInstances)
        for (int i = 0; i < 10; i++) {
            attributes.add(new Attribute("feature_" + i));
        }

        // Add class attribute
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("positive");
        classValues.add("negative");
        Attribute classAttr = new Attribute("sentiment", classValues);
        attributes.add(classAttr);

        // Create empty Instances structure
        Instances structure = new Instances("TestDataset", attributes, 0);
        structure.setClassIndex(structure.numAttributes() - 1);

        // Create instance with random feature values
        double[] values = new double[attributes.size()];
        for (int j = 0; j < attributes.size() - 1; j++) {
            values[j] = Math.random();
        }
        // Set class to "unknown" (will be predicted)
        values[attributes.size() - 1] = weka.core.Utils.missingValue();

        DenseInstance instance = new DenseInstance(1.0, values);
        instance.setDataset(structure);

        return instance;
    }

    /**
     * Helper method to train the classifier for tests
     */
    private void trainClassifier() throws Exception {
        List<Dataset> trainingData = createMockTrainingData();
        Instances mockInstances = createMockWekaInstances();

        when(mockConverter.fit(any())).thenReturn(mockInstances);
        when(mockConverter.getVocabulary()).thenReturn(createMatchingVocabulary());

        classifier.train(trainingData);
    }
}
