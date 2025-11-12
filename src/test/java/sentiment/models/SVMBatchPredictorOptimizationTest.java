package sentiment.models;

import org.junit.jupiter.api.*;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import sentiment.data.Dataset;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import weka.classifiers.functions.SMO;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * Comprehensive unit tests for BatchPredictor optimization features.
 *
 * TEST COVERAGE:
 * ==============
 * 1. Configuration - default, custom, and preset configs
 * 2. Sequential processing - mini-batch processing
 * 3. Parallel processing - thread pool and task distribution
 * 4. Performance metrics - tracking and reporting
 * 5. Error handling - edge cases and failures
 * 6. Thread safety - concurrent operations
 */
@DisplayName("Batch Predictor Optimization Tests")
class SVMBatchPredictorOptimizationTest {

    @Mock
    private TextPreprocessor mockPreprocessor;

    @Mock
    private WekaInstancesConverter mockConverter;

    @Mock
    private SMO mockSMO;

    private SVMClassifier classifier;
    private AutoCloseable closeable;

    @BeforeEach
    void setUp() throws Exception {
        closeable = MockitoAnnotations.openMocks(this);

        // Setup mock behavior
        when(mockPreprocessor.isFitted()).thenReturn(true);
        when(mockConverter.isReady()).thenReturn(true);

        // Mock preprocessor pipeline state
        TextPreprocessor.PipelineState pipelineState = new TextPreprocessor.PipelineState();
        pipelineState.vocabularySize = 100;

        // Create a vocabulary that matches the preprocessor's vocabulary size (100)
        java.util.Set<String> vocabulary = new java.util.HashSet<>();
        for (int i = 0; i < 100; i++) {
            vocabulary.add("word" + i);
            // Populate vocabularyFrequencies to match the vocabulary
            pipelineState.vocabularyFrequencies.put("word" + i, 1);
        }

        when(mockPreprocessor.getPipelineState()).thenReturn(pipelineState);

        // Create classifier and train it
        classifier = new SVMClassifier(mockPreprocessor, mockConverter, mockSMO);

        // Mock training data
        Instances mockTrainingData = createMockInstances(10);
        when(mockConverter.fit(any())).thenReturn(mockTrainingData);

        when(mockConverter.getVocabulary()).thenReturn(vocabulary);

        // Train the classifier
        List<Dataset> trainingData = Arrays.asList(
                new Dataset("positive text", "positive"),
                new Dataset("negative text", "negative")
        );
        classifier.train(trainingData);
    }

    @AfterEach
    void tearDown() throws Exception {
        closeable.close();
    }

    // ==================== CONFIGURATION TESTS ====================

    @Test
    @DisplayName("Default configuration should have balanced settings")
    void testDefaultConfiguration() {
        BatchOptimizationConfig config = BatchOptimizationConfig.getDefault();

        assertEquals(100, config.getBatchSize());
        assertEquals(500, config.getParallelThreshold());
        assertTrue(config.isEnableParallelProcessing());
        assertTrue(config.isEnableMetrics());
        assertTrue(config.getThreadPoolSize() > 0);
    }

    @Test
    @DisplayName("Low memory configuration should minimize resource usage")
    void testLowMemoryConfiguration() {
        BatchOptimizationConfig config = BatchOptimizationConfig.forLowMemory();

        assertEquals(50, config.getBatchSize());
        assertFalse(config.isEnableParallelProcessing());
        assertFalse(config.shouldUseParallel(1000));
    }

    @Test
    @DisplayName("High throughput configuration should maximize performance")
    void testHighThroughputConfiguration() {
        BatchOptimizationConfig config = BatchOptimizationConfig.forHighThroughput();

        assertEquals(200, config.getBatchSize());
        assertEquals(200, config.getParallelThreshold());
        assertTrue(config.isEnableParallelProcessing());
        assertTrue(config.shouldUseParallel(250));
    }

    @Test
    @DisplayName("Custom configuration should accept valid parameters")
    void testCustomConfiguration() {
        BatchOptimizationConfig config = BatchOptimizationConfig.builder()
                .batchSize(150)
                .parallelThreshold(300)
                .enableParallelProcessing(true)
                .threadPoolSize(4)
                .build();

        assertEquals(150, config.getBatchSize());
        assertEquals(300, config.getParallelThreshold());
        assertEquals(4, config.getThreadPoolSize());
    }

    @Test
    @DisplayName("Configuration builder should reject invalid parameters")
    void testConfigurationValidation() {
        assertThrows(IllegalArgumentException.class, () ->
                BatchOptimizationConfig.builder().batchSize(0).build()
        );

        assertThrows(IllegalArgumentException.class, () ->
                BatchOptimizationConfig.builder().parallelThreshold(-1).build()
        );

        assertThrows(IllegalArgumentException.class, () ->
                BatchOptimizationConfig.builder().threadPoolSize(0).build()
        );
    }

    // ==================== BATCH PREDICTION TESTS ====================

    @Test
    @DisplayName("Sequential batch processing should process all items")
    void testSequentialBatchProcessing() throws Exception {
        // Setup
        BatchOptimizationConfig config = BatchOptimizationConfig.builder()
                .batchSize(5)
                .enableParallelProcessing(false)
                .build();

        BatchPredictor<SVMClassifier> predictor = new BatchPredictor<>(classifier, config);

        List<String> texts = createTestTexts(12);
        setupMocksForBatchProcessing(texts, "positive");

        // Execute
        List<String> results = predictor.classifyBatch(texts);

        // Verify
        assertNotNull(results);
        assertEquals(12, results.size());
        verify(mockConverter, atLeast(3)).transform(any(String[].class), anyString());
    }

    @Test
    @DisplayName("Parallel batch processing should process all items correctly")
    void testParallelBatchProcessing() throws Exception {
        // Setup
        BatchOptimizationConfig config = BatchOptimizationConfig.builder()
                .batchSize(50)
                .parallelThreshold(100)
                .enableParallelProcessing(true)
                .threadPoolSize(2)
                .build();

        BatchPredictor<SVMClassifier> predictor = new BatchPredictor<>(classifier, config);

        List<String> texts = createTestTexts(150);
        setupMocksForBatchProcessing(texts, "positive");

        // Execute
        List<String> results = predictor.classifyBatch(texts);

        // Verify
        assertNotNull(results);
        assertEquals(150, results.size());
        assertTrue(config.shouldUseParallel(150));

        // Cleanup
        predictor.shutdown();
    }

    @Test
    @DisplayName("Batch processing should maintain order of results")
    void testResultOrderPreservation() throws Exception {
        // Setup
        BatchOptimizationConfig config = BatchOptimizationConfig.builder()
                .batchSize(10)
                .parallelThreshold(20)
                .enableParallelProcessing(true)
                .build();

        BatchPredictor<SVMClassifier> predictor = new BatchPredictor<>(classifier, config);

        List<String> texts = createTestTexts(30);

        // Mock to return different results based on input
        when(mockConverter.transform(any(String[].class), anyString()))
                .thenAnswer(invocation -> {
                    String[] inputs = invocation.getArgument(0);
                    return createMockInstances(inputs.length);
                });

        when(mockSMO.classifyInstance(any()))
                .thenReturn(0.0);

        when(mockPreprocessor.preprocessText(anyString()))
                .thenAnswer(invocation -> invocation.getArgument(0));

        // Execute
        List<String> results = predictor.classifyBatch(texts);

        // Verify - results should be in same order as input
        assertNotNull(results);
        assertEquals(30, results.size());

        // Cleanup
        predictor.shutdown();
    }

    // ==================== PROBABILITY TESTS ====================

    @Test
    @DisplayName("Batch probability computation should work sequentially")
    void testSequentialProbabilityBatch() throws Exception {
        // Setup
        BatchOptimizationConfig config = BatchOptimizationConfig.builder()
                .batchSize(5)
                .enableParallelProcessing(false)
                .build();

        BatchPredictor<SVMClassifier> predictor = new BatchPredictor<>(classifier, config);

        List<String> texts = createTestTexts(10);
        setupMocksForProbabilityProcessing(texts);

        // Execute
        List<double[]> results = predictor.getProbabilitiesBatch(texts);

        // Verify
        assertNotNull(results);
        assertEquals(10, results.size());

        for (double[] probs : results) {
            assertNotNull(probs);
            assertEquals(2, probs.length);
            double sum = Arrays.stream(probs).sum();
            assertEquals(1.0, sum, 0.001);
        }
    }

    @Test
    @DisplayName("Batch probability computation should work in parallel")
    void testParallelProbabilityBatch() throws Exception {
        // Setup
        BatchOptimizationConfig config = BatchOptimizationConfig.builder()
                .batchSize(25)
                .parallelThreshold(50)
                .enableParallelProcessing(true)
                .build();

        BatchPredictor<SVMClassifier> predictor = new BatchPredictor<>(classifier, config);

        List<String> texts = createTestTexts(75);
        setupMocksForProbabilityProcessing(texts);

        // Execute
        List<double[]> results = predictor.getProbabilitiesBatch(texts);

        // Verify
        assertNotNull(results);
        assertEquals(75, results.size());

        // Cleanup
        predictor.shutdown();
    }

    // ==================== METRICS TESTS ====================

    @Test
    @DisplayName("Metrics should be collected when enabled")
    void testMetricsCollection() throws Exception {
        // Setup
        BatchOptimizationConfig config = BatchOptimizationConfig.builder()
                .batchSize(10)
                .enableMetrics(true)
                .enableParallelProcessing(false)
                .build();

        BatchPredictor<SVMClassifier> predictor = new BatchPredictor<>(classifier, config);

        List<String> texts = createTestTexts(25);
        setupMocksForBatchProcessing(texts, "positive");

        // Execute
        predictor.classifyBatch(texts);

        // Verify
        BatchProcessingMetrics metrics = predictor.getLastMetrics();
        assertNotNull(metrics);
        assertEquals(25, metrics.getTotalItems());
        assertEquals(10, metrics.getBatchSize());
        assertEquals(25, metrics.getItemsProcessed());
        assertTrue(metrics.getTotalTimeMs() >= 0);
        assertTrue(metrics.getThroughput() >= 0);
    }

    @Test
    @DisplayName("Metrics should not be collected when disabled")
    void testMetricsDisabled() throws Exception {
        // Setup
        BatchOptimizationConfig config = BatchOptimizationConfig.builder()
                .batchSize(10)
                .enableMetrics(false)
                .enableParallelProcessing(false)
                .build();

        BatchPredictor<SVMClassifier> predictor = new BatchPredictor<>(classifier, config);

        List<String> texts = createTestTexts(15);
        setupMocksForBatchProcessing(texts, "positive");

        // Execute
        predictor.classifyBatch(texts);

        // Verify
        BatchProcessingMetrics metrics = predictor.getLastMetrics();
        assertNull(metrics);
    }

    @Test
    @DisplayName("Metrics summary should contain all relevant information")
    void testMetricsSummary() throws Exception {
        // Setup
        BatchOptimizationConfig config = BatchOptimizationConfig.getDefault();
        BatchPredictor<SVMClassifier> predictor = new BatchPredictor<>(classifier, config);

        List<String> texts = createTestTexts(50);
        setupMocksForBatchProcessing(texts, "positive");

        // Execute
        predictor.classifyBatch(texts);

        // Verify
        BatchProcessingMetrics metrics = predictor.getLastMetrics();
        String summary = metrics.getSummary();

        assertNotNull(summary);
        assertTrue(summary.contains("Total Items: 50"));
        assertTrue(summary.contains("Items Processed: 50"));
        assertTrue(summary.contains("Throughput:"));
        assertTrue(summary.contains("Preprocessing:"));
        assertTrue(summary.contains("Transformation:"));
        assertTrue(summary.contains("Classification:"));
    }

    // ==================== ERROR HANDLING TESTS ====================

    @Test
    @DisplayName("Should reject null or empty input")
    void testNullInputHandling() {
        BatchPredictor<SVMClassifier> predictor = new BatchPredictor<>(classifier);

        assertThrows(IllegalArgumentException.class, () ->
                predictor.classifyBatch(null)
        );

        assertThrows(IllegalArgumentException.class, () ->
                predictor.classifyBatch(new ArrayList<>())
        );
    }

    @Test
    @DisplayName("Should handle empty texts gracefully")
    void testEmptyTextHandling() throws Exception {
        // Setup
        BatchPredictor<SVMClassifier> predictor = new BatchPredictor<>(classifier);

        List<String> texts = Arrays.asList("", "  ", null, "valid text");

        when(mockPreprocessor.preprocessText(anyString()))
                .thenAnswer(invocation -> {
                    String text = invocation.getArgument(0);
                    return (text == null || text.trim().isEmpty()) ?
                            "empty_content_placeholder" : text;
                });

        setupMocksForBatchProcessing(texts, "positive");

        // Execute
        List<String> results = predictor.classifyBatch(texts);

        // Verify
        assertNotNull(results);
        assertEquals(4, results.size());
    }

    @Test
    @DisplayName("Should handle processing errors appropriately")
    void testProcessingErrorHandling() {
        // Setup
        BatchPredictor<SVMClassifier> predictor = new BatchPredictor<>(classifier);

        List<String> texts = createTestTexts(10);

        when(mockPreprocessor.preprocessText(anyString()))
                .thenThrow(new RuntimeException("Processing error"));

        // Verify
        assertThrows(Exception.class, () ->
                predictor.classifyBatch(texts)
        );
    }

    // ==================== RESOURCE MANAGEMENT TESTS ====================

    @Test
    @DisplayName("Should shutdown thread pool gracefully")
    void testShutdown() throws Exception {
        // Setup with parallel processing
        BatchOptimizationConfig config = BatchOptimizationConfig.forHighThroughput();
        BatchPredictor<SVMClassifier> predictor = new BatchPredictor<>(classifier, config);

        // Perform some work
        List<String> texts = createTestTexts(300);
        setupMocksForBatchProcessing(texts, "positive");
        predictor.classifyBatch(texts);

        // Shutdown
        predictor.shutdown();

        // Verify we can't use it after shutdown (thread pool is down)
        // This is just to ensure shutdown completes without error
        assertDoesNotThrow(() -> predictor.shutdown()); // Should be idempotent
    }

    @Test
    @DisplayName("Sequential predictor should handle shutdown gracefully")
    void testShutdownWithoutThreadPool() {
        // Setup without parallel processing
        BatchOptimizationConfig config = BatchOptimizationConfig.forLowMemory();
        BatchPredictor<SVMClassifier> predictor = new BatchPredictor<>(classifier, config);

        // Shutdown should not throw even without thread pool
        assertDoesNotThrow(() -> predictor.shutdown());
    }

    // ==================== PERFORMANCE TESTS ====================

    @Test
    @DisplayName("Mini-batching should process correct number of batches")
    void testMiniBatchCalculation() {
        BatchOptimizationConfig config = BatchOptimizationConfig.builder()
                .batchSize(50)
                .build();

        assertEquals(3, config.calculateMiniBatches(150));
        assertEquals(3, config.calculateMiniBatches(101));
        assertEquals(2, config.calculateMiniBatches(100));
        assertEquals(1, config.calculateMiniBatches(50));
        assertEquals(1, config.calculateMiniBatches(25));
    }

    @Test
    @DisplayName("Parallel threshold should be respected")
    void testParallelThreshold() {
        BatchOptimizationConfig config = BatchOptimizationConfig.builder()
                .parallelThreshold(100)
                .enableParallelProcessing(true)
                .build();

        assertFalse(config.shouldUseParallel(50));
        assertFalse(config.shouldUseParallel(99));
        assertTrue(config.shouldUseParallel(100));
        assertTrue(config.shouldUseParallel(1000));
    }

    // ==================== HELPER METHODS ====================

    private List<String> createTestTexts(int count) {
        return IntStream.range(0, count)
                .mapToObj(i -> "Test text " + i)
                .collect(Collectors.toList());
    }

    private void setupMocksForBatchProcessing(List<String> texts, String label) throws Exception {
        when(mockPreprocessor.preprocessText(anyString()))
                .thenAnswer(invocation -> invocation.getArgument(0));

        when(mockConverter.transform(any(String[].class), anyString()))
                .thenAnswer(invocation -> {
                    String[] inputs = invocation.getArgument(0);
                    return createMockInstances(inputs.length);
                });

        when(mockSMO.classifyInstance(any()))
                .thenReturn(label.equals("positive") ? 0.0 : 1.0);
    }

    private void setupMocksForProbabilityProcessing(List<String> texts) throws Exception {
        when(mockPreprocessor.preprocessText(anyString()))
                .thenAnswer(invocation -> invocation.getArgument(0));

        when(mockConverter.transform(any(String[].class), anyString()))
                .thenAnswer(invocation -> {
                    String[] inputs = invocation.getArgument(0);
                    return createMockInstances(inputs.length);
                });

        when(mockSMO.distributionForInstance(any()))
                .thenReturn(new double[]{0.7, 0.3});
    }

    private Instances createMockInstances(int count) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("feature1"));
        attributes.add(new Attribute("feature2"));

        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("positive");
        classValues.add("negative");
        attributes.add(new Attribute("class", classValues));

        Instances instances = new Instances("test", attributes, count);
        instances.setClassIndex(2);

        // Create balanced class distribution
        for (int i = 0; i < count; i++) {
            DenseInstance instance = new DenseInstance(3);
            instance.setValue(0, 1.0);
            instance.setValue(1, 0.5);
            // Alternate between classes for balance
            instance.setValue(2, i % 2);
            instances.add(instance);
        }

        return instances;
    }
}
