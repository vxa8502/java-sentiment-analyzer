package sentiment.evaluation;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import sentiment.data.Dataset;
import sentiment.evaluation.domain.FeatureImportanceResult;
import sentiment.evaluation.domain.FeatureStatistics;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for FeatureImportanceAnalyzer.
 *
 * These tests validate that the analyzer correctly:
 * 1. Extracts feature weights from trained SVM models
 * 2. Ranks features by discriminative power
 * 3. Computes meaningful statistics
 * 4. Handles edge cases gracefully
 */
class FeatureImportanceAnalyzerTest {

    private FeatureImportanceAnalyzer analyzer;

    @BeforeEach
    void setUp() {
        analyzer = new FeatureImportanceAnalyzer();
    }

    /**
     * Creates a simple training instances for testing.
     * This bypasses the need for full preprocessing pipeline.
     */
    private Instances createSimpleInstances() {
        // Create attributes
        ArrayList<weka.core.Attribute> attributes = new ArrayList<>();

        // Add a few word features
        attributes.add(new weka.core.Attribute("excellent"));
        attributes.add(new weka.core.Attribute("terrible"));
        attributes.add(new weka.core.Attribute("good"));
        attributes.add(new weka.core.Attribute("bad"));

        // Add class attribute
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("positive");
        classValues.add("negative");
        attributes.add(new weka.core.Attribute("sentiment", classValues));

        // Create dataset
        Instances instances = new Instances("TestData", attributes, 0);
        instances.setClassIndex(attributes.size() - 1);

        // Add some instances
        addInstance(instances, new double[]{1.0, 0.0, 0.5, 0.0, 0}); // positive
        addInstance(instances, new double[]{0.0, 1.0, 0.0, 0.8, 1}); // negative
        addInstance(instances, new double[]{0.9, 0.0, 0.7, 0.0, 0}); // positive
        addInstance(instances, new double[]{0.0, 0.9, 0.0, 1.0, 1}); // negative

        return instances;
    }

    private void addInstance(Instances dataset, double[] values) {
        weka.core.Instance inst = new weka.core.DenseInstance(1.0, values);
        inst.setDataset(dataset);
        dataset.add(inst);
    }

    @Test
    @DisplayName("Should analyze feature importance for trained SVM model")
    void testAnalyzeFeatureImportance() throws Exception {
        // Create simple training data with clear sentiment signals
        Instances trainedInstances = createSimpleInstances();

        // Train a simple SVM
        SMO smo = new SMO();
        smo.buildClassifier(trainedInstances);

        // Analyze feature importance
        FeatureImportanceResult result =
                analyzer.analyzeFeatureImportance(trainedInstances, smo, 10);

        // Validate results
        assertNotNull(result, "Result should not be null");
        assertNotNull(result.topFeatures(), "Top features should not be null");
        assertNotNull(result.allFeatures(), "All features should not be null");
        assertNotNull(result.statistics(), "Statistics should not be null");

        assertTrue(result.topFeatures().size() > 0, "Should have at least one feature");
        assertTrue(result.topFeatures().size() <= 10, "Should not exceed requested top-k");
        assertTrue(result.analysisTimeMs() >= 0, "Analysis time should be non-negative");

        // Validate that features are ranked by absolute weight (descending)
        for (int i = 0; i < result.topFeatures().size() - 1; i++) {
            double currentWeight = Math.abs(result.topFeatures().get(i).weight());
            double nextWeight = Math.abs(result.topFeatures().get(i + 1).weight());
            assertTrue(currentWeight >= nextWeight,
                    "Features should be ranked by descending absolute weight");
        }
    }

    @Test
    @DisplayName("Should compute meaningful statistics")
    void testStatisticsComputation() throws Exception {
        Instances trainedInstances = createSimpleInstances();

        SMO smo = new SMO();
        smo.buildClassifier(trainedInstances);

        FeatureImportanceResult result =
                analyzer.analyzeFeatureImportance(trainedInstances, smo, 50);

        FeatureStatistics stats = result.statistics();

        // Validate statistics are sensible
        assertTrue(stats.mean() >= 0, "Mean should be non-negative");
        assertTrue(stats.stdDev() >= 0, "StdDev should be non-negative");
        assertTrue(stats.median() >= 0, "Median should be non-negative");
        assertTrue(stats.percentile95() >= stats.median(), "95th percentile should be >= median");
        assertTrue(stats.totalFeatures() > 0, "Should have positive feature count");

        // Mean should be reasonably related to median for feature weights
        assertTrue(stats.mean() >= 0 && stats.median() >= 0,
                "Mean and median should both be non-negative");
    }

    @Test
    @DisplayName("Should handle top-k parameter correctly")
    void testTopKParameter() throws Exception {
        Instances trainedInstances = createSimpleInstances();

        SMO smo = new SMO();
        smo.buildClassifier(trainedInstances);

        // Test with different top-k values
        int[] topKValues = {5, 10, 20};

        for (int topK : topKValues) {
            FeatureImportanceResult result =
                    analyzer.analyzeFeatureImportance(trainedInstances, smo, topK);

            assertTrue(result.topFeatures().size() <= topK,
                    "Should not return more than " + topK + " features");
            assertTrue(result.topFeatures().size() > 0,
                    "Should return at least one feature");
        }
    }

    @Test
    @DisplayName("Should reject null or invalid inputs")
    void testInputValidation() throws Exception {
        assertThrows(IllegalArgumentException.class, () ->
                analyzer.analyzeFeatureImportance(null, new SMO(), 10),
                "Should reject null training data");

        // Create minimal valid instances
        Instances trainedInstances = createSimpleInstances();

        assertThrows(IllegalArgumentException.class, () ->
                analyzer.analyzeFeatureImportance(trainedInstances, null, 10),
                "Should reject null SVM");

        SMO smo = new SMO();
        assertThrows(IllegalArgumentException.class, () ->
                analyzer.analyzeFeatureImportance(trainedInstances, smo, 0),
                "Should reject topK <= 0");

        assertThrows(IllegalArgumentException.class, () ->
                analyzer.analyzeFeatureImportance(trainedInstances, smo, -5),
                "Should reject negative topK");
    }

    @Test
    @DisplayName("Should produce FeatureWeight objects with valid properties")
    void testFeatureWeightProperties() throws Exception {
        Instances trainedInstances = createSimpleInstances();

        SMO smo = new SMO();
        smo.buildClassifier(trainedInstances);

        FeatureImportanceResult result =
                analyzer.analyzeFeatureImportance(trainedInstances, smo, 10);

        for (sentiment.evaluation.domain.FeatureWeight fw : result.topFeatures()) {
            assertNotNull(fw.featureName(), "Feature name should not be null");
            assertFalse(fw.featureName().trim().isEmpty(), "Feature name should not be empty");
            assertTrue(Double.isFinite(fw.weight()), "Weight should be finite");
            assertTrue(Double.isFinite(fw.significance()), "Significance should be finite");
            assertTrue(fw.significance() >= 0, "Significance should be non-negative");
        }
    }

    @Test
    @DisplayName("Should handle models with many features")
    void testLargeFeatureSet() throws Exception {
        // Create dataset with many features
        Instances trainedInstances = createLargeInstances();

        SMO smo = new SMO();
        smo.buildClassifier(trainedInstances);

        // Request top 100 features
        FeatureImportanceResult result =
                analyzer.analyzeFeatureImportance(trainedInstances, smo, 100);

        assertTrue(result.allFeatures().size() > 0, "Should have features");
        assertTrue(result.topFeatures().size() <= 100, "Should respect top-k limit");

        // Verify all features are included in allFeatures
        assertEquals(result.allFeatures().size(), result.statistics().totalFeatures(),
                "Total features should match allFeatures size");
    }

    @Test
    @DisplayName("Should identify discriminative features for sentiment")
    void testDiscriminativeFeatures() throws Exception {
        Instances trainedInstances = createSimpleInstances();

        SMO smo = new SMO();
        smo.buildClassifier(trainedInstances);

        FeatureImportanceResult result =
                analyzer.analyzeFeatureImportance(trainedInstances, smo, 10);

        // Check that the features extracted are from our defined features
        List<String> topFeatureNames = result.topFeatures().stream()
                .map(fw -> fw.featureName().toLowerCase())
                .toList();

        // At least some features should be from our test data
        boolean hasKnownFeature = topFeatureNames.stream()
                .anyMatch(name -> name.equals("excellent") || name.equals("terrible") ||
                        name.equals("good") || name.equals("bad"));

        assertTrue(hasKnownFeature,
                "Top features should include features from training data");
    }

    @Test
    @DisplayName("Should handle multi-class classification (3 classes)")
    void testMultiClassClassification() throws Exception {
        // Create instances with 3 classes: positive, negative, neutral
        Instances multiClassInstances = createMultiClassInstances();

        SMO smo = new SMO();
        smo.buildClassifier(multiClassInstances);

        // Analyze feature importance - should not throw exception
        FeatureImportanceResult result =
                analyzer.analyzeFeatureImportance(multiClassInstances, smo, 10);

        // Validate results
        assertNotNull(result, "Result should not be null");
        assertNotNull(result.topFeatures(), "Top features should not be null");
        assertTrue(result.topFeatures().size() > 0, "Should have at least one feature");
        assertTrue(result.analysisTimeMs() >= 0, "Analysis time should be non-negative");

        // Verify feature weights are valid
        for (sentiment.evaluation.domain.FeatureWeight fw : result.topFeatures()) {
            assertTrue(Double.isFinite(fw.weight()), "Weight should be finite");
            assertTrue(Double.isFinite(fw.significance()), "Significance should be finite");
        }
    }

    private Instances createMultiClassInstances() {
        // Create attributes
        ArrayList<weka.core.Attribute> attributes = new ArrayList<>();

        // Add word features
        attributes.add(new weka.core.Attribute("excellent"));
        attributes.add(new weka.core.Attribute("terrible"));
        attributes.add(new weka.core.Attribute("okay"));
        attributes.add(new weka.core.Attribute("average"));

        // Add class attribute with 3 values
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("positive");
        classValues.add("negative");
        classValues.add("neutral");
        attributes.add(new weka.core.Attribute("sentiment", classValues));

        // Create dataset
        Instances instances = new Instances("MultiClassTestData", attributes, 0);
        instances.setClassIndex(attributes.size() - 1);

        // Add instances for each class
        addInstance(instances, new double[]{1.0, 0.0, 0.0, 0.0, 0}); // positive
        addInstance(instances, new double[]{0.0, 1.0, 0.0, 0.0, 1}); // negative
        addInstance(instances, new double[]{0.0, 0.0, 1.0, 0.8, 2}); // neutral
        addInstance(instances, new double[]{0.9, 0.0, 0.0, 0.0, 0}); // positive
        addInstance(instances, new double[]{0.0, 0.9, 0.0, 0.0, 1}); // negative
        addInstance(instances, new double[]{0.0, 0.0, 0.8, 1.0, 2}); // neutral

        return instances;
    }

    // ==================== HELPER METHODS ====================

    private Instances createLargeInstances() {
        // Create attributes with many features
        ArrayList<weka.core.Attribute> attributes = new ArrayList<>();

        // Add 50 word features
        for (int i = 0; i < 50; i++) {
            attributes.add(new weka.core.Attribute("feature_" + i));
        }

        // Add class attribute
        ArrayList<String> classValues = new ArrayList<>();
        classValues.add("positive");
        classValues.add("negative");
        attributes.add(new weka.core.Attribute("sentiment", classValues));

        // Create dataset
        Instances instances = new Instances("LargeTestData", attributes, 0);
        instances.setClassIndex(attributes.size() - 1);

        // Add 20 instances with random feature values
        java.util.Random rand = new java.util.Random(42);
        for (int i = 0; i < 20; i++) {
            double[] values = new double[51];
            for (int j = 0; j < 50; j++) {
                values[j] = rand.nextDouble();
            }
            values[50] = i % 2; // alternate between positive/negative
            addInstance(instances, values);
        }

        return instances;
    }
}
