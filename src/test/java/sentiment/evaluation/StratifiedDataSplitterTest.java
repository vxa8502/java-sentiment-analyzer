package sentiment.evaluation;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import sentiment.data.Dataset;
import sentiment.evaluation.StratifiedDataSplitter.DataSplit;

import java.util.*;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive unit tests for StratifiedDataSplitter.
 * <p>
 * Tests cover:
 * - Stratified train/validation/test splitting
 * - K-fold cross-validation
 * - Class distribution preservation
 * - Edge cases and validation
 * - Reproducibility
 */
@DisplayName("StratifiedDataSplitter Unit Tests")
class StratifiedDataSplitterTest {

    private static final double EPSILON = 1e-6;
    private static final long SEED = 42L;

    private List<Dataset> balancedData;
    private List<Dataset> imbalancedData;

    @BeforeEach
    void setUp() {
        // Create balanced dataset: 50 positive, 50 negative
        balancedData = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            balancedData.add(new Dataset("Positive text " + i, Dataset.SentimentLabel.POSITIVE, 0.9));
            balancedData.add(new Dataset("Negative text " + i, Dataset.SentimentLabel.NEGATIVE, 0.9));
        }

        // Create imbalanced dataset: 80 positive, 20 negative
        imbalancedData = new ArrayList<>();
        for (int i = 0; i < 80; i++) {
            imbalancedData.add(new Dataset("Positive text " + i, Dataset.SentimentLabel.POSITIVE, 0.9));
        }
        for (int i = 0; i < 20; i++) {
            imbalancedData.add(new Dataset("Negative text " + i, Dataset.SentimentLabel.NEGATIVE, 0.9));
        }
    }

    // ==================== THREE-WAY SPLIT TESTS ====================

    @Test
    @DisplayName("Three-way split should create correct proportions")
    void testThreeWaySplit_Proportions() {
        DataSplit split = StratifiedDataSplitter.stratifiedSplit(
                balancedData, 0.6, 0.2, 0.2, SEED);

        int total = balancedData.size();
        int expectedTrain = (int) (total * 0.6);
        int expectedVal = (int) (total * 0.2);
        int expectedTest = (int) (total * 0.2);

        // Allow small rounding differences
        assertEquals(expectedTrain, split.train.size(), 2);
        assertEquals(expectedVal, split.validation.size(), 2);
        assertEquals(expectedTest, split.test.size(), 2);

        // Total should match
        assertEquals(total, split.totalSize());
    }

    @Test
    @DisplayName("Three-way split should preserve class distribution")
    void testThreeWaySplit_PreservesDistribution() {
        DataSplit split = StratifiedDataSplitter.stratifiedSplit(
                balancedData, 0.6, 0.2, 0.2, SEED);

        // Check train set class balance
        double trainPosRatio = getClassRatio(split.train, Dataset.SentimentLabel.POSITIVE);
        assertEquals(0.5, trainPosRatio, 0.05);  // Should be ~50%

        // Check validation set class balance
        double valPosRatio = getClassRatio(split.validation, Dataset.SentimentLabel.POSITIVE);
        assertEquals(0.5, valPosRatio, 0.05);

        // Check test set class balance
        double testPosRatio = getClassRatio(split.test, Dataset.SentimentLabel.POSITIVE);
        assertEquals(0.5, testPosRatio, 0.05);
    }

    @Test
    @DisplayName("Three-way split should preserve imbalanced distribution")
    void testThreeWaySplit_PreservesImbalance() {
        DataSplit split = StratifiedDataSplitter.stratifiedSplit(
                imbalancedData, 0.6, 0.2, 0.2, SEED);

        // Original: 80% positive, 20% negative
        // All splits should maintain roughly this ratio
        double trainPosRatio = getClassRatio(split.train, Dataset.SentimentLabel.POSITIVE);
        double valPosRatio = getClassRatio(split.validation, Dataset.SentimentLabel.POSITIVE);
        double testPosRatio = getClassRatio(split.test, Dataset.SentimentLabel.POSITIVE);

        assertEquals(0.8, trainPosRatio, 0.1);
        assertEquals(0.8, valPosRatio, 0.1);
        assertEquals(0.8, testPosRatio, 0.1);
    }

    @Test
    @DisplayName("Three-way split should have no data leakage")
    void testThreeWaySplit_NoLeakage() {
        DataSplit split = StratifiedDataSplitter.stratifiedSplit(
                balancedData, 0.6, 0.2, 0.2, SEED);

        // Extract IDs
        Set<String> trainIds = split.train.stream().map(Dataset::getId).collect(Collectors.toSet());
        Set<String> valIds = split.validation.stream().map(Dataset::getId).collect(Collectors.toSet());
        Set<String> testIds = split.test.stream().map(Dataset::getId).collect(Collectors.toSet());

        // No overlap
        assertTrue(Collections.disjoint(trainIds, valIds));
        assertTrue(Collections.disjoint(trainIds, testIds));
        assertTrue(Collections.disjoint(valIds, testIds));
    }

    @Test
    @DisplayName("Three-way split should be reproducible with same seed")
    void testThreeWaySplit_Reproducibility() {
        DataSplit split1 = StratifiedDataSplitter.stratifiedSplit(
                balancedData, 0.6, 0.2, 0.2, SEED);

        DataSplit split2 = StratifiedDataSplitter.stratifiedSplit(
                balancedData, 0.6, 0.2, 0.2, SEED);

        // Should produce same split
        assertEquals(split1.train.size(), split2.train.size());
        assertEquals(split1.validation.size(), split2.validation.size());
        assertEquals(split1.test.size(), split2.test.size());

        // Check that at least some IDs match (random shuffling within classes might differ)
        Set<String> trainIds1 = split1.train.stream().map(Dataset::getId).collect(Collectors.toSet());
        Set<String> trainIds2 = split2.train.stream().map(Dataset::getId).collect(Collectors.toSet());
        assertEquals(trainIds1, trainIds2);
    }

    // ==================== TWO-WAY SPLIT TESTS ====================

    @Test
    @DisplayName("Two-way split should create correct proportions")
    void testTwoWaySplit_Proportions() {
        DataSplit split = StratifiedDataSplitter.stratifiedSplit(
                balancedData, 0.8, SEED);

        int total = balancedData.size();
        int expectedTrain = (int) (total * 0.8);
        int expectedTest = total - expectedTrain;

        assertEquals(expectedTrain, split.train.size(), 2);
        assertEquals(0, split.validation.size());  // Should be empty
        assertEquals(expectedTest, split.test.size(), 2);
    }

    @Test
    @DisplayName("Two-way split should preserve class distribution")
    void testTwoWaySplit_PreservesDistribution() {
        DataSplit split = StratifiedDataSplitter.stratifiedSplit(
                balancedData, 0.8, SEED);

        double trainPosRatio = getClassRatio(split.train, Dataset.SentimentLabel.POSITIVE);
        double testPosRatio = getClassRatio(split.test, Dataset.SentimentLabel.POSITIVE);

        assertEquals(0.5, trainPosRatio, 0.05);
        assertEquals(0.5, testPosRatio, 0.05);
    }

    // ==================== K-FOLD TESTS ====================

    @Test
    @DisplayName("K-fold should create k splits")
    void testKFold_CreatesSplits() {
        int k = 5;
        List<DataSplit> folds = StratifiedDataSplitter.stratifiedKFold(balancedData, k, SEED);

        assertEquals(k, folds.size());
    }

    @Test
    @DisplayName("K-fold splits should have roughly equal test set sizes")
    void testKFold_EqualTestSizes() {
        int k = 5;
        List<DataSplit> folds = StratifiedDataSplitter.stratifiedKFold(balancedData, k, SEED);

        int expectedTestSize = balancedData.size() / k;

        for (DataSplit fold : folds) {
            assertEquals(expectedTestSize, fold.test.size(), 5);  // Allow small variation
            assertEquals(0, fold.validation.size());  // K-fold doesn't use validation
        }
    }

    @Test
    @DisplayName("K-fold should preserve class distribution in each fold")
    void testKFold_PreservesDistribution() {
        int k = 5;
        List<DataSplit> folds = StratifiedDataSplitter.stratifiedKFold(balancedData, k, SEED);

        for (DataSplit fold : folds) {
            double trainPosRatio = getClassRatio(fold.train, Dataset.SentimentLabel.POSITIVE);
            double testPosRatio = getClassRatio(fold.test, Dataset.SentimentLabel.POSITIVE);

            // Should maintain 50-50 balance
            assertEquals(0.5, trainPosRatio, 0.1);
            assertEquals(0.5, testPosRatio, 0.1);
        }
    }

    @Test
    @DisplayName("K-fold should cover all data exactly once in test sets")
    void testKFold_CoversAllData() {
        int k = 5;
        List<DataSplit> folds = StratifiedDataSplitter.stratifiedKFold(balancedData, k, SEED);

        Set<String> allTestIds = new HashSet<>();
        for (DataSplit fold : folds) {
            Set<String> testIds = fold.test.stream().map(Dataset::getId).collect(Collectors.toSet());

            // No overlap between test sets
            assertTrue(Collections.disjoint(allTestIds, testIds));
            allTestIds.addAll(testIds);
        }

        // All data should be covered
        assertEquals(balancedData.size(), allTestIds.size());
    }

    @Test
    @DisplayName("K-fold should have no data leakage within folds")
    void testKFold_NoLeakage() {
        int k = 5;
        List<DataSplit> folds = StratifiedDataSplitter.stratifiedKFold(balancedData, k, SEED);

        for (DataSplit fold : folds) {
            Set<String> trainIds = fold.train.stream().map(Dataset::getId).collect(Collectors.toSet());
            Set<String> testIds = fold.test.stream().map(Dataset::getId).collect(Collectors.toSet());

            // Train and test should not overlap
            assertTrue(Collections.disjoint(trainIds, testIds));
        }
    }

    @Test
    @DisplayName("K-fold should work with k=10")
    void testKFold_TenFold() {
        int k = 10;
        List<DataSplit> folds = StratifiedDataSplitter.stratifiedKFold(balancedData, k, SEED);

        assertEquals(k, folds.size());

        int expectedTestSize = balancedData.size() / k;
        for (DataSplit fold : folds) {
            assertEquals(expectedTestSize, fold.test.size(), 3);
        }
    }

    // ==================== VALIDATION TESTS ====================

    @Test
    @DisplayName("Should throw when data is null")
    void testValidation_NullData() {
        assertThrows(IllegalArgumentException.class,
                () -> StratifiedDataSplitter.stratifiedSplit(null, 0.6, 0.2, 0.2, SEED));
    }

    @Test
    @DisplayName("Should throw when data is empty")
    void testValidation_EmptyData() {
        assertThrows(IllegalArgumentException.class,
                () -> StratifiedDataSplitter.stratifiedSplit(List.of(), 0.6, 0.2, 0.2, SEED));
    }

    @Test
    @DisplayName("Should throw when ratios don't sum to 1.0")
    void testValidation_InvalidRatioSum() {
        assertThrows(IllegalArgumentException.class,
                () -> StratifiedDataSplitter.stratifiedSplit(balancedData, 0.5, 0.3, 0.1, SEED));  // Sum = 0.9

        assertThrows(IllegalArgumentException.class,
                () -> StratifiedDataSplitter.stratifiedSplit(balancedData, 0.6, 0.3, 0.2, SEED));  // Sum = 1.1
    }

    @Test
    @DisplayName("Should throw when ratios are negative")
    void testValidation_NegativeRatios() {
        assertThrows(IllegalArgumentException.class,
                () -> StratifiedDataSplitter.stratifiedSplit(balancedData, -0.1, 0.6, 0.5, SEED));
    }

    @Test
    @DisplayName("Should throw when trainRatio is zero")
    void testValidation_ZeroTrainRatio() {
        assertThrows(IllegalArgumentException.class,
                () -> StratifiedDataSplitter.stratifiedSplit(balancedData, 0.0, 0.5, 0.5, SEED));
    }

    @Test
    @DisplayName("Should throw when k < 2 for k-fold")
    void testValidation_InvalidK() {
        assertThrows(IllegalArgumentException.class,
                () -> StratifiedDataSplitter.stratifiedKFold(balancedData, 1, SEED));

        assertThrows(IllegalArgumentException.class,
                () -> StratifiedDataSplitter.stratifiedKFold(balancedData, 0, SEED));
    }

    // ==================== EDGE CASES ====================

    @Test
    @DisplayName("Should handle small dataset")
    void testEdgeCase_SmallDataset() {
        List<Dataset> smallData = new ArrayList<>();
        smallData.add(new Dataset("Positive 1", Dataset.SentimentLabel.POSITIVE, 0.9));
        smallData.add(new Dataset("Negative 1", Dataset.SentimentLabel.NEGATIVE, 0.9));
        smallData.add(new Dataset("Positive 2", Dataset.SentimentLabel.POSITIVE, 0.9));
        smallData.add(new Dataset("Negative 2", Dataset.SentimentLabel.NEGATIVE, 0.9));

        DataSplit split = StratifiedDataSplitter.stratifiedSplit(
                smallData, 0.5, 0.25, 0.25, SEED);

        assertEquals(4, split.totalSize());
        assertFalse(split.train.isEmpty());
        assertFalse(split.test.isEmpty());
    }

    @Test
    @DisplayName("Should handle single class dataset")
    void testEdgeCase_SingleClass() {
        List<Dataset> singleClass = new ArrayList<>();
        for (int i = 0; i < 20; i++) {
            singleClass.add(new Dataset("Positive " + i, Dataset.SentimentLabel.POSITIVE, 0.9));
        }

        DataSplit split = StratifiedDataSplitter.stratifiedSplit(
                singleClass, 0.6, 0.2, 0.2, SEED);

        // Should still work
        assertEquals(20, split.totalSize());
        assertEquals(1.0, getClassRatio(split.train, Dataset.SentimentLabel.POSITIVE), EPSILON);
        assertEquals(1.0, getClassRatio(split.test, Dataset.SentimentLabel.POSITIVE), EPSILON);
    }

    @Test
    @DisplayName("Should handle three-class dataset")
    void testEdgeCase_ThreeClasses() {
        List<Dataset> threeClass = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            threeClass.add(new Dataset("Positive " + i, Dataset.SentimentLabel.POSITIVE, 0.9));
            threeClass.add(new Dataset("Negative " + i, Dataset.SentimentLabel.NEGATIVE, 0.9));
            threeClass.add(new Dataset("Neutral " + i, Dataset.SentimentLabel.NEUTRAL, 0.9));
        }

        DataSplit split = StratifiedDataSplitter.stratifiedSplit(
                threeClass, 0.6, 0.2, 0.2, SEED);

        // All three classes should be represented
        double trainPosRatio = getClassRatio(split.train, Dataset.SentimentLabel.POSITIVE);
        double trainNegRatio = getClassRatio(split.train, Dataset.SentimentLabel.NEGATIVE);
        double trainNeutRatio = getClassRatio(split.train, Dataset.SentimentLabel.NEUTRAL);

        // Each class should be ~1/3
        assertEquals(1.0 / 3.0, trainPosRatio, 0.1);
        assertEquals(1.0 / 3.0, trainNegRatio, 0.1);
        assertEquals(1.0 / 3.0, trainNeutRatio, 0.1);
    }

    @Test
    @DisplayName("Should allow validation ratio of zero")
    void testEdgeCase_ZeroValidationRatio() {
        DataSplit split = StratifiedDataSplitter.stratifiedSplit(
                balancedData, 0.8, 0.0, 0.2, SEED);

        assertEquals(0, split.validation.size());
        assertFalse(split.train.isEmpty());
        assertFalse(split.test.isEmpty());
    }

    // ==================== IMMUTABILITY TESTS ====================

    @Test
    @DisplayName("DataSplit lists should be immutable")
    @SuppressWarnings("DataFlowIssue") // Intentionally testing immutability violations
    void testDataSplit_Immutability() {
        DataSplit split = StratifiedDataSplitter.stratifiedSplit(
                balancedData, 0.6, 0.2, 0.2, SEED);

        // Should throw when trying to modify
        assertThrows(UnsupportedOperationException.class,
                () -> split.train.add(new Dataset("New", Dataset.SentimentLabel.POSITIVE, 0.9)));

        assertThrows(UnsupportedOperationException.class,
                () -> split.validation.add(new Dataset("New", Dataset.SentimentLabel.POSITIVE, 0.9)));

        assertThrows(UnsupportedOperationException.class,
                () -> split.test.add(new Dataset("New", Dataset.SentimentLabel.POSITIVE, 0.9)));
    }

    // ==================== TO STRING TESTS ====================

    @Test
    @DisplayName("DataSplit toString should be formatted correctly")
    void testDataSplit_ToString() {
        DataSplit split = StratifiedDataSplitter.stratifiedSplit(
                balancedData, 0.6, 0.2, 0.2, SEED);

        String str = split.toString();
        assertTrue(str.contains("train"));
        assertTrue(str.contains("val"));
        assertTrue(str.contains("test"));
        assertTrue(str.contains("total"));
    }

    // ==================== HELPER METHODS ====================

    /**
     * Calculate the ratio of a specific class in a dataset
     */
    private double getClassRatio(List<Dataset> data, Dataset.SentimentLabel targetClass) {
        if (data.isEmpty()) {
            return 0.0;
        }

        long count = data.stream()
                .filter(d -> d.getSentiment() == targetClass)
                .count();

        return (double) count / data.size();
    }
}
