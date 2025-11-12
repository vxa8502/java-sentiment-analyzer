package sentiment.evaluation;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive unit tests for ClassifierEvaluationResult.
 *
 * Tests cover:
 * - Constructor validation
 * - Getter methods
 * - Per-class metric accessors
 * - Immutability guarantees
 * - Edge cases
 * - String formatting
 */
@DisplayName("ClassifierEvaluationResult Unit Tests")
class ClassifierEvaluationResultTest {

    private ClassifierEvaluationResult basicResult;
    private ClassifierEvaluationResult advancedResult;

    @BeforeEach
    void setUp() {
        // Create a basic result (binary classification)
        basicResult = new ClassifierEvaluationResult(
                "SVM",
                0.85,  // accuracy
                new double[]{0.80, 0.90},  // precision
                new double[]{0.85, 0.85},  // recall
                new double[]{0.825, 0.875},  // f1
                0.85,  // macro precision
                0.85,  // macro recall
                0.85,  // macro f1
                0.86,  // weighted precision
                0.85,  // weighted recall
                0.855, // weighted f1
                new double[][]{{80, 20}, {15, 85}},  // confusion matrix
                new String[]{"NEGATIVE", "POSITIVE"},
                Map.of("vocabulary_size", 1000)
        );

        // Create advanced result with ROC-AUC, PR-AUC, calibration
        double[] predictedProbs = {0.9, 0.8, 0.7, 0.4, 0.3, 0.2};
        int[] actualLabels = {1, 1, 1, 0, 0, 0};
        CalibrationMetrics calibration = CalibrationMetrics.compute(predictedProbs, actualLabels, 3);

        advancedResult = new ClassifierEvaluationResult(
                "SVM-Advanced",
                0.85,
                new double[]{0.80, 0.90},
                new double[]{0.85, 0.85},
                new double[]{0.825, 0.875},
                0.85, 0.85, 0.85,
                0.86, 0.85, 0.855,
                new double[][]{{80, 20}, {15, 85}},
                new String[]{"NEGATIVE", "POSITIVE"},
                new double[]{0.88, 0.92},  // ROC-AUC
                0.90,  // macro ROC-AUC
                new double[]{0.85, 0.90},  // PR-AUC
                0.875,  // macro PR-AUC
                calibration,
                Map.of("test_samples", 200)
        );
    }

    // ==================== CONSTRUCTOR TESTS ====================

    @Test
    @DisplayName("Basic constructor should initialize all fields correctly")
    void testBasicConstructor() {
        assertNotNull(basicResult);
        assertEquals("SVM", basicResult.getAlgorithmName());
        assertEquals(0.85, basicResult.getAccuracy(), 1e-6);
        assertArrayEquals(new double[]{0.80, 0.90}, basicResult.getPrecision(), 1e-6);
        assertArrayEquals(new double[]{0.85, 0.85}, basicResult.getRecall(), 1e-6);
        assertArrayEquals(new double[]{0.825, 0.875}, basicResult.getF1Score(), 1e-6);
    }

    @Test
    @DisplayName("Advanced constructor should initialize all fields including ROC-AUC and calibration")
    void testAdvancedConstructor() {
        assertNotNull(advancedResult);
        assertTrue(advancedResult.hasAdvancedMetrics());
        assertArrayEquals(new double[]{0.88, 0.92}, advancedResult.getRocAUC(), 1e-6);
        assertEquals(0.90, advancedResult.getMacroAvgROCAUC(), 1e-6);
        assertArrayEquals(new double[]{0.85, 0.90}, advancedResult.getPrAUC(), 1e-6);
        assertEquals(0.875, advancedResult.getMacroAvgPRAUC(), 1e-6);
        assertNotNull(advancedResult.getCalibrationMetrics());
    }

    @Test
    @DisplayName("Constructor should handle null arrays defensively")
    void testConstructor_NullArrays() {
        ClassifierEvaluationResult result = new ClassifierEvaluationResult(
                "Test", 0.5,
                null, null, null,  // null arrays
                0.5, 0.5, 0.5,
                0.5, 0.5, 0.5,
                null, null,
                Map.of()
        );

        assertNotNull(result.getPrecision());
        assertNotNull(result.getRecall());
        assertNotNull(result.getF1Score());
        assertEquals(0, result.getPrecision().length);
    }

    // ==================== GETTER TESTS ====================

    @Test
    @DisplayName("Getters should return correct values")
    void testGetters() {
        assertEquals(0.85, basicResult.getMacroAvgPrecision(), 1e-6);
        assertEquals(0.85, basicResult.getMacroAvgRecall(), 1e-6);
        assertEquals(0.85, basicResult.getMacroAvgF1(), 1e-6);
        assertEquals(0.86, basicResult.getWeightedPrecision(), 1e-6);
        assertEquals(0.85, basicResult.getWeightedRecall(), 1e-6);
        assertEquals(0.855, basicResult.getWeightedF1(), 1e-6);
        assertArrayEquals(new String[]{"NEGATIVE", "POSITIVE"}, basicResult.getClassLabels());
    }

    @Test
    @DisplayName("Confusion matrix getter should return a copy")
    void testConfusionMatrix_ReturnsDeepCopy() {
        double[][] matrix1 = basicResult.getConfusionMatrix();
        double[][] matrix2 = basicResult.getConfusionMatrix();

        // Should be equal but not same reference
        assertNotSame(matrix1, matrix2);
        assertArrayEquals(matrix1[0], matrix2[0], 1e-6);

        // Modifying copy should not affect original
        matrix1[0][0] = 999;
        assertNotEquals(999, basicResult.getConfusionMatrix()[0][0]);
    }

    @Test
    @DisplayName("Array getters should return copies for immutability")
    void testArrayGetters_ReturnCopies() {
        double[] precision1 = basicResult.getPrecision();
        double[] precision2 = basicResult.getPrecision();

        assertNotSame(precision1, precision2);
        assertArrayEquals(precision1, precision2, 1e-6);

        // Modifying copy should not affect original
        precision1[0] = 999;
        assertNotEquals(999, basicResult.getPrecision()[0]);
    }

    // ==================== PER-CLASS METRICS TESTS ====================

    @Test
    @DisplayName("getPrecisionForClass should return correct value")
    void testGetPrecisionForClass() {
        assertEquals(0.80, basicResult.getPrecisionForClass(0), 1e-6);
        assertEquals(0.90, basicResult.getPrecisionForClass(1), 1e-6);
    }

    @Test
    @DisplayName("getPrecisionForClass should throw for invalid index")
    void testGetPrecisionForClass_InvalidIndex() {
        assertThrows(IllegalArgumentException.class,
                () -> basicResult.getPrecisionForClass(-1));
        assertThrows(IllegalArgumentException.class,
                () -> basicResult.getPrecisionForClass(10));
    }

    @Test
    @DisplayName("getRecallForClass should return correct value")
    void testGetRecallForClass() {
        assertEquals(0.85, basicResult.getRecallForClass(0), 1e-6);
        assertEquals(0.85, basicResult.getRecallForClass(1), 1e-6);
    }

    @Test
    @DisplayName("getRecallForClass should throw for invalid index")
    void testGetRecallForClass_InvalidIndex() {
        assertThrows(IllegalArgumentException.class,
                () -> basicResult.getRecallForClass(-1));
        assertThrows(IllegalArgumentException.class,
                () -> basicResult.getRecallForClass(10));
    }

    @Test
    @DisplayName("getF1ForClass should return correct value")
    void testGetF1ForClass() {
        assertEquals(0.825, basicResult.getF1ForClass(0), 1e-6);
        assertEquals(0.875, basicResult.getF1ForClass(1), 1e-6);
    }

    @Test
    @DisplayName("getF1ForClass should throw for invalid index")
    void testGetF1ForClass_InvalidIndex() {
        assertThrows(IllegalArgumentException.class,
                () -> basicResult.getF1ForClass(-1));
        assertThrows(IllegalArgumentException.class,
                () -> basicResult.getF1ForClass(10));
    }

    @Test
    @DisplayName("getMetricsForClass should return correct ClassMetrics by name")
    void testGetMetricsForClass_ByName() {
        ClassifierEvaluationResult.ClassMetrics metrics =
                basicResult.getMetricsForClass("POSITIVE");

        assertEquals("POSITIVE", metrics.getClassName());
        assertEquals(0.90, metrics.getPrecision(), 1e-6);
        assertEquals(0.85, metrics.getRecall(), 1e-6);
        assertEquals(0.875, metrics.getF1Score(), 1e-6);
    }

    @Test
    @DisplayName("getMetricsForClass should throw for unknown class name")
    void testGetMetricsForClass_UnknownClass() {
        assertThrows(IllegalArgumentException.class,
                () -> basicResult.getMetricsForClass("UNKNOWN"));
    }

    // ==================== ADVANCED METRICS TESTS ====================

    @Test
    @DisplayName("hasAdvancedMetrics should return false for basic result")
    void testHasAdvancedMetrics_BasicResult() {
        assertFalse(basicResult.hasAdvancedMetrics());
    }

    @Test
    @DisplayName("hasAdvancedMetrics should return true when ROC-AUC is present")
    void testHasAdvancedMetrics_WithROCAUC() {
        assertTrue(advancedResult.hasAdvancedMetrics());
    }

    @Test
    @DisplayName("Basic result should have null advanced metrics")
    void testBasicResult_NullAdvancedMetrics() {
        assertNull(basicResult.getMacroAvgROCAUC());
        assertNull(basicResult.getMacroAvgPRAUC());
        assertNull(basicResult.getCalibrationMetrics());
        assertEquals(0, basicResult.getRocAUC().length);
        assertEquals(0, basicResult.getPrAUC().length);
    }

    // ==================== ADDITIONAL STATS TESTS ====================

    @Test
    @DisplayName("getAdditionalStats should return immutable map")
    void testAdditionalStats_Immutability() {
        Map<String, Object> stats = basicResult.getAdditionalStats();
        assertEquals(1000, stats.get("vocabulary_size"));

        // Should throw when trying to modify
        assertThrows(UnsupportedOperationException.class,
                () -> stats.put("new_key", "new_value"));
    }

    @Test
    @DisplayName("Constructor should handle null additionalStats")
    void testConstructor_NullAdditionalStats() {
        ClassifierEvaluationResult result = new ClassifierEvaluationResult(
                "Test", 0.5,
                new double[]{0.5}, new double[]{0.5}, new double[]{0.5},
                0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                new double[][]{{1}}, new String[]{"A"},
                null  // null stats
        );

        assertNotNull(result.getAdditionalStats());
        assertTrue(result.getAdditionalStats().isEmpty());
    }

    // ==================== STRING FORMATTING TESTS ====================

    @Test
    @DisplayName("toString should return formatted summary")
    void testToString() {
        String result = basicResult.toString();
        assertTrue(result.contains("SVM"));
        assertTrue(result.contains("0.85"));
        assertTrue(result.contains("accuracy"));
    }

    @Test
    @DisplayName("toDetailedString should include all metrics")
    void testToDetailedString_BasicResult() {
        String detailed = basicResult.toDetailedString();

        // Should contain key sections
        assertTrue(detailed.contains("SVM Evaluation Results"));
        assertTrue(detailed.contains("Overall Accuracy"));
        assertTrue(detailed.contains("Per-Class Metrics"));
        assertTrue(detailed.contains("NEGATIVE"));
        assertTrue(detailed.contains("POSITIVE"));
        assertTrue(detailed.contains("Macro Averages"));
        assertTrue(detailed.contains("Weighted Averages"));
        assertTrue(detailed.contains("Confusion Matrix"));
        assertTrue(detailed.contains("Additional Statistics"));
        assertTrue(detailed.contains("vocabulary_size"));
    }

    @Test
    @DisplayName("toDetailedString should include advanced metrics when present")
    void testToDetailedString_AdvancedResult() {
        String detailed = advancedResult.toDetailedString();

        // Should contain advanced metrics
        assertTrue(detailed.contains("ROC-AUC"));
        assertTrue(detailed.contains("PR-AUC"));
        assertTrue(detailed.contains("Calibration Metrics"));
        assertTrue(detailed.contains("Brier Score"));
        assertTrue(detailed.contains("ECE"));
        assertTrue(detailed.contains("MCE"));
    }

    // ==================== EDGE CASES ====================

    @Test
    @DisplayName("Should handle single-class case")
    void testSingleClass() {
        ClassifierEvaluationResult result = new ClassifierEvaluationResult(
                "Test",
                1.0,
                new double[]{1.0},
                new double[]{1.0},
                new double[]{1.0},
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0,
                new double[][]{{100}},
                new String[]{"ONLY_CLASS"},
                Map.of()
        );

        assertEquals(1, result.getClassLabels().length);
        assertEquals(1.0, result.getPrecisionForClass(0));
    }

    @Test
    @DisplayName("Should handle multi-class case")
    void testMultiClass() {
        ClassifierEvaluationResult result = new ClassifierEvaluationResult(
                "MultiClass-SVM",
                0.75,
                new double[]{0.7, 0.8, 0.75},
                new double[]{0.72, 0.78, 0.76},
                new double[]{0.71, 0.79, 0.755},
                0.75, 0.753, 0.751,
                0.76, 0.77, 0.765,
                new double[][]{
                        {50, 10, 5},
                        {8, 55, 7},
                        {5, 5, 55}
                },
                new String[]{"CLASS_A", "CLASS_B", "CLASS_C"},
                Map.of()
        );

        assertEquals(3, result.getClassLabels().length);
        assertEquals(0.7, result.getPrecisionForClass(0), 1e-6);
        assertEquals(0.8, result.getPrecisionForClass(1), 1e-6);
        assertEquals(0.75, result.getPrecisionForClass(2), 1e-6);
    }

    // ==================== NESTED CLASS TESTS ====================

    @Test
    @DisplayName("ClassMetrics should store and return values correctly")
    void testClassMetrics() {
        ClassifierEvaluationResult.ClassMetrics metrics =
                new ClassifierEvaluationResult.ClassMetrics("TEST_CLASS", 0.85, 0.90, 0.875);

        assertEquals("TEST_CLASS", metrics.getClassName());
        assertEquals(0.85, metrics.getPrecision());
        assertEquals(0.90, metrics.getRecall());
        assertEquals(0.875, metrics.getF1Score());
    }

    @Test
    @DisplayName("ClassMetrics toString should be formatted correctly")
    void testClassMetrics_ToString() {
        ClassifierEvaluationResult.ClassMetrics metrics =
                new ClassifierEvaluationResult.ClassMetrics("TEST", 0.85, 0.90, 0.875);

        String str = metrics.toString();
        assertTrue(str.contains("TEST"));
        assertTrue(str.contains("0.85"));
        assertTrue(str.contains("0.90"));
        assertTrue(str.contains("0.875"));
    }
}
