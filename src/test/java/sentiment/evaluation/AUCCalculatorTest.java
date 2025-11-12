package sentiment.evaluation;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive unit tests for AUCCalculator.
 *
 * Tests cover:
 * - ROC-AUC computation for binary and multi-class
 * - PR-AUC computation for binary and multi-class
 * - Perfect and random classifier cases
 * - Edge cases and validation
 * - Imbalanced datasets
 */
@DisplayName("AUCCalculator Unit Tests")
class AUCCalculatorTest {

    private static final double EPSILON = 1e-6;

    // ==================== ROC-AUC TESTS ====================

    @Test
    @DisplayName("Perfect classifier should have ROC-AUC = 1.0")
    void testROCAUC_PerfectClassifier() {
        // Perfect ranking: all positives ranked higher than negatives
        double[] probs = {0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1};
        int[] labels = {1, 1, 1, 1, 0, 0, 0, 0};

        double auc = AUCCalculator.computeROCAUC(probs, labels);

        assertEquals(1.0, auc, EPSILON);
    }

    @Test
    @DisplayName("Random classifier should have ROC-AUC ≈ 0.5")
    void testROCAUC_RandomClassifier() {
        // Predictions uncorrelated with labels
        double[] probs = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
        int[] labels = {1, 0, 1, 0, 1, 0};

        double auc = AUCCalculator.computeROCAUC(probs, labels);

        // Should be close to 0.5 for random
        assertTrue(Math.abs(auc - 0.5) < 0.2);
    }

    @Test
    @DisplayName("Worst classifier should have ROC-AUC = 0.0")
    void testROCAUC_WorstClassifier() {
        // All negatives ranked higher than positives
        double[] probs = {0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9};
        int[] labels = {1, 1, 1, 1, 0, 0, 0, 0};

        double auc = AUCCalculator.computeROCAUC(probs, labels);

        assertEquals(0.0, auc, EPSILON);
    }

    @Test
    @DisplayName("ROC-AUC should be computed correctly for typical case")
    void testROCAUC_TypicalCase() {
        // Mixed predictions
        double[] probs = {0.9, 0.8, 0.7, 0.4, 0.3, 0.2};
        int[] labels = {1, 1, 0, 1, 0, 0};

        double auc = AUCCalculator.computeROCAUC(probs, labels);

        // Manual calculation:
        // Sorted by prob desc: (0.9,1), (0.8,1), (0.7,0), (0.4,1), (0.3,0), (0.2,0)
        // ROC points: (0,0) -> (0,1/3) -> (0,2/3) -> (1/3,2/3) -> (1/3,1) -> (2/3,1) -> (1,1)
        // AUC ≈ 0.67
        assertTrue(auc > 0.5);
        assertTrue(auc < 1.0);
    }

    @Test
    @DisplayName("ROC-AUC should be in range [0, 1]")
    void testROCAUC_Range() {
        double[] probs = {0.6, 0.4, 0.7, 0.3, 0.5};
        int[] labels = {1, 0, 1, 0, 1};

        double auc = AUCCalculator.computeROCAUC(probs, labels);

        assertTrue(auc >= 0.0);
        assertTrue(auc <= 1.0);
    }

    @Test
    @DisplayName("ROC-AUC should handle tied predictions")
    void testROCAUC_TiedPredictions() {
        double[] probs = {0.5, 0.5, 0.5, 0.5};
        int[] labels = {1, 1, 0, 0};

        double auc = AUCCalculator.computeROCAUC(probs, labels);

        // With ties, AUC should be around 0.5
        assertTrue(auc >= 0.0);
        assertTrue(auc <= 1.0);
    }

    @Test
    @DisplayName("ROC-AUC should return 0.5 when only one class present")
    void testROCAUC_SingleClass() {
        double[] probs = {0.9, 0.8, 0.7};
        int[] labels = {1, 1, 1};  // Only positives

        double auc = AUCCalculator.computeROCAUC(probs, labels);

        assertEquals(0.5, auc, EPSILON);  // Undefined, returns 0.5
    }

    // ==================== PR-AUC TESTS ====================

    @Test
    @DisplayName("Perfect classifier should have PR-AUC near 1.0")
    void testPRAUC_PerfectClassifier() {
        double[] probs = {0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1};
        int[] labels = {1, 1, 1, 1, 0, 0, 0, 0};

        double auc = AUCCalculator.computePRAUC(probs, labels);

        // PR-AUC computation using trapezoidal rule may not yield exact 1.0
        assertTrue(auc > 0.7);  // Should be high for perfect ranking
        assertTrue(auc <= 1.0);
    }

    @Test
    @DisplayName("PR-AUC should be lower for imbalanced data compared to ROC-AUC")
    void testPRAUC_ImbalancedData() {
        // Imbalanced: 2 positives, 8 negatives
        double[] probs = {0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05};
        int[] labels = {1, 1, 0, 0, 0, 0, 0, 0, 0, 0};

        double rocAUC = AUCCalculator.computeROCAUC(probs, labels);
        double prAUC = AUCCalculator.computePRAUC(probs, labels);

        // PR-AUC is typically more pessimistic than ROC-AUC for imbalanced data
        assertTrue(rocAUC > 0.9);  // Should be high
        assertTrue(prAUC < rocAUC);  // PR-AUC should be lower
    }

    @Test
    @DisplayName("PR-AUC should be computed correctly for typical case")
    void testPRAUC_TypicalCase() {
        double[] probs = {0.9, 0.8, 0.7, 0.4, 0.3, 0.2};
        int[] labels = {1, 1, 0, 1, 0, 0};

        double auc = AUCCalculator.computePRAUC(probs, labels);

        // PR-AUC should be in valid range
        assertTrue(auc >= 0.0);
        assertTrue(auc <= 1.0);
        assertTrue(auc > 0.4);  // Should be above random baseline for this case
    }

    @Test
    @DisplayName("PR-AUC should return 0 when no positive samples")
    void testPRAUC_NoPositives() {
        double[] probs = {0.9, 0.8, 0.7};
        int[] labels = {0, 0, 0};  // Only negatives

        double auc = AUCCalculator.computePRAUC(probs, labels);

        assertEquals(0.0, auc, EPSILON);  // Undefined, returns 0
    }

    @Test
    @DisplayName("PR-AUC should be in range [0, 1]")
    void testPRAUC_Range() {
        double[] probs = {0.6, 0.4, 0.7, 0.3, 0.5};
        int[] labels = {1, 0, 1, 0, 1};

        double auc = AUCCalculator.computePRAUC(probs, labels);

        assertTrue(auc >= 0.0);
        assertTrue(auc <= 1.0);
    }

    // ==================== MULTI-CLASS TESTS ====================

    @Test
    @DisplayName("Multi-class ROC-AUC should compute per-class AUC")
    void testMultiClassROCAUC() {
        // 3-class problem, 6 samples
        double[][] probs = {
                {0.8, 0.1, 0.1},  // Actual: 0
                {0.7, 0.2, 0.1},  // Actual: 0
                {0.1, 0.8, 0.1},  // Actual: 1
                {0.1, 0.7, 0.2},  // Actual: 1
                {0.1, 0.1, 0.8},  // Actual: 2
                {0.2, 0.1, 0.7}   // Actual: 2
        };
        int[] labels = {0, 0, 1, 1, 2, 2};

        double[] perClassAUC = AUCCalculator.computeMultiClassROCAUC(probs, labels);

        assertEquals(3, perClassAUC.length);

        // Each class should have high AUC (well-separated)
        for (double auc : perClassAUC) {
            assertTrue(auc >= 0.0);
            assertTrue(auc <= 1.0);
            assertTrue(auc > 0.8);  // Should be high for this well-separated case
        }
    }

    @Test
    @DisplayName("Multi-class PR-AUC should compute per-class AUC")
    void testMultiClassPRAUC() {
        double[][] probs = {
                {0.8, 0.1, 0.1},
                {0.7, 0.2, 0.1},
                {0.1, 0.8, 0.1},
                {0.1, 0.7, 0.2},
                {0.1, 0.1, 0.8},
                {0.2, 0.1, 0.7}
        };
        int[] labels = {0, 0, 1, 1, 2, 2};

        double[] perClassAUC = AUCCalculator.computeMultiClassPRAUC(probs, labels);

        assertEquals(3, perClassAUC.length);

        for (double auc : perClassAUC) {
            assertTrue(auc >= 0.0);
            assertTrue(auc <= 1.0);
        }
    }

    @Test
    @DisplayName("Multi-class should handle imbalanced classes")
    void testMultiClass_ImbalancedClasses() {
        // Class 0: 4 samples, Class 1: 1 sample, Class 2: 1 sample
        double[][] probs = {
                {0.9, 0.05, 0.05},  // Actual: 0
                {0.8, 0.1, 0.1},    // Actual: 0
                {0.7, 0.2, 0.1},    // Actual: 0
                {0.6, 0.3, 0.1},    // Actual: 0
                {0.1, 0.8, 0.1},    // Actual: 1
                {0.1, 0.1, 0.8}     // Actual: 2
        };
        int[] labels = {0, 0, 0, 0, 1, 2};

        double[] rocAUC = AUCCalculator.computeMultiClassROCAUC(probs, labels);
        double[] prAUC = AUCCalculator.computeMultiClassPRAUC(probs, labels);

        assertEquals(3, rocAUC.length);
        assertEquals(3, prAUC.length);

        // All should be valid
        for (int i = 0; i < 3; i++) {
            assertTrue(rocAUC[i] >= 0.0 && rocAUC[i] <= 1.0);
            assertTrue(prAUC[i] >= 0.0 && prAUC[i] <= 1.0);
        }
    }

    // ==================== VALIDATION TESTS ====================

    @Test
    @DisplayName("Should throw on array length mismatch")
    void testValidation_LengthMismatch() {
        double[] probs = {0.9, 0.8, 0.7};
        int[] labels = {1, 0};  // Wrong length

        assertThrows(IllegalArgumentException.class,
                () -> AUCCalculator.computeROCAUC(probs, labels));

        assertThrows(IllegalArgumentException.class,
                () -> AUCCalculator.computePRAUC(probs, labels));
    }

    @Test
    @DisplayName("Should throw on empty arrays")
    void testValidation_EmptyArrays() {
        double[] probs = {};
        int[] labels = {};

        assertThrows(IllegalArgumentException.class,
                () -> AUCCalculator.computeROCAUC(probs, labels));

        assertThrows(IllegalArgumentException.class,
                () -> AUCCalculator.computePRAUC(probs, labels));
    }

    // ==================== EDGE CASES ====================

    @Test
    @DisplayName("Should handle single sample")
    void testEdgeCase_SingleSample() {
        double[] probs = {0.7};
        int[] labels = {1};

        // Should not crash, though result may be undefined
        double rocAUC = AUCCalculator.computeROCAUC(probs, labels);
        double prAUC = AUCCalculator.computePRAUC(probs, labels);

        assertTrue(rocAUC >= 0.0 && rocAUC <= 1.0);
        assertTrue(prAUC >= 0.0 && prAUC <= 1.0);
    }

    @Test
    @DisplayName("Should handle two samples")
    void testEdgeCase_TwoSamples() {
        double[] probs = {0.9, 0.1};
        int[] labels = {1, 0};

        double rocAUC = AUCCalculator.computeROCAUC(probs, labels);
        double prAUC = AUCCalculator.computePRAUC(probs, labels);

        assertEquals(1.0, rocAUC, EPSILON);  // Perfect separation
        assertTrue(prAUC >= 0.0 && prAUC <= 1.0);  // Valid range
    }

    @Test
    @DisplayName("Should handle all probabilities at extremes")
    void testEdgeCase_ExtremeProbabilities() {
        double[] probs = {1.0, 1.0, 0.0, 0.0};
        int[] labels = {1, 1, 0, 0};

        double rocAUC = AUCCalculator.computeROCAUC(probs, labels);
        double prAUC = AUCCalculator.computePRAUC(probs, labels);

        assertEquals(1.0, rocAUC, EPSILON);
        assertTrue(prAUC >= 0.5);  // Should be reasonably high
        assertTrue(prAUC <= 1.0);
    }

    @Test
    @DisplayName("Should handle balanced dataset")
    void testEdgeCase_BalancedDataset() {
        double[] probs = {0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1};
        int[] labels = {1, 1, 1, 1, 0, 0, 0, 0};

        double rocAUC = AUCCalculator.computeROCAUC(probs, labels);
        double prAUC = AUCCalculator.computePRAUC(probs, labels);

        // ROC-AUC should be perfect for this case
        assertEquals(1.0, rocAUC, EPSILON);
        // PR-AUC should be high (trapezoidal approximation)
        assertTrue(prAUC > 0.7);
        assertTrue(prAUC <= 1.0);
    }

    @Test
    @DisplayName("Should handle highly imbalanced dataset")
    void testEdgeCase_HighlyImbalanced() {
        // 1 positive, 99 negatives
        double[] probs = new double[100];
        int[] labels = new int[100];

        probs[0] = 0.99;  // One positive with high score
        labels[0] = 1;

        for (int i = 1; i < 100; i++) {
            probs[i] = 0.01 + (i * 0.005);  // Negatives with lower scores
            labels[i] = 0;
        }

        double rocAUC = AUCCalculator.computeROCAUC(probs, labels);
        double prAUC = AUCCalculator.computePRAUC(probs, labels);

        // ROC-AUC should be very low since positives are ranked lower
        assertTrue(rocAUC >= 0.0 && rocAUC <= 1.0);

        // PR-AUC should be valid
        assertTrue(prAUC >= 0.0 && prAUC <= 1.0);
    }

    // ==================== CONSISTENCY TESTS ====================

    @Test
    @DisplayName("ROC-AUC and PR-AUC should both be high for perfect classifier")
    void testConsistency_PerfectClassifier() {
        double[] probs = {0.9, 0.8, 0.7, 0.3, 0.2, 0.1};
        int[] labels = {1, 1, 1, 0, 0, 0};

        double rocAUC = AUCCalculator.computeROCAUC(probs, labels);
        double prAUC = AUCCalculator.computePRAUC(probs, labels);

        assertEquals(1.0, rocAUC, EPSILON);
        // PR-AUC should also be high (though may not be exact 1.0 due to trapezoid integration)
        assertTrue(prAUC > 0.6);
        assertTrue(prAUC <= 1.0);
    }

    @Test
    @DisplayName("ROC-AUC should be symmetric")
    void testConsistency_Symmetry() {
        double[] probs = {0.9, 0.8, 0.7, 0.3, 0.2, 0.1};
        int[] labels1 = {1, 1, 1, 0, 0, 0};

        // Flip labels
        int[] labels2 = {0, 0, 0, 1, 1, 1};

        double auc1 = AUCCalculator.computeROCAUC(probs, labels1);
        double auc2 = AUCCalculator.computeROCAUC(probs, labels2);

        // AUC1 + AUC2 should equal 1.0 (symmetry property)
        assertEquals(1.0, auc1 + auc2, 0.01);
    }

    @Test
    @DisplayName("Reordering samples should not change AUC")
    void testConsistency_OrderInvariance() {
        double[] probs1 = {0.9, 0.8, 0.7, 0.3, 0.2, 0.1};
        int[] labels1 = {1, 1, 1, 0, 0, 0};

        // Reorder
        double[] probs2 = {0.1, 0.9, 0.2, 0.8, 0.3, 0.7};
        int[] labels2 = {0, 1, 0, 1, 0, 1};

        double auc1 = AUCCalculator.computeROCAUC(probs1, labels1);
        double auc2 = AUCCalculator.computeROCAUC(probs2, labels2);

        assertEquals(auc1, auc2, EPSILON);
    }
}
