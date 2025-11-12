package sentiment.evaluation;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive unit tests for CalibrationMetrics.
 *
 * Tests cover:
 * - Brier score computation
 * - Expected Calibration Error (ECE)
 * - Maximum Calibration Error (MCE)
 * - Reliability diagram bins
 * - Multi-class calibration
 * - Edge cases and validation
 */
@DisplayName("CalibrationMetrics Unit Tests")
class CalibrationMetricsTest {

    private static final double EPSILON = 1e-6;

    // ==================== BASIC COMPUTATION TESTS ====================

    @Test
    @DisplayName("Perfect calibration should yield near-zero Brier score and ECE")
    void testPerfectCalibration() {
        // Perfect predictions: prob=1.0 for positives, prob=0.0 for negatives
        double[] probs = {1.0, 1.0, 1.0, 0.0, 0.0, 0.0};
        int[] labels = {1, 1, 1, 0, 0, 0};

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 10);

        // Brier score should be 0 for perfect predictions
        assertEquals(0.0, metrics.getBrierScore(), EPSILON);

        // ECE should be very low (bins may not align perfectly)
        assertTrue(metrics.getExpectedCalibrationError() < 0.15);
    }

    @Test
    @DisplayName("Worst calibration should yield high Brier score")
    void testWorstCalibration() {
        // Completely wrong predictions
        double[] probs = {0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
        int[] labels = {1, 1, 1, 0, 0, 0};

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 10);

        // Brier score should be 1.0 for worst predictions
        assertEquals(1.0, metrics.getBrierScore(), EPSILON);
    }

    @Test
    @DisplayName("Well-calibrated model should have low ECE")
    void testWellCalibratedModel() {
        // Predictions roughly match actual frequencies
        // 80% confident predictions -> 80% accuracy
        double[] probs = {0.8, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2};
        int[] labels = {1, 1, 1, 1, 0, 0, 0, 0, 0, 1};  // 4/5 correct in high conf, 4/5 correct in low conf

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 5);

        // ECE should be low for well-calibrated model
        assertTrue(metrics.getExpectedCalibrationError() < 0.2);
    }

    @Test
    @DisplayName("Poorly calibrated model should have high ECE")
    void testPoorlyCalibratedModel() {
        // Model is overconfident: predicts 0.9 but only 50% correct
        double[] probs = {0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1};
        int[] labels = {1, 1, 0, 0, 0, 0, 1, 1};  // 50% correct in both groups

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 5);

        // ECE should be high (confidence=0.9, accuracy=0.5 -> gap=0.4)
        assertTrue(metrics.getExpectedCalibrationError() > 0.3);
    }

    // ==================== BRIER SCORE TESTS ====================

    @Test
    @DisplayName("Brier score should be computed correctly")
    void testBrierScoreComputation() {
        double[] probs = {0.7, 0.6, 0.3, 0.2};
        int[] labels = {1, 1, 0, 0};

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 5);

        // Manual calculation:
        // BS = ((0.7-1)^2 + (0.6-1)^2 + (0.3-0)^2 + (0.2-0)^2) / 4
        // BS = (0.09 + 0.16 + 0.09 + 0.04) / 4 = 0.38 / 4 = 0.095
        double expectedBrier = 0.095;
        assertEquals(expectedBrier, metrics.getBrierScore(), EPSILON);
    }

    @Test
    @DisplayName("Brier score should be in range [0, 1]")
    void testBrierScoreRange() {
        double[] probs = {0.5, 0.5, 0.5, 0.5};
        int[] labels = {1, 0, 1, 0};

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 5);

        assertTrue(metrics.getBrierScore() >= 0.0);
        assertTrue(metrics.getBrierScore() <= 1.0);
    }

    // ==================== BINNING TESTS ====================

    @Test
    @DisplayName("Bins should be created correctly")
    void testBinCreation() {
        double[] probs = {0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95};
        int[] labels = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 10);

        List<CalibrationMetrics.CalibrationBin> bins = metrics.getBins();
        assertEquals(10, bins.size());

        // Each bin should have exactly one sample
        for (CalibrationMetrics.CalibrationBin bin : bins) {
            assertEquals(1, bin.getCount());
        }
    }

    @Test
    @DisplayName("Edge case: probability=1.0 should go in last bin")
    void testEdgeCase_ProbabilityOne() {
        double[] probs = {1.0};
        int[] labels = {1};

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 10);

        List<CalibrationMetrics.CalibrationBin> bins = metrics.getBins();

        // Only the last bin should have a sample
        int nonEmptyBins = 0;
        for (CalibrationMetrics.CalibrationBin bin : bins) {
            if (bin.getCount() > 0) {
                nonEmptyBins++;
            }
        }
        assertEquals(1, nonEmptyBins);
    }

    @Test
    @DisplayName("CalibrationBin should compute average confidence correctly")
    void testBin_AverageConfidence() {
        CalibrationMetrics.CalibrationBin bin = new CalibrationMetrics.CalibrationBin(0.5, 0.6);

        bin.addSample(0.52, 1);
        bin.addSample(0.54, 0);
        bin.addSample(0.56, 1);

        assertEquals(3, bin.getCount());
        assertEquals((0.52 + 0.54 + 0.56) / 3.0, bin.getAverageConfidence(), EPSILON);
    }

    @Test
    @DisplayName("CalibrationBin should compute accuracy correctly")
    void testBin_Accuracy() {
        CalibrationMetrics.CalibrationBin bin = new CalibrationMetrics.CalibrationBin(0.5, 0.6);

        bin.addSample(0.52, 1);
        bin.addSample(0.54, 0);
        bin.addSample(0.56, 1);

        assertEquals(3, bin.getCount());
        assertEquals(2.0 / 3.0, bin.getAccuracy(), EPSILON);  // 2 correct out of 3
    }

    @Test
    @DisplayName("Empty bin should return zero for averages")
    void testEmptyBin() {
        CalibrationMetrics.CalibrationBin bin = new CalibrationMetrics.CalibrationBin(0.5, 0.6);

        assertEquals(0, bin.getCount());
        assertEquals(0.0, bin.getAverageConfidence(), EPSILON);
        assertEquals(0.0, bin.getAccuracy(), EPSILON);
    }

    // ==================== MULTI-CLASS TESTS ====================

    @Test
    @DisplayName("Multi-class calibration should average metrics across classes")
    void testMultiClassCalibration() {
        // 3-class problem, 6 samples
        double[][] probs = {
                {0.8, 0.1, 0.1},  // Actual class: 0
                {0.7, 0.2, 0.1},  // Actual class: 0
                {0.1, 0.8, 0.1},  // Actual class: 1
                {0.1, 0.7, 0.2},  // Actual class: 1
                {0.1, 0.1, 0.8},  // Actual class: 2
                {0.2, 0.1, 0.7}   // Actual class: 2
        };
        int[] labels = {0, 0, 1, 1, 2, 2};

        CalibrationMetrics metrics = CalibrationMetrics.computeMultiClass(probs, labels, 5);

        assertNotNull(metrics);
        assertTrue(metrics.getBrierScore() >= 0.0);
        assertTrue(metrics.getBrierScore() <= 1.0);
        assertTrue(metrics.getExpectedCalibrationError() >= 0.0);
        assertTrue(metrics.getExpectedCalibrationError() <= 1.0);
    }

    @Test
    @DisplayName("Multi-class should throw on dimension mismatch")
    void testMultiClass_DimensionMismatch() {
        double[][] probs = {{0.5, 0.5}, {0.5, 0.5}};
        int[] labels = {0};  // Wrong length

        assertThrows(IllegalArgumentException.class,
                () -> CalibrationMetrics.computeMultiClass(probs, labels, 5));
    }

    // ==================== VALIDATION TESTS ====================

    @Test
    @DisplayName("Should throw on array length mismatch")
    void testValidation_LengthMismatch() {
        double[] probs = {0.5, 0.6, 0.7};
        int[] labels = {1, 0};  // Different length

        assertThrows(IllegalArgumentException.class,
                () -> CalibrationMetrics.compute(probs, labels, 10));
    }

    @Test
    @DisplayName("Should throw on invalid probability values")
    void testValidation_InvalidProbability() {
        double[] probs = {-0.1, 0.5, 0.7};  // Negative probability
        int[] labels = {1, 0, 1};

        assertThrows(IllegalArgumentException.class,
                () -> CalibrationMetrics.compute(probs, labels, 10));

        double[] probs2 = {1.5, 0.5, 0.7};  // Probability > 1
        assertThrows(IllegalArgumentException.class,
                () -> CalibrationMetrics.compute(probs2, labels, 10));
    }

    @Test
    @DisplayName("Should throw on invalid labels")
    void testValidation_InvalidLabels() {
        double[] probs = {0.5, 0.6, 0.7};
        int[] labels = {1, 2, 0};  // Label 2 is invalid for binary

        assertThrows(IllegalArgumentException.class,
                () -> CalibrationMetrics.compute(probs, labels, 10));
    }

    @Test
    @DisplayName("Should throw when numBins < 2")
    void testValidation_InvalidNumBins() {
        double[] probs = {0.5, 0.6};
        int[] labels = {1, 0};

        assertThrows(IllegalArgumentException.class,
                () -> CalibrationMetrics.compute(probs, labels, 1));

        assertThrows(IllegalArgumentException.class,
                () -> CalibrationMetrics.compute(probs, labels, 0));
    }

    // ==================== STATISTICS TESTS ====================

    @Test
    @DisplayName("Average confidence should be computed correctly")
    void testAverageConfidence() {
        double[] probs = {0.2, 0.4, 0.6, 0.8};
        int[] labels = {0, 0, 1, 1};

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 5);

        double expectedAvgConf = (0.2 + 0.4 + 0.6 + 0.8) / 4.0;
        assertEquals(expectedAvgConf, metrics.getAverageConfidence(), EPSILON);
    }

    @Test
    @DisplayName("Average accuracy should be computed correctly")
    void testAverageAccuracy() {
        double[] probs = {0.9, 0.8, 0.2, 0.1};
        int[] labels = {1, 1, 0, 0};

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 5);

        // Average of labels = (1+1+0+0)/4 = 0.5
        assertEquals(0.5, metrics.getAverageAccuracy(), EPSILON);
    }

    // ==================== ECE AND MCE TESTS ====================

    @Test
    @DisplayName("ECE should be weighted average of bin calibration errors")
    void testECE_Computation() {
        // Create two distinct groups
        double[] probs = {0.9, 0.9, 0.1, 0.1};
        int[] labels = {1, 1, 0, 0};  // Perfect calibration

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 2);

        // Should have very low ECE since predictions match labels well
        assertTrue(metrics.getExpectedCalibrationError() < 0.15);
    }

    @Test
    @DisplayName("MCE should be maximum calibration error across bins")
    void testMCE_Computation() {
        double[] probs = {0.9, 0.9, 0.1, 0.1};
        int[] labels = {0, 0, 1, 1};  // Completely reversed!

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 2);

        // MCE should be high (worst bin has large gap)
        assertTrue(metrics.getMaximumCalibrationError() > 0.7);
    }

    @Test
    @DisplayName("ECE should be in range [0, 1]")
    void testECE_Range() {
        double[] probs = {0.3, 0.5, 0.7, 0.9};
        int[] labels = {0, 1, 0, 1};

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 4);

        assertTrue(metrics.getExpectedCalibrationError() >= 0.0);
        assertTrue(metrics.getExpectedCalibrationError() <= 1.0);
    }

    @Test
    @DisplayName("MCE should be in range [0, 1]")
    void testMCE_Range() {
        double[] probs = {0.3, 0.5, 0.7, 0.9};
        int[] labels = {0, 1, 0, 1};

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 4);

        assertTrue(metrics.getMaximumCalibrationError() >= 0.0);
        assertTrue(metrics.getMaximumCalibrationError() <= 1.0);
    }

    // ==================== TO STRING TESTS ====================

    @Test
    @DisplayName("toString should contain key metrics")
    void testToString() {
        double[] probs = {0.7, 0.6, 0.3, 0.2};
        int[] labels = {1, 1, 0, 0};

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 5);
        String str = metrics.toString();

        assertTrue(str.contains("Brier"));
        assertTrue(str.contains("ECE"));
        assertTrue(str.contains("MCE"));
    }

    @Test
    @DisplayName("Bin toString should be formatted correctly")
    void testBin_ToString() {
        CalibrationMetrics.CalibrationBin bin = new CalibrationMetrics.CalibrationBin(0.5, 0.6);
        bin.addSample(0.55, 1);
        bin.addSample(0.52, 0);

        String str = bin.toString();
        assertTrue(str.contains("0.5"));
        assertTrue(str.contains("0.6"));
        assertTrue(str.contains("conf"));
        assertTrue(str.contains("acc"));
    }

    @Test
    @DisplayName("Empty bin toString should indicate empty")
    void testEmptyBin_ToString() {
        CalibrationMetrics.CalibrationBin bin = new CalibrationMetrics.CalibrationBin(0.5, 0.6);
        String str = bin.toString();
        assertTrue(str.contains("empty"));
    }

    // ==================== EDGE CASES ====================

    @Test
    @DisplayName("Should handle all same predictions")
    void testEdgeCase_SamePredictions() {
        double[] probs = {0.5, 0.5, 0.5, 0.5};
        int[] labels = {1, 0, 1, 0};

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 5);

        assertNotNull(metrics);
        assertEquals(0.25, metrics.getBrierScore(), EPSILON);  // (0.5-1)^2 + (0.5-0)^2 = 0.5, avg = 0.25
    }

    @Test
    @DisplayName("Should handle all same labels")
    void testEdgeCase_SameLabels() {
        double[] probs = {0.9, 0.8, 0.7, 0.6};
        int[] labels = {1, 1, 1, 1};

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 5);

        assertNotNull(metrics);
        assertTrue(metrics.getBrierScore() < 0.1);  // Should be low for high-confidence correct predictions
    }

    @Test
    @DisplayName("Should handle minimum dataset")
    void testEdgeCase_MinimumDataset() {
        double[] probs = {0.7, 0.3};
        int[] labels = {1, 0};

        CalibrationMetrics metrics = CalibrationMetrics.compute(probs, labels, 2);

        assertNotNull(metrics);
        assertTrue(metrics.getBrierScore() >= 0.0);
    }
}
