package sentiment.data;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class DatasetUtilsTest {

    @Test
    void balanceByUndersampling_balancesImbalancedData() {
        // Create imbalanced dataset: 100 positive, 50 negative
        List<Dataset> data = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            data.add(createDataset("positive text " + i, Dataset.SentimentLabel.POSITIVE));
        }
        for (int i = 0; i < 50; i++) {
            data.add(createDataset("negative text " + i, Dataset.SentimentLabel.NEGATIVE));
        }

        List<Dataset> balanced = DatasetUtils.balanceByUndersampling(data, 42L);

        // Should have 50 of each class (100 total)
        assertEquals(100, balanced.size());
        long positiveCount = balanced.stream()
                .filter(d -> d.getSentiment() == Dataset.SentimentLabel.POSITIVE).count();
        long negativeCount = balanced.stream()
                .filter(d -> d.getSentiment() == Dataset.SentimentLabel.NEGATIVE).count();
        assertEquals(50, positiveCount);
        assertEquals(50, negativeCount);
    }

    @Test
    void balanceByUndersampling_preservesAlreadyBalancedData() {
        // Create already balanced dataset: 50 positive, 50 negative
        List<Dataset> data = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            data.add(createDataset("positive text " + i, Dataset.SentimentLabel.POSITIVE));
        }
        for (int i = 0; i < 50; i++) {
            data.add(createDataset("negative text " + i, Dataset.SentimentLabel.NEGATIVE));
        }

        List<Dataset> balanced = DatasetUtils.balanceByUndersampling(data, 42L);

        // Should return same size (within tolerance, no undersampling needed)
        assertEquals(100, balanced.size());
    }

    @Test
    void balanceByUndersampling_withinToleranceNoChange() {
        // Create nearly balanced dataset: 100 positive, 96 negative (96% ratio)
        List<Dataset> data = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            data.add(createDataset("positive text " + i, Dataset.SentimentLabel.POSITIVE));
        }
        for (int i = 0; i < 96; i++) {
            data.add(createDataset("negative text " + i, Dataset.SentimentLabel.NEGATIVE));
        }

        List<Dataset> balanced = DatasetUtils.balanceByUndersampling(data, 42L);

        // Should return original data (ratio 0.96 >= 0.95 tolerance)
        assertEquals(196, balanced.size());
    }

    @Test
    void balanceByUndersampling_handlesEmptyInput() {
        List<Dataset> data = new ArrayList<>();

        List<Dataset> balanced = DatasetUtils.balanceByUndersampling(data, 42L);

        assertTrue(balanced.isEmpty());
    }

    @Test
    void balanceByUndersampling_handlesNullInput() {
        List<Dataset> balanced = DatasetUtils.balanceByUndersampling(null, 42L);

        assertNull(balanced);
    }

    @Test
    void balanceByUndersampling_handlesOneClassEmpty() {
        // Only positive samples
        List<Dataset> data = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            data.add(createDataset("positive text " + i, Dataset.SentimentLabel.POSITIVE));
        }

        List<Dataset> balanced = DatasetUtils.balanceByUndersampling(data, 42L);

        // Should return original data (cannot balance)
        assertEquals(50, balanced.size());
    }

    @Test
    void balanceByUndersampling_isDeterministicWithSameSeed() {
        List<Dataset> data = createTestData(100, 50);

        List<Dataset> balanced1 = DatasetUtils.balanceByUndersampling(new ArrayList<>(data), 42L);
        List<Dataset> balanced2 = DatasetUtils.balanceByUndersampling(new ArrayList<>(data), 42L);

        // Same seed should produce same order
        assertEquals(balanced1.size(), balanced2.size());
        for (int i = 0; i < balanced1.size(); i++) {
            assertEquals(balanced1.get(i).getText(), balanced2.get(i).getText());
        }
    }

    @Test
    void balanceByUndersampling_differentSeedsDifferentOrder() {
        List<Dataset> data = createTestData(100, 50);

        List<Dataset> balanced1 = DatasetUtils.balanceByUndersampling(new ArrayList<>(data), 42L);
        List<Dataset> balanced2 = DatasetUtils.balanceByUndersampling(new ArrayList<>(data), 123L);

        // Different seeds should produce different order (with high probability)
        assertEquals(balanced1.size(), balanced2.size());
        boolean anyDifferent = false;
        for (int i = 0; i < balanced1.size(); i++) {
            if (!balanced1.get(i).getText().equals(balanced2.get(i).getText())) {
                anyDifferent = true;
                break;
            }
        }
        assertTrue(anyDifferent, "Different seeds should produce different ordering");
    }

    private List<Dataset> createTestData(int positiveCount, int negativeCount) {
        List<Dataset> data = new ArrayList<>();
        for (int i = 0; i < positiveCount; i++) {
            data.add(createDataset("positive text " + i, Dataset.SentimentLabel.POSITIVE));
        }
        for (int i = 0; i < negativeCount; i++) {
            data.add(createDataset("negative text " + i, Dataset.SentimentLabel.NEGATIVE));
        }
        return data;
    }

    private Dataset createDataset(String text, Dataset.SentimentLabel sentiment) {
        return new Dataset.Builder(text, sentiment).build();
    }
}
