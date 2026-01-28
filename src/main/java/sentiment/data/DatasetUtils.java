package sentiment.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * Utility methods for dataset manipulation.
 * Provides shared operations used across data preparation and training pipelines.
 */
public final class DatasetUtils {

    private static final Logger logger = LoggerFactory.getLogger(DatasetUtils.class);

    /**
     * Tolerance threshold for considering classes "already balanced".
     * If minority/majority ratio >= this value, no undersampling is performed.
     */
    private static final double BALANCE_TOLERANCE = 0.95;

    private DatasetUtils() {
        // Utility class - prevent instantiation
    }

    /**
     * Balance classes by undersampling the majority class to match the minority class.
     * This ensures fair model comparison across datasets with different class distributions.
     *
     * The method:
     * 1. Separates data by sentiment class
     * 2. If already balanced (within 5% tolerance), returns original data
     * 3. Otherwise, undersamples majority class to match minority class size
     * 4. Shuffles the result to mix classes
     *
     * @param data input dataset (may be imbalanced)
     * @param seed random seed for reproducible shuffling
     * @return balanced dataset with equal positive/negative samples
     */
    public static List<Dataset> balanceByUndersampling(List<Dataset> data, long seed) {
        if (data == null || data.isEmpty()) {
            return data;
        }

        // Separate by class
        List<Dataset> positive = data.stream()
                .filter(d -> d.getSentiment() == Dataset.SentimentLabel.POSITIVE)
                .collect(Collectors.toList());
        List<Dataset> negative = data.stream()
                .filter(d -> d.getSentiment() == Dataset.SentimentLabel.NEGATIVE)
                .collect(Collectors.toList());

        // Find minority class size
        int minoritySize = Math.min(positive.size(), negative.size());

        if (minoritySize == 0) {
            logger.warn("One class has zero samples - cannot balance. Returning original data.");
            return data;
        }

        // Check if already balanced (within tolerance)
        double ratio = (double) minoritySize / Math.max(positive.size(), negative.size());
        if (ratio >= BALANCE_TOLERANCE) {
            logger.info("Classes already balanced (ratio: {}) - no undersampling needed",
                    String.format("%.2f", ratio));
            return data;
        }

        // Undersample majority class
        List<Dataset> balancedPositive = positive.subList(0, minoritySize);
        List<Dataset> balancedNegative = negative.subList(0, minoritySize);

        // Combine and shuffle to mix classes
        List<Dataset> balanced = new ArrayList<>(minoritySize * 2);
        balanced.addAll(balancedPositive);
        balanced.addAll(balancedNegative);
        Collections.shuffle(balanced, new Random(seed));

        logger.info("Balanced classes: {} -> {} samples (undersampled {} from majority class)",
                data.size(), balanced.size(), data.size() - balanced.size());
        logger.info("Final distribution: positive={}, negative={}", minoritySize, minoritySize);

        return balanced;
    }
}
