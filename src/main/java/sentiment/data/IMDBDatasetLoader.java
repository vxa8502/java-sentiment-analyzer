package sentiment.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import sentiment.evaluation.StratifiedDataSplitter;
import sentiment.evaluation.StratifiedDataSplitter.DataSplit;

import java.util.List;

/**
 * Specialized loader for IMDB Large Movie Review Dataset.
 *
 * Sofia's Principle: "Dataset-specific loaders encapsulate domain knowledge."
 *
 * The IMDB dataset has specific characteristics:
 * - Binary sentiment (positive >= 7/10 rating, negative <= 4/10)
 * - 25K train, 25K test
 * - Balanced class distribution
 * - Movie reviews (longer text, complex language)
 *
 * @author Victoria Alabi
 */
@Component
public class IMDBDatasetLoader {

    private static final Logger logger = LoggerFactory.getLogger(IMDBDatasetLoader.class);

    private final SimpleDatasetLoader baseLoader;

    public IMDBDatasetLoader(SimpleDatasetLoader baseLoader) {
        this.baseLoader = baseLoader;
    }

    /**
     * Load IMDB training set (25,000 reviews).
     *
     * @return DatasetLoadResult with training data
     * @throws DataLoadingException if loading fails
     */
    public DatasetLoadResult loadTrainSet() throws DataLoadingException {
        String path = "src/main/resources/datasets/v1_raw/imdb_train.csv";
        logger.info("Loading IMDB training set from {}", path);

        DatasetLoadResult result = baseLoader.loadWithMetadata(path);

        // Validate expected size
        validateIMDBDataset(result.datasets(), "train", 25000);

        return result;
    }

    /**
     * Load IMDB test set (25,000 reviews).
     *
     * @return DatasetLoadResult with test data
     * @throws DataLoadingException if loading fails
     */
    public DatasetLoadResult loadTestSet() throws DataLoadingException {
        String path = "src/main/resources/datasets/v1_raw/imdb_test.csv";
        logger.info("Loading IMDB test set from {}", path);

        DatasetLoadResult result = baseLoader.loadWithMetadata(path);

        // Validate expected size
        validateIMDBDataset(result.datasets(), "test", 25000);

        return result;
    }

    /**
     * Load stratified subset for quick experiments.
     *
     * Marcus's insight: "Not every experiment needs 25K examples."
     *
     * @param subset "train" or "test"
     * @param size number of examples to load (will be balanced)
     * @return stratified subset
     */
    public DatasetLoadResult loadSubset(String subset, int size) throws DataLoadingException {
        if (size <= 0 || size > 25000) {
            throw new IllegalArgumentException("Subset size must be between 1 and 25000, got: " + size);
        }

        String path = subset.equals("train") ?
            "src/main/resources/datasets/v1_raw/imdb_train.csv" :
            "src/main/resources/datasets/v1_raw/imdb_test.csv";

        logger.info("Loading IMDB {} subset: {} examples", subset, size);

        DatasetLoadResult fullResult = baseLoader.loadWithMetadata(path);
        List<Dataset> fullDataset = fullResult.datasets();

        // Create stratified subset (maintain 50/50 balance)
        int perClass = size / 2;

        List<Dataset> subset_data = fullDataset.stream()
            .filter(d -> d.getSentiment() == Dataset.SentimentLabel.POSITIVE)
            .limit(perClass)
            .collect(java.util.stream.Collectors.toList());

        subset_data.addAll(
            fullDataset.stream()
                .filter(d -> d.getSentiment() == Dataset.SentimentLabel.NEGATIVE)
                .limit(perClass)
                .toList()
        );

        // Shuffle to avoid ordering bias
        java.util.Collections.shuffle(subset_data);

        logger.info("Created balanced subset: {} examples ({} pos, {} neg)",
            subset_data.size(),
            perClass,
            perClass);

        return new DatasetLoadResult(
            subset_data,
            "IMDB " + subset + " subset",
            path,
            fullResult.loadTimeMs()
        );
    }

    /**
     * Validate IMDB dataset properties.
     *
     * Sofia's checklist:
     * ✓ Expected size (25K for full sets)
     * ✓ Label balance (50/50)
     * ✓ No empty texts
     */
    private void validateIMDBDataset(List<Dataset> datasets, String setName, int expectedSize)
            throws DataLoadingException {

        // Check size
        if (datasets.size() != expectedSize) {
            logger.warn("IMDB {} set size mismatch: expected {}, got {}",
                setName, expectedSize, datasets.size());
        }

        // Check label balance
        long posCount = datasets.stream()
            .filter(d -> d.getSentiment() == Dataset.SentimentLabel.POSITIVE)
            .count();

        long negCount = datasets.stream()
            .filter(d -> d.getSentiment() == Dataset.SentimentLabel.NEGATIVE)
            .count();

        double balanceRatio = (double) posCount / negCount;

        logger.info("IMDB {} set: {} total ({} positive, {} negative, ratio: {:.3f})",
            setName, datasets.size(), posCount, negCount, balanceRatio);

        // Warn if imbalanced (should be exactly 50/50 for IMDB)
        if (Math.abs(balanceRatio - 1.0) > 0.01) {
            logger.warn("⚠ Label imbalance detected in IMDB {} set!", setName);
        } else {
            logger.info("✓ Labels are balanced");
        }

        // Check for empty texts
        long emptyCount = datasets.stream()
            .filter(d -> d.getText() == null || d.getText().trim().isEmpty())
            .count();

        if (emptyCount > 0) {
            throw new DataLoadingException(
                String.format("Found %d empty texts in IMDB %s set", emptyCount, setName),
                "imdb_" + setName + ".csv",
                "CSV"
            );
        }

        logger.info("✓ No empty texts found");
    }

    /**
     * Load IMDB dataset with proper train/validation/test split.
     *
     * Aria's Principle: "Proper evaluation protocol requires 3-way split."
     * Sofia's Principle: "Stratification ensures representative splits."
     *
     * EVALUATION PROTOCOL:
     * ====================
     * - Train (60%): Model training
     * - Validation (20%): Hyperparameter tuning and model selection
     * - Test (20%): Final unbiased performance evaluation (use ONCE)
     *
     * This prevents information leakage where test set is used for model selection,
     * leading to overoptimistic performance estimates.
     *
     * @return IMDBDataSplit with train, validation, and test sets
     * @throws DataLoadingException if loading fails
     */
    public IMDBDataSplit loadWithValSplit() throws DataLoadingException {
        return loadWithValSplit(0.6, 0.2, 0.2, 42L);
    }

    /**
     * Load IMDB dataset with custom split ratios.
     *
     * @param trainRatio proportion for training (e.g., 0.6)
     * @param valRatio proportion for validation (e.g., 0.2)
     * @param testRatio proportion for test (e.g., 0.2)
     * @param randomSeed seed for reproducibility
     * @return IMDBDataSplit with train, validation, and test sets
     * @throws DataLoadingException if loading fails
     */
    public IMDBDataSplit loadWithValSplit(
            double trainRatio,
            double valRatio,
            double testRatio,
            long randomSeed) throws DataLoadingException {

        logger.info("Loading IMDB dataset with {}/{}/{} train/val/test split (seed={})",
                (int)(trainRatio * 100), (int)(valRatio * 100), (int)(testRatio * 100), randomSeed);

        // Load full training set (we'll split it ourselves)
        String trainPath = "src/main/resources/datasets/v1_raw/imdb_train.csv";
        DatasetLoadResult trainResult = baseLoader.loadWithMetadata(trainPath);
        List<Dataset> fullTrainData = trainResult.datasets();

        // Load test set (keep as-is for benchmark comparisons)
        String testPath = "src/main/resources/datasets/v1_raw/imdb_test.csv";
        DatasetLoadResult testResult = baseLoader.loadWithMetadata(testPath);
        List<Dataset> testData = testResult.datasets();

        logger.info("Loaded {} training examples, {} test examples",
                fullTrainData.size(), testData.size());

        // Perform stratified split on training set to create train/val
        // Recalculate ratios: if we want 60/20/20 overall and test is separate,
        // we need to split the training set into 75/25 (which gives 60/20 of total)
        double adjustedTrainRatio = trainRatio / (trainRatio + valRatio);
        double adjustedValRatio = valRatio / (trainRatio + valRatio);

        logger.info("Splitting training set: {:.1f}% train, {:.1f}% validation",
                adjustedTrainRatio * 100, adjustedValRatio * 100);

        DataSplit split = StratifiedDataSplitter.stratifiedSplit(
                fullTrainData,
                adjustedTrainRatio,
                adjustedValRatio,
                0.0,  // No test set from this split
                randomSeed
        );

        List<Dataset> trainSplit = split.train;
        List<Dataset> valSplit = split.validation;

        // Validate splits
        logger.info("Final split sizes: train={}, val={}, test={}",
                trainSplit.size(), valSplit.size(), testData.size());

        validateSplitDistribution(trainSplit, "train");
        validateSplitDistribution(valSplit, "validation");
        validateSplitDistribution(testData, "test");

        return new IMDBDataSplit(trainSplit, valSplit, testData);
    }

    /**
     * Validate that a split maintains label balance.
     */
    private void validateSplitDistribution(List<Dataset> split, String name) {
        long posCount = split.stream()
                .filter(d -> d.getSentiment() == Dataset.SentimentLabel.POSITIVE)
                .count();
        long negCount = split.stream()
                .filter(d -> d.getSentiment() == Dataset.SentimentLabel.NEGATIVE)
                .count();

        double balanceRatio = (double) Math.min(posCount, negCount) / Math.max(posCount, negCount);

        logger.info("{} split: {} examples ({} pos, {} neg, ratio={:.3f})",
                name, split.size(), posCount, negCount, balanceRatio);

        if (balanceRatio < 0.95) {
            logger.warn("⚠ {} split is imbalanced (ratio={:.3f})", name, balanceRatio);
        } else {
            logger.info("✓ {} split is balanced", name);
        }
    }

    /**
     * Get dataset statistics for reporting.
     *
     * Aria's insight: "Always understand your data distribution."
     */
    public DatasetStatistics computeStatistics(List<Dataset> datasets) {
        return DatasetStatistics.compute(datasets);
    }

    /**
     * Result record for 3-way split.
     *
     * USAGE:
     * ======
     * ```java
     * IMDBDataSplit split = loader.loadWithValSplit();
     *
     * // Train model
     * classifier.train(split.train());
     *
     * // Tune hyperparameters on validation set
     * tuneHyperparameters(classifier, split.validation());
     *
     * // Final evaluation on test set (ONCE!)
     * ClassifierEvaluationResult finalResult = classifier.evaluate(split.test());
     * ```
     */
    public record IMDBDataSplit(
            List<Dataset> train,
            List<Dataset> validation,
            List<Dataset> test
    ) {
        public IMDBDataSplit {
            if (train == null || validation == null || test == null) {
                throw new IllegalArgumentException("All splits must be non-null");
            }
            if (train.isEmpty() || validation.isEmpty() || test.isEmpty()) {
                throw new IllegalArgumentException("All splits must be non-empty");
            }
        }

        /**
         * Get total dataset size.
         */
        public int totalSize() {
            return train.size() + validation.size() + test.size();
        }

        /**
         * Get split summary for logging.
         */
        public String summary() {
            return String.format("Train: %,d | Val: %,d | Test: %,d | Total: %,d",
                    train.size(), validation.size(), test.size(), totalSize());
        }
    }
}
