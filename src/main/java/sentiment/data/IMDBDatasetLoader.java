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
 * <p>Dataset-specific loaders encapsulate domain knowledge. The IMDB dataset
 * has specific characteristics that require specialized handling:
 * <ul>
 *   <li>Binary sentiment (positive: rating >= 7/10, negative: rating <= 4/10)</li>
 *   <li>25,000 training samples, 25,000 test samples</li>
 *   <li>Balanced class distribution (50/50)</li>
 *   <li>Movie reviews with longer text and complex language</li>
 * </ul>
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
     * Loads the IMDB training set containing 25,000 reviews.
     *
     * @return dataset load result with training data and metadata
     * @throws DataLoadingException if loading or validation fails
     */
    public DatasetLoadResult loadTrainSet() throws DataLoadingException {
        String path = "src/main/resources/datasets/v1_raw/imdb_train.csv";
        logger.info("Loading IMDB training set from {}", path);

        DatasetLoadResult result = baseLoader.loadWithMetadata(path);

        // Validate expected size
        validateIMDBDataset(result.datasets(), "train");

        return result;
    }

    /**
     * Loads the IMDB test set containing 25,000 reviews.
     *
     * @return dataset load result with test data and metadata
     * @throws DataLoadingException if loading or validation fails
     */
    public DatasetLoadResult loadTestSet() throws DataLoadingException {
        String path = "src/main/resources/datasets/v1_raw/imdb_test.csv";
        logger.info("Loading IMDB test set from {}", path);

        DatasetLoadResult result = baseLoader.loadWithMetadata(path);

        // Validate expected size
        validateIMDBDataset(result.datasets(), "test");

        return result;
    }

    /**
     * Loads a stratified subset of the IMDB dataset for quick experiments.
     *
     * <p>Creates a balanced subset by sampling equal numbers of positive and negative
     * examples. The subset is shuffled to avoid ordering bias. This is useful for
     * rapid prototyping where the full 25,000 examples are not necessary.
     *
     * @param subset the dataset to load ("train" or "test")
     * @param size the number of examples to load (must be between 1 and 25,000; will be balanced 50/50)
     * @return dataset load result with stratified subset
     * @throws IllegalArgumentException if size is not between 1 and 25,000
     * @throws DataLoadingException if loading fails
     */
    @SuppressWarnings("unused") // Public API method
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
     * Validates IMDB dataset properties and logs quality metrics.
     *
     * <p>Performs validation checks on:
     * <ul>
     *   <li>Expected size (25,000 for full sets)</li>
     *   <li>Label balance (should be 50/50 positive/negative)</li>
     *   <li>No empty or null text content</li>
     * </ul>
     *
     * @param datasets the list of datasets to validate
     * @param setName the name of the dataset set (e.g., "train", "test")
     * @throws DataLoadingException if validation fails critically (e.g., empty texts found)
     */
    private void validateIMDBDataset(List<Dataset> datasets, String setName)
            throws DataLoadingException {

        // Check size (IMDB full sets are always 25,000)
        int expectedSize = 25000;
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

        logger.info("IMDB {} set: {} total ({} positive, {} negative, ratio: {})",
            setName, datasets.size(), posCount, negCount, String.format("%.3f", balanceRatio));

        // Warn if imbalanced (should be exactly 50/50 for IMDB)
        if (Math.abs(balanceRatio - 1.0) > 0.01) {
            logger.warn(" Label imbalance detected in IMDB {} set!", setName);
        } else {
            logger.info(" Labels are balanced");
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

        logger.info(" No empty texts found");
    }

    /**
     * Loads IMDB dataset with proper train/validation/test split using default ratios.
     *
     * <p>Proper evaluation protocol requires a 3-way split with stratification to ensure
     * representative splits. The evaluation protocol is:
     * <ul>
     *   <li><b>Train (60%)</b>: Model training</li>
     *   <li><b>Validation (20%)</b>: Hyperparameter tuning and model selection</li>
     *   <li><b>Test (20%)</b>: Final unbiased performance evaluation (use ONCE only)</li>
     * </ul>
     *
     * <p>This prevents information leakage where the test set is used for model selection,
     * which would lead to overoptimistic performance estimates.
     *
     * @return IMDB data split with train, validation, and test sets
     * @throws DataLoadingException if loading or splitting fails
     */
    @SuppressWarnings("unused") // Public API method
    public IMDBDataSplit loadWithValSplit() throws DataLoadingException {
        return loadWithValSplit(0.6, 0.2, 0.2, 42L);
    }

    /**
     * Loads IMDB dataset with custom split ratios and random seed.
     *
     * <p>Combines both IMDB train and test files (50K total) and splits them
     * according to the specified ratios. This allows true custom splits but
     * means the test set won't match IMDB's official test set.
     *
     * @param trainRatio the proportion for training (e.g., 0.6 for 60%)
     * @param valRatio the proportion for validation (e.g., 0.2 for 20%)
     * @param testRatio the proportion for test (e.g., 0.2 for 20%)
     * @param randomSeed the random seed for reproducibility
     * @return IMDB data split with train, validation, and test sets
     * @throws IllegalArgumentException if ratios don't sum to 1.0
     * @throws DataLoadingException if loading or splitting fails
     */
    public IMDBDataSplit loadWithValSplit(
            double trainRatio,
            double valRatio,
            double testRatio,
            long randomSeed) throws DataLoadingException {

        // Validate ratios sum to 1.0
        double sum = trainRatio + valRatio + testRatio;
        if (Math.abs(sum - 1.0) > 0.001) {
            throw new IllegalArgumentException(
                String.format("Ratios must sum to 1.0, got: %.3f (train=%.2f, val=%.2f, test=%.2f)",
                    sum, trainRatio, valRatio, testRatio));
        }

        logger.info("Loading IMDB dataset with {}/{}/{} train/val/test split (seed={})",
                (int)(trainRatio * 100), (int)(valRatio * 100), (int)(testRatio * 100), randomSeed);

        // Load both IMDB files to get full 50K dataset
        String trainPath = "src/main/resources/datasets/v1_raw/imdb_train.csv";
        DatasetLoadResult trainResult = baseLoader.loadWithMetadata(trainPath);
        List<Dataset> trainData = trainResult.datasets();

        String testPath = "src/main/resources/datasets/v1_raw/imdb_test.csv";
        DatasetLoadResult testResult = baseLoader.loadWithMetadata(testPath);
        List<Dataset> testData = testResult.datasets();

        // Combine both files for true custom split
        List<Dataset> allData = new java.util.ArrayList<>(trainData);
        allData.addAll(testData);

        logger.info("Loaded {} total examples (combining train + test files)",
                allData.size());

        // Perform stratified 3-way split on combined dataset
        DataSplit split = StratifiedDataSplitter.stratifiedSplit(
                allData,
                trainRatio,
                valRatio,
                testRatio,
                randomSeed
        );

        List<Dataset> trainSplit = split.train;
        List<Dataset> valSplit = split.validation;
        List<Dataset> testSplit = split.test;

        // Validate splits
        logger.info("Final split sizes: train={}, val={}, test={} (total={})",
                trainSplit.size(), valSplit.size(), testSplit.size(), allData.size());

        // Verify actual ratios
        double actualTrainRatio = (double) trainSplit.size() / allData.size();
        double actualValRatio = (double) valSplit.size() / allData.size();
        double actualTestRatio = (double) testSplit.size() / allData.size();

        logger.info("Actual split ratios: {}%/{}%/{}%",
                String.format("%.1f", actualTrainRatio * 100),
                String.format("%.1f", actualValRatio * 100),
                String.format("%.1f", actualTestRatio * 100));

        validateSplitDistribution(trainSplit, "train");
        validateSplitDistribution(valSplit, "validation");
        validateSplitDistribution(testSplit, "test");

        return new IMDBDataSplit(trainSplit, valSplit, testSplit);
    }

    /**
     * Validates that a split maintains label balance and logs distribution metrics.
     *
     * @param split the dataset split to validate
     * @param name the name of the split for logging (e.g., "train", "validation", "test")
     */
    private void validateSplitDistribution(List<Dataset> split, String name) {
        long posCount = split.stream()
                .filter(d -> d.getSentiment() == Dataset.SentimentLabel.POSITIVE)
                .count();
        long negCount = split.stream()
                .filter(d -> d.getSentiment() == Dataset.SentimentLabel.NEGATIVE)
                .count();

        double balanceRatio = (double) Math.min(posCount, negCount) / Math.max(posCount, negCount);

        logger.info("{} split: {} examples ({} pos, {} neg, ratio={})",
                name, split.size(), posCount, negCount, String.format("%.3f", balanceRatio));

        if (balanceRatio < 0.95) {
            logger.warn(" {} split is imbalanced (ratio={})", name, String.format("%.3f", balanceRatio));
        } else {
            logger.info(" {} split is balanced", name);
        }
    }

    /**
     * Computes comprehensive dataset statistics for quality reporting.
     *
     * <p>Statistics include label distribution, text length metrics, duplicate detection,
     * and vocabulary diversity. Use this to understand your data distribution before training.
     *
     * @param datasets the list of datasets to analyze
     * @return dataset statistics with comprehensive metrics
     */
    public DatasetStatistics computeStatistics(List<Dataset> datasets) {
        return DatasetStatistics.compute(datasets);
    }

    /**
     * Record encapsulating a 3-way dataset split (train, validation, test).
     *
     * <p><b>Usage Example:</b>
     * <pre>{@code
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
     * }</pre>
     *
     * @param train the training dataset
     * @param validation the validation dataset for hyperparameter tuning
     * @param test the test dataset for final evaluation
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
         * Returns the total number of samples across all splits.
         *
         * @return combined size of train, validation, and test sets
         */
        public int totalSize() {
            return train.size() + validation.size() + test.size();
        }

        /**
         * Returns a formatted summary of the split sizes for logging.
         *
         * @return summary string with split sizes
         */
        public String summary() {
            return String.format("Train: %,d | Val: %,d | Test: %,d | Total: %,d",
                    train.size(), validation.size(), test.size(), totalSize());
        }
    }
}
