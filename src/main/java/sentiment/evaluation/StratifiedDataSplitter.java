package sentiment.evaluation;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sentiment.data.Dataset;

import java.util.*;
import java.util.stream.Collectors;

/**
 * ARIA'S MATHEMATICAL RIGOR: Proper Data Splitting for Unbiased Evaluation
 * =========================================================================
 *
 * "The model is only as honest as its data splitting strategy."
 *
 * CRITICAL PRINCIPLE:
 * ===================
 * Test set must be held out BEFORE any model selection or hyperparameter tuning.
 * Otherwise, information leakage causes overoptimistic performance estimates.
 *
 * PROPER WORKFLOW:
 * ================
 * 1. Split data: 60% train, 20% validation, 20% test
 * 2. Use train set for model training
 * 3. Use validation set for hyperparameter tuning and model selection
 * 4. Use test set ONCE for final unbiased performance estimate
 * 5. For robust estimates, use k-fold CV on train+val (NOT test)
 *
 * STRATIFICATION:
 * ===============
 * Ensures class distribution is preserved across splits.
 * Critical for imbalanced datasets.
 *
 * EXAMPLE:
 * ========
 * ```java
 * List<Dataset> data = loadData();
 *
 * // 60/20/20 split with stratification
 * DataSplit split = StratifiedDataSplitter.stratifiedSplit(data, 0.6, 0.2, 0.2, 42);
 *
 * // Train on train set
 * classifier.train(split.train);
 *
 * // Tune hyperparameters on validation set
 * tuneHyperparameters(classifier, split.validation);
 *
 * // Final evaluation on test set (ONCE)
 * ClassifierEvaluationResult finalResult = classifier.evaluate(split.test);
 * ```
 *
 * @author Aria (Mathematical Foundations Mentor)
 */
public class StratifiedDataSplitter {

    private static final Logger logger = LoggerFactory.getLogger(StratifiedDataSplitter.class);

    /**
     * Stratified train/validation/test split.
     *
     * GUARANTEES:
     * - Class distribution preserved in each split
     * - No data leakage between splits
     * - Reproducible with fixed seed
     *
     * @param data Full dataset
     * @param trainRatio Proportion for training (e.g., 0.6)
     * @param valRatio Proportion for validation (e.g., 0.2)
     * @param testRatio Proportion for test (e.g., 0.2)
     * @param randomSeed Seed for reproducibility
     * @return DataSplit with train, validation, and test sets
     */
    public static DataSplit stratifiedSplit(
            List<Dataset> data,
            double trainRatio,
            double valRatio,
            double testRatio,
            long randomSeed) {

        // Validate input
        if (data == null) {
            throw new IllegalArgumentException("Data cannot be null");
        }
        if (data.isEmpty()) {
            throw new IllegalArgumentException("Data cannot be empty");
        }

        // Validate ratios
        if (Math.abs(trainRatio + valRatio + testRatio - 1.0) > 1e-6) {
            throw new IllegalArgumentException(
                    String.format("Ratios must sum to 1.0, got %.3f", trainRatio + valRatio + testRatio));
        }

        if (trainRatio < 0 || valRatio < 0 || testRatio < 0) {
            throw new IllegalArgumentException("Ratios cannot be negative");
        }

        if (trainRatio <= 0 || valRatio < 0 || testRatio < 0) {
            throw new IllegalArgumentException("All ratios must be non-negative");
        }

        // Group by class
        Map<Dataset.SentimentLabel, List<Dataset>> byClass = data.stream()
                .collect(Collectors.groupingBy(Dataset::getSentiment));

        logger.info("Performing stratified split: train={}%, val={}%, test={}%",
                trainRatio * 100, valRatio * 100, testRatio * 100);

        List<Dataset> trainSet = new ArrayList<>();
        List<Dataset> valSet = new ArrayList<>();
        List<Dataset> testSet = new ArrayList<>();

        Random random = new Random(randomSeed);

        // Split each class proportionally
        for (Map.Entry<Dataset.SentimentLabel, List<Dataset>> entry : byClass.entrySet()) {
            Dataset.SentimentLabel label = entry.getKey();
            List<Dataset> classData = new ArrayList<>(entry.getValue());

            // Shuffle within class
            Collections.shuffle(classData, random);

            int total = classData.size();
            int trainSize = (int) (total * trainRatio);
            int valSize = (int) (total * valRatio);

            // Split
            List<Dataset> classTrain = classData.subList(0, trainSize);
            List<Dataset> classVal = classData.subList(trainSize, trainSize + valSize);
            List<Dataset> classTest = classData.subList(trainSize + valSize, total);

            trainSet.addAll(classTrain);
            valSet.addAll(classVal);
            testSet.addAll(classTest);

            logger.debug("Class {}: train={}, val={}, test={}",
                    label, classTrain.size(), classVal.size(), classTest.size());
        }

        // Shuffle combined sets (within each split)
        Collections.shuffle(trainSet, random);
        Collections.shuffle(valSet, random);
        Collections.shuffle(testSet, random);

        logger.info("Split complete: train={}, val={}, test={}",
                trainSet.size(), valSet.size(), testSet.size());

        // Verify class distribution
        verifyStratification(data, trainSet, valSet, testSet);

        return new DataSplit(trainSet, valSet, testSet);
    }

    /**
     * Two-way stratified split (train/test only).
     *
     * @param data Full dataset
     * @param trainRatio Proportion for training (e.g., 0.8)
     * @param randomSeed Seed for reproducibility
     * @return DataSplit with validation set empty
     */
    public static DataSplit stratifiedSplit(
            List<Dataset> data,
            double trainRatio,
            long randomSeed) {

        return stratifiedSplit(data, trainRatio, 0.0, 1.0 - trainRatio, randomSeed);
    }

    /**
     * K-fold stratified cross-validation splits.
     *
     * USAGE:
     * For model evaluation WITHOUT a held-out test set.
     * Typically used on train+val data, never on test data.
     *
     * @param data Dataset to split
     * @param k Number of folds (typically 5 or 10)
     * @param randomSeed Seed for reproducibility
     * @return List of k folds, each containing train and test split
     */
    public static List<DataSplit> stratifiedKFold(
            List<Dataset> data,
            int k,
            long randomSeed) {

        if (k < 2) {
            throw new IllegalArgumentException("k must be >= 2 for k-fold CV");
        }

        // Group by class
        Map<Dataset.SentimentLabel, List<Dataset>> byClass = data.stream()
                .collect(Collectors.groupingBy(Dataset::getSentiment));

        logger.info("Creating {}-fold stratified CV splits from {} samples", k, data.size());

        // Create k empty folds
        List<List<Dataset>> folds = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            folds.add(new ArrayList<>());
        }

        Random random = new Random(randomSeed);

        // Distribute each class across folds
        for (Map.Entry<Dataset.SentimentLabel, List<Dataset>> entry : byClass.entrySet()) {
            List<Dataset> classData = new ArrayList<>(entry.getValue());
            Collections.shuffle(classData, random);

            // Round-robin assignment to folds
            for (int i = 0; i < classData.size(); i++) {
                int foldIdx = i % k;
                folds.get(foldIdx).add(classData.get(i));
            }
        }

        // Create train/test splits for each fold
        List<DataSplit> cvSplits = new ArrayList<>();
        for (int testFoldIdx = 0; testFoldIdx < k; testFoldIdx++) {
            List<Dataset> testSet = folds.get(testFoldIdx);
            List<Dataset> trainSet = new ArrayList<>();

            // Combine all other folds as training set
            for (int i = 0; i < k; i++) {
                if (i != testFoldIdx) {
                    trainSet.addAll(folds.get(i));
                }
            }

            cvSplits.add(new DataSplit(trainSet, List.of(), testSet));

            logger.debug("Fold {}/{}: train={}, test={}",
                    testFoldIdx + 1, k, trainSet.size(), testSet.size());
        }

        return cvSplits;
    }

    /**
     * Verify that stratification preserved class distribution.
     */
    private static void verifyStratification(
            List<Dataset> original,
            List<Dataset> train,
            List<Dataset> val,
            List<Dataset> test) {

        Map<Dataset.SentimentLabel, Long> origDist = original.stream()
                .collect(Collectors.groupingBy(Dataset::getSentiment, Collectors.counting()));

        Map<Dataset.SentimentLabel, Long> trainDist = train.stream()
                .collect(Collectors.groupingBy(Dataset::getSentiment, Collectors.counting()));

        Map<Dataset.SentimentLabel, Long> valDist = val.isEmpty() ? Map.of() : val.stream()
                .collect(Collectors.groupingBy(Dataset::getSentiment, Collectors.counting()));

        Map<Dataset.SentimentLabel, Long> testDist = test.stream()
                .collect(Collectors.groupingBy(Dataset::getSentiment, Collectors.counting()));

        logger.info("=== Class Distribution Verification ===");
        for (Dataset.SentimentLabel label : origDist.keySet()) {
            double origPct = 100.0 * origDist.get(label) / original.size();
            double trainPct = 100.0 * trainDist.getOrDefault(label, 0L) / train.size();
            double valPct = val.isEmpty() ? 0.0 : 100.0 * valDist.getOrDefault(label, 0L) / val.size();
            double testPct = 100.0 * testDist.getOrDefault(label, 0L) / test.size();

            logger.info("{}: original={}%, train={}%, val={}%, test={}%",
                    label, origPct, trainPct, valPct, testPct);
        }
    }

    /**
     * Result of data splitting.
     */
    public static class DataSplit {
        public final List<Dataset> train;
        public final List<Dataset> validation;
        public final List<Dataset> test;

        public DataSplit(List<Dataset> train, List<Dataset> validation, List<Dataset> test) {
            this.train = List.copyOf(train);
            this.validation = List.copyOf(validation);
            this.test = List.copyOf(test);
        }

        public int totalSize() {
            return train.size() + validation.size() + test.size();
        }

        @Override
        public String toString() {
            return String.format("DataSplit{train=%d, val=%d, test=%d, total=%d}",
                    train.size(), validation.size(), test.size(), totalSize());
        }
    }
}
