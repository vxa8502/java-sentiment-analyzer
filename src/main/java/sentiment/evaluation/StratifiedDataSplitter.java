package sentiment.evaluation;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sentiment.data.Dataset;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Utility for creating stratified train/validation/test splits while preserving
 * class distribution across all subsets.
 * <p>
 * Stratification ensures each split maintains the same class proportions as the
 * original dataset, which is critical for imbalanced data and accurate evaluation.
 * <p>
 * The test set should be held out before any model selection or hyperparameter
 * tuning to prevent information leakage and ensure unbiased performance estimates.
 */
public class StratifiedDataSplitter {

    private static final Logger logger = LoggerFactory.getLogger(StratifiedDataSplitter.class);

    /**
     * Creates a stratified three-way split preserving class distribution in each subset.
     * Reproducible when using the same seed.
     *
     * @param data the full dataset to split
     * @param trainRatio proportion for training (e.g., 0.6)
     * @param valRatio proportion for validation (e.g., 0.2)
     * @param testRatio proportion for test (e.g., 0.2); ratios must sum to 1.0
     * @param randomSeed seed for reproducibility
     * @return data split containing train, validation, and test sets
     * @throws IllegalArgumentException if ratios don't sum to 1.0 or are negative
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

        if (trainRatio <= 0 || valRatio < 0 || testRatio < 0) {
            throw new IllegalArgumentException("All ratios must be non-negative and trainRatio must be positive");
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
     * Creates a stratified two-way split (train/test only) with an empty validation set.
     *
     * @param data the full dataset to split
     * @param trainRatio proportion for training (e.g., 0.8)
     * @param randomSeed seed for reproducibility
     * @return data split with train and test sets; validation set is empty
     */
    public static DataSplit stratifiedSplit(
            List<Dataset> data,
            double trainRatio,
            long randomSeed) {

        return stratifiedSplit(data, trainRatio, 0.0, 1.0 - trainRatio, randomSeed);
    }

    /**
     * Creates k-fold stratified cross-validation splits for model evaluation.
     * Each fold contains a train/test pair with stratified class distribution.
     * Intended for use on combined train+validation data, not on held-out test sets.
     *
     * @param data the dataset to split into folds
     * @param k number of folds (typically 5 or 10); must be at least 2
     * @param randomSeed seed for reproducibility
     * @return list of k folds, each containing a train and test split
     * @throws IllegalArgumentException if k < 2
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
     * Counts the number of samples per class in the given dataset.
     *
     * @param data the dataset to count
     * @return a map from sentiment label to count
     */
    private static Map<Dataset.SentimentLabel, Long> countByClass(List<Dataset> data) {
        if (data.isEmpty()) {
            return Map.of();
        }
        return data.stream()
                .collect(Collectors.groupingBy(Dataset::getSentiment, Collectors.counting()));
    }

    /**
     * Verifies and logs that stratification preserved class distribution across splits.
     *
     * @param original the original dataset
     * @param train the training split
     * @param val the validation split (possibly empty)
     * @param test the test split
     */
    private static void verifyStratification(
            List<Dataset> original,
            List<Dataset> train,
            List<Dataset> val,
            List<Dataset> test) {

        Map<Dataset.SentimentLabel, Long> origDist = countByClass(original);
        Map<Dataset.SentimentLabel, Long> trainDist = countByClass(train);
        Map<Dataset.SentimentLabel, Long> valDist = countByClass(val);
        Map<Dataset.SentimentLabel, Long> testDist = countByClass(test);

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
     * Immutable container for train, validation, and test data splits.
     * Defensive copies are made to ensure immutability.
     */
    public static class DataSplit {
        public final List<Dataset> train;
        public final List<Dataset> validation;
        public final List<Dataset> test;

        /**
         * Creates an immutable data split with defensive copies.
         *
         * @param train the training data
         * @param validation the validation data
         * @param test the test data
         */
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
