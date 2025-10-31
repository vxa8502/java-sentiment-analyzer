package sentiment.models;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.functions.SMO;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;

import java.util.Random;

/**
 * Thread-safe cross-validation utility for sentiment classifiers.
 *
 * ARCHITECTURAL DECISION:
 * ======================
 * Cross-validation is separated from the main classifier because:
 *
 * 1. Thread Safety: Creates independent classifier instances per fold,
 *    avoiding concurrent training conflicts with the main classifier
 *
 * 2. State Isolation: Each fold gets a fresh classifier instance,
 *    preventing state pollution between folds
 *
 * 3. Template Pattern Compliance: Doesn't bypass the training template's
 *    state machine and locking mechanisms
 *
 * 4. Separation of Concerns: Cross-validation is a model evaluation technique,
 *    not a core classification operation
 *
 * USAGE:
 * ======
 * <pre>
 * CrossValidator cvUtil = new CrossValidator(preprocessor, converter);
 * CrossValidationResult result = cvUtil.performCrossValidation(data, 10);
 * System.out.println(result.getSummary());
 * </pre>
 *
 * THREAD SAFETY:
 * ==============
 * ✅ Each fold creates its own classifier instance
 * ✅ No shared mutable state between folds
 * ✅ Safe to run multiple cross-validations concurrently
 */
public class CrossValidator {

    private static final Logger logger = LoggerFactory.getLogger(CrossValidator.class);

    private final TextPreprocessor preprocessor;
    private final WekaInstancesConverter converter;

    /**
     * Creates a new cross-validation utility.
     *
     * @param preprocessor Text preprocessor for feature extraction
     * @param converter Weka instances converter for data transformation
     */
    public CrossValidator(TextPreprocessor preprocessor, WekaInstancesConverter converter) {
        if (preprocessor == null || converter == null) {
            throw new IllegalArgumentException("Preprocessor and converter cannot be null");
        }
        this.preprocessor = preprocessor;
        this.converter = converter;

        logger.info("Created CrossValidationUtil - thread-safe fold isolation enabled");
    }

    /**
     * Performs stratified k-fold cross-validation on the given dataset.
     *
     * THREAD SAFETY: This method is thread-safe. Each fold creates an independent
     * classifier instance, so multiple concurrent cross-validations are safe.
     *
     * @param data Training data for cross-validation
     * @param numFolds Number of folds (typically 5 or 10)
     * @param randomSeed Random seed for reproducibility
     * @return Cross-validation results with per-fold and aggregate metrics
     * @throws Exception if validation fails
     */
    public CrossValidationResult performCrossValidation(Instances data, int numFolds, long randomSeed)
            throws Exception {

        validateInputs(data, numFolds);

        logger.info("Starting {}-fold cross-validation on {} instances (seed={})",
                numFolds, data.numInstances(), randomSeed);

        // Prepare stratified folds
        Instances shuffledData = prepareShuffledData(data, numFolds, randomSeed);
        Random random = new Random(randomSeed);

        // Track metrics for each fold
        double[] foldAccuracies = new double[numFolds];
        double[] foldMacroF1s = new double[numFolds];
        long[] foldTrainingTimes = new long[numFolds];
        long totalCVTime = 0;

        // Process each fold independently
        for (int fold = 0; fold < numFolds; fold++) {
            logger.info("Processing fold {}/{}", fold + 1, numFolds);
            long foldStartTime = System.currentTimeMillis();

            // Create train/test split for this fold
            Instances trainSet = shuffledData.trainCV(numFolds, fold, random);
            Instances testSet = shuffledData.testCV(numFolds, fold);

            // ✅ CRITICAL: Create independent classifier for this fold
            // Each fold gets its own SMO instance - no shared state!
            SMO foldClassifier = createAndTrainFoldClassifier(trainSet, fold + 1);

            // Record training time
            long trainingTime = System.currentTimeMillis() - foldStartTime;
            foldTrainingTimes[fold] = trainingTime;

            // Evaluate on test set
            FoldMetrics metrics = evaluateFold(foldClassifier, trainSet, testSet, fold + 1);
            foldAccuracies[fold] = metrics.accuracy;
            foldMacroF1s[fold] = metrics.macroF1;

            // Track total time
            long foldTotalTime = System.currentTimeMillis() - foldStartTime;
            totalCVTime += foldTotalTime;

            String accuracy = String.format("%.3f", foldAccuracies[fold]);
            String macroF1 = String.format("%.3f", foldMacroF1s[fold]);
            logger.info("Fold {}/{} complete: accuracy={}, macro-F1={}, time={}ms",
                    fold + 1, numFolds, accuracy, macroF1, foldTotalTime);
        }

        // Calculate aggregate statistics
        CrossValidationResult result = aggregateResults(
                numFolds, foldAccuracies, foldMacroF1s, foldTrainingTimes,
                totalCVTime, randomSeed);

        String accuracy = String.format("%.3f±%.3f", result.avgAccuracy, result.stdAccuracy);
        String macroF1 = String.format("%.3f±%.3f", result.avgMacroF1, result.stdMacroF1);
        logger.info("Cross-validation complete: accuracy={}, macro-F1={}, total={}ms",
                accuracy, macroF1, totalCVTime);

        return result;
    }

    /**
     * Convenience method with default random seed.
     */
    public CrossValidationResult performCrossValidation(Instances data, int numFolds)
            throws Exception {
        return performCrossValidation(data, numFolds, 42L);
    }

    // ==================== PRIVATE HELPER METHODS ====================

    private void validateInputs(Instances data, int numFolds) {
        if (data == null || data.numInstances() == 0) {
            throw new IllegalArgumentException("Data cannot be null or empty");
        }

        if (numFolds < 2 || numFolds > data.numInstances()) {
            throw new IllegalArgumentException(
                    String.format("Invalid numFolds=%d for dataset with %d instances",
                            numFolds, data.numInstances()));
        }

        if (data.classIndex() == -1) {
            throw new IllegalArgumentException("Data must have a class attribute set");
        }
    }

    private Instances prepareShuffledData(Instances data, int numFolds, long randomSeed) {
        logger.debug("Preparing stratified folds with seed={}", randomSeed);

        Instances shuffledData = new Instances(data);
        Random random = new Random(randomSeed);
        shuffledData.randomize(random);

        // Stratify to maintain class distribution in folds
        if (numFolds <= data.numInstances()) {
            shuffledData.stratify(numFolds);
            logger.debug("✅ Data stratified into {} folds", numFolds);
        } else {
            logger.warn("⚠️ Cannot stratify: numFolds ({}) > instances ({})",
                    numFolds, data.numInstances());
        }

        return shuffledData;
    }

    /**
     * Creates and trains an independent SMO classifier for a single fold.
     *
     * ✅ THREAD SAFETY: Each fold gets its own classifier instance.
     * No shared state with other folds or the main classifier.
     */
    private SMO createAndTrainFoldClassifier(Instances trainSet, int foldNumber) throws Exception {
        logger.debug("Training fold {} classifier on {} instances",
                foldNumber, trainSet.numInstances());

        // Create fresh SMO instance for this fold
        SMO foldSMO = new SMO();

        // Configure parameters (same logic as SVMClassifier)
        double complexityC = calculateOptimalC(trainSet.numInstances());
        foldSMO.setC(complexityC);
        logger.trace("  Fold {} C = {}", foldNumber, complexityC);

        double epsilon = calculateOptimalEpsilon(trainSet.numInstances());
        foldSMO.setEpsilon(epsilon);
        logger.trace("  Fold {} epsilon = {}", foldNumber, epsilon);

        foldSMO.setFilterType(new weka.core.SelectedTag(SMO.FILTER_NORMALIZE, SMO.TAGS_FILTER));
        logger.trace("  Fold {} normalization enabled", foldNumber);

        // Train the classifier
        foldSMO.buildClassifier(trainSet);
        logger.debug("✅ Fold {} classifier trained successfully", foldNumber);

        return foldSMO;
    }

    private FoldMetrics evaluateFold(SMO classifier, Instances trainStructure,
                                     Instances testSet, int foldNumber) throws Exception {
        logger.debug("Evaluating fold {} on {} test instances", foldNumber, testSet.numInstances());

        Evaluation evaluation = new Evaluation(trainStructure);
        evaluation.evaluateModel(classifier, testSet);

        double accuracy = evaluation.pctCorrect() / 100.0;

        // Calculate macro-averaged F1
        double foldF1Sum = 0.0;
        int numClasses = testSet.classAttribute().numValues();
        for (int i = 0; i < numClasses; i++) {
            double f1 = evaluation.fMeasure(i);
            foldF1Sum += Double.isNaN(f1) ? 0.0 : f1;
        }
        double macroF1 = foldF1Sum / numClasses;

        String accuracyStr = String.format("%.3f", accuracy);
        String macroF1Str = String.format("%.3f", macroF1);
        logger.debug("✅ Fold {} evaluation complete: accuracy={}, macro-F1={}",
                foldNumber, accuracyStr, macroF1Str);

        return new FoldMetrics(accuracy, macroF1);
    }

    private CrossValidationResult aggregateResults(int numFolds, double[] foldAccuracies,
                                                   double[] foldMacroF1s, long[] foldTrainingTimes,
                                                   long totalCVTime, long randomSeed) {

        double avgAccuracy = calculateMean(foldAccuracies);
        double stdAccuracy = calculateStdDev(foldAccuracies, avgAccuracy);

        double avgMacroF1 = calculateMean(foldMacroF1s);
        double stdMacroF1 = calculateStdDev(foldMacroF1s, avgMacroF1);

        long avgTrainingTime = calculateMean(foldTrainingTimes);

        return new CrossValidationResult(
                numFolds,
                foldAccuracies,
                foldMacroF1s,
                foldTrainingTimes,
                avgAccuracy,
                stdAccuracy,
                avgMacroF1,
                stdMacroF1,
                avgTrainingTime,
                totalCVTime,
                randomSeed
        );
    }

    // Statistical calculations
    private double calculateMean(double[] values) {
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum / values.length;
    }

    private long calculateMean(long[] values) {
        long sum = 0;
        for (long value : values) {
            sum += value;
        }
        return sum / values.length;
    }

    private double calculateStdDev(double[] values, double mean) {
        if (values.length < 2) return 0.0;

        double sumSquaredDiffs = 0.0;
        for (double value : values) {
            double diff = value - mean;
            sumSquaredDiffs += diff * diff;
        }

        return Math.sqrt(sumSquaredDiffs / (values.length - 1));
    }

    // Parameter calculation (same as SVMClassifier)
    private double calculateOptimalC(int numInstances) {
        if (numInstances < 1000) return 0.5;
        else if (numInstances < 10000) return 1.0;
        else return 2.0;
    }

    private double calculateOptimalEpsilon(int numInstances) {
        if (numInstances < 1000) return 1.0E-10;
        else return 1.0E-12;
    }

    // ==================== INNER CLASSES ====================

    /**
     * Metrics for a single fold
     */
    private static class FoldMetrics {
        final double accuracy;
        final double macroF1;

        FoldMetrics(double accuracy, double macroF1) {
            this.accuracy = accuracy;
            this.macroF1 = macroF1;
        }
    }

    /**
     * Complete cross-validation results
     */
    public static class CrossValidationResult {
        public final int numFolds;
        public final double[] foldAccuracies;
        public final double[] foldMacroF1s;
        public final long[] foldTrainingTimes;

        public final double avgAccuracy;
        public final double stdAccuracy;
        public final double avgMacroF1;
        public final double stdMacroF1;
        public final long avgTrainingTimeMs;
        public final long totalCVTimeMs;
        public final long randomSeed;

        public CrossValidationResult(
                int numFolds,
                double[] foldAccuracies,
                double[] foldMacroF1s,
                long[] foldTrainingTimes,
                double avgAccuracy,
                double stdAccuracy,
                double avgMacroF1,
                double stdMacroF1,
                long avgTrainingTimeMs,
                long totalCVTimeMs,
                long randomSeed) {

            this.numFolds = numFolds;
            this.foldAccuracies = foldAccuracies.clone();
            this.foldMacroF1s = foldMacroF1s.clone();
            this.foldTrainingTimes = foldTrainingTimes.clone();
            this.avgAccuracy = avgAccuracy;
            this.stdAccuracy = stdAccuracy;
            this.avgMacroF1 = avgMacroF1;
            this.stdMacroF1 = stdMacroF1;
            this.avgTrainingTimeMs = avgTrainingTimeMs;
            this.totalCVTimeMs = totalCVTimeMs;
            this.randomSeed = randomSeed;
        }


        public String getSummary() {
            String nl = System.lineSeparator();
            StringBuilder summary = new StringBuilder();
            summary.append(String.format("=== %d-Fold Cross-Validation Results ===%s", numFolds, nl));
            summary.append(String.format("Random Seed: %d%s%s", randomSeed, nl, nl));

            summary.append("AGGREGATE METRICS:").append(nl);
            summary.append(String.format("  Accuracy:  %.3f ± %.3f%s", avgAccuracy, stdAccuracy, nl));
            summary.append(String.format("  Macro F1:  %.3f ± %.3f%s", avgMacroF1, stdMacroF1, nl));
            summary.append(String.format("  Avg Training Time: %d ms%s", avgTrainingTimeMs, nl));
            summary.append(String.format("  Total CV Time: %d ms (%.2f sec)%s%s",
                    totalCVTimeMs, totalCVTimeMs / 1000.0, nl, nl));

            summary.append("PER-FOLD BREAKDOWN:").append(nl);
            for (int i = 0; i < numFolds; i++) {
                summary.append(String.format("  Fold %d: accuracy=%.3f, F1=%.3f, time=%dms%s",
                        i + 1, foldAccuracies[i], foldMacroF1s[i], foldTrainingTimes[i], nl));
            }

            return summary.toString();
        }

        public boolean isConsistent(double maxStdDev) {
            return stdAccuracy <= maxStdDev && stdMacroF1 <= maxStdDev;
        }

        public int getBestFold() {
            int bestFold = 0;
            double bestAccuracy = foldAccuracies[0];

            for (int i = 1; i < numFolds; i++) {
                if (foldAccuracies[i] > bestAccuracy) {
                    bestAccuracy = foldAccuracies[i];
                    bestFold = i;
                }
            }

            return bestFold;
        }

        public int getWorstFold() {
            int worstFold = 0;
            double worstAccuracy = foldAccuracies[0];

            for (int i = 1; i < numFolds; i++) {
                if (foldAccuracies[i] < worstAccuracy) {
                    worstAccuracy = foldAccuracies[i];
                    worstFold = i;
                }
            }

            return worstFold;
        }

        @Override
        public String toString() {
            return String.format(
                    "CrossValidationResult{folds=%d, accuracy=%.3f±%.3f, F1=%.3f±%.3f, time=%dms}",
                    numFolds, avgAccuracy, stdAccuracy, avgMacroF1, stdMacroF1, totalCVTimeMs
            );
        }
    }
}