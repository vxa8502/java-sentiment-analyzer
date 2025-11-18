package sentiment.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import sentiment.data.Dataset;
import sentiment.data.SimpleDatasetLoader;
import sentiment.data.DatasetLoadResult;
import sentiment.evaluation.StratifiedDataSplitter;
import sentiment.models.*;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Utility class for training and serializing sentiment analysis models offline.
 *
 * PURPOSE:
 * ========
 * This class enables pre-training models before application startup, eliminating
 * the need for training during server initialization. This approach:
 * - Reduces startup time dramatically (seconds instead of minutes)
 * - Enables model versioning and A/B testing
 * - Separates training concerns from serving concerns
 * - Allows training on larger datasets without impacting production
 *
 * USAGE:
 * ======
 * 1. Standalone: Run ModelTrainingCLI.main() to train all models
 * 2. Programmatic: Use this class in your own training pipelines
 * 3. CI/CD: Integrate into build process for automated model updates
 *
 * EXAMPLE:
 * ========
 * // With Spring (recommended):
 * @Autowired
 * ModelTrainer trainer;
 *
 * // Programmatic usage:
 * trainer.trainAndSave(
 *     "./data/datasets/Reviews.csv",
 *     "./models/svm-model.ser",
 *     AlgorithmType.SVM,
 *     10000
 * );
 */
@Component
public class ModelTrainer {

    private static final Logger logger = LoggerFactory.getLogger(ModelTrainer.class);

    private final SimpleDatasetLoader datasetLoader;
    private final TextPreprocessor textPreprocessor;
    private final WekaInstancesConverter wekaInstancesConverter;

    /**
     * Creates a ModelTrainer with Spring-injected components.
     */
    @Autowired
    public ModelTrainer(SimpleDatasetLoader datasetLoader,
                        TextPreprocessor textPreprocessor,
                        WekaInstancesConverter wekaInstancesConverter) {
        this.datasetLoader = datasetLoader;
        this.textPreprocessor = textPreprocessor;
        this.wekaInstancesConverter = wekaInstancesConverter;

        logger.info("ModelTrainer initialized with Spring-managed components");
    }

    /**
     * Trains a model and saves it to disk.
     *
     * This is the main entry point for offline model training.
     *
     * @param dataPath Path to training data (CSV, JSON, etc.)
     * @param outputPath Path where trained model should be saved
     * @param algorithmType Type of classifier to train
     * @param maxSamples Maximum number of training samples (0 = use all)
     * @return Training statistics
     * @throws Exception if training or saving fails
     */
    public TrainingResult trainAndSave(String dataPath, String outputPath,
                                        AlgorithmType algorithmType, int maxSamples)
            throws Exception {

        // CRITICAL: Reset preprocessing components before each training
        // This is necessary because they're Spring singletons that retain state
        logger.info("Resetting preprocessing components before training {}", algorithmType);
        resetPreprocessingComponents();

        logger.info("=== Starting Model Training ===");
        logger.info("Algorithm: {}", algorithmType.getDisplayName());
        logger.info("Data path: {}", dataPath);
        logger.info("Output path: {}", outputPath);
        logger.info("Max samples: {}", maxSamples > 0 ? maxSamples : "all");

        long startTime = System.currentTimeMillis();

        // Step 1: Load all data
        logger.info("Step 1/5: Loading data...");
        List<Dataset> allData = loadTrainingData(dataPath, maxSamples);
        long loadTime = System.currentTimeMillis() - startTime;
        logger.info("✅ Loaded {} samples in {}ms", allData.size(), loadTime);

        // Step 2: Perform stratified train/validation/test split (60/20/20)
        logger.info("Step 2/5: Performing stratified 60/20/20 train/val/test split...");
        StratifiedDataSplitter.DataSplit split = StratifiedDataSplitter.stratifiedSplit(
                allData,
                0.6,  // 60% train
                0.2,  // 20% validation
                0.2,  // 20% test
                42    // Fixed seed for reproducibility
        );
        logger.info("✅ Split complete: train={}, val={}, test={}",
                split.train.size(), split.validation.size(), split.test.size());

        // Step 3: Create and train classifier (ONLY on train set)
        logger.info("Step 3/5: Training {} classifier on TRAIN SET ONLY...", algorithmType.getDisplayName());
        long trainStartTime = System.currentTimeMillis();
        SentimentClassifier classifier = createClassifier(algorithmType);
        classifier.train(split.train);  // ✅ Train ONLY on training set
        long trainTime = System.currentTimeMillis() - trainStartTime;
        logger.info("✅ Training completed in {}ms ({:.2f}s)",
                trainTime, trainTime / 1000.0);

        // Step 4: Validate trained model (on train set for sanity check)
        logger.info("Step 4/5: Validating trained model...");
        validateModel(classifier, split.train);
        logger.info("✅ Model validation passed");

        // Step 5: Save model to disk
        logger.info("Step 5/5: Saving model to {}...", outputPath);
        long saveStartTime = System.currentTimeMillis();
        saveModel(classifier, outputPath, algorithmType);
        long saveTime = System.currentTimeMillis() - saveStartTime;
        logger.info("✅ Model saved in {}ms", saveTime);

        long totalTime = System.currentTimeMillis() - startTime;

        TrainingResult result = new TrainingResult(
                algorithmType,
                split.train.size(),
                split.validation.size(),
                split.test.size(),
                outputPath,
                trainTime,
                totalTime,
                classifier.isTrained(),
                split
        );

        logger.info("=== Training Complete ===");
        logger.info("{}", result);

        return result;
    }

    /**
     * Trains multiple models in parallel and saves them.
     *
     * @param dataPath Path to training data
     * @param outputDir Directory where models should be saved
     * @param algorithms List of algorithm types to train
     * @param maxSamples Maximum training samples per model
     * @return List of training results
     */
    public List<TrainingResult> trainMultipleModels(String dataPath, String outputDir,
                                                     List<AlgorithmType> algorithms,
                                                     int maxSamples) throws Exception {

        logger.info("Training {} models from {}", algorithms.size(), dataPath);

        // Create output directory if needed
        Files.createDirectories(Paths.get(outputDir));

        List<TrainingResult> results = new ArrayList<>();

        for (AlgorithmType algorithm : algorithms) {
            String outputPath = Paths.get(outputDir,
                    algorithm.name().toLowerCase() + "-model.ser").toString();

            try {
                TrainingResult result = trainAndSave(dataPath, outputPath, algorithm, maxSamples);
                results.add(result);
            } catch (Exception e) {
                logger.error("Failed to train {} model: {}", algorithm, e.getMessage(), e);
                results.add(TrainingResult.failed(algorithm, e.getMessage()));
            }
        }

        return results;
    }

    // ==================== PRIVATE HELPERS ====================

    /**
     * Reset preprocessing components to allow training multiple models sequentially.
     * This is necessary because Spring beans are singletons and retain state.
     */
    private void resetPreprocessingComponents() {
        try {
            textPreprocessor.reset();
            wekaInstancesConverter.reset();
            logger.debug("Successfully reset preprocessing components");
        } catch (Exception e) {
            logger.warn("Failed to reset preprocessing components: {}", e.getMessage());
        }
    }

    private List<Dataset> loadTrainingData(String dataPath, int maxSamples) throws Exception {
        DatasetLoadResult loadResult = datasetLoader.loadWithMetadata(dataPath);
        List<Dataset> allData = loadResult.datasets();

        logger.info("Loaded {} total samples from {} ({}ms)",
                allData.size(), loadResult.datasetType(), loadResult.loadTimeMs());

        // Log class distribution
        long positive = allData.stream().filter(d -> d.getSentiment() == Dataset.SentimentLabel.POSITIVE).count();
        long negative = allData.stream().filter(d -> d.getSentiment() == Dataset.SentimentLabel.NEGATIVE).count();
        long neutral = allData.stream().filter(d -> d.getSentiment() == Dataset.SentimentLabel.NEUTRAL).count();
        logger.info("Distribution: positive={}, negative={}, neutral={}", positive, negative, neutral);

        // Shuffle for better training
        List<Dataset> shuffled = new ArrayList<>(allData);
        Collections.shuffle(shuffled);

        // Limit samples if requested
        if (maxSamples > 0 && shuffled.size() > maxSamples) {
            logger.info("Limiting to {} samples (from {})", maxSamples, shuffled.size());
            return shuffled.subList(0, maxSamples);
        }

        return shuffled;
    }

    private SentimentClassifier createClassifier(AlgorithmType algorithmType) {
        return switch (algorithmType) {
            case SVM -> new SVMClassifier(textPreprocessor, wekaInstancesConverter);
            case NAIVE_BAYES -> new NaiveBayesClassifier(textPreprocessor, wekaInstancesConverter);
            case RANDOM_FOREST -> new RandomForestClassifier(textPreprocessor, wekaInstancesConverter);
            case LOGISTIC_REGRESSION -> new LogisticRegressionClassifier(textPreprocessor, wekaInstancesConverter);
            case NEURAL_NETWORK -> throw new UnsupportedOperationException(
                "Neural Network classifier not yet implemented. Use SVM, Naive Bayes, Random Forest, or Logistic Regression.");
            case UNKNOWN -> throw new IllegalArgumentException(
                "Cannot create classifier for UNKNOWN algorithm type");
        };
    }

    private void validateModel(SentimentClassifier classifier, List<Dataset> trainingData) throws Exception {
        if (!classifier.isTrained()) {
            throw new IllegalStateException("Classifier reports as not trained after training!");
        }

        // Test with a few samples
        int samplesToTest = Math.min(5, trainingData.size());
        logger.debug("Testing model with {} samples...", samplesToTest);

        for (int i = 0; i < samplesToTest; i++) {
            Dataset sample = trainingData.get(i);
            String prediction = classifier.classify(sample.getText());
            double[] probs = classifier.getClassificationProbabilities(sample.getText());

            logger.debug("Sample {}: actual={}, predicted={}, confidence={:.3f}",
                    i + 1, sample.getSentiment(), prediction, getMaxProb(probs));
        }
    }

    private double getMaxProb(double[] probs) {
        double max = 0.0;
        for (double prob : probs) {
            if (prob > max) max = prob;
        }
        return max;
    }

    @SuppressWarnings("unchecked")
    private void saveModel(SentimentClassifier classifier, String outputPath,
                           AlgorithmType algorithmType) throws IOException {

        Path modelPath = Paths.get(outputPath);
        Files.createDirectories(modelPath.getParent());

        // All classifiers extend ClassifierTrainingTemplate and use generic Weka persistence
        if (classifier instanceof ClassifierTrainingTemplate) {
            WekaModelPersistence<ClassifierTrainingTemplate<?>> persistence = new WekaModelPersistence<>();
            persistence.saveModel((ClassifierTrainingTemplate<?>) classifier, modelPath);
        } else {
            throw new UnsupportedOperationException(
                    "Classifier type " + algorithmType + " does not support persistence");
        }

        long fileSize = Files.size(modelPath);
        logger.info("Model file size: {} bytes ({} KB)", fileSize, fileSize / 1024);
    }

    // ==================== RESULT CLASSES ====================

    /**
     * Training result metadata.
     */
    public static class TrainingResult {
        private final AlgorithmType algorithm;
        private final int trainSampleCount;
        private final int valSampleCount;
        private final int testSampleCount;
        private final String outputPath;
        private final long trainingTimeMs;
        private final long totalTimeMs;
        private final boolean success;
        private final String errorMessage;
        private final StratifiedDataSplitter.DataSplit dataSplit;  // Store split for later evaluation

        public TrainingResult(AlgorithmType algorithm, int trainSampleCount, int valSampleCount,
                              int testSampleCount, String outputPath,
                              long trainingTimeMs, long totalTimeMs, boolean success,
                              StratifiedDataSplitter.DataSplit dataSplit) {
            this(algorithm, trainSampleCount, valSampleCount, testSampleCount, outputPath,
                    trainingTimeMs, totalTimeMs, success, null, dataSplit);
        }

        private TrainingResult(AlgorithmType algorithm, int trainSampleCount, int valSampleCount,
                               int testSampleCount, String outputPath,
                               long trainingTimeMs, long totalTimeMs, boolean success,
                               String errorMessage, StratifiedDataSplitter.DataSplit dataSplit) {
            this.algorithm = algorithm;
            this.trainSampleCount = trainSampleCount;
            this.valSampleCount = valSampleCount;
            this.testSampleCount = testSampleCount;
            this.outputPath = outputPath;
            this.trainingTimeMs = trainingTimeMs;
            this.totalTimeMs = totalTimeMs;
            this.success = success;
            this.errorMessage = errorMessage;
            this.dataSplit = dataSplit;
        }

        public static TrainingResult failed(AlgorithmType algorithm, String errorMessage) {
            return new TrainingResult(algorithm, 0, 0, 0, null, 0, 0, false, errorMessage, null);
        }

        public AlgorithmType getAlgorithm() { return algorithm; }
        public int getTrainSampleCount() { return trainSampleCount; }
        public int getValSampleCount() { return valSampleCount; }
        public int getTestSampleCount() { return testSampleCount; }
        public int getTotalSampleCount() { return trainSampleCount + valSampleCount + testSampleCount; }
        public String getOutputPath() { return outputPath; }
        public long getTrainingTimeMs() { return trainingTimeMs; }
        public long getTotalTimeMs() { return totalTimeMs; }
        public boolean isSuccess() { return success; }
        public String getErrorMessage() { return errorMessage; }
        public StratifiedDataSplitter.DataSplit getDataSplit() { return dataSplit; }

        @Override
        public String toString() {
            if (!success) {
                return String.format("TrainingResult{algorithm=%s, success=false, error='%s'}",
                        algorithm.getDisplayName(), errorMessage);
            }

            return String.format(
                    "TrainingResult{algorithm=%s, train=%d, val=%d, test=%d (total=%d), " +
                    "trainTime=%dms (%.2fs), totalTime=%dms (%.2fs), output='%s'}",
                    algorithm.getDisplayName(), trainSampleCount, valSampleCount, testSampleCount,
                    getTotalSampleCount(),
                    trainingTimeMs, trainingTimeMs / 1000.0,
                    totalTimeMs, totalTimeMs / 1000.0,
                    outputPath);
        }
    }
}
