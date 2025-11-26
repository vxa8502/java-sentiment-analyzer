package sentiment.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import sentiment.data.Dataset;
import sentiment.data.SimpleDatasetLoader;
import sentiment.data.DatasetLoadResult;
import sentiment.evaluation.FeatureImportanceAnalyzer;
import sentiment.evaluation.FeatureImportancePersistence;
import sentiment.evaluation.StratifiedDataSplitter;
import sentiment.evaluation.domain.FeatureImportanceResult;
import sentiment.models.*;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import weka.core.Instances;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Service for training and persisting sentiment analysis models.
 * Orchestrates data loading, stratified splitting, training, validation, and serialization.
 */
@Component
public class ModelTrainer {

    private static final Logger logger = LoggerFactory.getLogger(ModelTrainer.class);

    private final SimpleDatasetLoader datasetLoader;
    private final TextPreprocessor textPreprocessor;
    private final WekaInstancesConverter wekaInstancesConverter;

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
     * Trains a model and saves it to disk with stratified 60/20/20 train/val/test split.
     *
     * @param dataPath training data path
     * @param outputPath where to save trained model
     * @param algorithmType classifier algorithm
     * @param maxSamples max training samples (0 = all)
     * @param showFeatureImportance analyze feature importance
     * @param topFeaturesCount number of top features to display
     * @return training statistics
     */
    public TrainingResult trainAndSave(String dataPath, String outputPath,
                                        AlgorithmType algorithmType, int maxSamples,
                                        boolean showFeatureImportance, int topFeaturesCount)
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
        logger.info("✅ Training completed in {}ms ({}s)",
                trainTime, trainTime / 1000.0);

        // Step 4: Validate trained model (on train set for sanity check)
        logger.info("Step 4/5: Validating trained model...");
        validateModel(classifier, split.train);
        logger.info("✅ Model validation passed");

        // Step 5: Analyze feature importance (if requested)
        if (showFeatureImportance) {
            logger.info("Step 5/6: Analyzing feature importance...");
            analyzeAndPrintFeatureImportance(classifier, split, topFeaturesCount, outputPath, algorithmType);
        }

        // Step 6: Save model to disk
        int finalStep = showFeatureImportance ? 6 : 5;
        logger.info("Step {}/{}: Saving model to {}...", finalStep, finalStep, outputPath);
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
                classifier.isTrained()
        );

        logger.info("=== Training Complete ===");
        logger.info("{}", result);

        return result;
    }

    /**
     * Trains multiple models sequentially and saves them.
     */
    public List<TrainingResult> trainMultipleModels(String dataPath, String outputDir,
                                                     List<AlgorithmType> algorithms,
                                                     int maxSamples,
                                                     boolean showFeatureImportance,
                                                     int topFeaturesCount) throws Exception {

        logger.info("Training {} models from {}", algorithms.size(), dataPath);

        // Create output directory if needed
        Files.createDirectories(Paths.get(outputDir));

        List<TrainingResult> results = new ArrayList<>();

        for (AlgorithmType algorithm : algorithms) {
            String outputPath = Paths.get(outputDir,
                    algorithm.name().toLowerCase() + "-model.ser").toString();

            try {
                TrainingResult result = trainAndSave(dataPath, outputPath, algorithm, maxSamples,
                        showFeatureImportance, topFeaturesCount);
                results.add(result);
            } catch (Exception e) {
                logger.error("Failed to train {} model: {}", algorithm, e.getMessage(), e);
                results.add(TrainingResult.failed(algorithm, e.getMessage()));
            }
        }

        return results;
    }

    // ==================== PRIVATE HELPERS ====================

    private void resetPreprocessingComponents() {
        try {
            wekaInstancesConverter.reset();  // This now resets the preprocessor too
            textPreprocessor.reset();        // Defensive: ensure preprocessor is also reset
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

        // Shuffle and limit samples if requested
        if (maxSamples > 0 && allData.size() > maxSamples) {
            // Efficiently sample without shuffling entire dataset
            Random random = new Random();
            Set<Integer> selectedIndices = new HashSet<>();
            while (selectedIndices.size() < maxSamples) {
                selectedIndices.add(random.nextInt(allData.size()));
            }
            List<Dataset> sampled = selectedIndices.stream()
                    .map(allData::get)
                    .collect(Collectors.toList());
            logger.info("Limiting to {} samples (from {})", maxSamples, allData.size());
            return sampled;
        }

        // Shuffle all data for better training
        List<Dataset> shuffled = new ArrayList<>(allData);
        Collections.shuffle(shuffled);
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
            double[] probabilities = classifier.getClassificationProbabilities(sample.getText());

            logger.debug("Sample {}: actual={}, predicted={}, confidence={}",
                    i + 1, sample.getSentiment(), prediction, getMaxProb(probabilities));
        }
    }

    private double getMaxProb(double[] probabilities) {
        double max = 0.0;
        for (double prob : probabilities) {
            if (prob > max) max = prob;
        }
        return max;
    }

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

    private void analyzeAndPrintFeatureImportance(SentimentClassifier classifier,
                                                   StratifiedDataSplitter.DataSplit split,
                                                   int topFeaturesCount,
                                                   String modelPath,
                                                   AlgorithmType algorithmType) {
        try {
            logger.info("Converting training data to Weka Instances for feature analysis...");
            // Convert the training data to Weka Instances using the already-trained converter
            Instances trainedInstances = wekaInstancesConverter.transformDatasets(split.train);

            logger.info("Analyzing feature importance on {} training instances with {} features...",
                    trainedInstances.numInstances(), trainedInstances.numAttributes() - 1);

            // Analyze feature importance (with larger topK for saving to file)
            int fullTopK = Math.max(topFeaturesCount, 100); // Save top 100 minimum

            // Get the underlying Weka classifier
            if (!(classifier instanceof WekaClassifier wekaClassifierInterface)) {
                logger.error("Could not extract Weka classifier for feature importance analysis");
                return;
            }

            weka.classifiers.Classifier wekaClassifier = wekaClassifierInterface.getWekaClassifier();

            // Use single unified analyzer for all classifier types
            logger.info("Analyzing feature importance using perturbation method");
            FeatureImportanceAnalyzer analyzer = new FeatureImportanceAnalyzer();
            FeatureImportanceResult result = analyzer.analyzeFeatureImportance(
                    trainedInstances, wekaClassifier, fullTopK);

            // Print the results to console
            printFeatureImportanceReport(result, topFeaturesCount, algorithmType);

            // Save to JSON file for API serving
            try {
                Path featureImportancePath = FeatureImportancePersistence.getFeatureImportancePath(
                        Paths.get(modelPath));
                FeatureImportancePersistence.save(result, featureImportancePath);
                logger.info("✅ Feature importance saved to: {}", featureImportancePath);
                System.out.println("\n💾 Feature importance saved to: " + featureImportancePath);
                System.out.println("   Use this for runtime API exploration via /api/v1/model/feature-importance\n");
            } catch (IOException e) {
                logger.warn("Failed to save feature importance to file: {}", e.getMessage());
            }

        } catch (Exception e) {
            logger.error("Failed to analyze feature importance: {}", e.getMessage(), e);
            System.err.println("⚠️  Feature importance analysis failed: " + e.getMessage());
        }
    }

    private void printFeatureImportanceReport(FeatureImportanceResult result,
                                               int topFeaturesCount,
                                               AlgorithmType algorithmType) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("FEATURE IMPORTANCE ANALYSIS - " + algorithmType.getDisplayName());
        System.out.println("=".repeat(80));
        System.out.println("Analysis completed in " + result.analysisTimeMs() + "ms");
        System.out.println("\nStatistics:");
        System.out.println("  Total features: " + result.allFeatures().size());
        System.out.println("  Mean absolute weight: " + String.format("%.6f", result.statistics().mean()));
        System.out.println("  Std deviation: " + String.format("%.6f", result.statistics().stdDev()));
        System.out.println("  Median: " + String.format("%.6f", result.statistics().median()));
        System.out.println("  95th percentile: " + String.format("%.6f", result.statistics().percentile95()));
        System.out.println("=".repeat(80));

        result.printTopFeatures(topFeaturesCount);

        System.out.println("INTERPRETATION:");
        System.out.println("  • Features with high |weight| strongly influence predictions");
        System.out.println("  • Positive weights → positive sentiment, negative → negative sentiment");
        System.out.println("  • Features near zero are non-discriminative");
        System.out.println("  • This analysis helps understand what the model learned");
        System.out.println("=".repeat(80) + "\n");
    }

    /**
     * Training result with metrics.
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

        public TrainingResult(AlgorithmType algorithm, int trainSampleCount, int valSampleCount,
                              int testSampleCount, String outputPath,
                              long trainingTimeMs, long totalTimeMs, boolean success) {
            this(algorithm, trainSampleCount, valSampleCount, testSampleCount, outputPath,
                    trainingTimeMs, totalTimeMs, success, null);
        }

        private TrainingResult(AlgorithmType algorithm, int trainSampleCount, int valSampleCount,
                               int testSampleCount, String outputPath,
                               long trainingTimeMs, long totalTimeMs, boolean success,
                               String errorMessage) {
            this.algorithm = algorithm;
            this.trainSampleCount = trainSampleCount;
            this.valSampleCount = valSampleCount;
            this.testSampleCount = testSampleCount;
            this.outputPath = outputPath;
            this.trainingTimeMs = trainingTimeMs;
            this.totalTimeMs = totalTimeMs;
            this.success = success;
            this.errorMessage = errorMessage;
        }

        public static TrainingResult failed(AlgorithmType algorithm, String errorMessage) {
            return new TrainingResult(algorithm, 0, 0, 0, null, 0, 0, false, errorMessage);
        }

        public AlgorithmType getAlgorithm() { return algorithm; }
        public int getTrainSampleCount() { return trainSampleCount; }
        public int getValSampleCount() { return valSampleCount; }
        public int getTestSampleCount() { return testSampleCount; }
        public int getTotalSampleCount() { return trainSampleCount + valSampleCount + testSampleCount; }
        public String getOutputPath() { return outputPath; }
        public long getTrainingTimeMs() { return trainingTimeMs; }
        public boolean isSuccess() { return success; }
        public String getErrorMessage() { return errorMessage; }

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
