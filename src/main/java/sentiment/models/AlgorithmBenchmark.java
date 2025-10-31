package sentiment.models;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sentiment.data.Dataset;
import sentiment.evaluation.ClassifierEvaluationResult;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import weka.core.Instances;

import java.util.*;
import java.util.concurrent.*;

/**
 * Framework for systematic comparison of multiple sentiment analysis models.
 *
 * PURPOSE:
 * ========
 * This class enables comprehensive model evaluation to answer critical questions:
 * - "Which algorithm performs best on my data?"
 * - "What are the accuracy/speed trade-offs?"
 * - "Which model should I deploy to production?"
 *
 * DESIGN:
 * =======
 * - Trains multiple models on the same dataset with identical preprocessing
 * - Evaluates all models on the same test set for fair comparison
 * - Captures accuracy, timing, and resource metrics
 * - Generates comparison reports and recommendations
 *
 * SUPPORTED MODELS:
 * =================
 * - SVM (Support Vector Machine): High accuracy, slower training
 * - Naive Bayes: Fast baseline, good for large datasets
 * - Random Forest: Ensemble method, robust to noise
 * - Logistic Regression: Interpretable linear model
 *
 * USAGE EXAMPLE:
 * ==============
 * <pre>
 * AlgorithmBenchmark comparator = new AlgorithmBenchmark();
 * comparator.addModel(AlgorithmType.SVM);
 * comparator.addModel(AlgorithmType.NAIVE_BAYES);
 * comparator.addModel(AlgorithmType.RANDOM_FOREST);
 *
 * ComparisonResult result = comparator.compareModels(trainingData, testData);
 * System.out.println(result.toComparisonTable());
 * </pre>
 *
 * THREAD SAFETY:
 * ==============
 * This class is not thread-safe. Use one instance per comparison task.
 */
public class AlgorithmBenchmark {

    private static final Logger logger = LoggerFactory.getLogger(AlgorithmBenchmark.class);

    private final List<AlgorithmType> modelsToCompare;
    private final Map<AlgorithmType, ClassifierTrainingTemplate<?>> trainedModels;

    // Configuration
    private int maxFeatures = 5000;
    private boolean useParallelTraining = false;

    public AlgorithmBenchmark() {
        this.modelsToCompare = new ArrayList<>();
        this.trainedModels = new ConcurrentHashMap<>();
    }

    // ==================== CONFIGURATION ====================

    /**
     * Adds a model to the comparison.
     */
    public AlgorithmBenchmark addModel(AlgorithmType algorithmType) {
        if (algorithmType == AlgorithmType.UNKNOWN) {
            throw new IllegalArgumentException("Cannot add UNKNOWN algorithm type");
        }
        if (!modelsToCompare.contains(algorithmType)) {
            modelsToCompare.add(algorithmType);
            logger.info("Added {} to comparison", algorithmType.getDisplayName());
        }
        return this;
    }

    /**
     * Adds all standard models for comprehensive comparison.
     */
    public AlgorithmBenchmark addAllStandardModels() {
        addModel(AlgorithmType.SVM);
        addModel(AlgorithmType.NAIVE_BAYES);
        addModel(AlgorithmType.RANDOM_FOREST);
        addModel(AlgorithmType.LOGISTIC_REGRESSION);
        logger.info("Added all standard models for comparison");
        return this;
    }

    /**
     * Sets the maximum number of TF-IDF features to extract.
     */
    public AlgorithmBenchmark setMaxFeatures(int maxFeatures) {
        if (maxFeatures < 100 || maxFeatures > 50000) {
            throw new IllegalArgumentException("Max features must be between 100 and 50000");
        }
        this.maxFeatures = maxFeatures;
        return this;
    }

    /**
     * Enables or disables parallel model training.
     * When enabled, models train concurrently (faster but uses more CPU).
     */
    public AlgorithmBenchmark setParallelTraining(boolean useParallel) {
        this.useParallelTraining = useParallel;
        return this;
    }

    // ==================== COMPARISON EXECUTION ====================

    /**
     * Compares all configured models on the given dataset.
     *
     * @param trainingData Raw training datasets
     * @param testData Raw test datasets (will be preprocessed identically)
     * @return Comprehensive comparison results
     * @throws Exception if comparison fails
     */
    public ComparisonResult compareModels(List<Dataset> trainingData, List<Dataset> testData) throws Exception {
        if (modelsToCompare.isEmpty()) {
            throw new IllegalStateException("No models configured for comparison. Use addModel() first.");
        }

        if (trainingData == null || trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }

        if (testData == null || testData.isEmpty()) {
            throw new IllegalArgumentException("Test data cannot be null or empty");
        }

        logger.info("Starting model comparison with {} models", modelsToCompare.size());
        logger.info("Training data: {} datasets", trainingData.size());
        logger.info("Test data: {} datasets", testData.size());

        List<ClassificationMetrics> allMetrics = new ArrayList<>();

        if (useParallelTraining) {
            allMetrics = compareModelsParallel(trainingData, testData);
        } else {
            allMetrics = compareModelsSequential(trainingData, testData);
        }

        int totalTestInstances = testData.size();

        ComparisonResult result = new ComparisonResult(allMetrics, "Comparison Dataset", totalTestInstances);

        logger.info("Model comparison complete");
        return result;
    }

    /**
     * Sequential comparison (one model at a time).
     */
    private List<ClassificationMetrics> compareModelsSequential(List<Dataset> trainingData, List<Dataset> testData) throws Exception {
        List<ClassificationMetrics> allMetrics = new ArrayList<>();

        for (AlgorithmType algorithmType : modelsToCompare) {
            logger.info("Training and evaluating: {}", algorithmType.getDisplayName());

            try {
                ClassificationMetrics metrics = trainAndEvaluateModel(algorithmType, trainingData, testData);
                allMetrics.add(metrics);
                logger.info("Completed: {} - Accuracy: {:.2f}%",
                        algorithmType.getDisplayName(), metrics.getAccuracy() * 100);
            } catch (Exception e) {
                logger.error("Failed to train/evaluate {}: {}",
                        algorithmType.getDisplayName(), e.getMessage());
                throw e;
            }
        }

        return allMetrics;
    }

    /**
     * Parallel comparison (all models trained concurrently).
     */
    private List<ClassificationMetrics> compareModelsParallel(List<Dataset> trainingData, List<Dataset> testData) throws Exception {
        logger.info("Using parallel training for {} models", modelsToCompare.size());

        ExecutorService executor = Executors.newFixedThreadPool(
                Math.min(modelsToCompare.size(), Runtime.getRuntime().availableProcessors())
        );

        try {
            List<Future<ClassificationMetrics>> futures = new ArrayList<>();

            for (AlgorithmType algorithmType : modelsToCompare) {
                Future<ClassificationMetrics> future = executor.submit(() ->
                        trainAndEvaluateModel(algorithmType, trainingData, testData)
                );
                futures.add(future);
            }

            List<ClassificationMetrics> allMetrics = new ArrayList<>();
            for (Future<ClassificationMetrics> future : futures) {
                allMetrics.add(future.get());
            }

            return allMetrics;

        } finally {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(5, TimeUnit.MINUTES)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                executor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    /**
     * Trains and evaluates a single model, returning comprehensive metrics.
     */
    private ClassificationMetrics trainAndEvaluateModel(AlgorithmType algorithmType,
                                                List<Dataset> trainingData,
                                                List<Dataset> testData) throws Exception {

        // Step 1: Create preprocessing components
        sentiment.preprocessing.ContractionExpander contractionExpander = new sentiment.preprocessing.ContractionExpander();
        sentiment.preprocessing.AdvancedTokenizer tokenizer = new sentiment.preprocessing.AdvancedTokenizer(
                false,  // preserveNumbers
                1,      // minTokenLength
                true    // lowerCase
        );
        sentiment.preprocessing.IntelligentStopwordRemover stopwordRemover =
                new sentiment.preprocessing.IntelligentStopwordRemover("SENTIMENT_AWARE");

        // Create temporary preprocessor without feature extractor
        TextPreprocessor tempPreprocessor = new TextPreprocessor(
                null,  // featureExtractor - will set later
                contractionExpander,
                tokenizer,
                stopwordRemover,
                2,  // minWordLength
                true // removeStopwords
        );

        // Create feature extractor with the temp preprocessor
        sentiment.preprocessing.TFIDFFeatureExtractor featureExtractor = new sentiment.preprocessing.TFIDFFeatureExtractor(
                tempPreprocessor,
                maxFeatures,  // maxFeatures
                1,            // minTermFreq
                1.0,          // idfThreshold
                true,         // useTFIDF
                true,         // normalize
                true,         // useBigrams
                false,        // useTrigrams
                false         // useStemming
        );

        // Create final preprocessor with all components
        TextPreprocessor preprocessor = new TextPreprocessor(
                featureExtractor,
                contractionExpander,
                tokenizer,
                stopwordRemover,
                2,  // minWordLength
                true // removeStopwords
        );

        // Create converter with the final preprocessor
        WekaInstancesConverter converter = new WekaInstancesConverter(
                preprocessor,
                maxFeatures,  // maxFeatures
                1,            // minTermFreq
                1.0,          // idfThreshold
                true,         // useTFIDF
                true,         // normalize
                true,         // useBigrams
                false         // useTrigrams
        );

        ClassifierTrainingTemplate<?> model = createModel(algorithmType, preprocessor, converter);

        // Step 2: Train model and measure time
        long trainingStart = System.currentTimeMillis();
        model.train(trainingData);
        long trainingTime = System.currentTimeMillis() - trainingStart;

        // Step 3: Transform test data using the already-fitted pipeline
        Instances testInstances = converter.transformDatasets(testData);

        // Step 4: Evaluate and measure time
        ClassifierEvaluationResult evalResult = ((ClassifierEvaluator) model).evaluate(testInstances);
        long evaluationTime = (long) evalResult.getAdditionalStats().getOrDefault("evaluationTimeMs", 0L);

        // Step 5: Measure average inference time
        long inferenceTime = measureInferenceTime(model, testData);

        // Step 6: Build metrics
        ClassificationMetrics metrics = buildMetrics(
                algorithmType,
                evalResult,
                trainingTime,
                inferenceTime,
                evaluationTime,
                trainingData.size(),
                converter.getVocabulary().size()
        );

        // Store trained model
        trainedModels.put(algorithmType, model);

        return metrics;
    }

    /**
     * Creates a classifier instance based on algorithm type.
     */
    private ClassifierTrainingTemplate<?> createModel(AlgorithmType algorithmType,
                                                       TextPreprocessor preprocessor,
                                                       WekaInstancesConverter converter) {
        switch (algorithmType) {
            case SVM:
                return new SVMClassifier(preprocessor, converter);
            case NAIVE_BAYES:
                return new NaiveBayesClassifier(preprocessor, converter);
            case RANDOM_FOREST:
                return new RandomForestClassifier(preprocessor, converter);
            case LOGISTIC_REGRESSION:
                return new LogisticRegressionClassifier(preprocessor, converter);
            default:
                throw new IllegalArgumentException("Unsupported algorithm type: " + algorithmType);
        }
    }

    /**
     * Measures average inference time per instance.
     */
    private long measureInferenceTime(ClassifierTrainingTemplate<?> model, List<Dataset> testData) {
        // Sample up to 100 texts for timing
        List<Dataset> sampleData = testData.stream()
                .limit(100)
                .collect(java.util.stream.Collectors.toList());

        if (sampleData.isEmpty()) {
            return 0;
        }

        long start = System.currentTimeMillis();
        for (Dataset dataset : sampleData) {
            try {
                model.classify(dataset.getText());
            } catch (Exception e) {
                logger.warn("Inference timing failed: {}", e.getMessage());
            }
        }
        long total = System.currentTimeMillis() - start;

        return total / sampleData.size(); // Average per instance
    }

    /**
     * Builds ClassificationMetrics from evaluation results.
     */
    private ClassificationMetrics buildMetrics(AlgorithmType algorithmType,
                                       ClassifierEvaluationResult evalResult,
                                       long trainingTime,
                                       long inferenceTime,
                                       long evaluationTime,
                                       int trainingInstances,
                                       int featureCount) {

        return ClassificationMetrics.builder()
                .algorithmType(algorithmType)
                .modelName(algorithmType.getDisplayName())
                .accuracy(evalResult.getAccuracy())
                .precision(evalResult.getMacroAvgPrecision())
                .recall(evalResult.getMacroAvgRecall())
                .f1Score(evalResult.getMacroAvgF1())
                .perClassPrecision(evalResult.getPrecision())
                .perClassRecall(evalResult.getRecall())
                .perClassF1(evalResult.getF1Score())
                .trainingTimeMs(trainingTime)
                .inferenceTimeMs(inferenceTime)
                .evaluationTimeMs(evaluationTime)
                .confusionMatrix(evalResult.getConfusionMatrix())
                .correctlyClassified((int) evalResult.getAdditionalStats().getOrDefault("correctlyClassified", 0))
                .incorrectlyClassified((int) evalResult.getAdditionalStats().getOrDefault("incorrectlyClassified", 0))
                .classLabels(evalResult.getClassLabels())
                .trainingInstances(trainingInstances)
                .featureCount(featureCount)
                .build();
    }

    // ==================== ACCESSORS ====================

    /**
     * Returns a trained model by algorithm type (available after compareModels() completes).
     */
    public Optional<ClassifierTrainingTemplate<?>> getTrainedModel(AlgorithmType algorithmType) {
        return Optional.ofNullable(trainedModels.get(algorithmType));
    }

    /**
     * Returns all configured models.
     */
    public List<AlgorithmType> getModelsToCompare() {
        return new ArrayList<>(modelsToCompare);
    }

    /**
     * Clears all configured models and trained instances.
     */
    public void clear() {
        modelsToCompare.clear();
        trainedModels.clear();
        logger.info("Cleared all models from comparator");
    }
}
