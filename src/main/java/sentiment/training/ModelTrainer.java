package sentiment.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import sentiment.data.Dataset;
import sentiment.data.DatasetUtils;
import sentiment.data.SimpleDatasetLoader;
import sentiment.data.DatasetLoadResult;
import sentiment.data.SplitManifest;
import sentiment.evaluation.ClassifierEvaluationResult;
import sentiment.evaluation.FeatureImportanceAnalyzer;
import sentiment.evaluation.FeatureImportancePersistence;
import sentiment.evaluation.StratifiedDataSplitter;
import sentiment.evaluation.domain.FeatureImportanceResult;
import sentiment.models.*;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import sentiment.config.FeatureExtractionProperties;
import weka.core.Instances;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Service for training and persisting sentiment analysis models.
 * Orchestrates data loading, stratified splitting, training, validation, and serialization.
 */
@Component
public class ModelTrainer {

    private static final Logger logger = LoggerFactory.getLogger(ModelTrainer.class);

    private final SimpleDatasetLoader datasetLoader;
    private final ObjectProvider<TextPreprocessor> textPreprocessorProvider;
    private final ObjectProvider<WekaInstancesConverter> wekaInstancesConverterProvider;
    private final FeatureExtractionProperties featureConfig;

    @Autowired
    public ModelTrainer(SimpleDatasetLoader datasetLoader,
                        ObjectProvider<TextPreprocessor> textPreprocessorProvider,
                        ObjectProvider<WekaInstancesConverter> wekaInstancesConverterProvider,
                        FeatureExtractionProperties featureConfig) {
        this.datasetLoader = datasetLoader;
        this.textPreprocessorProvider = textPreprocessorProvider;
        this.wekaInstancesConverterProvider = wekaInstancesConverterProvider;
        this.featureConfig = featureConfig;

        logger.info("ModelTrainer initialized with Spring-managed components (prototype scope)");
    }

    /**
     * Trains a model and saves it to disk with stratified split.
     * If testDataPath is provided: 80/20 train/val split + separate test set (for pre-split datasets like Amazon)
     * If testDataPath is null: 60/20/20 train/val/test split from single file
     *
     * @param dataPath training data path (or full dataset path if testDataPath is null)
     * @param testDataPath optional separate test data path (null for single-file datasets)
     * @param outputPath where to save trained model
     * @param algorithmType classifier algorithm
     * @param maxSamples max training samples (0 = all)
     * @param showFeatureImportance analyze feature importance
     * @param topFeaturesCount number of top features to display
     * @param enableHyperparameterTuning enable grid search for SVM (increases training time 5-10x)
     * @return training statistics
     * @throws IllegalArgumentException if inputs are invalid
     * @throws Exception if training fails (data loading, model training, or persistence errors)
     */
    public TrainingResult trainAndSave(String dataPath, String testDataPath, String outputPath,
                                        AlgorithmType algorithmType, int maxSamples,
                                        boolean showFeatureImportance, int topFeaturesCount,
                                        boolean enableHyperparameterTuning)
            throws Exception {

        // Input validation
        if (dataPath == null || dataPath.trim().isEmpty()) {
            throw new IllegalArgumentException("Data path cannot be null or empty");
        }
        if (outputPath == null || outputPath.trim().isEmpty()) {
            throw new IllegalArgumentException("Output path cannot be null or empty");
        }
        if (algorithmType == null) {
            throw new IllegalArgumentException("Algorithm type cannot be null");
        }
        if (maxSamples < 0) {
            throw new IllegalArgumentException("Max samples cannot be negative");
        }
        if (topFeaturesCount <= 0) {
            throw new IllegalArgumentException("Top features count must be positive");
        }

        // Set up MDC context for this training session to enable tracing through logs
        String sessionId = UUID.randomUUID().toString().substring(0, 8);
        MDC.put("algorithmType", algorithmType.name());
        MDC.put("sessionId", sessionId);

        try {
            logger.info("=== Starting Model Training ===");
            logger.info("Algorithm: {}", algorithmType.getDisplayName());
            logger.info("Data path: {}", dataPath);
            logger.info("Output path: {}", outputPath);
            logger.info("Max samples: {}", maxSamples > 0 ? maxSamples : "all");

            long startTime = System.currentTimeMillis();

            // Step 1: Load data - prefer prepared splits if available
            logger.info("Step 1/5: Loading training data...");
            if (testDataPath != null && !testDataPath.trim().isEmpty()) {
                logger.info("Note: Ignoring separate test file - using prepared splits for consistency");
            }

            // Check for prepared splits (Phase 1 output from prepare_data.sh)
            String datasetName = inferDatasetName(dataPath);
            Path processedDir = Paths.get("data/processed", datasetName).toAbsolutePath();
            SplitManifest manifest = null;
            StratifiedDataSplitter.DataSplit split;
            String splitManifestHash = null;
            String testSplitHash = null;

            if (SplitManifest.exists(processedDir)) {
                // Use prepared splits (recommended path)
                logger.info("Found prepared splits for '{}' - loading from processed data", datasetName);
                manifest = SplitManifest.load(SplitManifest.getManifestPath(processedDir));

                // Verify checksums
                if (!manifest.verify(processedDir)) {
                    logger.warn("Split checksums do not match manifest! Data may have been corrupted.");
                    logger.warn("Consider running: ./scripts/prepare_data.sh --force");
                }

                // Load from prepared splits
                Path trainPath = processedDir.resolve(manifest.getTrain().getFile());
                Path testPath = processedDir.resolve(manifest.getTest().getFile());

                List<Dataset> trainData = datasetLoader.load(trainPath.toString());
                List<Dataset> testData = datasetLoader.load(testPath.toString());

                split = new StratifiedDataSplitter.DataSplit(trainData, List.of(), testData);

                // Store hashes for audit trail
                splitManifestHash = SplitManifest.calculateSha256(SplitManifest.getManifestPath(processedDir));
                testSplitHash = manifest.getTest().getSha256();

                logger.info(" Using prepared splits from {}", manifest.getCreatedAt());
                logger.info(" Train: {} samples, Test: {} samples", trainData.size(), testData.size());

            } else {
                // Fall back to creating splits (backward compatibility)
                logger.warn("No prepared splits found for '{}'. Run ./scripts/prepare_data.sh first!", datasetName);
                logger.warn("Falling back to dynamic split creation (not recommended for reproducibility)");

                List<Dataset> allData = loadTrainingData(dataPath, maxSamples);
                long loadTime = System.currentTimeMillis() - startTime;
                logger.info(" Loaded {} samples in {}ms", allData.size(), loadTime);

                // Step 2: Perform stratified train/test split (80/20)
                logger.info("Step 2/5: Performing stratified 80/20 train/test split...");
                split = StratifiedDataSplitter.stratifiedSplit(
                        allData,
                        0.8,  // 80% train
                        0.0,  // 0% validation (SVM uses internal CV for tuning)
                        0.2,  // 20% test
                        42    // Fixed seed for reproducibility
                );

                // Save train/test splits for reproducibility and auditability
                saveDataSplits(split.train, split.test, dataPath);
            }

            logger.info(" Split complete: train={}, test={}",
                    split.train.size(), split.test.size());

            // Step 3: Create and train classifier (ONLY on train set)
            logger.info("Step 3/5: Training {} classifier on TRAIN SET ONLY...", algorithmType.getDisplayName());
            long trainStartTime = System.currentTimeMillis();
            SentimentClassifier classifier = createClassifier(algorithmType);

            // Configure SVM-specific settings before training
            if (classifier instanceof SVMClassifier svm) {
                if (enableHyperparameterTuning) {
                    logger.info("Hyperparameter tuning ENABLED for SVM (training will take longer)");
                    svm.setHyperparameterTuning(true, 5);
                }
            }

            classifier.train(split.train);  //  Train ONLY on training set
            long trainTime = System.currentTimeMillis() - trainStartTime;
            logger.info(" Training completed in {}ms ({}s)",
                    trainTime, trainTime / 1000.0);

            // Log optimal config if hyperparameter tuning was used
            if (classifier instanceof SVMClassifier svm && svm.getOptimalConfig() != null) {
                SVMConfig optimal = svm.getOptimalConfig();
                logger.info("Grid search results: kernel={}, C={}, CV accuracy={}",
                    optimal.getKernelType().getDisplayName(), optimal.getC(),
                    String.format("%.3f", optimal.getCvAccuracy()));
            }

            // Step 4: Validate trained model (on train set for sanity check)
            logger.info("Step 4/5: Validating trained model...");
            validateModel(classifier, split.train);
            logger.info(" Model validation passed");

            // Step 5: Evaluate on test set
            logger.info("Step 5/7: Evaluating model on test set...");
            ClassifierEvaluationResult testEvaluation = evaluateOnTestSet(classifier, split.test);
            logger.info(" Test Accuracy: {}", String.format("%.3f", testEvaluation.getAccuracy()));
            logger.info(" Test Precision (macro): {}", String.format("%.3f", testEvaluation.getMacroAvgPrecision()));
            logger.info(" Test Recall (macro): {}", String.format("%.3f", testEvaluation.getMacroAvgRecall()));
            logger.info(" Test F1 (macro): {}", String.format("%.3f", testEvaluation.getMacroAvgF1()));

            // Step 6: Analyze feature importance (if requested)
            if (showFeatureImportance) {
                logger.info("Step 6/7: Analyzing feature importance...");
                analyzeAndPrintFeatureImportance(classifier, split, topFeaturesCount, outputPath, algorithmType);
            }

            // Step 7: Save model and metadata to disk
            int finalStep = showFeatureImportance ? 7 : 6;
            logger.info("Step {}/{}: Saving model and metadata to {}...", finalStep, finalStep, outputPath);
            long saveStartTime = System.currentTimeMillis();
            saveModel(classifier, outputPath, algorithmType);
            saveTrainingMetadata(dataPath, outputPath, algorithmType, split, testEvaluation, trainTime, classifier,
                    splitManifestHash, testSplitHash);
            long saveTime = System.currentTimeMillis() - saveStartTime;
            logger.info(" Model and metadata saved in {}ms", saveTime);

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
        } finally {
            // Clean up MDC context after training completes
            MDC.remove("algorithmType");
            MDC.remove("sessionId");
        }
    }

    /**
     * Backward-compatible wrapper: trains model with 60/20/20 split from single file.
     */
    public TrainingResult trainAndSave(String dataPath, String outputPath,
                                        AlgorithmType algorithmType, int maxSamples,
                                        boolean showFeatureImportance, int topFeaturesCount,
                                        boolean enableHyperparameterTuning)
            throws Exception {
        return trainAndSave(dataPath, null, outputPath, algorithmType, maxSamples,
                showFeatureImportance, topFeaturesCount, enableHyperparameterTuning);
    }

    /**
     * Trains multiple models sequentially and saves them.
     */
    public List<TrainingResult> trainMultipleModels(String dataPath, String outputDir,
                                                     List<AlgorithmType> algorithms,
                                                     int maxSamples,
                                                     boolean showFeatureImportance,
                                                     int topFeaturesCount,
                                                     boolean enableHyperparameterTuning) throws Exception {

        logger.info("Training {} models from {}", algorithms.size(), dataPath);

        // Create output directory if needed
        Files.createDirectories(Paths.get(outputDir));

        List<TrainingResult> results = new ArrayList<>();

        for (AlgorithmType algorithm : algorithms) {
            String outputPath = Paths.get(outputDir,
                    algorithm.name().toLowerCase() + "-model.ser").toString();

            try {
                TrainingResult result = trainAndSave(dataPath, null, outputPath, algorithm, maxSamples,
                        showFeatureImportance, topFeaturesCount, enableHyperparameterTuning);
                results.add(result);
            } catch (Exception e) {
                logger.error("Failed to train {} model: {}", algorithm, e.getMessage(), e);
                results.add(TrainingResult.failed(algorithm, e.getMessage()));
            }
        }

        return results;
    }

    // ==================== PRIVATE HELPERS ====================

    private List<Dataset> loadTrainingData(String dataPath, int maxSamples) throws Exception {
        // Load with STRATIFIED SAMPLING: preserves original class distribution when capping
        // SimpleDatasetLoader does a two-pass approach:
        //   1. Scan file to get class distribution
        //   2. Load proportionally to preserve ratio (e.g., 75/25 Yelp stays 75/25)
        DatasetLoadResult loadResult = datasetLoader.loadWithMetadata(dataPath, maxSamples);
        List<Dataset> allData = loadResult.datasets();

        logger.info("Loaded {} total samples from {} ({}ms)",
                allData.size(), loadResult.datasetType(), loadResult.loadTimeMs());

        // Log class distribution (neutrals filtered at load time for binary classification)
        long positive = allData.stream().filter(d -> d.getSentiment() == Dataset.SentimentLabel.POSITIVE).count();
        long negative = allData.stream().filter(d -> d.getSentiment() == Dataset.SentimentLabel.NEGATIVE).count();
        logger.info("Distribution after stratified load: positive={}, negative={}", positive, negative);

        // Shuffle to mix classes before splitting
        List<Dataset> shuffled = new ArrayList<>(allData);
        Collections.shuffle(shuffled, new Random(42)); // Fixed seed for reproducibility
        logger.info("Shuffled {} samples for training", shuffled.size());

        // Balance classes by undersampling majority class to match minority class
        // This ensures fair comparison across datasets (IMDB 50/50, Amazon 50/50, Yelp 75/25 -> 50/50)
        // Uses shared utility to ensure consistent balancing across training and data preparation
        return DatasetUtils.balanceByUndersampling(shuffled, 42L);
    }

    /**
     * Creates a new classifier instance with fresh preprocessing components.
     * Each classifier gets its own TextPreprocessor and WekaInstancesConverter
     * to avoid state conflicts when training multiple models.
     */
    private SentimentClassifier createClassifier(AlgorithmType algorithmType) {
        // Get fresh instances from prototype-scoped beans
        TextPreprocessor textPreprocessor = textPreprocessorProvider.getObject();
        WekaInstancesConverter wekaInstancesConverter = wekaInstancesConverterProvider.getObject();

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

    private ClassifierEvaluationResult evaluateOnTestSet(SentimentClassifier classifier,
                                                           List<Dataset> testData) throws Exception {
        if (!(classifier instanceof ClassifierTrainingTemplate)) {
            throw new UnsupportedOperationException("Classifier does not support evaluation");
        }

        ClassifierTrainingTemplate<?> template = (ClassifierTrainingTemplate<?>) classifier;

        // Convert test data to Weka Instances using the trained converter
        Instances testInstances = template.getConverter().transformDatasets(testData);

        // Evaluate using the classifier's evaluate method
        return template.evaluate(testInstances);
    }

    private void saveModel(SentimentClassifier classifier, String outputPath,
                           AlgorithmType algorithmType) throws IOException {

        Path modelPath = Paths.get(outputPath);
        Files.createDirectories(modelPath.getParent());

        // All classifiers extend ClassifierTrainingTemplate and use generic Weka persistence
        if (classifier instanceof ClassifierTrainingTemplate) {
            WekaModelPersistence<ClassifierTrainingTemplate<?>> persistence = new WekaModelPersistence<>();
            persistence.saveModel((ClassifierTrainingTemplate<?>) classifier, modelPath);

            // Log metadata for easy reference and verification
            try {
                WekaModelPersistence.ModelMetadata metadata = persistence.getModelMetadata(modelPath);
                logger.info("Model metadata: {}", metadata);
            } catch (Exception e) {
                logger.warn("Failed to read metadata after save: {}", e.getMessage());
            }
        } else {
            throw new UnsupportedOperationException(
                    "Classifier type " + algorithmType + " does not support persistence");
        }
    }

    private void saveTrainingMetadata(String dataPath, String modelPath,
                                       AlgorithmType algorithmType,
                                       StratifiedDataSplitter.DataSplit split,
                                       ClassifierEvaluationResult evaluation,
                                       long trainingTimeMs,
                                       SentimentClassifier classifier,
                                       String splitManifestHash,
                                       String testSplitHash) {
        try {
            Path modelFilePath = Paths.get(modelPath);
            Path metadataPath = TrainingMetadata.getMetadataPath(modelFilePath);
            String modelFileName = modelFilePath.getFileName().toString();
            String baseName = modelFileName.replaceAll("\\.[^.]+$", "");

            // Extract hyperparameters from trained model
            Map<String, Object> hyperparameters = extractHyperparameters(classifier, algorithmType);

            // Generate model ID from timestamp and algorithm
            String modelId = algorithmType.name().toLowerCase() + "-" +
                            java.time.Instant.now().toString().substring(0, 19).replace(":", "-");

            // Infer dataset info
            String datasetName = inferDatasetName(dataPath);
            String datasetVersion = inferDatasetVersion(datasetName);
            String datasetSource = inferDatasetSourceUrl(datasetName);

            // Calculate class balance from training data
            Map<String, Integer> classBalance = calculateClassBalance(split.train);

            // Get kernel info for SVM
            String kernel = null;
            String notes = null;
            if (classifier instanceof SVMClassifier svm && svm.getOptimalConfig() != null) {
                SVMConfig config = svm.getOptimalConfig();
                kernel = config.getKernelType().name().toLowerCase();

                // Generate notes based on whether grid search actually ran
                if (config.getCvAccuracy() != null) {
                    notes = String.format("Tuned SVM with C=%.4f, 5-fold CV accuracy=%.2f%%",
                            config.getC(), config.getCvAccuracy() * 100);
                } else {
                    notes = String.format("SVM with C=%.4f, %s kernel (default config, no grid search)",
                            config.getC(), kernel);
                }
            } else if (algorithmType == AlgorithmType.SVM) {
                kernel = "linear";
                notes = "SVM with default parameters";
            }

            // Get git commit hash
            String commitHash = getGitCommitHash();

            // Build feature importance file path
            String featureImportanceFile = baseName + "-feature-importance.json";

            // Build metadata with ALL fields populated
            TrainingMetadata metadata = TrainingMetadata.builder()
                    .modelId(modelId)
                    .algorithm(algorithmType)
                    .hyperparameters(hyperparameters)
                    .datasetPath(dataPath)
                    .datasetInfo(datasetName, datasetVersion, datasetSource, classBalance)
                    .trainingSamples(split.train.size())
                    .sampleCounts(split.train.size(), split.validation.size(), split.test.size())
                    .splitHashes(splitManifestHash, testSplitHash)
                    .preprocessing(featureConfig.getMaxFeatures(), featureConfig.isUseTfidf(), featureConfig.isUseBigrams())
                    .evaluationResults(evaluation)
                    .trainedAt(java.time.Instant.now())
                    .trainingDuration(trainingTimeMs)
                    .modelFile(modelFileName, Files.size(modelFilePath))
                    .modelInfo(algorithmType.name(), kernel, "Weka 3.9.6", "2.0.0", notes)
                    .artifacts(modelFileName, featureImportanceFile)
                    .reproducibility(42L, System.getProperty("java.version"), "3.9.6", commitHash)
                    .build();

            // Save metadata JSON
            metadata.save(metadataPath);
            logger.info("Training metadata saved to: {}", metadataPath);

        } catch (Exception e) {
            logger.error("ARTIFACT_SAVE_FAILED: metadata - {}", e.getMessage());
            logger.error("Training succeeded but metadata.json was NOT saved. Check permissions and disk space.");
        }
    }

    /**
     * Extract hyperparameters from trained classifier.
     */
    private Map<String, Object> extractHyperparameters(SentimentClassifier classifier, AlgorithmType algorithmType) {
        Map<String, Object> hyperparameters = new LinkedHashMap<>();
        hyperparameters.put("algorithm", algorithmType.name());

        if (classifier instanceof SVMClassifier svm && svm.getOptimalConfig() != null) {
            SVMConfig config = svm.getOptimalConfig();
            hyperparameters.put("kernel", config.getKernelType().name().toLowerCase());
            hyperparameters.put("c_parameter", config.getC());

            // Determine if grid search actually ran by checking for CV metrics
            boolean gridSearchRan = config.getCvAccuracy() != null;
            hyperparameters.put("tuning_method", gridSearchRan ? "grid_search" : "default");

            if (gridSearchRan) {
                hyperparameters.put("cv_folds", 5);
                hyperparameters.put("cv_accuracy", config.getCvAccuracy());
                if (config.getCvStdDev() != null) {
                    hyperparameters.put("cv_std_dev", config.getCvStdDev());
                }
            }
        } else if (algorithmType == AlgorithmType.SVM) {
            hyperparameters.put("kernel", "linear");
            hyperparameters.put("c_parameter", 1.0);
        } else if (algorithmType == AlgorithmType.LOGISTIC_REGRESSION) {
            hyperparameters.put("ridge", 1.0E-8); // Weka default
        } else if (algorithmType == AlgorithmType.RANDOM_FOREST) {
            hyperparameters.put("num_trees", 100); // Default
            hyperparameters.put("max_depth", 0); // Unlimited
        } else if (algorithmType == AlgorithmType.NAIVE_BAYES) {
            hyperparameters.put("use_kernel_estimator", false);
        }

        return hyperparameters;
    }

    /**
     * Calculate class balance from training data.
     */
    private Map<String, Integer> calculateClassBalance(List<Dataset> data) {
        Map<String, Integer> balance = new LinkedHashMap<>();
        long positive = data.stream().filter(d -> d.getSentiment() == Dataset.SentimentLabel.POSITIVE).count();
        long negative = data.stream().filter(d -> d.getSentiment() == Dataset.SentimentLabel.NEGATIVE).count();
        balance.put("positive", (int) positive);
        balance.put("negative", (int) negative);
        return balance;
    }

    /**
     * Infer dataset version from name.
     */
    private String inferDatasetVersion(String datasetName) {
        return switch (datasetName.toLowerCase()) {
            case "amazon_polarity" -> "2015-mcauley";
            case "imdb_50k" -> "2011-maas";
            case "yelp" -> "2015-yelp-challenge";
            default -> "1.0";
        };
    }

    /**
     * Infer dataset source URL from name.
     */
    private String inferDatasetSourceUrl(String datasetName) {
        return switch (datasetName.toLowerCase()) {
            case "amazon_polarity" -> "https://huggingface.co/datasets/amazon_polarity";
            case "imdb_50k" -> "https://ai.stanford.edu/~amaas/data/sentiment/";
            case "yelp" -> "https://www.yelp.com/dataset";
            default -> "custom_dataset";
        };
    }

    /**
     * Get current git commit hash.
     */
    private String getGitCommitHash() {
        try {
            ProcessBuilder pb = new ProcessBuilder("git", "rev-parse", "--short", "HEAD");
            pb.redirectErrorStream(true);
            Process process = pb.start();
            String output = new String(process.getInputStream().readAllBytes()).trim();
            process.waitFor();
            return process.exitValue() == 0 ? output : null;
        } catch (Exception e) {
            return null;
        }
    }

    /**
     * Save train and test splits to data/processed/{dataset}/ for reproducibility and auditability.
     * Enables exact reconstruction of what data the model trained on.
     *
     * IMPORTANT: If a manifest already exists (prepared splits), this method will NOT overwrite.
     * This preserves data integrity for cross-domain evaluation consistency.
     *
     * @param trainData Training portion from stratified split
     * @param testData Test portion from stratified split
     * @param dataPath Original data path (used to infer dataset name)
     */
    private void saveDataSplits(List<Dataset> trainData, List<Dataset> testData, String dataPath) {
        String datasetName = inferDatasetName(dataPath);
        Path processedDir = Paths.get("data/processed", datasetName).toAbsolutePath();

        // Check if manifest exists - if so, DON'T overwrite (immutable splits)
        if (SplitManifest.exists(processedDir)) {
            logger.info("Manifest exists at {} - NOT overwriting prepared splits", processedDir);
            logger.info("This preserves data integrity for reproducible cross-domain evaluation");
            return;
        }

        try {
            Files.createDirectories(processedDir);
        } catch (IOException e) {
            logger.error("ARTIFACT_SAVE_FAILED: Could not create directory {} - {}", processedDir, e.getMessage());
            return;
        }

        // Save both splits (backward compatibility path - no manifest)
        // Uses shared CSV writer to ensure format consistency with data preparation pipeline
        logger.warn("Saving splits without manifest (legacy mode). Run ./scripts/prepare_data.sh for proper setup.");
        try {
            SimpleDatasetLoader.writeCsv(trainData, processedDir.resolve("train.csv"));
            logger.info("Saved train split ({} samples) to: {}", trainData.size(), processedDir.resolve("train.csv"));
        } catch (IOException e) {
            logger.error("ARTIFACT_SAVE_FAILED: train split - {}", e.getMessage());
        }
        try {
            SimpleDatasetLoader.writeCsv(testData, processedDir.resolve("test.csv"));
            logger.info("Saved test split ({} samples) to: {}", testData.size(), processedDir.resolve("test.csv"));
        } catch (IOException e) {
            logger.error("ARTIFACT_SAVE_FAILED: test split - {}", e.getMessage());
        }
    }

    /**
     * Infer dataset name from data path.
     * Examples:
     *   data/raw/imdb_50k/IMDB Dataset.csv -> imdb_50k
     *   data/raw/amazon_polarity/train.csv -> amazon_polarity
     *   data/raw/yelp/yelp_reviews.csv -> yelp
     */
    private String inferDatasetName(String dataPath) {
        Path path = Paths.get(dataPath);
        Path parent = path.getParent();

        if (parent != null && parent.getParent() != null) {
            // Get the directory name (e.g., "imdb_50k" from "data/raw/imdb_50k/...")
            return parent.getFileName().toString();
        }

        // Fallback: use filename without extension
        String filename = path.getFileName().toString();
        int dotIndex = filename.lastIndexOf('.');
        return dotIndex > 0 ? filename.substring(0, dotIndex) : filename;
    }

    private void analyzeAndPrintFeatureImportance(SentimentClassifier classifier,
                                                   StratifiedDataSplitter.DataSplit split,
                                                   int topFeaturesCount,
                                                   String modelPath,
                                                   AlgorithmType algorithmType) {
        try {
            logger.info("Converting training data to Weka Instances for feature analysis...");

            // Extract the trained converter from the classifier
            WekaInstancesConverter trainedConverter = ((ClassifierTrainingTemplate<?>) classifier).getConverter();

            // Convert the training data to Weka Instances using the already-trained converter
            Instances trainedInstances = trainedConverter.transformDatasets(split.train);

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
                logger.info("Feature importance saved to: {}", featureImportancePath);
                logger.info("Use this for runtime API exploration via /api/v1/model/feature-importance");
            } catch (IOException e) {
                logger.warn("Failed to save feature importance to file: {}", e.getMessage());
            }

        } catch (Exception e) {
            logger.error("Failed to analyze feature importance: {}", e.getMessage(), e);
        }
    }

    private void printFeatureImportanceReport(FeatureImportanceResult result,
                                               int topFeaturesCount,
                                               AlgorithmType algorithmType) {
        logger.info("=".repeat(80));
        logger.info("FEATURE IMPORTANCE ANALYSIS - {}", algorithmType.getDisplayName());
        logger.info("=".repeat(80));
        logger.info("Analysis completed in {}ms", result.analysisTimeMs());
        logger.info("");
        logger.info("Statistics:");
        logger.info("  Total features: {}", result.allFeatures().size());
        logger.info("  Mean absolute weight: {}", String.format("%.6f", result.statistics().mean()));
        logger.info("  Std deviation: {}", String.format("%.6f", result.statistics().stdDev()));
        logger.info("  Median: {}", String.format("%.6f", result.statistics().median()));
        logger.info("  95th percentile: {}", String.format("%.6f", result.statistics().percentile95()));
        logger.info("=".repeat(80));

        result.logTopFeatures(topFeaturesCount, logger);

        logger.info("INTERPRETATION:");
        logger.info("  • Features with high |weight| strongly influence predictions");
        logger.info("  • Positive weights → positive sentiment, negative → negative sentiment");
        logger.info("  • Features near zero are non-discriminative");
        logger.info("  • This analysis helps understand what the model learned");
        logger.info("=".repeat(80));
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
