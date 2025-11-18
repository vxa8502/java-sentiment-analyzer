package sentiment.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import sentiment.data.Dataset;
import sentiment.data.SimpleDatasetLoader;
import sentiment.data.DatasetLoadResult;
import sentiment.models.SVMClassifier;
import sentiment.models.WekaModelPersistence;
import sentiment.models.SentimentClassifier;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Spring configuration for sentiment analysis components.
 *
 * Provides beans for:
 * - SentimentClassifier (trained and ready for use)
 * - Configuration properties
 *
 * NOTE: This configuration is disabled during training (profile=training)
 * to prevent conflicts with the model training process.
 */
@Configuration
@org.springframework.context.annotation.Profile("!training")
public class SentimentConfiguration {

    private static final Logger logger = LoggerFactory.getLogger(SentimentConfiguration.class);

    @Autowired
    private TextPreprocessor textPreprocessor;

    @Autowired
    private WekaInstancesConverter wekaInstancesConverter;

    @Autowired
    private SimpleDatasetLoader datasetLoader;

    @Value("${sentiment.data.training-file:./data/datasets/Reviews.csv}")
    private String trainingDataPath;

    @Value("${sentiment.data.max-training-samples:10000}")
    private int maxTrainingSamples;

    @Value("${sentiment.models.svm-model-path:./models/svm-model.ser}")
    private String svmModelPath;

    @Value("${sentiment.models.prefer-pretrained:true}")
    private boolean preferPretrained;

    @Value("${sentiment.models.require-pretrained:false}")
    private boolean requirePretrained;

    @Value("${sentiment.svm.class-imbalance-threshold:3.0}")
    private double classImbalanceThreshold;

    /**
     * Creates WekaModelPersistence bean for model save/load operations.
     */
    @Bean
    public WekaModelPersistence<SVMClassifier> wekaModelPersistence() {
        return new WekaModelPersistence<>();
    }

    /**
     * Creates and trains the sentiment classifier bean.
     *
     * This bean is eagerly initialized on application startup to ensure
     * the model is ready before accepting requests.
     *
     * UPDATED BEHAVIOR (Pre-trained Model Support):
     * ==============================================
     * 1. If prefer-pretrained=true and model exists, loads it (FAST STARTUP)
     * 2. If prefer-pretrained=true but no model exists:
     *    a. If require-pretrained=true, fails startup with clear error
     *    b. If require-pretrained=false, trains minimal fallback model (SLOW)
     * 3. If prefer-pretrained=false, always trains new model
     *
     * Production recommendation:
     * - Set prefer-pretrained=true and require-pretrained=true
     * - Pre-train models using sentiment.training.ModelTrainingCLI
     * - This ensures predictable startup time and prevents training in production
     *
     * Development recommendation:
     * - Set prefer-pretrained=true and require-pretrained=false
     * - This allows fallback training if models aren't pre-trained
     */
    @Bean
    public SentimentClassifier sentimentClassifier() {
        logger.info("Initializing sentiment classifier bean...");
        logger.info("Configuration: prefer-pretrained={}, require-pretrained={}",
                preferPretrained, requirePretrained);

        try {
            SVMClassifier classifier = new SVMClassifier(textPreprocessor, wekaInstancesConverter);
            classifier.setClassImbalanceThreshold(classImbalanceThreshold);
            WekaModelPersistence<SVMClassifier> persistence = wekaModelPersistence();

            if (preferPretrained) {
                // Try to load pre-trained model
                Path modelPath = Paths.get(svmModelPath);

                if (Files.exists(modelPath) && Files.isReadable(modelPath)) {
                    logger.info("Found pre-trained model at {}", svmModelPath);

                    try {
                        // Validate model before loading
                        if (persistence.isValidModel(modelPath)) {
                            logger.info("Model validation passed. Loading model...");
                            persistence.loadModel(classifier, modelPath);

                            // Log model metadata
                            WekaModelPersistence.ModelMetadata metadata = persistence.getModelMetadata(modelPath);
                            logger.info("✅ Pre-trained model loaded successfully: {}", metadata);
                            logger.info("Startup model loading completed - FAST PATH (no training)");

                            return classifier;
                        } else {
                            logger.warn("Model file exists but validation failed at: {}", modelPath);
                            throw new IOException("Model validation failed");
                        }
                    } catch (IOException | ClassNotFoundException e) {
                        logger.error("Failed to load pre-trained model from {}: {}",
                                svmModelPath, e.getMessage());

                        if (requirePretrained) {
                            throw new IllegalStateException(
                                    "Pre-trained model required but failed to load from: " + svmModelPath +
                                    "\nTo fix: Run sentiment.training.ModelTrainingCLI to generate models, " +
                                    "or set sentiment.models.require-pretrained=false", e);
                        } else {
                            logger.warn("Falling back to training a new model (SLOW PATH)...");
                        }
                    }
                } else {
                    logger.info("No pre-trained model found at {}", svmModelPath);

                    if (requirePretrained) {
                        throw new IllegalStateException(
                                "Pre-trained model required but not found at: " + svmModelPath +
                                "\n\nTo fix this error:" +
                                "\n1. Run the model training tool:" +
                                "\n   mvn exec:java -Dexec.mainClass=\"sentiment.training.ModelTrainingCLI\"" +
                                "\n\n2. Or set sentiment.models.require-pretrained=false to allow fallback training" +
                                "\n\n3. Or provide a valid model path via SENTIMENT_SVM_MODEL environment variable");
                    } else {
                        logger.warn("Falling back to training a new model (SLOW PATH)...");
                    }
                }
            }

            // Fallback: Train new model from scratch
            logger.info("Training new model from scratch (this will take time)...");
            trainMinimalModel(classifier);

            logger.info("Sentiment classifier bean initialized successfully");
            return classifier;

        } catch (IllegalStateException e) {
            // Re-throw configuration errors
            throw e;
        } catch (Exception e) {
            logger.error("Failed to initialize sentiment classifier", e);

            if (requirePretrained) {
                throw new IllegalStateException("Failed to initialize required pre-trained model", e);
            }

            // Return an untrained classifier rather than failing startup
            logger.warn("Returning untrained classifier - API will return 503 until model is trained");
            SVMClassifier fallbackClassifier = new SVMClassifier(textPreprocessor, wekaInstancesConverter);
            fallbackClassifier.setClassImbalanceThreshold(classImbalanceThreshold);
            return fallbackClassifier;
        }
    }

    /**
     * Resolves a resource path that may be prefixed with "classpath:" or be a regular file path.
     *
     * @param path The path to resolve (e.g., "classpath:datasets/Reviews.csv" or "/absolute/path/file.csv")
     * @return File object if path is valid, null otherwise
     */
    private java.io.File resolveResourcePath(String path) {
        if (path == null || path.trim().isEmpty()) {
            return null;
        }

        try {
            if (path.startsWith("classpath:")) {
                // Strip classpath: prefix and load from classpath
                String resourcePath = path.substring("classpath:".length());
                var resource = getClass().getClassLoader().getResource(resourcePath);
                if (resource != null) {
                    return new java.io.File(resource.toURI());
                }
                logger.debug("Classpath resource not found: {}", resourcePath);
                return null;
            } else {
                // Treat as regular file path
                return new java.io.File(path);
            }
        } catch (Exception e) {
            logger.debug("Could not resolve resource path {}: {}", path, e.getMessage());
            return null;
        }
    }

    /**
     * Trains model using the configured dataset file.
     * Loads data using DatasetLoaderRegistry with intelligent detection.
     */
    private void trainMinimalModel(SVMClassifier classifier) throws Exception {
        logger.info("Loading training data from: {}", trainingDataPath);

        try {
            // Resolve the path (handles classpath: prefix)
            String resolvedPath = trainingDataPath;
            if (trainingDataPath.startsWith("classpath:")) {
                java.io.File file = resolveResourcePath(trainingDataPath);
                if (file != null && file.exists()) {
                    resolvedPath = file.getAbsolutePath();
                    logger.debug("Resolved classpath resource to: {}", resolvedPath);
                } else {
                    logger.warn("Could not resolve classpath resource: {}", trainingDataPath);
                    throw new Exception("Training data file not found: " + trainingDataPath);
                }
            }

            // Load dataset using SimpleDatasetLoader
            DatasetLoadResult loadResult = datasetLoader.loadWithMetadata(resolvedPath);
            List<Dataset> allData = loadResult.datasets();

            logger.info("Loaded {} reviews from {} (took {}ms)",
                    allData.size(),
                    loadResult.datasetType(),
                    loadResult.loadTimeMs());

            // Limit training data for faster startup (with randomization to avoid bias)
            List<Dataset> trainingData;
            if (allData.size() > maxTrainingSamples) {
                List<Dataset> shuffled = new ArrayList<>(allData);
                Collections.shuffle(shuffled, new java.util.Random(42)); // Fixed seed for reproducibility
                trainingData = shuffled.subList(0, maxTrainingSamples);
                logger.info("Randomly sampled {} training samples from {} total",
                        maxTrainingSamples, allData.size());
            } else {
                trainingData = allData;
            }

            // Log dataset statistics
            long positive = trainingData.stream().filter(d -> d.getSentiment() == Dataset.SentimentLabel.POSITIVE).count();
            long negative = trainingData.stream().filter(d -> d.getSentiment() == Dataset.SentimentLabel.NEGATIVE).count();
            long neutral = trainingData.stream().filter(d -> d.getSentiment() == Dataset.SentimentLabel.NEUTRAL).count();

            logger.info("Dataset distribution: positive={}, negative={}, neutral={}", positive, negative, neutral);

            // Train the classifier
            classifier.train(trainingData);
            logger.info("Model training completed on {} samples", trainingData.size());

        } catch (Exception e) {
            logger.error("Failed to load training data from {}: {}", trainingDataPath, e.getMessage());
            logger.warn("Falling back to minimal hardcoded training data");

            // Fallback to minimal training data if file loading fails
            List<Dataset> minimalTrainingData = List.of(
                new Dataset("This is absolutely amazing and wonderful!", "positive"),
                new Dataset("I love this product, it's fantastic!", "positive"),
                new Dataset("Great quality and excellent service!", "positive"),
                new Dataset("Best purchase I've ever made!", "positive"),
                new Dataset("Highly recommend this to everyone!", "positive"),
                new Dataset("This is terrible and horrible!", "negative"),
                new Dataset("I hate this product, it's awful!", "negative"),
                new Dataset("Worst experience ever, very disappointed!", "negative"),
                new Dataset("Complete waste of money and time!", "negative"),
                new Dataset("Would not recommend to anyone!", "negative")
            );

            classifier.train(minimalTrainingData);
            logger.info("Minimal fallback model training completed");
        }
    }

    /**
     * Configuration properties for sentiment analysis.
     * Binds to sentiment.* properties in application.yml
     */
    @ConfigurationProperties(prefix = "sentiment")
    public static class SentimentProperties {
        private ModelConfig model = new ModelConfig();
        private PerformanceConfig performance = new PerformanceConfig();

        public ModelConfig getModel() {
            return model;
        }

        public void setModel(ModelConfig model) {
            this.model = model;
        }

        public PerformanceConfig getPerformance() {
            return performance;
        }

        public void setPerformance(PerformanceConfig performance) {
            this.performance = performance;
        }

        public static class ModelConfig {
            private String algorithm = "svm";
            private double confidenceThreshold = 0.7;
            private PreprocessingConfig preprocessing = new PreprocessingConfig();

            public String getAlgorithm() {
                return algorithm;
            }

            public void setAlgorithm(String algorithm) {
                this.algorithm = algorithm;
            }

            public double getConfidenceThreshold() {
                return confidenceThreshold;
            }

            public void setConfidenceThreshold(double confidenceThreshold) {
                this.confidenceThreshold = confidenceThreshold;
            }

            public PreprocessingConfig getPreprocessing() {
                return preprocessing;
            }

            public void setPreprocessing(PreprocessingConfig preprocessing) {
                this.preprocessing = preprocessing;
            }

            public static class PreprocessingConfig {
                private boolean removeStopwords = true;
                private boolean stemming = true;

                public boolean isRemoveStopwords() {
                    return removeStopwords;
                }

                public void setRemoveStopwords(boolean removeStopwords) {
                    this.removeStopwords = removeStopwords;
                }

                public boolean isStemming() {
                    return stemming;
                }

                public void setStemming(boolean stemming) {
                    this.stemming = stemming;
                }
            }
        }

        public static class PerformanceConfig {
            private boolean cachePredictions = false;
            private int maxBatchSize = 1000;

            public boolean isCachePredictions() {
                return cachePredictions;
            }

            public void setCachePredictions(boolean cachePredictions) {
                this.cachePredictions = cachePredictions;
            }

            public int getMaxBatchSize() {
                return maxBatchSize;
            }

            public void setMaxBatchSize(int maxBatchSize) {
                this.maxBatchSize = maxBatchSize;
            }
        }
    }
}
