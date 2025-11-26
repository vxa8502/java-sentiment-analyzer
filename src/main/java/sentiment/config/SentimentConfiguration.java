package sentiment.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import sentiment.models.SVMClassifier;
import sentiment.models.WekaModelPersistence;
import sentiment.models.SentimentClassifier;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

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

    @Value("${sentiment.models.svm-model-path:./models/svm-model.ser}")
    private String svmModelPath;

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
     * Creates the sentiment classifier bean by loading a pre-trained model.
     *
     * This bean is eagerly initialized on application startup to ensure
     * the model is ready before accepting requests.
     *
     * SIMPLIFIED BEHAVIOR:
     * ====================
     * - Always loads pre-trained model from disk (FAST STARTUP)
     * - Fails fast with clear error if model not found
     * - No fallback training (use /admin/train endpoint instead)
     *
     * To train models:
     * - Development: POST http://localhost:8080/admin/train
     * - CI/CD: Train models as part of build, package into Docker image
     * - Production: Bake pre-trained models into container
     *
     * This ensures predictable startup time and prevents training in production.
     */
    @Bean
    public SentimentClassifier sentimentClassifier() {
        logger.info("Initializing sentiment classifier bean...");
        logger.info("Loading pre-trained model from: {}", svmModelPath);

        SVMClassifier classifier = new SVMClassifier(textPreprocessor, wekaInstancesConverter);
        classifier.setClassImbalanceThreshold(classImbalanceThreshold);
        WekaModelPersistence<SVMClassifier> persistence = wekaModelPersistence();

        Path modelPath = Paths.get(svmModelPath);

        // Check if model exists
        if (!Files.exists(modelPath)) {
            String errorMsg = String.format(
                    "Pre-trained model not found at: %s\n\n" +
                    "To train a model:\n" +
                    "1. Use the admin API endpoint:\n" +
                    "   curl -X POST http://localhost:8080/admin/train \\\n" +
                    "     -H 'Content-Type: application/json' \\\n" +
                    "     -d '{\"dataPath\":\"./data/datasets/Reviews.csv\",\"outputPath\":\"%s\",\"algorithm\":\"SVM\",\"maxSamples\":10000}'\n\n" +
                    "2. Or set SENTIMENT_SVM_MODEL environment variable to an existing model path\n",
                    svmModelPath, svmModelPath);

            logger.error(errorMsg);
            throw new IllegalStateException(errorMsg);
        }

        if (!Files.isReadable(modelPath)) {
            throw new IllegalStateException("Model file exists but is not readable: " + svmModelPath);
        }

        try {
            // Validate model before loading
            if (!persistence.isValidModel(modelPath)) {
                throw new IllegalStateException("Model validation failed: " + svmModelPath);
            }

            logger.info("Model validation passed. Loading model...");
            persistence.loadModel(classifier, modelPath);

            // Log model metadata
            WekaModelPersistence.ModelMetadata metadata = persistence.getModelMetadata(modelPath);
            logger.info("✅ Pre-trained model loaded successfully: {}", metadata);
            logger.info("Startup model loading completed - FAST PATH");

            return classifier;

        } catch (IOException | ClassNotFoundException e) {
            String errorMsg = String.format(
                    "Failed to load pre-trained model from: %s\n" +
                    "Error: %s\n\n" +
                    "The model file may be corrupted. Try training a new model via /admin/train endpoint.",
                    svmModelPath, e.getMessage());

            logger.error(errorMsg, e);
            throw new IllegalStateException(errorMsg, e);
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
