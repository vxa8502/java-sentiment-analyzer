package sentiment.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import sentiment.models.*;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Spring configuration for sentiment analysis components.
 */
@Configuration
@org.springframework.boot.context.properties.EnableConfigurationProperties(FeatureExtractionProperties.class)
@org.springframework.context.annotation.Profile("!training")
public class SentimentConfiguration {

    private static final Logger logger = LoggerFactory.getLogger(SentimentConfiguration.class);

    @Autowired
    private ObjectProvider<TextPreprocessor> textPreprocessorProvider;

    @Autowired
    private ObjectProvider<WekaInstancesConverter> wekaInstancesConverterProvider;

    @Value("${sentiment.model-type}")
    private String modelType;

    @Value("${sentiment.models.svm-model-path:./models/svm-model.ser}")
    private String svmModelPath;

    @Value("${sentiment.models.naive-bayes-model-path:./models/naive_bayes-model.ser}")
    private String naiveBayesModelPath;

    @Value("${sentiment.models.random-forest-model-path:./models/random_forest-model.ser}")
    private String randomForestModelPath;

    @Value("${sentiment.models.logistic-model-path:./models/logistic_regression-model.ser}")
    private String logisticModelPath;

    @Value("${sentiment.models.production-model-path:./models/production/sentiment_model.ser}")
    private String productionModelPath;


    /**
     * Creates WekaModelPersistence bean for model save/load operations.
     * Generic type allows use with any Weka-based classifier.
     */
    @Bean
    @SuppressWarnings("rawtypes")
    public WekaModelPersistence wekaModelPersistence() {
        return new WekaModelPersistence<>();
    }

    /**
     * Provides the actual model path used for the loaded classifier.
     * This is needed for feature importance file lookup.
     */
    @Bean
    public String loadedModelPath() {
        if ("PRODUCTION".equalsIgnoreCase(modelType)) {
            return productionModelPath;
        }
        AlgorithmType algorithm = AlgorithmType.fromString(modelType);
        return getModelPath(algorithm);
    }

    /**
     * Creates the sentiment classifier bean by loading a pre-trained model.
     * This bean is eagerly initialized on application startup to ensure
     * the model is ready before accepting requests.
     * This ensures predictable startup time and prevents training in production.
     */
    @Bean
    public SentimentClassifier sentimentClassifier() {
        logger.info("Initializing sentiment classifier bean...");
        logger.info("Requested algorithm: {}", modelType);

        // Handle PRODUCTION model type specially - loads the promoted best model
        if ("PRODUCTION".equalsIgnoreCase(modelType)) {
            return loadProductionModel();
        }

        // Parse algorithm type
        AlgorithmType algorithm;
        try {
            algorithm = AlgorithmType.fromString(modelType);
        } catch (IllegalArgumentException e) {
            throw new IllegalStateException("Invalid sentiment.model-type: " + modelType + ". " + e.getMessage());
        }

        // Create classifier instance based on algorithm
        SentimentClassifier classifier = createClassifier(algorithm);
        String modelPath = getModelPath(algorithm);

        logger.info("Loading pre-trained {} model from: {}", algorithm.getDisplayName(), modelPath);

        @SuppressWarnings("unchecked")
        WekaModelPersistence<ClassifierTrainingTemplate<?>> persistence = wekaModelPersistence();
        Path path = Paths.get(modelPath);

        // Check if model exists
        if (!Files.exists(path)) {
            String errorMsg = String.format(
                    "Pre-trained model not found at: %s\n\n" +
                    "To train models, run from project root:\n" +
                    "  ./scripts/train_all_models.sh\n\n" +
                    "Or set the appropriate environment variable:\n" +
                    "  - SVM: SENTIMENT_SVM_MODEL\n" +
                    "  - Naive Bayes: SENTIMENT_NB_MODEL\n" +
                    "  - Random Forest: SENTIMENT_RF_MODEL\n" +
                    "  - Logistic Regression: SENTIMENT_LR_MODEL\n\n" +
                    "See docs/TRAINING.md for details.\n",
                    modelPath);

            logger.error(errorMsg);
            throw new IllegalStateException(errorMsg);
        }

        if (!Files.isReadable(path)) {
            throw new IllegalStateException("Model file exists but is not readable: " + modelPath);
        }

        try {
            // Validate model before loading
            if (!persistence.isValidModel(path)) {
                throw new IllegalStateException("Model validation failed: " + modelPath);
            }

            logger.info("Model validation passed. Loading model...");
            @SuppressWarnings("unchecked")
            ClassifierTrainingTemplate<?> template = (ClassifierTrainingTemplate<?>) classifier;
            persistence.loadModel(template, path);

            // Log model metadata
            WekaModelPersistence.ModelMetadata metadata = persistence.getModelMetadata(path);
            logger.info(" Pre-trained {} model loaded successfully: {}", algorithm.getDisplayName(), metadata);
            logger.info("Startup model loading completed - FAST PATH");

            return classifier;

        } catch (IOException | ClassNotFoundException e) {
            String errorMsg = String.format(
                    "Failed to load pre-trained model from: %s\n" +
                    "Error: %s\n\n" +
                    "The model file may be corrupted. Try training a new model via /admin/train endpoint.",
                    modelPath, e.getMessage());

            logger.error(errorMsg, e);
            throw new IllegalStateException(errorMsg, e);
        }
    }

    /**
     * Loads the production model (best generalizing model promoted by promote_to_production.sh).
     * Reads metadata to determine the algorithm type dynamically.
     */
    private SentimentClassifier loadProductionModel() {
        logger.info("Loading PRODUCTION model from: {}", productionModelPath);
        Path modelPath = Paths.get(productionModelPath);

        if (!Files.exists(modelPath)) {
            throw new IllegalStateException(
                "Production model not found at: " + productionModelPath + "\n" +
                "Run ./scripts/promote_to_production.sh to create the production model.");
        }

        // Read metadata to determine algorithm type
        Path metadataPath = Paths.get(productionModelPath.replace(".ser", ".metadata.json"));
        AlgorithmType algorithm = readAlgorithmFromMetadata(metadataPath);

        logger.info("Production model algorithm: {}", algorithm.getDisplayName());

        SentimentClassifier classifier = createClassifier(algorithm);
        @SuppressWarnings("unchecked")
        WekaModelPersistence<ClassifierTrainingTemplate<?>> persistence = wekaModelPersistence();

        try {
            if (!persistence.isValidModel(modelPath)) {
                throw new IllegalStateException("Production model validation failed: " + productionModelPath);
            }

            @SuppressWarnings("unchecked")
            ClassifierTrainingTemplate<?> template = (ClassifierTrainingTemplate<?>) classifier;
            persistence.loadModel(template, modelPath);

            logger.info("Production model loaded successfully");
            return classifier;

        } catch (IOException | ClassNotFoundException e) {
            throw new IllegalStateException("Failed to load production model: " + e.getMessage(), e);
        }
    }

    /**
     * Reads the algorithm type from model metadata JSON file.
     */
    private AlgorithmType readAlgorithmFromMetadata(Path metadataPath) {
        if (!Files.exists(metadataPath)) {
            logger.warn("Metadata file not found at: {}. Defaulting to SVM.", metadataPath);
            return AlgorithmType.SVM;
        }

        try {
            String content = Files.readString(metadataPath);
            // Simple parsing - look for "algorithm": "value"
            if (content.contains("\"algorithm\"")) {
                int start = content.indexOf("\"algorithm\"");
                int colonPos = content.indexOf(":", start);
                int quoteStart = content.indexOf("\"", colonPos + 1);
                int quoteEnd = content.indexOf("\"", quoteStart + 1);
                String algorithmStr = content.substring(quoteStart + 1, quoteEnd);
                return AlgorithmType.fromString(algorithmStr);
            }
            logger.warn("Algorithm not found in metadata. Defaulting to SVM.");
            return AlgorithmType.SVM;
        } catch (IOException e) {
            logger.warn("Failed to read metadata: {}. Defaulting to SVM.", e.getMessage());
            return AlgorithmType.SVM;
        }
    }

    /**
     * Factory method to create classifier instance based on algorithm type.
     * Gets fresh instances of preprocessing components from prototype-scoped beans.
     */
    private SentimentClassifier createClassifier(AlgorithmType algorithm) {
        // Get fresh instances for this classifier
        TextPreprocessor textPreprocessor = textPreprocessorProvider.getObject();
        WekaInstancesConverter wekaInstancesConverter = wekaInstancesConverterProvider.getObject();

        return switch (algorithm) {
            case SVM -> new SVMClassifier(textPreprocessor, wekaInstancesConverter);
            case NAIVE_BAYES -> new NaiveBayesClassifier(textPreprocessor, wekaInstancesConverter);
            case RANDOM_FOREST -> new RandomForestClassifier(textPreprocessor, wekaInstancesConverter);
            case LOGISTIC_REGRESSION -> new LogisticRegressionClassifier(textPreprocessor, wekaInstancesConverter);
            default -> throw new IllegalArgumentException(
                    "Algorithm not supported for runtime loading: " + algorithm + ". " +
                    "Supported: SVM, NAIVE_BAYES, RANDOM_FOREST, LOGISTIC_REGRESSION");
        };
    }

    /**
     * Returns the model file path for the specified algorithm.
     */
    private String getModelPath(AlgorithmType algorithm) {
        return switch (algorithm) {
            case SVM -> svmModelPath;
            case NAIVE_BAYES -> naiveBayesModelPath;
            case RANDOM_FOREST -> randomForestModelPath;
            case LOGISTIC_REGRESSION -> logisticModelPath;
            default -> throw new IllegalArgumentException("No model path configured for algorithm: " + algorithm);
        };
    }
}
