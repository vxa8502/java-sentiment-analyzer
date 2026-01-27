package sentiment.models;

import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import sentiment.training.TrainingMetadata;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.function.BiFunction;

/**
 * Model loader with metadata validation.
 * Ensures every model has companion metadata and validates compatibility.
 *
 * @author Victoria Alabi
 */
public class ModelLoader {

    /**
     * Load model with metadata validation.
     * Throws exception if metadata file is missing (non-negotiable requirement).
     *
     * @param modelPath path to .ser model file
     * @return loaded classifier with metadata attached
     * @throws IOException if model loading fails
     * @throws IllegalStateException if metadata file is missing
     */
    public static SentimentClassifier loadWithMetadata(String modelPath)
            throws IOException, ClassNotFoundException {

        Path path = Paths.get(modelPath);

        // Every model must have metadata for reproducibility
        Path metadataPath = TrainingMetadata.getMetadataPath(path);
        if (!Files.exists(metadataPath)) {
            throw new IllegalStateException(
                "Missing metadata file: " + metadataPath +
                ". All models must have companion metadata.json files."
            );
        }

        // Load metadata first
        TrainingMetadata metadata = TrainingMetadata.load(metadataPath);
        System.out.printf("✓ Loaded metadata: %s, trained on %s, accuracy=%.3f%n",
                metadata.getModelId(),
                metadata.getDataset().datasetName,
                metadata.getMetrics().testAccuracy);

        // Validate metadata points to correct model file
        if (metadata.getModelFile() != null &&
            !metadata.getModelFile().equals(path.getFileName().toString())) {
            System.err.printf("⚠ Metadata model_file mismatch: expected %s, got %s%n",
                    metadata.getModelFile(), path.getFileName());
        }

        // Load the actual model based on algorithm type
        SentimentClassifier classifier = loadModelByType(path, metadata);

        System.out.printf("✓ Loaded model: %s from %s%n",
                path.getFileName(), path.getParent());

        return classifier;
    }

    /**
     * Load model by algorithm type (dispatches to appropriate classifier factory).
     */
    private static SentimentClassifier loadModelByType(Path modelPath, TrainingMetadata metadata)
            throws IOException, ClassNotFoundException {

        AlgorithmType algorithmType = AlgorithmType.valueOf(metadata.getAlgorithm().toUpperCase());

        return switch (algorithmType) {
            case SVM -> loadClassifier(modelPath, SVMClassifier::new);
            case NAIVE_BAYES -> loadClassifier(modelPath, NaiveBayesClassifier::new);
            case RANDOM_FOREST -> loadClassifier(modelPath, RandomForestClassifier::new);
            case LOGISTIC_REGRESSION -> loadClassifier(modelPath, LogisticRegressionClassifier::new);
            default -> throw new IllegalArgumentException("Unsupported algorithm: " + algorithmType);
        };
    }

    /**
     * Creates the preprocessing pipeline components used by all classifier types.
     */
    private static PreprocessingPipeline createPreprocessingPipeline() {
        sentiment.config.FeatureExtractionProperties featureConfig =
            new sentiment.config.FeatureExtractionProperties();
        sentiment.preprocessing.ContractionExpander expander =
            new sentiment.preprocessing.ContractionExpander();
        sentiment.preprocessing.AdvancedTokenizer tokenizer =
            new sentiment.preprocessing.AdvancedTokenizer(false, 1, true);
        sentiment.preprocessing.IntelligentStopwordRemover stopwordRemover =
            new sentiment.preprocessing.IntelligentStopwordRemover();
        sentiment.preprocessing.TextPreprocessor preprocessor =
            new sentiment.preprocessing.TextPreprocessor(expander, tokenizer, stopwordRemover, featureConfig);
        sentiment.preprocessing.WekaInstancesConverter converter =
            new sentiment.preprocessing.WekaInstancesConverter(preprocessor, featureConfig);

        return new PreprocessingPipeline(preprocessor, converter);
    }

    /**
     * Container for preprocessing pipeline components.
     */
    private record PreprocessingPipeline(
            TextPreprocessor preprocessor,
            WekaInstancesConverter converter) {}

    /**
     * Generic classifier loader using a factory function.
     * Consolidates the common pattern of creating preprocessing pipeline,
     * instantiating classifier, and loading persisted model state.
     *
     * @param modelPath path to the serialized model file
     * @param factory constructor reference for the classifier (e.g., SVMClassifier::new)
     * @param <T> classifier type extending ClassifierTrainingTemplate
     * @return loaded and initialized classifier
     */
    private static <T extends ClassifierTrainingTemplate<?>> T loadClassifier(
            Path modelPath,
            BiFunction<TextPreprocessor, WekaInstancesConverter, T> factory)
            throws IOException, ClassNotFoundException {

        PreprocessingPipeline pipeline = createPreprocessingPipeline();
        T classifier = factory.apply(pipeline.preprocessor(), pipeline.converter());
        return new WekaModelPersistence<T>().loadModel(classifier, modelPath);
    }

    /**
     * Load all models for a given algorithm from models/<algorithm>/ directory.
     *
     * @param algorithm algorithm name (svm, naive_bayes, random_forest)
     * @return map of domain name -> classifier
     */
    public static java.util.Map<String, SentimentClassifier> loadAllForAlgorithm(String algorithm)
            throws IOException {

        java.util.Map<String, SentimentClassifier> models = new java.util.HashMap<>();
        Path algoDir = Paths.get("models", algorithm);

        if (!Files.exists(algoDir)) {
            System.err.println("⚠ Model directory not found: " + algoDir);
            return models;
        }

        // Find all .ser files
        try (var stream = Files.list(algoDir)) {
            stream.filter(p -> p.toString().endsWith(".ser"))
                  .forEach(modelPath -> {
                      try {
                          // Extract domain name from filename: imdb_50k_svm_model.ser -> imdb_50k
                          String filename = modelPath.getFileName().toString();
                          String domain = extractDomainName(filename, algorithm);

                          SentimentClassifier classifier = loadWithMetadata(modelPath.toString());
                          models.put(domain, classifier);

                          System.out.printf("✓ Loaded %s model trained on %s%n", algorithm, domain);

                      } catch (Exception e) {
                          System.err.println("✗ Failed to load " + modelPath + ": " + e.getMessage());
                      }
                  });
        }

        return models;
    }

    /**
     * Extract domain name from model filename.
     * Example: imdb_50k_svm_model.ser -> imdb_50k
     */
    private static String extractDomainName(String filename, String algorithm) {
        // Remove extension: imdb_50k_svm_model.ser -> imdb_50k_svm_model
        String base = filename.replace(".ser", "");

        // Remove algorithm suffix: imdb_50k_svm_model -> imdb_50k
        base = base.replace("_" + algorithm, "");
        base = base.replace("_model", "");

        return base;
    }
}
