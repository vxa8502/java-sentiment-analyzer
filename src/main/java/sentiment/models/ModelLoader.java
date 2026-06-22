package sentiment.models;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import sentiment.training.TrainingMetadata;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Set;
import java.util.function.BiFunction;

/**
 * Model loader with metadata validation.
 * Ensures every model has companion metadata and validates compatibility.
 *
 * @author Victoria Alabi
 */
public class ModelLoader {

    private static final Logger logger = LoggerFactory.getLogger(ModelLoader.class);

    /**
     * Allowed base directories for model loading.
     * Models can only be loaded from these directories or their subdirectories.
     * This prevents path traversal attacks.
     */
    private static final Set<String> ALLOWED_BASE_DIRS = Set.of(
            "models",
            "/app/models",
            "/tmp/models"  // For testing
    );

    /**
     * Load model with metadata validation.
     * Throws exception if metadata file is missing (non-negotiable requirement).
     *
     * @param modelPath path to .ser model file
     * @return loaded classifier with metadata attached
     * @throws IOException if model loading fails
     * @throws IllegalStateException if metadata file is missing
     * @throws SecurityException if model path is outside allowed directories
     */
    public static SentimentClassifier loadWithMetadata(String modelPath)
            throws IOException, ClassNotFoundException {

        Path path = Paths.get(modelPath);

        // Security: Validate path is within allowed directories
        validateModelPath(path);

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
        logger.info("Loaded metadata: {}, trained on {}, accuracy={}",
                metadata.getModelId(),
                metadata.getDataset().datasetName,
                String.format("%.3f", metadata.getMetrics().testAccuracy));

        // Validate metadata points to correct model file
        if (metadata.getModelFile() != null &&
            !metadata.getModelFile().equals(path.getFileName().toString())) {
            logger.warn("Metadata model_file mismatch: expected {}, got {}",
                    metadata.getModelFile(), path.getFileName());
        }

        // Load the actual model based on algorithm type
        SentimentClassifier classifier = loadModelByType(path, metadata);

        logger.info("Loaded model: {} from {}", path.getFileName(), path.getParent());

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
     * Result of loading multiple models, including both successes and failures.
     * This allows callers to handle partial failures appropriately.
     */
    public record BatchLoadResult(
            java.util.Map<String, SentimentClassifier> models,
            java.util.Map<String, String> failures
    ) {
        public boolean hasFailures() {
            return !failures.isEmpty();
        }

        public int successCount() {
            return models.size();
        }

        public int failureCount() {
            return failures.size();
        }
    }

    /**
     * Load all models for a given algorithm from models/<algorithm>/ directory.
     *
     * <p>Unlike silently swallowing exceptions, this method returns a {@link BatchLoadResult}
     * that includes both successfully loaded models and any failures with their error messages.
     * This allows callers to:
     * <ul>
     *   <li>Know if any models failed to load</li>
     *   <li>Get specific error messages for debugging</li>
     *   <li>Decide how to handle partial failures</li>
     * </ul>
     *
     * @param algorithm algorithm name (svm, naive_bayes, random_forest)
     * @return BatchLoadResult containing successfully loaded models and any failures
     * @throws IOException if the directory listing itself fails
     */
    public static BatchLoadResult loadAllForAlgorithm(String algorithm)
            throws IOException {

        java.util.Map<String, SentimentClassifier> models = new java.util.HashMap<>();
        java.util.Map<String, String> failures = new java.util.HashMap<>();
        Path algoDir = Paths.get("models", algorithm);

        if (!Files.exists(algoDir)) {
            logger.warn("Model directory not found: {}", algoDir);
            return new BatchLoadResult(models, failures);
        }

        // Find all .ser files
        try (var stream = Files.list(algoDir)) {
            stream.filter(p -> p.toString().endsWith(".ser"))
                  .forEach(modelPath -> {
                      String filename = modelPath.getFileName().toString();
                      String domain = extractDomainName(filename, algorithm);

                      try {
                          SentimentClassifier classifier = loadWithMetadata(modelPath.toString());
                          models.put(domain, classifier);
                          logger.info("Loaded {} model trained on {}", algorithm, domain);

                      } catch (Exception e) {
                          // Capture the failure instead of silently ignoring it
                          String errorMsg = String.format("%s: %s",
                                  e.getClass().getSimpleName(), e.getMessage());
                          failures.put(domain, errorMsg);
                          logger.error("Failed to load {} model from {}: {}",
                                  algorithm, modelPath, errorMsg, e);
                      }
                  });
        }

        // Log summary if there were any failures
        if (!failures.isEmpty()) {
            logger.warn("Loaded {}/{} {} models. Failures: {}",
                    models.size(),
                    models.size() + failures.size(),
                    algorithm,
                    failures.keySet());
        }

        return new BatchLoadResult(models, failures);
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

    /**
     * Validates that a model path is within allowed directories.
     *
     * <p>This prevents path traversal attacks where an attacker might try to load
     * arbitrary files from the filesystem using paths like "../../../etc/passwd"
     * or "/etc/sensitive/data.ser".
     *
     * @param modelPath the path to validate
     * @throws SecurityException if the path is outside allowed directories
     */
    private static void validateModelPath(Path modelPath) {
        // Normalize the path to resolve any ".." or "." components
        Path normalizedPath = modelPath.toAbsolutePath().normalize();
        String pathString = normalizedPath.toString();

        // Check if path contains suspicious traversal patterns
        String originalString = modelPath.toString();
        if (originalString.contains("..") || originalString.contains("//")) {
            logger.warn("Rejected model path with traversal pattern: {}", originalString);
            throw new SecurityException(
                    "Model path contains invalid traversal patterns: " + originalString);
        }

        // Check if the path is within an allowed base directory
        boolean isAllowed = false;
        for (String allowedBase : ALLOWED_BASE_DIRS) {
            Path allowedPath = Paths.get(allowedBase).toAbsolutePath().normalize();
            if (normalizedPath.startsWith(allowedPath)) {
                isAllowed = true;
                break;
            }
            // Also check relative paths (for local development)
            if (pathString.contains("/" + allowedBase + "/") ||
                pathString.contains("\\" + allowedBase + "\\") ||
                originalString.startsWith(allowedBase + "/") ||
                originalString.startsWith(allowedBase + "\\") ||
                originalString.equals(allowedBase)) {
                isAllowed = true;
                break;
            }
        }

        // Special case: allow paths that start with "models/" relative to current dir
        Path currentDir = Paths.get(".").toAbsolutePath().normalize();
        for (String allowedBase : ALLOWED_BASE_DIRS) {
            Path allowedRelative = currentDir.resolve(allowedBase).normalize();
            if (normalizedPath.startsWith(allowedRelative)) {
                isAllowed = true;
                break;
            }
        }

        if (!isAllowed) {
            logger.warn("Rejected model path outside allowed directories: {} (normalized: {})",
                    modelPath, normalizedPath);
            throw new SecurityException(
                    "Model path must be within allowed directories (models/, /app/models/, /tmp/models/). " +
                    "Received: " + modelPath);
        }

        // Validate file extension
        String filename = normalizedPath.getFileName().toString().toLowerCase();
        if (!filename.endsWith(".ser")) {
            logger.warn("Rejected model path with invalid extension: {}", modelPath);
            throw new SecurityException(
                    "Model file must have .ser extension. Received: " + filename);
        }

        logger.debug("Model path validated: {}", normalizedPath);
    }
}
