package sentiment.evaluation;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Handles saving and loading feature importance analysis results to/from JSON files.
 *
 * This enables:
 * 1. Persisting feature importance computed during training
 * 2. Loading pre-computed results at runtime for API serving
 * 3. Notebook-based exploration without recomputation
 *
 * File format: JSON
 * File naming convention: {model-name}-feature-importance.json
 * Example: svm-model-feature-importance.json
 */
public class FeatureImportancePersistence {

    private static final Logger logger = LoggerFactory.getLogger(FeatureImportancePersistence.class);
    private static final ObjectMapper objectMapper = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT);

    /**
     * Saves feature importance analysis results to a JSON file.
     *
     * @param result The analysis result to save
     * @param outputPath Path where JSON file should be saved
     * @throws IOException if saving fails
     */
    public static void save(FeatureImportanceAnalyzer.FeatureImportanceResult result,
                            Path outputPath) throws IOException {
        logger.info("Saving feature importance to: {}", outputPath);

        // Convert to serializable format
        SerializableFeatureImportance serializable = new SerializableFeatureImportance(
                result.topFeatures.stream()
                        .map(fw -> new SerializableFeatureWeight(fw.featureName, fw.weight, fw.significance))
                        .collect(Collectors.toList()),
                new SerializableStatistics(
                        result.statistics.mean,
                        result.statistics.stdDev,
                        result.statistics.median,
                        result.statistics.percentile95,
                        result.statistics.totalFeatures
                ),
                result.analysisTimeMs,
                System.currentTimeMillis()
        );

        // Ensure parent directory exists
        Files.createDirectories(outputPath.getParent());

        // Write to file
        objectMapper.writeValue(outputPath.toFile(), serializable);

        logger.info("Feature importance saved successfully ({} bytes)",
                Files.size(outputPath));
    }

    /**
     * Loads feature importance analysis results from a JSON file.
     *
     * @param inputPath Path to JSON file
     * @return Loaded feature importance data
     * @throws IOException if loading fails
     */
    public static SerializableFeatureImportance load(Path inputPath) throws IOException {
        logger.info("Loading feature importance from: {}", inputPath);

        if (!Files.exists(inputPath)) {
            throw new IOException("Feature importance file not found: " + inputPath);
        }

        SerializableFeatureImportance result = objectMapper.readValue(
                inputPath.toFile(),
                SerializableFeatureImportance.class
        );

        logger.info("Feature importance loaded successfully ({} features)",
                result.topFeatures.size());

        return result;
    }

    /**
     * Checks if a feature importance file exists for the given model path.
     *
     * @param modelPath Path to the model file (e.g., "./models/svm-model.ser")
     * @return Path to the corresponding feature importance file
     */
    public static Path getFeatureImportancePath(Path modelPath) {
        String modelFileName = modelPath.getFileName().toString();
        String baseName = modelFileName.replaceAll("\\.(ser|model)$", "");
        String featureImportanceFileName = baseName + "-feature-importance.json";

        return modelPath.getParent().resolve(featureImportanceFileName);
    }

    // ==================== SERIALIZABLE DATA CLASSES ====================

    /**
     * JSON-serializable representation of feature importance results.
     */
    public static class SerializableFeatureImportance {
        public List<SerializableFeatureWeight> topFeatures;
        public SerializableStatistics statistics;
        public long analysisTimeMs;
        public long timestamp;

        // Default constructor for Jackson
        public SerializableFeatureImportance() {}

        public SerializableFeatureImportance(List<SerializableFeatureWeight> topFeatures,
                                              SerializableStatistics statistics,
                                              long analysisTimeMs,
                                              long timestamp) {
            this.topFeatures = topFeatures;
            this.statistics = statistics;
            this.analysisTimeMs = analysisTimeMs;
            this.timestamp = timestamp;
        }
    }

    /**
     * JSON-serializable representation of a single feature weight.
     */
    public static class SerializableFeatureWeight {
        public String featureName;
        public double weight;
        public double significance;

        // Default constructor for Jackson
        public SerializableFeatureWeight() {}

        public SerializableFeatureWeight(String featureName, double weight, double significance) {
            this.featureName = featureName;
            this.weight = weight;
            this.significance = significance;
        }
    }

    /**
     * JSON-serializable representation of feature importance statistics.
     */
    public static class SerializableStatistics {
        public double mean;
        public double stdDev;
        public double median;
        public double percentile95;
        public int totalFeatures;

        // Default constructor for Jackson
        public SerializableStatistics() {}

        public SerializableStatistics(double mean, double stdDev, double median,
                                       double percentile95, int totalFeatures) {
            this.mean = mean;
            this.stdDev = stdDev;
            this.median = median;
            this.percentile95 = percentile95;
            this.totalFeatures = totalFeatures;
        }
    }
}
