package sentiment.evaluation;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sentiment.evaluation.domain.FeatureImportanceResult;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Handles saving and loading feature importance analysis results to/from JSON files.
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
    public static void save(FeatureImportanceResult result, Path outputPath) throws IOException {
        logger.info("Saving feature importance to: {}", outputPath);

        // Ensure parent directory exists
        Files.createDirectories(outputPath.getParent());

        // Write directly - domain types are already serializable
        objectMapper.writeValue(outputPath.toFile(), result);

        logger.info("Feature importance saved successfully ({} bytes)", Files.size(outputPath));
    }

    /**
     * Loads feature importance analysis results from a JSON file.
     *
     * @param inputPath Path to JSON file
     * @return Loaded feature importance data
     * @throws IOException if loading fails
     */
    public static FeatureImportanceResult load(Path inputPath) throws IOException {
        logger.info("Loading feature importance from: {}", inputPath);

        if (!Files.exists(inputPath)) {
            throw new IOException("Feature importance file not found: " + inputPath);
        }

        FeatureImportanceResult result = objectMapper.readValue(
                inputPath.toFile(),
                FeatureImportanceResult.class
        );

        logger.info("Feature importance loaded successfully ({} features)",
                result.topFeatures().size());

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
}
