package sentiment.data;

import java.util.List;

/**
 * Strategy interface for loading different types of datasets
 */
public interface DatasetLoader {

    /**
     * Load dataset from the specified file path
     * @param filePath Path to the dataset file
     * @return List of Dataset objects
     * @throws DataLoadingException if loading fails
     */
    List<Dataset> loadDataset(String filePath) throws DataLoadingException;

    /**
     * Get the supported file extensions for this loader
     * @return Array of supported extensions (e.g., [".csv", ".json"])
     */
    String[] getSupportedExtensions();

    /**
     * Get a human-readable name for this dataset type
     * @return Dataset type name (e.g., "Movie Reviews", "Twitter Data")
     */
    String getDatasetTypeName();

    /**
     * Validate that the file can be processed by this loader
     * @param filePath Path to validate
     * @return true if this loader can handle the file
     */
    default boolean canHandle(String filePath) {
        if (filePath == null || filePath.trim().isEmpty()) {
            return false;
        }

        String [] extensions = getSupportedExtensions();
        if (extensions == null || extensions.length == 0) {
            return false;
        }

        String lowerPath = filePath.toLowerCase();
        for (String ext : extensions) {
            if (lowerPath.endsWith(ext.toLowerCase())) {
                return true;
            }
        }
        return false;
    }
}