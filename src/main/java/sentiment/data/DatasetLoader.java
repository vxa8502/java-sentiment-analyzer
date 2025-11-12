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
     * Test compatibility of this loader with the given file by sampling the first N rows.
     * Returns error rate and diagnostic information to help determine the best loader.
     *
     * @param filePath Path to the dataset file
     * @param sampleSize Number of rows to sample (default: 100)
     * @return CompatibilityTestResult with error rate and diagnostics
     */
    CompatibilityTestResult testCompatibility(String filePath, int sampleSize);

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

    /**
     * Result of compatibility testing - contains error rate and diagnostic information
     */
    record CompatibilityTestResult(
            String loaderName,
            double errorRate,
            int sampledRows,
            int successfulRows,
            int failedRows,
            String failureReason,
            boolean canParse
    ) {
        /**
         * Create a successful compatibility test result
         */
        public static CompatibilityTestResult success(String loaderName, int sampledRows, int successfulRows, int failedRows) {
            double errorRate = sampledRows > 0 ? (double) failedRows / sampledRows : 1.0;
            return new CompatibilityTestResult(loaderName, errorRate, sampledRows, successfulRows, failedRows, null, true);
        }

        /**
         * Create a failed compatibility test result
         */
        public static CompatibilityTestResult failure(String loaderName, String failureReason) {
            return new CompatibilityTestResult(loaderName, 1.0, 0, 0, 0, failureReason, false);
        }

        /**
         * Check if this result indicates high compatibility (low error rate)
         */
        public boolean isHighlyCompatible() {
            return canParse && errorRate <= 0.10; // 10% or less errors
        }

        /**
         * Check if this result is better than another result
         */
        public boolean isBetterThan(CompatibilityTestResult other) {
            if (other == null) return true;
            if (!canParse && !other.canParse) return false;
            if (!canParse) return false;
            if (!other.canParse) return true;
            return this.errorRate < other.errorRate;
        }

        @Override
        public String toString() {
            if (!canParse) {
                return String.format("%s: FAILED (%s)", loaderName, failureReason);
            }
            return String.format("%s: %.1f%% error rate (%d/%d successful)",
                    loaderName, errorRate * 100, successfulRows, sampledRows);
        }
    }
}