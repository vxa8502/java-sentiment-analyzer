package sentiment.data;

import java.util.List;

/**
 * Strategy interface for loading sentiment analysis datasets.
 * <p>
 * Implementations provide format-specific loading logic and compatibility testing
 * to automatically select the appropriate loader for a given file.
 */
public interface DatasetLoader {

    /**
     * Loads the dataset from the specified file.
     *
     * @param filePath the path to the dataset file
     * @return the loaded dataset entries
     * @throws DataLoadingException if the file cannot be loaded or parsed
     */
    List<Dataset> loadDataset(String filePath) throws DataLoadingException;

    /**
     * Tests loader compatibility by sampling rows from the file.
     *
     * @param filePath the path to the dataset file
     * @param sampleSize the number of rows to sample
     * @return test result containing error rate and diagnostics
     */
    CompatibilityTestResult testCompatibility(String filePath, int sampleSize);

    /**
     * Returns the file extensions supported by this loader.
     *
     * @return array of supported extensions (e.g., <code>".csv"</code>, <code>".json"</code>)
     */
    String[] getSupportedExtensions();

    /**
     * Returns a human-readable name for this dataset type.
     *
     * @return the dataset type name (e.g., <code>"Movie Reviews"</code>, <code>"Twitter Data"</code>)
     */
    String getDatasetTypeName();

    /**
     * Validates whether this loader can process the specified file.
     *
     * @param filePath the file path to validate
     * @return {@code true} if this loader supports the file extension
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
     * Compatibility test result containing error rate and diagnostic information.
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
         * Creates a successful test result.
         *
         * @param loaderName the name of the loader
         * @param sampledRows the total number of rows sampled
         * @param successfulRows the number of successfully parsed rows
         * @param failedRows the number of failed rows
         * @return a successful compatibility test result
         */
        public static CompatibilityTestResult success(String loaderName, int sampledRows, int successfulRows, int failedRows) {
            double errorRate = sampledRows > 0 ? (double) failedRows / sampledRows : 1.0;
            return new CompatibilityTestResult(loaderName, errorRate, sampledRows, successfulRows, failedRows, null, true);
        }

        /**
         * Creates a failed test result.
         *
         * @param loaderName the name of the loader
         * @param failureReason the reason for failure
         * @return a failed compatibility test result
         */
        public static CompatibilityTestResult failure(String loaderName, String failureReason) {
            return new CompatibilityTestResult(loaderName, 1.0, 0, 0, 0, failureReason, false);
        }

        /**
         * Determines if the error rate indicates high compatibility.
         *
         * @return {@code true} if error rate is 10% or fewer
         */
        @SuppressWarnings("unused")
        public boolean isHighlyCompatible() {
            return canParse && errorRate <= 0.10; // 10% or fewer errors
        }

        /**
         * Compares this result to another for better compatibility.
         *
         * @param other the result to compare against
         * @return {@code true} if this result has lower error rate
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