package sentiment.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.List;

public abstract class BaseDatasetLoader implements DatasetLoader {
    protected final Logger logger = LoggerFactory.getLogger(getClass());

    @Override
    public final List<Dataset> loadDataset(String filePath) throws DataLoadingException {
        logger.info("Loading {} from: {}", getDatasetTypeName(), filePath);

        // Validate file can be read
        DatasetValidationUtils.validateFilePath(filePath, getDatasetTypeName());

        // Create statistics tracker
        DatasetLoadingStats stats = new DatasetLoadingStats();

        // Perform actual loading (implemented by subclasses)
        List<Dataset> datasets = doLoadDataset(filePath, stats);

        // Validate final result
        DatasetValidationUtils.validateFinalDataset(datasets, stats, filePath, getDatasetTypeName());

        // Log summary
        stats.logSummary(logger, filePath, getDatasetTypeName());

        return datasets;
    }

    @Override
    public CompatibilityTestResult testCompatibility(String filePath, int sampleSize) {
        // Validate basic file access
        try {
            DatasetValidationUtils.validateFilePath(filePath, getDatasetTypeName());
        } catch (DataLoadingException e) {
            return CompatibilityTestResult.failure(getDatasetTypeName(),
                "Cannot access file: " + e.getMessage());
        }

        // Check file extension
        if (!canHandle(filePath)) {
            return CompatibilityTestResult.failure(getDatasetTypeName(),
                "Unsupported file extension");
        }

        // Create statistics tracker for sampling
        DatasetLoadingStats stats = new DatasetLoadingStats();

        try {
            // Attempt to load a sample of the file
            List<Dataset> sample = doLoadDatasetSample(filePath, sampleSize, stats);

            // Calculate results
            int sampledRows = stats.getTotalRecords();
            int successfulRows = stats.getSuccessfulRecords();
            int failedRows = sampledRows - successfulRows;

            // If we couldn't parse anything, it's a complete failure
            if (sampledRows == 0) {
                return CompatibilityTestResult.failure(getDatasetTypeName(),
                    "Could not parse any records from file");
            }

            return CompatibilityTestResult.success(
                getDatasetTypeName(),
                sampledRows,
                successfulRows,
                failedRows
            );

        } catch (Exception e) {
            return CompatibilityTestResult.failure(getDatasetTypeName(),
                "Error during sampling: " + e.getMessage());
        }
    }

    /**
     * Load a sample of the dataset for compatibility testing.
     * Default implementation delegates to doLoadDataset, but subclasses can override
     * for more efficient sampling.
     *
     * @param filePath Path to the dataset file
     * @param sampleSize Number of rows to sample
     * @param stats Statistics tracker
     * @return List of sampled Dataset objects
     * @throws DataLoadingException if sampling fails
     */
    protected List<Dataset> doLoadDatasetSample(String filePath, int sampleSize,
                                                 DatasetLoadingStats stats) throws DataLoadingException {
        // Default: delegate to regular load method
        // Subclasses (especially CSV loaders) can override for more efficient sampling
        return doLoadDataset(filePath, stats);
    }

    // Single abstract method for subclasses to implement
    protected abstract List<Dataset> doLoadDataset(String filePath, DatasetLoadingStats stats)
            throws DataLoadingException;
}