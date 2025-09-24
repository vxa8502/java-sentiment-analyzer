package sentiment.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.List;

public abstract class BaseDatasetLoader implements DatasetLoader {
    protected final Logger logger = LoggerFactory.getLogger(getClass());

    @Override
    public final List<Dataset> loadDataset(String filePath) throws DataLoadingException {
        logger.info("Loading {} from: {}", getDatasetTypeName(), filePath);

        // Validate file - using utility directly
        DatasetValidationUtils.validateFilePath(filePath, getDatasetTypeName());

        // Create statistics tracker
        DatasetLoadingStats stats = new DatasetLoadingStats();

        // Perform actual loading (implemented by subclasses)
        List<Dataset> datasets = doLoadDataset(filePath, stats);

        // Validate final result - using utility directly
        DatasetValidationUtils.validateFinalDataset(datasets, stats, filePath, getDatasetTypeName());

        // Log layered summary and analyze
        stats.logLayeredSummary(logger, filePath, getDatasetTypeName());
        DatasetValidationUtils.analyzeSentimentDistribution(datasets, getDatasetTypeName());

        return datasets;
    }

    // Single abstract method for subclasses to implement
    protected abstract List<Dataset> doLoadDataset(String filePath, DatasetLoadingStats stats)
            throws DataLoadingException;
}