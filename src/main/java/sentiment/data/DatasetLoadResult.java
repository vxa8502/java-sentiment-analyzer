package sentiment.data;

import java.util.List;
import java.util.Map;

/**
 * Result of loading a single dataset file.
 *
 * Contains the loaded data, metadata about the dataset type,
 * loading time statistics, and analysis of the loaded data.
 *
 * Thread-safe immutable record.
 */
public record DatasetLoadResult(
        List<Dataset> datasets,
        String datasetType,
        String filePath,
        long loadTimeMs,
        DatasetAnalysis analysis
) {
    @Override
    public String toString() {
        return String.format("DatasetLoadResult{type='%s', samples=%d, loadTime=%dms}",
                datasetType, datasets.size(), loadTimeMs);
    }
}
