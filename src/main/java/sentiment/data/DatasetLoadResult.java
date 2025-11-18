package sentiment.data;

import java.util.List;

/**
 * Result of loading a single dataset file.
 *
 * Contains the loaded data, metadata about the dataset type,
 * and loading time statistics.
 *
 * Thread-safe immutable record.
 */
public record DatasetLoadResult(
        List<Dataset> datasets,
        String datasetType,
        String filePath,
        long loadTimeMs
) {
    @Override
    public String toString() {
        return String.format("DatasetLoadResult{type='%s', samples=%d, loadTime=%dms}",
                datasetType, datasets.size(), loadTimeMs);
    }
}
