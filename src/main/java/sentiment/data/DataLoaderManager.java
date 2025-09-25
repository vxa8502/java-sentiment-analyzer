package sentiment.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Simplified dataset loading manager using extension-based routing only.
 * No complex content detection - just maps file extensions to loaders.
 */
@Component
public class DataLoaderManager {

    private static final Logger logger = LoggerFactory.getLogger(DataLoaderManager.class);

    private final Map<String, DatasetLoader> loaderRegistry;
    private final List<DatasetLoader> availableLoaders;

    public DataLoaderManager() {
        // Initialize all available loaders
        this.availableLoaders = Arrays.asList(
                new MovieReviewsLoader(),
                new ProductReviewsLoader(),
                new TwitterDataLoader()
        );

        // Build registry for extension-based lookup
        this.loaderRegistry = new HashMap<>();
        for (DatasetLoader loader : availableLoaders) {
            for (String extension : loader.getSupportedExtensions()) {
                loaderRegistry.put(extension.toLowerCase(), loader);
            }
        }

        logger.info("Initialized DataLoaderManager with {} loaders supporting {} file types: {}",
                availableLoaders.size(),
                loaderRegistry.size(),
                loaderRegistry.keySet());
    }

    /**
     * Load dataset using intelligent detection (filename + header analysis)
     */
    public DatasetLoadResult loadDataset(String filePath) throws DataLoadingException {
        logger.info("Loading dataset from: {}", filePath);

        // Validate file exists
        validateFilePath(filePath);

        // Find best loader using intelligent detection
        DatasetLoader loader = findBestLoader(filePath);
        if (loader == null) {
            String extension = getFileExtension(filePath);
            throw new DataLoadingException(
                    "No loader found for file: " + filePath +
                            " (extension: " + extension + "). Supported extensions: " + getSupportedExtensions(),
                    filePath, "Unknown");
        }

        logger.info("Using {} for file: {}", loader.getDatasetTypeName(), filePath);

        // Load the data
        long startTime = System.currentTimeMillis();
        List<Dataset> datasets = loader.loadDataset(filePath);
        long loadTime = System.currentTimeMillis() - startTime;

        // Create result with basic analysis
        DatasetLoadResult result = new DatasetLoadResult(
                datasets,
                loader.getDatasetTypeName(),
                filePath,
                loadTime,
                analyzeDataset(datasets)
        );

        logger.info("Successfully loaded {} samples from {} in {}ms",
                datasets.size(), loader.getDatasetTypeName(), loadTime);

        return result;
    }

    /**
     * Load multiple datasets and combine them
     */
    public CombinedDatasetResult loadMultipleDatasets(String... filePaths) throws DataLoadingException {
        logger.info("Loading {} datasets for combination", filePaths.length);

        List<DatasetLoadResult> individualResults = new ArrayList<>();
        List<Dataset> allDatasets = new ArrayList<>();
        Map<String, Integer> datasetCounts = new HashMap<>();

        for (String filePath : filePaths) {
            DatasetLoadResult result = loadDataset(filePath);
            individualResults.add(result);
            allDatasets.addAll(result.datasets());

            // Track counts by dataset type
            datasetCounts.merge(result.datasetType(), result.datasets().size(), Integer::sum);
        }

        logger.info("Combined {} datasets: {} total samples with distribution: {}",
                filePaths.length, allDatasets.size(), datasetCounts);

        return new CombinedDatasetResult(allDatasets, individualResults, datasetCounts);
    }

    /**
     * Find the best loader using simple but effective detection
     */
    private DatasetLoader findBestLoader(String filePath) {
        // Step 1: Check if extension is supported at all
        String extension = getFileExtension(filePath).toLowerCase();
        if (!loaderRegistry.containsKey(extension)) {
            return null; // Unsupported file type
        }

        // Step 2: Use filename-based hints for quick detection
        String filename = getFileName(filePath).toLowerCase();

        if (filename.contains("movie") || filename.contains("imdb") || filename.contains("film")) {
            DatasetLoader movieLoader = findLoaderByType("Movie Reviews");
            if (movieLoader != null && movieLoader.canHandle(filePath)) {
                return movieLoader;
            }
        }

        if (filename.contains("twitter") || filename.contains("tweet") || filename.contains("social")) {
            DatasetLoader twitterLoader = findLoaderByType("Twitter Data");
            if (twitterLoader != null && twitterLoader.canHandle(filePath)) {
                return twitterLoader;
            }
        }

        if (filename.contains("product") || filename.contains("amazon") || filename.contains("review")) {
            DatasetLoader productLoader = findLoaderByType("Product Reviews");
            if (productLoader != null && productLoader.canHandle(filePath)) {
                return productLoader;
            }
        }

        // Step 3: Try header-based detection for CSV files
        if (extension.equals(".csv") || extension.equals(".tsv")) {
            DatasetLoader headerDetected = detectByHeaders(filePath);
            if (headerDetected != null) {
                return headerDetected;
            }
        }

        // Step 4: Fallback to first loader that can handle this extension
        return loaderRegistry.get(extension);
    }

    /**
     * Simple header-based detection for CSV files
     */
    private DatasetLoader detectByHeaders(String filePath) {
        try {
            String headerLine = readFirstLine(filePath);
            if (headerLine == null || headerLine.trim().isEmpty()) {
                return null;
            }

            String headers = headerLine.toLowerCase();

            // Movie review patterns
            if (headers.contains("movie") || headers.contains("film") || headers.contains("imdb")) {
                return findLoaderByType("Movie Reviews");
            }

            // Twitter patterns
            if (headers.contains("tweet") || headers.contains("twitter") || headers.contains("user")) {
                return findLoaderByType("Twitter Data");
            }

            // Product review patterns
            if (headers.contains("product") || headers.contains("amazon") ||
                    headers.contains("asin") || headers.contains("verified")) {
                return findLoaderByType("Product Reviews");
            }

        } catch (Exception e) {
            logger.debug("Failed to read headers from {}: {}", filePath, e.getMessage());
        }

        return null;
    }

    /**
     * Read just the first line of a file for header detection
     */
    private String readFirstLine(String filePath) {
        try (var reader = Files.newBufferedReader(Paths.get(filePath))) {
            return reader.readLine();
        } catch (Exception e) {
            logger.debug("Could not read first line from {}: {}", filePath, e.getMessage());
            return null;
        }
    }

    /**
     * Find loader by dataset type name
     */
    private DatasetLoader findLoaderByType(String datasetType) {
        return availableLoaders.stream()
                .filter(loader -> loader.getDatasetTypeName().equals(datasetType))
                .findFirst()
                .orElse(null);
    }

    /**
     * Get just filename from full path
     */
    private String getFileName(String filePath) {
        int lastSeparator = Math.max(filePath.lastIndexOf('/'), filePath.lastIndexOf('\\'));
        if (lastSeparator >= 0 && lastSeparator < filePath.length() - 1) {
            return filePath.substring(lastSeparator + 1);
        }
        return filePath;
    }

    /**
     * Extract file extension from path
     */
    private String getFileExtension(String filePath) {
        int lastDot = filePath.lastIndexOf('.');
        return lastDot > 0 ? filePath.substring(lastDot) : "";
    }

    /**
     * Basic file validation
     */
    private void validateFilePath(String filePath) throws DataLoadingException {
        if (filePath == null || filePath.trim().isEmpty()) {
            throw new DataLoadingException("File path cannot be null or empty", filePath, "Unknown");
        }

        Path path = Paths.get(filePath);
        if (!Files.exists(path)) {
            throw new DataLoadingException("File does not exist", filePath, "Unknown");
        }

        if (!Files.isReadable(path)) {
            throw new DataLoadingException("File is not readable", filePath, "Unknown");
        }

        try {
            long fileSize = Files.size(path);
            if (fileSize > 100 * 1024 * 1024) { // 100MB
                logger.warn("Large file detected: {} MB - loading may take time",
                        fileSize / (1024 * 1024));
            }
            if (fileSize == 0) {
                throw new DataLoadingException("File is empty", filePath, "Unknown");
            }
        } catch (DataLoadingException e) {
            throw e; // Re-throw our own exceptions
        } catch (Exception e) {
            logger.debug("Could not check file size: {}", e.getMessage());
        }
    }

    /**
     * Analyze loaded dataset for basic statistics
     */
    private DatasetAnalysis analyzeDataset(List<Dataset> datasets) {
        if (datasets.isEmpty()) {
            return new DatasetAnalysis(0, Map.of(), 0, 0, 0);
        }

        // Count sentiment distribution
        Map<Dataset.SentimentLabel, Long> sentimentCounts = datasets.stream()
                .collect(Collectors.groupingBy(Dataset::getSentiment, Collectors.counting()));

        // Calculate text length statistics
        IntSummaryStatistics lengthStats = datasets.stream()
                .mapToInt(Dataset::getTextLength)
                .summaryStatistics();

        return new DatasetAnalysis(
                datasets.size(),
                sentimentCounts,
                (int) lengthStats.getAverage(),
                lengthStats.getMin(),
                lengthStats.getMax()
        );
    }

    // Utility Methods

    /**
     * Get all supported file extensions
     */
    public List<String> getSupportedExtensions() {
        return new ArrayList<>(loaderRegistry.keySet());
    }

    /**
     * Get all available dataset type names
     */
    public List<String> getAvailableDatasetTypes() {
        return availableLoaders.stream()
                .map(DatasetLoader::getDatasetTypeName)
                .collect(Collectors.toList());
    }

    /**
     * Check if a file can be handled by any loader
     */
    public boolean canHandleFile(String filePath) {
        return findBestLoader(filePath) != null;
    }

    /**
     * Get the loader that would be used for a given file path
     */
    public DatasetLoader getLoaderForFile(String filePath) {
        return findBestLoader(filePath);
    }
}

// Result Records

record DatasetLoadResult(List<Dataset> datasets, String datasetType, String filePath, long loadTimeMs,
                         DatasetAnalysis analysis) {
    @Override
    public String toString() {
        return String.format("DatasetLoadResult{type='%s', samples=%d, loadTime=%dms}",
                datasetType, datasets.size(), loadTimeMs);
    }
}

record CombinedDatasetResult(List<Dataset> allDatasets, List<DatasetLoadResult> individualResults,
                             Map<String, Integer> datasetTypeCounts) {
    public int getTotalSamples() {
        return allDatasets.size();
    }

    @Override
    public String toString() {
        return String.format("CombinedDatasetResult{totalSamples=%d, types=%s}",
                allDatasets.size(), datasetTypeCounts);
    }
}

record DatasetAnalysis(int totalSamples, Map<Dataset.SentimentLabel, Long> sentimentDistribution, int avgTextLength,
                       int minTextLength, int maxTextLength) {
    public boolean isBalanced() {
        if (sentimentDistribution.size() < 2) return true;

        long max = sentimentDistribution.values().stream().mapToLong(Long::longValue).max().orElse(0);
        long min = sentimentDistribution.values().stream().mapToLong(Long::longValue).min().orElse(0);

        return (double) min / max >= 0.7; // Consider balanced if ratio >= 0.7
    }

    @Override
    public String toString() {
        return String.format("Analysis{samples=%d, sentiments=%s, avgLength=%d, balanced=%s}",
                totalSamples, sentimentDistribution, avgTextLength, isBalanced());
    }
}