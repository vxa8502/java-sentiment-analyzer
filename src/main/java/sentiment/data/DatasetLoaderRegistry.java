package sentiment.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.beans.factory.annotation.Autowired;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Enhanced dataset loading manager with memory protection.
 * Uses extension-based routing with enforced memory limits to prevent OOM errors.
 *
 * MEMORY PROTECTION:
 * - Per-file dataset limits (default: 100k datasets)
 * - Combined dataset limits (default: 500k datasets)
 * - File size checks (default: 500MB max)
 * - Configurable via application.properties
 */
@Component
public class DatasetLoaderRegistry {

    private static final Logger logger = LoggerFactory.getLogger(DatasetLoaderRegistry.class);

    private final Map<String, DatasetLoader> loaderRegistry;
    private final List<DatasetLoader> availableLoaders;
    private final DataLoadingLimits limits;  // ✅ Memory protection

    @Autowired
    public DatasetLoaderRegistry(DataLoadingLimits limits) {
        this.limits = limits;

        // Initialize all available loaders
        this.availableLoaders = Arrays.asList(
                new AmazonPolarityLoader(),
                new MovieReviewsLoader(),
                new ProductReviewsLoader(),
                new ServiceReviewsLoader()
        );

        // Build registry for extension-based lookup
        this.loaderRegistry = new HashMap<>();
        for (DatasetLoader loader : availableLoaders) {
            for (String extension : loader.getSupportedExtensions()) {
                loaderRegistry.put(extension.toLowerCase(), loader);
            }
        }

        logger.info("Initialized DatasetLoaderRegistry with {} loaders supporting {} file types: {}",
                availableLoaders.size(),
                loaderRegistry.size(),
                loaderRegistry.keySet());

        // Log memory limit configuration
        limits.logConfiguration();
    }

    /**
     * Load dataset using intelligent detection (filename + header analysis).
     * Enforces per-file memory limits.
     *
     * @param filePath Path to dataset file
     * @return DatasetLoadResult with loaded data and metadata
     * @throws DataLoadingException if file invalid or memory limits exceeded
     */
    public DatasetLoadResult loadDataset(String filePath) throws DataLoadingException {
        logger.info("Loading dataset from: {}", filePath);

        // Step 1: Validate file exists and size is acceptable
        validateFilePathWithLimits(filePath);

        // Step 2: Find best loader using intelligent detection
        DatasetLoader loader = findBestLoader(filePath);
        if (loader == null) {
            String extension = getFileExtension(filePath);
            throw new DataLoadingException(
                    "No loader found for file: " + filePath +
                            " (extension: " + extension + "). Supported extensions: " + getSupportedExtensions(),
                    filePath, "Unknown");
        }

        logger.info("Using {} for file: {}", loader.getDatasetTypeName(), filePath);

        // Step 3: Load the data with memory monitoring
        long startTime = System.currentTimeMillis();
        List<Dataset> datasets = loader.loadDataset(filePath);
        long loadTime = System.currentTimeMillis() - startTime;

        // Step 4: Analyze dataset to get class distribution
        DatasetAnalysis analysis = analyzeDataset(datasets);
        int numClasses = analysis.sentimentDistribution().size();

        // Step 5: STATISTICAL VALIDATION (Aria's requirement)
        // Validate sample size BEFORE enforcing memory limits
        DataLoadingLimits.SampleSizeValidation validation = limits.validateSampleSize(
                datasets.size(),
                numClasses,
                limits.getEstimatedFeatures()
        );

        // Log validation warnings but don't fail (allows user to proceed with understanding)
        if (validation.hasWarnings()) {
            logger.warn("=== Statistical Validation Warnings for {} ===", getFileName(filePath));
            for (String warning : validation.getWarnings()) {
                logger.warn("  - {}", warning);
            }
            logger.warn("==================================================");
        }

        // Step 6: ENFORCE per-file limit after loading
        limits.checkPerFileLimit(datasets.size(), filePath);

        // Step 7: Create result with analysis
        DatasetLoadResult result = new DatasetLoadResult(
                datasets,
                loader.getDatasetTypeName(),
                filePath,
                loadTime,
                analysis
        );

        logger.info("Successfully loaded {} samples from {} in {}ms (memory: ~{} MB)",
                datasets.size(), loader.getDatasetTypeName(), loadTime,
                limits.estimateMemoryUsageMb(datasets.size()));

        return result;
    }

    /**
     * Load multiple datasets and combine them with STRICT memory limits.
     *
     * MEMORY PROTECTION:
     * - Checks combined total against maxCombinedDatasets limit
     * - Warns when approaching limits
     * - Throws exception if strict enforcement enabled
     *
     * @param filePaths Paths to dataset files
     * @return CombinedDatasetResult with all loaded data
     * @throws DataLoadingException if memory limits exceeded
     */
    public CombinedDatasetResult loadMultipleDatasets(String... filePaths) throws DataLoadingException {
        logger.info("Loading {} datasets for combination with memory protection", filePaths.length);

        List<DatasetLoadResult> individualResults = new ArrayList<>();
        List<Dataset> allDatasets = new ArrayList<>();
        Map<String, Integer> datasetCounts = new HashMap<>();

        // Track running total for memory limit checks
        int runningTotal = 0;

        for (int i = 0; i < filePaths.length; i++) {
            String filePath = filePaths[i];

            logger.info("Loading file {}/{}: {}", i + 1, filePaths.length, filePath);

            // Load individual file (already enforces per-file limit)
            DatasetLoadResult result = loadDataset(filePath);
            individualResults.add(result);

            // Check if adding this file would exceed combined limit
            int newTotal = runningTotal + result.datasets().size();

            // ✅ MEMORY PROTECTION: Check combined limit BEFORE adding
            try {
                limits.checkCombinedLimit(newTotal, i + 1);
            } catch (DataLoadingException e) {
                logger.error("Combined dataset limit exceeded after loading {} files. " +
                                "Loaded {} datasets, attempted to add {} more",
                        i + 1, runningTotal, result.datasets().size());
                throw e;
            }

            // Safe to add - within limits
            allDatasets.addAll(result.datasets());
            runningTotal = newTotal;

            // Track counts by dataset type
            datasetCounts.merge(result.datasetType(), result.datasets().size(), Integer::sum);

            logger.info("Progress: {}/{} files loaded, {} total datasets ({:.1f}% of limit)",
                    i + 1, filePaths.length, runningTotal,
                    (double) runningTotal / limits.getMaxCombinedDatasets() * 100);
        }

        // Statistical validation for combined dataset
        DatasetAnalysis combinedAnalysis = analyzeDataset(allDatasets);
        int numClasses = combinedAnalysis.sentimentDistribution().size();

        DataLoadingLimits.SampleSizeValidation validation = limits.validateSampleSize(
                allDatasets.size(),
                numClasses,
                limits.getEstimatedFeatures()
        );

        if (validation.hasWarnings()) {
            logger.warn("=== Statistical Validation Warnings for Combined Dataset ===");
            for (String warning : validation.getWarnings()) {
                logger.warn("  - {}", warning);
            }
            logger.warn("===========================================================");
        }

        logger.info("Combined {} datasets: {} total samples with distribution: {} (memory: ~{} MB)",
                filePaths.length, allDatasets.size(), datasetCounts,
                limits.estimateMemoryUsageMb(allDatasets.size()));

        return new CombinedDatasetResult(allDatasets, individualResults, datasetCounts);
    }

    /**
     * Enhanced file validation with memory limit checks.
     *
     * @param filePath Path to validate
     * @throws DataLoadingException if file invalid or too large
     */
    private void validateFilePathWithLimits(String filePath) throws DataLoadingException {
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
            long fileSizeBytes = Files.size(path);
            long fileSizeMb = fileSizeBytes / (1024 * 1024);

            if (fileSizeBytes == 0) {
                throw new DataLoadingException("File is empty", filePath, "Unknown");
            }

            // ✅ MEMORY PROTECTION: Check file size before loading
            limits.checkFileSize(fileSizeMb, filePath);

            logger.info("File validated: {} ({} MB)", getFileName(filePath), fileSizeMb);

        } catch (DataLoadingException e) {
            throw e; // Re-throw our own exceptions
        } catch (Exception e) {
            logger.debug("Could not check file size: {}", e.getMessage());
        }
    }

    /**
     * Find the best loader by testing all compatible loaders and selecting the one
     * with the lowest error rate. This replaces unreliable filename-based detection.
     *
     * @param filePath Path to the dataset file
     * @return Best loader based on compatibility testing, or null if none are compatible
     * @throws DataLoadingException if best match still has >10% error rate
     */
    private DatasetLoader findBestLoader(String filePath) throws DataLoadingException {
        // Step 1: Check if extension is supported at all
        String extension = getFileExtension(filePath).toLowerCase();
        if (!loaderRegistry.containsKey(extension)) {
            return null; // Unsupported file type
        }

        // Step 2: Get all loaders that support this file extension
        List<DatasetLoader> candidateLoaders = availableLoaders.stream()
                .filter(loader -> loader.canHandle(filePath))
                .toList();

        if (candidateLoaders.isEmpty()) {
            return null;
        }

        // Step 3: Test compatibility with all candidate loaders
        logger.info("Testing {} candidate loaders for file: {}", candidateLoaders.size(), getFileName(filePath));

        DatasetLoader.CompatibilityTestResult bestResult = null;
        DatasetLoader bestLoader = null;
        int sampleSize = 100; // Test first 100 rows

        for (DatasetLoader loader : candidateLoaders) {
            logger.debug("Testing compatibility with {}", loader.getDatasetTypeName());

            DatasetLoader.CompatibilityTestResult result = loader.testCompatibility(filePath, sampleSize);

            logger.info("  {}", result);

            if (result.isBetterThan(bestResult)) {
                bestResult = result;
                bestLoader = loader;
            }
        }

        // Step 4: Validate the best result meets our confidence threshold
        if (bestResult == null || !bestResult.canParse()) {
            String supportedTypes = candidateLoaders.stream()
                    .map(DatasetLoader::getDatasetTypeName)
                    .collect(Collectors.joining(", "));

            throw new DataLoadingException(
                    String.format("No loader could parse file successfully. Tried: %s", supportedTypes),
                    filePath,
                    "Unknown"
            );
        }

        // Step 5: Check if error rate is acceptable (≤10%)
        final double CONFIDENCE_THRESHOLD = 0.10; // 10% max error rate
        if (bestResult.errorRate() > CONFIDENCE_THRESHOLD) {
            throw new DataLoadingException(
                    String.format(
                            "Best matching loader '%s' has error rate of %.1f%%, which exceeds acceptable threshold of %.1f%%. " +
                            "This file may be corrupted, in an unexpected format, or may require a custom loader. " +
                            "Sample results: %d/%d rows parsed successfully.",
                            bestResult.loaderName(),
                            bestResult.errorRate() * 100,
                            CONFIDENCE_THRESHOLD * 100,
                            bestResult.successfulRows(),
                            bestResult.sampledRows()
                    ),
                    filePath,
                    bestResult.loaderName()
            );
        }

        logger.info("Selected {} with {:.1f}% error rate ({}/{} successful)",
                bestResult.loaderName(),
                bestResult.errorRate() * 100,
                bestResult.successfulRows(),
                bestResult.sampledRows()
        );

        return bestLoader;
    }

    /**
     * Get just filename from full path.
     */
    private String getFileName(String filePath) {
        int lastSeparator = Math.max(filePath.lastIndexOf('/'), filePath.lastIndexOf('\\'));
        if (lastSeparator >= 0 && lastSeparator < filePath.length() - 1) {
            return filePath.substring(lastSeparator + 1);
        }
        return filePath;
    }

    /**
     * Extract file extension from path.
     */
    private String getFileExtension(String filePath) {
        int lastDot = filePath.lastIndexOf('.');
        return lastDot > 0 ? filePath.substring(lastDot) : "";
    }

    /**
     * Analyze loaded dataset for basic statistics.
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

    // ==================== UTILITY METHODS ====================

    /**
     * Get all supported file extensions.
     */
    public List<String> getSupportedExtensions() {
        return new ArrayList<>(loaderRegistry.keySet());
    }

    /**
     * Get all available dataset type names.
     */
    public List<String> getAvailableDatasetTypes() {
        return availableLoaders.stream()
                .map(DatasetLoader::getDatasetTypeName)
                .collect(Collectors.toList());
    }

    /**
     * Check if a file can be handled by any loader with acceptable error rate.
     * Returns false if no loader can handle the file or if error rate exceeds threshold.
     */
    public boolean canHandleFile(String filePath) {
        try {
            return findBestLoader(filePath) != null;
        } catch (DataLoadingException e) {
            logger.debug("File cannot be handled: {}", e.getMessage());
            return false;
        }
    }

    /**
     * Get the loader that would be used for a given file path.
     * @throws DataLoadingException if no suitable loader found or error rate too high
     */
    public DatasetLoader getLoaderForFile(String filePath) throws DataLoadingException {
        return findBestLoader(filePath);
    }

    /**
     * Get current memory limit configuration.
     */
    public DataLoadingLimits getLimits() {
        return limits;
    }

    /**
     * Get comprehensive loading statistics.
     */
    public LoadingCapabilityInfo getCapabilityInfo() {
        return new LoadingCapabilityInfo(
                availableLoaders.size(),
                loaderRegistry.keySet(),
                limits,
                limits.estimateMemoryUsageMb(limits.getMaxCombinedDatasets())
        );
    }
}

// ==================== RESULT RECORDS ====================

record CombinedDatasetResult(
        List<Dataset> allDatasets,
        List<DatasetLoadResult> individualResults,
        Map<String, Integer> datasetTypeCounts
) {
    public int getTotalSamples() {
        return allDatasets.size();
    }

    @Override
    public String toString() {
        return String.format("CombinedDatasetResult{totalSamples=%d, types=%s}",
                allDatasets.size(), datasetTypeCounts);
    }
}

record DatasetAnalysis(
        int totalSamples,
        Map<Dataset.SentimentLabel, Long> sentimentDistribution,
        int avgTextLength,
        int minTextLength,
        int maxTextLength
) {
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

record LoadingCapabilityInfo(
        int loaderCount,
        Set<String> supportedExtensions,
        DataLoadingLimits limits,
        long maxMemoryEstimateMb
) {
    @Override
    public String toString() {
        return String.format(
                "LoadingCapability{loaders=%d, extensions=%s, limits=%s, maxMemory=~%dMB}",
                loaderCount, supportedExtensions, limits, maxMemoryEstimateMb
        );
    }
}