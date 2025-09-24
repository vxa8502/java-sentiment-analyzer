package sentiment.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Enhanced dataset loading manager with intelligent content-based detection.
 */
@Component
public class DataLoaderManager {

    private static final Logger logger = LoggerFactory.getLogger(DataLoaderManager.class);

    private final Map<String, DatasetLoader> loaderRegistry;
    private final List<DatasetLoader> availableLoaders;
    private final DatasetTypeDetector typeDetector;

    public DataLoaderManager() {
        // Initialize all available loaders - ORDER MATTERS for fallback!
        this.availableLoaders = Arrays.asList(
                new MovieReviewsLoader(),      // Most specific first
                new ProductReviewsLoader(),    // Then product reviews
                new TwitterDataLoader()        // Most generic last
        );

        // Build registry for quick lookup by file extension
        this.loaderRegistry = new HashMap<>();
        for (DatasetLoader loader : availableLoaders) {
            for (String extension : loader.getSupportedExtensions()) {
                loaderRegistry.put(extension.toLowerCase(), loader);
            }
        }

        this.typeDetector = new DatasetTypeDetector();

        logger.info("Initialized DataLoaderManager with {} loaders supporting {} file types",
                availableLoaders.size(), loaderRegistry.size());
    }

    /**
     * Smart load: automatically detect dataset type and load appropriately
     */
    public DatasetLoadResult loadDataset(String filePath) throws DataLoadingException {
        logger.info("Loading dataset from: {}", filePath);

        // Validate file exists
        validateFilePath(filePath);

        // Find appropriate loader using enhanced detection
        DatasetLoader loader = findBestLoaderWithDetection(filePath);
        if (loader == null) {
            throw new DataLoadingException("No loader found for file type", filePath, "Unknown");
        }

        logger.info("Using {} for file: {}", loader.getDatasetTypeName(), filePath);

        // Load the data
        long startTime = System.currentTimeMillis();
        List<Dataset> datasets = loader.loadDataset(filePath);
        long loadTime = System.currentTimeMillis() - startTime;

        // Create result with metadata
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
     * Enhanced loader detection combining extension, content, and heuristics
     */
    private DatasetLoader findBestLoaderWithDetection(String filePath) {
        try {
            // Analyze file content to determine dataset type
            DatasetTypeDetector.DetectionResult detection = typeDetector.detectDatasetType(filePath);

            // Find loader matching detected type
            DatasetLoader detectedLoader = findLoaderByType(detection.detectedType());
            if (detectedLoader != null && detection.confidence() > 0.4) { // Lowered threshold
                return detectedLoader;
            }

            // If we detected something but confidence is low, still prefer it over generic fallback
            if (detectedLoader != null && detection.confidence() > 0.2) {
                return detectedLoader;
            }

            // Fallback to extension-based lookup
            String extension = getFileExtension(filePath);
            DatasetLoader extensionLoader = loaderRegistry.get(extension.toLowerCase());
            if (extensionLoader != null && extensionLoader.canHandle(filePath)) {
                return extensionLoader;
            }

            // Final fallback: try all loaders in priority order
            for (DatasetLoader candidate : availableLoaders) {
                if (candidate.canHandle(filePath)) {
                    return candidate;
                }
            }

        } catch (Exception e) {
            logger.error("Error during content detection: {}", e.getMessage(), e);

            // Error fallback - use extension only
            String extension = getFileExtension(filePath);
            DatasetLoader fallbackLoader = loaderRegistry.get(extension.toLowerCase());
            if (fallbackLoader != null) {
                return fallbackLoader;
            }
        }

        return null;
    }

    /**
     * Find loader by dataset type name
     */
    private DatasetLoader findLoaderByType(String datasetType) {
        if (datasetType == null) return null;

        return availableLoaders.stream()
                .filter(loader -> loader.getDatasetTypeName().equalsIgnoreCase(datasetType))
                .findFirst()
                .orElse(null);
    }

    // Existing methods remain unchanged...
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

        return new CombinedDatasetResult(allDatasets, individualResults, datasetCounts);
    }

    private String getFileExtension(String filePath) {
        int lastDot = filePath.lastIndexOf('.');
        return lastDot > 0 ? filePath.substring(lastDot) : "";
    }

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
                logger.warn("Large file detected: {} MB - loading may take time", fileSize / (1024 * 1024));
            }
        } catch (Exception e) {
            logger.debug("Could not check file size: {}", e.getMessage());
        }
    }

    private DatasetAnalysis analyzeDataset(List<Dataset> datasets) {
        if (datasets.isEmpty()) {
            return new DatasetAnalysis(0, Map.of(), 0, 0, 0);
        }

        Map<Dataset.SentimentLabel, Long> sentimentCounts = datasets.stream()
                .collect(Collectors.groupingBy(Dataset::getSentiment, Collectors.counting()));

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

    // Utility methods
    public List<String> getSupportedExtensions() {
        return new ArrayList<>(loaderRegistry.keySet());
    }

    public List<String> getAvailableDatasetTypes() {
        return availableLoaders.stream()
                .map(DatasetLoader::getDatasetTypeName)
                .collect(Collectors.toList());
    }

    public boolean canHandleFile(String filePath) {
        return findBestLoaderWithDetection(filePath) != null;
    }
}

/**
 * Content-based dataset type detector using header analysis and heuristics
 */
class DatasetTypeDetector {
    private static final Logger logger = LoggerFactory.getLogger(DatasetTypeDetector.class);

    // Header patterns for different dataset types (case-insensitive)
    private static final Map<String, Set<String>> HEADER_PATTERNS = Map.of(
            "Movie Reviews", Set.of("review", "movie", "film", "imdb", "sentiment", "polarity", "label", "text"),
            "Twitter Data", Set.of("tweet", "twitter", "user", "mention", "hashtag", "retweet", "@", "tweet_text", "target"),
            "Product Reviews", Set.of("product", "amazon", "asin", "overall", "rating", "verified", "summary")
    );

    // Content patterns for different dataset types
    private static final Map<String, Set<String>> CONTENT_PATTERNS = Map.of(
            "Movie Reviews", Set.of("director", "actor", "plot", "cinema", "screenplay", "character", "scene", "episode",
                    "acting", "ending", "cast", "series", "season", "drama", "comedy", "thriller"),
            "Twitter Data", Set.of("@", "#", "rt ", "http://", "https://", "pic.twitter", "via @", "retweet", "follow", "dm"),
            "Product Reviews", Set.of("bought", "purchase", "shipped", "delivery", "price", "amazon", "verified purchase",
                    "quality", "recommend", "item arrived", "customer service", "refund")
    );

    public DetectionResult detectDatasetType(String filePath) throws IOException {
        String extension = getFileExtension(filePath).toLowerCase();

        if (extension.equals(".json") || extension.equals(".jsonl")) {
            return detectJsonDatasetType(filePath);
        } else if (extension.equals(".csv") || extension.equals(".tsv")) {
            return detectCsvDatasetType(filePath);
        }

        return new DetectionResult("Unknown", 0.0, "Unsupported file extension");
    }

    private DetectionResult detectCsvDatasetType(String filePath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath, StandardCharsets.UTF_8))) {
            // Read header line
            String headerLine = reader.readLine();
            if (headerLine == null || headerLine.trim().isEmpty()) {
                return new DetectionResult("Unknown", 0.0, "Empty file or no header");
            }

            logger.debug("CSV header detected: {}", headerLine);

            // Read first few data lines for content analysis
            List<String> sampleLines = new ArrayList<>();
            String line;
            int count = 0;
            while ((line = reader.readLine()) != null && count < 20) { // Read more samples
                if (!line.trim().isEmpty()) {
                    sampleLines.add(line.toLowerCase());
                    count++;
                }
            }

            logger.debug("Read {} sample lines for content analysis", sampleLines.size());

            return analyzeContent(headerLine.toLowerCase(), sampleLines);
        } catch (IOException e) {
            logger.error("Error reading CSV file for detection: {}", e.getMessage());
            throw e;
        } catch (Exception e) {
            logger.error("Unexpected error during CSV detection: {}", e.getMessage(), e);
            return new DetectionResult("Unknown", 0.0, "Error during analysis: " + e.getMessage());
        }
    }

    private DetectionResult detectJsonDatasetType(String filePath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath, StandardCharsets.UTF_8))) {
            StringBuilder sample = new StringBuilder();
            String line;
            int count = 0;

            // Read first few lines/objects
            while ((line = reader.readLine()) != null && count < 10) {
                sample.append(line.toLowerCase()).append(" ");
                count++;
            }

            String content = sample.toString();
            List<String> sampleLines = List.of(content);

            // For JSON, we analyze field names and content together
            return analyzeContent(content, sampleLines);
        }
    }

    private DetectionResult analyzeContent(String header, List<String> sampleLines) {
        Map<String, Double> scores = new HashMap<>();

        logger.debug("Analyzing header: {}", header.length() > 200 ? header.substring(0, 200) + "..." : header);

        // Score based on header patterns
        for (Map.Entry<String, Set<String>> entry : HEADER_PATTERNS.entrySet()) {
            String datasetType = entry.getKey();
            Set<String> patterns = entry.getValue();

            long matchCount = patterns.stream()
                    .mapToLong(pattern -> header.contains(pattern) ? 1 : 0)
                    .sum();

            double headerScore = (double) matchCount / patterns.size();
            scores.put(datasetType, headerScore * 0.7); // Increased header weight: 70%

        }

        // Score based on content patterns (analyze all sample lines)
        for (Map.Entry<String, Set<String>> entry : CONTENT_PATTERNS.entrySet()) {
            String datasetType = entry.getKey();
            Set<String> patterns = entry.getValue();

            double totalContentScore = 0.0;
            int analyzedLines = Math.min(sampleLines.size(), 10); // Analyze up to 10 lines

            for (int i = 0; i < analyzedLines; i++) {
                String line = sampleLines.get(i);
                long matchCount = patterns.stream()
                        .mapToLong(pattern -> line.contains(pattern) ? 1 : 0)
                        .sum();

                if (matchCount > 0) {
                    totalContentScore += (double) matchCount / patterns.size();
                }
            }

            double avgContentScore = analyzedLines > 0 ? totalContentScore / analyzedLines : 0.0;
            scores.put(datasetType, scores.getOrDefault(datasetType, 0.0) + avgContentScore * 0.3); // Content weight: 30%
        }

        // Find best match
        Map.Entry<String, Double> bestMatch = scores.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .orElse(Map.entry("Unknown", 0.0));

        // More lenient threshold for detection
        String detectedType = bestMatch.getValue() > 0.1 ? bestMatch.getKey() : "Unknown"; // Lowered from 0.3
        double confidence = Math.min(bestMatch.getValue(), 1.0);
        String reasoning = generateReasoning(header, sampleLines, scores);

        return new DetectionResult(detectedType, confidence, reasoning);
    }

    private String generateReasoning(String header, List<String> sampleLines, Map<String, Double> scores) {
        StringBuilder reasoning = new StringBuilder();
        reasoning.append("Header analysis: ").append(header.length() > 100 ? header.substring(0, 100) + "..." : header);
        reasoning.append(". Scores: ");

        scores.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .forEach(entry -> reasoning.append(entry.getKey()).append("=").append(String.format("%.2f", entry.getValue())).append(" "));

        return reasoning.toString().trim();
    }

    private String getFileExtension(String filePath) {
        int lastDot = filePath.lastIndexOf('.');
        return lastDot > 0 ? filePath.substring(lastDot) : "";
    }

    /**
     * Result of dataset type detection
     */
    record DetectionResult(String detectedType, double confidence, String reasoning) {

        @Override
        public String toString() {
            return String.format("DetectionResult{type='%s', confidence=%.2f, reasoning='%s'}",
                    detectedType, confidence, reasoning);
        }
    }
}

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

        return (double) min / max >= 0.7;
    }

    @Override
    public String toString() {
        return String.format("Analysis{samples=%d, sentiments=%s, avgLength=%d, balanced=%s}",
                totalSamples, sentimentDistribution, avgTextLength, isBalanced());
    }
}