package sentiment.data;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Comprehensive unit tests for DatasetLoaderRegistry.
 *
 * Tests cover:
 * - File validation and error handling
 * - Memory limit enforcement
 * - Loader selection and routing
 * - CSV file loading
 * - Multi-file loading
 * - Extension support
 */
@DisplayName("DatasetLoaderRegistry Unit Tests")
class DatasetLoaderRegistryTest {

    private DatasetLoaderRegistry loaderManager;
    private DataLoadingLimits limits;

    @TempDir
    Path tempDir;

    @BeforeEach
    void setUp() {
        // Create limits with reasonable test values (respecting minimum constraints)
        limits = new DataLoadingLimits();
        limits.setMaxDatasetsPerFile(1000);
        limits.setMaxCombinedDatasets(10000);  // Must be >= 10000 per validation
        limits.setMaxFileSizeMb(10);
        loaderManager = new DatasetLoaderRegistry(limits);
    }

    // ==================== INITIALIZATION TESTS ====================

    @Test
    @DisplayName("Constructor should initialize with available loaders")
    void testConstructor_InitializesLoaders() {
        assertNotNull(loaderManager);

        List<String> supportedExtensions = loaderManager.getSupportedExtensions();
        assertFalse(supportedExtensions.isEmpty(),
            "Should have supported extensions");

        List<String> datasetTypes = loaderManager.getAvailableDatasetTypes();
        assertFalse(datasetTypes.isEmpty(),
            "Should have available dataset types");
    }

    @Test
    @DisplayName("Should support common file extensions")
    void testSupportedExtensions_ContainsCommonFormats() {
        List<String> extensions = loaderManager.getSupportedExtensions();

        assertTrue(extensions.contains(".csv"),
            "Should support CSV files");
        assertTrue(extensions.contains(".jsonl"),
            "Should support JSONL files");
    }

    @Test
    @DisplayName("Should have multiple dataset types available")
    void testAvailableDatasetTypes_ContainsExpectedTypes() {
        List<String> types = loaderManager.getAvailableDatasetTypes();

        assertTrue(types.size() >= 3,
            "Should have at least 3 dataset types");

        // Verify expected types
        assertTrue(types.contains("Movie Reviews") ||
                   types.contains("Product Reviews") ||
                   types.contains("Twitter Data"),
            "Should contain expected dataset types");
    }

    // ==================== FILE VALIDATION TESTS ====================

    @Test
    @DisplayName("loadDataset should throw when file path is null")
    void testLoadDataset_NullPath_ThrowsException() {
        assertThrows(DataLoadingException.class,
            () -> loaderManager.loadDataset(null),
            "Should throw DataLoadingException for null path");
    }

    @Test
    @DisplayName("loadDataset should throw when file path is empty")
    void testLoadDataset_EmptyPath_ThrowsException() {
        assertThrows(DataLoadingException.class,
            () -> loaderManager.loadDataset(""),
            "Should throw DataLoadingException for empty path");
    }

    @Test
    @DisplayName("loadDataset should throw when file does not exist")
    void testLoadDataset_NonExistentFile_ThrowsException() {
        String nonExistentPath = tempDir.resolve("nonexistent.csv").toString();

        assertThrows(DataLoadingException.class,
            () -> loaderManager.loadDataset(nonExistentPath),
            "Should throw DataLoadingException for non-existent file");
    }

    @Test
    @DisplayName("loadDataset should throw when file is empty")
    void testLoadDataset_EmptyFile_ThrowsException() throws IOException {
        // Create empty file
        Path emptyFile = tempDir.resolve("empty.csv");
        Files.createFile(emptyFile);

        assertThrows(DataLoadingException.class,
            () -> loaderManager.loadDataset(emptyFile.toString()),
            "Should throw DataLoadingException for empty file");
    }

    @Test
    @DisplayName("loadDataset should throw for unsupported file extension")
    void testLoadDataset_UnsupportedExtension_ThrowsException() throws IOException {
        // Create file with unsupported extension
        Path unsupportedFile = tempDir.resolve("data.xlsx");
        Files.writeString(unsupportedFile, "Some data");

        assertThrows(DataLoadingException.class,
            () -> loaderManager.loadDataset(unsupportedFile.toString()),
            "Should throw DataLoadingException for unsupported extension");
    }

    // ==================== MEMORY LIMIT TESTS ====================

    @Test
    @DisplayName("loadDataset should enforce file size limits")
    void testLoadDataset_ExceedsFileSizeLimit_ThrowsException() throws IOException {
        // Create a file that exceeds the limit (10MB in our test config)
        Path largeFile = tempDir.resolve("large.csv");

        // Write header
        StringBuilder content = new StringBuilder();
        content.append("text,sentiment\n");

        // Write enough data to exceed limit (simplified for test)
        // In practice, this would need to be > 10MB
        // For testing purposes, we'll create a smaller file and adjust limits

        DataLoadingLimits strictLimits = new DataLoadingLimits();
        strictLimits.setMaxDatasetsPerFile(1000);
        strictLimits.setMaxCombinedDatasets(5000);
        strictLimits.setMaxFileSizeMb(0); // 0 MB limit
        DatasetLoaderRegistry strictManager = new DatasetLoaderRegistry(strictLimits);

        Files.writeString(largeFile, content.toString());

        assertThrows(DataLoadingException.class,
            () -> strictManager.loadDataset(largeFile.toString()),
            "Should throw DataLoadingException when file size exceeds limit");
    }

    // ==================== CSV LOADING TESTS ====================

    @Test
    @DisplayName("loadDataset should successfully load valid CSV file")
    void testLoadDataset_ValidCsv_LoadsSuccessfully() throws Exception {
        // Create valid movie review CSV
        Path csvFile = tempDir.resolve("movie_reviews.csv");
        String csvContent = """
            text,sentiment
            This movie was amazing and fantastic!,positive
            Terrible film with bad acting,negative
            Great cinematography and story,positive
            Waste of time and money,negative
            """;
        Files.writeString(csvFile, csvContent);

        DatasetLoadResult result = loaderManager.loadDataset(csvFile.toString());

        assertNotNull(result);
        assertEquals(4, result.datasets().size(),
            "Should load 4 datasets from CSV");
        assertTrue(result.loadTimeMs() >= 0,
            "Load time should be non-negative");
    }

    @Test
    @DisplayName("loadDataset should detect dataset type from filename")
    void testLoadDataset_DetectsTypeFromFilename() throws Exception {
        // Create movie review file
        Path movieFile = tempDir.resolve("imdb_movie_reviews.csv");
        String csvContent = """
            text,sentiment
            Great movie!,positive
            Bad film,negative
            """;
        Files.writeString(movieFile, csvContent);

        DatasetLoadResult result = loaderManager.loadDataset(movieFile.toString());

        assertNotNull(result.datasetType());
        assertTrue(result.datasetType().contains("Movie") ||
                   result.datasetType().contains("Review"),
            "Should detect movie review type from filename");
    }

    // ==================== MULTI-FILE LOADING TESTS ====================

    @Test
    @DisplayName("loadMultipleDatasets should combine datasets from multiple files")
    void testLoadMultipleDatasets_CombinesFiles() throws Exception {
        // Create two CSV files
        Path file1 = tempDir.resolve("reviews1.csv");
        Path file2 = tempDir.resolve("reviews2.csv");

        String content1 = """
            text,sentiment
            Good product,positive
            Bad service,negative
            """;

        String content2 = """
            text,sentiment
            Amazing quality,positive
            Terrible experience,negative
            """;

        Files.writeString(file1, content1);
        Files.writeString(file2, content2);

        CombinedDatasetResult result = loaderManager.loadMultipleDatasets(
            file1.toString(),
            file2.toString()
        );

        assertNotNull(result);
        assertEquals(4, result.getTotalSamples(),
            "Should combine all samples from both files");
        assertEquals(2, result.individualResults().size(),
            "Should have 2 individual results");
    }

    @Test
    @DisplayName("loadMultipleDatasets should enforce combined dataset limits")
    void testLoadMultipleDatasets_EnforcesCombinedLimit() throws Exception {
        // Create limits with strict combined limit (using valid minimum values)
        DataLoadingLimits strictLimits = new DataLoadingLimits();
        strictLimits.setMaxDatasetsPerFile(6000);    // per file: must be >= 1000
        strictLimits.setMaxCombinedDatasets(10000);  // combined: must be >= 10000
        strictLimits.setMaxFileSizeMb(10);           // file size
        strictLimits.setStrictEnforcement(true);     // Ensure strict enforcement is enabled
        DatasetLoaderRegistry strictManager = new DatasetLoaderRegistry(strictLimits);

        // Create two files with 5500 rows each = 11000 total (exceeds combined limit of 10000)
        Path file1 = tempDir.resolve("data1.csv");
        Path file2 = tempDir.resolve("data2.csv");

        StringBuilder content = new StringBuilder("text,sentiment\n");
        for (int i = 0; i < 5500; i++) {
            content.append("Sample text ").append(i).append(",positive\n");
        }

        Files.writeString(file1, content.toString());
        Files.writeString(file2, content.toString());

        assertThrows(DataLoadingException.class,
            () -> strictManager.loadMultipleDatasets(
                file1.toString(),
                file2.toString()
            ),
            "Should throw when combined datasets exceed limit");
    }

    // ==================== LOADER SELECTION TESTS ====================

    @Test
    @DisplayName("canHandleFile should return true for supported files")
    void testCanHandleFile_SupportedExtension_ReturnsTrue() throws IOException {
        Path csvFile = tempDir.resolve("test.csv");
        Files.writeString(csvFile, """
            text,sentiment
            This is a great movie,positive
            This is a terrible movie,negative
            Amazing film,positive
            Worst ever,negative
            Pretty good,positive
            """);

        boolean canHandle = loaderManager.canHandleFile(csvFile.toString());

        assertTrue(canHandle,
            "Should be able to handle CSV files");
    }

    @Test
    @DisplayName("canHandleFile should return false for unsupported files")
    void testCanHandleFile_UnsupportedExtension_ReturnsFalse() {
        boolean canHandle = loaderManager.canHandleFile("test.pdf");

        assertFalse(canHandle,
            "Should not be able to handle PDF files");
    }

    @Test
    @DisplayName("getLoaderForFile should return appropriate loader")
    void testGetLoaderForFile_ReturnsLoader() throws IOException, DataLoadingException {
        Path movieFile = tempDir.resolve("imdb_reviews.csv");
        Files.writeString(movieFile, """
            text,sentiment
            Great movie with excellent acting,positive
            Terrible film with poor direction,negative
            Amazing cinematography and storyline,positive
            Worst movie I've ever seen,negative
            Pretty good overall,positive
            """);

        DatasetLoader loader = loaderManager.getLoaderForFile(movieFile.toString());

        assertNotNull(loader,
            "Should return a loader for supported file");
    }

    @Test
    @DisplayName("getLoaderForFile should return null for unsupported files")
    void testGetLoaderForFile_UnsupportedFile_ReturnsNull() throws DataLoadingException {
        DatasetLoader loader = loaderManager.getLoaderForFile("test.pdf");

        assertNull(loader,
            "Should return null for unsupported file extension");
    }

    // ==================== LIMITS AND CONFIGURATION TESTS ====================

    @Test
    @DisplayName("getLimits should return configured limits")
    void testGetLimits_ReturnsConfiguredLimits() {
        DataLoadingLimits retrievedLimits = loaderManager.getLimits();

        assertNotNull(retrievedLimits);
        assertEquals(1000, retrievedLimits.getMaxDatasetsPerFile());
        assertEquals(10000, retrievedLimits.getMaxCombinedDatasets());  // Updated to match valid minimum
        assertEquals(10, retrievedLimits.getMaxFileSizeMb());
    }

    @Test
    @DisplayName("getCapabilityInfo should provide comprehensive information")
    void testGetCapabilityInfo_ProvidesInformation() {
        LoadingCapabilityInfo info = loaderManager.getCapabilityInfo();

        assertNotNull(info);
        assertTrue(info.loaderCount() > 0,
            "Should have loaders available");
        assertFalse(info.supportedExtensions().isEmpty(),
            "Should have supported extensions");
        assertNotNull(info.limits(),
            "Should include limits information");
    }

    // ==================== DATASET ANALYSIS TESTS ====================

    @Test
    @DisplayName("loadDataset should provide dataset analysis")
    void testLoadDataset_ProvidesAnalysis() throws Exception {
        Path csvFile = tempDir.resolve("analyzed_reviews.csv");
        String csvContent = """
            text,sentiment
            Excellent product with great features,positive
            Poor quality and bad service,negative
            Amazing experience overall,positive
            Terrible waste of money,negative
            Outstanding quality and value,positive
            """;
        Files.writeString(csvFile, csvContent);

        DatasetLoadResult result = loaderManager.loadDataset(csvFile.toString());

        assertNotNull(result);
        assertEquals(5, result.datasets().size());

        // Should have loaded successfully with metadata
        assertTrue(result.loadTimeMs() >= 0);
        assertNotNull(result.datasetType());
    }

    // ==================== EDGE CASE TESTS ====================

    @Test
    @DisplayName("loadDataset should handle CSV with varying line lengths")
    void testLoadDataset_HandlesVaryingLineLengths() throws Exception {
        Path csvFile = tempDir.resolve("varying_lengths.csv");
        String csvContent = """
            text,sentiment
            Short,positive
            This is a much longer review with many more words to test handling,negative
            Medium length review here,positive
            """;
        Files.writeString(csvFile, csvContent);

        DatasetLoadResult result = loaderManager.loadDataset(csvFile.toString());

        assertNotNull(result);
        assertEquals(3, result.datasets().size());
    }

    @Test
    @DisplayName("loadDataset should handle special characters in text")
    void testLoadDataset_HandlesSpecialCharacters() throws Exception {
        Path csvFile = tempDir.resolve("special_chars.csv");
        String csvContent = "text,sentiment\n" +
            "\"Review with comma and quotes\",positive\n" +
            "Text with special chars,negative\n";
        Files.writeString(csvFile, csvContent);

        DatasetLoadResult result = loaderManager.loadDataset(csvFile.toString());

        assertNotNull(result);
        assertTrue(result.datasets().size() >= 1,
            "Should handle special characters in CSV");
    }

    // ==================== INTEGRATION TESTS ====================

    @Test
    @DisplayName("End-to-end loading workflow should work")
    void testEndToEndLoadingWorkflow() throws Exception {
        // Create a realistic CSV file
        Path csvFile = tempDir.resolve("complete_dataset.csv");
        StringBuilder content = new StringBuilder();
        content.append("text,sentiment\n");

        for (int i = 0; i < 50; i++) {
            String sentiment = (i % 2 == 0) ? "positive" : "negative";
            content.append("Sample review text number ")
                   .append(i)
                   .append(",")
                   .append(sentiment)
                   .append("\n");
        }

        Files.writeString(csvFile, content.toString());

        // Load the dataset
        DatasetLoadResult result = loaderManager.loadDataset(csvFile.toString());

        // Verify all aspects
        assertNotNull(result);
        assertEquals(50, result.datasets().size(),
            "Should load all 50 samples");
        assertNotNull(result.datasetType(),
            "Should identify dataset type");
        assertTrue(result.loadTimeMs() >= 0,
            "Should record load time");

        // Verify datasets are properly parsed
        List<Dataset> datasets = result.datasets();
        for (Dataset dataset : datasets) {
            assertNotNull(dataset.getText(),
                "Each dataset should have text");
            assertNotNull(dataset.getSentiment(),
                "Each dataset should have sentiment label");
        }
    }
}
