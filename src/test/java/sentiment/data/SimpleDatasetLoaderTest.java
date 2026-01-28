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

@DisplayName("SimpleDatasetLoader Tests")
class SimpleDatasetLoaderTest {

    private SimpleDatasetLoader loader;

    @TempDir
    Path tempDir;

    @BeforeEach
    void setUp() {
        loader = new SimpleDatasetLoader();
    }

    // ==================== BASIC CSV TESTS ====================

    @Test
    @DisplayName("Should load CSV with standard text,sentiment columns")
    void testLoadCsv_StandardColumns() throws Exception {
        Path csvFile = tempDir.resolve("standard.csv");
        Files.writeString(csvFile, """
            text,sentiment
            This is great,positive
            This is terrible,negative
            This is also good,positive
            """);

        List<Dataset> datasets = loader.load(csvFile.toString());

        assertEquals(3, datasets.size());
        assertEquals(Dataset.SentimentLabel.POSITIVE, datasets.get(0).getSentiment());
        assertEquals(Dataset.SentimentLabel.NEGATIVE, datasets.get(1).getSentiment());
        assertEquals(Dataset.SentimentLabel.POSITIVE, datasets.get(2).getSentiment());
    }

    @Test
    @DisplayName("Should handle CSV with review,label columns (flexible naming)")
    void testLoadCsv_AlternativeColumnNames() throws Exception {
        Path csvFile = tempDir.resolve("alternative.csv");
        Files.writeString(csvFile, """
            review,label
            Amazing product,pos
            Awful experience,neg
            """);

        List<Dataset> datasets = loader.load(csvFile.toString());

        assertEquals(2, datasets.size());
        assertEquals(Dataset.SentimentLabel.POSITIVE, datasets.get(0).getSentiment());
        assertEquals(Dataset.SentimentLabel.NEGATIVE, datasets.get(1).getSentiment());
    }

    @Test
    @DisplayName("Should handle numeric sentiment labels (0/1)")
    void testLoadCsv_NumericLabels() throws Exception {
        Path csvFile = tempDir.resolve("numeric.csv");
        Files.writeString(csvFile, """
            text,sentiment
            Good stuff,1
            Bad stuff,0
            """);

        List<Dataset> datasets = loader.load(csvFile.toString());

        assertEquals(2, datasets.size());
        assertEquals(Dataset.SentimentLabel.POSITIVE, datasets.get(0).getSentiment());
        assertEquals(Dataset.SentimentLabel.NEGATIVE, datasets.get(1).getSentiment());
    }

    @Test
    @DisplayName("Should handle star ratings (1-5, neutral filtered for binary classification)")
    void testLoadCsv_StarRatings() throws Exception {
        Path csvFile = tempDir.resolve("ratings.csv");
        Files.writeString(csvFile, """
            text,rating
            Loved it,5.0
            Hated it,1.0
            It was fine,3.0
            Pretty good,4.0
            Pretty bad,2.0
            """);

        List<Dataset> datasets = loader.load(csvFile.toString());

        // Neutral (3.0) is filtered out for binary classification
        assertEquals(4, datasets.size());
        assertEquals(Dataset.SentimentLabel.POSITIVE, datasets.get(0).getSentiment()); // 5.0
        assertEquals(Dataset.SentimentLabel.NEGATIVE, datasets.get(1).getSentiment()); // 1.0
        assertEquals(Dataset.SentimentLabel.POSITIVE, datasets.get(2).getSentiment()); // 4.0
        assertEquals(Dataset.SentimentLabel.NEGATIVE, datasets.get(3).getSentiment()); // 2.0
    }

    @Test
    @DisplayName("Should handle TSV files")
    void testLoadTsv_TabDelimited() throws Exception {
        Path tsvFile = tempDir.resolve("data.tsv");
        Files.writeString(tsvFile, """
            text\tsentiment
            Great product\tpositive
            Poor quality\tnegative
            """);

        List<Dataset> datasets = loader.load(tsvFile.toString());

        assertEquals(2, datasets.size());
        assertEquals(Dataset.SentimentLabel.POSITIVE, datasets.get(0).getSentiment());
    }

    // ==================== ERROR HANDLING TESTS ====================

    @Test
    @DisplayName("Should throw exception for missing text column")
    void testLoadCsv_MissingTextColumn_ThrowsException() throws Exception {
        Path csvFile = tempDir.resolve("missing_text.csv");
        Files.writeString(csvFile, """
            sentiment,other
            positive,data
            """);

        assertThrows(DataLoadingException.class, () -> loader.load(csvFile.toString()));
    }

    @Test
    @DisplayName("Should throw exception for missing sentiment column")
    void testLoadCsv_MissingSentimentColumn_ThrowsException() throws Exception {
        Path csvFile = tempDir.resolve("missing_sentiment.csv");
        Files.writeString(csvFile, """
            text,other
            Some text,data
            """);

        assertThrows(DataLoadingException.class, () -> loader.load(csvFile.toString()));
    }

    @Test
    @DisplayName("Should skip rows with empty text")
    void testLoadCsv_SkipsEmptyText() throws Exception {
        Path csvFile = tempDir.resolve("empty_text.csv");
        Files.writeString(csvFile, """
            text,sentiment
            Good text,positive
            ,negative
            More good text,positive
            """);

        List<Dataset> datasets = loader.load(csvFile.toString());

        assertEquals(2, datasets.size()); // Should skip the empty text row
    }

    @Test
    @DisplayName("Should skip rows with invalid sentiment")
    void testLoadCsv_SkipsInvalidSentiment() throws Exception {
        Path csvFile = tempDir.resolve("invalid_sentiment.csv");
        Files.writeString(csvFile, """
            text,sentiment
            Good text,positive
            Bad sentiment,invalid_label
            More text,negative
            """);

        List<Dataset> datasets = loader.load(csvFile.toString());

        assertEquals(2, datasets.size()); // Should skip invalid sentiment row
    }

    @Test
    @DisplayName("Should throw exception for non-existent file")
    void testLoad_NonExistentFile_ThrowsException() {
        assertThrows(DataLoadingException.class,
            () -> loader.load(tempDir.resolve("nonexistent.csv").toString()));
    }

    @Test
    @DisplayName("Should throw exception for empty file")
    void testLoad_EmptyFile_ThrowsException() throws IOException {
        Path emptyFile = tempDir.resolve("empty.csv");
        Files.createFile(emptyFile);

        assertThrows(DataLoadingException.class, () -> loader.load(emptyFile.toString()));
    }

    @Test
    @DisplayName("Should throw exception for unsupported file format")
    void testLoad_UnsupportedFormat_ThrowsException() throws IOException {
        Path xlsxFile = tempDir.resolve("data.xlsx");
        Files.writeString(xlsxFile, "fake excel data");

        DataLoadingException ex = assertThrows(DataLoadingException.class,
            () -> loader.load(xlsxFile.toString()));
        assertTrue(ex.getMessage().contains("Unsupported file format"));
    }

    @Test
    @DisplayName("Should throw exception when all rows are invalid")
    void testLoad_AllRowsInvalid_ThrowsException() throws IOException {
        Path csvFile = tempDir.resolve("all_invalid.csv");
        Files.writeString(csvFile, """
            text,sentiment
            ,
            ,
            """);

        assertThrows(DataLoadingException.class, () -> loader.load(csvFile.toString()));
    }

    // ==================== SPECIAL CHARACTERS & EDGE CASES ====================

    @Test
    @DisplayName("Should handle quoted text with commas")
    void testLoadCsv_QuotedText() throws Exception {
        Path csvFile = tempDir.resolve("quoted.csv");
        Files.writeString(csvFile, """
            text,sentiment
            "This text has a comma, and quotes",positive
            Normal text,negative
            """);

        List<Dataset> datasets = loader.load(csvFile.toString());

        assertEquals(2, datasets.size());
        assertTrue(datasets.get(0).getText().contains("comma"));
    }

    @Test
    @DisplayName("Should handle case-insensitive column headers")
    void testLoadCsv_CaseInsensitiveHeaders() throws Exception {
        Path csvFile = tempDir.resolve("case_insensitive.csv");
        Files.writeString(csvFile, """
            TEXT,SENTIMENT
            Upper case headers,POSITIVE
            """);

        List<Dataset> datasets = loader.load(csvFile.toString());

        assertEquals(1, datasets.size());
        assertEquals(Dataset.SentimentLabel.POSITIVE, datasets.get(0).getSentiment());
    }

    @Test
    @DisplayName("Should preserve original label in Dataset")
    void testLoadCsv_PreservesOriginalLabel() throws Exception {
        Path csvFile = tempDir.resolve("original_label.csv");
        Files.writeString(csvFile, """
            text,sentiment
            Test text,pos
            """);

        List<Dataset> datasets = loader.load(csvFile.toString());

        assertEquals("pos", datasets.get(0).getOriginalLabel());
    }

    // ==================== UNIMPLEMENTED FORMAT TESTS ====================

    @Test
    @DisplayName("Should throw exception for JSONL (not yet implemented)")
    void testLoad_JsonlNotImplemented() throws IOException {
        Path jsonlFile = tempDir.resolve("data.jsonl");
        Files.writeString(jsonlFile, "{\"text\":\"test\",\"sentiment\":\"positive\"}");

        DataLoadingException ex = assertThrows(DataLoadingException.class,
            () -> loader.load(jsonlFile.toString()));
        assertTrue(ex.getMessage().contains("not yet implemented"));
    }

    @Test
    @DisplayName("Should throw exception for TXT (not yet implemented)")
    void testLoad_PlaintextNotImplemented() throws IOException {
        Path txtFile = tempDir.resolve("data.txt");
        Files.writeString(txtFile, "__label__positive This is text");

        DataLoadingException ex = assertThrows(DataLoadingException.class,
            () -> loader.load(txtFile.toString()));
        assertTrue(ex.getMessage().contains("not yet implemented"));
    }

    // ==================== INTEGRATION TEST ====================

    @Test
    @DisplayName("Should handle realistic CSV with multiple rows (binary classification)")
    void testLoadCsv_RealisticDataset() throws Exception {
        Path csvFile = tempDir.resolve("realistic.csv");
        StringBuilder content = new StringBuilder("text,sentiment\n");

        // Binary classification: only positive and negative labels
        for (int i = 0; i < 100; i++) {
            String sentiment = (i % 2 == 0) ? "positive" : "negative";
            content.append(String.format("Review number %d with some text,", i))
                   .append(sentiment)
                   .append("\n");
        }

        Files.writeString(csvFile, content.toString());

        List<Dataset> datasets = loader.load(csvFile.toString());

        assertEquals(100, datasets.size());

        // Verify distribution (50/50 split for binary classification)
        long positiveCount = datasets.stream()
            .filter(d -> d.getSentiment() == Dataset.SentimentLabel.POSITIVE)
            .count();
        long negativeCount = datasets.stream()
            .filter(d -> d.getSentiment() == Dataset.SentimentLabel.NEGATIVE)
            .count();

        assertEquals(50, positiveCount);
        assertEquals(50, negativeCount);
    }

    // ==================== WRITE CSV TESTS ====================

    @Test
    @DisplayName("Should write and read back CSV roundtrip")
    void testWriteCsv_Roundtrip() throws Exception {
        // Create test data
        List<Dataset> original = List.of(
            new Dataset.Builder("This is a positive review", Dataset.SentimentLabel.POSITIVE).build(),
            new Dataset.Builder("This is a negative review", Dataset.SentimentLabel.NEGATIVE).build(),
            new Dataset.Builder("Another positive one", Dataset.SentimentLabel.POSITIVE).build()
        );

        // Write to CSV
        Path csvFile = tempDir.resolve("roundtrip.csv");
        SimpleDatasetLoader.writeCsv(original, csvFile);

        // Read back
        List<Dataset> loaded = loader.load(csvFile.toString());

        // Verify
        assertEquals(original.size(), loaded.size());
        for (int i = 0; i < original.size(); i++) {
            assertEquals(original.get(i).getText(), loaded.get(i).getText());
            assertEquals(original.get(i).getSentiment(), loaded.get(i).getSentiment());
        }
    }

    @Test
    @DisplayName("Should escape quotes in CSV output")
    void testWriteCsv_EscapesQuotes() throws Exception {
        List<Dataset> data = List.of(
            new Dataset.Builder("Text with \"quotes\" inside", Dataset.SentimentLabel.POSITIVE).build()
        );

        Path csvFile = tempDir.resolve("quotes.csv");
        SimpleDatasetLoader.writeCsv(data, csvFile);

        // Read back and verify quotes are preserved
        List<Dataset> loaded = loader.load(csvFile.toString());

        assertEquals(1, loaded.size());
        assertEquals("Text with \"quotes\" inside", loaded.get(0).getText());
    }

    @Test
    @DisplayName("Should handle empty list for CSV write")
    void testWriteCsv_EmptyList() throws Exception {
        List<Dataset> data = List.of();

        Path csvFile = tempDir.resolve("empty.csv");
        SimpleDatasetLoader.writeCsv(data, csvFile);

        // File should exist with just header
        assertTrue(Files.exists(csvFile));
        String content = Files.readString(csvFile);
        assertTrue(content.startsWith("review,sentiment"));
    }
}
