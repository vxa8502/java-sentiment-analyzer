package sentiment.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

/**
 * Command-line tool for validating sentiment analysis dataset quality.
 *
 * <p>The tool generates detailed reports identifying potential data quality issues
 * that could impact model training and performance.
 *
 * @author Victoria Alabi
 * @see DatasetStatistics
 * @see IMDBDatasetLoader
 */
@SpringBootApplication
@ComponentScan(basePackages = "sentiment.data")
public class DatasetValidationCLI implements CommandLineRunner {

    private static final Logger logger = LoggerFactory.getLogger(DatasetValidationCLI.class);

    private final IMDBDatasetLoader imdbLoader;

    /**
     * Constructs a new dataset validation CLI with the specified loader.
     *
     * @param imdbLoader the IMDB dataset loader for accessing training and test data
     */
    public DatasetValidationCLI(IMDBDatasetLoader imdbLoader) {
        this.imdbLoader = imdbLoader;
    }

    /**
     * Application entry point.
     *
     * <p>Configures Spring Boot to disable the banner for cleaner console output
     * and runs the validation CLI.
     *
     * @param args command-line arguments (currently unused)
     */
    public static void main(String[] args) {
        SpringApplication app = new SpringApplication(DatasetValidationCLI.class);
        app.setBannerMode(org.springframework.boot.Banner.Mode.OFF);
        app.run(args);
    }

    /**
     * Executes the dataset validation workflow.
     *
     * @param args command-line arguments (currently unused)
     * @throws Exception if dataset loading or validation fails
     */
    @Override
    public void run(String... args) throws Exception {
        printHeader();

        // Validate training set
        String separator = "=".repeat(70);
        logger.info("\n{}", separator);
        logger.info("VALIDATING TRAINING SET");
        logger.info(separator);

        DatasetLoadResult trainResult = imdbLoader.loadTrainSet();
        DatasetStatistics trainStats = imdbLoader.computeStatistics(trainResult.datasets());

        logger.info("\nLoad Time: {} ms", trainResult.loadTimeMs());
        logger.info(trainStats.generateReport());

        // Validate test set
        logger.info("\n{}", separator);
        logger.info("VALIDATING TEST SET");
        logger.info(separator);

        DatasetLoadResult testResult = imdbLoader.loadTestSet();
        DatasetStatistics testStats = imdbLoader.computeStatistics(testResult.datasets());

        logger.info("\nLoad Time: {} ms", testResult.loadTimeMs());
        logger.info(testStats.generateReport());

        // Cross-set comparison
        printCrossSetComparison(trainStats, testStats);

        // Sample data preview
        printDataSamples(trainResult.datasets());

        printFooter(trainStats, testStats);
    }

    /**
     * Prints the application header banner.
     */
    private void printHeader() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("IMDB DATASET VALIDATION & QUALITY AUDIT");
        System.out.println("=".repeat(70) + "\n");
    }

    /**
     * Compares training and test dataset statistics to detect distribution shifts.
     *
     * @param train training dataset statistics
     * @param test test dataset statistics
     */
    private void printCrossSetComparison(DatasetStatistics train, DatasetStatistics test) {
        String separator = "=".repeat(70);
        logger.info("\n{}", separator);
        logger.info("CROSS-SET COMPARISON");
        logger.info(separator);

        logger.info("\nSIZE COMPARISON:");
        logger.info("   Train: {} examples", String.format("%,d", train.getTotalExamples()));
        logger.info("   Test:  {} examples", String.format("%,d", test.getTotalExamples()));

        double sizeRatio = (double) test.getTotalExamples() / train.getTotalExamples();
        String splitStatus = Math.abs(sizeRatio - 1.0) < 0.01 ? " [EQUAL SPLIT]" : "";
        logger.info("   Ratio: {}{}", String.format("%.3f", sizeRatio), splitStatus);

        logger.info("\nTEXT LENGTH COMPARISON:");
        logger.info("   Train Avg: {} chars | Test Avg: {} chars",
            String.format("%.0f", train.getAvgTextLength()),
            String.format("%.0f", test.getAvgTextLength()));
        logger.info("   Train Median: {} chars | Test Median: {} chars",
            String.format("%.0f", train.getMedianTextLength()),
            String.format("%.0f", test.getMedianTextLength()));

        double lengthDiff = Math.abs(train.getAvgTextLength() - test.getAvgTextLength());
        double lengthDiffPct = (lengthDiff / train.getAvgTextLength()) * 100;
        String distStatus = lengthDiffPct < 10 ? " [SIMILAR]" : " [DISTRIBUTION SHIFT]";
        logger.info("   Difference: {}{}", String.format("%.1f%%", lengthDiffPct), distStatus);

        logger.info("\nLABEL BALANCE COMPARISON:");
        logger.info("   Train Balance Ratio: {}", String.format("%.3f", train.getLabelBalanceRatio()));
        logger.info("   Test Balance Ratio:  {}", String.format("%.3f", test.getLabelBalanceRatio()));

        boolean bothBalanced = train.isBalanced() && test.isBalanced();
        logger.info("   Status: {}", bothBalanced ? "[BALANCED]" : "[IMBALANCE DETECTED]");

        logger.info("\nVOCABULARY COMPARISON:");
        logger.info("   Train Vocabulary: {} unique tokens", String.format("%.0f", train.getVocabularySize()));
        logger.info("   Test Vocabulary:  {} unique tokens", String.format("%.0f", test.getVocabularySize()));

        double vocabRatio = test.getVocabularySize() / train.getVocabularySize();
        logger.info("   Test/Train Ratio: {}", String.format("%.3f", vocabRatio));

        if (vocabRatio > 1.1) {
            logger.warn("   WARNING: Test vocabulary significantly larger than train");
            logger.warn("   This may indicate out-of-vocabulary issues");
        } else {
            logger.info("   Test vocabulary coverage is reasonable");
        }
    }

    /**
     * Displays sample data from the dataset for manual inspection.
     *
     * <p>Shows the first 3 examples with sentiment labels, text length,
     * and a preview of the text content (truncated to 150 characters).
     *
     * @param datasets the dataset examples to preview
     */
    private void printDataSamples(java.util.List<Dataset> datasets) {
        String separator = "=".repeat(70);
        logger.info("\n{}", separator);
        logger.info("DATA PREVIEW (First 3 Examples)");
        logger.info(separator);

        for (int i = 0; i < Math.min(3, datasets.size()); i++) {
            Dataset d = datasets.get(i);
            String preview = d.getText().substring(0, Math.min(150, d.getText().length()));
            preview = preview.replace("\n", " ");

            logger.info("\n[Example {}]", i + 1);
            logger.info("  Sentiment: {}", d.getSentiment().getDisplayName().toUpperCase());
            logger.info("  Length: {} characters", d.getTextLength());
            logger.info("  Preview: \"{}{}\"", preview,
                d.getText().length() > 150 ? "..." : "");
        }
    }

    /**
     * Prints the validation summary with overall quality assessment.
     *
     * <p>Reports the pass/fail status of quality checks for both training
     * and test datasets, along with an overall readiness determination.
     *
     * @param train training dataset statistics
     * @param test test dataset statistics
     */
    private void printFooter(DatasetStatistics train, DatasetStatistics test) {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("VALIDATION SUMMARY");
        System.out.println("=".repeat(70));

        boolean allChecksPass = train.passesQualityChecks() && test.passesQualityChecks();

        System.out.println("\nTrain Set Quality: " +
            (train.passesQualityChecks() ? "PASS" : "WARNINGS DETECTED"));
        System.out.println("Test Set Quality:  " +
            (test.passesQualityChecks() ? "PASS" : "WARNINGS DETECTED"));

        System.out.println("\nOverall Status: " +
            (allChecksPass ? "READY FOR TRAINING" : "REVIEW WARNINGS BEFORE TRAINING"));

        System.out.println("\nValidation Complete:");
        System.out.println("  - Data quality validated");
        System.out.println("  - Label distributions checked");
        System.out.println("  - Train/test consistency verified");

        System.out.println("\n" + "=".repeat(70) + "\n");
    }
}
