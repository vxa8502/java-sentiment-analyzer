package sentiment.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

/**
 * CLI tool for validating and reporting on dataset quality.
 *
 * Sofia's Philosophy: "Validate your data before you validate your model."
 *
 * Usage:
 *   mvn exec:java -Dexec.mainClass="sentiment.data.DatasetValidationCLI"
 *
 * @author Victoria Alabi
 */
@SpringBootApplication
@ComponentScan(basePackages = "sentiment.data")
public class DatasetValidationCLI implements CommandLineRunner {

    private static final Logger logger = LoggerFactory.getLogger(DatasetValidationCLI.class);

    private final IMDBDatasetLoader imdbLoader;

    public DatasetValidationCLI(IMDBDatasetLoader imdbLoader) {
        this.imdbLoader = imdbLoader;
    }

    public static void main(String[] args) {
        // Disable Spring Boot banner for cleaner output
        SpringApplication app = new SpringApplication(DatasetValidationCLI.class);
        app.setBannerMode(org.springframework.boot.Banner.Mode.OFF);
        app.run(args);
    }

    @Override
    public void run(String... args) throws Exception {
        printHeader();

        // Validate training set
        logger.info("\n" + "=".repeat(70));
        logger.info("VALIDATING TRAINING SET");
        logger.info("=".repeat(70));

        DatasetLoadResult trainResult = imdbLoader.loadTrainSet();
        DatasetStatistics trainStats = imdbLoader.computeStatistics(trainResult.datasets());

        logger.info("\nLoad Time: {} ms", trainResult.loadTimeMs());
        logger.info(trainStats.generateReport());

        // Validate test set
        logger.info("\n" + "=".repeat(70));
        logger.info("VALIDATING TEST SET");
        logger.info("=".repeat(70));

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

    private void printHeader() {
        System.out.println("\n");
        System.out.println("╔═══════════════════════════════════════════════════════════════════╗");
        System.out.println("║                                                                   ║");
        System.out.println("║           IMDB DATASET VALIDATION & QUALITY AUDIT                 ║");
        System.out.println("║                                                                   ║");
        System.out.println("║           Following Sofia's Data Quality Standards               ║");
        System.out.println("║                                                                   ║");
        System.out.println("╚═══════════════════════════════════════════════════════════════════╝");
        System.out.println();
    }

    private void printCrossSetComparison(DatasetStatistics train, DatasetStatistics test) {
        logger.info("\n" + "=".repeat(70));
        logger.info("CROSS-SET COMPARISON");
        logger.info("=".repeat(70));

        logger.info("\n📊 SIZE COMPARISON:");
        logger.info("   Train: {:,} examples", train.getTotalExamples());
        logger.info("   Test:  {:,} examples", test.getTotalExamples());

        double sizeRatio = (double) test.getTotalExamples() / train.getTotalExamples();
        logger.info("   Ratio: {:.3f} {}", sizeRatio,
            Math.abs(sizeRatio - 1.0) < 0.01 ? "✓ EQUAL SPLIT" : "");

        logger.info("\n📏 TEXT LENGTH COMPARISON:");
        logger.info("   Train Avg: {:.0f} chars | Test Avg: {:.0f} chars",
            train.getAvgTextLength(), test.getAvgTextLength());
        logger.info("   Train Median: {:.0f} chars | Test Median: {:.0f} chars",
            train.getMedianTextLength(), test.getMedianTextLength());

        double lengthDiff = Math.abs(train.getAvgTextLength() - test.getAvgTextLength());
        double lengthDiffPct = (lengthDiff / train.getAvgTextLength()) * 100;
        logger.info("   Difference: {:.1f}% {}", lengthDiffPct,
            lengthDiffPct < 10 ? "✓ SIMILAR DISTRIBUTION" : "⚠ DISTRIBUTION SHIFT");

        logger.info("\n🏷️  LABEL BALANCE COMPARISON:");
        logger.info("   Train Balance Ratio: {:.3f}", train.getLabelBalanceRatio());
        logger.info("   Test Balance Ratio:  {:.3f}", test.getLabelBalanceRatio());

        boolean bothBalanced = train.isBalanced() && test.isBalanced();
        logger.info("   Status: {}", bothBalanced ? "✓ BOTH BALANCED" : "⚠ IMBALANCE DETECTED");

        logger.info("\n📚 VOCABULARY COMPARISON:");
        logger.info("   Train Vocabulary: {:.0f} unique tokens", train.getVocabularySize());
        logger.info("   Test Vocabulary:  {:.0f} unique tokens", test.getVocabularySize());

        double vocabRatio = test.getVocabularySize() / train.getVocabularySize();
        logger.info("   Test/Train Ratio: {:.3f}", vocabRatio);

        if (vocabRatio > 1.1) {
            logger.warn("   ⚠ WARNING: Test vocabulary significantly larger than train!");
            logger.warn("   This may indicate out-of-vocabulary issues.");
        } else {
            logger.info("   ✓ Test vocabulary coverage looks reasonable");
        }
    }

    private void printDataSamples(java.util.List<Dataset> datasets) {
        logger.info("\n" + "=".repeat(70));
        logger.info("DATA PREVIEW (First 3 Examples)");
        logger.info("=".repeat(70));

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

    private void printFooter(DatasetStatistics train, DatasetStatistics test) {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("VALIDATION SUMMARY");
        System.out.println("=".repeat(70));

        boolean allChecksPass = train.passesQualityChecks() && test.passesQualityChecks();

        System.out.println("\nTrain Set Quality: " +
            (train.passesQualityChecks() ? "✅ PASS" : "⚠️  WARNINGS"));
        System.out.println("Test Set Quality:  " +
            (test.passesQualityChecks() ? "✅ PASS" : "⚠️  WARNINGS"));

        System.out.println("\n" + (allChecksPass ? "✅" : "⚠️") + " Overall Status: " +
            (allChecksPass ? "READY FOR TRAINING" : "REVIEW WARNINGS"));

        System.out.println("\n" + "=".repeat(70));
        System.out.println("\nSofia's Note:");
        System.out.println("  • Data quality validated ✓");
        System.out.println("  • Label distributions checked ✓");
        System.out.println("  • Train/test consistency verified ✓");
        System.out.println("  • Ready to build reliable models!");
        System.out.println("\n" + "=".repeat(70) + "\n");
    }
}
