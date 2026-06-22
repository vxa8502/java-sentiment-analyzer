package sentiment.evaluation;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.condition.EnabledIf;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import sentiment.data.Dataset;
import sentiment.data.SimpleDatasetLoader;
import sentiment.models.SentimentClassifier;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verification tests for negation edge-case accuracy.
 *
 * <p>On the dedicated negation edge-case test set, the classifier shall achieve
 * at least 95% accuracy, correctly classifying polarity-reversing constructions
 * such as "not recommend" as negative.
 *
 * <p><b>IMPORTANT:</b> This test logs a warning when prerequisites are missing.
 *
 * <p>Prerequisites:
 * <ul>
 *   <li>Production model at ./models/production/sentiment_model.ser</li>
 *   <li>Negation dataset at ./data/test/negation_edge_cases.csv</li>
 * </ul>
 */
@SpringBootTest(properties = {
    "sentiment.model-type=PRODUCTION",
    "sentiment.models.production-model-path=./models/production/sentiment_model.ser"
})
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@Tag("integration")
@DisplayName("Negation Edge-Case Accuracy Verification")
@EnabledIf("productionModelExists")
public class NegationEdgeCaseTest {

    private static final Logger logger = LoggerFactory.getLogger(NegationEdgeCaseTest.class);

    private static final String PRODUCTION_MODEL_PATH =
            "./models/production/sentiment_model.ser";
    private static final Path NEGATION_DATASET_PATH =
            Paths.get("./data/test/negation_edge_cases.csv");

    /**
     * Required accuracy threshold: 95% on negation edge cases.
     */
    private static final double REQUIRED_ACCURACY = 0.95;

    @Autowired
    private SentimentClassifier classifier;

    private List<Dataset> negationEdgeCases;

    /**
     * Condition for @EnabledIf - checks if prerequisites exist.
     */
    static boolean productionModelExists() {
        boolean modelExists = Files.exists(Paths.get(PRODUCTION_MODEL_PATH));
        boolean datasetExists = Files.exists(NEGATION_DATASET_PATH);

        if (!modelExists || !datasetExists) {
            System.err.println("WARNING: Negation edge-case test skipped - prerequisites missing");
            if (!modelExists) {
                System.err.println("  Missing: " + PRODUCTION_MODEL_PATH);
            }
            if (!datasetExists) {
                System.err.println("  Missing: " + NEGATION_DATASET_PATH);
            }
        }

        return modelExists && datasetExists;
    }

    @BeforeAll
    void loadDataset() throws Exception {
        assertTrue(Files.exists(Paths.get(PRODUCTION_MODEL_PATH)),
                "Production model required: " + PRODUCTION_MODEL_PATH);
        assertTrue(Files.exists(NEGATION_DATASET_PATH),
                "Negation dataset required: " + NEGATION_DATASET_PATH);
        assertTrue(classifier.isTrained(), "Loaded model should be trained");

        logger.info("Loading negation edge-case dataset from: {}", NEGATION_DATASET_PATH);
        SimpleDatasetLoader loader = new SimpleDatasetLoader();
        negationEdgeCases = loader.load(NEGATION_DATASET_PATH.toString());

        logger.info("Loaded {} negation edge cases for evaluation", negationEdgeCases.size());
        assertTrue(negationEdgeCases.size() >= 50,
                "Negation dataset should have at least 50 cases for statistical validity");
    }

    @Test
    @DisplayName("Classifier achieves >= 95% accuracy on negation edge cases")
    void verifyNegationEdgeCaseAccuracy() throws Exception {
        int correct = 0;
        int total = 0;
        List<MisclassificationRecord> misclassifications = new ArrayList<>();

        for (Dataset sample : negationEdgeCases) {
            String text = sample.getText();
            String expected = sample.getSentiment().getDisplayName().toLowerCase();

            SentimentClassifier.ClassificationResult result = classifier.classifyWithProbabilities(text);
            String predicted = result.label().toLowerCase();

            total++;
            if (expected.equals(predicted)) {
                correct++;
            } else {
                misclassifications.add(new MisclassificationRecord(
                        text, expected, predicted, result.confidence()));
            }
        }

        double accuracy = (double) correct / total;

        logger.info("Negation Edge-Case Evaluation Results:");
        logger.info("  Total samples:  {}", total);
        logger.info("  Correct:        {}", correct);
        logger.info("  Misclassified:  {}", misclassifications.size());
        logger.info("  Accuracy:       {:.2f}%", accuracy * 100);
        logger.info("  Required:       {:.2f}%", REQUIRED_ACCURACY * 100);
        logger.info("  Status:         {}", accuracy >= REQUIRED_ACCURACY ? "PASS" : "FAIL");

        if (!misclassifications.isEmpty()) {
            logger.warn("Misclassified samples ({}):", misclassifications.size());
            for (MisclassificationRecord record : misclassifications) {
                logger.warn("  '{}' -> expected: {}, predicted: {} (conf: {:.3f})",
                        truncate(record.text, 40), record.expected, record.predicted, record.confidence);
            }
        }

        assertTrue(accuracy >= REQUIRED_ACCURACY,
                String.format(
                    "Negation edge-case accuracy %.2f%% is below required %.2f%%. " +
                    "Misclassified %d of %d samples.",
                    accuracy * 100, REQUIRED_ACCURACY * 100, misclassifications.size(), total));
    }

    @Test
    @DisplayName("Dataset should have balanced positive and negative samples")
    void verifyDatasetBalance() {
        long positiveCount = negationEdgeCases.stream()
                .filter(d -> d.getSentiment().getDisplayName().equalsIgnoreCase("positive"))
                .count();
        long negativeCount = negationEdgeCases.size() - positiveCount;

        double ratio = (double) Math.min(positiveCount, negativeCount) /
                       Math.max(positiveCount, negativeCount);

        logger.info("Dataset balance: {} positive, {} negative (ratio: {:.2f})",
                positiveCount, negativeCount, ratio);

        assertTrue(ratio >= 0.3,
                String.format("Dataset is too imbalanced (ratio: %.2f). " +
                        "Minimum ratio 0.3 required for valid accuracy measurement.", ratio));
    }

    private record MisclassificationRecord(
            String text,
            String expected,
            String predicted,
            double confidence
    ) {}

    private static String truncate(String text, int maxLength) {
        if (text == null) return "";
        if (text.length() <= maxLength) return text;
        return text.substring(0, maxLength - 3) + "...";
    }
}
