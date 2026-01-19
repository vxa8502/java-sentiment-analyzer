package sentiment.evaluation;

import sentiment.data.Dataset;
import sentiment.data.SimpleDatasetLoader;
import sentiment.models.ModelLoader;
import sentiment.models.SentimentClassifier;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Evaluates models on edge case challenge sets following Sofia's recommendations.
 * Tests model robustness on sarcasm, negation, mixed sentiment, and domain jargon.
 *
 * Sofia's warning: "If your model gets 95% accuracy on clean test data but bombs on
 * edge cases, it's memorizing patterns, not learning sentiment."
 *
 * @author Victoria Alabi
 */
public class EdgeCaseEvaluator {

    private static final String EDGE_CASES_DIR = "data/raw/edge_cases";
    private static final String[] EDGE_CASE_TYPES = {
        "sarcasm", "mixed_sentiment", "negation_heavy", "domain_jargon"
    };

    /**
     * Result for a single edge case category
     */
    public static class EdgeCaseResult {
        public final String category;
        public final int totalSamples;
        public final int correct;
        public final double accuracy;
        public final List<PredictionFailure> failures;

        public EdgeCaseResult(String category, int totalSamples, int correct,
                            List<PredictionFailure> failures) {
            this.category = category;
            this.totalSamples = totalSamples;
            this.correct = correct;
            this.accuracy = totalSamples > 0 ? (double) correct / totalSamples : 0.0;
            this.failures = failures;
        }
    }

    /**
     * Single prediction failure on an edge case
     */
    public static class PredictionFailure {
        public final Dataset sample;
        public final String predicted;
        public final String actual;
        public final double confidence;

        public PredictionFailure(Dataset sample, String predicted, String actual, double confidence) {
            this.sample = sample;
            this.predicted = predicted;
            this.actual = actual;
            this.confidence = confidence;
        }
    }

    /**
     * Complete edge case evaluation report
     */
    public static class EdgeCaseReport {
        public final String modelName;
        public final Map<String, EdgeCaseResult> results;
        public final double overallAccuracy;

        public EdgeCaseReport(String modelName, Map<String, EdgeCaseResult> results) {
            this.modelName = modelName;
            this.results = results;

            // Calculate overall accuracy
            int totalSamples = results.values().stream().mapToInt(r -> r.totalSamples).sum();
            int totalCorrect = results.values().stream().mapToInt(r -> r.correct).sum();
            this.overallAccuracy = totalSamples > 0 ? (double) totalCorrect / totalSamples : 0.0;
        }

        /**
         * Generate formatted console report
         */
        public String generateReport() {
            StringBuilder sb = new StringBuilder();

            sb.append("\n╔══════════════════════════════════════════════════════════════╗\n");
            sb.append("║            EDGE CASE EVALUATION REPORT                       ║\n");
            sb.append("╚══════════════════════════════════════════════════════════════╝\n\n");

            sb.append(String.format("Model: %s%n", modelName));
            sb.append(String.format("Overall Edge Case Accuracy: %.2f%%%n%n", overallAccuracy * 100));

            sb.append("Performance by Category:\n");
            sb.append("─".repeat(65)).append("\n");
            sb.append(String.format("%-20s %10s %10s %12s%n", "Category", "Correct", "Total", "Accuracy"));
            sb.append("─".repeat(65)).append("\n");

            // Sort by accuracy (ascending) to show worst categories first
            results.entrySet().stream()
                .sorted(Comparator.comparingDouble(e -> e.getValue().accuracy))
                .forEach(entry -> {
                    EdgeCaseResult result = entry.getValue();
                    sb.append(String.format("%-20s %10d %10d %11.2f%%%n",
                        result.category,
                        result.correct,
                        result.totalSamples,
                        result.accuracy * 100));
                });

            sb.append("\n");
            sb.append("Expected Performance (Sofia's baseline):\n");
            sb.append("  Sarcasm:         50-60% (genuinely hard)\n");
            sb.append("  Mixed Sentiment: 60-70% (depends on neutral labeling)\n");
            sb.append("  Negation:        75-85% (grammar helps)\n");
            sb.append("  Domain Jargon:   70-80% (depends on training overlap)\n");
            sb.append("  Overall:         >70% (model is robust)\n\n");

            // Show worst failures for each category
            sb.append("Sample Failures by Category:\n");
            sb.append("─".repeat(65)).append("\n");

            for (EdgeCaseResult result : results.values()) {
                if (result.failures.isEmpty()) continue;

                sb.append(String.format("%n%s (%d failures):%n", result.category, result.failures.size()));

                // Show up to 3 failures per category
                result.failures.stream()
                    .limit(3)
                    .forEach(failure -> {
                        sb.append(String.format("  • Predicted: %-10s | Actual: %-10s | Conf: %.3f%n",
                            failure.predicted, failure.actual, failure.confidence));
                        sb.append(String.format("    \"%s\"%n",
                            failure.sample.getText().length() > 100
                                ? failure.sample.getText().substring(0, 97) + "..."
                                : failure.sample.getText()));
                    });
            }

            sb.append("\n");
            sb.append("═".repeat(65)).append("\n");

            // Warning if overall accuracy is too low
            if (overallAccuracy < 0.7) {
                sb.append("\n⚠ WARNING: Overall edge case accuracy is below 70%!\n");
                sb.append("This model may be brittle and could fail on real-world data.\n");
                sb.append("Consider retraining with more diverse examples or using a different algorithm.\n");
            }

            return sb.toString();
        }
    }

    /**
     * Evaluate model on all edge case categories
     */
    public static EdgeCaseReport evaluate(SentimentClassifier model, String modelName) {
        Map<String, EdgeCaseResult> results = new LinkedHashMap<>();

        for (String category : EDGE_CASE_TYPES) {
            Path edgeCaseFile = Paths.get(EDGE_CASES_DIR, category + ".csv");

            if (!Files.exists(edgeCaseFile)) {
                System.err.println("⚠ Edge case file not found: " + edgeCaseFile);
                continue;
            }

            try {
                SimpleDatasetLoader loader = new SimpleDatasetLoader();
                List<Dataset> edgeCases = loader.load(edgeCaseFile.toString());

                if (edgeCases.isEmpty()) {
                    System.err.println("⚠ No edge cases loaded from: " + edgeCaseFile);
                    continue;
                }

                int correct = 0;
                List<PredictionFailure> failures = new ArrayList<>();

                for (Dataset sample : edgeCases) {
                    try {
                        String predicted = model.classify(sample.getText());
                        String actual = sample.getSentiment().name();

                        if (predicted.equalsIgnoreCase(actual)) {
                            correct++;
                        } else {
                            // Get confidence for failure analysis
                            double[] probs = model.getClassificationProbabilities(sample.getText());
                            double confidence = Arrays.stream(probs).max().orElse(0.0);
                            failures.add(new PredictionFailure(sample, predicted, actual, confidence));
                        }
                    } catch (Exception e) {
                        // Prediction failed, count as incorrect
                        failures.add(new PredictionFailure(sample, "ERROR", sample.getSentiment().name(), 0.0));
                    }
                }

                EdgeCaseResult result = new EdgeCaseResult(category, edgeCases.size(), correct, failures);
                results.put(category, result);

                System.out.printf("✓ Evaluated %s: %d/%d correct (%.2f%%)%n",
                    category, correct, edgeCases.size(), result.accuracy * 100);

            } catch (Exception e) {
                System.err.println("✗ Failed to evaluate " + category + ": " + e.getMessage());
            }
        }

        return new EdgeCaseReport(modelName, results);
    }

    /**
     * CLI entry point
     */
    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: java EdgeCaseEvaluator <algorithm> <domain>");
            System.err.println();
            System.err.println("Arguments:");
            System.err.println("  algorithm  Model algorithm (svm, naive_bayes, random_forest, logistic_regression)");
            System.err.println("  domain     Training domain (imdb_50k, amazon_polarity, yelp)");
            System.err.println();
            System.err.println("Examples:");
            System.err.println("  java EdgeCaseEvaluator svm imdb_50k");
            System.err.println("  java EdgeCaseEvaluator logistic_regression yelp");
            System.exit(1);
        }

        String algorithm = args[0];
        String domain = args[1];

        try {
            // Load model
            System.out.println("Loading " + algorithm + " model trained on " + domain + "...");
            String modelPath = String.format("models/%s/%s_%s_model.ser", algorithm, domain, algorithm);
            SentimentClassifier model = ModelLoader.loadWithMetadata(modelPath);
            System.out.println("✓ Model loaded\n");

            // Evaluate on edge cases
            System.out.println("Evaluating on edge case challenge sets...\n");
            String modelName = algorithm + " (trained on " + domain + ")";
            EdgeCaseReport report = EdgeCaseEvaluator.evaluate(model, modelName);

            // Print report
            System.out.println(report.generateReport());

        } catch (Exception e) {
            System.err.println("Error during evaluation: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
