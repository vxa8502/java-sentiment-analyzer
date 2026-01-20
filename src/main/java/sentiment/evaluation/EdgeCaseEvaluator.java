package sentiment.evaluation;

import sentiment.config.EdgeCaseConfig;
import sentiment.data.Dataset;
import sentiment.data.SimpleDatasetLoader;
import sentiment.models.ModelLoader;
import sentiment.models.SentimentClassifier;
import sentiment.training.TrainingMetadata;
import sentiment.training.TrainingMetadata.EdgeCaseEvaluationResult;
import sentiment.training.TrainingMetadata.CategoryPerformance;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
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

    // Default values (used if no config provided)
    private static final String DEFAULT_EDGE_CASES_DIR = "data/raw/edge_cases";
    private static final String[] DEFAULT_EDGE_CASE_TYPES = {
        "sarcasm", "mixed_sentiment", "negation_heavy", "domain_jargon"
    };

    // Instance config (null means use defaults)
    private final EdgeCaseConfig config;

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
            sb.append("─".repeat(75)).append("\n");
            sb.append(String.format("%-18s %8s %8s %10s %18s%n",
                "Category", "Correct", "Total", "Accuracy", "95% CI"));
            sb.append("─".repeat(75)).append("\n");

            // Sort by accuracy (ascending) to show worst categories first
            results.entrySet().stream()
                .sorted(Comparator.comparingDouble(e -> e.getValue().accuracy))
                .forEach(entry -> {
                    EdgeCaseResult result = entry.getValue();
                    // Calculate Wilson CI for display
                    CategoryPerformance perf = CategoryPerformance.create(
                        result.correct, result.totalSamples, 0.95);
                    sb.append(String.format("%-18s %8d %8d %9.1f%% %7.1f%% - %.1f%%%n",
                        result.category,
                        result.correct,
                        result.totalSamples,
                        result.accuracy * 100,
                        perf.ciLower * 100,
                        perf.ciUpper * 100));
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

    // Constructors

    /**
     * Create evaluator with default config.
     */
    public EdgeCaseEvaluator() {
        this.config = null;
    }

    /**
     * Create evaluator with specified config.
     */
    public EdgeCaseEvaluator(EdgeCaseConfig config) {
        this.config = config;
    }

    // Config accessors

    private String getEdgeCasesDir() {
        return config != null ? config.getDirectories().getEdgeCasesDir() : DEFAULT_EDGE_CASES_DIR;
    }

    private String[] getEdgeCaseTypes() {
        return config != null ? config.getCategoryIds() : DEFAULT_EDGE_CASE_TYPES;
    }

    private double getConfidenceLevel() {
        return config != null ? config.getEvaluation().getConfidenceLevel() : 0.95;
    }

    private int getWarnThreshold() {
        return config != null ? config.getEvaluation().getWarnIfSamplesBelow() : 30;
    }

    /**
     * Evaluate model on all edge case categories (instance method using config)
     */
    public EdgeCaseReport evaluateModel(SentimentClassifier model, String modelName) {
        Map<String, EdgeCaseResult> results = new LinkedHashMap<>();
        String edgeCasesDir = getEdgeCasesDir();
        String[] edgeCaseTypes = getEdgeCaseTypes();
        int warnThreshold = getWarnThreshold();

        for (String category : edgeCaseTypes) {
            Path edgeCaseFile = Paths.get(edgeCasesDir, category + ".csv");

            if (!Files.exists(edgeCaseFile)) {
                System.err.println("  [WARN] Edge case file not found: " + edgeCaseFile);
                continue;
            }

            try {
                SimpleDatasetLoader loader = new SimpleDatasetLoader();
                List<Dataset> edgeCases = loader.load(edgeCaseFile.toString());

                if (edgeCases.isEmpty()) {
                    System.err.println("  [WARN] No edge cases loaded from: " + edgeCaseFile);
                    continue;
                }

                // Warn if sample size is too small
                if (edgeCases.size() < warnThreshold) {
                    System.err.printf("  [WARN] %s has only %d samples (recommend %d+ for reliable CI)%n",
                        category, edgeCases.size(), warnThreshold);
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
                            double[] probs = model.getClassificationProbabilities(sample.getText());
                            double confidence = Arrays.stream(probs).max().orElse(0.0);
                            failures.add(new PredictionFailure(sample, predicted, actual, confidence));
                        }
                    } catch (Exception e) {
                        failures.add(new PredictionFailure(sample, "ERROR", sample.getSentiment().name(), 0.0));
                    }
                }

                EdgeCaseResult result = new EdgeCaseResult(category, edgeCases.size(), correct, failures);
                results.put(category, result);

                System.out.printf("  [OK] %s: %d/%d correct (%.1f%%)%n",
                    category, correct, edgeCases.size(), result.accuracy * 100);

            } catch (Exception e) {
                System.err.println("  [ERR] Failed to evaluate " + category + ": " + e.getMessage());
            }
        }

        return new EdgeCaseReport(modelName, results);
    }

    /**
     * Convert report to persistable metadata format.
     */
    public EdgeCaseEvaluationResult toMetadataFormat(EdgeCaseReport report) {
        EdgeCaseEvaluationResult result = new EdgeCaseEvaluationResult();
        result.evaluatedAt = Instant.now().toString();
        result.configVersion = config != null ? config.getVersion() : "default";
        result.confidenceLevel = getConfidenceLevel();

        // Overall stats
        int totalSamples = report.results.values().stream().mapToInt(r -> r.totalSamples).sum();
        int totalCorrect = report.results.values().stream().mapToInt(r -> r.correct).sum();
        result.overallSamples = totalSamples;
        result.overallCorrect = totalCorrect;
        result.overallAccuracy = totalSamples > 0 ? (double) totalCorrect / totalSamples : 0.0;

        // Per-category with confidence intervals
        result.categories = new LinkedHashMap<>();
        for (Map.Entry<String, EdgeCaseResult> entry : report.results.entrySet()) {
            EdgeCaseResult ecr = entry.getValue();
            CategoryPerformance perf = CategoryPerformance.create(
                ecr.correct, ecr.totalSamples, getConfidenceLevel());
            result.categories.put(entry.getKey(), perf);
        }

        return result;
    }

    /**
     * Persist edge case results to model metadata file.
     */
    public void persistToMetadata(Path metadataPath, EdgeCaseReport report) throws IOException {
        EdgeCaseEvaluationResult evalResult = toMetadataFormat(report);

        // Load existing metadata
        TrainingMetadata metadata = TrainingMetadata.load(metadataPath);

        // Update with edge case results
        metadata.setEdgeCasePerformance(evalResult);

        // Save back
        metadata.save(metadataPath);

        System.out.println("  [OK] Persisted edge case results to: " + metadataPath);
    }

    /**
     * Evaluate model on all edge case categories (static method for backwards compatibility)
     */
    public static EdgeCaseReport evaluate(SentimentClassifier model, String modelName) {
        // Use default evaluator for backwards compatibility
        return new EdgeCaseEvaluator().evaluateModel(model, modelName);
    }

    /**
     * CLI entry point
     */
    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: java EdgeCaseEvaluator <algorithm> <domain> [--persist]");
            System.err.println();
            System.err.println("Arguments:");
            System.err.println("  algorithm  Model algorithm (svm, naive_bayes, random_forest, logistic_regression)");
            System.err.println("  domain     Training domain (imdb_50k, amazon_polarity, yelp)");
            System.err.println("  --persist  Save results to model metadata file");
            System.err.println();
            System.err.println("Config: Reads from config/edge-case-evaluation.json (or EDGE_CASE_CONFIG env var)");
            System.err.println();
            System.err.println("Examples:");
            System.err.println("  java EdgeCaseEvaluator svm imdb_50k");
            System.err.println("  java EdgeCaseEvaluator logistic_regression yelp --persist");
            System.exit(1);
        }

        String algorithm = args[0];
        String domain = args[1];
        boolean persist = args.length > 2 && args[2].equals("--persist");

        try {
            // Load config
            EdgeCaseConfig config;
            try {
                config = EdgeCaseConfig.load();
                System.out.println("Config loaded: " + config);
            } catch (IOException e) {
                System.out.println("No config file found, using defaults");
                config = EdgeCaseConfig.defaults();
            }

            // Create evaluator with config
            EdgeCaseEvaluator evaluator = new EdgeCaseEvaluator(config);

            // Load model
            String modelsDir = config.getDirectories().getModelsDir();
            String modelPath = String.format("%s/%s/%s_%s_model.ser", modelsDir, algorithm, domain, algorithm);
            System.out.println("\nLoading " + algorithm + " model trained on " + domain + "...");
            SentimentClassifier model = ModelLoader.loadWithMetadata(modelPath);
            System.out.println("  [OK] Model loaded\n");

            // Evaluate on edge cases
            System.out.println("Evaluating on edge case challenge sets:");
            String modelName = algorithm + " (trained on " + domain + ")";
            EdgeCaseReport report = evaluator.evaluateModel(model, modelName);

            // Print report
            System.out.println(report.generateReport());

            // Print confidence intervals
            System.out.println("\nConfidence Intervals (Wilson score, " +
                String.format("%.0f%%", config.getEvaluation().getConfidenceLevel() * 100) + " CI):");
            System.out.println("─".repeat(65));
            System.out.printf("%-20s %10s %15s %15s%n", "Category", "Accuracy", "CI Lower", "CI Upper");
            System.out.println("─".repeat(65));

            EdgeCaseEvaluationResult evalResult = evaluator.toMetadataFormat(report);
            for (Map.Entry<String, CategoryPerformance> entry : evalResult.categories.entrySet()) {
                CategoryPerformance perf = entry.getValue();
                System.out.printf("%-20s %9.1f%% %14.1f%% %14.1f%%%n",
                    entry.getKey(),
                    perf.accuracy * 100,
                    perf.ciLower * 100,
                    perf.ciUpper * 100);
            }
            System.out.println();

            // Persist if requested
            if (persist) {
                Path metadataPath = Paths.get(modelsDir, algorithm,
                    domain + "_" + algorithm + "_model.metadata.json");

                if (Files.exists(metadataPath)) {
                    evaluator.persistToMetadata(metadataPath, report);
                } else {
                    System.err.println("  [WARN] Metadata file not found: " + metadataPath);
                    System.err.println("         Edge case results NOT persisted.");
                }
            }

        } catch (Exception e) {
            System.err.println("Error during evaluation: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
