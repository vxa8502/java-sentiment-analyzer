package sentiment.evaluation;

import sentiment.data.Dataset;
import sentiment.data.SimpleDatasetLoader;
import sentiment.models.ModelLoader;
import sentiment.models.SentimentClassifier;

import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Analyzes model prediction errors to identify failure patterns.
 * Following Sofia's method: find real failures, then manually categorize them.
 *
 * Usage:
 * 1. Run this tool to find misclassifications
 * 2. Review worst predictions (high confidence errors)
 * 3. Manually categorize them into edge case sets
 * 4. Add to datasets/edge_cases/*.csv
 *
 * @author Victoria Alabi
 */
public class ErrorAnalyzer {

    /**
     * Result of a single prediction error
     */
    public static class PredictionError {
        public final Dataset sample;
        public final String predictedLabel;
        public final String actualLabel;
        public final double confidence;
        public final double[] probabilities;

        public PredictionError(Dataset sample, String predictedLabel, String actualLabel,
                             double confidence, double[] probabilities) {
            this.sample = sample;
            this.predictedLabel = predictedLabel;
            this.actualLabel = actualLabel;
            this.confidence = confidence;
            this.probabilities = probabilities;
        }

        /**
         * Check if this is a high-confidence error (model was very wrong)
         */
        public boolean isHighConfidenceError(double threshold) {
            return confidence >= threshold;
        }

        /**
         * Format for CSV export
         */
        public String toCsvRow() {
            // Escape quotes in text
            String escapedText = sample.getText().replace("\"", "\"\"");
            return String.format("\"%s\",%s,%s,%.4f",
                escapedText, actualLabel, predictedLabel, confidence);
        }

        @Override
        public String toString() {
            return String.format("Error[conf=%.3f, pred=%s, actual=%s]: \"%s\"",
                confidence, predictedLabel, actualLabel,
                sample.getText().length() > 80
                    ? sample.getText().substring(0, 77) + "..."
                    : sample.getText());
        }
    }

    /**
     * Complete error analysis report
     */
    public static class ErrorAnalysisReport {
        public final String modelName;
        public final String testDataName;
        public final int totalSamples;
        public final int totalErrors;
        public final double errorRate;
        public final List<PredictionError> allErrors;

        public ErrorAnalysisReport(String modelName, String testDataName,
                                 int totalSamples, List<PredictionError> allErrors) {
            this.modelName = modelName;
            this.testDataName = testDataName;
            this.totalSamples = totalSamples;
            this.allErrors = allErrors;
            this.totalErrors = allErrors.size();
            this.errorRate = totalSamples > 0 ? (double) totalErrors / totalSamples : 0.0;
        }

        /**
         * Get top N worst predictions (highest confidence errors)
         */
        public List<PredictionError> getWorstPredictions(int limit) {
            return allErrors.stream()
                .sorted(Comparator.comparingDouble((PredictionError e) -> e.confidence).reversed())
                .limit(limit)
                .collect(Collectors.toList());
        }

        /**
         * Get high confidence errors (confidence >= threshold)
         */
        public List<PredictionError> getHighConfidenceErrors(double threshold) {
            return allErrors.stream()
                .filter(e -> e.confidence >= threshold)
                .sorted(Comparator.comparingDouble((PredictionError e) -> e.confidence).reversed())
                .collect(Collectors.toList());
        }

        /**
         * Export errors to CSV for manual review
         */
        public String exportToCsv() {
            StringBuilder sb = new StringBuilder();
            sb.append("text,actual_sentiment,predicted_sentiment,confidence\n");

            for (PredictionError error : allErrors) {
                sb.append(error.toCsvRow()).append("\n");
            }

            return sb.toString();
        }

        /**
         * Generate formatted console report
         */
        public String generateReport(int topN) {
            StringBuilder sb = new StringBuilder();

            sb.append("\n╔══════════════════════════════════════════════════════════════╗\n");
            sb.append("║               ERROR ANALYSIS REPORT                          ║\n");
            sb.append("╚══════════════════════════════════════════════════════════════╝\n\n");

            sb.append(String.format("Model:             %s%n", modelName));
            sb.append(String.format("Test Data:         %s%n", testDataName));
            sb.append(String.format("Total Samples:     %d%n", totalSamples));
            sb.append(String.format("Total Errors:      %d%n", totalErrors));
            sb.append(String.format("Error Rate:        %.2f%%%n", errorRate * 100));
            sb.append(String.format("Accuracy:          %.2f%%%n%n", (1 - errorRate) * 100));

            // High confidence errors breakdown
            long highConfErrors = getHighConfidenceErrors(0.8).size();
            long medConfErrors = allErrors.stream()
                .filter(e -> e.confidence >= 0.6 && e.confidence < 0.8).count();
            long lowConfErrors = allErrors.stream()
                .filter(e -> e.confidence < 0.6).count();

            sb.append("Error Breakdown by Confidence:\n");
            sb.append("─".repeat(65)).append("\n");
            sb.append(String.format("  High (≥0.80):      %5d (%.1f%% of errors)%n",
                highConfErrors, (double) highConfErrors / totalErrors * 100));
            sb.append(String.format("  Medium (0.60-0.80): %5d (%.1f%% of errors)%n",
                medConfErrors, (double) medConfErrors / totalErrors * 100));
            sb.append(String.format("  Low (<0.60):       %5d (%.1f%% of errors)%n%n",
                lowConfErrors, (double) lowConfErrors / totalErrors * 100));

            // Top N worst predictions
            sb.append(String.format("Top %d Worst Predictions:%n", topN));
            sb.append("─".repeat(65)).append("\n");

            List<PredictionError> worst = getWorstPredictions(topN);
            for (int i = 0; i < worst.size(); i++) {
                PredictionError error = worst.get(i);
                sb.append(String.format("%n%d. Confidence: %.3f%n", i + 1, error.confidence));
                sb.append(String.format("   Predicted: %-10s | Actual: %-10s%n",
                    error.predictedLabel, error.actualLabel));
                sb.append(String.format("   Text: \"%s\"%n",
                    error.sample.getText().length() > 150
                        ? error.sample.getText().substring(0, 147) + "..."
                        : error.sample.getText()));
            }

            sb.append("\n");
            sb.append("═".repeat(65)).append("\n");
            sb.append("\nNext Steps:\n");
            sb.append("1. Review the worst predictions above\n");
            sb.append("2. Manually categorize them (sarcasm, negation, mixed, jargon)\n");
            sb.append("3. Add to datasets/edge_cases/*.csv\n");
            sb.append("4. Use --export to save all errors to CSV for batch review\n");

            return sb.toString();
        }
    }

    /**
     * Analyze prediction errors on a test dataset
     *
     * @param model trained classifier
     * @param testData test samples
     * @return error analysis report
     */
    public static ErrorAnalysisReport analyze(SentimentClassifier model, List<Dataset> testData,
                                             String modelName, String testDataName) {
        List<PredictionError> errors = new ArrayList<>();

        for (Dataset sample : testData) {
            try {
                String predicted = model.classify(sample.getText());
                String actual = sample.getSentiment().name();

                // Check if prediction is wrong
                if (!predicted.equalsIgnoreCase(actual)) {
                    double[] probs = model.getClassificationProbabilities(sample.getText());

                    // Find confidence (max probability)
                    double confidence = Arrays.stream(probs).max().orElse(0.0);

                    errors.add(new PredictionError(sample, predicted, actual, confidence, probs));
                }
            } catch (Exception e) {
                // Prediction failed - skip this sample
            }
        }

        return new ErrorAnalysisReport(modelName, testDataName, testData.size(), errors);
    }

    /**
     * CLI entry point for error analysis
     */
    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println("Usage: java ErrorAnalyzer <algorithm> <domain> <test-data-path> [--export] [--top-n N]");
            System.err.println();
            System.err.println("Arguments:");
            System.err.println("  algorithm      Model algorithm (svm, naive_bayes, random_forest, logistic_regression)");
            System.err.println("  domain         Training domain (imdb_50k, amazon_polarity, yelp)");
            System.err.println("  test-data-path Path to test data CSV");
            System.err.println("  --export       Export all errors to CSV (optional)");
            System.err.println("  --top-n N      Show top N worst predictions (default: 20)");
            System.err.println();
            System.err.println("Examples:");
            System.err.println("  java ErrorAnalyzer svm imdb_50k data/processed/imdb_50k/test.csv");
            System.err.println("  java ErrorAnalyzer logistic_regression yelp data/processed/yelp/test.csv --top-n 50");
            System.err.println("  java ErrorAnalyzer naive_bayes amazon_polarity data/processed/amazon_polarity/test.csv --export");
            System.exit(1);
        }

        String algorithm = args[0];
        String domain = args[1];
        String testDataPath = args[2];
        boolean export = Arrays.asList(args).contains("--export");
        int topN = 20;

        // Parse --top-n argument
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals("--top-n")) {
                topN = Integer.parseInt(args[i + 1]);
                break;
            }
        }

        try {
            // Load model
            System.out.println("Loading " + algorithm + " model trained on " + domain + "...");
            String modelPath = String.format("models/%s/%s_%s_model.ser", algorithm, domain, algorithm);
            SentimentClassifier model = ModelLoader.loadWithMetadata(modelPath);
            System.out.println("✓ Model loaded\n");

            // Load test data
            System.out.println("Loading test data from: " + testDataPath);
            SimpleDatasetLoader loader = new SimpleDatasetLoader();
            List<Dataset> testData = loader.load(testDataPath);
            System.out.println("✓ Loaded " + testData.size() + " test samples\n");

            // Analyze errors
            System.out.println("Analyzing prediction errors...");
            String modelName = algorithm + " (trained on " + domain + ")";
            ErrorAnalysisReport report = ErrorAnalyzer.analyze(model, testData, modelName, testDataPath);

            // Print report
            System.out.println(report.generateReport(topN));

            // Export to CSV if requested
            if (export) {
                String outputFile = String.format("error_analysis_%s_%s.csv", algorithm, domain);
                java.nio.file.Files.writeString(Paths.get(outputFile), report.exportToCsv());
                System.out.println("\n✓ Exported all " + report.totalErrors + " errors to: " + outputFile);
            }

        } catch (Exception e) {
            System.err.println("Error during analysis: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
