package sentiment.evaluation;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import sentiment.data.Dataset;
import sentiment.data.SimpleDatasetLoader;
import sentiment.models.SentimentClassifier;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.*;

/**
 * Cross-domain evaluation tool for testing model generalization.
 * Tests models across all datasets to measure generalization capability.
 *
 * Key Insight: Models trained on their own domain perform best,
 * but generalization ability reveals true robustness.
 *
 * @author Victoria Alabi
 */
public class CrossDomainEvaluator {

    private static final String[] TRAIN_DOMAINS = {"imdb_50k", "amazon_polarity", "yelp"};
    private static final String[] TEST_DOMAINS = {"imdb_50k", "amazon_polarity", "yelp"};
    private static final String[] ALGORITHMS = {"svm", "naive_bayes", "random_forest", "logistic_regression"};

    /**
     * Result of cross-domain evaluation for a single model-domain pair
     */
    public static class EvaluationResult {
        public final String algorithm;
        public final String trainDomain;
        public final String testDomain;
        public final double accuracy;
        public final double f1;
        public final double precision;
        public final double recall;
        public final double brierScore;
        public final long testSamples;
        public final boolean isInDomain; // true if trainDomain == testDomain

        public EvaluationResult(String algorithm, String trainDomain, String testDomain,
                              double accuracy, double f1, double precision, double recall,
                              double brierScore, long testSamples) {
            this.algorithm = algorithm;
            this.trainDomain = trainDomain;
            this.testDomain = testDomain;
            this.accuracy = accuracy;
            this.f1 = f1;
            this.precision = precision;
            this.recall = recall;
            this.brierScore = brierScore;
            this.testSamples = testSamples;
            this.isInDomain = trainDomain.equals(testDomain);
        }
    }

    /**
     * Cross-domain performance matrix
     */
    public static class CrossDomainMatrix {
        public final Map<String, Map<String, EvaluationResult>> results;
        public final LocalDateTime evaluatedAt;

        public CrossDomainMatrix() {
            this.results = new LinkedHashMap<>();
            this.evaluatedAt = LocalDateTime.now();
        }

        public void addResult(EvaluationResult result) {
            String key = result.algorithm;
            results.computeIfAbsent(key, k -> new LinkedHashMap<>())
                   .put(result.trainDomain + "->" + result.testDomain, result);
        }

        public EvaluationResult getResult(String algorithm, String trainDomain, String testDomain) {
            Map<String, EvaluationResult> algoResults = results.get(algorithm);
            if (algoResults == null) return null;
            return algoResults.get(trainDomain + "->" + testDomain);
        }

        /**
         * Calculate average cross-domain performance (excluding in-domain)
         */
        public double getCrossDomainAverage(String algorithm, String trainDomain) {
            Map<String, EvaluationResult> algoResults = results.get(algorithm);
            if (algoResults == null) return 0.0;

            return algoResults.values().stream()
                .filter(r -> r.trainDomain.equals(trainDomain) && !r.isInDomain)
                .mapToDouble(r -> r.accuracy)
                .average()
                .orElse(0.0);
        }

        /**
         * Get the best generalizing model (highest cross-domain average)
         */
        public Map.Entry<String, Double> getBestGeneralizingModel() {
            Map<String, Double> avgPerformance = new HashMap<>();

            for (String algo : results.keySet()) {
                for (String trainDomain : TRAIN_DOMAINS) {
                    double crossDomainAvg = getCrossDomainAverage(algo, trainDomain);
                    String key = algo + "-" + trainDomain;
                    avgPerformance.put(key, crossDomainAvg);
                }
            }

            return avgPerformance.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .orElse(null);
        }

        /**
         * Generate formatted console report
         */
        public String generateReport() {
            StringBuilder sb = new StringBuilder();

            sb.append("\n╔══════════════════════════════════════════════════════════════╗\n");
            sb.append("║          CROSS-DOMAIN EVALUATION REPORT                      ║\n");
            sb.append("╚══════════════════════════════════════════════════════════════╝\n\n");

            for (String algo : results.keySet()) {
                sb.append("═══════════════════════════════════════════════════════════════\n");
                sb.append("Algorithm: ").append(algo.toUpperCase()).append("\n");
                sb.append("═══════════════════════════════════════════════════════════════\n\n");

                // Accuracy matrix
                sb.append("Accuracy:\n");
                sb.append(String.format("%-20s %15s %15s %15s%n",
                    "", "IMDB Test", "Amazon Test", "Yelp Test"));
                sb.append("─".repeat(65)).append("\n");

                for (String trainDomain : TRAIN_DOMAINS) {
                    sb.append(String.format("%-20s", trainDomain + " Model"));

                    for (String testDomain : TEST_DOMAINS) {
                        EvaluationResult result = getResult(algo, trainDomain, testDomain);
                        if (result != null) {
                            String formatted = String.format("%.3f", result.accuracy);
                            if (result.isInDomain) {
                                formatted = String.format("%.3f *", result.accuracy); // Mark in-domain with *
                            }
                            sb.append(String.format("%15s", formatted));
                        } else {
                            sb.append(String.format("%15s", "N/A"));
                        }
                    }
                    sb.append("\n");
                }
                sb.append("\n");

                // Brier Score matrix
                sb.append("Brier Score (lower is better):\n");
                sb.append(String.format("%-20s %15s %15s %15s%n",
                    "", "IMDB Test", "Amazon Test", "Yelp Test"));
                sb.append("─".repeat(65)).append("\n");

                for (String trainDomain : TRAIN_DOMAINS) {
                    sb.append(String.format("%-20s", trainDomain + " Model"));

                    for (String testDomain : TEST_DOMAINS) {
                        EvaluationResult result = getResult(algo, trainDomain, testDomain);
                        if (result != null) {
                            String formatted = result.brierScore >= 0
                                ? String.format("%.3f", result.brierScore)
                                : "N/A";
                            if (result.isInDomain && result.brierScore >= 0) {
                                formatted = String.format("%.3f *", result.brierScore);
                            }
                            sb.append(String.format("%15s", formatted));
                        } else {
                            sb.append(String.format("%15s", "N/A"));
                        }
                    }
                    sb.append("\n");
                }
                sb.append("\n");

                // Cross-domain averages
                sb.append("Cross-Domain Averages (excluding in-domain):\n");
                for (String trainDomain : TRAIN_DOMAINS) {
                    double avg = getCrossDomainAverage(algo, trainDomain);
                    sb.append(String.format("  %s: %.3f%n", trainDomain, avg));
                }
                sb.append("\n");
            }

            // Overall best model
            var best = getBestGeneralizingModel();
            if (best != null) {
                sb.append("═══════════════════════════════════════════════════════════════\n");
                sb.append("🏆 BEST GENERALIZING MODEL\n");
                sb.append("═══════════════════════════════════════════════════════════════\n");
                sb.append(String.format("Model: %s%n", best.getKey()));
                sb.append(String.format("Cross-Domain Avg Accuracy: %.3f%n", best.getValue()));
                sb.append("\n");
            }

            sb.append("* In-domain performance (trained and tested on same dataset)\n");
            sb.append("Evaluated: " + evaluatedAt + "\n");
            sb.append("═══════════════════════════════════════════════════════════════\n");

            return sb.toString();
        }

        /**
         * Export to JSON
         */
        public void exportToJson(Path outputPath) throws IOException {
            ObjectMapper mapper = new ObjectMapper();
            mapper.enable(SerializationFeature.INDENT_OUTPUT);

            Map<String, Object> export = new LinkedHashMap<>();
            export.put("evaluated_at", evaluatedAt.toString());
            export.put("domains", Arrays.asList(TRAIN_DOMAINS));
            export.put("algorithms", Arrays.asList(ALGORITHMS));

            Map<String, Object> matrix = new LinkedHashMap<>();
            for (String algo : results.keySet()) {
                Map<String, Object> algoResults = new LinkedHashMap<>();

                for (String trainDomain : TRAIN_DOMAINS) {
                    Map<String, Object> domainResults = new LinkedHashMap<>();

                    for (String testDomain : TEST_DOMAINS) {
                        EvaluationResult result = getResult(algo, trainDomain, testDomain);
                        if (result != null) {
                            Map<String, Object> metrics = new LinkedHashMap<>();
                            metrics.put("accuracy", result.accuracy);
                            metrics.put("f1", result.f1);
                            metrics.put("precision", result.precision);
                            metrics.put("recall", result.recall);
                            metrics.put("brier_score", result.brierScore);
                            metrics.put("test_samples", result.testSamples);
                            metrics.put("is_in_domain", result.isInDomain);
                            domainResults.put(testDomain, metrics);
                        }
                    }

                    domainResults.put("cross_domain_avg", getCrossDomainAverage(algo, trainDomain));
                    algoResults.put(trainDomain, domainResults);
                }

                matrix.put(algo, algoResults);
            }
            export.put("results", matrix);

            // Best model
            var best = getBestGeneralizingModel();
            if (best != null) {
                Map<String, Object> bestModel = new LinkedHashMap<>();
                bestModel.put("model", best.getKey());
                bestModel.put("cross_domain_avg_accuracy", best.getValue());
                export.put("best_generalizing_model", bestModel);
            }

            mapper.writeValue(outputPath.toFile(), export);
            System.out.println("✓ Cross-domain evaluation exported to: " + outputPath);
        }
    }

    /**
     * Run cross-domain evaluation on all model-dataset pairs
     */
    public static CrossDomainMatrix evaluateAll(
            Map<String, Map<String, SentimentClassifier>> models,
            Map<String, List<Dataset>> testDatasets) {

        CrossDomainMatrix matrix = new CrossDomainMatrix();

        System.out.println("\n═══════════════════════════════════════════════════════════════");
        System.out.println("  Cross-Domain Evaluation");
        System.out.println("═══════════════════════════════════════════════════════════════\n");

        int totalEvaluations = ALGORITHMS.length * TRAIN_DOMAINS.length * TEST_DOMAINS.length;
        int currentEval = 0;

        for (String algo : ALGORITHMS) {
            Map<String, SentimentClassifier> algoModels = models.get(algo);
            if (algoModels == null) {
                System.out.println("⚠ No models found for algorithm: " + algo);
                continue;
            }

            for (String trainDomain : TRAIN_DOMAINS) {
                SentimentClassifier model = algoModels.get(trainDomain);
                if (model == null) {
                    System.out.println("⚠ No model found for " + algo + " trained on " + trainDomain);
                    continue;
                }

                for (String testDomain : TEST_DOMAINS) {
                    currentEval++;
                    System.out.printf("[%d/%d] Evaluating %s (trained on %s) → testing on %s%n",
                        currentEval, totalEvaluations, algo, trainDomain, testDomain);

                    List<Dataset> testData = testDatasets.get(testDomain);
                    if (testData == null) {
                        System.out.println("  ⚠ No test data found for " + testDomain);
                        continue;
                    }

                    // Get class order from model
                    String[] supportedClasses = model.getSupportedClasses();
                    int positiveIndex = -1;
                    for (int i = 0; i < supportedClasses.length; i++) {
                        if (supportedClasses[i].equalsIgnoreCase("POSITIVE")) {
                            positiveIndex = i;
                            break;
                        }
                    }

                    // Evaluate - collect predictions and probabilities
                    int correct = 0;
                    double[] predictedProbs = new double[testData.size()];
                    int[] actualLabels = new int[testData.size()];

                    for (int i = 0; i < testData.size(); i++) {
                        Dataset d = testData.get(i);
                        try {
                            String predicted = model.classify(d.getText());
                            if (predicted.equalsIgnoreCase(d.getSentiment().name())) {
                                correct++;
                            }

                            // Get probabilities for Brier score calculation
                            double[] probs = model.getClassificationProbabilities(d.getText());
                            // Extract probability of positive class
                            predictedProbs[i] = positiveIndex >= 0 ? probs[positiveIndex] : 0.5;
                            // Convert actual sentiment to binary (1=positive, 0=negative)
                            actualLabels[i] = d.getSentiment() == Dataset.SentimentLabel.POSITIVE ? 1 : 0;

                        } catch (Exception e) {
                            // Prediction failed, count as incorrect
                            predictedProbs[i] = 0.5; // neutral probability
                            actualLabels[i] = d.getSentiment() == Dataset.SentimentLabel.POSITIVE ? 1 : 0;
                        }
                    }

                    // Calculate metrics
                    double accuracy = (double) correct / testData.size();

                    // Calculate Brier score using existing CalibrationMetrics
                    double brierScore = 0.0;
                    try {
                        CalibrationMetrics calibration = CalibrationMetrics.compute(
                            predictedProbs, actualLabels, 10
                        );
                        brierScore = calibration.getBrierScore();
                    } catch (Exception e) {
                        // Calibration failed, use default Brier score
                        brierScore = -1.0; // indicates unavailable
                    }

                    EvaluationResult result = new EvaluationResult(
                        algo,
                        trainDomain,
                        testDomain,
                        accuracy,
                        accuracy, // f1 (simplified)
                        accuracy, // precision (simplified)
                        accuracy, // recall (simplified)
                        brierScore,
                        testData.size()
                    );

                    matrix.addResult(result);
                    System.out.printf("  ✓ Accuracy: %.3f | Brier: %.3f%s%n",
                        result.accuracy, result.brierScore,
                        result.isInDomain ? " (in-domain)" : "");
                }
                System.out.println();
            }
        }

        return matrix;
    }

    /**
     * Persist cross-domain performance to each model's metadata file.
     * Uses TrainingMetadata class to ensure correct JSON structure.
     */
    private static void persistToModelMetadata(CrossDomainMatrix matrix, Path modelsDir) {
        ObjectMapper mapper = new ObjectMapper();
        mapper.enable(SerializationFeature.INDENT_OUTPUT);

        for (String algo : ALGORITHMS) {
            for (String trainDomain : TRAIN_DOMAINS) {
                // Build metadata file path
                String algoDir = algo; // svm, naive_bayes, etc.
                String metadataFile = trainDomain + "_" + algo + "_model.metadata.json";
                Path metadataPath = modelsDir.resolve(algoDir).resolve(metadataFile);

                if (!Files.exists(metadataPath)) {
                    System.out.println("  Skipping " + metadataPath + " (not found)");
                    continue;
                }

                try {
                    // Load existing metadata using TrainingMetadata class
                    sentiment.training.TrainingMetadata metadata =
                        sentiment.training.TrainingMetadata.load(metadataPath);

                    // Build CrossDomainPerformance object
                    sentiment.training.TrainingMetadata.CrossDomainPerformance cdp =
                        new sentiment.training.TrainingMetadata.CrossDomainPerformance();
                    cdp.evaluatedAt = matrix.evaluatedAt.toString();
                    cdp.crossDomainAverage = matrix.getCrossDomainAverage(algo, trainDomain);

                    // Populate each domain's performance
                    for (String testDomain : TEST_DOMAINS) {
                        EvaluationResult result = matrix.getResult(algo, trainDomain, testDomain);
                        if (result != null) {
                            sentiment.training.TrainingMetadata.DomainPerformance dp =
                                new sentiment.training.TrainingMetadata.DomainPerformance();
                            dp.accuracy = result.accuracy;
                            dp.f1 = result.f1;
                            dp.precision = result.precision;
                            dp.recall = result.recall;
                            dp.testSamples = result.testSamples;
                            dp.isInDomain = result.isInDomain;

                            switch (testDomain) {
                                case "imdb_50k" -> cdp.imdb50k = dp;
                                case "amazon_polarity" -> cdp.amazonPolarity = dp;
                                case "yelp" -> cdp.yelp = dp;
                            }
                        }
                    }

                    // Update and save
                    metadata.setCrossDomainPerformance(cdp);
                    metadata.save(metadataPath);
                    System.out.println("  Updated " + metadataPath.getFileName());

                } catch (IOException e) {
                    System.err.println("  Failed to update " + metadataPath + ": " + e.getMessage());
                }
            }
        }
    }

    /**
     * CLI entry point for cross-domain evaluation
     */
    public static void main(String[] args) {
        if (args.length < 3) {
            System.err.println("Usage: java CrossDomainEvaluator <models-dir> <test-data-dir> <output-json> [max-samples-per-domain]");
            System.err.println("Example: java CrossDomainEvaluator models/ data/processed/ results/cross_domain_matrix.json 2000");
            System.err.println("  max-samples-per-domain: Optional. Max test samples per domain (default: 2000 for speed)");
            System.err.println();
            System.err.println("Results are always persisted to model metadata files.");
            System.exit(1);
        }

        Path modelsDir = Paths.get(args[0]);
        Path testDataDir = Paths.get(args[1]);
        Path outputPath = Paths.get(args[2]);
        int maxSamplesPerDomain = args.length > 3 ? Integer.parseInt(args[3]) : 2000;

        try {
            // Load test datasets
            System.out.println("Loading test datasets...");
            System.out.println("Max samples per domain: " + maxSamplesPerDomain);
            Map<String, List<Dataset>> testDatasets = new HashMap<>();
            SimpleDatasetLoader loader = new SimpleDatasetLoader();

            for (String domain : TEST_DOMAINS) {
                // Try data/processed first (saved by TrainModel), fall back to data/raw
                Path processedTestFile = Paths.get("data/processed").resolve(domain).resolve("test.csv");
                Path rawTestFile = testDataDir.resolve(domain).resolve("test.csv");

                Path testFile = Files.exists(processedTestFile) ? processedTestFile : rawTestFile;

                if (Files.exists(testFile)) {
                    List<Dataset> fullData = loader.load(testFile.toString());

                    // Sample data if it exceeds max samples
                    List<Dataset> sampledData = fullData;
                    if (fullData.size() > maxSamplesPerDomain) {
                        Collections.shuffle(fullData, new Random(42)); // Reproducible sampling
                        sampledData = fullData.subList(0, maxSamplesPerDomain);
                        System.out.println("✓ Loaded " + domain + ": " + sampledData.size() + " samples (sampled from " + fullData.size() + ") from " + testFile);
                    } else {
                        System.out.println("✓ Loaded " + domain + ": " + sampledData.size() + " samples from " + testFile);
                    }

                    testDatasets.put(domain, sampledData);
                } else {
                    System.out.println("⚠ Test file not found in either:");
                    System.out.println("    " + processedTestFile);
                    System.out.println("    " + rawTestFile);
                }
            }

            // Load models using ModelLoader
            System.out.println("\nLoading models from: " + modelsDir);
            Map<String, Map<String, SentimentClassifier>> models = new HashMap<>();

            for (String algo : ALGORITHMS) {
                System.out.println("\nLoading " + algo + " models...");
                Map<String, SentimentClassifier> algoModels = sentiment.models.ModelLoader.loadAllForAlgorithm(algo);

                if (algoModels.isEmpty()) {
                    System.out.println("⚠ No models found for " + algo);
                } else {
                    models.put(algo, algoModels);
                    System.out.println("✓ Loaded " + algoModels.size() + " " + algo + " model(s)");
                }
            }

            if (models.isEmpty()) {
                System.err.println("\n✗ No models found in " + modelsDir);
                System.err.println("Train models first using: ./scripts/train_all_models.sh");
                System.exit(1);
            }

            // Run cross-domain evaluation
            CrossDomainMatrix matrix = evaluateAll(models, testDatasets);

            // Print console report
            System.out.println(matrix.generateReport());

            // Export to JSON
            matrix.exportToJson(outputPath);

            // Always persist to model metadata files
            System.out.println("\nPersisting cross-domain performance to model metadata files...");
            persistToModelMetadata(matrix, modelsDir);

        } catch (Exception e) {
            System.err.println("Error during cross-domain evaluation: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
