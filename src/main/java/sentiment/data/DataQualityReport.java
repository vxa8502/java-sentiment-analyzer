package sentiment.data;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Comprehensive data quality analysis tool following Sofia's recommendations.
 * Detects biases, measures label quality, identifies edge cases, and generates detailed reports.
 *
 * @author Victoria Alabi
 */
public class DataQualityReport {

    private final String datasetName;
    private final DatasetStatistics statistics;
    private final BiasDetectionResults biases;
    private final LabelQualityMetrics labelQuality;
    private final EdgeCaseAnalysis edgeCases;
    private final LocalDateTime generatedAt;

    private DataQualityReport(Builder builder) {
        this.datasetName = builder.datasetName;
        this.statistics = builder.statistics;
        this.biases = builder.biases;
        this.labelQuality = builder.labelQuality;
        this.edgeCases = builder.edgeCases;
        this.generatedAt = LocalDateTime.now();
    }

    /**
     * Generate comprehensive quality report for a dataset
     */
    public static DataQualityReport analyze(String datasetName, List<Dataset> data) {
        System.out.println("Generating quality report for: " + datasetName);

        Builder builder = new Builder();
        builder.datasetName = datasetName;

        // Compute basic statistics
        builder.statistics = DatasetStatistics.compute(data);
        System.out.println("✓ Basic statistics computed");

        // Detect biases
        builder.biases = detectBiases(data);
        System.out.println("✓ Bias detection complete");

        // Analyze label quality
        builder.labelQuality = analyzeLabelQuality(data);
        System.out.println("✓ Label quality analysis complete");

        // Identify edge cases
        builder.edgeCases = identifyEdgeCases(data);
        System.out.println("✓ Edge case identification complete");

        return builder.build();
    }

    /**
     * Detect various types of biases in the dataset
     */
    private static BiasDetectionResults detectBiases(List<Dataset> data) {
        BiasDetectionResults results = new BiasDetectionResults();

        // 1. Length Bias - do positive/negative reviews have different lengths?
        Map<Dataset.SentimentLabel, List<Integer>> lengthsByLabel = data.stream()
            .collect(Collectors.groupingBy(
                Dataset::getSentiment,
                Collectors.mapping(d -> d.getText().length(), Collectors.toList())
            ));

        Map<Dataset.SentimentLabel, Double> avgLengthByLabel = new HashMap<>();
        for (var entry : lengthsByLabel.entrySet()) {
            double avg = entry.getValue().stream().mapToInt(Integer::intValue).average().orElse(0);
            avgLengthByLabel.put(entry.getKey(), avg);
        }

        results.lengthBias = avgLengthByLabel;
        results.hasLengthBias = detectSignificantLengthBias(avgLengthByLabel);

        // 2. Duplicate Detection - potential data leakage
        Map<String, Long> textCounts = data.stream()
            .collect(Collectors.groupingBy(d -> d.getText().toLowerCase().trim(), Collectors.counting()));

        results.duplicateRate = textCounts.values().stream()
            .filter(count -> count > 1)
            .count() / (double) data.size();

        // 3. Label Consistency - same text, different labels
        Map<String, Set<Dataset.SentimentLabel>> textLabels = new HashMap<>();
        for (Dataset d : data) {
            textLabels.computeIfAbsent(d.getText().toLowerCase().trim(), k -> new HashSet<>())
                     .add(d.getSentiment());
        }

        long inconsistentLabels = textLabels.values().stream()
            .filter(labels -> labels.size() > 1)
            .count();

        results.labelInconsistencyRate = inconsistentLabels / (double) textLabels.size();

        return results;
    }

    private static boolean detectSignificantLengthBias(Map<Dataset.SentimentLabel, Double> avgLengthByLabel) {
        if (avgLengthByLabel.size() < 2) return false;

        List<Double> lengths = new ArrayList<>(avgLengthByLabel.values());
        double max = Collections.max(lengths);
        double min = Collections.min(lengths);

        // Significant if difference > 20%
        return (max - min) / max > 0.20;
    }

    /**
     * Analyze label quality and estimate noise
     */
    private static LabelQualityMetrics analyzeLabelQuality(List<Dataset> data) {
        LabelQualityMetrics metrics = new LabelQualityMetrics();

        // Sample reviews for quality estimation
        Random random = new Random(42);
        List<Dataset> sample = data.stream()
            .filter(d -> random.nextDouble() < 0.01) // 1% sample
            .limit(100)
            .toList();

        // Heuristic-based noise detection
        int suspiciousLabels = 0;
        List<String> suspiciousExamples = new ArrayList<>();

        for (Dataset d : sample) {
            String text = d.getText().toLowerCase();
            boolean hasPositiveWords = containsPositiveWords(text);
            boolean hasNegativeWords = containsNegativeWords(text);

            // Suspicious if label doesn't match obvious sentiment words
            if (d.getSentiment() == Dataset.SentimentLabel.POSITIVE && hasNegativeWords && !hasPositiveWords) {
                suspiciousLabels++;
                suspiciousExamples.add("POSITIVE label but negative text: " + truncate(d.getText(), 80));
            } else if (d.getSentiment() == Dataset.SentimentLabel.NEGATIVE && hasPositiveWords && !hasNegativeWords) {
                suspiciousLabels++;
                suspiciousExamples.add("NEGATIVE label but positive text: " + truncate(d.getText(), 80));
            }
        }

        metrics.estimatedNoiseRate = suspiciousLabels / (double) sample.size();
        metrics.suspiciousExamples = suspiciousExamples.stream().limit(10).toList();
        metrics.sampleSize = sample.size();

        return metrics;
    }

    /**
     * Identify edge cases: sarcasm, mixed sentiment, negation
     */
    private static EdgeCaseAnalysis identifyEdgeCases(List<Dataset> data) {
        EdgeCaseAnalysis analysis = new EdgeCaseAnalysis();

        for (Dataset d : data) {
            String text = d.getText().toLowerCase();

            // Sarcasm detection (basic heuristics)
            if (containsSarcasmMarkers(text)) {
                analysis.sarcasmCandidates++;
                if (analysis.sarcasmExamples.size() < 5) {
                    analysis.sarcasmExamples.add(truncate(d.getText(), 100));
                }
            }

            // Mixed sentiment (has both positive and negative words)
            if (containsPositiveWords(text) && containsNegativeWords(text)) {
                analysis.mixedSentimentCandidates++;
                if (analysis.mixedSentimentExamples.size() < 5) {
                    analysis.mixedSentimentExamples.add(truncate(d.getText(), 100));
                }
            }

            // Heavy negation
            if (containsHeavyNegation(text)) {
                analysis.negationHeavyCandidates++;
                if (analysis.negationExamples.size() < 5) {
                    analysis.negationExamples.add(truncate(d.getText(), 100));
                }
            }
        }

        analysis.totalSamples = data.size();
        return analysis;
    }

    // Helper methods for sentiment word detection
    private static boolean containsPositiveWords(String text) {
        String[] positiveWords = {"great", "excellent", "amazing", "wonderful", "fantastic",
                                   "love", "loved", "brilliant", "perfect", "best", "good"};
        return Arrays.stream(positiveWords).anyMatch(text::contains);
    }

    private static boolean containsNegativeWords(String text) {
        String[] negativeWords = {"terrible", "awful", "horrible", "worst", "bad", "hate",
                                   "hated", "disappointing", "poor", "waste", "broken"};
        return Arrays.stream(negativeWords).anyMatch(text::contains);
    }

    private static boolean containsSarcasmMarkers(String text) {
        String[] sarcasmMarkers = {"oh great", "sure", "yeah right", "so good", "obviously",
                                    "totally", "definitely not", "just what i needed"};
        return Arrays.stream(sarcasmMarkers).anyMatch(text::contains);
    }

    private static boolean containsHeavyNegation(String text) {
        long negationCount = Arrays.stream(new String[]{"not ", "n't ", "never ", "no ", "nothing"})
            .filter(text::contains)
            .count();
        return negationCount >= 2;
    }

    private static String truncate(String text, int maxLength) {
        if (text.length() <= maxLength) return text;
        return text.substring(0, maxLength) + "...";
    }

    /**
     * Generate formatted console report
     */
    public String generateConsoleReport() {
        StringBuilder sb = new StringBuilder();

        sb.append("\n╔══════════════════════════════════════════════════════════════╗\n");
        sb.append(String.format("║       DATA QUALITY REPORT: %-32s ║\n", datasetName));
        sb.append("╚══════════════════════════════════════════════════════════════╝\n\n");

        // Basic Statistics
        sb.append(statistics.generateReport());

        // Bias Detection
        sb.append("\n[!] BIAS DETECTION\n");
        sb.append(String.format("   Length Bias: %s\n", biases.hasLengthBias ? "DETECTED" : "✓ Not detected"));
        if (!biases.lengthBias.isEmpty()) {
            for (var entry : biases.lengthBias.entrySet()) {
                sb.append(String.format("     - %s: avg %.0f chars\n",
                    entry.getKey().getDisplayName(), entry.getValue()));
            }
        }
        sb.append(String.format("   Duplicate Rate: %.2f%%\n", biases.duplicateRate * 100));
        sb.append(String.format("   Label Inconsistency: %.2f%%\n\n", biases.labelInconsistencyRate * 100));

        // Label Quality
        sb.append("[*] LABEL QUALITY\n");
        sb.append(String.format("   Estimated Noise Rate: %.1f%% (based on %d samples)\n",
            labelQuality.estimatedNoiseRate * 100, labelQuality.sampleSize));
        if (!labelQuality.suspiciousExamples.isEmpty()) {
            sb.append("   Suspicious Examples:\n");
            for (String example : labelQuality.suspiciousExamples.stream().limit(3).toList()) {
                sb.append("     • " + example + "\n");
            }
        }
        sb.append("\n");

        // Edge Cases
        sb.append("[>] EDGE CASE ANALYSIS\n");
        sb.append(String.format("   Sarcasm Candidates: %,d (%.1f%%)\n",
            edgeCases.sarcasmCandidates,
            (edgeCases.sarcasmCandidates * 100.0) / edgeCases.totalSamples));
        sb.append(String.format("   Mixed Sentiment: %,d (%.1f%%)\n",
            edgeCases.mixedSentimentCandidates,
            (edgeCases.mixedSentimentCandidates * 100.0) / edgeCases.totalSamples));
        sb.append(String.format("   Heavy Negation: %,d (%.1f%%)\n",
            edgeCases.negationHeavyCandidates,
            (edgeCases.negationHeavyCandidates * 100.0) / edgeCases.totalSamples));

        sb.append("\n══════════════════════════════════════════════════════════════\n");
        sb.append("Generated: " + generatedAt + "\n");

        return sb.toString();
    }

    /**
     * Export to JSON for documentation
     */
    public void exportToJson(Path outputPath) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.enable(SerializationFeature.INDENT_OUTPUT);

        Map<String, Object> report = new LinkedHashMap<>();
        report.put("dataset_name", datasetName);
        report.put("generated_at", generatedAt.toString());

        // Statistics
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("total_examples", statistics.getTotalExamples());
        stats.put("unique_texts", statistics.getUniqueTexts());
        stats.put("duplicate_count", statistics.getDuplicateCount());
        stats.put("vocabulary_size", statistics.getVocabularySize());
        stats.put("avg_text_length", statistics.getAvgTextLength());
        stats.put("median_text_length", statistics.getMedianTextLength());
        stats.put("min_text_length", statistics.getMinTextLength());
        stats.put("max_text_length", statistics.getMaxTextLength());
        stats.put("label_distribution", statistics.getLabelCounts());
        stats.put("label_balance_ratio", statistics.getLabelBalanceRatio());
        report.put("statistics", stats);

        // Biases
        Map<String, Object> biasesMap = new LinkedHashMap<>();
        biasesMap.put("has_length_bias", biases.hasLengthBias);
        biasesMap.put("avg_length_by_label", biases.lengthBias);
        biasesMap.put("duplicate_rate", biases.duplicateRate);
        biasesMap.put("label_inconsistency_rate", biases.labelInconsistencyRate);
        report.put("biases", biasesMap);

        // Label Quality
        Map<String, Object> qualityMap = new LinkedHashMap<>();
        qualityMap.put("estimated_noise_rate", labelQuality.estimatedNoiseRate);
        qualityMap.put("sample_size", labelQuality.sampleSize);
        qualityMap.put("suspicious_examples", labelQuality.suspiciousExamples);
        report.put("label_quality", qualityMap);

        // Edge Cases
        Map<String, Object> edgeMap = new LinkedHashMap<>();
        edgeMap.put("sarcasm_candidates", edgeCases.sarcasmCandidates);
        edgeMap.put("mixed_sentiment_candidates", edgeCases.mixedSentimentCandidates);
        edgeMap.put("negation_heavy_candidates", edgeCases.negationHeavyCandidates);
        edgeMap.put("total_samples", edgeCases.totalSamples);
        report.put("edge_cases", edgeMap);

        mapper.writeValue(outputPath.toFile(), report);
        System.out.println("✓ Quality report exported to: " + outputPath);
    }

    // Inner classes for organizing results
    private static class BiasDetectionResults {
        Map<Dataset.SentimentLabel, Double> lengthBias = new HashMap<>();
        boolean hasLengthBias = false;
        double duplicateRate = 0.0;
        double labelInconsistencyRate = 0.0;
    }

    private static class LabelQualityMetrics {
        double estimatedNoiseRate = 0.0;
        int sampleSize = 0;
        List<String> suspiciousExamples = new ArrayList<>();
    }

    private static class EdgeCaseAnalysis {
        int sarcasmCandidates = 0;
        int mixedSentimentCandidates = 0;
        int negationHeavyCandidates = 0;
        int totalSamples = 0;
        List<String> sarcasmExamples = new ArrayList<>();
        List<String> mixedSentimentExamples = new ArrayList<>();
        List<String> negationExamples = new ArrayList<>();
    }

    private static class Builder {
        private String datasetName;
        private DatasetStatistics statistics;
        private BiasDetectionResults biases;
        private LabelQualityMetrics labelQuality;
        private EdgeCaseAnalysis edgeCases;

        public DataQualityReport build() {
            return new DataQualityReport(this);
        }
    }

    /**
     * CLI entry point for generating quality reports
     */
    public static void main(String[] args) {
        if (args.length < 2) {
            System.err.println("Usage: java DataQualityReport <dataset-name> <path-to-csv> [output-json-path]");
            System.err.println("Example: java DataQualityReport imdb_50k data/processed/imdb_50k/train.csv datasets/imdb_50k/quality_report.json");
            System.exit(1);
        }

        String datasetName = args[0];
        Path inputPath = Paths.get(args[1]);
        Path outputPath = args.length > 2 ? Paths.get(args[2]) : null;

        try {
            // Load dataset
            System.out.println("Loading dataset from: " + inputPath);
            SimpleDatasetLoader loader = new SimpleDatasetLoader();
            List<Dataset> data = loader.load(inputPath.toString());

            // Generate report
            DataQualityReport report = DataQualityReport.analyze(datasetName, data);

            // Print to console
            System.out.println(report.generateConsoleReport());

            // Export to JSON if output path provided
            if (outputPath != null) {
                report.exportToJson(outputPath);
            }

        } catch (Exception e) {
            System.err.println("Error generating quality report: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
