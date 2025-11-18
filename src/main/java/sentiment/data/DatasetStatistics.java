package sentiment.data;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Comprehensive dataset statistics for validation and analysis.
 *
 * Sofia's Philosophy: "You can't validate what you can't measure."
 *
 * This class computes key metrics for dataset quality assessment:
 * - Label distribution (balance check)
 * - Text length statistics (edge case detection)
 * - Vocabulary diversity (representation check)
 * - Duplicate detection (data leakage prevention)
 *
 * @author Victoria Alabi
 */
public class DatasetStatistics {

    // Basic counts
    private final int totalExamples;
    private final Map<Dataset.SentimentLabel, Long> labelCounts;

    // Text statistics
    private final double avgTextLength;
    private final int minTextLength;
    private final int maxTextLength;
    private final double medianTextLength;

    // Quality metrics
    private final int duplicateCount;
    private final int uniqueTexts;
    private final double vocabularySize;

    // Distribution analysis
    private final Map<Dataset.SentimentLabel, Double> labelPercentages;
    private final double labelBalanceRatio; // Closest to 1.0 = balanced

    private DatasetStatistics(Builder builder) {
        this.totalExamples = builder.totalExamples;
        this.labelCounts = builder.labelCounts;
        this.avgTextLength = builder.avgTextLength;
        this.minTextLength = builder.minTextLength;
        this.maxTextLength = builder.maxTextLength;
        this.medianTextLength = builder.medianTextLength;
        this.duplicateCount = builder.duplicateCount;
        this.uniqueTexts = builder.uniqueTexts;
        this.vocabularySize = builder.vocabularySize;
        this.labelPercentages = builder.labelPercentages;
        this.labelBalanceRatio = builder.labelBalanceRatio;
    }

    /**
     * Compute comprehensive statistics for a dataset.
     *
     * Sofia's checklist:
     * ✓ Label distribution
     * ✓ Text length distribution
     * ✓ Duplicate detection
     * ✓ Vocabulary diversity
     */
    public static DatasetStatistics compute(List<Dataset> datasets) {
        if (datasets == null || datasets.isEmpty()) {
            throw new IllegalArgumentException("Cannot compute statistics for empty dataset");
        }

        Builder builder = new Builder();

        // Basic counts
        builder.totalExamples = datasets.size();

        // Label distribution
        builder.labelCounts = datasets.stream()
            .collect(Collectors.groupingBy(Dataset::getSentiment, Collectors.counting()));

        builder.labelPercentages = new HashMap<>();
        for (Map.Entry<Dataset.SentimentLabel, Long> entry : builder.labelCounts.entrySet()) {
            double percentage = (entry.getValue() * 100.0) / builder.totalExamples;
            builder.labelPercentages.put(entry.getKey(), percentage);
        }

        // Calculate balance ratio (for binary classification)
        if (builder.labelCounts.size() == 2) {
            List<Long> counts = new ArrayList<>(builder.labelCounts.values());
            long max = Math.max(counts.get(0), counts.get(1));
            long min = Math.min(counts.get(0), counts.get(1));
            builder.labelBalanceRatio = (double) min / max;
        } else {
            // Multi-class: use min/max ratio across all classes
            long max = Collections.max(builder.labelCounts.values());
            long min = Collections.min(builder.labelCounts.values());
            builder.labelBalanceRatio = (double) min / max;
        }

        // Text length statistics
        List<Integer> lengths = datasets.stream()
            .map(Dataset::getTextLength)
            .sorted()
            .collect(Collectors.toList());

        builder.minTextLength = lengths.get(0);
        builder.maxTextLength = lengths.get(lengths.size() - 1);
        builder.avgTextLength = lengths.stream()
            .mapToInt(Integer::intValue)
            .average()
            .orElse(0.0);

        // Median
        int midpoint = lengths.size() / 2;
        builder.medianTextLength = lengths.size() % 2 == 0 ?
            (lengths.get(midpoint - 1) + lengths.get(midpoint)) / 2.0 :
            lengths.get(midpoint);

        // Duplicate detection
        Set<String> uniqueTextsSet = new HashSet<>();
        for (Dataset d : datasets) {
            uniqueTextsSet.add(d.getText().toLowerCase().trim());
        }

        builder.uniqueTexts = uniqueTextsSet.size();
        builder.duplicateCount = builder.totalExamples - builder.uniqueTexts;

        // Vocabulary diversity (unique word count across entire dataset)
        Set<String> vocabulary = new HashSet<>();
        for (Dataset d : datasets) {
            String[] words = d.getText().toLowerCase().split("\\s+");
            vocabulary.addAll(Arrays.asList(words));
        }
        builder.vocabularySize = vocabulary.size();

        return builder.build();
    }

    // Getters
    public int getTotalExamples() { return totalExamples; }
    public Map<Dataset.SentimentLabel, Long> getLabelCounts() { return labelCounts; }
    public double getAvgTextLength() { return avgTextLength; }
    public int getMinTextLength() { return minTextLength; }
    public int getMaxTextLength() { return maxTextLength; }
    public double getMedianTextLength() { return medianTextLength; }
    public int getDuplicateCount() { return duplicateCount; }
    public int getUniqueTexts() { return uniqueTexts; }
    public double getVocabularySize() { return vocabularySize; }
    public Map<Dataset.SentimentLabel, Double> getLabelPercentages() { return labelPercentages; }
    public double getLabelBalanceRatio() { return labelBalanceRatio; }

    /**
     * Check if dataset is well-balanced.
     * Sofia's threshold: Ratio > 0.8 is acceptable, > 0.95 is excellent.
     */
    public boolean isBalanced() {
        return labelBalanceRatio >= 0.8;
    }

    /**
     * Check if there are significant duplicates.
     * Sofia's threshold: < 1% duplicates is acceptable.
     */
    public boolean hasMinimalDuplicates() {
        double duplicateRate = (duplicateCount * 100.0) / totalExamples;
        return duplicateRate < 1.0;
    }

    /**
     * Check if text lengths are reasonable (no degenerate cases).
     * Sofia's rule: Min > 10 chars, no empty texts.
     */
    public boolean hasReasonableTextLengths() {
        return minTextLength > 10 && maxTextLength < 100000;
    }

    /**
     * Overall data quality assessment.
     *
     * @return true if dataset passes all quality checks
     */
    public boolean passesQualityChecks() {
        return isBalanced() && hasMinimalDuplicates() && hasReasonableTextLengths();
    }

    /**
     * Generate a formatted report for logging.
     */
    public String generateReport() {
        StringBuilder sb = new StringBuilder();

        sb.append("\n╔════════════════════════════════════════════════════════════╗\n");
        sb.append("║           DATASET QUALITY REPORT (Sofia's Audit)           ║\n");
        sb.append("╚════════════════════════════════════════════════════════════╝\n\n");

        // Basic info
        sb.append("📊 DATASET SIZE\n");
        sb.append(String.format("   Total Examples: %,d\n", totalExamples));
        sb.append(String.format("   Unique Texts: %,d\n", uniqueTexts));
        sb.append(String.format("   Duplicates: %,d (%.2f%%)\n",
            duplicateCount, (duplicateCount * 100.0) / totalExamples));
        sb.append(String.format("   Vocabulary Size: %,.0f unique tokens\n\n", vocabularySize));

        // Label distribution
        sb.append("🏷️  LABEL DISTRIBUTION\n");
        for (Map.Entry<Dataset.SentimentLabel, Long> entry : labelCounts.entrySet()) {
            double percentage = labelPercentages.get(entry.getKey());
            sb.append(String.format("   %s: %,d (%.1f%%)\n",
                entry.getKey().getDisplayName().toUpperCase(),
                entry.getValue(),
                percentage));
        }
        sb.append(String.format("   Balance Ratio: %.3f ", labelBalanceRatio));
        sb.append(isBalanced() ? "✓ BALANCED\n\n" : "⚠ IMBALANCED\n\n");

        // Text length stats
        sb.append("📏 TEXT LENGTH STATISTICS\n");
        sb.append(String.format("   Average: %.0f characters\n", avgTextLength));
        sb.append(String.format("   Median: %.0f characters\n", medianTextLength));
        sb.append(String.format("   Min: %,d characters\n", minTextLength));
        sb.append(String.format("   Max: %,d characters\n", maxTextLength));
        sb.append(String.format("   Range: %,d characters\n\n", maxTextLength - minTextLength));

        // Quality assessment
        sb.append("✅ QUALITY CHECKS\n");
        sb.append(String.format("   [%s] Label Balance (ratio > 0.8)\n",
            isBalanced() ? "✓" : "✗"));
        sb.append(String.format("   [%s] Minimal Duplicates (< 1%%)\n",
            hasMinimalDuplicates() ? "✓" : "✗"));
        sb.append(String.format("   [%s] Reasonable Text Lengths (> 10 chars)\n",
            hasReasonableTextLengths() ? "✓" : "✗"));
        sb.append(String.format("\n   Overall: %s\n",
            passesQualityChecks() ? "✓ PASS" : "⚠ WARNINGS DETECTED"));

        sb.append("\n═══════════════════════════════════════════════════════════\n");

        return sb.toString();
    }

    @Override
    public String toString() {
        return generateReport();
    }

    // Builder pattern
    private static class Builder {
        private int totalExamples;
        private Map<Dataset.SentimentLabel, Long> labelCounts;
        private double avgTextLength;
        private int minTextLength;
        private int maxTextLength;
        private double medianTextLength;
        private int duplicateCount;
        private int uniqueTexts;
        private double vocabularySize;
        private Map<Dataset.SentimentLabel, Double> labelPercentages;
        private double labelBalanceRatio;

        public DatasetStatistics build() {
            return new DatasetStatistics(this);
        }
    }
}
