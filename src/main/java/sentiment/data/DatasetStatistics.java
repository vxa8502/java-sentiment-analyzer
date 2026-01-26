package sentiment.data;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Immutable statistics for dataset validation and quality assessment.
 *
 * @author Victoria Alabi
 */
@SuppressWarnings("unused") // Public API - methods used by CLI tools and external consumers
public class DatasetStatistics {

    /** Total number of examples in the dataset. */
    private final int totalExamples;

    /** Count of examples for each sentiment label. */
    private final Map<Dataset.SentimentLabel, Long> labelCounts;

    /** Average text length in characters. */
    private final double avgTextLength;

    /** Minimum text length in characters. */
    private final int minTextLength;

    /** Maximum text length in characters. */
    private final int maxTextLength;

    /** Median text length in characters. */
    private final double medianTextLength;

    /** Number of duplicate texts (case-insensitive). */
    private final int duplicateCount;

    /** Number of unique texts. */
    private final int uniqueTexts;

    /** Total vocabulary size (unique tokens). */
    private final double vocabularySize;

    /** Percentage distribution of each label. */
    private final Map<Dataset.SentimentLabel, Double> labelPercentages;

    /** Ratio of smallest to largest class (0.0 to 1.0, higher is more balanced). */
    private final double labelBalanceRatio;

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
     * Computes statistics for the given dataset.
     *
     * @param datasets the dataset examples to analyze
     * @return computed statistics
     * @throws IllegalArgumentException if datasets is null or empty
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

        // Calculate balance ratio: min/max across all classes (works for binary and multi-class)
        long max = Collections.max(builder.labelCounts.values());
        long min = Collections.min(builder.labelCounts.values());
        builder.labelBalanceRatio = (double) min / max;

        // Text length statistics
        List<Integer> lengths = datasets.stream()
            .map(Dataset::getTextLength)
            .sorted()
            .toList();

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

    /** @return total number of examples */
    public int getTotalExamples() { return totalExamples; }

    /** @return count of examples for each label */
    public Map<Dataset.SentimentLabel, Long> getLabelCounts() { return labelCounts; }

    /** @return average text length in characters */
    public double getAvgTextLength() { return avgTextLength; }

    /** @return minimum text length in characters */
    public int getMinTextLength() { return minTextLength; }

    /** @return maximum text length in characters */
    public int getMaxTextLength() { return maxTextLength; }

    /** @return median text length in characters */
    public double getMedianTextLength() { return medianTextLength; }

    /** @return number of duplicate texts */
    public int getDuplicateCount() { return duplicateCount; }

    /** @return number of unique texts */
    public int getUniqueTexts() { return uniqueTexts; }

    /** @return vocabulary size (unique tokens) */
    public double getVocabularySize() { return vocabularySize; }

    /** @return percentage distribution of each label (0-100) */
    public Map<Dataset.SentimentLabel, Double> getLabelPercentages() { return labelPercentages; }

    /** @return label balance ratio (1.0 = perfectly balanced) */
    public double getLabelBalanceRatio() { return labelBalanceRatio; }

    /** @return true if balance ratio >= 0.8 */
    public boolean isBalanced() {
        return labelBalanceRatio >= 0.8;
    }

    /** @return true if duplicate rate < 1% */
    public boolean hasMinimalDuplicates() {
        double duplicateRate = (duplicateCount * 100.0) / totalExamples;
        return duplicateRate < 1.0;
    }

    /** @return true if text lengths are reasonable (min > 10, max < 100000) */
    public boolean hasReasonableTextLengths() {
        return minTextLength > 10 && maxTextLength < 100000;
    }

    /** @return true if all quality checks pass */
    public boolean passesQualityChecks() {
        return isBalanced() && hasMinimalDuplicates() && hasReasonableTextLengths();
    }

    /** @return formatted report of all statistics */
    public String generateReport() {
        StringBuilder sb = new StringBuilder();

        sb.append("\n╔════════════════════════════════════════════════════════════╗\n");
        sb.append("║              DATASET QUALITY REPORT                        ║\n");
        sb.append("╚════════════════════════════════════════════════════════════╝\n\n");

        // Basic info
        sb.append("[#] DATASET SIZE\n");
        sb.append(String.format("   Total Examples: %,d\n", totalExamples));
        sb.append(String.format("   Unique Texts: %,d\n", uniqueTexts));
        sb.append(String.format("   Duplicates: %,d (%.2f%%)\n",
            duplicateCount, (duplicateCount * 100.0) / totalExamples));
        sb.append(String.format("   Vocabulary Size: %,.0f unique tokens\n\n", vocabularySize));

        // Label distribution
        sb.append("[%] LABEL DISTRIBUTION\n");
        for (Map.Entry<Dataset.SentimentLabel, Long> entry : labelCounts.entrySet()) {
            double percentage = labelPercentages.get(entry.getKey());
            sb.append(String.format("   %s: %,d (%.1f%%)\n",
                entry.getKey().getDisplayName().toUpperCase(),
                entry.getValue(),
                percentage));
        }
        sb.append(String.format("   Balance Ratio: %.3f ", labelBalanceRatio));
        sb.append(isBalanced() ? "(BALANCED)\n\n" : "(IMBALANCED)\n\n");

        // Text length stats
        sb.append("[~] TEXT LENGTH STATISTICS\n");
        sb.append(String.format("   Average: %.0f characters\n", avgTextLength));
        sb.append(String.format("   Median: %.0f characters\n", medianTextLength));
        sb.append(String.format("   Min: %,d characters\n", minTextLength));
        sb.append(String.format("   Max: %,d characters\n", maxTextLength));
        sb.append(String.format("   Range: %,d characters\n\n", maxTextLength - minTextLength));

        // Quality assessment
        sb.append("[?] QUALITY CHECKS\n");
        sb.append(String.format("   [%s] Label Balance (ratio > 0.8)\n",
            isBalanced() ? "x" : " "));
        sb.append(String.format("   [%s] Minimal Duplicates (< 1%%)\n",
            hasMinimalDuplicates() ? "x" : " "));
        sb.append(String.format("   [%s] Reasonable Text Lengths (> 10 chars)\n",
            hasReasonableTextLengths() ? "x" : " "));
        sb.append(String.format("\n   Overall: %s\n",
            passesQualityChecks() ? "PASS" : "WARNINGS DETECTED"));

        sb.append("\n═══════════════════════════════════════════════════════════\n");

        return sb.toString();
    }

    @Override
    public String toString() {
        return generateReport();
    }

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
