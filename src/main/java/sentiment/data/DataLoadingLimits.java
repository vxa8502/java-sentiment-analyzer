package sentiment.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;
import org.springframework.validation.annotation.Validated;

import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotNull;

/**
 * Configuration for dataset loading memory limits.
 * Prevents OOM errors when loading large datasets by enforcing reasonable limits.
 *
 * All limits are configurable via application.properties:
 *
 * sentiment.data.limits.max-datasets-per-file=100000
 * sentiment.data.limits.max-combined-datasets=500000
 * sentiment.data.limits.max-file-size-mb=500
 * sentiment.data.limits.warn-threshold-percent=80
 */
@Configuration
@ConfigurationProperties(prefix = "sentiment.data.limits")
@Validated
public class DataLoadingLimits {

    private static final Logger logger = LoggerFactory.getLogger(DataLoadingLimits.class);

    /**
     * Maximum number of datasets to load from a single file.
     * Default: 100,000 datasets (~100MB assuming 1KB per text)
     */
    @NotNull
    @Min(1000)
    private Integer maxDatasetsPerFile = 100_000;

    /**
     * Maximum total datasets when combining multiple files.
     * Default: 500,000 datasets (~500MB assuming 1KB per text)
     */
    @NotNull
    @Min(10000)
    private Integer maxCombinedDatasets = 500_000;

    /**
     * Maximum file size in MB before warning.
     * Default: 500MB
     */
    @NotNull
    @Min(1)
    private Integer maxFileSizeMb = 500;

    /**
     * Percentage threshold (0-100) to trigger warnings before hitting hard limits.
     * Default: 80% (warn when reaching 80% of limit)
     */
    @NotNull
    @Min(50)
    private Integer warnThresholdPercent = 80;

    /**
     * Whether to enforce limits strictly (throw exception) or just warn.
     * Default: true (strict enforcement)
     */
    private boolean strictEnforcement = true;

    /**
     * Estimated number of features for statistical validation.
     * Used to calculate minimum required samples for statistical power.
     * Default: 5000 (typical TF-IDF feature count)
     */
    @NotNull
    @Min(100)
    private Integer estimatedFeatures = 5000;

    // Getters and Setters

    public Integer getMaxDatasetsPerFile() {
        return maxDatasetsPerFile;
    }

    public void setMaxDatasetsPerFile(Integer maxDatasetsPerFile) {
        if (maxDatasetsPerFile != null && maxDatasetsPerFile >= 1000) {
            this.maxDatasetsPerFile = maxDatasetsPerFile;
            logger.info("Updated maxDatasetsPerFile to: {}", maxDatasetsPerFile);
        }
    }

    public Integer getMaxCombinedDatasets() {
        return maxCombinedDatasets;
    }

    public void setMaxCombinedDatasets(Integer maxCombinedDatasets) {
        if (maxCombinedDatasets != null && maxCombinedDatasets >= 10000) {
            this.maxCombinedDatasets = maxCombinedDatasets;
            logger.info("Updated maxCombinedDatasets to: {}", maxCombinedDatasets);
        }
    }

    public Integer getMaxFileSizeMb() {
        return maxFileSizeMb;
    }

    public void setMaxFileSizeMb(Integer maxFileSizeMb) {
        if (maxFileSizeMb != null && maxFileSizeMb >= 1) {
            this.maxFileSizeMb = maxFileSizeMb;
            logger.info("Updated maxFileSizeMb to: {}", maxFileSizeMb);
        }
    }

    public Integer getWarnThresholdPercent() {
        return warnThresholdPercent;
    }

    public void setWarnThresholdPercent(Integer warnThresholdPercent) {
        if (warnThresholdPercent != null && warnThresholdPercent >= 50 && warnThresholdPercent <= 100) {
            this.warnThresholdPercent = warnThresholdPercent;
            logger.info("Updated warnThresholdPercent to: {}", warnThresholdPercent);
        }
    }

    public boolean isStrictEnforcement() {
        return strictEnforcement;
    }

    public void setStrictEnforcement(boolean strictEnforcement) {
        this.strictEnforcement = strictEnforcement;
        logger.info("Updated strictEnforcement to: {}", strictEnforcement);
    }

    public Integer getEstimatedFeatures() {
        return estimatedFeatures;
    }

    public void setEstimatedFeatures(Integer estimatedFeatures) {
        if (estimatedFeatures != null && estimatedFeatures >= 100) {
            this.estimatedFeatures = estimatedFeatures;
            logger.info("Updated estimatedFeatures to: {}", estimatedFeatures);
        }
    }

    // Utility Methods

    /**
     * Check if a dataset count exceeds the per-file limit.
     *
     * @param count Current dataset count
     * @param filePath File being loaded (for logging)
     * @throws DataLoadingException if strict enforcement is enabled and limit exceeded
     */
    public void checkPerFileLimit(int count, String filePath) throws DataLoadingException {
        if (count > maxDatasetsPerFile) {
            String message = String.format(
                    "Dataset count %d exceeds maximum per-file limit of %d for file: %s",
                    count, maxDatasetsPerFile, getFileName(filePath)
            );

            if (strictEnforcement) {
                logger.error("HARD LIMIT EXCEEDED: {}", message);
                throw new DataLoadingException(message, filePath, "Memory Limit");
            } else {
                logger.warn("SOFT LIMIT EXCEEDED: {} (strict enforcement disabled)", message);
            }
        }

        // Warn when approaching limit
        int warnThreshold = (maxDatasetsPerFile * warnThresholdPercent) / 100;
        if (count > warnThreshold && count <= maxDatasetsPerFile) {
            double percentUsed = (double) count / maxDatasetsPerFile * 100;
            String percent = String.format("%.1f", percentUsed);
            logger.warn("Approaching per-file limit for {}: {} datasets ({}% of limit)",
                    getFileName(filePath), count, percent);
        }
    }

    /**
     * Check if combined dataset count exceeds the limit.
     *
     * @param totalCount Total datasets across all files
     * @param fileCount Number of files being combined
     * @throws DataLoadingException if strict enforcement is enabled and limit exceeded
     */
    public void checkCombinedLimit(int totalCount, int fileCount) throws DataLoadingException {
        if (totalCount > maxCombinedDatasets) {
            String message = String.format(
                    "Combined dataset count %d exceeds maximum limit of %d (from %d files)",
                    totalCount, maxCombinedDatasets, fileCount
            );

            if (strictEnforcement) {
                logger.error("HARD LIMIT EXCEEDED: {}", message);
                throw new DataLoadingException(message,
                        String.format("Combined from %d files", fileCount), "Memory Limit");
            } else {
                logger.warn("SOFT LIMIT EXCEEDED: {} (strict enforcement disabled)", message);
            }
        }

        // Warn when approaching limit
        int warnThreshold = (maxCombinedDatasets * warnThresholdPercent) / 100;
        if (totalCount > warnThreshold && totalCount <= maxCombinedDatasets) {
            double percentUsed = (double) totalCount / maxCombinedDatasets * 100;
            String percent = String.format("%.1f", percentUsed);
            logger.warn("Approaching combined dataset limit: {} datasets ({}% of limit) from {} files",
                    totalCount, percent, fileCount);
        }
    }

    /**
     * Check if file size is too large before attempting to load.
     *
     * @param fileSizeMb File size in MB
     * @param filePath File being checked
     * @throws DataLoadingException if strict enforcement is enabled and size too large
     */
    public void checkFileSize(long fileSizeMb, String filePath) throws DataLoadingException {
        if (fileSizeMb > maxFileSizeMb) {
            String message = String.format(
                    "File size %d MB exceeds maximum limit of %d MB for file: %s",
                    fileSizeMb, maxFileSizeMb, getFileName(filePath)
            );

            if (strictEnforcement) {
                logger.error("HARD LIMIT EXCEEDED: {}", message);
                throw new DataLoadingException(message, filePath, "Memory Limit");
            } else {
                logger.warn("SOFT LIMIT EXCEEDED: {} (strict enforcement disabled)", message);
            }
        }

        // Warn for large files even below limit
        long warnThreshold = (maxFileSizeMb * warnThresholdPercent) / 100;
        if (fileSizeMb > warnThreshold && fileSizeMb <= maxFileSizeMb) {
            double percentUsed = (double) fileSizeMb / maxFileSizeMb * 100;
            String percent = String.format("%.1f", percentUsed);
            logger.warn("Loading large file {}: {} MB ({}% of limit) - may take time",
                    getFileName(filePath), fileSizeMb, percent);
        }
    }

    /**
     * Get estimated memory usage for a dataset count.
     * Assumes average 1KB per text sample.
     *
     * @param datasetCount Number of datasets
     * @return Estimated memory in MB
     */
    public long estimateMemoryUsageMb(int datasetCount) {
        // Conservative estimate: 1KB per text + overhead
        return (datasetCount * 1024L) / (1024L * 1024L);
    }

    /**
     * Validate if the proposed sample size is statistically sufficient.
     * Aria's requirement: Prove that sample sizes meet statistical power requirements.
     *
     * This method checks:
     * 1. Minimum samples per class for statistical significance
     * 2. Sample size for reliable confidence interval estimation
     *
     * @param proposedSize Number of samples being loaded
     * @param numClasses Number of distinct classes in the dataset (e.g., 3 for positive/negative/neutral)
     * @param numFeatures Number of features in the model (for statistical power calculation)
     * @return ValidationResult containing warnings and recommendations
     */
    public SampleSizeValidation validateSampleSize(int proposedSize, int numClasses, int numFeatures) {
        SampleSizeValidation result = new SampleSizeValidation();

        // 1. Minimum samples per class for statistical significance
        // Rule: 30 base samples + 10 * log(numFeatures) to account for feature dimensionality
        // Justification: Based on Central Limit Theorem and degrees of freedom requirements
        int minPerClass = (int) Math.ceil(30 + 10 * Math.log(numFeatures));
        int recommendedMinTotal = minPerClass * numClasses;

        if (proposedSize < recommendedMinTotal) {
            String warning = String.format(
                "Sample size %d may be insufficient for %d classes with %d features. " +
                "Recommend at least %d samples (%d per class) for statistical power.",
                proposedSize, numClasses, numFeatures, recommendedMinTotal, minPerClass
            );
            logger.warn(warning);
            result.addWarning(warning);
            result.setSufficient(false);
        }

        // 2. Sample size for confidence interval estimation
        // Formula: n = (Z^2 * p * (1-p)) / E^2
        // Where: Z = 1.96 (95% CI), p = 0.5 (max variance), E = 0.05 (±5% margin)
        // Result: n ≈ 384 for 95% confidence with ±5% margin of error
        double z = 1.96;  // 95% confidence level
        double p = 0.25;  // Conservative variance estimate (0.5 * 0.5)
        double e = 0.05;  // ±5% margin of error
        double requiredForCI = (z * z * p) / (e * e);

        if (proposedSize < requiredForCI) {
            String warning = String.format(
                "Sample size %d insufficient for tight confidence intervals. " +
                "Need at least %d samples for ±%.1f%% margin at 95%% confidence (currently achievable margin: ±%.2f%%).",
                proposedSize, (int)Math.ceil(requiredForCI), e * 100,
                z * Math.sqrt(p / proposedSize) * 100
            );
            logger.warn(warning);
            result.addWarning(warning);
            result.setSufficient(false);
        }

        // 3. Check for class imbalance concerns
        // If we have fewer than 100 samples per class on average, warn about potential imbalance issues
        int avgPerClass = proposedSize / numClasses;
        if (avgPerClass < 100) {
            String warning = String.format(
                "Average samples per class (%d) is low. " +
                "Class imbalance may significantly impact model performance. " +
                "Consider stratified sampling to ensure balanced representation.",
                avgPerClass
            );
            logger.warn(warning);
            result.addWarning(warning);
        }

        // 4. Model complexity vs sample size ratio (curse of dimensionality)
        // Rule of thumb: Need at least 10-20 samples per feature for reliable generalization
        int minForFeatures = numFeatures * 10;  // Conservative estimate
        if (proposedSize < minForFeatures) {
            String warning = String.format(
                "Sample size %d may be too small for %d features (ratio: %.1f samples/feature). " +
                "Risk of overfitting. Recommend at least %d samples (10x features) or feature reduction.",
                proposedSize, numFeatures, (double)proposedSize / numFeatures, minForFeatures
            );
            logger.warn(warning);
            result.addWarning(warning);
            result.setSufficient(false);
        }

        // Success case
        if (result.isSufficient()) {
            logger.info("Sample size {} validated: Sufficient for {} classes with {} features " +
                       "({}% of recommended minimum, {:.1f} samples/feature)",
                       proposedSize, numClasses, numFeatures,
                       (proposedSize * 100 / recommendedMinTotal),
                       (double)proposedSize / numFeatures);
        }

        return result;
    }

    /**
     * Result of sample size validation with warnings and recommendations.
     */
    public static class SampleSizeValidation {
        private boolean sufficient = true;
        private final java.util.List<String> warnings = new java.util.ArrayList<>();

        public boolean isSufficient() {
            return sufficient;
        }

        public void setSufficient(boolean sufficient) {
            this.sufficient = sufficient;
        }

        public java.util.List<String> getWarnings() {
            return warnings;
        }

        public void addWarning(String warning) {
            this.warnings.add(warning);
        }

        public boolean hasWarnings() {
            return !warnings.isEmpty();
        }
    }

    /**
     * Log current limit configuration.
     */
    public void logConfiguration() {
        logger.info("=== Dataset Loading Limits Configuration ===");
        logger.info("Max datasets per file: {}", maxDatasetsPerFile);
        logger.info("Max combined datasets: {}", maxCombinedDatasets);
        logger.info("Max file size: {} MB", maxFileSizeMb);
        logger.info("Warning threshold: {}%", warnThresholdPercent);
        logger.info("Strict enforcement: {}", strictEnforcement);
        logger.info("Estimated max memory per file: ~{} MB",
                estimateMemoryUsageMb(maxDatasetsPerFile));
        logger.info("Estimated max memory combined: ~{} MB",
                estimateMemoryUsageMb(maxCombinedDatasets));
        logger.info("===========================================");
    }

    private String getFileName(String filePath) {
        if (filePath == null || filePath.isEmpty()) {
            return "unknown file";
        }

        int lastSeparator = Math.max(filePath.lastIndexOf('/'), filePath.lastIndexOf('\\'));
        if (lastSeparator >= 0 && lastSeparator < filePath.length() - 1) {
            return filePath.substring(lastSeparator + 1);
        }

        return filePath;
    }

    @Override
    public String toString() {
        return String.format(
                "DataLoadingLimits{perFile=%d, combined=%d, fileSize=%dMB, warnAt=%d%%, strict=%s}",
                maxDatasetsPerFile, maxCombinedDatasets, maxFileSizeMb,
                warnThresholdPercent, strictEnforcement
        );
    }
}