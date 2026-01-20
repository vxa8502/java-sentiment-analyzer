package sentiment.config;

import jakarta.validation.constraints.*;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;
import org.springframework.validation.annotation.Validated;

/**
 * Shared configuration properties for feature extraction across the pipeline.
 *
 * <p>This is the single source of truth for feature extraction parameters used by:
 * <ul>
 *   <li>TextPreprocessor - for vocabulary building and tokenization</li>
 *   <li>TFIDFFeatureExtractor - for TF-IDF transformation</li>
 *   <li>WekaInstancesConverter - for Weka data structure conversion</li>
 * </ul>
 *
 * <p>All components must use these shared settings to ensure pipeline consistency
 * and prevent vocabulary mismatches.
 *
 * <p>Configuration is loaded from {@code sentiment.features.*} properties:
 * <pre>
 * sentiment:
 *   features:
 *     max-features: 5000
 *     min-term-freq: 1
 *     use-tfidf: true
 *     use-bigrams: true
 *     min-word-length: 2
 *     batch-size: 1000
 * </pre>
 *
 * @see sentiment.preprocessing.TextPreprocessor
 * @see sentiment.preprocessing.WekaInstancesConverter
 */
@ConfigurationProperties(prefix = "sentiment.features")
@Validated
public class FeatureExtractionProperties {

    /**
     * Maximum number of features to extract for TF-IDF.
     *
     * <p>Controls vocabulary size and model complexity:
     * <ul>
     *   <li>100-1000: Fast training, lower accuracy (for testing)</li>
     *   <li>5000: Balanced (default)</li>
     *   <li>10000+: Higher accuracy, slower, more memory</li>
     * </ul>
     *
     * <p>Range: [100, 50000]. Default: 5000
     */
    @Min(value = 100, message = "sentiment.features.max-features must be >= 100 (got: ${validatedValue})")
    @Max(value = 50000, message = "sentiment.features.max-features must be <= 50000 (got: ${validatedValue})")
    private int maxFeatures = 5000;

    /**
     * Minimum term frequency for a word to be included.
     *
     * <p>Filters out rare terms that appear in fewer than N documents:
     * <ul>
     *   <li>1: Keep all terms (default, prevents vocabulary=0)</li>
     *   <li>2: Require term appears in at least 2 documents</li>
     *   <li>5+: More aggressive filtering of rare terms</li>
     * </ul>
     *
     * <p>Range: [1, 100]. Default: 1
     */
    @Min(value = 1, message = "sentiment.features.min-term-freq must be >= 1 (got: ${validatedValue})")
    @Max(value = 100, message = "sentiment.features.min-term-freq must be <= 100 (got: ${validatedValue})")
    private int minTermFreq = 1;

    /**
     * Enable TF-IDF transformation for feature extraction.
     *
     * <p>When true, features are weighted by term frequency-inverse document frequency.
     * When false, uses simple term frequency (bag-of-words).
     *
     * <p>TF-IDF generally improves accuracy by downweighting common words.
     *
     * <p>Default: true
     */
    private boolean useTfidf = true;

    /**
     * Enable bigram features (unigrams + bigrams).
     *
     * <p>When true, extracts both single words and word pairs:
     * <ul>
     *   <li>"not good" → features: ["not", "good", "not_good"]</li>
     *   <li>Captures negation and context better</li>
     *   <li>Increases feature count ~2-3x</li>
     * </ul>
     *
     * <p>IMPORTANT: When enabled, max-features should be >= 1000 to accommodate
     * additional bigram features.
     *
     * <p>Default: true
     */
    private boolean useBigrams = true;

    /**
     * Minimum word length to include in tokenization.
     *
     * <p>Filters out very short words that rarely carry semantic meaning.
     * <ul>
     *   <li>1: Include single-letter words (I, a)</li>
     *   <li>2: Exclude single letters (default)</li>
     *   <li>3+: More aggressive filtering</li>
     * </ul>
     *
     * <p>Range: [1, 10]. Default: 2
     */
    @Min(value = 1, message = "sentiment.features.min-word-length must be >= 1 (got: ${validatedValue})")
    @Max(value = 10, message = "sentiment.features.min-word-length must be <= 10 (got: ${validatedValue})")
    private int minWordLength = 2;

    /**
     * Batch size for processing large datasets.
     *
     * <p>Controls memory usage during preprocessing:
     * <ul>
     *   <li>100: Low memory, slower (for development/testing)</li>
     *   <li>1000: Balanced (default)</li>
     *   <li>5000+: High throughput, more memory (for production)</li>
     * </ul>
     *
     * <p>Range: [10, 10000]. Default: 1000
     */
    @Min(value = 10, message = "sentiment.features.batch-size must be >= 10 (got: ${validatedValue})")
    @Max(value = 10000, message = "sentiment.features.batch-size must be <= 10000 (got: ${validatedValue})")
    private int batchSize = 1000;

    /**
     * Threshold for triggering MI-based feature selection.
     *
     * <p>When raw vocabulary exceeds this threshold, mutual information (MI)
     * is used to select the most discriminative features before applying
     * the final max-features limit.
     *
     * <p>Flow: raw vocab → MI selection (if > threshold) → max-features limit
     *
     * <p>Range: [10000, 100000]. Default: 50000
     */
    @Min(value = 10000, message = "sentiment.features.mi-selection-threshold must be >= 10000")
    @Max(value = 100000, message = "sentiment.features.mi-selection-threshold must be <= 100000")
    private int miSelectionThreshold = 50000;

    /**
     * Validates that when bigrams are enabled, max-features is sufficient.
     *
     * <p>Bigrams increase feature count significantly. This validation ensures
     * max-features is large enough to capture both unigrams and bigrams effectively.
     */
    @AssertTrue(message = "When sentiment.features.use-bigrams is true, max-features should be >= 1000")
    private boolean isValidBigramConfig() {
        // Allow small max-features when bigrams are disabled
        if (!useBigrams) {
            return true;
        }
        // Require at least 1000 features when bigrams are enabled
        return maxFeatures >= 1000;
    }

    /**
     * Validates that feature extraction parameters are reasonable.
     * Max features should be significantly larger than min term frequency
     * to ensure adequate vocabulary size.
     */
    @AssertTrue(message = "max-features should be at least 10x min-term-freq to avoid empty vocabulary")
    private boolean isFeatureConfigReasonable() {
        return maxFeatures >= (minTermFreq * 10);
    }

    // Getters and setters

    public int getMaxFeatures() {
        return maxFeatures;
    }

    public void setMaxFeatures(int maxFeatures) {
        this.maxFeatures = maxFeatures;
    }

    public int getMinTermFreq() {
        return minTermFreq;
    }

    public void setMinTermFreq(int minTermFreq) {
        this.minTermFreq = minTermFreq;
    }

    public boolean isUseTfidf() {
        return useTfidf;
    }

    public void setUseTfidf(boolean useTfidf) {
        this.useTfidf = useTfidf;
    }

    public boolean isUseBigrams() {
        return useBigrams;
    }

    public void setUseBigrams(boolean useBigrams) {
        this.useBigrams = useBigrams;
    }

    public int getMinWordLength() {
        return minWordLength;
    }

    public void setMinWordLength(int minWordLength) {
        this.minWordLength = minWordLength;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    public int getMiSelectionThreshold() {
        return miSelectionThreshold;
    }

    public void setMiSelectionThreshold(int miSelectionThreshold) {
        this.miSelectionThreshold = miSelectionThreshold;
    }
}
