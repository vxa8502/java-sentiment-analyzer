package sentiment.config;

/**
 * Test utility for creating FeatureExtractionProperties instances in tests.
 */
public class TestFeatureConfig {

    /**
     * Creates a default FeatureExtractionProperties for testing.
     */
    public static FeatureExtractionProperties createDefault() {
        FeatureExtractionProperties config = new FeatureExtractionProperties();
        config.setMaxFeatures(5000);
        config.setMinTermFreq(1);
        config.setUseTfidf(true);
        config.setUseBigrams(true);
        config.setMinWordLength(2);
        config.setBatchSize(1000);
        return config;
    }

    /**
     * Creates a minimal FeatureExtractionProperties for fast testing.
     */
    public static FeatureExtractionProperties createMinimal() {
        FeatureExtractionProperties config = new FeatureExtractionProperties();
        config.setMaxFeatures(100);
        config.setMinTermFreq(1);
        config.setUseTfidf(true);
        config.setUseBigrams(false);  // Disable bigrams for faster tests
        config.setMinWordLength(2);
        config.setBatchSize(10);
        return config;
    }

    /**
     * Creates a custom FeatureExtractionProperties for testing specific scenarios.
     */
    public static FeatureExtractionProperties create(int maxFeatures, int minTermFreq,
                                                       boolean useTfidf, boolean useBigrams,
                                                       int minWordLength, int batchSize) {
        FeatureExtractionProperties config = new FeatureExtractionProperties();
        config.setMaxFeatures(maxFeatures);
        config.setMinTermFreq(minTermFreq);
        config.setUseTfidf(useTfidf);
        config.setUseBigrams(useBigrams);
        config.setMinWordLength(minWordLength);
        config.setBatchSize(batchSize);
        return config;
    }
}
