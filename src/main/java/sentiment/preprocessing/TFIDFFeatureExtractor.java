package sentiment.preprocessing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.beans.factory.annotation.Autowired;
import sentiment.data.Dataset;
import sentiment.util.ValidationUtils;
import weka.core.*;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.*;
import java.util.stream.Collectors;

/**
 * TF-IDF feature extractor that accepts pre-cleaned text.
 *
 * DEPENDENCY SIMPLIFICATION:
 * ==========================
 * This component does NOT inject text cleaning dependencies.
 * Callers are responsible for pre-cleaning datasets before passing them here.
 *
 * BENEFITS:
 * =========
 * ✅ Single Responsibility - only handles feature extraction
 * ✅ No circular dependencies - clean dependency graph
 * ✅ Testable - easy to test with mock cleaned data
 * ✅ Reusable - works with any text cleaning strategy
 *
 * RESPONSIBILITIES:
 * =================
 * 1. Configure Weka filters (StringToWordVector, Normalize)
 * 2. Train filters on pre-cleaned text during fit()
 * 3. Transform pre-cleaned text during inference
 * 4. Provide feature extraction analysis and statistics
 *
 * THREAD SAFETY (via FilterTrainingTemplate):
 * ===========================================
 * - Training phase: write lock (exclusive access)
 * - Inference phase: read lock (concurrent reads safe)
 * - State machine: UNINITIALIZED → TRAINING → READY
 */
@Component
public class TFIDFFeatureExtractor extends FilterTrainingTemplate<Instances> {

    private static final String VERSION = "1.0.0";
    private static final Logger logger = LoggerFactory.getLogger(TFIDFFeatureExtractor.class);

    // Immutable configuration
    private final int maxFeatures;
    private final int minTermFreq;
    private final double maxTermFreq;
    private final boolean useTfIdf;
    private final boolean useBigrams;
    private final boolean useTrigrams;
    private final boolean normalizeFeatures;
    private final boolean outputWordCounts;

    // REFACTORED: No text cleaning dependency - accepts pre-cleaned text
    // Caller is responsible for cleaning datasets before passing to this component

    // Trained filters (mutable, but immutable after training)
    private StringToWordVector stringToWordVectorFilter;
    private Normalize normalizationFilter;

    // Training metadata
    private VocabularyStats vocabularyStats;
    private Instances trainingStructure;

    @Autowired
    public TFIDFFeatureExtractor(
            @Value("${sentiment.features.max-features:5000}") int maxFeatures,
            @Value("${sentiment.features.min-term-freq:2}") int minTermFreq,
            @Value("${sentiment.features.max-term-freq:0.9}") double maxTermFreq,
            @Value("${sentiment.features.use-tfidf:true}") boolean useTfIdf,
            @Value("${sentiment.features.use-bigrams:true}") boolean useBigrams,
            @Value("${sentiment.features.use-trigrams:false}") boolean useTrigrams,
            @Value("${sentiment.features.normalize-features:true}") boolean normalizeFeatures,
            @Value("${sentiment.features.output-word-counts:true}") boolean outputWordCounts) {

        validateConfiguration(maxFeatures, minTermFreq, maxTermFreq);

        this.maxFeatures = maxFeatures;
        this.minTermFreq = minTermFreq;
        this.maxTermFreq = maxTermFreq;
        this.useTfIdf = useTfIdf;
        this.useBigrams = useBigrams;
        this.useTrigrams = useTrigrams;
        this.normalizeFeatures = normalizeFeatures;
        this.outputWordCounts = outputWordCounts;

        logger.info("TFIDFFeatureExtractor initialized (accepts pre-cleaned text). " +
                        "Configuration: maxFeatures={}, minTermFreq={}, useTfIdf={}, useBigrams={}",
                maxFeatures, minTermFreq, useTfIdf, useBigrams);
    }

    // ==================== TEMPLATE METHOD IMPLEMENTATIONS ====================

    /**
     * SUBCLASS HOOK: Implement TF-IDF specific training logic.
     * Called by base class fit() within write lock.
     */
    @Override
    protected Instances doFit(List<Dataset> datasets) throws Exception {
        logger.info("Training TF-IDF filters on {} samples (expects pre-cleaned text)", datasets.size());

        // Step 1: Create raw Weka instances (assumes datasets contain pre-cleaned text)
        Instances rawInstances = createRawInstances(datasets);
        this.trainingStructure = new Instances(rawInstances, 0);  // Store structure

        // Step 2: Train StringToWordVector filter
        trainFeatureFilter(rawInstances);

        // Step 3: Apply TF-IDF transformation
        Instances tfidfInstances = Filter.useFilter(rawInstances, stringToWordVectorFilter);

        // Step 4: Train normalization filter if enabled
        Instances finalInstances;
        if (normalizeFeatures) {
            trainNormalizationFilter(tfidfInstances);
            finalInstances = Filter.useFilter(tfidfInstances, normalizationFilter);
        } else {
            finalInstances = tfidfInstances;
        }

        // Step 5: Generate statistics
        generateFeatureStats(finalInstances);

        logger.info("TF-IDF training complete. Vocabulary: {} terms", vocabularyStats.numFeatures);
        return finalInstances;
    }

    /**
     * SUBCLASS HOOK: Clear TF-IDF specific resources.
     * Called by base class reset() within write lock.
     */
    @Override
    protected void doClearResources() {
        stringToWordVectorFilter = null;
        normalizationFilter = null;
        vocabularyStats = null;
        trainingStructure = null;
    }

    /**
     * SUBCLASS HOOK: Provide logger instance to base class.
     */
    @Override
    protected Logger getLogger() {
        return logger;
    }

    // ==================== INFERENCE METHODS (THREAD-SAFE) ====================

    /**
     * Transform a single pre-cleaned text using trained filters.
     * Thread-safe after fit() completes.
     *
     * @param cleanedText Pre-cleaned text (caller must clean before passing)
     */
    public Instance transform(String cleanedText) {
        ValidationUtils.requireNonEmpty(cleanedText);

        // Use base class executeInference for thread-safe execution
        return executeFilterInference(() -> {
            logger.debug("INFERENCE: Transforming pre-cleaned text (thread-safe): '{}'",
                    cleanedText.substring(0, Math.min(50, cleanedText.length())));

            Instances singleInstanceSet = createSingleInstanceSet(cleanedText);

            // Apply trained filters (thread-safe: Filter.useFilter creates new Instances)
            Instances transformed = Filter.useFilter(singleInstanceSet, stringToWordVectorFilter);

            if (normalizeFeatures && normalizationFilter != null) {
                transformed = Filter.useFilter(transformed, normalizationFilter);
            }

            return transformed.instance(0);
        });
    }

    /**
     * Transform multiple pre-cleaned texts using trained filters.
     * Thread-safe batch transformation.
     *
     * @param cleanedTexts Pre-cleaned texts (caller must clean before passing)
     */
    public Instances transform(List<String> cleanedTexts) {
        if (cleanedTexts == null || cleanedTexts.isEmpty()) {
            throw new IllegalArgumentException("Texts cannot be null or empty");
        }

        return executeFilterInference(() -> {
            logger.info("INFERENCE: Batch transforming {} pre-cleaned texts (thread-safe)", cleanedTexts.size());

            // Create batch instances from pre-cleaned texts
            List<Dataset> tempDatasets = cleanedTexts.stream()
                    .map(text -> new Dataset.Builder(text, Dataset.SentimentLabel.NEUTRAL).build())
                    .collect(Collectors.toList());

            Instances rawInstances = createRawInstances(tempDatasets);

            // Apply trained filters
            Instances transformed = Filter.useFilter(rawInstances, stringToWordVectorFilter);
            if (normalizeFeatures && normalizationFilter != null) {
                transformed = Filter.useFilter(transformed, normalizationFilter);
            }

            logger.info("INFERENCE complete: {} texts -> {} features",
                    cleanedTexts.size(), transformed.numAttributes() - 1);

            return transformed;
        });
    }

    // ==================== TRAINING HELPER METHODS ====================

    private Instances createRawInstances(List<Dataset> datasets) {
        logger.debug("Creating raw Weka instances from {} pre-cleaned datasets", datasets.size());

        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("text", (ArrayList<String>) null));

        ArrayList<String> classValues = new ArrayList<>(Arrays.asList("positive", "negative", "neutral"));
        attributes.add(new Attribute("class_label", classValues));

        Instances instances = new Instances("SentimentDataRaw", attributes, datasets.size());
        instances.setClassIndex(instances.numAttributes() - 1);

        for (Dataset dataset : datasets) {
            DenseInstance instance = new DenseInstance(2);
            instance.setValue(attributes.get(0), dataset.getText());
            instance.setValue(attributes.get(1), dataset.getSentiment().getDisplayName());
            instance.setDataset(instances);
            instances.add(instance);
        }

        return instances;
    }

    private Instances createSingleInstanceSet(String cleanedText) {
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("text", (ArrayList<String>) null));

        ArrayList<String> classValues = new ArrayList<>(Arrays.asList("positive", "negative", "neutral", "unknown"));
        attributes.add(new Attribute("class_label", classValues));

        Instances singleSet = new Instances("SingleText", attributes, 1);
        singleSet.setClassIndex(singleSet.numAttributes() - 1);

        DenseInstance instance = new DenseInstance(2);
        instance.setValue(attributes.get(0), cleanedText);
        instance.setValue(attributes.get(1), "unknown");
        instance.setDataset(singleSet);
        singleSet.add(instance);

        return singleSet;
    }

    private void trainFeatureFilter(Instances rawInstances) throws Exception {
        logger.info("Training TF-IDF filter: maxFeatures={}, minTermFreq={}, useTfIdf={}",
                maxFeatures, minTermFreq, useTfIdf);

        stringToWordVectorFilter = new StringToWordVector();
        stringToWordVectorFilter.setAttributeIndices("first");
        stringToWordVectorFilter.setWordsToKeep(maxFeatures);
        stringToWordVectorFilter.setMinTermFreq(minTermFreq);
        stringToWordVectorFilter.setDoNotOperateOnPerClassBasis(false);
        stringToWordVectorFilter.setLowerCaseTokens(true);
        stringToWordVectorFilter.setOutputWordCounts(outputWordCounts);

        if (useTfIdf) {
            stringToWordVectorFilter.setTFTransform(true);
            stringToWordVectorFilter.setIDFTransform(true);
            logger.debug("TF-IDF transformation enabled");
        }

        if (useBigrams || useTrigrams) {
            NGramTokenizer ngramTokenizer = new NGramTokenizer();
            ngramTokenizer.setNGramMinSize(1);

            if (useTrigrams) {
                ngramTokenizer.setNGramMaxSize(3);
                logger.debug("N-gram tokenizer configured for unigrams, bigrams, and trigrams");
            } else if (useBigrams) {
                ngramTokenizer.setNGramMaxSize(2);
                logger.debug("N-gram tokenizer configured for unigrams and bigrams");
            }

            stringToWordVectorFilter.setTokenizer(ngramTokenizer);
        }

        stringToWordVectorFilter.setInputFormat(rawInstances);
        logger.info("Feature filter training complete");
    }

    private void trainNormalizationFilter(Instances tfidfInstances) throws Exception {
        logger.info("Training Normalize filter");
        normalizationFilter = new Normalize();
        normalizationFilter.setInputFormat(tfidfInstances);
        logger.info("Normalize filter trained");
    }

    private void generateFeatureStats(Instances instances) {
        int numFeatures = instances.numAttributes() - 1;
        int numInstances = instances.numInstances();

        int totalValues = numFeatures * numInstances;
        int nonZeroValues = 0;

        for (int i = 0; i < numInstances; i++) {
            Instance instance = instances.instance(i);
            for (int j = 0; j < numFeatures; j++) {
                if (instance.value(j) != 0) {
                    nonZeroValues++;
                }
            }
        }

        double sparsity = totalValues > 0 ? 1.0 - (double) nonZeroValues / totalValues : 0.0;

        vocabularyStats = new VocabularyStats(
                numFeatures, numInstances, sparsity,
                calculateAverageDocumentLength(instances),
                calculateVocabularyDistribution(instances)
        );

        logger.info("Feature statistics: {}", vocabularyStats);
    }

    private double calculateAverageDocumentLength(Instances instances) {
        if (instances.numInstances() == 0) return 0.0;

        double totalLength = 0;
        int numFeatures = instances.numAttributes() - 1;

        for (int i = 0; i < instances.numInstances(); i++) {
            Instance instance = instances.instance(i);
            int documentLength = 0;

            for (int j = 0; j < numFeatures; j++) {
                if (instance.value(j) > 0) {
                    documentLength++;
                }
            }
            totalLength += documentLength;
        }

        return totalLength / instances.numInstances();
    }

    private Map<String, Double> calculateVocabularyDistribution(Instances instances) {
        Map<String, Double> distribution = new HashMap<>();
        int numFeatures = instances.numAttributes() - 1;

        if (numFeatures == 0) return distribution;

        double[] termCounts = new double[numFeatures];

        for (int i = 0; i < instances.numInstances(); i++) {
            Instance instance = instances.instance(i);
            for (int j = 0; j < numFeatures; j++) {
                if (instance.value(j) > 0) {
                    termCounts[j]++;
                }
            }
        }

        double numDocs = instances.numInstances();
        Arrays.sort(termCounts);

        distribution.put("min_doc_freq", termCounts[0] / numDocs);
        distribution.put("max_doc_freq", termCounts[numFeatures - 1] / numDocs);
        distribution.put("median_doc_freq", termCounts[numFeatures / 2] / numDocs);
        distribution.put("mean_doc_freq", Arrays.stream(termCounts).average().orElse(0.0) / numDocs);

        return distribution;
    }

    // ==================== CONFIGURATION AND UTILITIES ====================

    private void validateConfiguration(int maxFeatures, int minTermFreq, double maxTermFreq) {
        if (maxFeatures <= 0) {
            throw new IllegalArgumentException("maxFeatures must be > 0, got: " + maxFeatures);
        }
        if (minTermFreq < 1) {
            throw new IllegalArgumentException("minTermFreq must be >= 1, got: " + minTermFreq);
        }
        if (maxTermFreq <= 0 || maxTermFreq > 1.0) {
            throw new IllegalArgumentException("maxTermFreq must be in (0, 1], got: " + maxTermFreq);
        }
    }

    public String getVersion() {
        return VERSION;
    }

    public FeatureConfig getConfiguration() {
        return new FeatureConfig(maxFeatures, minTermFreq, maxTermFreq,
                useTfIdf, useBigrams, useTrigrams, normalizeFeatures, outputWordCounts);
    }

    public VocabularyStats getVocabularyStats() {
        // Thread-safe: read within inference context if needed
        return executeFilterInference(() -> vocabularyStats);
    }

    @Override
    protected String getSubclassDiagnostics() {
        StringBuilder diag = new StringBuilder();
        diag.append("=== TFIDFFeatureExtractor Specific Diagnostics ===\n");
        diag.append(String.format("StringToWordVector filter: %s\n",
                stringToWordVectorFilter != null ? "trained" : "null"));
        diag.append(String.format("Normalization filter: %s\n",
                normalizationFilter != null ? "trained" : "null"));
        diag.append(String.format("Configuration: maxFeatures=%d, minTermFreq=%d, useTfIdf=%s\n",
                maxFeatures, minTermFreq, useTfIdf));
        if (vocabularyStats != null) {
            diag.append(String.format("Vocabulary stats: %s\n", vocabularyStats));
        }
        return diag.toString();
    }

    // ==================== ANALYSIS METHODS ====================

    /**
     * Extract features with comprehensive analysis from pre-cleaned datasets.
     * Thread-safe after fit() completes.
     *
     * @param datasets Datasets with pre-cleaned text
     */
    public FeatureExtractionResult extractFeaturesWithAnalysis(List<Dataset> datasets) {
        if (datasets == null || datasets.isEmpty()) {
            throw new IllegalArgumentException("Datasets cannot be null or empty");
        }

        return executeFilterInference(() -> {
            logger.info("Extracting features with analysis for {} pre-cleaned datasets", datasets.size());

            // Transform using trained filters (expects pre-cleaned text)
            List<String> cleanedTexts = datasets.stream()
                    .map(Dataset::getText)
                    .collect(Collectors.toList());
            Instances features = transform(cleanedTexts);

            // Perform analysis
            VocabularyAnalysis vocabAnalysis = analyzeVocabulary(features);
            FeatureQualityMetrics qualityMetrics = assessFeatureQuality(features);

            return new FeatureExtractionResult(features, vocabAnalysis,
                    qualityMetrics, vocabularyStats);
        });
    }

    /**
     * Analyze vocabulary composition.
     */
    private VocabularyAnalysis analyzeVocabulary(Instances instances) {
        int numFeatures = instances.numAttributes() - 1;
        List<String> topFeatures = new ArrayList<>();

        for (int i = 0; i < Math.min(20, numFeatures); i++) {
            topFeatures.add(instances.attribute(i).name());
        }

        long unigramCount = topFeatures.stream()
                .filter(name -> !name.contains(" "))
                .count();
        long bigramCount = topFeatures.stream()
                .filter(name -> name.split(" ").length == 2)
                .count();
        long trigramCount = topFeatures.stream()
                .filter(name -> name.split(" ").length == 3)
                .count();

        return new VocabularyAnalysis(numFeatures, topFeatures,
                (int) unigramCount, (int) bigramCount, (int) trigramCount);
    }

    /**
     * Assess feature quality metrics.
     */
    private FeatureQualityMetrics assessFeatureQuality(Instances instances) {
        int numFeatures = instances.numAttributes() - 1;
        int numInstances = instances.numInstances();

        double[] featureDensities = new double[numFeatures];
        double[] featureVariances = new double[numFeatures];

        for (int j = 0; j < numFeatures; j++) {
            double[] values = new double[numInstances];
            for (int i = 0; i < numInstances; i++) {
                values[i] = instances.instance(i).value(j);
            }

            long nonZeroCount = Arrays.stream(values).filter(v -> v != 0).count();
            featureDensities[j] = (double) nonZeroCount / numInstances;

            double mean = Arrays.stream(values).average().orElse(0.0);
            featureVariances[j] = Arrays.stream(values)
                    .map(v -> Math.pow(v - mean, 2))
                    .average().orElse(0.0);
        }

        double avgDensity = Arrays.stream(featureDensities).average().orElse(0.0);
        double avgVariance = Arrays.stream(featureVariances).average().orElse(0.0);

        int highVarianceFeatures = (int) Arrays.stream(featureVariances)
                .filter(v -> v > avgVariance * 1.5).count();

        return new FeatureQualityMetrics(avgDensity, avgVariance, highVarianceFeatures);
    }

    /**
     * Demonstrate feature extraction with sample data.
     */
    public void demonstrateFeatureExtraction(List<String> sampleTexts, List<String> sentiments) {
        if (sampleTexts.size() != sentiments.size()) {
            throw new IllegalArgumentException("Sample texts and sentiments must have same size");
        }

        logger.info("=== TF-IDF Feature Extraction Demonstration ===");
        logger.info("Current filter state: {}", getFilterState());

        List<Dataset> sampleDatasets = new ArrayList<>();
        for (int i = 0; i < sampleTexts.size(); i++) {
            Dataset.SentimentLabel sentiment = Dataset.SentimentLabel.fromString(sentiments.get(i));
            sampleDatasets.add(new Dataset.Builder(sampleTexts.get(i), sentiment)
                    .id("demo_" + i)
                    .build());
        }

        FeatureExtractionResult result = extractFeaturesWithAnalysis(sampleDatasets);

        logger.info("Sample data: {} texts", sampleTexts.size());
        logger.info("Features extracted: {}", result.instances.numAttributes() - 1);
        logger.info("Final filter state: {}", getFilterState());
        logger.info("Vocabulary analysis: {}", result.vocabularyAnalysis);
        logger.info("Quality metrics: {}", result.qualityMetrics);
        logger.info("Vocabulary stats: {}", result.vocabularyStats);

        for (int i = 0; i < Math.min(3, result.instances.numInstances()); i++) {
            Instance instance = result.instances.instance(i);
            logger.info("Sample {}: {} non-zero features", i, countNonZeroFeatures(instance));
        }

        logger.info("=== End Feature Extraction Demonstration ===");
    }

    private int countNonZeroFeatures(Instance instance) {
        int count = 0;
        for (int i = 0; i < instance.numAttributes() - 1; i++) {
            if (instance.value(i) != 0) {
                count++;
            }
        }
        return count;
    }

    // ==================== DATA CLASSES ====================

    public static class FeatureExtractionResult {
        public final Instances instances;
        public final VocabularyAnalysis vocabularyAnalysis;
        public final FeatureQualityMetrics qualityMetrics;
        public final VocabularyStats vocabularyStats;

        public FeatureExtractionResult(Instances instances, VocabularyAnalysis vocabularyAnalysis,
                                       FeatureQualityMetrics qualityMetrics, VocabularyStats vocabularyStats) {
            this.instances = instances;
            this.vocabularyAnalysis = vocabularyAnalysis;
            this.qualityMetrics = qualityMetrics;
            this.vocabularyStats = vocabularyStats;
        }
    }

    public static class VocabularyAnalysis {
        public final int totalFeatures;
        public final List<String> topFeatures;
        public final int unigramCount;
        public final int bigramCount;
        public final int trigramCount;

        public VocabularyAnalysis(int totalFeatures, List<String> topFeatures,
                                  int unigramCount, int bigramCount, int trigramCount) {
            this.totalFeatures = totalFeatures;
            this.topFeatures = topFeatures;
            this.unigramCount = unigramCount;
            this.bigramCount = bigramCount;
            this.trigramCount = trigramCount;
        }

        @Override
        public String toString() {
            return String.format("VocabAnalysis{features=%d, unigrams=%d, bigrams=%d, trigrams=%d, top=%s}",
                    totalFeatures, unigramCount, bigramCount, trigramCount,
                    topFeatures.stream().limit(5).collect(Collectors.toList()));
        }
    }

    public static class FeatureQualityMetrics {
        public final double averageDensity;
        public final double averageVariance;
        public final int highVarianceFeatures;

        public FeatureQualityMetrics(double averageDensity, double averageVariance, int highVarianceFeatures) {
            this.averageDensity = averageDensity;
            this.averageVariance = averageVariance;
            this.highVarianceFeatures = highVarianceFeatures;
        }

        @Override
        public String toString() {
            return String.format("QualityMetrics{density=%.3f, variance=%.3f, highVarFeatures=%d}",
                    averageDensity, averageVariance, highVarianceFeatures);
        }
    }

    public static class VocabularyStats {
        public final int numFeatures;
        public final int numDocuments;
        public final double sparsity;
        public final double averageDocumentLength;
        public final Map<String, Double> distribution;

        public VocabularyStats(int numFeatures, int numDocuments, double sparsity,
                               double averageDocumentLength, Map<String, Double> distribution) {
            this.numFeatures = numFeatures;
            this.numDocuments = numDocuments;
            this.sparsity = sparsity;
            this.averageDocumentLength = averageDocumentLength;
            this.distribution = distribution;
        }

        @Override
        public String toString() {
            return String.format("VocabStats{features=%d, docs=%d, sparsity=%.3f, avgDocLen=%.1f}",
                    numFeatures, numDocuments, sparsity, averageDocumentLength);
        }
    }

    public static class FeatureConfig {
        public final int maxFeatures;
        public final int minTermFreq;
        public final double maxTermFreq;
        public final boolean useTfIdf;
        public final boolean useBigrams;
        public final boolean useTrigrams;
        public final boolean normalizeFeatures;
        public final boolean outputWordCounts;

        public FeatureConfig(int maxFeatures, int minTermFreq, double maxTermFreq,
                             boolean useTfIdf, boolean useBigrams, boolean useTrigrams,
                             boolean normalizeFeatures, boolean outputWordCounts) {
            this.maxFeatures = maxFeatures;
            this.minTermFreq = minTermFreq;
            this.maxTermFreq = maxTermFreq;
            this.useTfIdf = useTfIdf;
            this.useBigrams = useBigrams;
            this.useTrigrams = useTrigrams;
            this.normalizeFeatures = normalizeFeatures;
            this.outputWordCounts = outputWordCounts;
        }

        @Override
        public String toString() {
            return String.format("FeatureConfig{maxFeatures=%d, minTermFreq=%d, maxTermFreq=%.2f, " +
                            "tfidf=%s, bigrams=%s, trigrams=%s, normalize=%s, wordCounts=%s}",
                    maxFeatures, minTermFreq, maxTermFreq, useTfIdf, useBigrams,
                    useTrigrams, normalizeFeatures, outputWordCounts);
        }
    }

    // ==================== LEGACY API (DEPRECATED) ====================

    /**
     * @deprecated Use explicit fit() then transform() workflow instead.
     */
    @Deprecated
    public Instances extractFeatures(List<Dataset> datasets) {
        logger.warn("DEPRECATED: extractFeatures() mixes training and inference. " +
                "Use fit() for training, then transform() for inference.");

        if (!isReady()) {
            return fit(datasets);
        }

        // Already trained, just transform
        List<String> texts = datasets.stream()
                .map(Dataset::getText)
                .collect(Collectors.toList());
        return transform(texts);
    }

    /**
     * @deprecated Use transform(text) instead
     */
    @Deprecated
    public Instance transformSingleText(String text) {
        logger.warn("DEPRECATED: Use transform(text) instead");
        return transform(text);
    }
}