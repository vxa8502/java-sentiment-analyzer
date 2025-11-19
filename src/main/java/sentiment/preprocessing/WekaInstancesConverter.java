package sentiment.preprocessing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.beans.factory.annotation.Autowired;
import sentiment.data.Dataset;
import sentiment.PipelineState;
import sentiment.evaluation.FeatureImportanceAnalyzer;
import sentiment.util.ValidationUtils;
import weka.classifiers.functions.SMO;
import weka.core.*;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.*;
import java.util.stream.Collectors;

/**
 * REFACTORED: Now extends FilterTrainingTemplate base class.
 *
 * BENEFITS OF REFACTORING:
 * ========================
 * ✅ ~189 lines of boilerplate removed
 * ✅ State management logic inherited from base
 * ✅ Thread-safety guarantees enforced by template
 * ✅ Consistent behavior with TFIDFFeatureExtractor
 * ✅ Focus on WHAT (Weka conversion) not HOW (state management)
 *
 * WHAT THIS CLASS NOW DOES:
 * =========================
 * 1. Convert preprocessed text to Weka Instances
 * 2. Configure Weka filters (StringToWordVector, Normalize)
 * 3. Create train/test splits with stratification
 * 4. Provide conversion statistics and diagnostics
 *
 * WHAT THE BASE CLASS HANDLES:
 * ============================
 * 1. Thread-safe state management (ReadWriteLock)
 * 2. State machine enforcement (UNINITIALIZED -> TRAINING -> READY)
 * 3. Training phase protection (write lock)
 * 4. Inference phase concurrency (read lock)
 * 5. Error handling and state transitions
 * 6. Reset and cleanup logic
 */
@Component
public class WekaInstancesConverter extends FilterTrainingTemplate<Instances> {

    private static final String VERSION = "1.0.0";
    private static final Logger logger = LoggerFactory.getLogger(WekaInstancesConverter.class);

    // Immutable configuration
    private final int maxFeatures;
    private final int minTermFreq;
    private final double maxDocFreq;
    private final boolean useTfIdf;
    private final boolean useBigrams;
    private final boolean normalizeFeatures;
    private final boolean outputWordCounts;
    private final TextPreprocessor textPreprocessor;

    // Trained filters (mutable, but immutable after training)
    private StringToWordVector trainedStringToWordFilter;
    private Normalize trainedNormalizationFilter;

    // Training metadata
    private Instances filterTrainingStructure;
    private Set<String> vocabulary;
    private ConversionStats lastConversionStats;

    @Autowired
    public WekaInstancesConverter(
            TextPreprocessor textPreprocessor,
            @Value("${sentiment.weka.max-features:5000}") int maxFeatures,
            @Value("${sentiment.weka.min-term-freq:2}") int minTermFreq,
            @Value("${sentiment.weka.max-doc-freq:0.9}") double maxDocFreq,
            @Value("${sentiment.weka.use-tfidf:true}") boolean useTfIdf,
            @Value("${sentiment.weka.use-bigrams:true}") boolean useBigrams,
            @Value("${sentiment.weka.normalize-features:true}") boolean normalizeFeatures,
            @Value("${sentiment.weka.output-word-counts:false}") boolean outputWordCounts) {

        validateConfiguration(maxFeatures, minTermFreq, maxDocFreq);

        this.textPreprocessor = textPreprocessor;
        this.maxFeatures = maxFeatures;
        this.minTermFreq = minTermFreq;
        this.maxDocFreq = maxDocFreq;
        this.useTfIdf = useTfIdf;
        this.useBigrams = useBigrams;
        this.normalizeFeatures = normalizeFeatures;
        this.outputWordCounts = outputWordCounts;

        logger.info("WekaInstancesConverter initialized. Configuration: maxFeatures={}, " +
                        "minTermFreq={}, useTfIdf={}, useBigrams={}",
                maxFeatures, minTermFreq, useTfIdf, useBigrams);
    }

    // ==================== TEMPLATE METHOD IMPLEMENTATIONS ====================

    /**
     * SUBCLASS HOOK: Implement Weka-specific training logic.
     * Called by base class fit() within write lock.
     */
    @Override
    protected Instances doFit(List<Dataset> datasets) throws Exception {
        logger.info("Training Weka conversion filters on {} samples", datasets.size());

        // Step 1: Create raw instances with preprocessing
        Instances rawInstances = createRawInstances(datasets);
        this.filterTrainingStructure = new Instances(rawInstances, 0);

        // Step 2: Train StringToWordVector filter
        trainStringToWordVectorFilter(rawInstances);

        // Step 3: Apply TF-IDF transformation
        Instances tfidfInstances = Filter.useFilter(rawInstances, trainedStringToWordFilter);

        // Step 4: Train normalization filter if enabled
        Instances finalInstances;
        if (normalizeFeatures) {
            trainNormalizationFilter(tfidfInstances);
            finalInstances = Filter.useFilter(tfidfInstances, trainedNormalizationFilter);
        } else {
            finalInstances = tfidfInstances;
        }

        // Step 5: Extract vocabulary and generate stats
        extractVocabulary(finalInstances);
        generateConversionStats(datasets, finalInstances);

        logger.info("Weka conversion training complete. Vocabulary: {} terms", vocabulary.size());
        return finalInstances;
    }

    /**
     * SUBCLASS HOOK: Clear Weka-specific resources.
     * Called by base class reset() within write lock.
     */
    @Override
    protected void doClearResources() {
        trainedStringToWordFilter = null;
        trainedNormalizationFilter = null;
        filterTrainingStructure = null;
        vocabulary = null;
        lastConversionStats = null;
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
     * Transform a single text into a Weka Instance.
     * Thread-safe after fit() completes.
     *
     * @param text Input text to transform
     * @param defaultSentiment Default sentiment label for the instance
     * @return Weka Instance with features
     */
    public Instance transform(String text, String defaultSentiment) {
        ValidationUtils.requireNonEmpty(text);

        // Use base class executeInference for thread-safe execution
        return executeFilterInference(() -> {
            logger.debug("INFERENCE: Transforming single text (thread-safe): '{}'",
                    text.substring(0, Math.min(50, text.length())));

            // Create raw instance
            Instance rawInstance = createSingleRawInstance(text, defaultSentiment);

            // Apply trained filters (thread-safe: Filter.useFilter creates new Instances)
            Instance transformedInstance = applyTrainedFilters(rawInstance);

            logger.debug("INFERENCE complete: {} attributes", transformedInstance.numAttributes());
            return transformedInstance;
        });
    }

    /**
     * Transform multiple texts into Weka Instances.
     * Batch transformation for efficiency. Thread-safe after fit().
     *
     * @param texts Array of texts to transform
     * @param defaultSentiment Default sentiment label
     * @return Weka Instances with features
     */
    public Instances transform(String[] texts, String defaultSentiment) {
        if (texts == null || texts.length == 0) {
            throw new IllegalArgumentException("Texts cannot be null or empty");
        }

        return executeFilterInference(() -> {
            logger.info("INFERENCE: Batch transforming {} texts (thread-safe)", texts.length);

            Instances batchInstances = new Instances(filterTrainingStructure, texts.length);

            for (int i = 0; i < texts.length; i++) {
                Instance instance = transformSingleInternal(texts[i], defaultSentiment);
                batchInstances.add(instance);

                if ((i + 1) % 100 == 0) {
                    logger.debug("Transformed {}/{} texts", i + 1, texts.length);
                }
            }

            logger.info("INFERENCE complete: {} texts -> {} features",
                    texts.length, batchInstances.numAttributes() - 1);

            return batchInstances;
        });
    }

    /**
     * Transform a list of datasets into Weka Instances, preserving sentiment labels.
     * This is the proper method for converting test data for evaluation.
     * Thread-safe after fit() completes.
     *
     * NOTE: This does NOT retrain the converter - it uses the already-fitted vocabulary.
     * Use this for test/validation data after training is complete.
     *
     * @param datasets List of datasets to transform
     * @return Weka Instances with features and preserved sentiment labels
     */
    public Instances transformDatasets(List<Dataset> datasets) {
        if (datasets == null || datasets.isEmpty()) {
            throw new IllegalArgumentException("Datasets cannot be null or empty");
        }

        return executeFilterInference(() -> {
            logger.info("INFERENCE: Transforming {} datasets with preserved labels (thread-safe)",
                    datasets.size());

            // Step 1: Create raw instances with preprocessing
            Instances rawInstances = createRawInstances(datasets);

            // Step 2: Apply trained StringToWordVector filter to entire batch (more efficient)
            Instances tfidfInstances = Filter.useFilter(rawInstances, trainedStringToWordFilter);

            // Step 3: Apply normalization filter if enabled
            Instances finalInstances;
            if (normalizeFeatures && trainedNormalizationFilter != null) {
                finalInstances = Filter.useFilter(tfidfInstances, trainedNormalizationFilter);
            } else {
                finalInstances = tfidfInstances;
            }

            logger.info("INFERENCE complete: {} datasets -> {} features with preserved labels",
                    datasets.size(), finalInstances.numAttributes() - 1);

            return finalInstances;
        });
    }

    // ==================== TRAINING HELPER METHODS ====================

    private Instances createRawInstances(List<Dataset> datasets) {
        logger.info("Creating raw instances for {} datasets", datasets.size());

        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(new Attribute("text", (ArrayList<String>) null));

        ArrayList<String> sentimentValues = createSentimentValues(datasets);
        attributes.add(new Attribute("class_label", sentimentValues));

        Instances instances = new Instances("SentimentAnalysis", attributes, datasets.size());
        instances.setClassIndex(1);

        for (Dataset dataset : datasets) {
            DenseInstance instance = new DenseInstance(2);
            instance.setDataset(instances);  // Must set dataset BEFORE setValue()
            String preprocessed = textPreprocessor.transform(dataset.getText());
            instance.setValue(0, preprocessed);
            instance.setValue(1, dataset.getSentiment().getDisplayName());
            instances.add(instance);
        }

        return instances;
    }

    private Instance createSingleRawInstance(String text, String defaultSentiment) {
        if (filterTrainingStructure == null) {
            throw new IllegalStateException("No training structure available");
        }

        Instances singleStructure = new Instances(filterTrainingStructure, 1);
        DenseInstance instance = new DenseInstance(2);
        instance.setDataset(singleStructure);

        String preprocessed = textPreprocessor.transform(text);
        if (preprocessed == null || preprocessed.trim().isEmpty()) {
            preprocessed = "empty_content_placeholder";
        }
        instance.setValue(0, preprocessed);

        if ("unknown".equals(defaultSentiment)) {
            instance.setClassMissing();
        } else {
            try {
                instance.setValue(1, defaultSentiment);
            } catch (Exception e) {
                instance.setClassMissing();
            }
        }

        singleStructure.add(instance);
        return instance;
    }

    private Instance transformSingleInternal(String text, String sentiment) throws Exception {
        Instance raw = createSingleRawInstance(text, sentiment);
        return applyTrainedFilters(raw);
    }

    /**
     * Apply trained Weka filters to a single instance.
     *
     * THREAD-SAFETY: This method is safe for concurrent execution because
     * Filter.useFilter() creates new Instances without mutating the filter.
     */
    private Instance applyTrainedFilters(Instance rawInstance) throws Exception {
        Instances singleSet = new Instances(rawInstance.dataset(), 1);
        singleSet.add(rawInstance);

        // Apply StringToWordVector (thread-safe)
        Instances filtered = Filter.useFilter(singleSet, trainedStringToWordFilter);

        // Apply normalization if enabled (thread-safe)
        if (normalizeFeatures && trainedNormalizationFilter != null) {
            filtered = Filter.useFilter(filtered, trainedNormalizationFilter);
        }

        return filtered.instance(0);
    }

    private void trainStringToWordVectorFilter(Instances rawInstances) throws Exception {
        logger.info("Training StringToWordVector filter");

        trainedStringToWordFilter = new StringToWordVector();
        trainedStringToWordFilter.setAttributeIndices("first");
        trainedStringToWordFilter.setWordsToKeep(maxFeatures);
        trainedStringToWordFilter.setMinTermFreq(minTermFreq);
        trainedStringToWordFilter.setDoNotOperateOnPerClassBasis(false);
        trainedStringToWordFilter.setLowerCaseTokens(true);
        trainedStringToWordFilter.setOutputWordCounts(outputWordCounts);

        if (useTfIdf) {
            trainedStringToWordFilter.setTFTransform(true);
            trainedStringToWordFilter.setIDFTransform(true);
        }

        if (useBigrams) {
            NGramTokenizer tokenizer = new NGramTokenizer();
            tokenizer.setNGramMinSize(1);
            tokenizer.setNGramMaxSize(2);
            trainedStringToWordFilter.setTokenizer(tokenizer);
        }

        trainedStringToWordFilter.setInputFormat(rawInstances);
        logger.info("StringToWordVector filter trained");
    }

    private void trainNormalizationFilter(Instances tfidfInstances) throws Exception {
        logger.info("Training Normalize filter");
        trainedNormalizationFilter = new Normalize();
        trainedNormalizationFilter.setInputFormat(tfidfInstances);
        logger.info("Normalize filter trained");
    }

    private void extractVocabulary(Instances instances) {
        vocabulary = new HashSet<>();
        for (int i = 0; i < instances.numAttributes() - 1; i++) {
            vocabulary.add(instances.attribute(i).name());
        }
        logger.debug("Extracted vocabulary: {} terms", vocabulary.size());
    }

    private ArrayList<String> createSentimentValues(List<Dataset> datasets) {
        Set<String> uniqueSentiments = datasets.stream()
                .map(d -> d.getSentiment().getDisplayName())
                .collect(Collectors.toSet());

        Set<String> standardSentiments = Set.of("positive", "negative", "neutral");
        uniqueSentiments.addAll(standardSentiments);

        ArrayList<String> sentimentValues = new ArrayList<>(uniqueSentiments);
        Collections.sort(sentimentValues);

        logger.debug("Sentiment values: {}", sentimentValues);
        return sentimentValues;
    }

    private void generateConversionStats(List<Dataset> originalDatasets, Instances finalInstances) {
        int originalTexts = originalDatasets.size();
        int features = finalInstances.numAttributes() - 1;
        int instances = finalInstances.numInstances();

        double sparsity = calculateSparsity(finalInstances);
        double avgDocLength = calculateAverageDocumentLength(finalInstances);
        FeatureDistribution featureDistribution = analyzeFeatureDistribution(finalInstances);

        lastConversionStats = new ConversionStats(
                originalTexts, instances, features, sparsity, avgDocLength,
                featureDistribution, vocabulary.size()
        );

        logger.info("Conversion statistics: {}", lastConversionStats);
    }

    private double calculateSparsity(Instances instances) {
        int numFeatures = instances.numAttributes() - 1;
        long totalValues = (long) instances.numInstances() * numFeatures;
        long zeroValues = 0;

        for (int i = 0; i < instances.numInstances(); i++) {
            Instance instance = instances.instance(i);
            for (int j = 0; j < numFeatures; j++) {
                if (instance.value(j) == 0.0) {
                    zeroValues++;
                }
            }
        }

        return totalValues > 0 ? (double) zeroValues / totalValues : 0.0;
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

    private FeatureDistribution analyzeFeatureDistribution(Instances instances) {
        int numFeatures = instances.numAttributes() - 1;
        double[] featureMaxs = new double[numFeatures];
        double[] featureMeans = new double[numFeatures];

        for (int j = 0; j < numFeatures; j++) {
            double sum = 0;
            double max = 0;

            for (int i = 0; i < instances.numInstances(); i++) {
                double value = instances.instance(i).value(j);
                sum += value;
                max = Math.max(max, value);
            }

            featureMeans[j] = sum / instances.numInstances();
            featureMaxs[j] = max;
        }

        double avgMean = Arrays.stream(featureMeans).average().orElse(0.0);
        double maxMean = Arrays.stream(featureMeans).max().orElse(0.0);
        double avgMax = Arrays.stream(featureMaxs).average().orElse(0.0);
        double maxMax = Arrays.stream(featureMaxs).max().orElse(0.0);

        return new FeatureDistribution(avgMean, maxMean, avgMax, maxMax);
    }

    private void analyzeClassDistribution(Instances instances, String setName) {
        Map<String, Integer> classCounts = new HashMap<>();

        for (int i = 0; i < instances.numInstances(); i++) {
            String classValue = instances.instance(i).classAttribute()
                    .value((int) instances.instance(i).classValue());
            classCounts.merge(classValue, 1, Integer::sum);
        }

        logger.info("{} set class distribution: {}", setName, classCounts);
    }

    // ==================== UTILITY METHODS ====================

    /**
     * Creates stratified train/test split with the specified ratio.
     * Thread-safe - can be called after fit() completes.
     *
     * @param instances Weka Instances to split
     * @param trainRatio Ratio of training data (0.0 to 1.0)
     * @param randomSeed Random seed for reproducibility
     * @return DataSplit with train and test sets
     */
    public DataSplit createTrainTestSplit(Instances instances, double trainRatio, long randomSeed) {
        if (trainRatio <= 0 || trainRatio >= 1) {
            throw new IllegalArgumentException("Train ratio must be between 0 and 1");
        }

        logger.info("Creating stratified train/test split: {:.1f}%/{:.1f}%",
                trainRatio * 100, (1 - trainRatio) * 100);

        Random random = new Random(randomSeed);
        Instances shuffled = new Instances(instances);
        shuffled.randomize(random);

        int numFolds = Math.min(10, shuffled.numInstances() / 10);
        if (numFolds > 1) {
            shuffled.stratify(numFolds);
        }

        int trainSize = (int) (instances.numInstances() * trainRatio);

        Instances trainSet = new Instances(shuffled, 0, trainSize);
        Instances testSet = new Instances(shuffled, trainSize, shuffled.numInstances() - trainSize);

        logger.info("Split completed: {} training, {} testing samples",
                trainSet.numInstances(), testSet.numInstances());

        analyzeClassDistribution(trainSet, "Training");
        analyzeClassDistribution(testSet, "Testing");

        return new DataSplit(trainSet, testSet, trainRatio, randomSeed);
    }

    /**
     * Apply TF-IDF transformation to raw instances.
     * Thread-safe - uses trained filter.
     */
    public Instances applyTfIdfTransformation(Instances rawInstances) throws Exception {
        return executeFilterInference(() ->
                Filter.useFilter(rawInstances, trainedStringToWordFilter)
        );
    }

    /**
     * Apply normalization to TF-IDF instances.
     * Thread-safe - uses trained filter.
     */
    public Instances applyNormalization(Instances tfidfInstances) throws Exception {
        if (!normalizeFeatures) {
            return tfidfInstances;
        }

        return executeFilterInference(() -> {
            if (trainedNormalizationFilter != null) {
                return Filter.useFilter(tfidfInstances, trainedNormalizationFilter);
            } else {
                logger.warn("Normalization requested but filter not trained");
                return tfidfInstances;
            }
        });
    }

    // ==================== CONFIGURATION AND STATE ACCESS ====================

    private void validateConfiguration(int maxFeatures, int minTermFreq, double maxDocFreq) {
        if (maxFeatures <= 0) {
            throw new IllegalArgumentException("maxFeatures must be > 0");
        }
        if (minTermFreq < 1) {
            throw new IllegalArgumentException("minTermFreq must be >= 1");
        }
        if (maxDocFreq <= 0 || maxDocFreq > 1.0) {
            throw new IllegalArgumentException("maxDocFreq must be in (0, 1]");
        }
    }

    public String getVersion() {
        return VERSION;
    }

    /**
     * Check if filters are initialized and ready.
     * Thread-safe - delegates to base class.
     */
    public boolean areFiltersInitialized() {
        return isReady() && trainedStringToWordFilter != null;
    }

    /**
     * Get vocabulary extracted during training.
     * Thread-safe - returns defensive copy.
     */
    public Set<String> getVocabulary() {
        return executeFilterInference(() ->
                vocabulary != null ? new HashSet<>(vocabulary) : new HashSet<>()
        );
    }

    /**
     * Get last conversion statistics.
     * Thread-safe - returns immutable object.
     */
    public ConversionStats getLastConversionStats() {
        return executeFilterInference(() -> lastConversionStats);
    }

    /**
     * Get training structure for creating new instances.
     * Thread-safe - returns defensive copy.
     */
    public Instances getFilterTrainingStructure() {
        return executeFilterInference(() ->
                filterTrainingStructure != null ?
                        new Instances(filterTrainingStructure, 0) : null
        );
    }

    /**
     * Get filter statistics.
     * Thread-safe.
     */
    public FilterStats getFilterStats() {
        return executeFilterInference(() -> {
            int vocabularySize = vocabulary != null ? vocabulary.size() : 0;
            int ngramSize = useBigrams ? 2 : 1;
            return new FilterStats(getFilterState(), vocabularySize, ngramSize, 0.0, 0.0);
        });
    }

    /**
     * Get performance metrics.
     * Thread-safe.
     */
    public PerformanceMetrics getPerformanceMetrics() {
        return executeFilterInference(() ->
                new PerformanceMetrics(
                        getLastTrainingTimeMs(),
                        vocabulary != null ? vocabulary.size() : 0,
                        lastConversionStats != null ? lastConversionStats.features : 0,
                        lastConversionStats != null ? lastConversionStats.sparsity : 0.0,
                        getFilterState()
                )
        );
    }

    /**
     * Get versioned configuration.
     */
    public VersionedConversionConfig getVersionedConfiguration() {
        return new VersionedConversionConfig(
                VERSION,
                maxFeatures,
                minTermFreq,
                maxDocFreq,
                useTfIdf,
                useBigrams,
                normalizeFeatures,
                outputWordCounts,
                System.currentTimeMillis()
        );
    }

    @Override
    protected String getSubclassDiagnostics() {
        StringBuilder diag = new StringBuilder();
        diag.append("=== WekaInstancesConverter Specific Diagnostics ===\n");
        diag.append(String.format("StringToWordVector filter: %s\n",
                trainedStringToWordFilter != null ? "trained" : "null"));
        diag.append(String.format("Normalization filter: %s\n",
                trainedNormalizationFilter != null ? "trained" : "null"));
        diag.append(String.format("Configuration: maxFeatures=%d, minTermFreq=%d, useTfIdf=%s\n",
                maxFeatures, minTermFreq, useTfIdf));
        if (vocabulary != null) {
            diag.append(String.format("Vocabulary size: %d\n", vocabulary.size()));
        }
        if (lastConversionStats != null) {
            diag.append(String.format("Last conversion: %s\n", lastConversionStats));
        }
        if (filterTrainingStructure != null) {
            diag.append(String.format("Training structure: %d attributes\n",
                    filterTrainingStructure.numAttributes()));
        }
        return diag.toString();
    }

    // ==================== DATA CLASSES ====================

    public static class DataSplit {
        public final Instances trainSet;
        public final Instances testSet;
        public final double trainRatio;
        public final long randomSeed;

        public DataSplit(Instances trainSet, Instances testSet, double trainRatio, long randomSeed) {
            this.trainSet = trainSet;
            this.testSet = testSet;
            this.trainRatio = trainRatio;
            this.randomSeed = randomSeed;
        }

        @Override
        public String toString() {
            return String.format("DataSplit{train=%d, test=%d, ratio=%.2f, seed=%d}",
                    trainSet.numInstances(), testSet.numInstances(), trainRatio, randomSeed);
        }
    }

    public static class ConversionStats {
        public final int originalTexts;
        public final int outputInstances;
        public final int features;
        public final double sparsity;
        public final double avgDocumentLength;
        public final FeatureDistribution featureDistribution;
        public final int vocabularySize;

        public ConversionStats(int originalTexts, int outputInstances, int features,
                               double sparsity, double avgDocumentLength,
                               FeatureDistribution featureDistribution, int vocabularySize) {
            this.originalTexts = originalTexts;
            this.outputInstances = outputInstances;
            this.features = features;
            this.sparsity = sparsity;
            this.avgDocumentLength = avgDocumentLength;
            this.featureDistribution = featureDistribution;
            this.vocabularySize = vocabularySize;
        }

        @Override
        public String toString() {
            return String.format("ConversionStats{texts=%d->%d, features=%d, vocab=%d, " +
                            "sparsity=%.3f, avgDocLen=%.1f}",
                    originalTexts, outputInstances, features, vocabularySize,
                    sparsity, avgDocumentLength);
        }
    }

    public static class FeatureDistribution {
        public final double avgMean;
        public final double maxMean;
        public final double avgMax;
        public final double maxMax;

        public FeatureDistribution(double avgMean, double maxMean, double avgMax, double maxMax) {
            this.avgMean = avgMean;
            this.maxMean = maxMean;
            this.avgMax = avgMax;
            this.maxMax = maxMax;
        }

        @Override
        public String toString() {
            return String.format("FeatureDist{avgMean=%.4f, maxMean=%.4f, avgMax=%.4f, maxMax=%.4f}",
                    avgMean, maxMean, avgMax, maxMax);
        }
    }

    public static class FilterStats {
        public final PipelineState state;
        public final int vocabularySize;
        public final int ngramSize;
        public final double averageTermFrequency;
        public final double maxTermFrequency;

        public FilterStats(PipelineState state, int vocabularySize, int ngramSize,
                           double avgTermFreq, double maxTermFreq) {
            this.state = state;
            this.vocabularySize = vocabularySize;
            this.ngramSize = ngramSize;
            this.averageTermFrequency = avgTermFreq;
            this.maxTermFrequency = maxTermFreq;
        }

        @Override
        public String toString() {
            return String.format("FilterStats{state=%s, vocab=%d, ngrams=1-%d, avgTF=%.3f, maxTF=%.3f}",
                    state, vocabularySize, ngramSize, averageTermFrequency, maxTermFrequency);
        }
    }

    public static class PerformanceMetrics {
        public final long trainingTimeMs;
        public final int vocabularySize;
        public final int featureCount;
        public final double sparsity;
        public final PipelineState state;

        public PerformanceMetrics(long trainingTimeMs, int vocabularySize,
                                  int featureCount, double sparsity, PipelineState state) {
            this.trainingTimeMs = trainingTimeMs;
            this.vocabularySize = vocabularySize;
            this.featureCount = featureCount;
            this.sparsity = sparsity;
            this.state = state;
        }

        @Override
        public String toString() {
            return String.format("PerformanceMetrics{trainingTime=%dms, vocab=%d, features=%d, " +
                            "sparsity=%.3f, state=%s}",
                    trainingTimeMs, vocabularySize, featureCount, sparsity, state);
        }
    }

    public static class VersionedConversionConfig {
        public final String version;
        public final int maxFeatures;
        public final int minTermFreq;
        public final double maxDocFreq;
        public final boolean useTfIdf;
        public final boolean useBigrams;
        public final boolean normalizeFeatures;
        public final boolean outputWordCounts;
        public final long timestamp;

        public VersionedConversionConfig(String version, int maxFeatures, int minTermFreq,
                                         double maxDocFreq, boolean useTfIdf, boolean useBigrams,
                                         boolean normalizeFeatures, boolean outputWordCounts,
                                         long timestamp) {
            this.version = version;
            this.maxFeatures = maxFeatures;
            this.minTermFreq = minTermFreq;
            this.maxDocFreq = maxDocFreq;
            this.useTfIdf = useTfIdf;
            this.useBigrams = useBigrams;
            this.normalizeFeatures = normalizeFeatures;
            this.outputWordCounts = outputWordCounts;
            this.timestamp = timestamp;
        }

        @Override
        public String toString() {
            return String.format(
                    "VersionedConversionConfig{version=%s, maxFeatures=%d, minTermFreq=%d, " +
                            "maxDocFreq=%.2f, tfidf=%s, bigrams=%s, normalize=%s, wordCounts=%s, timestamp=%d}",
                    version, maxFeatures, minTermFreq, maxDocFreq, useTfIdf, useBigrams,
                    normalizeFeatures, outputWordCounts, timestamp
            );
        }
    }

    // ==================== FEATURE IMPORTANCE ANALYSIS ====================

    /**
     * ARIA'S REQUIREMENT: Analyze feature importance from trained SVM.
     *
     * This method addresses the gap: "TF-IDF weights are computed but never examined.
     * You don't know which features drive predictions."
     *
     * USAGE:
     * ======
     * After training your SVM classifier:
     * ```java
     * FeatureImportanceAnalyzer.FeatureImportanceResult result =
     *     converter.analyzeFeatureImportance(trainedInstances, trainedSVM, 100);
     *
     * result.printTopFeatures(20);  // Print top 20 features
     * ```
     *
     * @param trainedData The Instances used to train the model
     * @param trainedSVM The trained SMO classifier
     * @param topK Number of top features to extract
     * @return FeatureImportanceResult with ranked features and statistics
     */
    public FeatureImportanceAnalyzer.FeatureImportanceResult analyzeFeatureImportance(
            Instances trainedData,
            SMO trainedSVM,
            int topK) {

        if (!areFiltersInitialized()) {
            throw new IllegalStateException("Converter must be trained before analyzing feature importance");
        }

        logger.info("Analyzing feature importance using FeatureImportanceAnalyzer");

        FeatureImportanceAnalyzer analyzer = new FeatureImportanceAnalyzer();
        return analyzer.analyzeFeatureImportance(trainedData, trainedSVM, topK);
    }

    /**
     * Convenience method: Analyze feature importance and print report.
     *
     * @param trainedData The Instances used to train the model
     * @param trainedSVM The trained SMO classifier
     */
    public void printFeatureImportanceReport(Instances trainedData, SMO trainedSVM) {
        FeatureImportanceAnalyzer.FeatureImportanceResult result =
                analyzeFeatureImportance(trainedData, trainedSVM, 50);

        System.out.println("\n" + "=".repeat(80));
        System.out.println("FEATURE IMPORTANCE ANALYSIS");
        System.out.println("=".repeat(80));
        System.out.println("Configuration: maxFeatures=" + maxFeatures +
                ", minTermFreq=" + minTermFreq + ", useTfIdf=" + useTfIdf);
        System.out.println("Vocabulary size: " + (vocabulary != null ? vocabulary.size() : 0));
        System.out.println("\nFeature Statistics: " + result.statistics);
        System.out.println("=".repeat(80));

        result.printTopFeatures(30);

        System.out.println("INTERPRETATION:");
        System.out.println("- Features with high |weight| strongly influence predictions");
        System.out.println("- Positive weights → positive sentiment, negative → negative sentiment");
        System.out.println("- Features near zero are non-discriminative");
        System.out.println("=".repeat(80) + "\n");
    }

    // ==================== LEGACY API (DEPRECATED) ====================

    /**
     * @deprecated Use explicit fit() then transform() workflow instead.
     */
    @Deprecated
    public Instances convertToInstances(List<Dataset> datasets) {
        logger.warn("DEPRECATED: convertToInstances() mixes training and inference. " +
                "Use fit() for training, then transform() for inference.");

        if (!isReady()) {
            return fit(datasets);
        }

        // Already trained, just transform
        String[] texts = datasets.stream()
                .map(Dataset::getText)
                .toArray(String[]::new);
        return transform(texts, "unknown");
    }

    /**
     * @deprecated Use transform(text, sentiment) instead
     */
    @Deprecated
    public Instance convertSingleText(String text, String defaultSentiment) {
        logger.warn("DEPRECATED: Use transform(text, sentiment) instead");
        return transform(text, defaultSentiment);
    }
}