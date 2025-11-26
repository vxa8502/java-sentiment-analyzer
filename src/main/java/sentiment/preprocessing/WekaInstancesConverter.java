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
 * Converts preprocessed text to Weka Instances for machine learning.
 * <p>
 * Configures and trains TF-IDF vectorization and normalization filters,
 * then transforms text into numeric feature vectors. Supports stratified
 * train/test splitting and provides conversion statistics.
 * <p>
 * Thread-safe for concurrent inference after training completes.
 *
 * @see FilterTrainingTemplate for state management and thread safety
 */
@Component
public class WekaInstancesConverter extends FilterTrainingTemplate<Instances> {

    private static final String VERSION = "1.0.0";
    private static final Logger logger = LoggerFactory.getLogger(WekaInstancesConverter.class);

    // Immutable configuration
    private final int maxFeatures;
    private final int minTermFreq;
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

    @Autowired
    public WekaInstancesConverter(
            TextPreprocessor textPreprocessor,
            @Value("${sentiment.weka.max-features:5000}") int maxFeatures,
            @Value("${sentiment.weka.min-term-freq:2}") int minTermFreq,
            @Value("${sentiment.weka.use-tfidf:true}") boolean useTfIdf,
            @Value("${sentiment.weka.use-bigrams:true}") boolean useBigrams,
            @Value("${sentiment.weka.normalize-features:true}") boolean normalizeFeatures,
            @Value("${sentiment.weka.output-word-counts:false}") boolean outputWordCounts) {

        validateConfiguration(maxFeatures, minTermFreq);

        this.textPreprocessor = textPreprocessor;
        this.maxFeatures = maxFeatures;
        this.minTermFreq = minTermFreq;
        this.useTfIdf = useTfIdf;
        this.useBigrams = useBigrams;
        this.normalizeFeatures = normalizeFeatures;
        this.outputWordCounts = outputWordCounts;

        logger.info("WekaInstancesConverter initialized. Configuration: maxFeatures={}, " +
                        "minTermFreq={}, useTfIdf={}, useBigrams={}",
                maxFeatures, minTermFreq, useTfIdf, useBigrams);
    }

    /**
     * Trains Weka filters on provided datasets.
     * Called by base class within write lock.
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

        // Step 5: Extract vocabulary
        extractVocabulary(finalInstances);

        logger.info("Weka conversion training complete. Vocabulary: {} terms", vocabulary.size());
        return finalInstances;
    }

    /**
     * Clears trained filters and cached state.
     * Called by base class within write lock.
     */
    @Override
    protected void doClearResources() {
        trainedStringToWordFilter = null;
        trainedNormalizationFilter = null;
        filterTrainingStructure = null;
        vocabulary = null;
    }

    @Override
    protected Logger getLogger() {
        return logger;
    }

    /**
     * Transforms a single text into a Weka Instance using trained filters.
     *
     * @param text text to transform
     * @param defaultSentiment sentiment label
     * @return Weka Instance with TF-IDF features
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
     * Transforms multiple texts into Weka Instances (batch operation).
     *
     * @param texts texts to transform
     * @param defaultSentiment sentiment label
     * @return Weka Instances with TF-IDF features
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
     * Transforms datasets into Weka Instances with preserved labels.
     * Uses trained filters without retraining. Intended for test/validation data.
     *
     * @param datasets datasets to transform
     * @return Weka Instances with TF-IDF features and labels
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
     * Applies trained filters to a single instance.
     * Thread-safe (Filter.useFilter creates new Instances).
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


    private void validateConfiguration(int maxFeatures, int minTermFreq) {
        if (maxFeatures <= 0) {
            throw new IllegalArgumentException("maxFeatures must be > 0");
        }
        if (minTermFreq < 1) {
            throw new IllegalArgumentException("minTermFreq must be >= 1");
        }
    }

    public String getVersion() {
        return VERSION;
    }

    public Set<String> getVocabulary() {
        return executeFilterInference(() ->
                vocabulary != null ? new HashSet<>(vocabulary) : new HashSet<>()
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
        if (filterTrainingStructure != null) {
            diag.append(String.format("Training structure: %d attributes\n",
                    filterTrainingStructure.numAttributes()));
        }
        return diag.toString();
    }

}