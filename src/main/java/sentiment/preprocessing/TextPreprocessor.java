package sentiment.preprocessing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.context.annotation.Scope;
import org.springframework.beans.factory.annotation.Autowired;
import sentiment.config.FeatureExtractionProperties;
import sentiment.data.Dataset;
import sentiment.util.ValidationUtils;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Stateful text preprocessing pipeline with thread-safe training and inference.
 * <p>
 * Use {@link #fit(List)} to train the pipeline on labeled data, then use
 * {@link #transform(String)} for inference. Training captures vocabulary statistics
 * and uses mutual information for feature selection. All methods are thread-safe
 * using read-write locks.
 * <p>
 * For fine-grained control over individual preprocessing steps, use
 * {@link #cleanText(String)}, {@link #tokenize(String)}, and
 * {@link #removeStopwords(List)} independently.
 * <p>
 * <b>Bean Scope:</b> Prototype - each training run gets a fresh instance to avoid
 * state conflicts when training multiple models.
 */
@Component
@Scope("prototype")
public class TextPreprocessor {

    private static final String VERSION = "1.0.0";
    private static final Logger logger = LoggerFactory.getLogger(TextPreprocessor.class);

    // Immutable configuration parameters - set via constructor
    private final int minWordLength;
    private final boolean preserveEmoticons;

    // Clean dependency injection
    private final ContractionExpander contractionExpander;
    private final AdvancedTokenizer advancedTokenizer;
    private final IntelligentStopwordRemover stopwordRemover;
    private final FeatureExtractionProperties featureConfig;

    // Pipeline state management - THREAD-SAFE with ReadWriteLock
    private final ReadWriteLock stateLock = new ReentrantReadWriteLock();
    private volatile boolean isFitted = false;
    private PipelineState pipelineState;

    // Compiled regex patterns for efficiency
    private static final Pattern URL_PATTERN = Pattern.compile(
            "https?://[-\\w.]+(?:[:\\d]+)?(?:/[\\w/_.]*(?:\\?[\\w&=%.]*)?(?:#[\\w.]*)?)?",
            Pattern.CASE_INSENSITIVE
    );

    private static final Pattern EMAIL_PATTERN = Pattern.compile(
            "\\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}\\b",
            Pattern.CASE_INSENSITIVE
    );

    private static final Pattern MENTION_PATTERN = Pattern.compile("@[a-zA-Z0-9_]+");
    private static final Pattern HASHTAG_PATTERN = Pattern.compile("#[a-zA-Z0-9_]+");
    private static final Pattern HTML_PATTERN = Pattern.compile("<[^>]+>");
    private static final Pattern EXTRA_WHITESPACE_PATTERN = Pattern.compile("\\s+");
    private static final Pattern REPEATED_CHARS_PATTERN = Pattern.compile("(.)\\1{3,}");

    // Emoticon patterns - preserve these for sentiment analysis
    private static final Pattern POSITIVE_EMOTICONS = Pattern.compile(
            "[:;=8][)\\]}>D]|[)\\]}>D][:;=8]|:\\)|;\\)|:D|:P|\\^_\\^|\\^\\^|<3"
    );
    private static final Pattern NEGATIVE_EMOTICONS = Pattern.compile(
            "[:;=8][({<\\[]|[({<\\[][:;=8]|:\\(|;\\(|D:|</3|>:\\(|:\\["
    );

    /**
     * Creates a text preprocessor with the specified dependencies and configuration.
     *
     * @param contractionExpander expands contractions before tokenization
     * @param advancedTokenizer tokenizes cleaned text
     * @param stopwordRemover filters stopwords from tokens
     * @param featureConfig shared feature extraction configuration
     * @throws IllegalArgumentException if configuration is invalid
     */
    @Autowired
    public TextPreprocessor(
            ContractionExpander contractionExpander,
            AdvancedTokenizer advancedTokenizer,
            IntelligentStopwordRemover stopwordRemover,
            FeatureExtractionProperties featureConfig) {

        // Validate configuration at construction time
        int minWordLength = featureConfig.getMinWordLength();
        if (minWordLength < 1) {
            throw new IllegalArgumentException(
                    "Invalid minWordLength: " + minWordLength + ". Must be >= 1");
        }

        this.contractionExpander = contractionExpander;
        this.advancedTokenizer = advancedTokenizer;
        this.stopwordRemover = stopwordRemover;
        this.featureConfig = featureConfig;
        this.minWordLength = minWordLength;
        this.preserveEmoticons = true; // Default behavior for sentiment analysis
        this.pipelineState = new PipelineState();

        logger.info("TextPreprocessor initialized. Thread safety: ReadWriteLock. " +
                        "Configuration: minWordLength={}, maxFeatures={}, useTfidf={}, useBigrams={}",
                minWordLength, featureConfig.getMaxFeatures(),
                featureConfig.isUseTfidf(), featureConfig.isUseBigrams());
    }

    /**
     * Trains the preprocessing pipeline on labeled data.
     * Captures vocabulary statistics and applies mutual information-based feature selection.
     * Must be called before using {@link #transform(String)}.
     *
     * @param data training datasets with text and labels
     * @throws IllegalArgumentException if data is null or empty
     */
    public void fit(List<Dataset> data) {
        if (data == null || data.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }

        stateLock.writeLock().lock();  // WRITE LOCK for training
        try {
            if (isFitted) {
                logger.warn("Pipeline already fitted. Refitting with new data.");
            }

            logger.info("Fitting preprocessing pipeline on {} samples", data.size());
            long startTime = System.currentTimeMillis();

            try {
                // Step 1: Preprocess all training texts
                List<String> preprocessedTexts = new ArrayList<>();
                for (Dataset dataset : data) {
                    String preprocessed = preprocessText(dataset.getText());
                    preprocessedTexts.add(preprocessed);
                }

                // Step 2: Extract and store vocabulary statistics with MI-based feature selection
                pipelineState.captureVocabularyStatsWithPrincipledSelection(
                        preprocessedTexts, data, featureConfig.getMiSelectionThreshold());

                // Step 3: Store configuration
                pipelineState.storeConfiguration(this);

                isFitted = true;
                long duration = System.currentTimeMillis() - startTime;

                logger.info("Pipeline fitting completed in {}ms. Vocabulary size: {}, Fitted: {}",
                        duration, pipelineState.vocabularySize, isFitted);

            } catch (Exception e) {
                logger.error("Pipeline fitting failed: {}", e.getMessage(), e);
                throw new RuntimeException("Failed to fit preprocessing pipeline", e);
            }

        } finally {
            stateLock.writeLock().unlock();  // Always release write lock
        }
    }

    /**
     * Transforms text through the fitted preprocessing pipeline.
     * Thread-safe for concurrent inference after training.
     *
     * @param text input text to preprocess
     * @return preprocessed text with cleaning, tokenization, and stopword removal applied
     * @throws IllegalStateException if pipeline not fitted
     */
    public String transform(String text) {
        if (ValidationUtils.isNullOrEmpty(text)) {
            logger.debug("Received null or empty text for transformation");
            return "";
        }

        stateLock.readLock().lock();  // READ LOCK for concurrent inference
        try {
            if (!isFitted) {
                throw new IllegalStateException(
                        "Pipeline must be fitted before transforming text. Call fit() first.");
            }

            logger.debug("Transforming text through fitted pipeline: '{}'",
                    text.substring(0, Math.min(50, text.length())));

            return preprocessText(text);

        } finally {
            stateLock.readLock().unlock();  // Always release read lock
        }
    }

    /**
     * Cleans text by removing URLs, emails, HTML tags, and normalizing special characters.
     * Expands contractions and preserves emoticons if configured.
     *
     * @param rawText raw input text
     * @return cleaned text ready for tokenization
     */
    public String cleanText(String rawText) {
        if (rawText == null || rawText.trim().isEmpty()) {
            logger.warn("Received null or empty text for cleaning");
            return "";
        }

        String cleaned = rawText;

        // URLs and emails - use lowercase tokens to survive lowercasing step
        cleaned = URL_PATTERN.matcher(cleaned).replaceAll(" url_token ");
        cleaned = EMAIL_PATTERN.matcher(cleaned).replaceAll(" email_token ");

        // Social media - use lowercase tokens
        cleaned = MENTION_PATTERN.matcher(cleaned).replaceAll(" mention_token ");
        cleaned = HASHTAG_PATTERN.matcher(cleaned).replaceAll(match ->
                " hashtag_" + match.group().substring(1).toLowerCase() + " ");

        // Emoticons (preserve if configured) - use lowercase tokens
        if (preserveEmoticons) {
            cleaned = POSITIVE_EMOTICONS.matcher(cleaned).replaceAll(" positive_emoticon ");
            cleaned = NEGATIVE_EMOTICONS.matcher(cleaned).replaceAll(" negative_emoticon ");
        }

        // HTML tags
        cleaned = HTML_PATTERN.matcher(cleaned).replaceAll(" ");

        // Expand contractions before lowercasing
        // (Order doesn't affect final tokens, but maintains readable intermediate output)
        cleaned = contractionExpander.expand(cleaned);

        // Lowercase for normalization
        cleaned = cleaned.toLowerCase();

        // Repeated characters
        cleaned = REPEATED_CHARS_PATTERN.matcher(cleaned).replaceAll("$1$1");

        // Let tokenizer handle punctuation intelligently - it's designed to preserve
        // sentiment-bearing punctuation (!, ?, etc.) while filtering noise

        // Whitespace
        cleaned = EXTRA_WHITESPACE_PATTERN.matcher(cleaned).replaceAll(" ");

        return cleaned.trim();
    }

    /**
     * Tokenizes cleaned text into individual words.
     *
     * @param cleanedText cleaned text from {@link #cleanText(String)}
     * @return list of tokens
     */
    public List<String> tokenize(String cleanedText) {
        if (cleanedText == null || cleanedText.trim().isEmpty()) {
            return new ArrayList<>();
        }

        return advancedTokenizer.tokenize(cleanedText);
    }

    /**
     * Removes stopwords from tokenized text.
     *
     * @param tokens list of tokens from {@link #tokenize(String)}
     * @return filtered tokens with stopwords removed
     */
    public List<String> removeStopwords(List<String> tokens) {
        if (tokens == null || tokens.isEmpty()) {
            return new ArrayList<>();
        }

        return stopwordRemover.removeStopwords(tokens);
    }

    /**
     * Internal helper that chains the complete preprocessing pipeline.
     * Used by both fit() and transform() to ensure consistency.
     *
     * @param rawText raw input text
     * @return fully preprocessed text
     */
    private String preprocessText(String rawText) {
        if (rawText == null || rawText.trim().isEmpty()) {
            return "";
        }

        String cleaned = cleanText(rawText);
        List<String> tokens = tokenize(cleaned);
        List<String> filtered = removeStopwords(tokens);

        return String.join(" ", filtered);
    }

    /**
     * Saves the fitted pipeline state to disk for later reuse.
     *
     * @param path file path to save state
     * @throws IOException if writing fails
     * @throws IllegalStateException if pipeline not fitted
     */
    public void saveState(Path path) throws IOException {
        stateLock.readLock().lock();  // READ LOCK for safe state access
        try {
            if (!isFitted) {
                throw new IllegalStateException("Cannot save unfitted pipeline");
            }

            logger.info("Saving pipeline state to: {}", path);

            Files.createDirectories(path.getParent());

            try (ObjectOutputStream oos = new ObjectOutputStream(
                    new BufferedOutputStream(Files.newOutputStream(path)))) {

                oos.writeObject(VERSION);
                oos.writeObject(pipelineState);
                oos.writeBoolean(isFitted);
                oos.writeInt(minWordLength);
                oos.writeBoolean(preserveEmoticons);

                logger.info("Pipeline state saved successfully");
            }

        } finally {
            stateLock.readLock().unlock();
        }
    }

    /**
     * Loads a previously saved pipeline state from disk.
     *
     * @param path file path to load state from
     * @throws IOException if reading fails or file doesn't exist
     * @throws IllegalArgumentException if file doesn't exist
     */
    public void loadState(Path path) throws IOException {
        if (!Files.exists(path)) {
            throw new IllegalArgumentException("State file does not exist: " + path);
        }

        stateLock.writeLock().lock();  // WRITE LOCK for state modification
        try {
            logger.info("Loading pipeline state from: {}", path);

            try (ObjectInputStream ois = new ObjectInputStream(
                    new BufferedInputStream(Files.newInputStream(path)))) {

                String savedVersion = (String) ois.readObject();
                pipelineState = (PipelineState) ois.readObject();
                isFitted = ois.readBoolean();

                int savedMinWordLength = ois.readInt();
                boolean savedPreserveEmoticons = ois.readBoolean();

                if (!VERSION.equals(savedVersion)) {
                    logger.warn("Loaded version {} differs from current version {}",
                               savedVersion, VERSION);
                }

                if (savedMinWordLength != this.minWordLength) {
                    logger.warn("Loaded minWordLength {} differs from current config {}",
                               savedMinWordLength, this.minWordLength);
                }

                if (savedPreserveEmoticons != this.preserveEmoticons) {
                    logger.warn("Loaded preserveEmoticons {} differs from current config {}",
                               savedPreserveEmoticons, this.preserveEmoticons);
                }

                logger.info("Pipeline state loaded. Vocabulary: {}, Fitted: {}",
                        pipelineState.vocabularySize, isFitted);

            } catch (ClassNotFoundException e) {
                throw new IOException("Incompatible pipeline state format", e);
            }

        } finally {
            stateLock.writeLock().unlock();
        }
    }

    /**
     * Resets the pipeline to an unfitted state, clearing all learned vocabulary statistics.
     */
    public void reset() {
        stateLock.writeLock().lock();  // WRITE LOCK for state reset
        try {
            isFitted = false;
            pipelineState = new PipelineState();
            logger.info("Pipeline reset to unfitted state");
        } finally {
            stateLock.writeLock().unlock();
        }
    }

    /**
     * Returns the pipeline version.
     *
     * @return version string
     */
    public String getVersion() {
        return VERSION;
    }

    /**
     * Checks if the pipeline has been trained and is ready for inference.
     *
     * @return true if fitted, false otherwise
     */
    public boolean isFitted() {
        return isFitted;  // volatile read is safe
    }

    /**
     * Returns the current pipeline state including vocabulary statistics.
     *
     * @return pipeline state snapshot
     */
    public PipelineState getPipelineState() {
        stateLock.readLock().lock();  // READ LOCK for safe access
        try {
            return pipelineState;
        } finally {
            stateLock.readLock().unlock();
        }
    }

    /**
     * Returns version information for all pipeline components.
     *
     * @return comprehensive version info
     */
    @SuppressWarnings("unused") // Public API for monitoring and diagnostics
    public PipelineVersionInfo getVersionInfo() {
        return new PipelineVersionInfo(
                VERSION,
                contractionExpander.getVersion(),
                advancedTokenizer.getVersion(),
                stopwordRemover.getVersion()
        );
    }

    /**
     * Computes preprocessing statistics for the given datasets.
     *
     * @param data datasets to analyze
     * @return preprocessing statistics including word counts and analysis
     */
    @SuppressWarnings("unused") // Public API for pipeline analysis and monitoring
    public PreprocessingStats getPreprocessingStats(List<Dataset> data) {
        if (data == null || data.isEmpty()) {
            return new PreprocessingStats(0, 0, 0, 0, new CleaningMetrics(), null, null);
        }

        int totalTexts = data.size();
        int totalOriginalWords = 0;
        int totalCleanedWords = 0;
        int totalFilteredWords = 0;
        CleaningMetrics aggregatedMetrics = new CleaningMetrics();

        String sampleText = data.get(0).getText();
        AdvancedTokenizer.TokenizationAnalysis tokenAnalysis =
                advancedTokenizer.analyzeTokenization(cleanText(sampleText));

        IntelligentStopwordRemover.StopwordAnalysis stopwordAnalysis =
                stopwordRemover.analyzeStopwordRemoval(tokenize(cleanText(sampleText)));

        for (Dataset dataset : data) {
            String originalText = dataset.getText();
            String cleanedText = cleanText(originalText);
            List<String> tokens = tokenize(cleanedText);
            List<String> filtered = removeStopwords(tokens);

            totalOriginalWords += originalText.split("\\s+").length;
            totalCleanedWords += tokens.size();
            totalFilteredWords += filtered.size();
        }

        return new PreprocessingStats(
                totalTexts,
                totalOriginalWords,
                totalCleanedWords,
                totalFilteredWords,
                aggregatedMetrics,
                tokenAnalysis,
                stopwordAnalysis
        );
    }

    /**
     * Returns a summary of the pipeline configuration and component statistics.
     *
     * @return pipeline summary
     */
    public PipelineSummary getPipelineSummary() {
        return new PipelineSummary(
                this.getClass().getSimpleName(),
                contractionExpander.getStats(),
                advancedTokenizer.getClass().getSimpleName(),
                stopwordRemover.getConfigurationSummary()
        );
    }

    /**
     * Serializable pipeline state for persistence.
     * Contains vocabulary statistics and configuration from training.
     */
    public static class PipelineState implements Serializable {
        @java.io.Serial
        private static final long serialVersionUID = 1L;

        public int vocabularySize = 0;
        public Map<String, Integer> vocabularyFrequencies = new HashMap<>();
        public long fittingTimestamp = 0;
        public int trainingSampleCount = 0;

        // Configuration snapshot
        public int minWordLength = 2;
        public boolean preserveEmoticons = true;

        /**
         * @deprecated Use {@link #captureVocabularyStatsWithPrincipledSelection(List, List)} instead.
         */
        @Deprecated
        public void captureVocabularyStats(List<String> preprocessedTexts) {
            Set<String> uniqueTokens = new HashSet<>();
            Map<String, Integer> frequencies = new HashMap<>();

            for (String text : preprocessedTexts) {
                String[] tokens = text.split("\\s+");
                for (String token : tokens) {
                    if (!token.isEmpty()) {
                        uniqueTokens.add(token);
                        frequencies.merge(token, 1, Integer::sum);
                    }
                }
            }

            this.vocabularySize = uniqueTokens.size();
            this.vocabularyFrequencies = frequencies;
            this.fittingTimestamp = System.currentTimeMillis();
            this.trainingSampleCount = preprocessedTexts.size();
        }

        /**
         * Captures vocabulary statistics using mutual information-based feature selection.
         * Selects features that maximize discriminative power rather than raw frequency.
         *
         * @param preprocessedTexts preprocessed text samples
         * @param originalDatasets original datasets with labels
         * @param miThreshold threshold for triggering MI-based feature selection
         * @throws IllegalArgumentException if sizes don't match
         */
        public void captureVocabularyStatsWithPrincipledSelection(
                List<String> preprocessedTexts,
                List<Dataset> originalDatasets,
                int miThreshold) {

            if (preprocessedTexts.size() != originalDatasets.size()) {
                throw new IllegalArgumentException(
                    "Preprocessed texts and datasets must have same size");
            }

            logger.info("Computing vocabulary statistics with MI-based feature selection");

            // Step 1: Precompute document token sets (EFFICIENCY FIX)
            // This avoids 500M substring scans for large vocabularies
            List<Set<String>> docTokenSets = new ArrayList<>(preprocessedTexts.size());
            Map<String, Integer> frequencies = new HashMap<>();

            for (String text : preprocessedTexts) {
                String[] tokens = text.split("\\s+");
                Set<String> tokenSet = new HashSet<>();

                for (String token : tokens) {
                    if (!token.isEmpty()) {
                        tokenSet.add(token);
                        frequencies.merge(token, 1, Integer::sum);
                    }
                }

                docTokenSets.add(tokenSet);
            }

            logger.info("Initial vocabulary size: {}", frequencies.size());

            // Step 2: Compute mutual information for each term
            Map<String, Double> mutualInformation = new HashMap<>();
            for (String term : frequencies.keySet()) {
                double mi = computeMutualInformation(term, docTokenSets, originalDatasets);
                mutualInformation.put(term, mi);
            }

            // Step 3: Apply vocabulary size limit with MI-based selection
            // Threshold is configurable via sentiment.features.mi-selection-threshold (default: 50000)
            if (frequencies.size() > miThreshold) {
                logger.warn("Vocabulary size {} exceeds limit {}. Selecting top-{} by mutual information",
                           frequencies.size(), miThreshold, miThreshold);

                // Select top-k features by INFORMATION GAIN, not frequency
                this.vocabularyFrequencies = mutualInformation.entrySet().stream()
                    .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                    .limit(miThreshold)
                    .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        e -> frequencies.get(e.getKey()),  // Store frequency, not MI
                        (e1, e2) -> e1,
                        LinkedHashMap::new
                    ));

                this.vocabularySize = miThreshold;

                // Step 4: Report information loss
                double totalMI = mutualInformation.values().stream()
                    .mapToDouble(Double::doubleValue).sum();
                double retainedMI = vocabularyFrequencies.keySet().stream()
                    .mapToDouble(mutualInformation::get).sum();

                logger.info("Feature selection retained {}",
                           String.format("%.2f%% of mutual information (%.4f / %.4f)",
                               100.0 * retainedMI / totalMI, retainedMI, totalMI));

                // Report some statistics about discarded features
                long discardedFeatures = frequencies.size() - miThreshold;
                logger.info("Discarded {} features with low discriminative power", discardedFeatures);

            } else {
                // No feature selection needed
                this.vocabularySize = frequencies.size();
                this.vocabularyFrequencies = frequencies;
                logger.info("Vocabulary size {} within limit. No feature selection applied.",
                           this.vocabularySize);
            }

            this.fittingTimestamp = System.currentTimeMillis();
            this.trainingSampleCount = preprocessedTexts.size();
        }

        /**
         * Computes mutual information between a term and class labels using multi-class formula.
         * Higher scores indicate greater discriminative power.
         *
         * @param term term to evaluate
         * @param docTokenSets precomputed token sets for each document
         * @param originalDatasets original datasets with labels
         * @return mutual information score
         */
        private double computeMutualInformation(
                String term,
                List<Set<String>> docTokenSets,
                List<Dataset> originalDatasets) {

            int totalDocs = docTokenSets.size();

            // Build contingency table: counts[termPresence][classLabel]
            Map<String, Integer> classGivenTerm = new HashMap<>();
            Map<String, Integer> classGivenNoTerm = new HashMap<>();
            Map<String, Integer> classTotal = new HashMap<>();
            int withTerm = 0;
            int withoutTerm = 0;

            for (int i = 0; i < totalDocs; i++) {
                boolean hasTerm = docTokenSets.get(i).contains(term);
                String label = originalDatasets.get(i).getSentiment().toString();

                classTotal.merge(label, 1, Integer::sum);

                if (hasTerm) {
                    withTerm++;
                    classGivenTerm.merge(label, 1, Integer::sum);
                } else {
                    withoutTerm++;
                    classGivenNoTerm.merge(label, 1, Integer::sum);
                }
            }

            // Compute MI using correct multi-class formula:
            // MI(T; C) = Σₜ Σ_c P(t,c) log(P(t,c) / (P(t) P(c)))
            double mi = 0.0;

            // Sum over all cells in the 2×k contingency table
            for (String classLabel : classTotal.keySet()) {
                // Cell: (term present, this class)
                int n_t1_c = classGivenTerm.getOrDefault(classLabel, 0);
                if (n_t1_c > 0) {
                    double p_t1_c = (double) n_t1_c / totalDocs;  // P(t=1, c)
                    double p_t1 = (double) withTerm / totalDocs;   // P(t=1)
                    double p_c = (double) classTotal.get(classLabel) / totalDocs; // P(c)
                    mi += p_t1_c * Math.log(p_t1_c / (p_t1 * p_c));
                }

                // Cell: (term absent, this class)
                int n_t0_c = classGivenNoTerm.getOrDefault(classLabel, 0);
                if (n_t0_c > 0) {
                    double p_t0_c = (double) n_t0_c / totalDocs;  // P(t=0, c)
                    double p_t0 = (double) withoutTerm / totalDocs; // P(t=0)
                    double p_c = (double) classTotal.get(classLabel) / totalDocs; // P(c)
                    mi += p_t0_c * Math.log(p_t0_c / (p_t0 * p_c));
                }
            }

            return Math.max(0.0, mi);  // MI should be non-negative
        }


        /**
         * Stores configuration snapshot from the preprocessor.
         *
         * @param preprocessor preprocessor to extract configuration from
         */
        public void storeConfiguration(TextPreprocessor preprocessor) {
            this.minWordLength = preprocessor.minWordLength;
            this.preserveEmoticons = preprocessor.preserveEmoticons;
        }

        @Override
        public String toString() {
            return String.format("PipelineState{vocab=%d, samples=%d, timestamp=%d}",
                    vocabularySize, trainingSampleCount, fittingTimestamp);
        }
    }

    public record PreprocessingStats(
            int totalTexts,
            int originalWords,
            int cleanedWords,
            int filteredWords,
            CleaningMetrics cleaningMetrics,
            AdvancedTokenizer.TokenizationAnalysis tokenizationAnalysis,
            IntelligentStopwordRemover.StopwordAnalysis stopwordAnalysis) {

        @Override
        public String toString() {
            return String.format(
                    "PreprocessingStats{texts=%d, originalWords=%d, cleanedWords=%d, filteredWords=%d, " +
                            "cleaning=%s, tokenization=%s, stopwords=%s}",
                    totalTexts, originalWords, cleanedWords, filteredWords, cleaningMetrics,
                    tokenizationAnalysis, stopwordAnalysis
            );
        }

        @SuppressWarnings("unused") // Public API for statistics analysis
        public double getCompressionRatio() {
            return originalWords > 0 ? (double) filteredWords / originalWords : 0.0;
        }

        @SuppressWarnings("unused") // Public API for statistics analysis
        public double getTokenizationEfficiency() {
            return originalWords > 0 ? (double) cleanedWords / originalWords : 0.0;
        }

        @SuppressWarnings("unused") // Public API for statistics analysis
        public double getStopwordFilteringRatio() {
            return cleanedWords > 0 ? (double) filteredWords / cleanedWords : 0.0;
        }
    }

    public record PipelineSummary(
            String preprocessorName,
            ContractionExpander.ContractionStats contractionStats,
            String tokenizerName,
            String stopwordConfig) {
    }

    public static class CleaningMetrics {
        public int urlsFound = 0;
        public int emailsFound = 0;
        public int mentionsFound = 0;
        public int hashtagsFound = 0;
        public int htmlTagsFound = 0;
        public boolean contractionsExpanded = false;
        public boolean repeatedCharsFound = false;
        public int positiveMoods = 0;
        public int negativeMoods = 0;

        @Override
        public String toString() {
            return String.format(
                    "CleaningMetrics{urls=%d, emails=%d, mentions=%d, hashtags=%d, html=%d, " +
                            "contractions=%s, repeated=%s, +mood=%d, -mood=%d}",
                    urlsFound, emailsFound, mentionsFound, hashtagsFound, htmlTagsFound,
                    contractionsExpanded, repeatedCharsFound, positiveMoods, negativeMoods
            );
        }
    }

    public record PipelineVersionInfo(
            String preprocessorVersion,
            String contractionExpanderVersion,
            String tokenizerVersion,
            String stopwordRemoverVersion) {

        /**
         * Returns a compact version string for the entire pipeline.
         *
         * @return compact version string
         */
        @SuppressWarnings("unused") // Public API for version reporting
        public String getCompactVersion() {
            return String.format("Pipeline-v%s (CE:%s|TOK:%s|SW:%s)",
                    preprocessorVersion, contractionExpanderVersion, tokenizerVersion,
                    stopwordRemoverVersion);
        }
    }

    /**
     * Sets the fitted state for model loading.
     * This is called when a model is loaded from disk to mark the preprocessor as ready.
     *
     * @param fitted whether the preprocessor should be marked as fitted
     */
    public void setFitted(boolean fitted) {
        stateLock.writeLock().lock();
        try {
            this.isFitted = fitted;
        } finally {
            stateLock.writeLock().unlock();
        }
    }

}