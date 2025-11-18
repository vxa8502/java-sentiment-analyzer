package sentiment.preprocessing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Lazy;
import org.springframework.stereotype.Component;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.beans.factory.annotation.Autowired;
import sentiment.data.Dataset;
import sentiment.util.ValidationUtils;
import weka.core.Instances;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.util.stream.Collectors;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Text preprocessing pipeline with stateful training and thread safety.
 *
 * REFACTORED: Circular dependency eliminated
 * ===========================================
 * TFIDFFeatureExtractor no longer injects this class.
 * This class provides pre-cleaned datasets to feature extractors.
 *
 * WORKFLOW:
 * =========
 * 1. fit(data) - Train pipeline on raw data (cleans + captures vocab stats)
 * 2. transform(text) - Apply cleaning to new text (thread-safe inference)
 * 3. preprocessDatasets(datasets) - Bulk cleaning for feature extraction
 *
 * THREAD SAFETY:
 * ==============
 * - Training phase: stateLock.writeLock() - exclusive access
 * - Inference phase: stateLock.readLock() - concurrent reads safe
 */
@Component
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
     * Constructor with dependency injection.
     *
     * Dependencies:
     * - contractionExpander: Expands contractions ("don't" → "do not")
     * - advancedTokenizer: Advanced tokenization logic
     * - stopwordRemover: Intelligent stopword filtering
     */
    @Autowired
    public TextPreprocessor(
            ContractionExpander contractionExpander,
            AdvancedTokenizer advancedTokenizer,
            IntelligentStopwordRemover stopwordRemover,
            @Value("${sentiment.preprocessing.min-word-length:2}") int minWordLength,
            @Value("${sentiment.preprocessing.preserve-emoticons:true}") boolean preserveEmoticons) {

        // Validate configuration at construction time
        if (minWordLength < 1) {
            throw new IllegalArgumentException(
                    "Invalid minWordLength: " + minWordLength + ". Must be >= 1");
        }

        this.contractionExpander = contractionExpander;
        this.advancedTokenizer = advancedTokenizer;
        this.stopwordRemover = stopwordRemover;
        this.minWordLength = minWordLength;
        this.preserveEmoticons = preserveEmoticons;
        this.pipelineState = new PipelineState();

        logger.info("TextPreprocessor initialized. Thread safety: ReadWriteLock. " +
                        "Configuration: minWordLength={}, preserveEmoticons={}",
                minWordLength, preserveEmoticons);
    }

    // ==================== TRAINING PHASE ====================

    /**
     * TRAINING PHASE: Fit the preprocessing pipeline on training data.
     *
     * This method trains any stateful components and captures vocabulary statistics.
     * Must be called ONCE before using transform().
     *
     * Thread safety: Uses WRITE lock - exclusive access during training
     *
     * @param data Training dataset
     * @throws IllegalArgumentException if data is null or empty
     * @throws IllegalStateException if pipeline is already fitted
     */
    public void fit(List<Dataset> data) {
        if (data == null || data.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }

        stateLock.writeLock().lock();  // ✅ WRITE LOCK for training
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
                    String cleaned = cleanText(dataset.getText());
                    List<String> tokens = tokenize(cleaned);
                    List<String> filtered = removeStopwords(tokens);
                    preprocessedTexts.add(String.join(" ", filtered));
                }

                // Step 2: Extract and store vocabulary statistics with MI-based feature selection
                pipelineState.captureVocabularyStatsWithPrincipledSelection(preprocessedTexts, data);

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
            stateLock.writeLock().unlock();  // ✅ Always release write lock
        }
    }

    // ==================== INFERENCE PHASE ====================

    /**
     * INFERENCE PHASE: Transform a single text through the fitted pipeline.
     *
     * Thread-safe after fit() completes. Must call fit() before using this method.
     *
     * Thread safety: Uses READ lock - concurrent reads safe!
     *
     * @param text Input text to transform
     * @return Preprocessed text
     * @throws IllegalStateException if pipeline not fitted
     * @throws IllegalArgumentException if text is null or empty
     */
    public String transform(String text) {
        if (ValidationUtils.isNullOrEmpty(text)) {
            logger.debug("Received null or empty text for transformation");
            return "";
        }

        stateLock.readLock().lock();  // ✅ READ LOCK for concurrent inference
        try {
            if (!isFitted) {
                throw new IllegalStateException(
                        "Pipeline must be fitted before transforming text. Call fit() first.");
            }

            logger.debug("Transforming text through fitted pipeline: '{}'",
                    text.substring(0, Math.min(50, text.length())));

            return preprocessText(text);

        } finally {
            stateLock.readLock().unlock();  // ✅ Always release read lock
        }
    }

    // ==================== CORE PREPROCESSING METHODS ====================

    /**
     * Clean text by removing noise, URLs, HTML, etc.
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

        // Expand contractions BEFORE lowercase
        cleaned = contractionExpander.expand(cleaned);

        // Lowercase
        cleaned = cleaned.toLowerCase();

        // Repeated characters
        cleaned = REPEATED_CHARS_PATTERN.matcher(cleaned).replaceAll("$1$1");

        // Special characters
        cleaned = cleaned.replaceAll("[^a-zA-Z0-9\\s_]", " ");

        // Whitespace
        cleaned = EXTRA_WHITESPACE_PATTERN.matcher(cleaned).replaceAll(" ");

        return cleaned.trim();
    }

    /**
     * Tokenize using AdvancedTokenizer
     */
    public List<String> tokenize(String cleanedText) {
        if (cleanedText == null || cleanedText.trim().isEmpty()) {
            return new ArrayList<>();
        }

        return advancedTokenizer.tokenize(cleanedText);
    }

    /**
     * Remove stopwords using IntelligentStopwordRemover
     */
    public List<String> removeStopwords(List<String> tokens) {
        if (tokens == null || tokens.isEmpty()) {
            return new ArrayList<>();
        }

        return stopwordRemover.removeStopwords(tokens);
    }

    /**
     * Complete preprocessing pipeline
     */
    public String preprocessText(String rawText) {
        if (rawText == null || rawText.trim().isEmpty()) {
            return "";
        }

        String cleaned = cleanText(rawText);
        List<String> tokens = tokenize(cleaned);
        List<String> filtered = removeStopwords(tokens);

        return String.join(" ", filtered);
    }

    // ==================== DATASET PREPROCESSING ====================

    /**
     * Preprocess multiple datasets - returns new datasets with cleaned text.
     * This is the primary method for preparing data for feature extraction.
     *
     * @param rawDatasets Datasets with raw text
     * @return New datasets with pre-cleaned text
     */
    public List<Dataset> preprocessDatasets(List<Dataset> rawDatasets) {
        if (rawDatasets == null || rawDatasets.isEmpty()) {
            throw new IllegalArgumentException("Datasets cannot be null or empty");
        }

        logger.info("Preprocessing {} datasets for feature extraction", rawDatasets.size());

        return rawDatasets.stream()
                .map(dataset -> {
                    String cleanedText = preprocessText(dataset.getText());
                    return new Dataset.Builder(cleanedText, dataset.getSentiment())
                            .id(dataset.getId())
                            .confidence(dataset.getConfidence())
                            .source(dataset.getSource())
                            .timestamp(dataset.getTimestamp())
                            .build();
                })
                .collect(Collectors.toList());
    }

    // ==================== STATE MANAGEMENT ====================

    /**
     * Save pipeline state to disk
     */
    public void saveState(Path path) throws IOException {
        stateLock.readLock().lock();  // ✅ READ LOCK for safe state access
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
     * Load pipeline state from disk
     */
    public void loadState(Path path) throws IOException {
        if (!Files.exists(path)) {
            throw new IllegalArgumentException("State file does not exist: " + path);
        }

        stateLock.writeLock().lock();  // ✅ WRITE LOCK for state modification
        try {
            logger.info("Loading pipeline state from: {}", path);

            try (ObjectInputStream ois = new ObjectInputStream(
                    new BufferedInputStream(Files.newInputStream(path)))) {

                String savedVersion = (String) ois.readObject();
                pipelineState = (PipelineState) ois.readObject();
                isFitted = ois.readBoolean();

                int savedMinWordLength = ois.readInt();
                boolean savedPreserveEmoticons = ois.readBoolean();

                if (savedMinWordLength != this.minWordLength) {
                    logger.warn("Loaded minWordLength differs from current config");
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
     * Reset pipeline to unfitted state
     */
    public void reset() {
        stateLock.writeLock().lock();  // ✅ WRITE LOCK for state reset
        try {
            isFitted = false;
            pipelineState = new PipelineState();
            logger.info("Pipeline reset to unfitted state");
        } finally {
            stateLock.writeLock().unlock();
        }
    }

    // ==================== STATE ACCESS ====================

    public String getVersion() {
        return VERSION;
    }

    /**
     * Check if pipeline is fitted and ready for transformation
     */
    public boolean isFitted() {
        return isFitted;  // volatile read is safe
    }

    /**
     * Get pipeline state information (thread-safe)
     */
    public PipelineState getPipelineState() {
        stateLock.readLock().lock();  // ✅ READ LOCK for safe access
        try {
            return pipelineState;
        } finally {
            stateLock.readLock().unlock();
        }
    }

    /**
     * Get comprehensive version information for all pipeline components
     */
    public PipelineVersionInfo getVersionInfo() {
        return new PipelineVersionInfo(
                VERSION,
                contractionExpander.getVersion(),
                advancedTokenizer.getVersion(),
                stopwordRemover.getVersion()
        );
    }

    /**
     * Get comprehensive preprocessing statistics using all components
     */
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
     * Process a single text through the complete pipeline
     */
    public String processSingleText(String rawText) {
        if (rawText == null || rawText.trim().isEmpty()) {
            return "";
        }

        logger.debug("Processing single text through integrated pipeline");
        String result = preprocessText(rawText);

        logger.debug("Single text processing complete: '{}' -> '{}'",
                rawText.substring(0, Math.min(30, rawText.length())),
                result.substring(0, Math.min(30, result.length())));

        return result;
    }

    /**
     * Get the current preprocessing pipeline summary
     */
    public PipelineSummary getPipelineSummary() {
        return new PipelineSummary(
                this.getClass().getSimpleName(),
                contractionExpander.getStats(),
                advancedTokenizer.getClass().getSimpleName(),
                stopwordRemover.getConfigurationSummary()
        );
    }

    // ==================== PIPELINE STATE CLASS ====================

    /**
     * Serializable pipeline state for persistence
     */
    public static class PipelineState implements Serializable {
        private static final long serialVersionUID = 1L;

        public int vocabularySize = 0;
        public Map<String, Integer> vocabularyFrequencies = new HashMap<>();
        public long fittingTimestamp = 0;
        public int trainingSampleCount = 0;

        // Configuration snapshot
        public int minWordLength = 2;
        public boolean preserveEmoticons = true;

        /**
         * DEPRECATED: Frequency-based vocabulary capture without feature selection.
         * Use captureVocabularyStatsWithPrincipledSelection() instead.
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
         * Principled vocabulary capture with Mutual Information-based feature selection.
         *
         * THEORETICAL FOUNDATION:
         * ======================
         * Mutual Information I(X;Y) measures how much knowing feature X reduces uncertainty about class Y.
         *
         * I(X;Y) = H(Y) - H(Y|X)
         * where:
         *   H(Y) = entropy of class distribution
         *   H(Y|X) = conditional entropy of class given feature
         *
         * This is provably optimal for feature selection because:
         * 1. It directly quantifies discriminative power
         * 2. It's invariant to feature scaling
         * 3. It handles both positive and negative class associations
         *
         * Unlike frequency-based selection which discards rare but potentially discriminative terms,
         * MI-based selection preserves features that reduce classification uncertainty.
         *
         * @param preprocessedTexts Preprocessed text samples
         * @param originalDatasets Original datasets with labels
         */
        public void captureVocabularyStatsWithPrincipledSelection(
                List<String> preprocessedTexts,
                List<Dataset> originalDatasets) {

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
            final int MAX_VOCAB_SIZE = 50000;

            if (frequencies.size() > MAX_VOCAB_SIZE) {
                logger.warn("Vocabulary size {} exceeds limit {}. Selecting top-{} by mutual information",
                           frequencies.size(), MAX_VOCAB_SIZE, MAX_VOCAB_SIZE);

                // Select top-k features by INFORMATION GAIN, not frequency
                this.vocabularyFrequencies = mutualInformation.entrySet().stream()
                    .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                    .limit(MAX_VOCAB_SIZE)
                    .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        e -> frequencies.get(e.getKey()),  // Store frequency, not MI
                        (e1, e2) -> e1,
                        LinkedHashMap::new
                    ));

                this.vocabularySize = MAX_VOCAB_SIZE;

                // Step 4: Report information loss
                double totalMI = mutualInformation.values().stream()
                    .mapToDouble(Double::doubleValue).sum();
                double retainedMI = vocabularyFrequencies.keySet().stream()
                    .mapToDouble(mutualInformation::get).sum();

                logger.info("Feature selection retained {:.2f}% of mutual information ({:.4f} / {:.4f})",
                           100.0 * retainedMI / totalMI, retainedMI, totalMI);

                // Report some statistics about discarded features
                long discardedFeatures = frequencies.size() - MAX_VOCAB_SIZE;
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
         * Compute Mutual Information between a term and class labels.
         *
         * I(term; class) = H(class) - H(class|term)
         *
         * This quantifies how much knowing whether the term appears reduces uncertainty
         * about the class label.
         *
         * MATHEMATICAL DERIVATION:
         * ========================
         * MI = Σ_x Σ_y P(x,y) * log(P(x,y) / (P(x) * P(y)))
         *
         * where:
         *   x ∈ {term present, term absent}
         *   y ∈ {positive, negative, neutral, ...}
         *
         * @param term The term to evaluate
         * @param docTokenSets Precomputed token sets for each document (CORRECTNESS FIX)
         * @param originalDatasets Original datasets with labels
         * @return Mutual information score (higher = more discriminative)
         */
        private double computeMutualInformation(
                String term,
                List<Set<String>> docTokenSets,
                List<Dataset> originalDatasets) {

            int totalDocs = docTokenSets.size();

            // Count documents with/without term for each class
            int withTerm = 0;
            int withoutTerm = 0;
            Map<String, Integer> classGivenTerm = new HashMap<>();
            Map<String, Integer> classGivenNoTerm = new HashMap<>();
            Map<String, Integer> classTotal = new HashMap<>();

            for (int i = 0; i < totalDocs; i++) {
                // CORRECTNESS FIX: Use proper token matching, not substring matching
                // Previously: preprocessedTexts.get(i).contains(term)
                // This was incorrect because "good" would match "goodness", "goody", etc.
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

            // Compute mutual information
            double mi = 0.0;

            // For each class
            for (String classLabel : classTotal.keySet()) {
                int n11 = classGivenTerm.getOrDefault(classLabel, 0);  // term present, class present
                int n10 = withTerm - n11;                               // term present, class absent
                int n01 = classGivenNoTerm.getOrDefault(classLabel, 0); // term absent, class present
                int n00 = withoutTerm - n01;                            // term absent, class absent

                // Compute MI contribution for this class
                mi += computeMIComponent(n11, n10, n01, n00, totalDocs);
            }

            return Math.max(0.0, mi);  // MI should be non-negative
        }

        /**
         * Compute MI component for a 2x2 contingency table.
         *
         * This implements the formula:
         * MI = Σ_i Σ_j (n_ij / N) * log((N * n_ij) / (row_i * col_j))
         *
         * @param n11 Count of (term present, class present)
         * @param n10 Count of (term present, class absent)
         * @param n01 Count of (term absent, class present)
         * @param n00 Count of (term absent, class absent)
         * @param total Total number of documents
         * @return MI contribution from this contingency table
         */
        private double computeMIComponent(int n11, int n10, int n01, int n00, int total) {
            double mi = 0.0;

            // Row and column totals
            int row1 = n11 + n10;  // term present
            int row0 = n01 + n00;  // term absent
            int col1 = n11 + n01;  // class present
            int col0 = n10 + n00;  // class absent

            // Add each cell's contribution to MI
            mi += addMITerm(n11, row1, col1, total);
            mi += addMITerm(n10, row1, col0, total);
            mi += addMITerm(n01, row0, col1, total);
            mi += addMITerm(n00, row0, col0, total);

            return mi;
        }

        /**
         * Add a single term to the MI calculation with proper handling of edge cases.
         *
         * Handles the case where n_ij = 0 (since 0 * log(0) = 0 by convention in information theory)
         *
         * @param nij Count in cell (i,j)
         * @param rowTotal Total for row i
         * @param colTotal Total for column j
         * @param total Overall total
         * @return Contribution of this cell to MI
         */
        private double addMITerm(int nij, int rowTotal, int colTotal, int total) {
            if (nij == 0 || rowTotal == 0 || colTotal == 0) {
                return 0.0;  // 0 * log(0) = 0 by convention
            }

            double expected = (double) rowTotal * colTotal / total;

            // Add epsilon smoothing for numerical stability when expected is near-zero
            if (expected < 1e-10) {
                return 0.0;
            }

            double pij = (double) nij / total;

            // MI contribution: P(i,j) * log(P(i,j) / (P(i) * P(j)))
            return pij * Math.log(nij / expected);
        }

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

    public static class PreprocessingStats {
        public final int totalTexts;
        public final int originalWords;
        public final int cleanedWords;
        public final int filteredWords;
        public final CleaningMetrics cleaningMetrics;
        public final AdvancedTokenizer.TokenizationAnalysis tokenizationAnalysis;
        public final IntelligentStopwordRemover.StopwordAnalysis stopwordAnalysis;

        public PreprocessingStats(int totalTexts, int originalWords, int cleanedWords,
                                  int filteredWords, CleaningMetrics cleaningMetrics,
                                  AdvancedTokenizer.TokenizationAnalysis tokenizationAnalysis,
                                  IntelligentStopwordRemover.StopwordAnalysis stopwordAnalysis) {
            this.totalTexts = totalTexts;
            this.originalWords = originalWords;
            this.cleanedWords = cleanedWords;
            this.filteredWords = filteredWords;
            this.cleaningMetrics = cleaningMetrics;
            this.tokenizationAnalysis = tokenizationAnalysis;
            this.stopwordAnalysis = stopwordAnalysis;
        }

        @Override
        public String toString() {
            return String.format(
                    "PreprocessingStats{texts=%d, originalWords=%d, cleanedWords=%d, filteredWords=%d, " +
                            "cleaning=%s, tokenization=%s, stopwords=%s}",
                    totalTexts, originalWords, cleanedWords, filteredWords, cleaningMetrics,
                    tokenizationAnalysis, stopwordAnalysis
            );
        }

        public double getCompressionRatio() {
            return originalWords > 0 ? (double) filteredWords / originalWords : 0.0;
        }

        public double getTokenizationEfficiency() {
            return originalWords > 0 ? (double) cleanedWords / originalWords : 0.0;
        }

        public double getStopwordFilteringRatio() {
            return cleanedWords > 0 ? (double) filteredWords / cleanedWords : 0.0;
        }
    }

    public static class PipelineSummary {
        public final String preprocessorName;
        public final ContractionExpander.ContractionStats contractionStats;
        public final String tokenizerName;
        public final String stopwordConfig;

        public PipelineSummary(String preprocessorName,
                               ContractionExpander.ContractionStats contractionStats,
                               String tokenizerName, String stopwordConfig) {
            this.preprocessorName = preprocessorName;
            this.contractionStats = contractionStats;
            this.tokenizerName = tokenizerName;
            this.stopwordConfig = stopwordConfig;
        }

        @Override
        public String toString() {
            return String.format(
                    "PipelineSummary{preprocessor=%s, contractions=%s, tokenizer=%s, stopwords=%s}",
                    preprocessorName, contractionStats, tokenizerName, stopwordConfig
            );
        }
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

    public static class PipelineVersionInfo {
        public final String preprocessorVersion;
        public final String contractionExpanderVersion;
        public final String tokenizerVersion;
        public final String stopwordRemoverVersion;

        public PipelineVersionInfo(String preprocessorVersion,
                                   String contractionExpanderVersion,
                                   String tokenizerVersion,
                                   String stopwordRemoverVersion) {
            this.preprocessorVersion = preprocessorVersion;
            this.contractionExpanderVersion = contractionExpanderVersion;
            this.tokenizerVersion = tokenizerVersion;
            this.stopwordRemoverVersion = stopwordRemoverVersion;
        }

        @Override
        public String toString() {
            return String.format(
                    "PipelineVersionInfo{preprocessor=%s, contractionExpander=%s, " +
                            "tokenizer=%s, stopwordRemover=%s}",
                    preprocessorVersion, contractionExpanderVersion, tokenizerVersion,
                    stopwordRemoverVersion
            );
        }

        /**
         * Get a compact version string for the entire pipeline
         */
        public String getCompactVersion() {
            return String.format("Pipeline-v%s (CE:%s|TOK:%s|SW:%s)",
                    preprocessorVersion, contractionExpanderVersion, tokenizerVersion,
                    stopwordRemoverVersion);
        }
    }

    // ========== Demonstration and Testing Methods ==========

    public void demonstrateIntegratedPipeline(String sampleText) {
        logger.info("=== Integrated Text Preprocessing Pipeline Demonstration ===");
        logger.info("Original text: '{}'", sampleText);
        logger.info("Pipeline fitted: {}", isFitted);

        PipelineSummary summary = getPipelineSummary();
        logger.info("Pipeline summary: {}", summary);

        logger.info("\n--- Step 1: Text Cleaning ---");
        String cleaned = cleanText(sampleText);
        logger.info("Cleaned: '{}'", cleaned);

        logger.info("\n--- Step 2: Advanced Tokenization ---");
        advancedTokenizer.demonstrateTokenization(cleaned);

        logger.info("\n--- Step 3: Intelligent Stopword Removal ---");
        List<String> tokens = tokenize(cleaned);
        stopwordRemover.demonstrateStopwordRemoval(String.join(" ", tokens));

        logger.info("\n--- Step 4: Complete Preprocessing ---");
        String finalPreprocessed = preprocessText(sampleText);
        logger.info("Final preprocessed result: '{}'", finalPreprocessed);

        logger.info("=== End Integrated Pipeline Demonstration ===");
    }

    public void demonstrateCleaning(String sampleText) {
        logger.info("=== Text Cleaning Demonstration (Integrated Pipeline) ===");
        logger.info("Original: '{}'", sampleText);

        String cleaned = cleanText(sampleText);
        logger.info("Cleaned:  '{}'", cleaned);

        List<String> tokens = tokenize(cleaned);
        logger.info("Advanced Tokens: {}", tokens);

        List<String> filtered = removeStopwords(tokens);
        logger.info("Filtered: {}", filtered);

        logger.info("Final:    '{}'", String.join(" ", filtered));
        logger.info("=== End Demonstration ===");
    }

    /**
     * Demonstrate the fit/transform workflow
     */
    public void demonstrateFitTransform(List<Dataset> trainingData, String newText) {
        logger.info("=== Fit/Transform Workflow Demonstration ===");

        logger.info("Step 1: Initial state - Fitted: {}", isFitted);

        logger.info("Step 2: Fitting pipeline on {} training samples", trainingData.size());
        fit(trainingData);

        logger.info("Step 3: Pipeline state after fitting - Fitted: {}", isFitted);
        logger.info("  Vocabulary size: {}", pipelineState.vocabularySize);
        logger.info("  Training samples: {}", pipelineState.trainingSampleCount);

        logger.info("Step 4: Transforming new text: '{}'",
                newText.substring(0, Math.min(50, newText.length())));
        String transformed = transform(newText);
        logger.info("  Transformed result: '{}'", transformed);

        logger.info("=== End Fit/Transform Demonstration ===");
    }

    /**
     * Demonstrate state persistence
     */
    public void demonstrateStatePersistence(Path savePath, Path loadPath) throws IOException {
        logger.info("=== State Persistence Demonstration ===");

        logger.info("Step 1: Current state - Fitted: {}", isFitted);
        if (isFitted) {
            logger.info("  Saving state to: {}", savePath);
            saveState(savePath);
            logger.info("  State saved successfully");
        } else {
            logger.warn("  Cannot save - pipeline not fitted");
        }

        logger.info("Step 2: Resetting pipeline");
        reset();
        logger.info("  After reset - Fitted: {}", isFitted);

        if (Files.exists(loadPath)) {
            logger.info("Step 3: Loading state from: {}", loadPath);
            loadState(loadPath);
            logger.info("  State loaded - Fitted: {}", isFitted);
            logger.info("  Vocabulary size: {}", pipelineState.vocabularySize);
        } else {
            logger.warn("  Load path does not exist: {}", loadPath);
        }

        logger.info("=== End State Persistence Demonstration ===");
    }
}