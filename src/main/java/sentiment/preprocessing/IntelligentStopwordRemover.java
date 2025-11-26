package sentiment.preprocessing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.beans.factory.annotation.Value;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Intelligent stopword removal that adapts to different text domains.
 * Provides multiple stopword removal strategies optimized for sentiment analysis.
 */
@Component
public class IntelligentStopwordRemover {

    private static final String VERSION = "1.0.0";
    private static final Logger logger = LoggerFactory.getLogger(IntelligentStopwordRemover.class);

    @Value("${sentiment.stopwords.strategy:SENTIMENT_AWARE}")
    private String stopwordStrategy;

    @Value("${sentiment.stopwords.preserve-negations:true}")
    private boolean preserveNegations;

    @Value("${sentiment.stopwords.preserve-intensifiers:true}")
    private boolean preserveIntensifiers;

    /**
     * Default constructor for Spring injection
     */
    public IntelligentStopwordRemover() {
        // Fields will be injected by Spring
    }

    /**
     * Constructor for manual initialization (non-Spring usage)
     */
    public IntelligentStopwordRemover(String stopwordStrategy, boolean preserveNegations, boolean preserveIntensifiers) {
        this.stopwordStrategy = stopwordStrategy;
        this.preserveNegations = preserveNegations;
        this.preserveIntensifiers = preserveIntensifiers;
    }

    /**
     * Convenience constructor with default settings
     */
    public IntelligentStopwordRemover(String stopwordStrategy) {
        this(stopwordStrategy, true, true);
    }

    // Enhanced stopword sets for different strategies
    private static final Set<String> BASIC_STOPWORDS = Set.of(
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
            "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
            "were", "will", "with", "would", "could", "should", "this", "these",
            "they", "them", "there", "their", "than", "then", "been", "have", "had"
    );

    private static final Set<String> EXTENDED_STOPWORDS = new HashSet<>(BASIC_STOPWORDS);
    static {
        EXTENDED_STOPWORDS.addAll(Set.of(
                "do", "does", "did", "can", "cannot", "i", "me", "my", "myself", "we",
                "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                "yourselves", "him", "his", "himself", "she", "her", "hers", "herself",
                "us", "they", "them", "their", "theirs", "themselves", "what", "which",
                "who", "whom", "whose", "where", "when", "why", "how", "all", "any",
                "both", "each", "few", "more", "most", "other", "some", "such", "no",
                "nor", "only", "own", "same", "so", "up", "very", "just", "now"
        ));
    }

    // Words that carry sentiment meaning and should be preserved
    private static final Set<String> SENTIMENT_BEARING_WORDS = Set.of(
            "not", "no", "never", "none", "nothing", "nobody", "nowhere",
            "very", "really", "quite", "extremely", "incredibly", "absolutely",
            "totally", "completely", "utterly", "highly", "particularly",
            "rather", "somewhat", "fairly", "pretty", "much", "too", "so"
    );

    // Negation words that are critical for sentiment analysis
    private static final Set<String> NEGATION_WORDS = Set.of(
            "not", "no", "never", "none", "nothing", "nobody", "nowhere",
            "neither", "nor", "cannot"
    );

    // Intensifier words that amplify sentiment
    private static final Set<String> INTENSIFIER_WORDS = Set.of(
            "very", "really", "quite", "extremely", "incredibly", "absolutely",
            "totally", "completely", "utterly", "highly", "particularly",
            "rather", "somewhat", "fairly", "pretty", "much", "too", "so",
            "super", "ultra", "mega", "hyper", "exceptionally", "remarkably"
    );

    /**
     * Main stopword removal method with strategy selection
     */
    public List<String> removeStopwords(List<String> tokens) {
        if (tokens == null || tokens.isEmpty()) {
            return new ArrayList<>();
        }

        logger.debug("Removing stopwords using strategy: {} from {} tokens",
                stopwordStrategy, tokens.size());

        StopwordStrategy strategy = StopwordStrategy.valueOf(stopwordStrategy.toUpperCase());
        List<String> filtered = applyStopwordStrategy(tokens, strategy);

        logger.debug("Stopword removal complete: {} -> {} tokens using {} strategy",
                tokens.size(), filtered.size(), strategy);

        return filtered;
    }

    /**
     * Apply the selected stopword removal strategy
     */
    private List<String> applyStopwordStrategy(List<String> tokens, StopwordStrategy strategy) {
        return switch (strategy) {
            case BASIC -> removeBasicStopwords(tokens);
            case EXTENDED -> removeExtendedStopwords(tokens);
            case SENTIMENT_AWARE -> removeSentimentAwareStopwords(tokens);
            case MINIMAL -> removeMinimalStopwords(tokens);
        };
    }

    /**
     * Basic stopword removal - conservative approach
     */
    private List<String> removeBasicStopwords(List<String> tokens) {
        return tokens.stream()
                .filter(token -> !isBasicStopword(token))
                .collect(Collectors.toList());
    }

    /**
     * Extended stopword removal - more aggressive filtering
     */
    private List<String> removeExtendedStopwords(List<String> tokens) {
        return tokens.stream()
                .filter(token -> !isExtendedStopword(token))
                .collect(Collectors.toList());
    }

    /**
     * Sentiment-aware stopword removal - preserves sentiment-critical words
     */
    private List<String> removeSentimentAwareStopwords(List<String> tokens) {
        return tokens.stream()
                .filter(token -> !isSentimentAwareStopword(token))
                .collect(Collectors.toList());
    }

    /**
     * Minimal stopword removal - only removes the most common function words
     */
    private List<String> removeMinimalStopwords(List<String> tokens) {
        Set<String> minimalStops = Set.of("the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by");
        return tokens.stream()
                .filter(token -> !minimalStops.contains(token.toLowerCase()))
                .collect(Collectors.toList());
    }

    /**
     * Check if token is a basic stopword
     */
    private boolean isBasicStopword(String token) {
        return BASIC_STOPWORDS.contains(token.toLowerCase());
    }

    /**
     * Check if token is an extended stopword
     */
    private boolean isExtendedStopword(String token) {
        String normalized = token.toLowerCase();

        // Don't remove sentiment-bearing words even in extended mode
        if (SENTIMENT_BEARING_WORDS.contains(normalized)) {
            return false;
        }

        return EXTENDED_STOPWORDS.contains(normalized);
    }

    /**
     * Sentiment-aware stopword detection - the most sophisticated approach
     */
    private boolean isSentimentAwareStopword(String token) {
        // Always preserve negation words
        if (preserveNegations && NEGATION_WORDS.contains(token)) {
            return false;
        }

        // Always preserve intensifiers
        if (preserveIntensifiers && INTENSIFIER_WORDS.contains(token)) {
            return false;
        }

        // Don't remove preserved sentiment indicators (from tokenizer)
        if (isPreservedSentimentToken(token)) {
            return false;
        }

        // Don't remove hyphenated words (often meaningful compounds)
        if (token.contains("-")) {
            return false;
        }

        // Don't remove meaningful punctuation
        if (token.matches("[!?]+")) {
            return false;
        }

        // Standard stopword check
        return EXTENDED_STOPWORDS.contains(token);
    }

    /**
     * Check if token is a preserved sentiment indicator from preprocessing
     */
    private boolean isPreservedSentimentToken(String token) {
        return token.startsWith("positive_emoticon") ||
                token.startsWith("negative_emoticon") ||
                token.startsWith("url_token") ||
                token.startsWith("email_token") ||
                token.startsWith("mention_token") ||
                token.startsWith("hashtag_") ||
                token.equals("number_token");
    }

    /**
     * Analyze stopword removal impact
     */
    public StopwordAnalysis analyzeStopwordRemoval(List<String> tokens) {
        if (tokens == null || tokens.isEmpty()) {
            return new StopwordAnalysis(0, 0, 0, 0, 0, 0);
        }

        int total = tokens.size();
        int basicRemoved = (int) tokens.stream().filter(this::isBasicStopword).count();
        int extendedRemoved = (int) tokens.stream().filter(this::isExtendedStopword).count();
        int sentimentAwareRemoved = (int) tokens.stream().filter(this::isSentimentAwareStopword).count();
        int negationsFound = (int) tokens.stream().filter(token -> NEGATION_WORDS.contains(token.toLowerCase())).count();
        int intensifiersFound = (int) tokens.stream().filter(token -> INTENSIFIER_WORDS.contains(token.toLowerCase())).count();

        return new StopwordAnalysis(total, basicRemoved, extendedRemoved,
                sentimentAwareRemoved, negationsFound, intensifiersFound);
    }

    /**
     * Compare different stopword removal strategies on the same text
     */
    public StopwordComparisonResult compareStrategies(List<String> tokens) {
        if (tokens == null || tokens.isEmpty()) {
            return new StopwordComparisonResult(new HashMap<>(), tokens);
        }

        Map<StopwordStrategy, List<String>> results = new HashMap<>();

        for (StopwordStrategy strategy : StopwordStrategy.values()) {
            results.put(strategy, applyStopwordStrategy(tokens, strategy));
        }

        return new StopwordComparisonResult(results, tokens);
    }

    /**
     * Stopword removal strategies
     */
    public enum StopwordStrategy {
        BASIC("Basic - Conservative removal"),
        EXTENDED("Extended - Aggressive removal"),
        SENTIMENT_AWARE("Sentiment Aware - Preserves sentiment words"),
        MINIMAL("Minimal - Only most common function words");

        private final String description;

        StopwordStrategy(String description) {
            this.description = description;
        }

        @Override
        public String toString() {
            return description;
        }
    }

    /**
     * Analysis results for stopword removal impact
     */
    public static class StopwordAnalysis {
        public final int totalTokens;
        public final int basicRemoved;
        public final int extendedRemoved;
        public final int sentimentAwareRemoved;
        public final int negationsFound;
        public final int intensifiersFound;

        public StopwordAnalysis(int totalTokens, int basicRemoved, int extendedRemoved,
                                int sentimentAwareRemoved, int negationsFound, int intensifiersFound) {
            this.totalTokens = totalTokens;
            this.basicRemoved = basicRemoved;
            this.extendedRemoved = extendedRemoved;
            this.sentimentAwareRemoved = sentimentAwareRemoved;
            this.negationsFound = negationsFound;
            this.intensifiersFound = intensifiersFound;
        }

        @Override
        public String toString() {
            return String.format(
                    "StopwordAnalysis{total=%d, basic=%d(%.1f%%), extended=%d(%.1f%%), sentimentAware=%d(%.1f%%), negations=%d, intensifiers=%d}",
                    totalTokens, basicRemoved, getPercentage(basicRemoved),
                    extendedRemoved, getPercentage(extendedRemoved),
                    sentimentAwareRemoved, getPercentage(sentimentAwareRemoved),
                    negationsFound, intensifiersFound
            );
        }

        private double getPercentage(int removed) {
            return totalTokens > 0 ? (double) removed / totalTokens * 100 : 0;
        }
    }

    /**
     * Comparison results between different stopword strategies
     */
    public static class StopwordComparisonResult {
        public final Map<StopwordStrategy, List<String>> results;
        public final List<String> originalTokens;

        public StopwordComparisonResult(Map<StopwordStrategy, List<String>> results,
                                        List<String> originalTokens) {
            this.results = results;
            this.originalTokens = originalTokens;
        }

        public void printComparison() {
            logger.debug("\nStopword Strategy Comparison:");
            logger.debug("Original ({}): {}", originalTokens.size(), originalTokens);

            for (Map.Entry<StopwordStrategy, List<String>> entry : results.entrySet()) {
                List<String> tokens = entry.getValue();
                double retention = !originalTokens.isEmpty() ?
                        (double) tokens.size() / originalTokens.size() * 100 : 0;
                logger.debug("{} ({} tokens, {}% retained): {}",
                        entry.getKey(), tokens.size(), String.format("%.1f", retention), tokens);
            }
        }
    }

    /**
     * Get configuration summary
     */
    public String getConfigurationSummary() {
        return String.format(
                "IntelligentStopwordRemover{strategy=%s, preserveNegations=%s, preserveIntensifiers=%s}",
                stopwordStrategy, preserveNegations, preserveIntensifiers
        );
    }

    public String getVersion() {
        return VERSION;
    }
}