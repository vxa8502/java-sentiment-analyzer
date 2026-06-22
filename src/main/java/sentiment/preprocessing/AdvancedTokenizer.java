package sentiment.preprocessing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.beans.factory.annotation.Value;
import sentiment.util.ValidationUtils;

import java.util.*;
import java.util.regex.Pattern;
import java.util.regex.Matcher;
import java.util.stream.Collectors;

/**
 * Advanced text tokenizer with configurable handling of hyphenated words, numbers, and punctuation.
 * <p>
 * Provides sophisticated tokenization beyond simple whitespace splitting, including preservation
 * of hyphenated compound words (e.g., "well-known", "state-of-the-art"), intelligent filtering
 * of single characters and short tokens, and optional number handling.
 */
@Component
public class AdvancedTokenizer {

    private static final String VERSION = "1.0.0";
    private static final Logger logger = LoggerFactory.getLogger(AdvancedTokenizer.class);

    private final boolean preserveNumbers;
    private final int minTokenLength;
    private final boolean preserveHyphenated;

    // Pre-compiled regex patterns for efficiency
    private static final Pattern HYPHENATED_WORD_PATTERN = Pattern.compile(
            "\\b[a-zA-Z]+(?:-[a-zA-Z]+)+\\b"
    );



    private static final Pattern WORD_BOUNDARY_PATTERN = Pattern.compile(
            "\\b\\w+\\b|[.!?]+|[,;:]+"
    );

    private static final Pattern NUMBER_PATTERN = Pattern.compile(
            "\\b\\d+(?:\\.\\d+)?\\b|\\b\\d{1,3}(?:,\\d{3})*(?:\\.\\d+)?\\b"
    );

    // Pre-compiled patterns to avoid regex compilation on every call (performance fix)
    private static final Pattern WHITESPACE_PATTERN = Pattern.compile("\\s+");
    private static final Pattern CONTAINS_WORD_CHAR_PATTERN = Pattern.compile(".*\\w.*");

    // Pre-compiled patterns for token filtering (avoid String.matches() per token)
    private static final Pattern NON_ALPHANUMERIC_PATTERN = Pattern.compile("[^a-zA-Z0-9]+");
    private static final Pattern EXCLAMATION_QUESTION_PATTERN = Pattern.compile("[!?]+");
    private static final Pattern PUNCTUATION_SEQUENCE_PATTERN = Pattern.compile("[.!?]+|[,;:]+");

    private static final Set<String> MEANINGFUL_SINGLE_CHARS = Set.of(
            "i", "!", "?", ".", ":", ";"
    );

    /**
     * Constructs an AdvancedTokenizer with Spring-injected configuration.
     */
    public AdvancedTokenizer(
            @Value("${sentiment.tokenizer.preserve-numbers:false}") boolean preserveNumbers,
            @Value("${sentiment.tokenizer.min-token-length:1}") int minTokenLength,
            @Value("${sentiment.tokenizer.preserve-hyphenated:true}") boolean preserveHyphenated) {

        // Validate configuration at construction time
        if (minTokenLength < 1) {
            throw new IllegalArgumentException(
                    "Invalid minTokenLength: " + minTokenLength + ". Must be >= 1");
        }

        this.preserveNumbers = preserveNumbers;
        this.minTokenLength = minTokenLength;
        this.preserveHyphenated = preserveHyphenated;

        logger.info("AdvancedTokenizer initialized with validated configuration: " +
                        "preserveNumbers={}, minLength={}, preserveHyphenated={}",
                preserveNumbers, minTokenLength, preserveHyphenated);
    }

    /**
     * Tokenizes text into a list of tokens using advanced processing rules.
     * @param text the input text to tokenize (null or empty returns empty list)
     * @return list of tokens after applying all processing rules
     */
    public List<String> tokenize(String text) {
        if (ValidationUtils.isNullOrEmpty(text)) {
            logger.debug("Received null or empty text for tokenization");
            return new ArrayList<>();
        }

        logger.debug("Advanced tokenization starting for text length: {}", text.length());

        TokenizationMetrics metrics = new TokenizationMetrics();
        List<String> tokens;

        String textWithPlaceholders = text;

        Map<String, String> hyphenatedPlaceholders = new HashMap<>();
        if (preserveHyphenated) {
            textWithPlaceholders = extractAndPreserveHyphenated(textWithPlaceholders,
                    hyphenatedPlaceholders, metrics);
        }

        tokens = performMainTokenization(textWithPlaceholders, metrics);
        tokens = restoreHyphenatedWords(tokens, hyphenatedPlaceholders);
        tokens = applyFinalFiltering(tokens, metrics);

        logger.debug("Tokenization complete: {} input chars -> {} tokens, metrics: {}",
                text.length(), tokens.size(), metrics);

        return tokens;
    }

    private String extractAndPreserveHyphenated(String text,
                                                Map<String, String> placeholders,
                                                TokenizationMetrics metrics) {
        Matcher hyphenMatcher = HYPHENATED_WORD_PATTERN.matcher(text);
        StringBuilder sb = new StringBuilder();

        while (hyphenMatcher.find()) {
            String hyphenated = hyphenMatcher.group();

            if (isValidHyphenatedWord(hyphenated)) {
                String placeholder = "HYPHEN_PLACEHOLDER_" + placeholders.size();
                placeholders.put(placeholder, hyphenated);
                hyphenMatcher.appendReplacement(sb, " " + placeholder + " ");
                metrics.hyphenatedWordsFound++;
            } else {
                hyphenMatcher.appendReplacement(sb, " " + hyphenated.replace("-", " ") + " ");
            }
        }
        hyphenMatcher.appendTail(sb);

        String result = sb.toString();
        logger.debug("Hyphenated words processed: {} preserved", metrics.hyphenatedWordsFound);
        return result;
    }

    private boolean isValidHyphenatedWord(String word) {
        String[] parts = word.split("-");

        for (String part : parts) {
            if (part.length() < 2) {
                return false;
            }
        }

        Set<String> commonCompoundEndings = Set.of(
                "like", "based", "free", "proof", "ready", "aware", "wise"
        );

        Set<String> commonCompoundPrefixes = Set.of(
                "self", "well", "multi", "non", "pre", "post", "anti", "pro", "co"
        );

        String lastPart = parts[parts.length - 1].toLowerCase();
        String firstPart = parts[0].toLowerCase();

        return commonCompoundEndings.contains(lastPart) ||
                commonCompoundPrefixes.contains(firstPart) ||
                parts.length >= 3;
    }

    private List<String> performMainTokenization(String text, TokenizationMetrics metrics) {
        List<String> tokens = new ArrayList<>();

        Matcher wordMatcher = WORD_BOUNDARY_PATTERN.matcher(text);

        while (wordMatcher.find()) {
            String token = wordMatcher.group().trim();

            if (!token.isEmpty()) {
                tokens.add(token);
                metrics.rawTokensFound++;
            }
        }

        // Use pre-compiled pattern instead of split("\\s+") to avoid regex compilation per call
        String[] remainingTokens = WHITESPACE_PATTERN.split(text);
        for (String token : remainingTokens) {
            String trimmed = token.trim();
            if (!trimmed.isEmpty() && !tokens.contains(trimmed)) {
                // Use pre-compiled pattern instead of matches(".*\\w.*")
                if (CONTAINS_WORD_CHAR_PATTERN.matcher(token).matches()) {
                    tokens.add(trimmed);
                    metrics.additionalTokensFound++;
                }
            }
        }

        logger.debug("Main tokenization: {} raw tokens, {} additional tokens found",
                metrics.rawTokensFound, metrics.additionalTokensFound);

        return tokens;
    }

    private List<String> restoreHyphenatedWords(List<String> tokens,
                                                Map<String, String> hyphenatedPlaceholders) {
        return tokens.stream()
                .map(token -> hyphenatedPlaceholders.getOrDefault(token, token))
                .collect(Collectors.toList());
    }

    private List<String> applyFinalFiltering(List<String> tokens, TokenizationMetrics metrics) {
        List<String> filtered = new ArrayList<>();

        for (String token : tokens) {
            if (token.trim().isEmpty()) {
                metrics.emptyTokensFiltered++;
                continue;
            }

            if (NUMBER_PATTERN.matcher(token).matches()) {
                if (preserveNumbers) {
                    filtered.add("NUMBER_TOKEN");
                    metrics.numbersPreserved++;
                } else {
                    metrics.numbersFiltered++;
                }
                continue;
            }

            if (token.length() == 1) {
                if (MEANINGFUL_SINGLE_CHARS.contains(token)) {
                    filtered.add(token);
                    metrics.meaningfulSingleCharsPreserved++;
                } else {
                    metrics.singleCharsFiltered++;
                }
                continue;
            }

            if (token.length() < minTokenLength) {
                metrics.shortTokensFiltered++;
                continue;
            }

            // Use pre-compiled pattern instead of String.matches() (performance fix)
            if (NON_ALPHANUMERIC_PATTERN.matcher(token).matches()) {
                if (isMeaningfulPunctuation(token)) {
                    filtered.add(token);
                    metrics.punctuationPreserved++;
                } else {
                    metrics.punctuationFiltered++;
                }
                continue;
            }

            filtered.add(token);
            metrics.tokensKept++;
        }

        logger.debug("Final filtering complete: {} -> {} tokens, filtering metrics: {}",
                tokens.size(), filtered.size(), metrics.getFilteringStats());

        return filtered;
    }

    private boolean isMeaningfulPunctuation(String punctuation) {
        // Use pre-compiled pattern instead of String.matches() (performance fix)
        return EXCLAMATION_QUESTION_PATTERN.matcher(punctuation).matches() ||
                punctuation.equals("...") ||
                punctuation.equals("!!") ||
                punctuation.equals("???");
    }

    /**
     * Analyzes text to provide pre-tokenization statistics.
     *
     * @param text the text to analyze (null or empty returns zero-valued analysis)
     * @return analysis containing counts of hyphenated words, numbers, punctuation, and total words
     */
    public TokenizationAnalysis analyzeTokenization(String text) {
        if (ValidationUtils.isNullOrEmpty(text)) {
            return new TokenizationAnalysis(0, 0, 0, 0);
        }

        int hyphenatedWords = countMatches(HYPHENATED_WORD_PATTERN, text);
        int numbers = countMatches(NUMBER_PATTERN, text);
        // Use pre-compiled pattern instead of Pattern.compile() per call (performance fix)
        int punctuationSequences = countMatches(PUNCTUATION_SEQUENCE_PATTERN, text);
        // Use pre-compiled pattern instead of String.split() (performance fix)
        int totalWords = WHITESPACE_PATTERN.split(text).length;

        return new TokenizationAnalysis(hyphenatedWords,
                numbers, punctuationSequences, totalWords);
    }

    private int countMatches(Pattern pattern, String text) {
        return (int) pattern.matcher(text).results().count();
    }

    /**
     * Enumeration of token types for classification and analysis.
     */
    public enum TokenType {
        HYPHENATED("Hyphenated Words"),
        CONTRACTION("Contractions"),
        NUMBER("Numbers"),
        PUNCTUATION("Punctuation"),
        SINGLE_CHAR("Single Characters"),
        REGULAR_WORD("Regular Words"),
        OTHER("Other");

        private final String displayName;

        TokenType(String displayName) {
            this.displayName = displayName;
        }

        @Override
        public String toString() {
            return displayName;
        }
    }

    /**
     * Tracks detailed metrics during tokenization for debugging and analysis.
     * <p>
     * Records counts of tokens found, preserved, and filtered at each processing stage.
     */
    public static class TokenizationMetrics {
        public int hyphenatedWordsFound = 0;
        public int rawTokensFound = 0;
        public int additionalTokensFound = 0;

        public int emptyTokensFiltered = 0;
        public int numbersPreserved = 0;
        public int numbersFiltered = 0;
        public int meaningfulSingleCharsPreserved = 0;
        public int singleCharsFiltered = 0;
        public int shortTokensFiltered = 0;
        public int punctuationPreserved = 0;
        public int punctuationFiltered = 0;
        public int tokensKept = 0;

        public String getFilteringStats() {
            return String.format(
                    "kept=%d, filtered(empty=%d, short=%d, singleChar=%d, numbers=%d, punct=%d), preserved(numbers=%d, singleChar=%d, punct=%d)",
                    tokensKept, emptyTokensFiltered, shortTokensFiltered, singleCharsFiltered,
                    numbersFiltered, punctuationFiltered, numbersPreserved,
                    meaningfulSingleCharsPreserved, punctuationPreserved
            );
        }

        @Override
        public String toString() {
            return String.format(
                    "TokenMetrics{hyphenated=%d, rawTokens=%d, additional=%d, %s}", hyphenatedWordsFound, rawTokensFound,
                    additionalTokensFound, getFilteringStats()
            );
        }
    }

    /**
     * Pre-tokenization analysis statistics.
     *
     * @param hyphenatedWords count of hyphenated compound words found
     * @param numbers count of numeric sequences found
     * @param punctuationSequences count of punctuation sequences found
     * @param totalWords total word count (whitespace-split)
     */
    public record TokenizationAnalysis(int hyphenatedWords, int numbers,
                                       int punctuationSequences, int totalWords) {

        @Override
        public String toString() {
            return String.format(
                    "TokenAnalysis{words=%d, hyphenated=%d, numbers=%d, punctuation=%d}",
                    totalWords, hyphenatedWords, numbers, punctuationSequences
            );
        }
    }

    /**
     * Compares simple whitespace tokenization against advanced tokenization.
     *
     * @param text the text to tokenize and compare
     * @return comparison object containing both tokenization results and statistics
     */
    public TokenizationComparison compareTokenization(String text) {
        List<String> simpleTokens = Arrays.stream(text.split("\\s+"))
                .filter(token -> !token.trim().isEmpty())
                .filter(token -> token.length() >= minTokenLength)
                .collect(Collectors.toList());

        List<String> advancedTokens = tokenize(text);

        return new TokenizationComparison(simpleTokens, advancedTokens, text);
    }

    /**
     * Comparison of simple versus advanced tokenization results.
     * <p>
     * Provides side-by-side comparison to demonstrate the differences between
     * basic whitespace splitting and advanced tokenization logic.
     */
    public static class TokenizationComparison {
        public final List<String> simpleTokens;
        public final List<String> advancedTokens;
        public final String originalText;
        public final int improvementCount;

        public TokenizationComparison(List<String> simpleTokens, List<String> advancedTokens, String originalText) {
            this.simpleTokens = simpleTokens;
            this.advancedTokens = advancedTokens;
            this.originalText = originalText;
            this.improvementCount = Math.abs(advancedTokens.size() - simpleTokens.size());
        }
    }

    /**
     * Returns the version of this tokenizer implementation.
     *
     * @return version string
     */
    public String getVersion() {
        return VERSION;
    }
}