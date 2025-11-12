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

    private static final Set<String> MEANINGFUL_SINGLE_CHARS = Set.of(
            "I", "a", "!", "?", ".", ":", ";"
    );

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

        String[] remainingTokens = text.split("\\s+");
        for (String token : remainingTokens) {
            if (!token.trim().isEmpty() && !tokens.contains(token.trim())) {
                if (token.matches(".*\\w.*")) {
                    tokens.add(token.trim());
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

            String normalizedToken = token.trim().toLowerCase();

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
                if (MEANINGFUL_SINGLE_CHARS.contains(token) ||
                        MEANINGFUL_SINGLE_CHARS.contains(token.toUpperCase())) {
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

            if (token.matches("[^a-zA-Z0-9]+")) {
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

    private boolean isMeaningfulPunctuation(String punct) {
        return punct.matches("[!?]+") ||
                punct.equals("...") ||
                punct.equals("!!") ||
                punct.equals("???");
    }

    public TokenizationAnalysis analyzeTokenization(String text) {
        if (ValidationUtils.isNullOrEmpty(text)) {
            return new TokenizationAnalysis(0, 0, 0, 0);
        }

        int hyphenatedWords = countMatches(HYPHENATED_WORD_PATTERN, text);
        int numbers = countMatches(NUMBER_PATTERN, text);
        int punctuationSequences = countMatches(Pattern.compile("[.!?]+|[,;:]+"), text);
        int totalWords = text.split("\\s+").length;

        return new TokenizationAnalysis(hyphenatedWords,
                numbers, punctuationSequences, totalWords);
    }

    private int countMatches(Pattern pattern, String text) {
        return (int) pattern.matcher(text).results().count();
    }

    public void demonstrateTokenization(String sampleText) {
        logger.info("=== Advanced Tokenization Demonstration ===");
        logger.info("Original text: '{}'", sampleText);

        TokenizationAnalysis analysis = analyzeTokenization(sampleText);
        logger.info("Pre-tokenization analysis: {}", analysis);

        List<String> tokens = tokenize(sampleText);
        logger.info("Tokenized ({} tokens): {}", tokens.size(), tokens);

        Map<TokenType, List<String>> categorizedTokens = categorizeTokens(tokens);
        for (Map.Entry<TokenType, List<String>> entry : categorizedTokens.entrySet()) {
            if (!entry.getValue().isEmpty()) {
                logger.info("{}: {}", entry.getKey(), entry.getValue());
            }
        }

        logger.info("=== End Tokenization Demonstration ===");
    }

    private Map<TokenType, List<String>> categorizeTokens(List<String> tokens) {
        Map<TokenType, List<String>> categorized = new EnumMap<>(TokenType.class);

        for (TokenType type : TokenType.values()) {
            categorized.put(type, new ArrayList<>());
        }

        for (String token : tokens) {
            TokenType type = classifyToken(token);
            categorized.get(type).add(token);
        }

        return categorized;
    }

    private TokenType classifyToken(String token) {
        if (token.contains("-") && HYPHENATED_WORD_PATTERN.matcher(token).matches()) {
            return TokenType.HYPHENATED;
        }
        if (NUMBER_PATTERN.matcher(token).matches()) {
            return TokenType.NUMBER;
        }
        if (token.matches("[^a-zA-Z0-9]+")) {
            return TokenType.PUNCTUATION;
        }
        if (token.length() == 1) {
            return TokenType.SINGLE_CHAR;
        }
        if (token.matches("\\w+")) {
            return TokenType.REGULAR_WORD;
        }
        return TokenType.OTHER;
    }

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

    public TokenizationComparison compareTokenization(String text) {
        List<String> simpleTokens = Arrays.stream(text.split("\\s+"))
                .filter(token -> !token.trim().isEmpty())
                .filter(token -> token.length() >= minTokenLength)
                .collect(Collectors.toList());

        List<String> advancedTokens = tokenize(text);

        return new TokenizationComparison(simpleTokens, advancedTokens, text);
    }

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

        public void printComparison() {
            logger.debug("Tokenization Comparison for: {}", originalText);
            logger.debug("Simple ({}): {}", simpleTokens.size(), simpleTokens);
            logger.debug("Advanced ({}): {}", advancedTokens.size(), advancedTokens);
            logger.debug("Difference: {} tokens", improvementCount);
        }
    }

    public String getVersion() {
        return VERSION;
    }
}