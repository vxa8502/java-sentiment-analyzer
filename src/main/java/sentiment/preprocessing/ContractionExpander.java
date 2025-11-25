package sentiment.preprocessing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import sentiment.util.ValidationUtils;

import java.util.Map;
import java.util.regex.Pattern;

/**
 * Expands English contractions to normalize text for sentiment analysis.
 * <p>
 * Preserves capitalization for component reusability and debugging clarity,
 * though downstream processing typically lowercases all tokens for ML.
 * <p>
 * Uses word boundary matching to prevent false matches (e.g., "candy" won't match "can't").
 * Thread-safe with pre-compiled regex patterns for performance.
 */
@Component
public class ContractionExpander {

    private static final String VERSION = "1.0.0";
    private static final Logger logger = LoggerFactory.getLogger(ContractionExpander.class);

    // Comprehensive contractions map with case-insensitive matching
    private static final Map<String, String> CONTRACTIONS = Map.<String, String>ofEntries(
            // Common negative contractions
            Map.entry("don't", "do not"),
            Map.entry("doesn't", "does not"),
            Map.entry("didn't", "did not"),
            Map.entry("won't", "will not"),
            Map.entry("wouldn't", "would not"),
            Map.entry("can't", "cannot"),
            Map.entry("couldn't", "could not"),
            Map.entry("shouldn't", "should not"),
            Map.entry("mustn't", "must not"),
            Map.entry("needn't", "need not"),
            Map.entry("daren't", "dare not"),
            Map.entry("mayn't", "may not"),
            Map.entry("mightn't", "might not"),
            Map.entry("oughtn't", "ought not"),

            // "Be" verb contractions
            Map.entry("isn't", "is not"),
            Map.entry("aren't", "are not"),
            Map.entry("wasn't", "was not"),
            Map.entry("weren't", "were not"),

            // "Have" verb contractions
            Map.entry("haven't", "have not"),
            Map.entry("hasn't", "has not"),
            Map.entry("hadn't", "had not"),

            // Positive contractions with pronouns
            Map.entry("i'm", "I am"),
            Map.entry("you're", "you are"),
            Map.entry("he's", "he is"),
            Map.entry("she's", "she is"),
            Map.entry("it's", "it is"),
            Map.entry("we're", "we are"),
            Map.entry("they're", "they are"),
            Map.entry("that's", "that is"),
            Map.entry("there's", "there is"),
            Map.entry("here's", "here is"),
            Map.entry("what's", "what is"),
            Map.entry("where's", "where is"),
            Map.entry("when's", "when is"),
            Map.entry("why's", "why is"),
            Map.entry("how's", "how is"),
            Map.entry("who's", "who is"),

            // "Have" contractions
            Map.entry("i've", "I have"),
            Map.entry("you've", "you have"),
            Map.entry("we've", "we have"),
            Map.entry("they've", "they have"),

            // "Will" contractions
            Map.entry("i'll", "I will"),
            Map.entry("you'll", "you will"),
            Map.entry("he'll", "he will"),
            Map.entry("she'll", "she will"),
            Map.entry("it'll", "it will"),
            Map.entry("we'll", "we will"),
            Map.entry("they'll", "they will"),
            Map.entry("that'll", "that will"),

            // "Would" contractions
            Map.entry("i'd", "I would"),
            Map.entry("you'd", "you would"),
            Map.entry("he'd", "he would"),
            Map.entry("she'd", "she would"),
            Map.entry("we'd", "we would"),
            Map.entry("they'd", "they would"),
            Map.entry("that'd", "that would"),

            // Informal contractions
            Map.entry("ain't", "am not"),
            Map.entry("y'all", "you all"),
            Map.entry("'til", "until"),
            Map.entry("'cause", "because"),
            Map.entry("'bout", "about"),
            Map.entry("'round", "around"),
            Map.entry("o'clock", "of the clock"),

            // Let's handle "let us"
            Map.entry("let's", "let us")
    );

    // Pre-compiled patterns for efficiency
    private static final Map<Pattern, String> COMPILED_PATTERNS;

    static {
        COMPILED_PATTERNS = CONTRACTIONS.entrySet().stream()
                .collect(java.util.stream.Collectors.toMap(
                        entry -> Pattern.compile("\\b" + Pattern.quote(entry.getKey()) + "\\b",
                                Pattern.CASE_INSENSITIVE),
                        Map.Entry::getValue
                ));

        logger.debug("ContractionExpander initialized with {} contraction patterns",
                COMPILED_PATTERNS.size());
    }

    /**
     * Expands contractions in the given text using word boundary matching.
     * @param text the input text containing potential contractions
     * @return text with contractions expanded to full forms
     */
    public String expand(String text) {
        if (ValidationUtils.isNullOrEmpty(text)) {
            logger.debug("Received null or empty text for contraction expansion");
            return text != null ? text : "";
        }

        logger.debug("Expanding contractions in text of length: {}", text.length());

        String expanded = text;
        int expansionsCount = 0;

        // Apply each contraction pattern
        for (Map.Entry<Pattern, String> entry : COMPILED_PATTERNS.entrySet()) {
            Pattern pattern = entry.getKey();
            String replacement = entry.getValue();

            String beforeReplacement = expanded;
            expanded = pattern.matcher(expanded).replaceAll(matchResult -> {
                String matched = matchResult.group();
                return preserveCapitalization(matched, replacement);
            });

            // Count actual replacements made
            if (!beforeReplacement.equals(expanded)) {
                expansionsCount++;
            }
        }

        logger.debug("Contraction expansion complete: {} patterns applied", expansionsCount);

        if (expansionsCount > 0) {
            logger.trace("Text before expansion: '{}'", text);
            logger.trace("Text after expansion:  '{}'", expanded);
        }

        return expanded;
    }

    /**
     * Preserves capitalization patterns when expanding contractions.
     * <p>
     * Maintains readable output for debugging and supports reuse in contexts
     * where case preservation matters (though the sentiment pipeline lowercases later).
     */
    private String preserveCapitalization(String original, String replacement) {
        if (original.equals(original.toUpperCase())) {
            // All caps
            return replacement.toUpperCase();
        } else if (Character.isUpperCase(original.charAt(0))) {
            // First letter capitalized
            return capitalizeFirstLetter(replacement);
        } else {
            // Lowercase
            return replacement.toLowerCase();
        }
    }

    /**
     * Capitalizes the first letter of a string
     */
    private String capitalizeFirstLetter(String text) {
        if (text == null || text.isEmpty()) {
            return text;
        }
        return Character.toUpperCase(text.charAt(0)) + text.substring(1).toLowerCase();
    }

    /**
     * Get statistics about available contractions
     */
    public ContractionStats getStats() {
        int totalContractions = CONTRACTIONS.size();
        long negativeContractions = CONTRACTIONS.keySet().stream()
                .filter(key -> key.contains("n't"))
                .count();

        return new ContractionStats(totalContractions, (int) negativeContractions);
    }

    /**
     * Statistics about contraction expansion capabilities
     */
    public record ContractionStats(int totalContractions, int negativeContractions) {

        public String toString() {
            return String.format("ContractionStats{total=%d, negative=%d}",
                    totalContractions, negativeContractions);
        }
    }


    /**
     * Check if text contains any known contractions
     */
    public boolean hasContractions(String text) {
        if (ValidationUtils.isNullOrEmpty(text)) {
            return false;
        }

        return COMPILED_PATTERNS.keySet().stream()
                .anyMatch(pattern -> pattern.matcher(text).find());
    }

    /**
     * Count total contractions in text
     */
    public int countContractions(String text) {
        if (ValidationUtils.isNullOrEmpty(text)) {
            return 0;
        }

        return (int) COMPILED_PATTERNS.keySet().stream()
                .mapToLong(pattern -> pattern.matcher(text).results().count())
                .sum();
    }

    public String getVersion() {
        return VERSION;
    }
}