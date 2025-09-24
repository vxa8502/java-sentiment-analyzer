package sentiment.data;

import java.util.regex.Pattern;

public class TextCleaningUtils {
    // Common patterns
    public static final Pattern HTML_TAGS = Pattern.compile("<[^>]+>");
    public static final Pattern EXCESSIVE_WHITESPACE = Pattern.compile("\\s+");
    public static final Pattern CONTROL_CHARS = Pattern.compile("""
            \\p{Cntrl}&&[^
            	]""");
    public static final Pattern NEWLINES_TO_SPACES = Pattern.compile("[\n\r\t]+");

    // Twitter-specific patterns
    private static final Pattern URL_PATTERN = Pattern.compile(
            "https?://[-\\w.]+(?:[:\\d]+)?(?:/[\\w/_.]*(?:\\?[\\w&=%.]*)?(?:#\\w*)?)?",
            Pattern.CASE_INSENSITIVE);
    private static final Pattern MENTION_PATTERN = Pattern.compile("@\\w+");
    private static final Pattern HASHTAG_PATTERN = Pattern.compile("#\\w+");
    private static final Pattern RT_PATTERN = Pattern.compile("^RT\\s+", Pattern.CASE_INSENSITIVE);

    // Product review patterns
    private static final Pattern REVIEW_ARTIFACTS = Pattern.compile(
            "(?i)(verified purchase|helpful\\?.*|\\d+ of \\d+ people found this helpful)");

    public static String basicTextCleaning(String text) {
        if (text == null || text.trim().isEmpty()) {
            return "";
        }

        String cleaned = text;
        cleaned = HTML_TAGS.matcher(cleaned).replaceAll(" ");
        cleaned = CONTROL_CHARS.matcher(cleaned).replaceAll("");
        cleaned = NEWLINES_TO_SPACES.matcher(cleaned).replaceAll(" ");
        cleaned = EXCESSIVE_WHITESPACE.matcher(cleaned).replaceAll(" ");

        return cleaned.trim();
    }

    public static String cleanMovieReviewText(String text) {
        String cleaned = basicTextCleaning(text);

        // Movie review specific cleaning
        cleaned = cleaned.replaceAll("\\*\\*\\*SPOILER ALERT\\*\\*\\*", "");
        cleaned = cleaned.replaceAll("\\[SPOILER]", "");

        return cleaned;
    }

    public static String cleanTweetText(String text) {
        String cleaned = basicTextCleaning(text);

        // Twitter specific cleaning
        cleaned = RT_PATTERN.matcher(cleaned).replaceAll("");
        cleaned = URL_PATTERN.matcher(cleaned).replaceAll(" URL ");
        cleaned = MENTION_PATTERN.matcher(cleaned).replaceAll(" USER ");
        cleaned = HASHTAG_PATTERN.matcher(cleaned).replaceAll(" ");

        return cleaned;
    }

    public static String cleanProductReviewText(String text) {
        String cleaned = basicTextCleaning(text);

        // Product review specific cleaning
        cleaned = REVIEW_ARTIFACTS.matcher(cleaned).replaceAll("");
        cleaned = cleaned.replaceAll("(?i)verified purchase", "");

        return cleaned;
    }
}