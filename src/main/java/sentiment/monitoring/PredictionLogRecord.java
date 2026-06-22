package sentiment.monitoring;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.HexFormat;

/**
 * Immutable record representing a single prediction for logging and drift analysis.
 *
 * <p>Uses text hash (not raw text) for privacy and storage efficiency.
 * The hash allows duplicate detection without storing sensitive content.
 *
 * <p><b>Performance optimization:</b> Hash computation is deferred to when the record
 * is serialized (in {@link #toJsonLine()}), which happens in the background logger
 * thread rather than on the inference hot path. This helps meet P99 &lt; 3 ms latency SLO.
 */
public final class PredictionLogRecord {
    private static final DateTimeFormatter ISO_FORMATTER = DateTimeFormatter.ISO_INSTANT;
    private static final HexFormat HEX_FORMAT = HexFormat.of();

    /**
     * Maximum text length to hash. Texts longer than this are truncated before hashing
     * to prevent DoS attacks via CPU exhaustion from hashing extremely large texts.
     * 10KB is sufficient for identifying duplicate predictions while limiting CPU impact.
     */
    private static final int MAX_HASH_TEXT_LENGTH = 10_000;

    private final Instant timestamp;
    private final String text;  // Kept for lazy hashing, not serialized
    private final int textLength;
    private final String predictedLabel;
    private final double confidence;
    private final long processingTimeMs;
    private final String modelVersion;

    // Lazily computed hash (computed on first access in background thread)
    private volatile String textHash;

    private PredictionLogRecord(
            Instant timestamp,
            String text,
            int textLength,
            String predictedLabel,
            double confidence,
            long processingTimeMs,
            String modelVersion) {
        this.timestamp = timestamp;
        this.text = text;
        this.textLength = textLength;
        this.predictedLabel = predictedLabel;
        this.confidence = confidence;
        this.processingTimeMs = processingTimeMs;
        this.modelVersion = modelVersion;
    }

    /**
     * Creates a PredictionLogRecord with deferred text hashing.
     *
     * <p>Hash computation is deferred to {@link #toJsonLine()} to keep the
     * inference hot path fast.
     *
     * @param text            the input text (hashed lazily when serialized)
     * @param predictedLabel  the predicted sentiment label
     * @param confidence      the confidence score [0, 1]
     * @param processingTimeMs inference latency in milliseconds
     * @param modelVersion    the model version or algorithm name
     * @return a new PredictionLogRecord
     */
    public static PredictionLogRecord create(
            String text,
            String predictedLabel,
            double confidence,
            long processingTimeMs,
            String modelVersion) {
        return new PredictionLogRecord(
                Instant.now(),
                text,
                text != null ? text.length() : 0,
                predictedLabel,
                confidence,
                processingTimeMs,
                modelVersion
        );
    }

    // Accessors for drift statistics (no hash needed)

    public Instant timestamp() {
        return timestamp;
    }

    public int textLength() {
        return textLength;
    }

    public String predictedLabel() {
        return predictedLabel;
    }

    public double confidence() {
        return confidence;
    }

    public long processingTimeMs() {
        return processingTimeMs;
    }

    public String modelVersion() {
        return modelVersion;
    }

    /**
     * Returns the text hash, computing it lazily if needed.
     * Thread-safe via volatile field.
     */
    public String textHash() {
        String hash = textHash;
        if (hash == null) {
            hash = computeTextHash(text);
            textHash = hash;
        }
        return hash;
    }

    /**
     * Computes SHA-256 hash of input text, truncated to 16 hex characters.
     * Returns "null" for null input.
     *
     * <p>Security: Texts longer than {@link #MAX_HASH_TEXT_LENGTH} are truncated
     * before hashing to prevent DoS attacks via CPU exhaustion.
     */
    public static String computeTextHash(String text) {
        if (text == null) {
            return "null";
        }
        try {
            // Limit text length to prevent DoS via large text hashing
            String textToHash = text.length() > MAX_HASH_TEXT_LENGTH
                    ? text.substring(0, MAX_HASH_TEXT_LENGTH)
                    : text;

            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(textToHash.getBytes(StandardCharsets.UTF_8));
            return HEX_FORMAT.formatHex(hash).substring(0, 16);
        } catch (NoSuchAlgorithmException e) {
            // SHA-256 is guaranteed to be available in Java
            throw new RuntimeException("SHA-256 not available", e);
        }
    }

    /**
     * Serializes this record to a JSON line (single line, no trailing newline).
     *
     * <p>Hash is computed here (lazily) so this should be called from background
     * thread, not from inference hot path.
     */
    public String toJsonLine() {
        return String.format(
                "{\"timestamp\":\"%s\",\"textHash\":\"%s\",\"textLength\":%d," +
                        "\"predictedLabel\":\"%s\",\"confidence\":%.6f," +
                        "\"processingTimeMs\":%d,\"modelVersion\":\"%s\"}",
                ISO_FORMATTER.format(timestamp),
                escapeJson(textHash()),  // Lazy hash computation happens here
                textLength,
                escapeJson(predictedLabel),
                confidence,
                processingTimeMs,
                escapeJson(modelVersion)
        );
    }

    /**
     * Parses a JSON line into a PredictionLogRecord.
     *
     * <p>Simple parser for the specific format produced by toJsonLine().
     * For production use, consider using Jackson or Gson.
     */
    public static PredictionLogRecord fromJsonLine(String json) {
        String timestampStr = extractJsonString(json, "timestamp");
        String textHash = extractJsonString(json, "textHash");
        int textLength = extractJsonInt(json, "textLength");
        String predictedLabel = extractJsonString(json, "predictedLabel");
        double confidence = extractJsonDouble(json, "confidence");
        long processingTimeMs = extractJsonLong(json, "processingTimeMs");
        String modelVersion = extractJsonString(json, "modelVersion");

        PredictionLogRecord record = new PredictionLogRecord(
                Instant.parse(timestampStr),
                null,  // Original text not available when parsing
                textLength,
                predictedLabel,
                confidence,
                processingTimeMs,
                modelVersion
        );
        record.textHash = textHash;  // Set pre-computed hash
        return record;
    }

    private static String escapeJson(String value) {
        if (value == null) {
            return "null";
        }
        return value
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    private static String extractJsonString(String json, String key) {
        String pattern = "\"" + key + "\":\"";
        int start = json.indexOf(pattern);
        if (start == -1) {
            return null;
        }
        start += pattern.length();
        int end = json.indexOf("\"", start);
        while (end > 0 && json.charAt(end - 1) == '\\') {
            end = json.indexOf("\"", end + 1);
        }
        return json.substring(start, end)
                .replace("\\\"", "\"")
                .replace("\\n", "\n")
                .replace("\\r", "\r")
                .replace("\\t", "\t")
                .replace("\\\\", "\\");
    }

    private static int extractJsonInt(String json, String key) {
        String value = extractJsonNumber(json, key);
        return Integer.parseInt(value);
    }

    private static long extractJsonLong(String json, String key) {
        String value = extractJsonNumber(json, key);
        return Long.parseLong(value);
    }

    private static double extractJsonDouble(String json, String key) {
        String value = extractJsonNumber(json, key);
        return Double.parseDouble(value);
    }

    private static String extractJsonNumber(String json, String key) {
        String pattern = "\"" + key + "\":";
        int start = json.indexOf(pattern);
        if (start == -1) {
            return "0";
        }
        start += pattern.length();
        int end = start;
        while (end < json.length()) {
            char c = json.charAt(end);
            if (c == ',' || c == '}') {
                break;
            }
            end++;
        }
        return json.substring(start, end).trim();
    }
}
