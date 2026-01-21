package sentiment.data;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;

/**
 * Manifest for prepared data splits.
 * Tracks metadata and checksums to ensure data integrity and reproducibility.
 * Acts as a "lock file" - splits won't be overwritten if manifest exists.
 */
public class SplitManifest {

    public static final String MANIFEST_FILENAME = "splits.manifest.json";
    public static final String VERSION = "1.0";

    @JsonProperty("version")
    private String version = VERSION;

    @JsonProperty("domain")
    private String domain;

    @JsonProperty("created_at")
    private String createdAt;

    @JsonProperty("seed")
    private int seed;

    @JsonProperty("max_samples")
    private int maxSamples;

    @JsonProperty("raw_source")
    private String rawSource;

    @JsonProperty("train")
    private SplitInfo train;

    @JsonProperty("test")
    private SplitInfo test;

    /**
     * Information about a single split (train or test).
     */
    public static class SplitInfo {
        @JsonProperty("file")
        private String file;

        @JsonProperty("samples")
        private int samples;

        @JsonProperty("sha256")
        private String sha256;

        @JsonProperty("positive")
        private int positive;

        @JsonProperty("negative")
        private int negative;

        public SplitInfo() {}

        public SplitInfo(String file, int samples, String sha256, int positive, int negative) {
            this.file = file;
            this.samples = samples;
            this.sha256 = sha256;
            this.positive = positive;
            this.negative = negative;
        }

        // Getters
        public String getFile() { return file; }
        public int getSamples() { return samples; }
        public String getSha256() { return sha256; }
        public int getPositive() { return positive; }
        public int getNegative() { return negative; }

        // Setters for Jackson
        public void setFile(String file) { this.file = file; }
        public void setSamples(int samples) { this.samples = samples; }
        public void setSha256(String sha256) { this.sha256 = sha256; }
        public void setPositive(int positive) { this.positive = positive; }
        public void setNegative(int negative) { this.negative = negative; }
    }

    public SplitManifest() {}

    // Builder pattern for easy construction
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private final SplitManifest manifest = new SplitManifest();

        public Builder domain(String domain) {
            manifest.domain = domain;
            return this;
        }

        public Builder seed(int seed) {
            manifest.seed = seed;
            return this;
        }

        public Builder maxSamples(int maxSamples) {
            manifest.maxSamples = maxSamples;
            return this;
        }

        public Builder rawSource(String rawSource) {
            manifest.rawSource = rawSource;
            return this;
        }

        public Builder train(SplitInfo train) {
            manifest.train = train;
            return this;
        }

        public Builder test(SplitInfo test) {
            manifest.test = test;
            return this;
        }

        public Builder createdAt(Instant instant) {
            manifest.createdAt = instant.toString();
            return this;
        }

        public SplitManifest build() {
            if (manifest.createdAt == null) {
                manifest.createdAt = Instant.now().toString();
            }
            return manifest;
        }
    }

    // Persistence methods
    public void save(Path path) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
        mapper.writeValue(path.toFile(), this);
    }

    public static SplitManifest load(Path path) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        return mapper.readValue(path.toFile(), SplitManifest.class);
    }

    /**
     * Check if manifest exists for a domain.
     */
    public static boolean exists(Path processedDir) {
        return Files.exists(processedDir.resolve(MANIFEST_FILENAME));
    }

    /**
     * Get the manifest path for a processed directory.
     */
    public static Path getManifestPath(Path processedDir) {
        return processedDir.resolve(MANIFEST_FILENAME);
    }

    /**
     * Calculate SHA-256 hash of a file.
     */
    public static String calculateSha256(Path filePath) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] fileBytes = Files.readAllBytes(filePath);
            byte[] hashBytes = digest.digest(fileBytes);

            StringBuilder hexString = new StringBuilder();
            for (byte b : hashBytes) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) {
                    hexString.append('0');
                }
                hexString.append(hex);
            }
            return hexString.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IOException("SHA-256 algorithm not available", e);
        }
    }

    /**
     * Verify that the current files match the manifest checksums.
     */
    public boolean verify(Path processedDir) throws IOException {
        Path trainPath = processedDir.resolve(train.file);
        Path testPath = processedDir.resolve(test.file);

        if (!Files.exists(trainPath) || !Files.exists(testPath)) {
            return false;
        }

        String trainHash = calculateSha256(trainPath);
        String testHash = calculateSha256(testPath);

        return trainHash.equals(train.sha256) && testHash.equals(test.sha256);
    }

    // Getters
    public String getVersion() { return version; }
    public String getDomain() { return domain; }
    public String getCreatedAt() { return createdAt; }
    public int getSeed() { return seed; }
    public int getMaxSamples() { return maxSamples; }
    public String getRawSource() { return rawSource; }
    public SplitInfo getTrain() { return train; }
    public SplitInfo getTest() { return test; }

    // Setters for Jackson
    public void setVersion(String version) { this.version = version; }
    public void setDomain(String domain) { this.domain = domain; }
    public void setCreatedAt(String createdAt) { this.createdAt = createdAt; }
    public void setSeed(int seed) { this.seed = seed; }
    public void setMaxSamples(int maxSamples) { this.maxSamples = maxSamples; }
    public void setRawSource(String rawSource) { this.rawSource = rawSource; }
    public void setTrain(SplitInfo train) { this.train = train; }
    public void setTest(SplitInfo test) { this.test = test; }

    @Override
    public String toString() {
        return String.format("SplitManifest{domain='%s', created='%s', train=%d samples, test=%d samples}",
                domain, createdAt, train != null ? train.samples : 0, test != null ? test.samples : 0);
    }
}
