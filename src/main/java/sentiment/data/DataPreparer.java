package sentiment.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sentiment.evaluation.StratifiedDataSplitter;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Prepares and manages immutable data splits for training.
 * Creates train/test splits with manifest files that act as "lock files" to prevent overwrites.
 *
 * Usage:
 *   DataPreparer preparer = new DataPreparer(datasetLoader);
 *   preparer.prepare("imdb_50k", rawPath, 50000, 42);  // Creates splits if not exist
 *   preparer.verify("imdb_50k");                        // Verifies checksums
 *   preparer.forceReset("imdb_50k", rawPath, 50000, 42); // Regenerates (explicit)
 */
public class DataPreparer {

    private static final Logger logger = LoggerFactory.getLogger(DataPreparer.class);
    private static final String PROCESSED_DIR = "data/processed";

    private final SimpleDatasetLoader datasetLoader;

    public DataPreparer(SimpleDatasetLoader datasetLoader) {
        this.datasetLoader = datasetLoader;
    }

    /**
     * Prepare data splits for a domain if they don't already exist.
     *
     * @param domain Dataset domain name (e.g., "imdb_50k", "amazon_polarity", "yelp")
     * @param rawDataPath Path to raw data file
     * @param maxSamples Maximum samples to load (0 = unlimited)
     * @param seed Random seed for reproducible splits
     * @return The manifest for the prepared splits
     * @throws IOException If file operations fail
     * @throws DataLoadingException If data loading fails
     */
    public SplitManifest prepare(String domain, String rawDataPath, int maxSamples, int seed)
            throws IOException, DataLoadingException {

        Path processedDir = getProcessedDir(domain);
        Path manifestPath = SplitManifest.getManifestPath(processedDir);

        // Check if splits already exist
        if (SplitManifest.exists(processedDir)) {
            logger.info("[SKIP] Splits already exist for domain '{}' at {}", domain, processedDir);
            logger.info("       Use forceReset() to regenerate.");
            return SplitManifest.load(manifestPath);
        }

        logger.info("=== Preparing Data Splits for '{}' ===", domain);
        logger.info("Raw source: {}", rawDataPath);
        logger.info("Max samples: {}", maxSamples > 0 ? maxSamples : "unlimited");
        logger.info("Random seed: {}", seed);

        return createSplits(domain, rawDataPath, maxSamples, seed, processedDir);
    }

    /**
     * Force regeneration of splits even if they exist.
     * Requires explicit call - prepare() won't overwrite.
     */
    public SplitManifest forceReset(String domain, String rawDataPath, int maxSamples, int seed)
            throws IOException, DataLoadingException {

        Path processedDir = getProcessedDir(domain);

        if (SplitManifest.exists(processedDir)) {
            logger.warn("[FORCE] Regenerating splits for domain '{}' (existing will be overwritten)", domain);
        }

        return createSplits(domain, rawDataPath, maxSamples, seed, processedDir);
    }

    /**
     * Verify that existing splits match their manifest checksums.
     */
    public boolean verify(String domain) throws IOException {
        Path processedDir = getProcessedDir(domain);
        Path manifestPath = SplitManifest.getManifestPath(processedDir);

        if (!SplitManifest.exists(processedDir)) {
            logger.error("[VERIFY FAIL] No manifest found for domain '{}'", domain);
            return false;
        }

        SplitManifest manifest = SplitManifest.load(manifestPath);
        boolean valid = manifest.verify(processedDir);

        if (valid) {
            logger.info("[VERIFY OK] Splits for '{}' match manifest checksums", domain);
        } else {
            logger.error("[VERIFY FAIL] Splits for '{}' do NOT match manifest checksums!", domain);
            logger.error("              Data may have been corrupted or modified.");
        }

        return valid;
    }

    /**
     * Load the manifest for a domain (if it exists).
     */
    public Optional<SplitManifest> loadManifest(String domain) {
        Path processedDir = getProcessedDir(domain);

        if (!SplitManifest.exists(processedDir)) {
            return Optional.empty();
        }

        try {
            return Optional.of(SplitManifest.load(SplitManifest.getManifestPath(processedDir)));
        } catch (IOException e) {
            logger.error("Failed to load manifest for '{}': {}", domain, e.getMessage());
            return Optional.empty();
        }
    }

    /**
     * Check if prepared splits exist for a domain.
     */
    public boolean splitsExist(String domain) {
        return SplitManifest.exists(getProcessedDir(domain));
    }

    // ===== Private Implementation =====

    /**
     * Get the processed data directory for a domain.
     * Centralizes path construction to ensure consistency across all methods.
     */
    private Path getProcessedDir(String domain) {
        return Paths.get(PROCESSED_DIR, domain).toAbsolutePath();
    }

    private SplitManifest createSplits(String domain, String rawDataPath, int maxSamples, int seed,
                                        Path processedDir) throws IOException, DataLoadingException {

        long startTime = System.currentTimeMillis();

        // Step 1: Load raw data with stratified sampling
        logger.info("Step 1/4: Loading raw data with stratified sampling...");
        List<Dataset> allData = loadAndBalance(rawDataPath, maxSamples, seed);
        logger.info("         Loaded {} balanced samples", allData.size());

        // Step 2: Perform 80/20 stratified split
        logger.info("Step 2/4: Performing 80/20 stratified split...");
        StratifiedDataSplitter.DataSplit split = StratifiedDataSplitter.stratifiedSplit(
                allData, 0.8, 0.0, 0.2, seed);
        logger.info("         Train: {}, Test: {}", split.train.size(), split.test.size());

        // Step 3: Save splits to CSV
        logger.info("Step 3/4: Saving splits to {}...", processedDir);
        Files.createDirectories(processedDir);

        Path trainPath = processedDir.resolve("train.csv");
        Path testPath = processedDir.resolve("test.csv");

        saveSplitToCsv(split.train, trainPath);
        saveSplitToCsv(split.test, testPath);

        // Step 4: Create and save manifest
        logger.info("Step 4/4: Creating manifest with checksums...");

        // Calculate class balance
        long trainPositive = split.train.stream()
                .filter(d -> d.getSentiment() == Dataset.SentimentLabel.POSITIVE).count();
        long trainNegative = split.train.size() - trainPositive;
        long testPositive = split.test.stream()
                .filter(d -> d.getSentiment() == Dataset.SentimentLabel.POSITIVE).count();
        long testNegative = split.test.size() - testPositive;

        // Calculate SHA-256 hashes
        String trainHash = SplitManifest.calculateSha256(trainPath);
        String testHash = SplitManifest.calculateSha256(testPath);

        SplitManifest manifest = SplitManifest.builder()
                .domain(domain)
                .seed(seed)
                .maxSamples(maxSamples)
                .rawSource(rawDataPath)
                .createdAt(Instant.now())
                .train(new SplitManifest.SplitInfo(
                        "train.csv", split.train.size(), trainHash, (int) trainPositive, (int) trainNegative))
                .test(new SplitManifest.SplitInfo(
                        "test.csv", split.test.size(), testHash, (int) testPositive, (int) testNegative))
                .build();

        manifest.save(SplitManifest.getManifestPath(processedDir));

        long elapsed = System.currentTimeMillis() - startTime;
        logger.info("=== Split Preparation Complete ({} ms) ===", elapsed);
        logger.info("Manifest: {}", SplitManifest.getManifestPath(processedDir));
        logger.info("Train: {} samples ({} pos, {} neg)", split.train.size(), trainPositive, trainNegative);
        logger.info("Test: {} samples ({} pos, {} neg)", split.test.size(), testPositive, testNegative);
        logger.info("Train hash: {}...", trainHash.substring(0, 16));
        logger.info("Test hash: {}...", testHash.substring(0, 16));

        return manifest;
    }

    /**
     * Load data with stratified sampling and balance classes.
     */
    private List<Dataset> loadAndBalance(String rawDataPath, int maxSamples, int seed)
            throws DataLoadingException {

        // Load with stratified sampling
        List<Dataset> allData = datasetLoader.load(rawDataPath, maxSamples);

        // Shuffle to mix classes
        List<Dataset> shuffled = new ArrayList<>(allData);
        Collections.shuffle(shuffled, new Random(seed));

        // Balance classes by undersampling majority
        return balanceClasses(shuffled, seed);
    }

    /**
     * Balance classes by undersampling the majority class.
     */
    private List<Dataset> balanceClasses(List<Dataset> data, int seed) {
        List<Dataset> positive = data.stream()
                .filter(d -> d.getSentiment() == Dataset.SentimentLabel.POSITIVE)
                .collect(Collectors.toList());
        List<Dataset> negative = data.stream()
                .filter(d -> d.getSentiment() == Dataset.SentimentLabel.NEGATIVE)
                .collect(Collectors.toList());

        int minoritySize = Math.min(positive.size(), negative.size());

        if (minoritySize == 0) {
            logger.warn("One class has zero samples - cannot balance");
            return data;
        }

        // Check if already balanced (within 5% tolerance)
        double ratio = (double) Math.min(positive.size(), negative.size()) /
                       Math.max(positive.size(), negative.size());
        if (ratio >= 0.95) {
            logger.info("Classes already balanced (ratio: {}) - no undersampling needed",
                    String.format("%.2f", ratio));
            return data;
        }

        // Undersample majority class
        List<Dataset> balancedPositive = positive.subList(0, minoritySize);
        List<Dataset> balancedNegative = negative.subList(0, minoritySize);

        List<Dataset> balanced = new ArrayList<>(minoritySize * 2);
        balanced.addAll(balancedPositive);
        balanced.addAll(balancedNegative);
        Collections.shuffle(balanced, new Random(seed));

        logger.info("Balanced classes: {} -> {} samples (undersampled {} from majority)",
                data.size(), balanced.size(), data.size() - balanced.size());

        return balanced;
    }

    /**
     * Save a data split to CSV file.
     */
    private void saveSplitToCsv(List<Dataset> data, Path filePath) throws IOException {
        try (FileWriter writer = new FileWriter(filePath.toFile())) {
            writer.write("review,sentiment\n");

            for (Dataset sample : data) {
                String sentiment = sample.getSentiment().name().toLowerCase();
                String text = escapeCsv(sample.getText());
                writer.write("\"" + text + "\"," + sentiment + "\n");
            }
        }
        logger.info("Saved {} samples to {}", data.size(), filePath);
    }

    /**
     * Escape CSV special characters in text.
     */
    private String escapeCsv(String text) {
        if (text == null) return "";
        return text.replace("\"", "\"\"");
    }

    // ===== CLI Entry Point =====

    /**
     * CLI for data preparation.
     * Usage: java DataPreparer <domain> <rawPath> [maxSamples] [seed] [--force|--verify]
     */
    public static void main(String[] args) {
        if (args.length < 1) {
            System.out.println("Usage: java DataPreparer <domain> [rawPath] [maxSamples] [seed] [--force|--verify]");
            System.out.println("  domain:     Dataset domain (e.g., imdb_50k, amazon_polarity, yelp)");
            System.out.println("  rawPath:    Path to raw data file (required for prepare, optional for verify)");
            System.out.println("  maxSamples: Max samples to load (default: 50000)");
            System.out.println("  seed:       Random seed (default: 42)");
            System.out.println("  --force:    Force regeneration even if splits exist");
            System.out.println("  --verify:   Verify existing splits match manifest checksums");
            System.exit(1);
        }

        String domain = args[0];
        boolean verify = Arrays.asList(args).contains("--verify");
        boolean force = Arrays.asList(args).contains("--force");

        try {
            SimpleDatasetLoader loader = new SimpleDatasetLoader();
            DataPreparer preparer = new DataPreparer(loader);

            if (verify) {
                // Verify mode: check checksums
                boolean valid = preparer.verify(domain);
                if (valid) {
                    System.out.println("[OK] Splits verified for " + domain);
                    System.exit(0);
                } else {
                    System.err.println("[FAIL] Verification failed for " + domain);
                    System.exit(1);
                }
            } else {
                // Prepare mode: create or update splits
                if (args.length < 2) {
                    System.err.println("Error: rawPath is required for prepare mode");
                    System.exit(1);
                }

                String rawPath = args[1];
                int maxSamples = 50000;
                int seed = 42;

                // Parse optional numeric arguments (skip flags)
                boolean maxSamplesSet = false;
                for (int i = 2; i < args.length; i++) {
                    if (!args[i].startsWith("--")) {
                        if (!maxSamplesSet) {
                            maxSamples = Integer.parseInt(args[i]);
                            maxSamplesSet = true;
                        } else {
                            seed = Integer.parseInt(args[i]);
                        }
                    }
                }

                SplitManifest manifest;
                if (force) {
                    manifest = preparer.forceReset(domain, rawPath, maxSamples, seed);
                } else {
                    manifest = preparer.prepare(domain, rawPath, maxSamples, seed);
                }

                System.out.println("\nManifest: " + manifest);
            }

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
