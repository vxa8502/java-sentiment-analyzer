package sentiment.data;

import sentiment.evaluation.StratifiedDataSplitter;

import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Utility to regenerate train/test splits from raw data.
 * Uses the same logic as ModelTrainer to ensure consistency.
 *
 * This allows populating data/processed/ with train.csv files
 * without retraining models.
 */
public class RegenerateSplits {

    private static final int MAX_SAMPLES = 50000;
    private static final long RANDOM_SEED = 42;
    private static final double TRAIN_RATIO = 0.8;
    private static final double TEST_RATIO = 0.2;

    private static final Map<String, String> DATASET_PATHS = Map.of(
        "imdb_50k", "data/raw/imdb_50k/IMDB Dataset.csv",
        "amazon_polarity", "data/raw/amazon_polarity/train.csv",
        "yelp", "data/raw/yelp/yelp_reviews.csv"
    );

    public static void main(String[] args) {
        System.out.println("=".repeat(60));
        System.out.println("REGENERATE TRAIN/TEST SPLITS");
        System.out.println("=".repeat(60));
        System.out.println("Using same logic as ModelTrainer:");
        System.out.println("  - Max samples: " + MAX_SAMPLES);
        System.out.println("  - Random seed: " + RANDOM_SEED);
        System.out.println("  - Split ratio: " + (int)(TRAIN_RATIO * 100) + "/" + (int)(TEST_RATIO * 100));
        System.out.println();

        SimpleDatasetLoader loader = new SimpleDatasetLoader();

        for (Map.Entry<String, String> entry : DATASET_PATHS.entrySet()) {
            String datasetName = entry.getKey();
            String rawPath = entry.getValue();

            System.out.println("Processing: " + datasetName);
            System.out.println("-".repeat(40));

            try {
                // Step 1: Load with stratified sampling (same as ModelTrainer.loadTrainingData)
                DatasetLoadResult loadResult = loader.loadWithMetadata(rawPath, MAX_SAMPLES);
                List<Dataset> allData = loadResult.datasets();
                System.out.println("  Loaded: " + allData.size() + " samples");

                // Step 2: Shuffle with fixed seed
                List<Dataset> shuffled = new ArrayList<>(allData);
                Collections.shuffle(shuffled, new Random(RANDOM_SEED));

                // Step 3: Balance classes (undersample majority to minority)
                List<Dataset> balanced = balanceClasses(shuffled);
                System.out.println("  After balancing: " + balanced.size() + " samples");

                // Step 4: Stratified 80/20 split
                StratifiedDataSplitter.DataSplit split = StratifiedDataSplitter.stratifiedSplit(
                    balanced, TRAIN_RATIO, 0.0, TEST_RATIO, RANDOM_SEED
                );
                System.out.println("  Train: " + split.train.size() + ", Test: " + split.test.size());

                // Step 5: Save to data/processed/
                Path processedDir = Paths.get("data/processed", datasetName);
                Files.createDirectories(processedDir);

                saveToCsv(split.train, processedDir.resolve("train.csv"));
                saveToCsv(split.test, processedDir.resolve("test.csv"));

                System.out.println("  Saved to: " + processedDir);
                System.out.println();

            } catch (Exception e) {
                System.err.println("  ERROR: " + e.getMessage());
                e.printStackTrace();
            }
        }

        System.out.println("=".repeat(60));
        System.out.println("COMPLETE");
        System.out.println("=".repeat(60));
    }

    /**
     * Balance classes by undersampling majority to match minority.
     * Exact same logic as ModelTrainer.balanceClasses().
     */
    private static List<Dataset> balanceClasses(List<Dataset> data) {
        Map<Dataset.SentimentLabel, List<Dataset>> byClass = data.stream()
            .collect(Collectors.groupingBy(Dataset::getSentiment));

        // Find minority class size
        int minSize = byClass.values().stream()
            .mapToInt(List::size)
            .min()
            .orElse(0);

        // Sample each class down to minority size
        List<Dataset> balanced = new ArrayList<>();
        Random random = new Random(RANDOM_SEED);

        for (List<Dataset> classData : byClass.values()) {
            List<Dataset> shuffledClass = new ArrayList<>(classData);
            Collections.shuffle(shuffledClass, random);
            balanced.addAll(shuffledClass.subList(0, Math.min(minSize, shuffledClass.size())));
        }

        // Final shuffle
        Collections.shuffle(balanced, new Random(RANDOM_SEED));
        return balanced;
    }

    /**
     * Save dataset to CSV file.
     */
    private static void saveToCsv(List<Dataset> data, Path filePath) throws IOException {
        try (FileWriter writer = new FileWriter(filePath.toFile())) {
            writer.write("review,sentiment\n");

            for (Dataset sample : data) {
                String sentiment = sample.getSentiment().name().toLowerCase();
                String text = escapeCsv(sample.getText());
                writer.write("\"" + text + "\"," + sentiment + "\n");
            }
        }
    }

    private static String escapeCsv(String text) {
        if (text == null) return "";
        return text.replace("\"", "\"\"");
    }
}
