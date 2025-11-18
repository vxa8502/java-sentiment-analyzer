package sentiment.data;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * Specialized loader for Twitter Sentiment140 dataset.
 *
 * DATASET CHARACTERISTICS:
 * =======================
 * - Source: http://help.sentiment140.com/for-students
 * - Size: 1.6M tweets (we use 20K subset: 10K pos, 10K neg)
 * - Format: CSV with 6 fields: polarity,id,date,query,user,text
 * - Polarity encoding: 0 = negative, 4 = positive
 * - Text: Short (10-140 chars), informal, emoji, hashtags, @mentions
 *
 * DOMAIN DIFFERENCES VS. IMDB:
 * ============================
 * | Characteristic | IMDB Reviews | Twitter Sentiment140 |
 * |----------------|--------------|----------------------|
 * | Text Length    | ~1,300 chars | ~80 chars            |
 * | Language       | Formal       | Informal, slang      |
 * | Sentiment      | Nuanced      | Direct, emoji-heavy  |
 * | Noise          | Low          | High (typos)         |
 *
 * This contrast is intentional - tests model robustness across domains.
 *
 * @author Victoria Alabi
 */
@Component
public class TwitterDatasetLoader {

    private static final Logger logger = LoggerFactory.getLogger(TwitterDatasetLoader.class);

    private final SimpleDatasetLoader baseLoader;

    public TwitterDatasetLoader(SimpleDatasetLoader baseLoader) {
        this.baseLoader = baseLoader;
    }

    /**
     * Load Twitter Sentiment140 training set.
     *
     * CSV Format: polarity,id,date,query,user,text
     * - polarity: 0 = negative, 4 = positive
     * - We only care about 'polarity' and 'text' columns
     *
     * @param path Path to sentiment140 CSV file
     * @return DatasetLoadResult with Twitter data
     * @throws DataLoadingException if loading fails
     */
    public DatasetLoadResult loadSentiment140(String path) throws DataLoadingException {
        logger.info("Loading Twitter Sentiment140 dataset from {}", path);

        DatasetLoadResult result = baseLoader.loadWithMetadata(path);

        // Validate Twitter dataset characteristics
        validateTwitterDataset(result.datasets());

        return result;
    }

    /**
     * Load balanced subset for experiments.
     *
     * Marcus's insight: "20K tweets is enough for domain transfer experiments."
     *
     * @param path Path to sentiment140 CSV
     * @param size Total size (will be balanced 50/50)
     * @return Stratified subset
     * @throws DataLoadingException if loading fails
     */
    public DatasetLoadResult loadBalancedSubset(String path, int size) throws DataLoadingException {
        if (size <= 0 || size > 1600000) {
            throw new IllegalArgumentException("Subset size must be between 1 and 1,600,000, got: " + size);
        }

        logger.info("Loading Twitter balanced subset: {} examples", size);

        DatasetLoadResult fullResult = baseLoader.loadWithMetadata(path);
        List<Dataset> fullDataset = fullResult.datasets();

        // Create stratified subset (maintain 50/50 balance)
        int perClass = size / 2;

        List<Dataset> subset = fullDataset.stream()
            .filter(d -> d.getSentiment() == Dataset.SentimentLabel.POSITIVE)
            .limit(perClass)
            .collect(java.util.stream.Collectors.toList());

        subset.addAll(
            fullDataset.stream()
                .filter(d -> d.getSentiment() == Dataset.SentimentLabel.NEGATIVE)
                .limit(perClass)
                .toList()
        );

        // Shuffle to avoid ordering bias
        java.util.Collections.shuffle(subset);

        logger.info("Created balanced Twitter subset: {} examples ({} pos, {} neg)",
            subset.size(), perClass, perClass);

        return new DatasetLoadResult(
            subset,
            "Twitter Sentiment140 subset",
            path,
            fullResult.loadTimeMs()
        );
    }

    /**
     * Validate Twitter dataset properties.
     *
     * Expected characteristics:
     * - Short text (avg 50-100 chars)
     * - High noise (typos, slang, emoji)
     * - Binary sentiment only
     */
    private void validateTwitterDataset(List<Dataset> datasets) throws DataLoadingException {
        if (datasets.isEmpty()) {
            throw new DataLoadingException("Twitter dataset is empty", "twitter", "CSV");
        }

        // Check average text length (should be much shorter than IMDB)
        double avgLength = datasets.stream()
            .mapToInt(Dataset::getTextLength)
            .average()
            .orElse(0.0);

        logger.info("Twitter dataset avg text length: {:.0f} chars (expected: 50-150)", avgLength);

        if (avgLength > 500) {
            logger.warn("⚠ Twitter text unusually long (avg={:.0f}). Expected short tweets!", avgLength);
        }

        // Check label distribution
        long posCount = datasets.stream()
            .filter(d -> d.getSentiment() == Dataset.SentimentLabel.POSITIVE)
            .count();

        long negCount = datasets.stream()
            .filter(d -> d.getSentiment() == Dataset.SentimentLabel.NEGATIVE)
            .count();

        long neutralCount = datasets.stream()
            .filter(d -> d.getSentiment() == Dataset.SentimentLabel.NEUTRAL)
            .count();

        logger.info("Twitter dataset: {} total ({} pos, {} neg, {} neutral)",
            datasets.size(), posCount, negCount, neutralCount);

        if (neutralCount > 0) {
            logger.warn("⚠ Found {} neutral tweets - Sentiment140 should be binary only", neutralCount);
        }
    }

    /**
     * Get dataset statistics for reporting.
     */
    public DatasetStatistics computeStatistics(List<Dataset> datasets) {
        return DatasetStatistics.compute(datasets);
    }
}
