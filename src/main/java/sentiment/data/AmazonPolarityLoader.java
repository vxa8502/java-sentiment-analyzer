package sentiment.data;

import org.apache.commons.csv.CSVRecord;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Loader for Amazon Review Polarity Dataset.
 *
 * Dataset format (CSV with 3 columns, no header):
 * - Column 0: polarity (1 = negative, 2 = positive)
 * - Column 1: title (review heading)
 * - Column 2: text (review body)
 *
 * Source: https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews
 * Citation: Xiang Zhang, Junbo Zhao, Yann LeCun. Character-level Convolutional Networks
 * for Text Classification. Advances in Neural Information Processing Systems 28 (NIPS 2015).
 */
class AmazonPolarityLoader extends CsvLoaderBase {

    private static final String DATASET_TYPE = "Amazon Review Polarity";
    private static final String[] SUPPORTED_EXTENSIONS = {".csv"};

    private static final int POLARITY_COLUMN = 0;
    private static final int TITLE_COLUMN = 1;
    private static final int TEXT_COLUMN = 2;

    @Override
    protected Dataset processRecord(CSVRecord record, DatasetLoadingStats stats, List<String> headers) {
        int recordNumber = (int) record.getRecordNumber();

        // Amazon Polarity format: polarity,title,text (no header row)
        if (record.size() < 3) {
            stats.incrementParseErrors();
            return null;
        }

        try {
            // Parse polarity (1 = negative, 2 = positive)
            String polarityStr = record.get(POLARITY_COLUMN).trim();
            int polarity = Integer.parseInt(polarityStr);

            Dataset.SentimentLabel sentiment;
            if (polarity == 1) {
                sentiment = Dataset.SentimentLabel.NEGATIVE;
            } else if (polarity == 2) {
                sentiment = Dataset.SentimentLabel.POSITIVE;
            } else {
                stats.incrementParseErrors();
                return null;
            }

            // Get title and text
            String title = record.get(TITLE_COLUMN);
            String text = record.get(TEXT_COLUMN);

            // Combine title and text for richer context
            String combinedText = title + ". " + text;

            // Validate text
            DatasetValidationUtils.ValidationResult textValidation =
                    DatasetValidationUtils.validateRawText(combinedText, DATASET_TYPE, recordNumber, stats);
            if (!textValidation.isValid()) {
                return null;
            }

            String rawText = textValidation.getText();

            return new Dataset.Builder(rawText, sentiment)
                    .source("amazon_polarity")
                    .originalLabel(polarityStr)
                    .timestamp(LocalDateTime.now())
                    .build();

        } catch (NumberFormatException | IndexOutOfBoundsException e) {
            stats.incrementParseErrors();
            return null;
        }
    }

    @Override
    protected double getMaxErrorRate() {
        return 0.05; // Amazon Polarity is clean, allow max 5% error rate
    }

    @Override
    public String[] getSupportedExtensions() {
        return SUPPORTED_EXTENSIONS.clone();
    }

    @Override
    public String getDatasetTypeName() {
        return DATASET_TYPE;
    }
}
