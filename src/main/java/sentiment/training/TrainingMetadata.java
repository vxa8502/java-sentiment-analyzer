package sentiment.training;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import sentiment.evaluation.ClassifierEvaluationResult;
import sentiment.models.AlgorithmType;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Human-readable training metadata for reproducibility.
 * Saved as JSON alongside model .ser files.
 */
public class TrainingMetadata {

    @JsonProperty("model_info")
    private ModelInfo modelInfo;

    @JsonProperty("training_data")
    private DatasetInfo dataset;

    @JsonProperty("preprocessing")
    private PreprocessingInfo preprocessing;

    @JsonProperty("hyperparameters")
    private Map<String, Object> hyperparameters;

    @JsonProperty("performance")
    private MetricsInfo metrics;

    @JsonProperty("cross_domain_performance")
    private Map<String, DomainPerformance> crossDomainPerformance;

    @JsonProperty("artifacts")
    private ArtifactsInfo artifacts;

    @JsonProperty("reproducibility")
    private ReproducibilityInfo reproducibility;

    @JsonProperty("trained_at")
    private String trainedAt;

    @JsonProperty("training_duration_seconds")
    private long trainingDurationSeconds;

    @JsonProperty("model_id")
    private String modelId;

    @JsonProperty("algorithm")
    private String algorithm;

    @JsonProperty("model_file")
    private String modelFile;

    @JsonProperty("model_size_bytes")
    private long modelSizeBytes;

    public static class ModelInfo {
        @JsonProperty("algorithm")
        public String algorithm;

        @JsonProperty("kernel")
        public String kernel;

        @JsonProperty("framework")
        public String framework; // e.g., "Weka 3.9.6"

        @JsonProperty("version")
        public String version = "1.0.0";

        @JsonProperty("notes")
        public String notes;
    }

    public static class DatasetInfo {
        @JsonProperty("dataset")
        public String datasetName; // e.g., "imdb_50k"

        @JsonProperty("dataset_version")
        public String datasetVersion; // e.g., "1.0"

        @JsonProperty("split")
        public String split = "train"; // train/val/test

        @JsonProperty("num_samples")
        public int numSamples;

        @JsonProperty("class_balance")
        public Map<String, Integer> classBalance; // e.g., {"positive": 12500, "negative": 12500}

        @JsonProperty("source")
        public String source;

        @JsonProperty("path")
        public String path;

        @JsonProperty("samples")
        public SampleCounts samples;

        @JsonProperty("hash")
        public String hash; // Optional: SHA-256 of dataset file
    }

    public static class SampleCounts {
        @JsonProperty("train")
        public int train;

        @JsonProperty("val")
        public int val;

        @JsonProperty("test")
        public int test;
    }

    public static class DomainPerformance {
        @JsonProperty("accuracy")
        public double accuracy;

        @JsonProperty("f1")
        public double f1;

        @JsonProperty("precision")
        public double precision;

        @JsonProperty("recall")
        public double recall;
    }

    public static class ArtifactsInfo {
        @JsonProperty("model_file")
        public String modelFile;

        @JsonProperty("preprocessor_state")
        public String preprocessorState;

        @JsonProperty("vocabulary_file")
        public String vocabularyFile;

        @JsonProperty("feature_importance")
        public String featureImportance;
    }

    public static class ReproducibilityInfo {
        @JsonProperty("random_seed")
        public long randomSeed = 42L;

        @JsonProperty("java_version")
        public String javaVersion;

        @JsonProperty("weka_version")
        public String wekaVersion;

        @JsonProperty("commit_hash")
        public String commitHash;
    }

    public static class PreprocessingInfo {
        @JsonProperty("lowercase")
        public boolean lowercase = true;

        @JsonProperty("remove_stopwords")
        public boolean removeStopwords = false;

        @JsonProperty("max_features")
        public int maxFeatures = 5000;

        @JsonProperty("min_word_frequency")
        public int minWordFrequency = 2;

        @JsonProperty("use_tfidf")
        public boolean useTfidf = true;

        @JsonProperty("use_bigrams")
        public boolean useBigrams = true;

        @JsonProperty("tokenizer")
        public String tokenizer = "AdvancedTokenizer";
    }

    public static class MetricsInfo {
        @JsonProperty("test_accuracy")
        public double testAccuracy;

        @JsonProperty("test_precision")
        public double testPrecision;

        @JsonProperty("test_recall")
        public double testRecall;

        @JsonProperty("test_f1")
        public double testF1;

        @JsonProperty("confusion_matrix")
        public int[][] confusionMatrix;

        @JsonProperty("roc_auc")
        public Double rocAuc;
    }

    // Builder for easy construction
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private final TrainingMetadata metadata = new TrainingMetadata();

        public Builder modelId(String modelId) {
            metadata.modelId = modelId;
            return this;
        }

        public Builder algorithm(AlgorithmType algorithm) {
            metadata.algorithm = algorithm.name();
            return this;
        }

        public Builder hyperparameters(Map<String, Object> params) {
            metadata.hyperparameters = new LinkedHashMap<>(params);
            return this;
        }

        public Builder datasetPath(String path) {
            if (metadata.dataset == null) metadata.dataset = new DatasetInfo();
            metadata.dataset.path = path;
            metadata.dataset.source = inferDatasetSource(path);
            return this;
        }

        public Builder sampleCounts(int train, int val, int test) {
            if (metadata.dataset == null) metadata.dataset = new DatasetInfo();
            metadata.dataset.samples = new SampleCounts();
            metadata.dataset.samples.train = train;
            metadata.dataset.samples.val = val;
            metadata.dataset.samples.test = test;
            return this;
        }

        public Builder preprocessing(int maxFeatures, boolean useTfidf, boolean useBigrams) {
            metadata.preprocessing = new PreprocessingInfo();
            metadata.preprocessing.maxFeatures = maxFeatures;
            metadata.preprocessing.useTfidf = useTfidf;
            metadata.preprocessing.useBigrams = useBigrams;
            return this;
        }

        public Builder evaluationResults(ClassifierEvaluationResult eval) {
            metadata.metrics = new MetricsInfo();
            metadata.metrics.testAccuracy = eval.getAccuracy();
            metadata.metrics.testPrecision = eval.getMacroAvgPrecision();
            metadata.metrics.testRecall = eval.getMacroAvgRecall();
            metadata.metrics.testF1 = eval.getMacroAvgF1();
            // Convert double[][] to int[][] for JSON serialization
            metadata.metrics.confusionMatrix = convertConfusionMatrix(eval.getConfusionMatrix());
            metadata.metrics.rocAuc = eval.getMacroAvgROCAUC();
            return this;
        }

        private int[][] convertConfusionMatrix(double[][] matrix) {
            if (matrix == null) return null;
            int[][] result = new int[matrix.length][];
            for (int i = 0; i < matrix.length; i++) {
                result[i] = new int[matrix[i].length];
                for (int j = 0; j < matrix[i].length; j++) {
                    result[i][j] = (int) Math.round(matrix[i][j]);
                }
            }
            return result;
        }

        public Builder trainedAt(Instant instant) {
            metadata.trainedAt = instant.toString();
            return this;
        }

        public Builder trainingDuration(long durationMs) {
            metadata.trainingDurationSeconds = durationMs / 1000;
            return this;
        }

        public Builder modelFile(String file, long sizeBytes) {
            metadata.modelFile = file;
            metadata.modelSizeBytes = sizeBytes;
            return this;
        }

        public Builder modelInfo(String algorithm, String kernel, String framework, String version, String notes) {
            metadata.modelInfo = new ModelInfo();
            metadata.modelInfo.algorithm = algorithm;
            metadata.modelInfo.kernel = kernel;
            metadata.modelInfo.framework = framework;
            metadata.modelInfo.version = version;
            metadata.modelInfo.notes = notes;
            return this;
        }

        public Builder artifacts(String modelFile, String featureImportance, String preprocessorState, String vocabularyFile) {
            metadata.artifacts = new ArtifactsInfo();
            metadata.artifacts.modelFile = modelFile;
            metadata.artifacts.featureImportance = featureImportance;
            metadata.artifacts.preprocessorState = preprocessorState;
            metadata.artifacts.vocabularyFile = vocabularyFile;
            return this;
        }

        public Builder reproducibility(long randomSeed, String javaVersion, String wekaVersion, String commitHash) {
            metadata.reproducibility = new ReproducibilityInfo();
            metadata.reproducibility.randomSeed = randomSeed;
            metadata.reproducibility.javaVersion = javaVersion;
            metadata.reproducibility.wekaVersion = wekaVersion;
            metadata.reproducibility.commitHash = commitHash;
            return this;
        }

        public Builder datasetInfo(String datasetName, String datasetVersion, String source, Map<String, Integer> classBalance) {
            if (metadata.dataset == null) metadata.dataset = new DatasetInfo();
            metadata.dataset.datasetName = datasetName;
            metadata.dataset.datasetVersion = datasetVersion;
            metadata.dataset.source = source;
            metadata.dataset.classBalance = classBalance;
            return this;
        }

        public Builder trainingSamples(int numSamples) {
            if (metadata.dataset == null) metadata.dataset = new DatasetInfo();
            metadata.dataset.numSamples = numSamples;
            return this;
        }

        public TrainingMetadata build() {
            return metadata;
        }

        private String inferDatasetSource(String path) {
            if (path.toLowerCase().contains("reviews")) {
                return "amazon_reviews_polarity";
            }
            return "custom_dataset";
        }
    }

    // Persistence
    public void save(Path path) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
        mapper.writeValue(path.toFile(), this);
    }

    public static TrainingMetadata load(Path path) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        return mapper.readValue(path.toFile(), TrainingMetadata.class);
    }

    /**
     * Get metadata file path for a model file.
     * Example: models/svm-model.ser -> models/svm-model.metadata.json
     */
    public static Path getMetadataPath(Path modelPath) {
        String fileName = modelPath.getFileName().toString();
        String baseName = fileName.replaceAll("\\.[^.]+$", "");
        return modelPath.getParent().resolve(baseName + ".metadata.json");
    }

    // Getters
    public String getModelId() { return modelId; }
    public String getAlgorithm() { return algorithm; }
    public Map<String, Object> getHyperparameters() { return hyperparameters; }
    public DatasetInfo getDataset() { return dataset; }
    public PreprocessingInfo getPreprocessing() { return preprocessing; }
    public MetricsInfo getMetrics() { return metrics; }
    public String getTrainedAt() { return trainedAt; }
    public long getTrainingDurationSeconds() { return trainingDurationSeconds; }
    public String getModelFile() { return modelFile; }
    public long getModelSizeBytes() { return modelSizeBytes; }
}
