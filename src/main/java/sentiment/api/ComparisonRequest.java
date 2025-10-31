package sentiment.api;

import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import sentiment.models.AlgorithmType;

import java.util.ArrayList;
import java.util.List;

/**
 * Request object for model comparison API endpoint.
 *
 * Allows clients to specify which models to compare and configuration options.
 */
public class ComparisonRequest {

    @NotNull(message = "Dataset name is required")
    private String datasetName;

    private List<String> models;  // Model names (e.g., "SVM", "Naive Bayes")

    private boolean compareAll = false;  // If true, compare all standard models

    private Integer maxFeatures = 5000;  // Max TF-IDF features

    private boolean parallelTraining = false;  // Enable parallel model training

    private Double trainTestSplit = 0.8;  // Train/test split ratio (0.0-1.0)

    // ==================== CONSTRUCTORS ====================

    public ComparisonRequest() {
        this.models = new ArrayList<>();
    }

    public ComparisonRequest(String datasetName) {
        this.datasetName = datasetName;
        this.models = new ArrayList<>();
    }

    // ==================== GETTERS AND SETTERS ====================

    public String getDatasetName() {
        return datasetName;
    }

    public void setDatasetName(String datasetName) {
        this.datasetName = datasetName;
    }

    public List<String> getModels() {
        return models;
    }

    public void setModels(List<String> models) {
        this.models = models != null ? models : new ArrayList<>();
    }

    public boolean isCompareAll() {
        return compareAll;
    }

    public void setCompareAll(boolean compareAll) {
        this.compareAll = compareAll;
    }

    public Integer getMaxFeatures() {
        return maxFeatures;
    }

    public void setMaxFeatures(Integer maxFeatures) {
        this.maxFeatures = maxFeatures;
    }

    public boolean isParallelTraining() {
        return parallelTraining;
    }

    public void setParallelTraining(boolean parallelTraining) {
        this.parallelTraining = parallelTraining;
    }

    public Double getTrainTestSplit() {
        return trainTestSplit;
    }

    public void setTrainTestSplit(Double trainTestSplit) {
        this.trainTestSplit = trainTestSplit;
    }

    // ==================== HELPER METHODS ====================

    /**
     * Parses model names to AlgorithmType list.
     */
    public List<AlgorithmType> getAlgorithmTypes() {
        List<AlgorithmType> types = new ArrayList<>();
        for (String modelName : models) {
            try {
                types.add(AlgorithmType.fromString(modelName));
            } catch (IllegalArgumentException e) {
                // Skip invalid model names
            }
        }
        return types;
    }

    @Override
    public String toString() {
        return String.format(
                "ComparisonRequest[dataset=%s, models=%s, compareAll=%s, maxFeatures=%d]",
                datasetName, models, compareAll, maxFeatures
        );
    }
}
