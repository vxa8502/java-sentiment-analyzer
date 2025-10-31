package sentiment.api;

import sentiment.models.ComparisonResult;
import sentiment.models.ClassificationMetrics;

import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Response object for model comparison API endpoint.
 *
 * Contains all comparison results in a format suitable for API consumption.
 */
public class ComparisonResponse {

    private String datasetName;
    private int testInstances;
    private int modelsCompared;
    private long timestamp;
    private long totalComparisonTimeMs;

    private List<ModelPerformance> models;
    private BestModels bestModels;
    private ComparisonInsights insights;

    // ==================== CONSTRUCTORS ====================

    public ComparisonResponse() {
        this.models = new ArrayList<>();
        this.timestamp = Instant.now().toEpochMilli();
    }

    /**
     * Creates a ComparisonResponse from a ComparisonResult.
     */
    public static ComparisonResponse fromComparisonResult(ComparisonResult result, long comparisonTimeMs) {
        ComparisonResponse response = new ComparisonResponse();

        response.datasetName = result.getDatasetName();
        response.testInstances = result.getTestInstances();
        response.modelsCompared = result.getClassificationMetrics().size();
        response.timestamp = result.getComparisonTimestamp();
        response.totalComparisonTimeMs = comparisonTimeMs;

        // Convert model metrics to API-friendly format
        for (ClassificationMetrics metrics : result.getClassificationMetrics()) {
            response.models.add(new ModelPerformance(metrics));
        }

        // Extract best models
        response.bestModels = new BestModels(result);

        // Generate insights
        response.insights = new ComparisonInsights(result);

        return response;
    }

    // ==================== GETTERS AND SETTERS ====================

    public String getDatasetName() {
        return datasetName;
    }

    public void setDatasetName(String datasetName) {
        this.datasetName = datasetName;
    }

    public int getTestInstances() {
        return testInstances;
    }

    public void setTestInstances(int testInstances) {
        this.testInstances = testInstances;
    }

    public int getModelsCompared() {
        return modelsCompared;
    }

    public void setModelsCompared(int modelsCompared) {
        this.modelsCompared = modelsCompared;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public long getTotalComparisonTimeMs() {
        return totalComparisonTimeMs;
    }

    public void setTotalComparisonTimeMs(long totalComparisonTimeMs) {
        this.totalComparisonTimeMs = totalComparisonTimeMs;
    }

    public List<ModelPerformance> getModels() {
        return models;
    }

    public void setModels(List<ModelPerformance> models) {
        this.models = models;
    }

    public BestModels getBestModels() {
        return bestModels;
    }

    public void setBestModels(BestModels bestModels) {
        this.bestModels = bestModels;
    }

    public ComparisonInsights getInsights() {
        return insights;
    }

    public void setInsights(ComparisonInsights insights) {
        this.insights = insights;
    }

    // ==================== NESTED CLASSES ====================

    /**
     * Performance metrics for a single model.
     */
    public static class ModelPerformance {
        private String modelName;
        private String algorithmType;
        private double accuracy;
        private double precision;
        private double recall;
        private double f1Score;
        private long trainingTimeMs;
        private long inferenceTimeMs;
        private int correctlyClassified;
        private int incorrectlyClassified;

        public ModelPerformance() {
        }

        public ModelPerformance(ClassificationMetrics metrics) {
            this.modelName = metrics.getModelName();
            this.algorithmType = metrics.getAlgorithmType().name();
            this.accuracy = metrics.getAccuracy();
            this.precision = metrics.getPrecision();
            this.recall = metrics.getRecall();
            this.f1Score = metrics.getF1Score();
            this.trainingTimeMs = metrics.getTrainingTimeMs();
            this.inferenceTimeMs = metrics.getInferenceTimeMs();
            this.correctlyClassified = metrics.getCorrectlyClassified();
            this.incorrectlyClassified = metrics.getIncorrectlyClassified();
        }

        // Getters and setters
        public String getModelName() { return modelName; }
        public void setModelName(String modelName) { this.modelName = modelName; }

        public String getAlgorithmType() { return algorithmType; }
        public void setAlgorithmType(String algorithmType) { this.algorithmType = algorithmType; }

        public double getAccuracy() { return accuracy; }
        public void setAccuracy(double accuracy) { this.accuracy = accuracy; }

        public double getPrecision() { return precision; }
        public void setPrecision(double precision) { this.precision = precision; }

        public double getRecall() { return recall; }
        public void setRecall(double recall) { this.recall = recall; }

        public double getF1Score() { return f1Score; }
        public void setF1Score(double f1Score) { this.f1Score = f1Score; }

        public long getTrainingTimeMs() { return trainingTimeMs; }
        public void setTrainingTimeMs(long trainingTimeMs) { this.trainingTimeMs = trainingTimeMs; }

        public long getInferenceTimeMs() { return inferenceTimeMs; }
        public void setInferenceTimeMs(long inferenceTimeMs) { this.inferenceTimeMs = inferenceTimeMs; }

        public int getCorrectlyClassified() { return correctlyClassified; }
        public void setCorrectlyClassified(int correctlyClassified) { this.correctlyClassified = correctlyClassified; }

        public int getIncorrectlyClassified() { return incorrectlyClassified; }
        public void setIncorrectlyClassified(int incorrectlyClassified) { this.incorrectlyClassified = incorrectlyClassified; }
    }

    /**
     * Best models across different criteria.
     */
    public static class BestModels {
        private String bestAccuracy;
        private String bestF1;
        private String fastestTraining;
        private String fastestInference;

        public BestModels() {
        }

        public BestModels(ComparisonResult result) {
            ClassificationMetrics best;

            best = result.getBestByAccuracy();
            this.bestAccuracy = best != null ? best.getModelName() : null;

            best = result.getBestByF1();
            this.bestF1 = best != null ? best.getModelName() : null;

            best = result.getFastestTraining();
            this.fastestTraining = best != null ? best.getModelName() : null;

            best = result.getFastestInference();
            this.fastestInference = best != null ? best.getModelName() : null;
        }

        // Getters and setters
        public String getBestAccuracy() { return bestAccuracy; }
        public void setBestAccuracy(String bestAccuracy) { this.bestAccuracy = bestAccuracy; }

        public String getBestF1() { return bestF1; }
        public void setBestF1(String bestF1) { this.bestF1 = bestF1; }

        public String getFastestTraining() { return fastestTraining; }
        public void setFastestTraining(String fastestTraining) { this.fastestTraining = fastestTraining; }

        public String getFastestInference() { return fastestInference; }
        public void setFastestInference(String fastestInference) { this.fastestInference = fastestInference; }
    }

    /**
     * Key insights from the comparison.
     */
    public static class ComparisonInsights {
        private double accuracySpread;
        private double trainingTimeRatio;
        private double inferenceTimeRatio;
        private Map<String, String> recommendations;

        public ComparisonInsights() {
            this.recommendations = new HashMap<>();
        }

        public ComparisonInsights(ComparisonResult result) {
            this.accuracySpread = result.getAccuracySpread();
            this.trainingTimeRatio = result.getTrainingTimeRatio();
            this.inferenceTimeRatio = result.getInferenceTimeRatio();

            this.recommendations = new HashMap<>();

            ComparisonResult.ModelRecommendation rec;

            rec = result.getRecommendation(true, false);
            recommendations.put("maxAccuracy", rec.getClassificationMetrics().getModelName());

            rec = result.getRecommendation(false, true);
            recommendations.put("realTime", rec.getClassificationMetrics().getModelName());

            rec = result.getRecommendation(false, false);
            recommendations.put("balanced", rec.getClassificationMetrics().getModelName());
        }

        // Getters and setters
        public double getAccuracySpread() { return accuracySpread; }
        public void setAccuracySpread(double accuracySpread) { this.accuracySpread = accuracySpread; }

        public double getTrainingTimeRatio() { return trainingTimeRatio; }
        public void setTrainingTimeRatio(double trainingTimeRatio) { this.trainingTimeRatio = trainingTimeRatio; }

        public double getInferenceTimeRatio() { return inferenceTimeRatio; }
        public void setInferenceTimeRatio(double inferenceTimeRatio) { this.inferenceTimeRatio = inferenceTimeRatio; }

        public Map<String, String> getRecommendations() { return recommendations; }
        public void setRecommendations(Map<String, String> recommendations) { this.recommendations = recommendations; }
    }
}
