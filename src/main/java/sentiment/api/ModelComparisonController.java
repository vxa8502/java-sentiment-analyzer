package sentiment.api;

import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import sentiment.data.DatasetLoaderRegistry;
import sentiment.data.Dataset;
import sentiment.models.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * REST API controller for model comparison operations.
 *
 * Endpoints:
 * - POST /api/models/compare - Compare multiple models on a dataset
 * - GET /api/models/available - List all available models
 *
 * USAGE EXAMPLES:
 * ===============
 *
 * Compare all models on a dataset:
 * POST /api/models/compare
 * {
 *   "datasetName": "movie_reviews",
 *   "compareAll": true,
 *   "maxFeatures": 5000,
 *   "parallelTraining": true
 * }
 *
 * Compare specific models:
 * POST /api/models/compare
 * {
 *   "datasetName": "movie_reviews",
 *   "models": ["SVM", "Naive Bayes", "Random Forest"],
 *   "maxFeatures": 3000
 * }
 */
@RestController
@RequestMapping("/api/models")
@org.springframework.context.annotation.Profile("!training")
public class ModelComparisonController {

    private static final Logger logger = LoggerFactory.getLogger(ModelComparisonController.class);

    private final DatasetLoaderRegistry dataLoaderManager;

    public ModelComparisonController(DatasetLoaderRegistry dataLoaderManager) {
        this.dataLoaderManager = dataLoaderManager;
        logger.info("ModelComparisonController initialized");
    }

    /**
     * Compares multiple models on a specified dataset.
     *
     * This endpoint:
     * 1. Loads the specified dataset
     * 2. Splits into train/test sets
     * 3. Trains all requested models
     * 4. Evaluates and compares performance
     * 5. Returns comprehensive comparison results
     *
     * @param request Comparison configuration
     * @return Comparison results with performance metrics
     */
    @PostMapping("/compare")
    public ResponseEntity<?> compareModels(@Valid @RequestBody ComparisonRequest request) {
        long startTime = System.currentTimeMillis();

        try {
            logger.info("Model comparison request: {}", request);

            // Step 1: Validate request
            validateRequest(request);

            // Step 2: Load dataset
            logger.info("Loading dataset: {}", request.getDatasetName());
            sentiment.data.DatasetLoadResult loadResult = dataLoaderManager.loadDataset(request.getDatasetName());
            List<Dataset> allData = loadResult.datasets();

            if (allData.isEmpty()) {
                return ResponseEntity.badRequest()
                        .body(new ErrorResponse("Dataset is empty or could not be loaded"));
            }

            logger.info("Loaded {} dataset records from {}", allData.size(), loadResult.filePath());

            // Step 3: Split into train/test
            DataSplit split = splitData(allData, request.getTrainTestSplit());
            logger.info("Data split: {} training, {} testing",
                    split.trainingData.size(), split.testData.size());

            // Step 4: Configure comparator
            AlgorithmBenchmark comparator = new AlgorithmBenchmark()
                    .setMaxFeatures(request.getMaxFeatures())
                    .setParallelTraining(request.isParallelTraining());

            // Step 5: Add models to compare
            if (request.isCompareAll()) {
                comparator.addAllStandardModels();
                logger.info("Comparing all standard models");
            } else {
                List<AlgorithmType> types = request.getAlgorithmTypes();
                if (types.isEmpty()) {
                    return ResponseEntity.badRequest()
                            .body(new ErrorResponse("No valid models specified. Use 'compareAll' or provide model names."));
                }
                for (AlgorithmType type : types) {
                    comparator.addModel(type);
                }
                logger.info("Comparing {} models: {}", types.size(), types);
            }

            // Step 6: Run comparison
            logger.info("Starting model comparison...");
            ComparisonResult result = comparator.compareModels(split.trainingData, split.testData);

            long totalTime = System.currentTimeMillis() - startTime;
            logger.info("Model comparison complete in {}ms", totalTime);

            // Step 7: Build response
            ComparisonResponse response = ComparisonResponse.fromComparisonResult(result, totalTime);

            return ResponseEntity.ok(response);

        } catch (IllegalArgumentException e) {
            logger.error("Invalid request: {}", e.getMessage());
            return ResponseEntity.badRequest()
                    .body(new ErrorResponse("Invalid request: " + e.getMessage()));

        } catch (Exception e) {
            logger.error("Model comparison failed", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(new ErrorResponse("Comparison failed: " + e.getMessage()));
        }
    }

    /**
     * Returns a list of all available model algorithms.
     *
     * GET /api/models/available
     *
     * Response example:
     * {
     *   "models": [
     *     {"name": "SVM", "displayName": "SVM (SMO)", "description": "Support Vector Machine..."},
     *     {"name": "NAIVE_BAYES", "displayName": "Naive Bayes", "description": "Probabilistic classifier..."},
     *     ...
     *   ]
     * }
     */
    @GetMapping("/available")
    public ResponseEntity<AvailableModelsResponse> getAvailableModels() {
        logger.info("Available models request");

        List<ModelInfo> models = new ArrayList<>();
        for (AlgorithmType type : AlgorithmType.values()) {
            if (type != AlgorithmType.UNKNOWN && type != AlgorithmType.NEURAL_NETWORK) {
                models.add(new ModelInfo(
                        type.name(),
                        type.getDisplayName(),
                        type.getDescription()
                ));
            }
        }

        AvailableModelsResponse response = new AvailableModelsResponse(models);
        return ResponseEntity.ok(response);
    }

    // ==================== HELPER METHODS ====================

    private void validateRequest(ComparisonRequest request) {
        if (request.getDatasetName() == null || request.getDatasetName().trim().isEmpty()) {
            throw new IllegalArgumentException("Dataset name is required");
        }

        if (!request.isCompareAll() && (request.getModels() == null || request.getModels().isEmpty())) {
            throw new IllegalArgumentException("Either set compareAll=true or provide a list of models");
        }

        if (request.getMaxFeatures() < 100 || request.getMaxFeatures() > 50000) {
            throw new IllegalArgumentException("maxFeatures must be between 100 and 50000");
        }

        if (request.getTrainTestSplit() < 0.1 || request.getTrainTestSplit() > 0.95) {
            throw new IllegalArgumentException("trainTestSplit must be between 0.1 and 0.95");
        }
    }

    /**
     * Splits datasets into training and test sets.
     */
    private DataSplit splitData(List<Dataset> allData, double trainRatio) {
        // Shuffle the dataset instances
        List<Dataset> shuffledData = new ArrayList<>(allData);
        Collections.shuffle(shuffledData);

        // Split based on ratio
        int trainSize = (int) (shuffledData.size() * trainRatio);

        List<Dataset> trainingData = shuffledData.subList(0, trainSize);
        List<Dataset> testData = shuffledData.subList(trainSize, shuffledData.size());

        return new DataSplit(
                new ArrayList<>(trainingData),  // Create new lists to avoid sublist issues
                new ArrayList<>(testData)
        );
    }

    // ==================== NESTED CLASSES ====================

    private static class DataSplit {
        final List<Dataset> trainingData;
        final List<Dataset> testData;

        DataSplit(List<Dataset> trainingData, List<Dataset> testData) {
            this.trainingData = trainingData;
            this.testData = testData;
        }
    }

    /**
     * Response for available models endpoint.
     */
    public static class AvailableModelsResponse {
        private List<ModelInfo> models;

        public AvailableModelsResponse() {
        }

        public AvailableModelsResponse(List<ModelInfo> models) {
            this.models = models;
        }

        public List<ModelInfo> getModels() {
            return models;
        }

        public void setModels(List<ModelInfo> models) {
            this.models = models;
        }
    }

    /**
     * Information about a single model.
     */
    public static class ModelInfo {
        private String name;
        private String displayName;
        private String description;

        public ModelInfo() {
        }

        public ModelInfo(String name, String displayName, String description) {
            this.name = name;
            this.displayName = displayName;
            this.description = description;
        }

        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public String getDisplayName() {
            return displayName;
        }

        public void setDisplayName(String displayName) {
            this.displayName = displayName;
        }

        public String getDescription() {
            return description;
        }

        public void setDescription(String description) {
            this.description = description;
        }
    }

    /**
     * Error response for failed requests.
     */
    public static class ErrorResponse {
        private String error;
        private long timestamp;

        public ErrorResponse() {
            this.timestamp = System.currentTimeMillis();
        }

        public ErrorResponse(String error) {
            this.error = error;
            this.timestamp = System.currentTimeMillis();
        }

        public String getError() {
            return error;
        }

        public void setError(String error) {
            this.error = error;
        }

        public long getTimestamp() {
            return timestamp;
        }

        public void setTimestamp(long timestamp) {
            this.timestamp = timestamp;
        }
    }
}
