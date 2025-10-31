package sentiment.models;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Instance;
import sentiment.preprocessing.TextPreprocessor;
import sentiment.preprocessing.WekaInstancesConverter;
import sentiment.evaluation.ClassifierEvaluationResult;
import sentiment.data.Dataset;

import javax.annotation.PreDestroy;
import java.util.*;

/**
 * Naive Bayes sentiment classifier using Weka's NaiveBayes implementation.
 *
 * MODEL SELECTION RATIONALE:
 * ==========================
 * Naive Bayes is a fast, probabilistic classifier ideal for baseline comparisons:
 *
 * 1. Computational Efficiency: Extremely fast training and inference, making it excellent
 *    for real-time applications and large-scale text classification.
 *
 * 2. Probabilistic Interpretation: Provides well-calibrated probability estimates that
 *    are interpretable and useful for confidence thresholding.
 *
 * 3. Small Data Performance: Works well even with limited training data due to its
 *    strong independence assumptions, which reduce parameter space.
 *
 * 4. Baseline Benchmark: Industry-standard baseline for text classification, making it
 *    valuable for comparing against more complex models.
 *
 * 5. Low Memory Footprint: Requires minimal memory as it only stores feature probabilities,
 *    not entire training instances.
 *
 * Limitations:
 * - Feature Independence Assumption: Assumes features are conditionally independent given
 *   the class, which is often violated in text (e.g., "not good" vs "good").
 * - Less Accurate for Complex Patterns: May underperform on datasets where feature
 *   interactions are critical for correct classification.
 *
 * Trade-offs vs. SVM:
 * + Faster training (10-100x speedup on large datasets)
 * + Lower memory usage
 * + Better calibrated probabilities
 * - Lower accuracy on complex patterns (typically 3-5% lower than SVM)
 * - Sensitive to irrelevant features
 *
 * ARCHITECTURE:
 * =============
 * Follows the same pipeline as SVMClassifier:
 * 1. Accepts raw List<Dataset> in train()
 * 2. Fits preprocessor (text cleaning, tokenization)
 * 3. Fits feature extractor (TF-IDF vectorization)
 * 4. Trains Naive Bayes on transformed features
 *
 * THREAD SAFETY:
 * ==============
 * - Training: Exclusive write lock (modifies model state)
 * - Inference: Concurrent read lock (thread-safe predictions)
 */
public class NaiveBayesClassifier extends ClassifierTrainingTemplate<ClassifierEvaluationResult>
        implements ClassifierEvaluator {

    private static final Logger logger = LoggerFactory.getLogger(NaiveBayesClassifier.class);

    private volatile boolean instanceStructureValidated = false;

    private NaiveBayes naiveBayes;

    private final TextPreprocessor preprocessor;
    private final WekaInstancesConverter converter;

    private Instances trainingDataStructure;
    private String[] supportedClasses;

    /**
     * Creates a new thread-safe Naive Bayes classifier with default configuration.
     */
    public NaiveBayesClassifier(TextPreprocessor preprocessor, WekaInstancesConverter converter) {
        if (preprocessor == null || converter == null) {
            throw new IllegalArgumentException("Preprocessor and converter cannot be null");
        }

        this.preprocessor = preprocessor;
        this.converter = converter;
        this.naiveBayes = new NaiveBayes();

        logger.info("Created NaiveBayesClassifier - fast probabilistic baseline");
    }

    /**
     * Creates classifier with custom NaiveBayes configuration.
     */
    public NaiveBayesClassifier(TextPreprocessor preprocessor, WekaInstancesConverter converter,
                                 NaiveBayes customNaiveBayes) {
        if (preprocessor == null || converter == null || customNaiveBayes == null) {
            throw new IllegalArgumentException("All dependencies must be non-null");
        }

        this.preprocessor = preprocessor;
        this.converter = converter;
        this.naiveBayes = customNaiveBayes;

        logger.info("Created NaiveBayesClassifier with custom configuration");
    }

    @Override
    public AlgorithmType getAlgorithmType() {
        return AlgorithmType.NAIVE_BAYES;
    }

    @Override
    public String[] getSupportedClasses() {
        requireTrained();
        return supportedClasses != null ? supportedClasses.clone() : new String[0];
    }

    @Override
    protected Logger getLogger() {
        return logger;
    }

    private int getTrainingInstanceCount() {
        return trainingDataStructure != null ? trainingDataStructure.numInstances() : 0;
    }

    private int getFeatureCount() {
        return trainingDataStructure != null ? trainingDataStructure.numAttributes() - 1 : 0;
    }

    // ==================== TEMPLATE METHOD IMPLEMENTATIONS ====================

    @Override
    protected ClassifierEvaluationResult doTrain(List<Dataset> rawDatasets) throws Exception {
        if (rawDatasets == null || rawDatasets.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }

        logger.info("Training Naive Bayes on {} raw datasets with full pipeline", rawDatasets.size());

        // Step 1: Fit preprocessing pipeline
        logger.info("Step 1/3: Fitting preprocessing pipeline");
        preprocessor.fit(rawDatasets);
        logger.info("Preprocessor fitted. Vocabulary: {}",
                preprocessor.getPipelineState().vocabularySize);

        // Step 2: Fit feature extraction
        logger.info("Step 2/3: Fitting feature extraction");
        Instances trainingInstances = converter.fit(rawDatasets);
        logger.info("Converter fitted. Features: {}, Vocabulary: {}",
                trainingInstances.numAttributes() - 1,
                converter.getVocabulary().size());

        // Step 3: Train Naive Bayes
        logger.info("Step 3/3: Training Naive Bayes classifier");
        validateTrainingData(trainingInstances);
        performModelTraining(trainingInstances);
        finalizeTraining(trainingInstances);

        logger.info("Naive Bayes training complete. Pipeline ready for inference.");

        return null;
    }

    @Override
    protected void doClearResources() {
        trainingDataStructure = null;
        supportedClasses = null;
        instanceStructureValidated = false;
    }

    @PreDestroy
    public void cleanup() {
        logger.info("Cleaning up NaiveBayesClassifier resources");
        doClearResources();
    }

    // ==================== TRAINING HELPERS ====================

    private void validateTrainingData(Instances data) {
        if (data.numInstances() < 10) {
            logger.warn("Small training set ({} instances)", data.numInstances());
        }

        if (data.classIndex() == -1) {
            throw new IllegalArgumentException("Training data must have class attribute set");
        }

        if (data.classAttribute().numValues() < 2) {
            throw new IllegalArgumentException("Need at least 2 classes");
        }

        logDatasetStatistics(data);
    }

    private void logDatasetStatistics(Instances data) {
        logger.info("Dataset: {} instances, {} features, {} classes",
                data.numInstances(), data.numAttributes() - 1,
                data.classAttribute().numValues());

        int[] classCounts = new int[data.classAttribute().numValues()];
        for (int i = 0; i < data.numInstances(); i++) {
            classCounts[(int) data.instance(i).classValue()]++;
        }

        for (int i = 0; i < classCounts.length; i++) {
            String percentage = String.format("%.1f", (classCounts[i] * 100.0) / data.numInstances());
            logger.info("  {}: {} ({}%)",
                    data.classAttribute().value(i),
                    classCounts[i],
                    percentage);
        }
    }

    private void performModelTraining(Instances trainingData) throws Exception {
        logger.info("Training Naive Bayes model on {} instances", trainingData.numInstances());
        naiveBayes.buildClassifier(trainingData);
        logger.info("Naive Bayes model training complete");
    }

    private void finalizeTraining(Instances trainingData) {
        this.trainingDataStructure = new Instances(trainingData, 0);

        this.supportedClasses = new String[trainingData.classAttribute().numValues()];
        for (int i = 0; i < supportedClasses.length; i++) {
            supportedClasses[i] = trainingData.classAttribute().value(i);
        }

        this.instanceStructureValidated = false;

        logger.info("Training finalized. Model ready. Classes: {}",
                String.join(", ", supportedClasses));
    }

    // ==================== CLASSIFICATION (THREAD-SAFE) ====================

    @Override
    public String classify(String text) throws Exception {
        requireTrained();

        if (text == null || text.trim().isEmpty()) {
            throw new IllegalArgumentException("Text cannot be null or empty");
        }

        return executeInference(() -> {
            logger.debug("INFERENCE: Classifying text: '{}'",
                    text.substring(0, Math.min(50, text.length())));

            Instance instance = converter.transform(text, "unknown");
            instance.setDataset(trainingDataStructure);

            double classIndex = naiveBayes.classifyInstance(instance);
            String predicted = supportedClasses[(int) classIndex];

            logger.debug("Classification result: {}", predicted);
            return predicted;
        });
    }

    @Override
    public double[] getClassificationProbabilities(String text) throws Exception {
        requireTrained();

        if (text == null || text.trim().isEmpty()) {
            throw new IllegalArgumentException("Text cannot be null or empty");
        }

        return executeInference(() -> {
            logger.debug("INFERENCE: Getting probabilities for: '{}'",
                    text.substring(0, Math.min(50, text.length())));

            Instance instance = converter.transform(text, "unknown");
            instance.setDataset(trainingDataStructure);

            double[] probs = naiveBayes.distributionForInstance(instance);
            logger.debug("Probability distribution: {}", formatProbabilities(probs));

            return probs;
        });
    }

    private String formatProbabilities(double[] probs) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < probs.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%s: %.3f", supportedClasses[i], probs[i]));
        }
        sb.append("]");
        return sb.toString();
    }

    // ==================== EVALUATION (THREAD-SAFE) ====================

    @Override
    public ClassifierEvaluationResult evaluate(Instances testData) throws Exception {
        requireTrained();

        if (testData == null || testData.numInstances() == 0) {
            throw new IllegalArgumentException("Test data cannot be null or empty");
        }

        logger.info("Evaluating on {} test instances", testData.numInstances());

        return executeInference(() -> {
            validateTestDataStructure(testData);
            return performEvaluation(testData);
        });
    }

    private void validateTestDataStructure(Instances testData) throws Exception {
        if (testData.numAttributes() != trainingDataStructure.numAttributes()) {
            throw new Exception(String.format(
                    "Attribute mismatch: training=%d, test=%d",
                    trainingDataStructure.numAttributes(), testData.numAttributes()));
        }

        logger.info("Test data validated: {} instances", testData.numInstances());
    }

    private ClassifierEvaluationResult performEvaluation(Instances testData) throws Exception {
        long startTime = System.currentTimeMillis();

        Evaluation evaluation = new Evaluation(trainingDataStructure);
        evaluation.evaluateModel(naiveBayes, testData);

        long evaluationTime = System.currentTimeMillis() - startTime;

        ClassifierEvaluationResult result = buildEvaluationResult(
                evaluation, testData, evaluationTime);

        String accuracy = String.format("%.3f", result.getAccuracy());
        logger.info("Evaluation complete in {}ms: accuracy={}",
                evaluationTime, accuracy);

        return result;
    }

    private ClassifierEvaluationResult buildEvaluationResult(
            Evaluation evaluation, Instances testData, long evaluationTimeMs) throws Exception {

        double accuracy = evaluation.pctCorrect() / 100.0;
        int numClasses = supportedClasses.length;

        double[] precision = new double[numClasses];
        double[] recall = new double[numClasses];
        double[] f1Score = new double[numClasses];

        for (int i = 0; i < numClasses; i++) {
            final int classIndex = i;
            precision[i] = safeMetric(() -> evaluation.precision(classIndex));
            recall[i] = safeMetric(() -> evaluation.recall(classIndex));
            f1Score[i] = safeMetric(() -> evaluation.fMeasure(classIndex));
        }

        double macroAvgPrecision = Arrays.stream(precision).average().orElse(0.0);
        double macroAvgRecall = Arrays.stream(recall).average().orElse(0.0);
        double macroAvgF1 = Arrays.stream(f1Score).average().orElse(0.0);

        double weightedPrecision = safeMetric(() -> evaluation.weightedPrecision());
        double weightedRecall = safeMetric(() -> evaluation.weightedRecall());
        double weightedF1 = safeMetric(() -> evaluation.weightedFMeasure());

        double[][] confusionMatrix = evaluation.confusionMatrix();

        Map<String, Object> stats = buildAdditionalStats(evaluation, testData, evaluationTimeMs);

        return new ClassifierEvaluationResult(
                getAlgorithmName(), accuracy,
                precision, recall, f1Score,
                macroAvgPrecision, macroAvgRecall, macroAvgF1,
                weightedPrecision, weightedRecall, weightedF1,
                confusionMatrix, supportedClasses, stats
        );
    }

    private double safeMetric(MetricSupplier supplier) {
        try {
            double value = supplier.get();
            return Double.isNaN(value) ? 0.0 : value;
        } catch (Exception e) {
            return 0.0;
        }
    }

    @FunctionalInterface
    private interface MetricSupplier {
        double get() throws Exception;
    }

    private Map<String, Object> buildAdditionalStats(
            Evaluation evaluation, Instances testData, long evaluationTimeMs) {

        Map<String, Object> stats = new HashMap<>();
        stats.put("totalInstances", testData.numInstances());
        stats.put("correctlyClassified", (int) evaluation.correct());
        stats.put("incorrectlyClassified", (int) evaluation.incorrect());
        stats.put("evaluationTimeMs", evaluationTimeMs);
        stats.put("kappa", safeMetric(() -> evaluation.kappa()));

        if (lastTrainingTimeMs > 0) {
            stats.put("trainingTimeMs", lastTrainingTimeMs);
        }

        if (converter.isReady()) {
            stats.put("vocabularySize", converter.getVocabulary().size());
        }

        return stats;
    }

    // ==================== MODEL SUMMARY ====================

    @Override
    public String getModelSummary() {
        requireTrained();

        return String.format(
                "=== Naive Bayes Classifier Summary ===\n\n" +
                        "Algorithm: %s\n" +
                        "State: %s\n\n" +
                        "Training: %d instances, %d features\n" +
                        "Classes: %d (%s)\n" +
                        "Training Time: %dms\n" +
                        "Vocabulary: %d terms\n\n" +
                        "Performance Characteristics:\n" +
                        "  - Fast training and inference\n" +
                        "  - Probabilistic predictions\n" +
                        "  - Low memory footprint",
                AlgorithmType.NAIVE_BAYES.getDisplayName(),
                classifierState,
                getTrainingInstanceCount(),
                getFeatureCount(),
                supportedClasses.length,
                String.join(", ", supportedClasses),
                lastTrainingTimeMs,
                converter.getVocabulary().size()
        );
    }

    @Override
    protected String getSubclassDiagnostics() {
        return String.format(
                "=== NaiveBayesClassifier Diagnostics ===\n" +
                        "NaiveBayes: %s\n" +
                        "Training: %d instances, %d features\n" +
                        "Supported classes: %s\n" +
                        "Instance validation cached: %s",
                naiveBayes != null ? "initialized" : "null",
                getTrainingInstanceCount(),
                getFeatureCount(),
                supportedClasses != null ? String.join(", ", supportedClasses) : "none",
                instanceStructureValidated
        );
    }

    // ==================== ACCESSORS ====================

    public NaiveBayes getNaiveBayes() {
        return naiveBayes;
    }

    void setNaiveBayes(NaiveBayes naiveBayes) {
        this.naiveBayes = naiveBayes;
    }

    Instances getTrainingStructure() {
        return trainingDataStructure;
    }

    void setTrainingMetadata(Instances structure, String[] classes) {
        this.trainingDataStructure = structure;
        this.supportedClasses = classes.clone();
        this.instanceStructureValidated = false;
    }

    public TextPreprocessor getPreprocessor() {
        return preprocessor;
    }

    public WekaInstancesConverter getConverter() {
        return converter;
    }
}
