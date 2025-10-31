package sentiment.models;

import weka.core.Instances;
import sentiment.evaluation.ClassifierEvaluationResult;

/**
 * Interface for classifier evaluation operations.
 *
 * DESIGN PRINCIPLE: Interface Segregation
 * ========================================
 * Separated from core SentimentClassifier to follow ISP:
 * - Not all classifiers need evaluation capabilities
 * - Some use cases only need prediction, not model analysis
 * - Allows lightweight classifier implementations
 *
 * Implement this interface if your classifier supports:
 * - Performance evaluation on test sets
 * - Model introspection and summaries
 * - Quality metrics and diagnostics
 *
 * @author Sentiment Analysis Team
 * @since 2.0
 */
public interface ClassifierEvaluator {

    /**
     * Evaluates classifier performance on test data and returns comprehensive metrics.
     *
     * Performs full evaluation including accuracy, precision, recall, F1-score,
     * confusion matrix, and algorithm-specific metrics. Critical for model
     * comparison and performance analysis.
     *
     * Implementation Notes:
     * - Test data should use same preprocessing as training data
     * - Should not modify the trained model
     * - Consider stratified evaluation for imbalanced datasets
     * - Include timing metrics for performance analysis
     *
     * @param testData Weka Instances containing preprocessed test examples
     *                Must have same structure as training data
     * @return Comprehensive evaluation results including all key metrics
     * @throws Exception if evaluation fails or data format mismatches
     * @throws IllegalStateException if classifier hasn't been trained
     * @throws IllegalArgumentException if testData is null or incompatible
     */
    ClassifierEvaluationResult evaluate(Instances testData) throws Exception;

    /**
     * Returns human-readable summary of the trained model.
     *
     * Provides key information about the model architecture, parameters,
     * training statistics, and performance characteristics. Useful for
     * debugging, documentation, and model comparison.
     *
     * Example output:
     * "SVM Classifier (SMO)
     *  - Kernel: Polynomial (degree=2)
     *  - Training instances: 25,000
     *  - Features: 10,000 (TF-IDF)
     *  - Training time: 45.2s
     *  - Cross-validation accuracy: 84.7%"
     *
     * @return Multi-line string with model details and statistics
     * @throws IllegalStateException if classifier hasn't been trained
     */
    String getModelSummary();
}