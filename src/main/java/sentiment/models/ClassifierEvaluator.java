package sentiment.models;

import weka.core.Instances;
import sentiment.evaluation.ClassifierEvaluationResult;

/**
 * Interface for classifier evaluation operations.
 * <p>
 * <strong>DESIGN PRINCIPLE: Interface Segregation</strong>
 * <p>
 * Separated from core SentimentClassifier to follow ISP:
 * <ul>
 *   <li>Not all classifiers need evaluation capabilities</li>
 *   <li>Some use cases only need prediction, not model analysis</li>
 *   <li>Allows lightweight classifier implementations</li>
 * </ul>
 * <p>
 * Implement this interface if your classifier supports:
 * <ul>
 *   <li>Performance evaluation on test sets</li>
 *   <li>Model introspection and summaries</li>
 *   <li>Quality metrics and diagnostics</li>
 * </ul>
 *
 */
public interface ClassifierEvaluator {

    /**
     * Evaluates classifier performance on test data and returns comprehensive metrics.
     * <p>
     * Performs full evaluation including accuracy, precision, recall, F1-score,
     * confusion matrix, and algorithm-specific metrics. Critical for model
     * comparison and performance analysis.
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
     * <p>
     * Provides key information about the model architecture, parameters,
     * training statistics, and performance characteristics. Useful for
     * debugging, documentation, and model comparison.
     * <p>
     * Example output:
     * <pre>
     * SVM Classifier (SMO)
     *  - Kernel: Polynomial (degree=2)
     *  - Training instances: 25,000
     *  - Features: 10,000 (TF-IDF)
     *  - Training time: 45.2s
     *  - Cross-validation accuracy: 84.7%
     * </pre>
     *
     * @return Multi-line string with model details and statistics
     * @throws IllegalStateException if classifier hasn't been trained
     */
    String getModelSummary();
}