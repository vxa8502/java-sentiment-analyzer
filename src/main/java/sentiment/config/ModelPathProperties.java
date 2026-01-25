package sentiment.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

/**
 * Configuration properties for model file paths.
 *
 * <p>Centralizes all model path configuration under {@code sentiment.models.*}:
 * <pre>
 * sentiment:
 *   models:
 *     production-model-path: /app/models/production/sentiment_model.ser
 *     svm-model-path: /app/models/svm-model.ser
 *     naive-bayes-model-path: /app/models/naive_bayes-model.ser
 *     random-forest-model-path: /app/models/random_forest-model.ser
 *     logistic-model-path: /app/models/logistic_regression-model.ser
 * </pre>
 *
 * @see SentimentConfiguration
 */
@ConfigurationProperties(prefix = "sentiment.models")
@Validated
public class ModelPathProperties {

    /**
     * Path to the production model (best generalizing model).
     * This is the model promoted by promote_to_production.sh script.
     */
    private String productionModelPath = "./models/production/sentiment_model.ser";

    /**
     * Path to the pre-trained SVM model.
     */
    private String svmModelPath = "./models/svm-model.ser";

    /**
     * Path to the pre-trained Naive Bayes model.
     */
    private String naiveBayesModelPath = "./models/naive_bayes-model.ser";

    /**
     * Path to the pre-trained Random Forest model.
     */
    private String randomForestModelPath = "./models/random_forest-model.ser";

    /**
     * Path to the pre-trained Logistic Regression model.
     */
    private String logisticModelPath = "./models/logistic_regression-model.ser";

    // Getters and setters

    public String getProductionModelPath() {
        return productionModelPath;
    }

    public void setProductionModelPath(String productionModelPath) {
        this.productionModelPath = productionModelPath;
    }

    public String getSvmModelPath() {
        return svmModelPath;
    }

    public void setSvmModelPath(String svmModelPath) {
        this.svmModelPath = svmModelPath;
    }

    public String getNaiveBayesModelPath() {
        return naiveBayesModelPath;
    }

    public void setNaiveBayesModelPath(String naiveBayesModelPath) {
        this.naiveBayesModelPath = naiveBayesModelPath;
    }

    public String getRandomForestModelPath() {
        return randomForestModelPath;
    }

    public void setRandomForestModelPath(String randomForestModelPath) {
        this.randomForestModelPath = randomForestModelPath;
    }

    public String getLogisticModelPath() {
        return logisticModelPath;
    }

    public void setLogisticModelPath(String logisticModelPath) {
        this.logisticModelPath = logisticModelPath;
    }
}
