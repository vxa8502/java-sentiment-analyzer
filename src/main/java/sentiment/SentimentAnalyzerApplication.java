package sentiment;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;


/**
 * Spring Boot entry point for the Sentiment Analyzer REST API.
 * <br> For model training, use {@link sentiment.training.TrainModel}  instead.
 */
@SpringBootApplication
public class SentimentAnalyzerApplication {
    public static void main(String[] args) {
        SpringApplication.run(SentimentAnalyzerApplication.class, args);
    }
}
