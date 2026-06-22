package sentiment;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * Spring Boot entry point for the Sentiment Analyzer REST API.
 *
 * <p>For model training, use {@link sentiment.training.TrainModel} instead.
 */
@SpringBootApplication
@EnableScheduling
public class SentimentAnalyzerApplication {
    public static void main(String[] args) {
        SpringApplication.run(SentimentAnalyzerApplication.class, args);
    }
}
