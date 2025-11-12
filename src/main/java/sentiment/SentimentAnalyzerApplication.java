package sentiment;

import org.springframework.boot.SpringApplication;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Spring Boot entry point for the Sentiment Analyzer REST API.
 *
 * This class bootstraps the Spring container and initializes all components
 * including the web server for the REST API.
 *
 * The application provides sentiment analysis via REST endpoints:
 * - POST /api/sentiment/analyze - Single text analysis
 * - POST /api/sentiment/batch - Batch text analysis
 * - GET /api/health - Health check
 *
 * Default port: 8080 (configurable in application.yml)
 *
 * To run the demo: Use SentimentAnalysisDemo.main() for command-line demos.
 * To run the API: Run this class to start the web server.
 */
@SpringBootApplication
public class SentimentAnalyzerApplication {
    private static final Logger logger = LoggerFactory.getLogger(SentimentAnalyzerApplication.class);

    public static void main(String[] args) {
        logger.info("Starting Sentiment Analyzer REST API...");
        SpringApplication.run(SentimentAnalyzerApplication.class, args);
        logger.info("REST API started successfully on port 8080");
        logger.info("Available endpoints:");
        logger.info("  POST http://localhost:8080/api/sentiment/analyze");
        logger.info("  POST http://localhost:8080/api/sentiment/batch");
        logger.info("  GET  http://localhost:8080/api/health");
    }
}
