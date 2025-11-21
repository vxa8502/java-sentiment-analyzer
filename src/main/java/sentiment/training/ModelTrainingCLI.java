package sentiment.training;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Profile;
import sentiment.models.AlgorithmType;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

/**
 * Standalone application for pre-training sentiment analysis models.
 *
 * PURPOSE:
 * ========
 * This class provides a command-line interface for training models offline,
 * before application deployment. This approach dramatically improves production
 * startup time and enables better model lifecycle management.
 *
 * USAGE:
 * ======
 * Basic usage (trains all models with defaults):
 *   java -cp sentiment-analyzer.jar sentiment.training.ModelTrainingCLI
 *
 * Custom data path:
 *   java -cp sentiment-analyzer.jar sentiment.training.ModelTrainingCLI \
 *     --data ./custom/dataset.csv
 *
 * Specific algorithm:
 *   java -cp sentiment-analyzer.jar sentiment.training.ModelTrainingCLI \
 *     --algorithm svm --data ./data/Reviews.csv --output ./models/
 *
 * Limit training samples:
 *   java -cp sentiment-analyzer.jar sentiment.training.ModelTrainingCLI \
 *     --max-samples 50000
 *
 * Train multiple algorithms:
 *   java -cp sentiment-analyzer.jar sentiment.training.ModelTrainingCLI \
 *     --algorithms svm,naive_bayes,random_forest
 *
 * MAVEN EXECUTION:
 * ================
 * mvn exec:java -Dexec.mainClass="sentiment.training.ModelTrainingCLI" \
 *   -Dexec.args="--data ./data/datasets/Reviews.csv --output ./models/"
 *
 * INTEGRATION WITH BUILD:
 * =======================
 * Add to pom.xml for automatic model training during build:
 * <pre>
 * {@code
 * <plugin>
 *   <groupId>org.codehaus.mojo</groupId>
 *   <artifactId>exec-maven-plugin</artifactId>
 *   <version>3.1.0</version>
 *   <executions>
 *     <execution>
 *       <id>train-models</id>
 *       <phase>process-classes</phase>
 *       <goals>
 *         <goal>java</goal>
 *       </goals>
 *       <configuration>
 *         <mainClass>sentiment.training.ModelTrainingCLI</mainClass>
 *         <arguments>
 *           <argument>--data</argument>
 *           <argument>./data/datasets/Reviews.csv</argument>
 *           <argument>--output</argument>
 *           <argument>./models/</argument>
 *         </arguments>
 *       </configuration>
 *     </execution>
 *   </executions>
 * </plugin>
 * }
 * </pre>
 */
@SpringBootApplication
@ComponentScan(basePackages = {"sentiment"})
@Profile("!test")
public class ModelTrainingCLI implements CommandLineRunner {

    private static final Logger logger = LoggerFactory.getLogger(ModelTrainingCLI.class);

    // Default configuration
    private static final String DEFAULT_DATA_PATH = "./data/datasets/Reviews.csv";
    private static final String DEFAULT_OUTPUT_DIR = "./models";
    private static final int DEFAULT_MAX_SAMPLES = 10000;

    private final ModelTrainer trainer;

    public ModelTrainingCLI(ModelTrainer trainer) {
        this.trainer = trainer;
    }

    public static void main(String[] args) {
        System.setProperty("spring.main.banner-mode", "off");
        System.setProperty("logging.level.org.springframework", "ERROR");
        System.setProperty("spring.main.web-application-type", "none");
        System.setProperty("spring.profiles.active", "training");
        SpringApplication.run(ModelTrainingCLI.class, args);
    }

    @Override
    public void run(String... args) throws Exception {
        try {
            logger.info("========================================");
            logger.info("  Sentiment Model Training Tool");
            logger.info("========================================");

            // Parse command-line arguments
            Config config = parseArguments(args);

            // Validate inputs
            validateConfiguration(config);

            // Train models
            List<ModelTrainer.TrainingResult> results;

            if (config.singleAlgorithm != null) {
                // Train single model
                logger.info("Training single model: {}", config.singleAlgorithm.getDisplayName());
                String outputPath = Paths.get(config.outputDir,
                        config.singleAlgorithm.name().toLowerCase() + "-model.ser").toString();

                ModelTrainer.TrainingResult result = trainer.trainAndSave(
                        config.dataPath,
                        outputPath,
                        config.singleAlgorithm,
                        config.maxSamples,
                        config.showFeatureImportance,
                        config.topFeaturesCount
                );

                results = List.of(result);

            } else {
                // Train multiple models
                logger.info("Training {} models: {}",
                        config.algorithms.size(),
                        config.algorithms.stream()
                                .map(AlgorithmType::getDisplayName)
                                .toList());

                results = trainer.trainMultipleModels(
                        config.dataPath,
                        config.outputDir,
                        config.algorithms,
                        config.maxSamples,
                        config.showFeatureImportance,
                        config.topFeaturesCount
                );
            }

            // Print summary
            printSummary(results);

            // Exit with appropriate code
            boolean allSucceeded = results.stream().allMatch(ModelTrainer.TrainingResult::isSuccess);
            System.exit(allSucceeded ? 0 : 1);

        } catch (Exception e) {
            logger.error("Training failed with error: {}", e.getMessage(), e);
            System.err.println("ERROR: " + e.getMessage());
            System.exit(1);
        }
    }

    private static Config parseArguments(String[] args) {
        Config config = new Config();

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--data", "-d":
                    if (i + 1 < args.length) {
                        config.dataPath = args[++i];
                    }
                    break;

                case "--output", "-o":
                    if (i + 1 < args.length) {
                        config.outputDir = args[++i];
                    }
                    break;

                case "--algorithm", "-a":
                    if (i + 1 < args.length) {
                        config.singleAlgorithm = parseAlgorithm(args[++i]);
                    }
                    break;

                case "--algorithms":
                    if (i + 1 < args.length) {
                        String[] algNames = args[++i].split(",");
                        config.algorithms = Arrays.stream(algNames)
                                .map(String::trim)
                                .map(ModelTrainingCLI::parseAlgorithm)
                                .toList();
                    }
                    break;

                case "--max-samples", "-n":
                    if (i + 1 < args.length) {
                        config.maxSamples = Integer.parseInt(args[++i]);
                    }
                    break;

                case "--show-feature-importance", "--feature-importance":
                    config.showFeatureImportance = true;
                    break;

                case "--top-features":
                    if (i + 1 < args.length) {
                        config.topFeaturesCount = Integer.parseInt(args[++i]);
                    }
                    break;

                case "--help", "-h":
                    printUsage();
                    System.exit(0);
                    break;

                default:
                    System.err.println("Unknown argument: " + args[i]);
                    printUsage();
                    System.exit(1);
            }
        }

        return config;
    }

    private static AlgorithmType parseAlgorithm(String name) {
        return switch (name.toLowerCase()) {
            case "svm" -> AlgorithmType.SVM;
            case "naive_bayes", "nb" -> AlgorithmType.NAIVE_BAYES;
            case "random_forest", "rf" -> AlgorithmType.RANDOM_FOREST;
            case "logistic", "logistic_regression", "lr" -> AlgorithmType.LOGISTIC_REGRESSION;
            default -> throw new IllegalArgumentException("Unknown algorithm: " + name);
        };
    }

    private static void validateConfiguration(Config config) throws Exception {
        // Check data file exists
        Path dataPath = Paths.get(config.dataPath);
        if (!Files.exists(dataPath)) {
            throw new IllegalArgumentException("Data file does not exist: " + config.dataPath);
        }

        if (!Files.isReadable(dataPath)) {
            throw new IllegalArgumentException("Data file is not readable: " + config.dataPath);
        }

        // Create output directory if needed
        Path outputDir = Paths.get(config.outputDir);
        if (!Files.exists(outputDir)) {
            logger.info("Creating output directory: {}", config.outputDir);
            Files.createDirectories(outputDir);
        }

        // Validate max samples
        if (config.maxSamples < 0) {
            throw new IllegalArgumentException("max-samples must be >= 0");
        }

        // If no algorithms specified, use defaults
        if (config.singleAlgorithm == null && config.algorithms.isEmpty()) {
            config.algorithms = Arrays.asList(
                    AlgorithmType.SVM,
                    AlgorithmType.NAIVE_BAYES,
                    AlgorithmType.RANDOM_FOREST
            );
            logger.info("No algorithms specified, using defaults: SVM, Naive Bayes, Random Forest");
        }
    }

    private static void printSummary(List<ModelTrainer.TrainingResult> results) {
        logger.info("");
        logger.info("========================================");
        logger.info("  Training Summary");
        logger.info("========================================");

        int successful = 0;
        int failed = 0;
        long totalTrainingTime = 0;

        for (ModelTrainer.TrainingResult result : results) {
            if (result.isSuccess()) {
                successful++;
                totalTrainingTime += result.getTrainingTimeMs();
                logger.info("✅ {} - train={}, val={}, test={} (total={}), trained in {:.2f}s, saved to {}",
                        result.getAlgorithm().getDisplayName(),
                        result.getTrainSampleCount(),
                        result.getValSampleCount(),
                        result.getTestSampleCount(),
                        result.getTotalSampleCount(),
                        result.getTrainingTimeMs() / 1000.0,
                        result.getOutputPath());
            } else {
                failed++;
                logger.error("❌ {} - FAILED: {}",
                        result.getAlgorithm().getDisplayName(),
                        result.getErrorMessage());
            }
        }

        logger.info("");
        logger.info("Total: {} successful, {} failed", successful, failed);
        logger.info("Total training time: {:.2f}s", totalTrainingTime / 1000.0);
        logger.info("========================================");
    }

    private static void printUsage() {
        System.out.println("\nUsage: ModelTrainingCLI [options]");
        System.out.println("\nOptions:");
        System.out.println("  --data, -d <path>           Path to training data CSV file");
        System.out.println("                              (default: " + DEFAULT_DATA_PATH + ")");
        System.out.println("");
        System.out.println("  --output, -o <dir>          Output directory for trained models");
        System.out.println("                              (default: " + DEFAULT_OUTPUT_DIR + ")");
        System.out.println("");
        System.out.println("  --algorithm, -a <name>      Train single algorithm");
        System.out.println("                              Options: svm, naive_bayes, random_forest, logistic");
        System.out.println("");
        System.out.println("  --algorithms <list>         Train multiple algorithms (comma-separated)");
        System.out.println("                              Example: svm,naive_bayes,random_forest");
        System.out.println("");
        System.out.println("  --max-samples, -n <num>     Maximum training samples (0 = use all)");
        System.out.println("                              (default: " + DEFAULT_MAX_SAMPLES + ")");
        System.out.println("");
        System.out.println("  --show-feature-importance   Analyze and display feature importance after training");
        System.out.println("                              (works for all model types)");
        System.out.println("");
        System.out.println("  --top-features <num>        Number of top features to display");
        System.out.println("                              (default: 30, used with --show-feature-importance)");
        System.out.println("");
        System.out.println("  --help, -h                  Show this help message");
        System.out.println("");
        System.out.println("Examples:");
        System.out.println("  # Train all default models");
        System.out.println("  java -jar sentiment-analyzer.jar sentiment.training.ModelTrainingCLI");
        System.out.println("");
        System.out.println("  # Train only SVM with 50k samples");
        System.out.println("  java -jar sentiment-analyzer.jar sentiment.training.ModelTrainingCLI \\");
        System.out.println("    --algorithm svm --max-samples 50000");
        System.out.println("");
        System.out.println("  # Train specific algorithms");
        System.out.println("  java -jar sentiment-analyzer.jar sentiment.training.ModelTrainingCLI \\");
        System.out.println("    --algorithms svm,naive_bayes --data ./data/reviews.csv");
        System.out.println("");
        System.out.println("  # Train SVM with feature importance analysis");
        System.out.println("  java -jar sentiment-analyzer.jar sentiment.training.ModelTrainingCLI \\");
        System.out.println("    --algorithm svm --show-feature-importance --top-features 50");
        System.out.println("");
        System.out.println("  # Train Naive Bayes with feature importance");
        System.out.println("  java -jar sentiment-analyzer.jar sentiment.training.ModelTrainingCLI \\");
        System.out.println("    --algorithm naive_bayes --show-feature-importance");
        System.out.println("");
    }

    private static class Config {
        String dataPath = DEFAULT_DATA_PATH;
        String outputDir = DEFAULT_OUTPUT_DIR;
        AlgorithmType singleAlgorithm = null;
        List<AlgorithmType> algorithms = List.of();
        int maxSamples = DEFAULT_MAX_SAMPLES;
        boolean showFeatureImportance = false;
        int topFeaturesCount = 30;
    }
}
