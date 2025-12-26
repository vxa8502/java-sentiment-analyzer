package sentiment.training;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import sentiment.config.FeatureExtractionProperties;
import sentiment.models.AlgorithmType;

/**
 * Bootstrap tool for training initial models using minimal Spring context.
 */
public class TrainModel {

    @Configuration
    @ComponentScan(basePackages = {
        "sentiment.training",
        "sentiment.preprocessing",
        "sentiment.data",
        "sentiment.models",
        "sentiment.evaluation"
    })
    @EnableConfigurationProperties(FeatureExtractionProperties.class)
    static class TrainingConfig {}

    public static void main(String[] args) {
        if (args.length < 3) {
            printUsage();
            System.exit(1);
        }

        String dataPath = args[0];
        String outputPath = args[1];
        String algorithmName = args[2];
        int maxSamples = args.length > 3 ? Integer.parseInt(args[3]) : 100000;
        boolean showFeatureImportance = args.length > 4 && Boolean.parseBoolean(args[4]);
        int topFeaturesCount = args.length > 5 ? Integer.parseInt(args[5]) : 30;
        boolean enableHyperparameterTuning = args.length > 6 && Boolean.parseBoolean(args[6]);
        String testDataPath = args.length > 7 ? args[7] : null;

        AlgorithmType algorithmType;
        try {
            algorithmType = AlgorithmType.valueOf(algorithmName.toUpperCase().replace("-", "_"));
        } catch (IllegalArgumentException e) {
            System.err.println("Invalid algorithm: " + algorithmName);
            System.err.println("Valid options: svm, naive_bayes, random_forest, logistic_regression");
            System.exit(1);
            return;
        }

        System.out.println("========================================");
        System.out.println("  Sentiment Model Training Tool");
        System.out.println("========================================");
        System.out.println("Data: " + dataPath);
        if (testDataPath != null) {
            System.out.println("Test data: " + testDataPath + " (using official test set)");
        }
        System.out.println("Output: " + outputPath);
        System.out.println("Algorithm: " + algorithmType.getDisplayName());
        System.out.println("Max samples: " + maxSamples);
        System.out.println("Feature importance: " + showFeatureImportance);
        System.out.println("Hyperparameter tuning: " + enableHyperparameterTuning);
        System.out.println("========================================\n");

        try {
            // Create minimal Spring context (NOT Spring Boot - just DI container)
            System.out.println("Initializing Spring context...");
            AnnotationConfigApplicationContext context =
                new AnnotationConfigApplicationContext(TrainingConfig.class);

            ModelTrainer trainer = context.getBean(ModelTrainer.class);

            System.out.println("Starting training...\n");

            // Train and save (with optional test data path for pre-split datasets)
            ModelTrainer.TrainingResult result = trainer.trainAndSave(
                    dataPath,
                    testDataPath,
                    outputPath,
                    algorithmType,
                    maxSamples,
                    showFeatureImportance,
                    topFeaturesCount,
                    enableHyperparameterTuning
            );

            // Clean up Spring context
            context.close();

            if (result.isSuccess()) {
                System.out.println("\n========================================");
                System.out.println("   Training Complete!");
                System.out.println("========================================");
                System.out.println("Model saved to: " + result.getOutputPath());
                System.out.println("Training time: " + (result.getTrainingTimeMs() / 1000.0) + "s");
                System.out.println("Train samples: " + result.getTrainSampleCount());
                System.out.println("Validation samples: " + result.getValSampleCount());
                System.out.println("Test samples: " + result.getTestSampleCount());
                System.out.println("Total samples: " + result.getTotalSampleCount());
                System.out.println("========================================\n");
                System.exit(0);
            } else {
                System.err.println("\n Training failed: " + result.getErrorMessage());
                System.exit(1);
            }

        } catch (Exception e) {
            System.err.println("\n Training failed: " + e.getClass().getSimpleName() + ": " + e.getMessage());
            if (e.getCause() != null) {
                System.err.println("Caused by: " + e.getCause().getMessage());
            }
            System.exit(1);
        }
    }

    private static void printUsage() {
        System.out.println("\nUsage: TrainModel <dataPath> <outputPath> <algorithm> [maxSamples] [showFeatureImportance] [topFeaturesCount] [enableHyperparameterTuning] [testDataPath]");
        System.out.println("\nAlgorithms: svm, naive_bayes, random_forest, logistic_regression");
        System.out.println("\nExamples:");
        System.out.println("  Single-file dataset (60/20/20 split):");
        System.out.println("    java TrainModel data.csv models/svm.ser svm 100000 false 30 false");
        System.out.println("\n  Pre-split dataset (80/20 train/val + official test):");
        System.out.println("    java TrainModel train.csv models/svm.ser svm 100000 false 30 false test.csv");
        System.out.println("\n  With hyperparameter tuning:");
        System.out.println("    java TrainModel data.csv models/svm.ser svm 100000 false 30 true");
    }
}
