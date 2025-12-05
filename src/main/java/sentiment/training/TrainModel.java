package sentiment.training;

import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
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
    static class TrainingConfig {}

    public static void main(String[] args) {
        if (args.length < 2) {
            printUsage();
            System.exit(1);
        }

        String dataPath = args[0];
        String outputPath = args[1];
        int maxSamples = args.length > 2 ? Integer.parseInt(args[2]) : 10000;
        boolean showFeatureImportance = args.length > 3 && Boolean.parseBoolean(args[3]);
        int topFeaturesCount = args.length > 4 ? Integer.parseInt(args[4]) : 30;
        boolean enableHyperparameterTuning = args.length > 5 && Boolean.parseBoolean(args[5]);

        System.out.println("========================================");
        System.out.println("  Sentiment Model Training Tool");
        System.out.println("========================================");
        System.out.println("Data: " + dataPath);
        System.out.println("Output: " + outputPath);
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

            // Train and save
            ModelTrainer.TrainingResult result = trainer.trainAndSave(
                    dataPath,
                    outputPath,
                    AlgorithmType.SVM,
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
        System.out.println("\nUsage: TrainModel <dataPath> <outputPath> [maxSamples] [showFeatureImportance] [topFeaturesCount] [enableHyperparameterTuning]");
        System.out.println("\nExample:");
        System.out.println("  mvn exec:java -Dexec.mainClass=\"sentiment.training.TrainModel\" \\");
        System.out.println("    -Dexec.args=\"./data/datasets/Reviews.csv ./models/svm-model.ser 10000 true 30 false\"");
        System.out.println("\nWith hyperparameter tuning (slower but more accurate):");
        System.out.println("  mvn exec:java -Dexec.mainClass=\"sentiment.training.TrainModel\" \\");
        System.out.println("    -Dexec.args=\"./data/datasets/Reviews.csv ./models/svm-model.ser 10000 true 30 true\"");
    }
}
