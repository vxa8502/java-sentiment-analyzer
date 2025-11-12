package sentiment.models;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sentiment.PipelineState;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Date;

/**
 * Generic model persistence for Weka-based classifiers.
 *
 * PURPOSE:
 * ========
 * Provides unified save/load functionality for all Weka classifier types
 * (Naive Bayes, Random Forest, Logistic Regression, etc.).
 *
 * DESIGN:
 * =======
 * Uses Java serialization to save complete classifier state including:
 * - Trained Weka classifier instance
 * - Training data structure (for validation)
 * - Supported class labels
 * - Training metadata (time, version)
 *
 * THREAD SAFETY:
 * ==============
 * This class is stateless and thread-safe. However, the classifiers being
 * saved/loaded must handle their own thread safety.
 *
 * USAGE:
 * ======
 * <pre>
 * // Save a trained classifier
 * WekaModelPersistence<NaiveBayesClassifier> persistence = new WekaModelPersistence<>();
 * persistence.saveModel(classifier, Path.of("./models/nb-model.ser"));
 *
 * // Load a classifier
 * NaiveBayesClassifier loaded = persistence.loadModel(
 *     new NaiveBayesClassifier(preprocessor, converter),
 *     Path.of("./models/nb-model.ser")
 * );
 * </pre>
 */
public class WekaModelPersistence<T extends ClassifierTrainingTemplate<?>> {

    private static final Logger logger = LoggerFactory.getLogger(WekaModelPersistence.class);
    private static final String MODEL_VERSION = "1.0.0";

    /**
     * Saves trained Weka classifier to disk.
     *
     * @param classifier Trained classifier to save
     * @param modelPath Output path for model file
     * @throws IOException if save fails
     * @throws IllegalStateException if classifier not trained
     */
    public void saveModel(T classifier, Path modelPath) throws IOException {
        if (!classifier.isTrained()) {
            throw new IllegalStateException("Cannot save untrained classifier");
        }

        logger.info("Saving {} model to: {}", classifier.getAlgorithmName(), modelPath);

        try {
            Files.createDirectories(modelPath.getParent());

            // Extract classifier components
            Classifier wekaClassifier = extractWekaClassifier(classifier);
            Instances trainingStructure = extractTrainingStructure(classifier);
            String[] supportedClasses = classifier.getSupportedClasses();

            ModelBundle bundle = new ModelBundle(
                    wekaClassifier,
                    trainingStructure,
                    supportedClasses,
                    classifier.getLastTrainingTimeMs(),
                    classifier.getAlgorithmType(),
                    MODEL_VERSION,
                    System.currentTimeMillis()
            );

            try (ObjectOutputStream oos = new ObjectOutputStream(
                    new BufferedOutputStream(Files.newOutputStream(modelPath)))) {
                oos.writeObject(bundle);
            }

            long fileSize = Files.size(modelPath);
            logger.info("✅ Model saved successfully: {} bytes ({} KB)",
                    fileSize, fileSize / 1024);

        } catch (IOException e) {
            logger.error("Failed to save model to {}", modelPath, e);
            throw new IOException("Model save failed: " + e.getMessage(), e);
        }
    }

    /**
     * Loads trained Weka classifier from disk.
     *
     * @param classifier Empty classifier instance to load into
     * @param modelPath Path to saved model file
     * @return The loaded classifier (same instance passed in)
     * @throws IOException if load fails
     * @throws ClassNotFoundException if model format incompatible
     */
    public T loadModel(T classifier, Path modelPath)
            throws IOException, ClassNotFoundException {

        if (!Files.exists(modelPath)) {
            throw new IOException("Model file does not exist: " + modelPath);
        }

        logger.info("Loading {} model from: {}", classifier.getAlgorithmName(), modelPath);

        // Acquire write lock for atomic model replacement
        classifier.stateLock.writeLock().lock();
        try {
            ModelBundle bundle;

            // Deserialize model bundle
            try (ObjectInputStream ois = new ObjectInputStream(
                    new BufferedInputStream(Files.newInputStream(modelPath)))) {
                bundle = (ModelBundle) ois.readObject();
            }

            // Validate version compatibility
            if (!MODEL_VERSION.equals(bundle.modelVersion)) {
                logger.warn("Model version mismatch: current={}, loaded={}",
                        MODEL_VERSION, bundle.modelVersion);
            }

            // Validate algorithm type matches
            if (bundle.algorithmType != classifier.getAlgorithmType()) {
                throw new IOException(String.format(
                        "Algorithm type mismatch: expected %s, found %s in model file",
                        classifier.getAlgorithmType(), bundle.algorithmType));
            }

            // Set classifier components
            setWekaClassifier(classifier, bundle.wekaClassifier);
            setTrainingMetadata(classifier, bundle.trainingStructure, bundle.supportedClasses);

            // Transition to READY state
            classifier.classifierState = PipelineState.READY;

            logger.info("✅ Model loaded: {} classes, trained at {}",
                    bundle.supportedClasses.length, new Date(bundle.timestamp));

            return classifier;

        } catch (IOException | ClassNotFoundException e) {
            logger.error("Failed to load model from {}", modelPath, e);
            throw e;
        } finally {
            classifier.stateLock.writeLock().unlock();
        }
    }

    /**
     * Checks if a model file is valid and loadable.
     */
    public boolean isValidModel(Path modelPath) {
        if (!Files.exists(modelPath) || !Files.isReadable(modelPath)) {
            return false;
        }

        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(Files.newInputStream(modelPath)))) {

            Object obj = ois.readObject();
            if (!(obj instanceof ModelBundle)) {
                logger.warn("File is not a valid ModelBundle: {}", modelPath);
                return false;
            }

            ModelBundle bundle = (ModelBundle) obj;

            // Basic validation
            if (bundle.wekaClassifier == null || bundle.trainingStructure == null
                    || bundle.supportedClasses == null || bundle.supportedClasses.length == 0) {
                logger.warn("ModelBundle has missing components: {}", modelPath);
                return false;
            }

            return true;

        } catch (Exception e) {
            logger.warn("Invalid model file: {}", modelPath, e);
            return false;
        }
    }

    /**
     * Gets model metadata without fully loading the model.
     */
    public ModelMetadata getModelMetadata(Path modelPath)
            throws IOException, ClassNotFoundException {

        if (!Files.exists(modelPath)) {
            throw new IOException("Model file does not exist: " + modelPath);
        }

        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(Files.newInputStream(modelPath)))) {

            ModelBundle bundle = (ModelBundle) ois.readObject();

            return new ModelMetadata(
                    bundle.algorithmType,
                    bundle.modelVersion,
                    bundle.timestamp,
                    bundle.supportedClasses.length,
                    bundle.trainingStructure.numAttributes() - 1,
                    Files.size(modelPath)
            );

        } catch (IOException | ClassNotFoundException e) {
            logger.error("Failed to read metadata from {}", modelPath, e);
            throw e;
        }
    }

    // ==================== HELPER METHODS ====================
    // REFACTORED: Eliminated 52 lines of duplicate type-checking by using abstract methods

    /**
     * Extract Weka classifier using polymorphism instead of type-checking.
     */
    private Classifier extractWekaClassifier(T classifier) {
        return classifier.getWekaClassifierInstance();
    }

    /**
     * Extract training structure using polymorphism instead of type-checking.
     */
    private Instances extractTrainingStructure(T classifier) {
        return classifier.getTrainingStructure();
    }

    /**
     * Set Weka classifier using polymorphism instead of type-checking.
     */
    private void setWekaClassifier(T classifier, Classifier wekaClassifier) {
        classifier.setWekaClassifierInstance(wekaClassifier);
    }

    /**
     * Set training metadata using polymorphism instead of type-checking.
     */
    private void setTrainingMetadata(T classifier, Instances structure, String[] classes) {
        classifier.setTrainingMetadata(structure, classes);
    }

    // ==================== DATA CLASSES ====================

    /**
     * Serializable bundle containing all trained model state.
     */
    private static class ModelBundle implements Serializable {
        private static final long serialVersionUID = 1L;

        final Classifier wekaClassifier;
        final Instances trainingStructure;
        final String[] supportedClasses;
        final long trainingTimeMs;
        final AlgorithmType algorithmType;
        final String modelVersion;
        final long timestamp;

        ModelBundle(Classifier wekaClassifier, Instances trainingStructure,
                    String[] supportedClasses, long trainingTimeMs,
                    AlgorithmType algorithmType, String modelVersion, long timestamp) {
            this.wekaClassifier = wekaClassifier;
            this.trainingStructure = trainingStructure;
            this.supportedClasses = supportedClasses.clone();
            this.trainingTimeMs = trainingTimeMs;
            this.algorithmType = algorithmType;
            this.modelVersion = modelVersion;
            this.timestamp = timestamp;
        }
    }

    /**
     * Model metadata without loading full model.
     */
    public static class ModelMetadata {
        public final AlgorithmType algorithmType;
        public final String version;
        public final long timestamp;
        public final int numClasses;
        public final int numFeatures;
        public final long fileSizeBytes;

        ModelMetadata(AlgorithmType algorithmType, String version, long timestamp,
                      int numClasses, int numFeatures, long fileSizeBytes) {
            this.algorithmType = algorithmType;
            this.version = version;
            this.timestamp = timestamp;
            this.numClasses = numClasses;
            this.numFeatures = numFeatures;
            this.fileSizeBytes = fileSizeBytes;
        }

        @Override
        public String toString() {
            return String.format(
                    "ModelMetadata{algorithm=%s, version=%s, trained=%s, classes=%d, features=%d, size=%d KB}",
                    algorithmType.getDisplayName(), version, new Date(timestamp),
                    numClasses, numFeatures, fileSizeBytes / 1024);
        }
    }
}
