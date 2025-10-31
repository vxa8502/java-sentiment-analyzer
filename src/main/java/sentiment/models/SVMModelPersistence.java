package sentiment.models;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sentiment.PipelineState;
import weka.classifiers.functions.SMO;
import weka.core.Instances;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Date;

/**
 * ✅ UPDATED: Simplified model persistence to match memory-optimized classifier
 *
 * Model persistence operations for SVM classifier.
 * Saves/loads complete trained state atomically.
 *
 * CHANGES:
 * ========
 * - Updated setTrainingMetadata() call to use simplified signature
 * - No longer stores duplicate instance/feature counts
 * - Structure alone is source of truth
 */
public class SVMModelPersistence {

    private static final Logger logger = LoggerFactory.getLogger(SVMModelPersistence.class);
    private static final String MODEL_VERSION = "1.0.0";

    /**
     * Saves trained SVM classifier to disk.
     *
     * @param classifier Trained classifier to save
     * @param modelPath Output path for model file
     * @throws IOException if save fails
     * @throws IllegalStateException if classifier not trained
     */
    public void saveModel(SVMClassifier classifier, Path modelPath) throws IOException {
        if (!classifier.isTrained()) {
            throw new IllegalStateException("Cannot save untrained classifier");
        }

        logger.info("Saving SVM model to: {}", modelPath);

        try {
            Files.createDirectories(modelPath.getParent());

            ModelBundle bundle = new ModelBundle(
                    classifier.getSMO(),
                    classifier.getTrainingStructure(),
                    classifier.getSupportedClasses(),
                    classifier.getLastTrainingTimeMs(),
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
     * Loads trained SVM classifier from disk.
     *
     * ✅ UPDATED: Uses simplified setTrainingMetadata() signature
     *
     * CRITICAL: Replaces entire SMO instance atomically.
     * Uses classifier's write lock for exclusive access during replacement.
     *
     * @param classifier Classifier instance to load into
     * @param modelPath Path to saved model file
     * @throws IOException if load fails
     * @throws ClassNotFoundException if model format incompatible
     */
    public void loadModel(SVMClassifier classifier, Path modelPath)
            throws IOException, ClassNotFoundException {

        if (!Files.exists(modelPath)) {
            throw new IOException("Model file does not exist: " + modelPath);
        }

        logger.info("Loading SVM model from: {}", modelPath);

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

            // ✅ ATOMIC REPLACEMENT: Set all state at once
            classifier.setSMO(bundle.smoClassifier);

            // ✅ MEMORY FIX: Simplified call - structure contains all needed info
            classifier.setTrainingMetadata(
                    bundle.trainingStructure,
                    bundle.supportedClasses
            );

            // Transition to READY state
            classifier.classifierState = PipelineState.READY;

            logger.info("✅ Model loaded: {} classes, trained at {}",
                    bundle.supportedClasses.length, new Date(bundle.timestamp));
            logger.debug("SMO parameters: C={}, epsilon={}",
                    bundle.smoClassifier.getC(), bundle.smoClassifier.getEpsilon());

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
            if (bundle.smoClassifier == null || bundle.trainingStructure == null
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

    // ==================== DATA CLASSES ====================

    /**
     * Serializable bundle containing all trained model state.
     */
    private static class ModelBundle implements Serializable {
        private static final long serialVersionUID = 1L;

        final SMO smoClassifier;
        final Instances trainingStructure;
        final String[] supportedClasses;
        final long trainingTimeMs;
        final String modelVersion;
        final long timestamp;

        ModelBundle(SMO smoClassifier, Instances trainingStructure,
                    String[] supportedClasses, long trainingTimeMs,
                    String modelVersion, long timestamp) {
            this.smoClassifier = smoClassifier;
            this.trainingStructure = trainingStructure;
            this.supportedClasses = supportedClasses.clone();
            this.trainingTimeMs = trainingTimeMs;
            this.modelVersion = modelVersion;
            this.timestamp = timestamp;
        }
    }

    /**
     * Model metadata without loading full model.
     */
    public static class ModelMetadata {
        public final String version;
        public final long timestamp;
        public final int numClasses;
        public final int numFeatures;
        public final long fileSizeBytes;

        ModelMetadata(String version, long timestamp, int numClasses,
                      int numFeatures, long fileSizeBytes) {
            this.version = version;
            this.timestamp = timestamp;
            this.numClasses = numClasses;
            this.numFeatures = numFeatures;
            this.fileSizeBytes = fileSizeBytes;
        }

        @Override
        public String toString() {
            return String.format(
                    "ModelMetadata{version=%s, trained=%s, classes=%d, features=%d, size=%d KB}",
                    version, new Date(timestamp), numClasses, numFeatures,
                    fileSizeBytes / 1024);
        }
    }
}