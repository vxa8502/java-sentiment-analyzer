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
import java.util.Set;

/**
 * Generic model persistence for Weka-based classifiers.
 */
public class WekaModelPersistence<T extends ClassifierTrainingTemplate<?>> {

    private static final Logger logger = LoggerFactory.getLogger(WekaModelPersistence.class);
    private static final String MODEL_VERSION = "2.0.0";

    /**
     * Set of model versions that are compatible with the current code.
     * Allows loading older models that are still structurally compatible.
     */
    private static final Set<String> COMPATIBLE_VERSIONS = Set.of(
            "2.0.0", // Current version (production model)
            "1.0.0"  // Legacy version (backward compatibility)
    );

    /**
     * Dangerous classes that must NEVER be deserialized (RCE gadget chains).
     * These classes can be used to execute arbitrary code during deserialization.
     */
    private static final Set<String> BLOCKED_CLASSES = Set.of(
            "java.lang.ProcessBuilder",              // RCE: command execution
            "java.lang.Runtime",                     // RCE: command execution
            "java.lang.ProcessBuilder$Redirect",     // RCE: process I/O redirect
            "javax.script.ScriptEngineManager",      // RCE: script execution
            "javax.script.CompiledScript"            // RCE: compiled scripts
    );

    /**
     * Dangerous package prefixes that must NEVER be deserialized.
     */
    private static final Set<String> BLOCKED_PACKAGES = Set.of(
            "java.lang.reflect.",                    // Reflection attacks
            "java.lang.invoke.",                     // MethodHandle attacks
            "javax.management.",                     // JMX attacks
            "com.sun.jndi.",                         // JNDI injection
            "com.sun.rowset.",                       // XMLDecoder attacks
            "org.apache.commons.collections.functors.", // Commons Collections gadgets
            "org.apache.commons.collections4.functors.", // Commons Collections 4 gadgets
            "org.apache.xalan.",                     // Xalan gadgets
            "com.mchange.",                          // C3P0 gadgets
            "org.hibernate.",                        // Hibernate gadgets
            "org.springframework.beans.factory.",    // Spring gadgets
            "org.springframework.aop.",              // Spring AOP gadgets
            "org.springframework.transaction.support.", // Spring TX gadgets
            "com.sun.org.apache.xalan.",             // Internal Xalan
            "com.sun.org.apache.xml.",               // Internal XML
            "sun.rmi."                               // RMI attacks
    );

    /**
     * Allowed package prefixes for model deserialization.
     */
    private static final Set<String> ALLOWED_PACKAGES = Set.of(
            "sentiment.",                            // Our model/preprocessing classes
            "weka.",                                 // Weka ML library
            "no.uib.cipr.matrix.",                   // Matrix library
            "java.lang.",                            // Primitives, String, etc.
            "java.util.",                            // Collections
            "java.io.",                              // Serializable, streams
            "java.math.",                            // BigDecimal, BigInteger
            "java.text.",                            // Formatters
            "java.time.",                            // Date/time classes
            "[L",                                    // Object arrays
            "[I", "[J", "[D", "[F", "[B", "[S", "[C", "[Z" // Primitive arrays
    );

    /**
     * ObjectInputFilter to prevent deserialization attacks (RCE).
     *
     * <p>SECURITY: Uses a programmatic filter for precise control over allowed classes.
     * Explicitly blocks known RCE gadget classes while allowing legitimate ML model classes.
     *
     * <p>This filter prevents attacks like:
     * <ul>
     *   <li>ProcessBuilder/Runtime command execution</li>
     *   <li>Reflection-based attacks</li>
     *   <li>JNDI injection</li>
     *   <li>Commons Collections gadget chains</li>
     * </ul>
     */
    private static final ObjectInputFilter MODEL_DESERIALIZATION_FILTER = filterInfo -> {
        Class<?> clazz = filterInfo.serialClass();

        // Arrays and primitives - check array depth limit
        if (clazz == null) {
            // Null class means checking depth/array limits - allow
            return filterInfo.depth() > 100 ? ObjectInputFilter.Status.REJECTED
                                            : ObjectInputFilter.Status.UNDECIDED;
        }

        String className = clazz.getName();

        // BLOCK: Check against explicit blocklist first
        if (BLOCKED_CLASSES.contains(className)) {
            logger.warn("SECURITY: Blocked deserialization of dangerous class: {}", className);
            return ObjectInputFilter.Status.REJECTED;
        }

        // BLOCK: Check against dangerous package prefixes
        for (String blockedPrefix : BLOCKED_PACKAGES) {
            if (className.startsWith(blockedPrefix)) {
                logger.warn("SECURITY: Blocked deserialization of class in dangerous package: {}", className);
                return ObjectInputFilter.Status.REJECTED;
            }
        }

        // ALLOW: Check against allowed package prefixes
        for (String allowedPrefix : ALLOWED_PACKAGES) {
            if (className.startsWith(allowedPrefix)) {
                return ObjectInputFilter.Status.ALLOWED;
            }
        }

        // ALLOW: Primitive types and their arrays
        if (clazz.isPrimitive() || clazz.isArray()) {
            return ObjectInputFilter.Status.ALLOWED;
        }

        // DENY: Everything else
        logger.warn("SECURITY: Blocked deserialization of unexpected class: {}", className);
        return ObjectInputFilter.Status.REJECTED;
    };

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

            // Extract converter components (filters and structure)
            sentiment.preprocessing.WekaInstancesConverter converter = classifier.getConverter();
            weka.filters.unsupervised.attribute.StringToWordVector stringToWordFilter = null;
            weka.filters.unsupervised.attribute.Normalize normalizeFilter = null;
            Instances filterTrainingStructure = null;

            if (converter != null) {
                stringToWordFilter = converter.getTrainedStringToWordFilter();
                normalizeFilter = converter.getTrainedNormalizationFilter();
                // CRITICAL: Save the RAW structure (before filters), not the final structure
                filterTrainingStructure = converter.getFilterTrainingStructure();
            }

            ModelBundle bundle = new ModelBundle(
                    wekaClassifier,
                    trainingStructure,
                    supportedClasses,
                    classifier.getLastTrainingTimeMs(),
                    classifier.getAlgorithmType(),
                    MODEL_VERSION,
                    System.currentTimeMillis(),
                    stringToWordFilter,
                    normalizeFilter,
                    filterTrainingStructure
            );

            try (ObjectOutputStream oos = new ObjectOutputStream(
                    new BufferedOutputStream(Files.newOutputStream(modelPath)))) {
                oos.writeObject(bundle);
            }

            long fileSize = Files.size(modelPath);
            logger.info("Model saved successfully: {} bytes ({} KB)",
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
        classifier.acquireWriteLock();
        try {
            ModelBundle bundle;

            // Deserialize model bundle (with security filter to prevent RCE)
            try (ObjectInputStream ois = createSecureObjectInputStream(
                    Files.newInputStream(modelPath))) {
                bundle = (ModelBundle) ois.readObject();
            }

            // Validate version compatibility - reject incompatible versions
            if (!COMPATIBLE_VERSIONS.contains(bundle.modelVersion)) {
                throw new IOException(String.format(
                        "Incompatible model version: loaded=%s, supported=%s. " +
                        "Please retrain the model with the current application version.",
                        bundle.modelVersion, COMPATIBLE_VERSIONS));
            }
            if (!MODEL_VERSION.equals(bundle.modelVersion)) {
                logger.info("Loading compatible model version {} (current: {})",
                        bundle.modelVersion, MODEL_VERSION);
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

            // Restore converter filters if available (version 2+ bundles)
            sentiment.preprocessing.WekaInstancesConverter converter = classifier.getConverter();
            if (converter != null && bundle.stringToWordFilter != null) {
                converter.setTrainedStringToWordFilter(bundle.stringToWordFilter);
                converter.setTrainedNormalizationFilter(bundle.normalizeFilter);
                // Note: filterTrainingStructure is already set by setTrainingMetadata
            }

            // Transition to READY state
            classifier.setState(PipelineState.READY);

            logger.info("Model loaded: {} classes, trained at {}",
                    bundle.supportedClasses.length, new Date(bundle.timestamp));

            return classifier;

        } catch (IOException | ClassNotFoundException e) {
            logger.error("Failed to load model from {}", modelPath, e);
            throw e;
        } finally {
            classifier.releaseWriteLock();
        }
    }

    /**
     * Checks if a model file is valid and loadable.
     */
    public boolean isValidModel(Path modelPath) {
        if (!Files.exists(modelPath) || !Files.isReadable(modelPath)) {
            return false;
        }

        try (ObjectInputStream ois = createSecureObjectInputStream(
                Files.newInputStream(modelPath))) {

            Object obj = ois.readObject();
            if (!(obj instanceof ModelBundle bundle)) {
                logger.warn("File is not a valid ModelBundle: {}", modelPath);
                return false;
            }

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

        try (ObjectInputStream ois = createSecureObjectInputStream(
                Files.newInputStream(modelPath))) {

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

    // HELPER METHODS

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

    /**
     * Creates a secure ObjectInputStream with deserialization filter applied.
     * Prevents RCE attacks by only allowing known safe classes to be deserialized.
     */
    private static ObjectInputStream createSecureObjectInputStream(InputStream in) throws IOException {
        ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(in));
        ois.setObjectInputFilter(MODEL_DESERIALIZATION_FILTER);
        return ois;
    }

    // DATA CLASSES

    /**
     * Serializable bundle containing all trained model state.
     *
     * IMPORTANT: This must remain backwards compatible. New models can include converter filters,
     * but old models without filters should still load successfully.
     */
    private static class ModelBundle implements Serializable {
        @Serial
        private static final long serialVersionUID = 2L; // Version 2: includes filter state

        final Classifier wekaClassifier;
        final Instances trainingStructure;
        final String[] supportedClasses;
        final long trainingTimeMs;
        final AlgorithmType algorithmType;
        final String modelVersion;
        final long timestamp;
        // Optional fields for converter state (may be null in old models)
        final weka.filters.unsupervised.attribute.StringToWordVector stringToWordFilter;
        final weka.filters.unsupervised.attribute.Normalize normalizeFilter;
        final Instances filterTrainingStructure;

        // Constructor for old format (backwards compatibility)
        ModelBundle(Classifier wekaClassifier, Instances trainingStructure,
                    String[] supportedClasses, long trainingTimeMs,
                    AlgorithmType algorithmType, String modelVersion, long timestamp) {
            this(wekaClassifier, trainingStructure, supportedClasses, trainingTimeMs,
                 algorithmType, modelVersion, timestamp, null, null, null);
        }

        // Constructor for new format (with filters)
        ModelBundle(Classifier wekaClassifier, Instances trainingStructure,
                    String[] supportedClasses, long trainingTimeMs,
                    AlgorithmType algorithmType, String modelVersion, long timestamp,
                    weka.filters.unsupervised.attribute.StringToWordVector stringToWordFilter,
                    weka.filters.unsupervised.attribute.Normalize normalizeFilter,
                    Instances filterTrainingStructure) {
            this.wekaClassifier = wekaClassifier;
            this.trainingStructure = trainingStructure;
            this.supportedClasses = supportedClasses.clone();
            this.trainingTimeMs = trainingTimeMs;
            this.algorithmType = algorithmType;
            this.modelVersion = modelVersion;
            this.timestamp = timestamp;
            this.stringToWordFilter = stringToWordFilter;
            this.normalizeFilter = normalizeFilter;
            this.filterTrainingStructure = filterTrainingStructure;
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
