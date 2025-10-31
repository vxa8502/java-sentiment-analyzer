package sentiment.models;

import org.slf4j.Logger;
import sentiment.data.Dataset;
import sentiment.PipelineState;

import javax.annotation.PreDestroy;
import java.util.List;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Template base class for classifier training with explicit train/predict workflow.
 *
 * ✅ UPDATED: Now uses unified PipelineState instead of ClassifierState
 *
 * DESIGN PATTERN: Template Method + Strategy
 * ==========================================
 * This class implements the INVARIANT parts of classifier training:
 * - Thread-safe state management (ReadWriteLock)
 * - State machine enforcement (UNINITIALIZED -> TRAINING -> READY)
 * - Training phase protection (write lock)
 * - Inference phase concurrency (read lock)
 *
 * Subclasses implement the VARIANT parts:
 * - Specific algorithm configuration (SVM, Naive Bayes, etc.)
 * - Classifier training logic (doTrain)
 * - Classification methods (classify, getClassificationProbabilities)
 * - Resource cleanup (doClearResources)
 *
 * Optional interfaces for subclasses:
 * - Implement ClassifierEvaluator if supporting evaluation
 * - Implement ClassifierPersistence if supporting save/load
 *
 * THREAD SAFETY GUARANTEE:
 * ========================
 * After train() completes successfully:
 * ✅ Multiple threads can call inference methods concurrently
 * ✅ No synchronization overhead during inference (ReadWriteLock.readLock)
 * ✅ State mutations only during training (ReadWriteLock.writeLock)
 *
 * USAGE PATTERN:
 * ==============
 * 1. Application startup: classifier.train(trainingData) - ONCE
 * 2. Request handling: classifier.classify(text) - MANY times, concurrently
 * 3. Application shutdown: classifier.cleanup() - automatic via @PreDestroy
 *
 * STATE MACHINE:
 * ==============
 * UNINITIALIZED --train()--> TRAINING --success--> READY
 *      ^                        |                    |
 *      |                     failure              reset()
 *      |                        |                    |
 *      +--------<------------ERROR---------<---------+
 *
 * @param <T> Type of training result (optional, can be Void)
 */
public abstract class ClassifierTrainingTemplate<T> implements SentimentClassifier {

    // Thread-safe state management
    protected final ReadWriteLock stateLock = new ReentrantReadWriteLock();
    protected volatile PipelineState classifierState = PipelineState.UNINITIALIZED;

    // Performance tracking
    protected long lastTrainingTimeMs = 0;

    // ==================== TEMPLATE METHOD: TRAINING PHASE ====================

    /**
     * TRAINING PHASE: Train the classifier on training data.
     *
     * This method is NOT thread-safe and should be called ONCE during initialization.
     * After successful completion, inference methods become thread-safe.
     *
     * TEMPLATE METHOD PATTERN:
     * 1. Validate input (common)
     * 2. Acquire write lock (common)
     * 3. Validate state machine (common)
     * 4. Transition to TRAINING (common)
     * 5. **Call doTrain() - SUBCLASS IMPLEMENTS THIS**
     * 6. Transition to READY (common)
     * 7. Release write lock (common)
     *
     * @param trainingData Training datasets
     * @throws IllegalStateException if already training or in error state
     * @throws IllegalArgumentException if trainingData is null/empty
     */
    @Override
    public final void train(List<Dataset> trainingData) throws Exception {
        validateTrainInput(trainingData);

        stateLock.writeLock().lock(); // ⚠️ EXCLUSIVE ACCESS
        try {
            validateStateBeforeTraining();

            getLogger().info("Starting TRAINING phase on {} samples. State: {} -> TRAINING",
                    trainingData.size(), classifierState);
            long startTime = System.currentTimeMillis();

            // ENFORCED transition: current -> TRAINING
            transitionToState(PipelineState.TRAINING);

            try {
                // ★ SUBCLASS HOOK: Implement specific classifier training logic
                T result = doTrain(trainingData);

                // ENFORCED transition: TRAINING -> READY (success)
                transitionToState(PipelineState.READY);

                this.lastTrainingTimeMs = System.currentTimeMillis() - startTime;
                getLogger().info("TRAINING completed successfully in {}ms. State: TRAINING -> READY. " +
                        "NOW THREAD-SAFE for concurrent inference.", lastTrainingTimeMs);

            } catch (Exception e) {
                // ENFORCED transition: TRAINING -> ERROR (failure)
                transitionToState(PipelineState.ERROR);
                getLogger().error("TRAINING failed. State: TRAINING -> ERROR", e);
                throw new Exception("Failed to train classifier: " + e.getMessage(), e);
            }

        } finally {
            stateLock.writeLock().unlock(); // ✅ Always release
        }
    }

    // ==================== TEMPLATE METHOD: INFERENCE PHASE ====================

    /**
     * Execute inference task with thread-safe read lock.
     *
     * USAGE IN SUBCLASS:
     * <pre>
     * public String classify(String text) {
     *     return executeInference(() -> {
     *         // Your classification logic here
     *         Instance instance = convertTextToInstance(text);
     *         double result = classifier.classifyInstance(instance);
     *         return mapToLabel(result);
     *     });
     * }
     * </pre>
     *
     * @param task Lambda containing inference logic
     * @return Result of inference task
     */
    protected final <R> R executeInference(InferenceTask<R> task) throws Exception {
        stateLock.readLock().lock(); // ⚠️ CONCURRENT READS ALLOWED
        try {
            validateReadyForInference();
            return task.execute();

        } catch (IllegalStateException e) {
            throw e; // Re-throw state errors
        } catch (Exception e) {
            getLogger().error("Inference failed in state {}", classifierState, e);
            throw new Exception("Failed to execute inference: " + e.getMessage(), e);
        } finally {
            stateLock.readLock().unlock(); // ✅ Always release
        }
    }

    /**
     * Functional interface for inference tasks.
     * Allows lambdas that throw checked exceptions.
     */
    @FunctionalInterface
    protected interface InferenceTask<R> {
        R execute() throws Exception;
    }

    // ==================== ABSTRACT METHODS FOR SUBCLASSES ====================

    /**
     * Subclass implements specific classifier training logic.
     *
     * CALLED WITHIN: train() method, protected by write lock
     * THREAD SAFETY: NOT required - exclusive access guaranteed
     *
     * IMPLEMENTATION GUIDELINES:
     * 1. Configure algorithm parameters
     * 2. Build/train the classifier
     * 3. Store trained model state
     * 4. Generate training statistics
     *
     * @param trainingData Training datasets
     * @return Training result (can be null if no result needed)
     * @throws Exception if training fails (will transition to ERROR state)
     */
    protected abstract T doTrain(List<Dataset> trainingData) throws Exception;

    /**
     * Subclass implements resource cleanup logic.
     *
     * CALLED WITHIN: reset() method, protected by write lock
     * THREAD SAFETY: NOT required - exclusive access guaranteed
     *
     * IMPLEMENTATION GUIDELINES:
     * 1. Set classifier references to null
     * 2. Clear cached data structures
     * 3. Release any held resources
     */
    protected abstract void doClearResources();

    /**
     * Subclass provides its logger instance.
     * Required for consistent logging across template and subclass.
     */
    protected abstract Logger getLogger();

    // ==================== STATE MANAGEMENT ====================

    /**
     * ENFORCED state transition with validation.
     * All state changes MUST go through this method.
     */
    private void transitionToState(PipelineState newState) {
        PipelineState oldState = classifierState;
        oldState.validateTransition(newState); // Throws if invalid
        classifierState = newState;
        getLogger().debug("State transition: {} -> {}", oldState, newState);
    }

    /**
     * Validate classifier is ready for inference.
     *
     * @throws IllegalStateException if not in READY state
     */
    protected void validateReadyForInference() {
        if (classifierState != PipelineState.READY) {
            throw new IllegalStateException(
                    "Cannot perform inference - not trained. Current state: " + classifierState +
                            ". Call train() with training data first.");
        }
    }

    /**
     * Validate state allows training to start.
     */
    private void validateStateBeforeTraining() {
        if (classifierState == PipelineState.TRAINING) {
            throw new IllegalStateException(
                    "Training already in progress. Current state: " + classifierState);
        }

        if (classifierState == PipelineState.READY) {
            getLogger().warn("Classifier already trained. Call reset() before retraining.");
            throw new IllegalStateException(
                    "Already trained. Current state: " + classifierState + ". Call reset() first.");
        }
    }

    /**
     * Validate train() input parameters.
     */
    private void validateTrainInput(List<Dataset> trainingData) {
        if (trainingData == null || trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
    }

    // ==================== RESET AND CLEANUP ====================

    /**
     * Reset classifier to UNINITIALIZED state.
     *
     * NOT thread-safe - should only be called during:
     * - Application shutdown
     * - Controlled retraining scenarios
     * - Error recovery
     */
    public void reset() {
        stateLock.writeLock().lock(); // ⚠️ EXCLUSIVE ACCESS
        try {
            getLogger().info("Resetting classifier from state: {}", classifierState);

            // Clear subclass resources
            doClearResources();

            // Reset timing
            lastTrainingTimeMs = 0;

            // Transition to UNINITIALIZED
            if (classifierState != PipelineState.UNINITIALIZED) {
                if (classifierState == PipelineState.TRAINING) {
                    transitionToState(PipelineState.ERROR);
                }
                transitionToState(PipelineState.UNINITIALIZED);
            }

            getLogger().info("Reset complete. New state: {}", classifierState);

        } finally {
            stateLock.writeLock().unlock(); // ✅ Always release
        }
    }

    /**
     * Automatic cleanup on bean destruction.
     * Spring calls this via @PreDestroy.
     */
    @PreDestroy
    public void cleanup() {
        getLogger().info("Cleaning up classifier resources");
        reset();
    }

    // ==================== READ-ONLY STATE ACCESS ====================

    /**
     * Get current classifier state.
     * Thread-safe: volatile read
     */
    public PipelineState getClassifierState() {
        return classifierState;
    }

    /**
     * Check if classifier is ready for inference.
     * Thread-safe: volatile read
     */
    @Override
    public boolean isTrained() {
        return classifierState == PipelineState.READY;
    }

    /**
     * Get training time in milliseconds.
     * Thread-safe: volatile read
     */
    public long getLastTrainingTimeMs() {
        return lastTrainingTimeMs;
    }

    /**
     * Get comprehensive diagnostics string.
     */
    public String getDiagnostics() {
        stateLock.readLock().lock();
        try {
            StringBuilder diag = new StringBuilder();
            diag.append("=== Classifier Training Template Diagnostics ===\n");
            diag.append(String.format("Current state: %s\n", classifierState));
            diag.append(String.format("State description: %s\n", classifierState.getDescription()));
            diag.append(String.format("Is ready: %s\n", classifierState.isReady()));
            diag.append(String.format("Can start training: %s\n", classifierState.canStartTraining()));
            diag.append(String.format("Is error: %s\n", classifierState.isError()));
            diag.append(String.format("Last training time: %d ms\n", lastTrainingTimeMs));

            // Subclass diagnostics
            String subclassDiag = getSubclassDiagnostics();
            if (subclassDiag != null && !subclassDiag.isEmpty()) {
                diag.append("\n").append(subclassDiag);
            }

            return diag.toString();
        } finally {
            stateLock.readLock().unlock();
        }
    }

    /**
     * Subclass can override to provide additional diagnostics.
     */
    protected String getSubclassDiagnostics() {
        return "";
    }
}