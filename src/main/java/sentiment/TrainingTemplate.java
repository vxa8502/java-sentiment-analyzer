package sentiment;

import org.slf4j.Logger;
import sentiment.data.Dataset;

import jakarta.annotation.PreDestroy;
import java.util.List;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Abstract template for trainable components with thread-safe state management.
 * Implements Template Method pattern with state machine: UNINITIALIZED → TRAINING → READY.
 * Uses ReadWriteLock for training exclusivity and concurrent inference.
 *
 * @param <T> type of training result
 */
public abstract class TrainingTemplate<T> {

    // Thread-safe state management
    // Using lazy initialization to handle Spring CGLIB proxy issues
    private volatile ReadWriteLock stateLock;
    protected volatile PipelineState pipelineState;

    // Performance tracking
    protected volatile long lastTrainingTimeMs;

    /**
     * Gets or creates state lock with lazy initialization for CGLIB proxy compatibility.
     */
    private ReadWriteLock getStateLock() {
        if (stateLock == null) {
            synchronized (this) {
                if (stateLock == null) {
                    stateLock = new ReentrantReadWriteLock();
                }
            }
        }
        return stateLock;
    }

    /** Ensures pipeline state is initialized (lazy init for CGLIB proxies). */
    private void ensureStateInitialized() {
        if (pipelineState == null) {
            pipelineState = PipelineState.UNINITIALIZED;
        }
    }

    // TEMPLATE METHOD: TRAINING PHASE

    /**
     * Trains the component on provided data. NOT thread-safe - call once during initialization.
     *
     * @param trainingData training datasets
     * @return training result
     * @throws IllegalStateException if already training
     * @throws IllegalArgumentException if trainingData null/empty
     */
    protected final T trainInternal(List<Dataset> trainingData) throws Exception {
        validateTrainingInput(trainingData);
        ensureStateInitialized();

        getStateLock().writeLock().lock(); // EXCLUSIVE ACCESS
        try {
            validateStateBeforeTraining();

            getLogger().info("Starting TRAINING phase on {} samples. State: {} -> TRAINING",
                    trainingData.size(), pipelineState);
            long startTime = System.currentTimeMillis();

            // ENFORCED transition: current -> TRAINING
            transitionToState(PipelineState.TRAINING);

            try {
                //  SUBCLASS HOOK: Implement specific training logic
                T result = doTrain(trainingData);

                // ENFORCED transition: TRAINING -> READY (success)
                transitionToState(PipelineState.READY);

                this.lastTrainingTimeMs = System.currentTimeMillis() - startTime;
                getLogger().info("TRAINING completed successfully in {}ms. State: TRAINING -> READY. " +
                        "NOW THREAD-SAFE for concurrent inference.", lastTrainingTimeMs);

                return result;

            } catch (Exception e) {
                // ENFORCED transition: TRAINING -> ERROR (failure)
                transitionToState(PipelineState.ERROR);
                getLogger().error("TRAINING failed. State: TRAINING -> ERROR", e);
                throw new Exception("Failed to train " + getComponentType() + ": " + e.getMessage(), e);
            }

        } finally {
            getStateLock().writeLock().unlock(); // Always release
        }
    }

    // TEMPLATE METHOD: INFERENCE PHASE

    /**
     * Executes inference task with read lock (allows concurrent execution).
     *
     * @param <R> return type
     * @param task inference logic
     * @return inference result
     * @throws IllegalStateException if not READY
     */
    protected final <R> R executeInference(InferenceTask<R> task) throws Exception {
        getStateLock().readLock().lock(); // CONCURRENT READS ALLOWED
        try {
            validateReadyForInference();
            return task.execute();

        } catch (IllegalStateException e) {
            throw e; // Re-throw state errors
        } catch (Exception e) {
            getLogger().error("Inference failed in state {}", pipelineState, e);
            throw new Exception("Failed to execute inference: " + e.getMessage(), e);
        } finally {
            getStateLock().readLock().unlock(); // Always release
        }
    }

    /** Functional interface for inference tasks with exception support. */
    @FunctionalInterface
    protected interface InferenceTask<R> {
        R execute() throws Exception;
    }

    // ABSTRACT METHODS FOR SUBCLASSES

    /**
     * Implements component-specific training logic (called within write lock).
     *
     * @param trainingData training datasets
     * @return training result (nullable)
     */
    protected abstract T doTrain(List<Dataset> trainingData) throws Exception;

    /** Implements component-specific resource cleanup (called within write lock). */
    protected abstract void doClearResources();

    /** Returns logger for this component. */
    protected abstract Logger getLogger();

    /** Returns component type name (e.g., "classifier", "filter"). */
    protected abstract String getComponentType();

    // STATE MANAGEMENT

    /** Performs validated state transition. */
    private void transitionToState(PipelineState newState) {
        PipelineState oldState = pipelineState;
        oldState.validateTransition(newState); // Throws if invalid
        pipelineState = newState;
        getLogger().debug("State transition: {} -> {}", oldState, newState);
    }

    /** Validates component is in READY state. */
    protected void validateReadyForInference() {
        if (pipelineState != PipelineState.READY) {
            throw new IllegalStateException(
                    "Cannot perform inference - not trained. Current state: " + pipelineState +
                            ". Call train() with training data first.");
        }
    }

    /** Validates state allows training to start. */
    private void validateStateBeforeTraining() {
        if (pipelineState == PipelineState.TRAINING) {
            throw new IllegalStateException(
                    "Training already in progress. Current state: " + pipelineState);
        }

        if (pipelineState == PipelineState.READY) {
            getLogger().warn("{} already trained. Call reset() before retraining.",
                    capitalize(getComponentType()));
            throw new IllegalStateException(
                    "Already trained. Current state: " + pipelineState + ". Call reset() first.");
        }
    }

    /** Validates training input is non-null and non-empty. */
    private void validateTrainingInput(List<Dataset> trainingData) {
        if (trainingData == null || trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
    }

    // RESET AND CLEANUP

    /** Resets component to UNINITIALIZED state (NOT thread-safe). */
    public void reset() {
        getStateLock().writeLock().lock(); // EXCLUSIVE ACCESS
        try {
            getLogger().info("Resetting {} from state: {}", getComponentType(), pipelineState);

            // Clear subclass resources
            doClearResources();

            // Reset timing
            lastTrainingTimeMs = 0;

            // Transition to UNINITIALIZED
            if (pipelineState != PipelineState.UNINITIALIZED) {
                if (pipelineState == PipelineState.TRAINING) {
                    transitionToState(PipelineState.ERROR);
                }
                transitionToState(PipelineState.UNINITIALIZED);
            }

            getLogger().info("Reset complete. New state: {}", pipelineState);

        } finally {
            getStateLock().writeLock().unlock(); // Always release
        }
    }

    /** Cleanup hook called by Spring on bean destruction. */
    @PreDestroy
    public void cleanup() {
        getLogger().info("Cleaning up {} resources", getComponentType());
        reset();
    }

    // READ-ONLY STATE ACCESS

    /** Gets current pipeline state (thread-safe). */
    public PipelineState getState() {
        return pipelineState;
    }

    /** Checks if component is in READY state (thread-safe). */
    public boolean isReady() {
        return pipelineState == PipelineState.READY;
    }

    /** Gets last training time in milliseconds (thread-safe). */
    public long getLastTrainingTimeMs() {
        return lastTrainingTimeMs;
    }

    // DIAGNOSTICS

    /** Gets comprehensive diagnostics information. */
    @SuppressWarnings("unused")
    public String getDiagnostics() {
        getStateLock().readLock().lock();
        try {
            StringBuilder diag = new StringBuilder();
            diag.append("=== ").append(capitalize(getComponentType()))
                    .append(" Training Template Diagnostics ===\n");
            diag.append(String.format("Current state: %s\n", pipelineState));
            diag.append(String.format("State description: %s\n", pipelineState.getDescription()));
            diag.append(String.format("Is ready: %s\n", pipelineState.isReady()));
            diag.append(String.format("Can start training: %s\n", pipelineState.canStartTraining()));
            diag.append(String.format("Is error: %s\n", pipelineState.isError()));
            diag.append(String.format("Last training time: %d ms\n", lastTrainingTimeMs));

            // Subclass diagnostics
            String subclassDiag = getSubclassDiagnostics();
            if (subclassDiag != null && !subclassDiag.isEmpty()) {
                diag.append("\n").append(subclassDiag);
            }

            return diag.toString();
        } finally {
            getStateLock().readLock().unlock();
        }
    }

    /** Returns subclass-specific diagnostics (override for custom info). */
    protected String getSubclassDiagnostics() {
        return "";
    }

    // PERSISTENCE SUPPORT

    /** Acquires write lock for external operations. MUST release via {@link #releaseWriteLock()}. */
    public void acquireWriteLock() {
        getStateLock().writeLock().lock();
    }

    /** Releases write lock after external operations (call in finally block). */
    public void releaseWriteLock() {
        getStateLock().writeLock().unlock();
    }

    /** Sets state directly (bypasses validation - use only for model loading). */
    public void setState(PipelineState state) {
        this.pipelineState = state;
    }

    // UTILITY METHODS

    /** Capitalizes first letter of string. */
    private String capitalize(String str) {
        if (str == null || str.isEmpty()) {
            return str;
        }
        return Character.toUpperCase(str.charAt(0)) + str.substring(1);
    }
}
