package sentiment;

import org.slf4j.Logger;
import sentiment.data.Dataset;

import javax.annotation.PreDestroy;
import java.util.List;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Abstract base template for all trainable components (classifiers, filters, etc.).
 * <p>
 * Provides unified thread-safe state management, lifecycle control, and training workflow
 * using the Template Method pattern. Eliminates ~280 lines of duplicate code between
 * ClassifierTrainingTemplate and FilterTrainingTemplate.
 *
 * <h3>Design Pattern</h3>
 * <p>This class implements the invariant parts of training workflows:
 * <ul>
 *   <li>Thread-safe state management using {@code ReadWriteLock}</li>
 *   <li>State machine enforcement (UNINITIALIZED → TRAINING → READY)</li>
 *   <li>Training phase protection (write lock)</li>
 *   <li>Inference phase concurrency (read lock)</li>
 *   <li>Lazy initialization for Spring CGLIB proxy compatibility</li>
 * </ul>
 *
 * <p>Subclasses implement the variant parts:
 * <ul>
 *   <li>Specific training logic ({@link #doTrain(List)})</li>
 *   <li>Resource cleanup ({@link #doClearResources()})</li>
 *   <li>Component-specific diagnostics ({@link #getSubclassDiagnostics()})</li>
 *   <li>Logger instance ({@link #getLogger()})</li>
 * </ul>
 *
 * <h3>State Machine</h3>
 * <pre>
 * UNINITIALIZED --train()--> TRAINING --success--> READY
 *      ^                        |                    |
 *      |                     failure              reset()
 *      |                        |                    |
 *      +--------&lt;----------ERROR---------&lt;---------+
 * </pre>
 *
 * @param <T> type of training result (e.g., Instances, Void)
 */
public abstract class TrainingTemplate<T> {

    // Thread-safe state management
    // Using lazy initialization to handle Spring CGLIB proxy issues
    private volatile ReadWriteLock stateLock;
    protected volatile PipelineState pipelineState;

    // Performance tracking
    protected volatile long lastTrainingTimeMs;

    /**
     * Gets or creates the state lock using lazy initialization for CGLIB proxy compatibility.
     * <p>
     * Ensures the lock is always available, even when Spring proxies bypass normal initialization.
     *
     * @return the state lock instance
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

    /**
     * Ensures pipeline state is initialized using lazy initialization for CGLIB proxy compatibility.
     */
    private void ensureStateInitialized() {
        if (pipelineState == null) {
            pipelineState = PipelineState.UNINITIALIZED;
        }
    }

    // TEMPLATE METHOD: TRAINING PHASE

    /**
     * Trains the component on the provided training data.
     * <p>
     * NOT DESIGNED for concurrent training calls - call ONCE during initialization.
     * After successful completion, inference methods become thread-safe for concurrent access.
     *
     * @param trainingData the training datasets
     * @return training result (e.g., transformed Instances)
     * @throws IllegalStateException if already training or in error state
     * @throws IllegalArgumentException if trainingData is null or empty
     * @throws Exception if training fails
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
                // ★ SUBCLASS HOOK: Implement specific training logic
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
     * Executes an inference task with thread-safe read lock protection.
     * <p>
     * This method allows multiple concurrent reads while ensuring no state modifications
     * occur during inference. The read lock permits concurrent execution by multiple threads.
     *
     * <p><b>Example usage in subclass:</b>
     * <pre>{@code
     * public Instance transform(String text) {
     *     return executeInference(() -> {
     *         // Your transformation logic here
     *         return filter.transform(input);
     *     });
     * }
     * }</pre>
     *
     * @param <R> the return type of the inference task
     * @param task lambda containing inference logic
     * @return result of the inference task
     * @throws IllegalStateException if not in READY state
     * @throws Exception if inference execution fails
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

    /**
     * Functional interface for inference tasks that allows lambdas to throw checked exceptions.
     *
     * @param <R> the return type of the task
     */
    @FunctionalInterface
    protected interface InferenceTask<R> {
        /**
         * Executes the inference task.
         *
         * @return the result of the inference
         * @throws Exception if the inference fails
         */
        R execute() throws Exception;
    }

    // ABSTRACT METHODS FOR SUBCLASSES

    /**
     * Implements component-specific training logic.
     * <p>
     * Called by {@link #trainInternal(List)} within write lock. Thread safety is not required
     * as exclusive access is guaranteed.
     *
     * @param trainingData the training datasets
     * @return training result (can be null if no result is needed)
     * @throws Exception if training fails (will transition to ERROR state)
     */
    protected abstract T doTrain(List<Dataset> trainingData) throws Exception;

    /**
     * Implements component-specific resource cleanup logic.
     * <p>
     * Called by {@link #reset()} within write lock. Thread safety is not required
     * as exclusive access is guaranteed.
     */
    protected abstract void doClearResources();

    /**
     * Provides the logger instance for this component.
     * <p>
     * Required for consistent logging across template and subclass implementations.
     *
     * @return the logger instance
     */
    protected abstract Logger getLogger();

    /**
     * Returns the component type name for logging and error messages.
     * <p>
     * Examples: "classifier", "filter", "preprocessor"
     *
     * @return component type name (lowercase, singular)
     */
    protected abstract String getComponentType();

    // STATE MANAGEMENT

    /**
     * Performs state transition with validation.
     * <p>
     * All state changes must go through this method to ensure proper state machine enforcement.
     *
     * @param newState the target state
     * @throws IllegalStateException if transition is invalid
     */
    private void transitionToState(PipelineState newState) {
        PipelineState oldState = pipelineState;
        oldState.validateTransition(newState); // Throws if invalid
        pipelineState = newState;
        getLogger().debug("State transition: {} -> {}", oldState, newState);
    }

    /**
     * Validates that the component is ready for inference.
     *
     * @throws IllegalStateException if not in READY state
     */
    protected void validateReadyForInference() {
        if (pipelineState != PipelineState.READY) {
            throw new IllegalStateException(
                    "Cannot perform inference - not trained. Current state: " + pipelineState +
                            ". Call train() with training data first.");
        }
    }

    /**
     * Validates that the current state allows training to start.
     *
     * @throws IllegalStateException if already training or already trained
     */
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

    /**
     * Validates training input parameters.
     *
     * @param trainingData the training data to validate
     * @throws IllegalArgumentException if trainingData is null or empty
     */
    private void validateTrainingInput(List<Dataset> trainingData) {
        if (trainingData == null || trainingData.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
    }

    // RESET AND CLEANUP

    /**
     * Resets the component to UNINITIALIZED state.
     * <p>
     * This method is NOT thread-safe and should only be called during:
     * <ul>
     *   <li>Application shutdown</li>
     *   <li>Controlled retraining scenarios</li>
     *   <li>Error recovery</li>
     * </ul>
     */
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

    /**
     * Performs automatic cleanup on bean destruction.
     * <p>
     * Spring calls this method automatically via {@code @PreDestroy} annotation.
     */
    @PreDestroy
    public void cleanup() {
        getLogger().info("Cleaning up {} resources", getComponentType());
        reset();
    }

    // READ-ONLY STATE ACCESS

    /**
     * Gets the current pipeline state.
     * <p>
     * Thread-safe via volatile read.
     *
     * @return the current pipeline state
     */
    public PipelineState getState() {
        return pipelineState;
    }

    /**
     * Checks if the component is ready for inference.
     * <p>
     * Thread-safe via volatile read.
     *
     * @return {@code true} if in READY state, {@code false} otherwise
     */
    public boolean isReady() {
        return pipelineState == PipelineState.READY;
    }

    /**
     * Gets the last training time in milliseconds.
     * <p>
     * Thread-safe via volatile read.
     *
     * @return training duration in milliseconds
     */
    public long getLastTrainingTimeMs() {
        return lastTrainingTimeMs;
    }

    // DIAGNOSTICS

    /**
     * Gets comprehensive diagnostics information.
     *
     * @return formatted diagnostics string
     */
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

    /**
     * Provides additional subclass-specific diagnostics.
     * <p>
     * Subclasses can override this method to include component-specific diagnostic information.
     *
     * @return subclass diagnostic information, or empty string if none
     */
    protected String getSubclassDiagnostics() {
        return "";
    }

    // PERSISTENCE SUPPORT

    /**
     * Acquires the write lock for external operations (e.g., model persistence).
     * <p>
     * Public to allow persistence utilities and subclasses access.
     * <b>IMPORTANT:</b> Caller MUST release the lock via {@link #releaseWriteLock()} in a finally block.
     */
    public void acquireWriteLock() {
        getStateLock().writeLock().lock();
    }

    /**
     * Releases the write lock after external operations.
     * <p>
     * Public to allow persistence utilities and subclasses access.
     * Should be called in a finally block after {@link #acquireWriteLock()}.
     */
    public void releaseWriteLock() {
        getStateLock().writeLock().unlock();
    }

    /**
     * Sets the pipeline state directly.
     * <p>
     * Public to allow persistence utilities and subclasses to set state during model loading.
     * <b>WARNING:</b> This bypasses state machine validation - use with extreme caution!
     * Only for use by persistence utilities during model restoration.
     *
     * @param state the state to set
     */
    public void setState(PipelineState state) {
        this.pipelineState = state;
    }

    //  UTILITY METHODS

    /**
     * Capitalizes the first letter of a string.
     *
     * @param str string to capitalize
     * @return capitalized string
     */
    private String capitalize(String str) {
        if (str == null || str.isEmpty()) {
            return str;
        }
        return Character.toUpperCase(str.charAt(0)) + str.substring(1);
    }
}
