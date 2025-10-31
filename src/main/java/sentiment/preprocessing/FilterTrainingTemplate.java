package sentiment.preprocessing;

import org.slf4j.Logger;
import sentiment.data.Dataset;
import sentiment.PipelineState;

import jakarta.annotation.PreDestroy;
import java.util.List;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Template base class for Weka filter training with explicit fit/transform workflow.
 * ✅ UPDATED: Now uses unified PipelineState instead of FilterState
 *
 * DESIGN PATTERN: Template Method + Strategy
 * ==========================================
 * This class implements the INVARIANT parts of filter training:
 * - Thread-safe state management (ReadWriteLock)
 * - State machine enforcement (UNINITIALIZED -> TRAINING -> READY)
 * - Training phase protection (write lock)
 * - Inference phase concurrency (read lock)
 *
 * Subclasses implement the VARIANT parts:
 * - Specific filter configuration (StringToWordVector, Normalize, etc.)
 * - Filter training logic (doFit)
 * - Resource cleanup (doClearResources)
 *
 * THREAD SAFETY GUARANTEE:
 * ========================
 * After fit() completes successfully:
 * ✅ Multiple threads can call inference methods concurrently
 * ✅ No synchronization overhead during inference (ReadWriteLock.readLock)
 * ✅ State mutations only during training (ReadWriteLock.writeLock)
 *
 * USAGE PATTERN:
 * ==============
 * 1. Application startup: extractor.fit(trainingData) - ONCE
 * 2. Request handling: extractor.transform(text) - MANY times, concurrently
 * 3. Application shutdown: extractor.cleanup() - automatic via @PreDestroy
 *
 * STATE MACHINE:
 * ==============
 * UNINITIALIZED --fit()--> TRAINING --success--> READY
 *      ^                      |                    |
 *      |                   failure              reset()
 *      |                      |                    |
 *      +--------<----------ERROR---------<---------+
 *
 * @param <T> Type of training data result (e.g., Instances)
 */
public abstract class FilterTrainingTemplate<T> {

    // Thread-safe state management
    // Using lazy initialization to handle Spring CGLIB proxy issues
    private ReadWriteLock stateLock;
    protected volatile PipelineState filterState;

    // Performance tracking
    protected long lastTrainingTimeMs;

    /**
     * Get or create the state lock (lazy initialization for CGLIB proxy compatibility).
     * This approach ensures the lock is always available, even if Spring proxies
     * bypass normal initialization.
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
     * Ensure pipeline state is initialized (lazy initialization for CGLIB proxy compatibility).
     */
    private void ensureStateInitialized() {
        if (filterState == null) {
            filterState = PipelineState.UNINITIALIZED;
        }
    }

    // ==================== TEMPLATE METHOD: TRAINING PHASE ====================

    /**
     * TRAINING PHASE: Fit the filter on training data.
     *
     * This method is NOT thread-safe and should be called ONCE during initialization.
     * After successful completion, inference methods become thread-safe.
     *
     * TEMPLATE METHOD PATTERN:
     * 1. Validate input (common)
     * 2. Acquire write lock (common)
     * 3. Validate state machine (common)
     * 4. Transition to TRAINING (common)
     * 5. **Call doFit() - SUBCLASS IMPLEMENTS THIS**
     * 6. Transition to READY (common)
     * 7. Release write lock (common)
     *
     * @param datasets Training data
     * @return Training result (e.g., transformed Instances)
     * @throws IllegalStateException if already training or in error state
     * @throws IllegalArgumentException if datasets is null/empty
     */
    public final T fit(List<Dataset> datasets) {
        validateFitInput(datasets);
        ensureStateInitialized(); // Ensure state is initialized before locking

        getStateLock().writeLock().lock(); // ⚠️ EXCLUSIVE ACCESS
        try {
            validateStateBeforeTraining();

            getLogger().info("Starting TRAINING phase on {} samples. State: {} -> TRAINING",
                    datasets.size(), filterState);
            long startTime = System.currentTimeMillis();

            // ENFORCED transition: current -> TRAINING
            transitionToState(PipelineState.TRAINING);

            try {
                // ★ SUBCLASS HOOK: Implement specific filter training logic
                T result = doFit(datasets);

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
                throw new RuntimeException("Failed to train filter: " + e.getMessage(), e);
            }

        } finally {
            getStateLock().writeLock().unlock(); // ✅ Always release
        }
    }

    // ==================== TEMPLATE METHOD: INFERENCE PHASE ====================

    /**
     * Execute inference task with thread-safe read lock.
     *
     * USAGE IN SUBCLASS:
     * <pre>
     * public Instance transform(String text) {
     *     return executeInference(() -> {
     *         // Your transformation logic here
     *         Instances result = Filter.useFilter(input, trainedFilter);
     *         return result.instance(0);
     *     });
     * }
     * </pre>
     *
     * @param task Lambda containing inference logic
     * @return Result of inference task
     */
    protected final <R> R executeInference(InferenceTask<R> task) {
        getStateLock().readLock().lock(); // ⚠️ CONCURRENT READS ALLOWED
        try {
            validateReadyForInference();
            return task.execute();

        } catch (IllegalStateException e) {
            throw e; // Re-throw state errors
        } catch (Exception e) {
            getLogger().error("Inference failed in state {}", filterState, e);
            throw new RuntimeException("Failed to execute inference: " + e.getMessage(), e);
        } finally {
            getStateLock().readLock().unlock(); // ✅ Always release
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
     * Subclass implements specific filter training logic.
     *
     * CALLED WITHIN: fit() method, protected by write lock
     * THREAD SAFETY: NOT required - exclusive access guaranteed
     *
     * IMPLEMENTATION GUIDELINES:
     * 1. Train your filters (StringToWordVector, Normalize, etc.)
     * 2. Apply transformations to training data
     * 3. Store trained filters in subclass fields
     * 4. Generate statistics/metadata
     * 5. Return training result
     *
     * @param datasets Training data
     * @return Training result
     * @throws Exception if training fails (will transition to ERROR state)
     */
    protected abstract T doFit(List<Dataset> datasets) throws Exception;

    /**
     * Subclass implements resource cleanup logic.
     *
     * CALLED WITHIN: reset() method, protected by write lock
     * THREAD SAFETY: NOT required - exclusive access guaranteed
     *
     * IMPLEMENTATION GUIDELINES:
     * 1. Set filter references to null
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
        PipelineState oldState = filterState;
        oldState.validateTransition(newState); // Throws if invalid
        filterState = newState;
        getLogger().debug("State transition: {} -> {}", oldState, newState);
    }

    /**
     * Validate filter is ready for inference.
     *
     * @throws IllegalStateException if not in READY state
     */
    protected void validateReadyForInference() {
        if (filterState != PipelineState.READY) {
            throw new IllegalStateException(
                    "Cannot perform inference - not trained. Current state: " + filterState +
                            ". Call fit() with training data first.");
        }
    }

    /**
     * Validate state allows training to start.
     */
    private void validateStateBeforeTraining() {
        if (filterState == PipelineState.TRAINING) {
            throw new IllegalStateException(
                    "Training already in progress. Current state: " + filterState);
        }

        if (filterState == PipelineState.READY) {
            getLogger().warn("Filter already trained. Call reset() before retraining.");
            throw new IllegalStateException(
                    "Already trained. Current state: " + filterState + ". Call reset() first.");
        }
    }

    /**
     * Validate fit() input parameters.
     */
    private void validateFitInput(List<Dataset> datasets) {
        if (datasets == null || datasets.isEmpty()) {
            throw new IllegalArgumentException("Training data cannot be null or empty");
        }
    }

    // ==================== RESET AND CLEANUP ====================

    /**
     * Reset filter to UNINITIALIZED state.
     *
     * NOT thread-safe - should only be called during:
     * - Application shutdown
     * - Controlled retraining scenarios
     * - Error recovery
     */
    public void reset() {
        stateLock.writeLock().lock(); // ⚠️ EXCLUSIVE ACCESS
        try {
            getLogger().info("Resetting filter from state: {}", filterState);

            // Clear subclass resources
            doClearResources();

            // Reset timing
            lastTrainingTimeMs = 0;

            // Transition to UNINITIALIZED (valid from any state for reset)
            if (filterState != PipelineState.UNINITIALIZED) {
                if (filterState == PipelineState.TRAINING) {
                    transitionToState(PipelineState.ERROR);
                }
                transitionToState(PipelineState.UNINITIALIZED);
            }

            getLogger().info("Reset complete. New state: {}", filterState);

        } finally {
            getStateLock().writeLock().unlock(); // ✅ Always release
        }
    }

    /**
     * Automatic cleanup on bean destruction.
     * Spring calls this via @PreDestroy.
     */
    @PreDestroy
    public void cleanup() {
        getLogger().info("Cleaning up filter resources");
        reset();
    }

    // ==================== READ-ONLY STATE ACCESS ====================

    /**
     * Get current filter state.
     * Thread-safe: volatile read
     */
    public PipelineState getFilterState() {
        return filterState;
    }

    /**
     * Check if filter is ready for inference.
     * Thread-safe: volatile read
     */
    public boolean isReady() {
        return filterState == PipelineState.READY;
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
        getStateLock().readLock().lock();
        try {
            StringBuilder diag = new StringBuilder();
            diag.append("=== Filter Training Template Diagnostics ===\n");
            diag.append(String.format("Current state: %s\n", filterState));
            diag.append(String.format("State description: %s\n", filterState.getDescription()));
            diag.append(String.format("Is ready: %s\n", filterState.isReady()));
            diag.append(String.format("Can start training: %s\n", filterState.canStartTraining()));
            diag.append(String.format("Is error: %s\n", filterState.isError()));
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
     * Subclass can override to provide additional diagnostics.
     */
    protected String getSubclassDiagnostics() {
        return "";
    }
}