package sentiment.preprocessing;

import org.slf4j.Logger;
import sentiment.data.Dataset;
import sentiment.TrainingTemplate;

import java.util.List;

/**
 * Template base class for Weka filter training with explicit fit/transform workflow.
 * <p>
 * Extends {@link TrainingTemplate} to provide unified state management and lifecycle control.
 * This class adds filter-specific naming conventions (fit/transform) on top of the base
 * training template pattern.
 *
 * <h3>Design Pattern</h3>
 * <p>Inherits from TrainingTemplate:
 * <ul>
 *   <li>Thread-safe state management using {@code ReadWriteLock}</li>
 *   <li>State machine enforcement (UNINITIALIZED → TRAINING → READY)</li>
 *   <li>Training phase protection (write lock)</li>
 *   <li>Inference phase concurrency (read lock)</li>
 * </ul>
 *
 * <p>Subclasses implement:
 * <ul>
 *   <li>Specific filter configuration (StringToWordVector, Normalize, etc.)</li>
 *   <li>Filter training logic ({@link #doFit(List)})</li>
 *   <li>Resource cleanup ({@link #doClearResources()})</li>
 * </ul>
 *
 * <h3>Thread Safety</h3>
 * <p>After {@link #fit(List)} completes successfully:
 * <ul>
 *   <li>Multiple threads can call inference methods concurrently</li>
 *   <li>No synchronization overhead during inference (read lock)</li>
 *   <li>State mutations only during training (write lock)</li>
 * </ul>
 *
 * <h3>Usage Pattern</h3>
 * <ol>
 *   <li>Application startup: {@code extractor.fit(trainingData)} (once)</li>
 *   <li>Request handling: {@code extractor.transform(text)} (many times, concurrently)</li>
 *   <li>Application shutdown: {@code extractor.cleanup()} (automatic via {@code @PreDestroy})</li>
 * </ol>
 *
 * <h3>State Machine</h3>
 * <pre>
 * UNINITIALIZED --fit()--> TRAINING --success--> READY
 *      ^                      |                    |
 *      |                   failure              reset()
 *      |                      |                    |
 *      +--------&lt;----------ERROR---------&lt;---------+
 * </pre>
 *
 * @param <T> type of training data result (e.g., Instances)
 */
public abstract class FilterTrainingTemplate<T> extends TrainingTemplate<T> {

    /**
     * Fits the filter on training data.
     * <p>
     * This is a convenience wrapper around {@link #train(List)} that uses filter-specific
     * naming conventions (fit vs train). Delegates to the base template method pattern.
     *
     * @param datasets training data
     * @return training result (e.g., transformed Instances)
     * @throws IllegalStateException if already training or in error state
     * @throws IllegalArgumentException if datasets is null or empty
     * @throws RuntimeException if training fails
     */
    public final T fit(List<Dataset> datasets) {
        try {
            return trainInternal(datasets);
        } catch (Exception e) {
            // Convert checked Exception to RuntimeException for filter API compatibility
            if (e instanceof RuntimeException) {
                throw (RuntimeException) e;
            }
            throw new RuntimeException(e.getMessage(), e);
        }
    }

    /**
     * Implements filter-specific training logic.
     * <p>
     * Called within {@link #fit(List)} method, protected by write lock.
     * Thread safety is not required as exclusive access is guaranteed.
     *
     * <p><b>Implementation guidelines:</b>
     * <ol>
     *   <li>Train filters (StringToWordVector, Normalize, etc.)</li>
     *   <li>Apply transformations to training data</li>
     *   <li>Store trained filters in subclass fields</li>
     *   <li>Generate statistics and metadata</li>
     *   <li>Return training result</li>
     * </ol>
     *
     * @param datasets training data
     * @return training result
     * @throws Exception if training fails (transitions to ERROR state)
     */
    protected abstract T doFit(List<Dataset> datasets) throws Exception;

    /**
     * Delegates to {@link #doFit(List)} for filter-specific API.
     * <p>
     * This method is called by the base class {@link TrainingTemplate#train(List)}.
     * It delegates to the filter-specific {@link #doFit(List)} method to maintain
     * API naming conventions.
     *
     * @param trainingData the training datasets
     * @return training result from doFit
     * @throws Exception if training fails
     */
    @Override
    protected final T doTrain(List<Dataset> trainingData) throws Exception {
        return doFit(trainingData);
    }

    /**
     * Returns the component type for logging.
     *
     * @return "filter"
     */
    @Override
    protected final String getComponentType() {
        return "filter";
    }

    /**
     * Gets the current filter state.
     * <p>
     * Convenience method that delegates to {@link #getState()}.
     * Thread-safe via volatile read.
     *
     * @return the current pipeline state
     */
    public sentiment.PipelineState getFilterState() {
        return getState();
    }

    /**
     * Executes an inference task with thread-safe read lock, converting exceptions to RuntimeException.
     * <p>
     * This is a convenience wrapper around {@link TrainingTemplate#executeInference(InferenceTask)}
     * that converts checked exceptions to RuntimeException for filter API compatibility.
     *
     * @param <R> the return type of the inference task
     * @param task lambda containing inference logic
     * @return result of inference task
     * @throws IllegalStateException if not in READY state
     * @throws RuntimeException if inference execution fails
     */
    protected final <R> R executeFilterInference(InferenceTask<R> task) {
        try {
            return executeInference(task);
        } catch (Exception e) {
            if (e instanceof RuntimeException) {
                throw (RuntimeException) e;
            }
            throw new RuntimeException(e.getMessage(), e);
        }
    }
}
