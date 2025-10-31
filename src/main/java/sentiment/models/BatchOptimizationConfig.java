package sentiment.models;

/**
 * Configuration for batch processing optimization in SVM classifier.
 *
 * OPTIMIZATION STRATEGIES:
 * ========================
 * 1. Mini-batching: Process large batches in smaller chunks to balance memory and throughput
 * 2. Parallel processing: Leverage multiple CPU cores for independent predictions
 * 3. Adaptive sizing: Automatically adjust batch size based on input size
 *
 * PERFORMANCE CONSIDERATIONS:
 * ===========================
 * - Smaller batches: Lower memory, higher overhead, better for constrained environments
 * - Larger batches: Higher throughput, more memory, better for high-volume scenarios
 * - Parallel processing: Best for CPU-bound operations with large batches (>100 items)
 *
 * DEFAULT CONFIGURATION:
 * ======================
 * - Batch size: 100 (balanced for most use cases)
 * - Parallel processing: Enabled for batches > 500 items
 * - Parallel threshold: 500 items
 * - Thread pool size: Runtime.availableProcessors()
 */
public class BatchOptimizationConfig {

    // Default configuration values
    private static final int DEFAULT_BATCH_SIZE = 100;
    private static final int DEFAULT_PARALLEL_THRESHOLD = 500;
    private static final boolean DEFAULT_ENABLE_PARALLEL = true;
    private static final boolean DEFAULT_ENABLE_METRICS = true;

    private final int batchSize;
    private final int parallelThreshold;
    private final boolean enableParallelProcessing;
    private final boolean enableMetrics;
    private final int threadPoolSize;

    private BatchOptimizationConfig(Builder builder) {
        this.batchSize = builder.batchSize;
        this.parallelThreshold = builder.parallelThreshold;
        this.enableParallelProcessing = builder.enableParallelProcessing;
        this.enableMetrics = builder.enableMetrics;
        this.threadPoolSize = builder.threadPoolSize;
    }

    /**
     * Creates a default configuration with balanced settings.
     */
    public static BatchOptimizationConfig getDefault() {
        return builder().build();
    }

    /**
     * Creates a configuration optimized for low memory environments.
     */
    public static BatchOptimizationConfig forLowMemory() {
        return builder()
                .batchSize(50)
                .enableParallelProcessing(false)
                .build();
    }

    /**
     * Creates a configuration optimized for high throughput.
     */
    public static BatchOptimizationConfig forHighThroughput() {
        return builder()
                .batchSize(200)
                .parallelThreshold(200)
                .enableParallelProcessing(true)
                .build();
    }

    /**
     * Creates a builder for custom configuration.
     */
    public static Builder builder() {
        return new Builder();
    }

    // Getters
    public int getBatchSize() {
        return batchSize;
    }

    public int getParallelThreshold() {
        return parallelThreshold;
    }

    public boolean isEnableParallelProcessing() {
        return enableParallelProcessing;
    }

    public boolean isEnableMetrics() {
        return enableMetrics;
    }

    public int getThreadPoolSize() {
        return threadPoolSize;
    }

    /**
     * Determines if parallel processing should be used for the given input size.
     */
    public boolean shouldUseParallel(int inputSize) {
        return enableParallelProcessing && inputSize >= parallelThreshold;
    }

    /**
     * Calculates the number of mini-batches needed for the given input size.
     */
    public int calculateMiniBatches(int inputSize) {
        return (int) Math.ceil((double) inputSize / batchSize);
    }

    @Override
    public String toString() {
        return String.format(
                "BatchOptimizationConfig{batchSize=%d, parallelThreshold=%d, " +
                "enableParallel=%s, enableMetrics=%s, threadPoolSize=%d}",
                batchSize, parallelThreshold, enableParallelProcessing,
                enableMetrics, threadPoolSize
        );
    }

    /**
     * Builder for BatchOptimizationConfig.
     */
    public static class Builder {
        private int batchSize = DEFAULT_BATCH_SIZE;
        private int parallelThreshold = DEFAULT_PARALLEL_THRESHOLD;
        private boolean enableParallelProcessing = DEFAULT_ENABLE_PARALLEL;
        private boolean enableMetrics = DEFAULT_ENABLE_METRICS;
        private int threadPoolSize = Runtime.getRuntime().availableProcessors();

        /**
         * Sets the batch size for mini-batch processing.
         * @param batchSize Size of each mini-batch (must be > 0)
         */
        public Builder batchSize(int batchSize) {
            if (batchSize <= 0) {
                throw new IllegalArgumentException("Batch size must be positive");
            }
            this.batchSize = batchSize;
            return this;
        }

        /**
         * Sets the threshold for enabling parallel processing.
         * @param threshold Minimum number of items to trigger parallel processing
         */
        public Builder parallelThreshold(int threshold) {
            if (threshold < 0) {
                throw new IllegalArgumentException("Parallel threshold cannot be negative");
            }
            this.parallelThreshold = threshold;
            return this;
        }

        /**
         * Enables or disables parallel processing.
         */
        public Builder enableParallelProcessing(boolean enable) {
            this.enableParallelProcessing = enable;
            return this;
        }

        /**
         * Enables or disables performance metrics collection.
         */
        public Builder enableMetrics(boolean enable) {
            this.enableMetrics = enable;
            return this;
        }

        /**
         * Sets the thread pool size for parallel processing.
         * @param size Number of threads (must be > 0)
         */
        public Builder threadPoolSize(int size) {
            if (size <= 0) {
                throw new IllegalArgumentException("Thread pool size must be positive");
            }
            this.threadPoolSize = size;
            return this;
        }

        /**
         * Builds the configuration.
         */
        public BatchOptimizationConfig build() {
            return new BatchOptimizationConfig(this);
        }
    }
}
