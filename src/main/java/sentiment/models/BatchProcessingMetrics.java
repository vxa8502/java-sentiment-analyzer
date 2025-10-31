package sentiment.models;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Performance metrics for batch processing operations.
 *
 * Tracks key performance indicators to help identify bottlenecks and
 * optimize batch processing configuration.
 *
 * METRICS TRACKED:
 * ================
 * - Total processing time
 * - Preprocessing time
 * - Transformation time
 * - Classification time
 * - Items processed
 * - Throughput (items/second)
 *
 * THREAD SAFETY:
 * ==============
 * Uses AtomicLong for thread-safe metric accumulation during parallel processing.
 */
public class BatchProcessingMetrics {

    private final long startTimeMs;
    private long endTimeMs;

    // Time breakdowns (in milliseconds)
    private final AtomicLong preprocessingTimeMs = new AtomicLong(0);
    private final AtomicLong transformationTimeMs = new AtomicLong(0);
    private final AtomicLong classificationTimeMs = new AtomicLong(0);

    // Processing stats
    private final AtomicLong itemsProcessed = new AtomicLong(0);
    private final AtomicLong miniBatchesProcessed = new AtomicLong(0);
    private final int totalItems;
    private final int batchSize;
    private final boolean parallelMode;

    /**
     * Creates new metrics tracker for a batch operation.
     */
    public BatchProcessingMetrics(int totalItems, int batchSize, boolean parallelMode) {
        this.startTimeMs = System.currentTimeMillis();
        this.totalItems = totalItems;
        this.batchSize = batchSize;
        this.parallelMode = parallelMode;
    }

    /**
     * Records preprocessing time.
     */
    public void recordPreprocessingTime(long timeMs) {
        preprocessingTimeMs.addAndGet(timeMs);
    }

    /**
     * Records transformation time.
     */
    public void recordTransformationTime(long timeMs) {
        transformationTimeMs.addAndGet(timeMs);
    }

    /**
     * Records classification time.
     */
    public void recordClassificationTime(long timeMs) {
        classificationTimeMs.addAndGet(timeMs);
    }

    /**
     * Records items processed in a mini-batch.
     */
    public void recordItemsProcessed(int count) {
        itemsProcessed.addAndGet(count);
        miniBatchesProcessed.incrementAndGet();
    }

    /**
     * Marks the batch operation as complete.
     */
    public void markComplete() {
        this.endTimeMs = System.currentTimeMillis();
    }

    /**
     * Gets total processing time in milliseconds.
     */
    public long getTotalTimeMs() {
        long end = endTimeMs > 0 ? endTimeMs : System.currentTimeMillis();
        return end - startTimeMs;
    }

    /**
     * Calculates throughput in items per second.
     */
    public double getThroughput() {
        long totalTime = getTotalTimeMs();
        if (totalTime == 0) return 0.0;
        return (itemsProcessed.get() * 1000.0) / totalTime;
    }

    /**
     * Gets average time per item in milliseconds.
     */
    public double getAverageTimePerItem() {
        long processed = itemsProcessed.get();
        if (processed == 0) return 0.0;
        return (double) getTotalTimeMs() / processed;
    }

    // Getters
    public long getPreprocessingTimeMs() {
        return preprocessingTimeMs.get();
    }

    public long getTransformationTimeMs() {
        return transformationTimeMs.get();
    }

    public long getClassificationTimeMs() {
        return classificationTimeMs.get();
    }

    public long getItemsProcessed() {
        return itemsProcessed.get();
    }

    public long getMiniBatchesProcessed() {
        return miniBatchesProcessed.get();
    }

    public int getTotalItems() {
        return totalItems;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public boolean isParallelMode() {
        return parallelMode;
    }

    /**
     * Returns a formatted summary of the metrics.
     */
    public String getSummary() {
        return String.format(
                "Batch Processing Metrics:\n" +
                "  Total Items: %d\n" +
                "  Items Processed: %d\n" +
                "  Mini-batches: %d (size: %d)\n" +
                "  Parallel Mode: %s\n" +
                "  Total Time: %dms\n" +
                "  Preprocessing: %dms (%.1f%%)\n" +
                "  Transformation: %dms (%.1f%%)\n" +
                "  Classification: %dms (%.1f%%)\n" +
                "  Throughput: %.1f items/sec\n" +
                "  Avg Time/Item: %.2fms",
                totalItems,
                itemsProcessed.get(),
                miniBatchesProcessed.get(),
                batchSize,
                parallelMode ? "YES" : "NO",
                getTotalTimeMs(),
                preprocessingTimeMs.get(),
                getPercentage(preprocessingTimeMs.get()),
                transformationTimeMs.get(),
                getPercentage(transformationTimeMs.get()),
                classificationTimeMs.get(),
                getPercentage(classificationTimeMs.get()),
                getThroughput(),
                getAverageTimePerItem()
        );
    }

    private double getPercentage(long timeMs) {
        long total = getTotalTimeMs();
        if (total == 0) return 0.0;
        return (timeMs * 100.0) / total;
    }

    @Override
    public String toString() {
        return String.format(
                "BatchMetrics{items=%d/%d, time=%dms, throughput=%.1f items/sec}",
                itemsProcessed.get(), totalItems, getTotalTimeMs(), getThroughput()
        );
    }
}
