package sentiment.monitoring;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import sentiment.config.DriftDetectionProperties;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Thread-safe collector for drift detection statistics.
 *
 * <p>Maintains two windows:
 * <ul>
 *   <li><b>Reference window:</b> First N predictions after startup (baseline distribution)</li>
 *   <li><b>Comparison window:</b> Most recent M predictions (sliding window)</li>
 * </ul>
 *
 * <p>The system operates in two states:
 * <ul>
 *   <li>COLLECTING_REFERENCE: Filling the reference window</li>
 *   <li>MONITORING: Reference complete, comparing against recent predictions</li>
 * </ul>
 *
 * <p><b>Performance optimization:</b> Incoming predictions are added to a lock-free
 * {@link ConcurrentLinkedQueue} on the hot path. The queue is drained to the windowed
 * structures periodically by {@link DriftDetector}, moving lock acquisition off the
 * critical inference path to meet P99 &lt; 3 ms latency SLO.
 */
@Component
@ConditionalOnProperty(name = "sentiment.drift.enabled", havingValue = "true", matchIfMissing = true)
public class DriftStatistics {

    private static final Logger logger = LoggerFactory.getLogger(DriftStatistics.class);

    private final DriftDetectionProperties config;

    // Bounded queue for incoming predictions (hot path) - prevents OOM under load
    // Size is 2x the reference window to allow burst absorption
    private static final int DEFAULT_MAX_PENDING = 10000;
    private final BlockingQueue<PredictionLogRecord> pendingRecords;
    private final AtomicInteger pendingCount = new AtomicInteger(0);
    private final AtomicLong droppedCount = new AtomicLong(0);

    // Windowed structures (accessed only during periodic drain)
    private final Deque<PredictionLogRecord> referenceWindow;
    private final Deque<PredictionLogRecord> comparisonWindow;
    private final ReadWriteLock windowLock = new ReentrantReadWriteLock();

    private volatile State state = State.COLLECTING_REFERENCE;

    public enum State {
        COLLECTING_REFERENCE,
        MONITORING
    }

    public DriftStatistics(DriftDetectionProperties config) {
        this.config = config;
        this.referenceWindow = new ArrayDeque<>(config.getReferenceWindowSize());
        this.comparisonWindow = new ArrayDeque<>(config.getComparisonWindowSize());

        // Bounded queue: 2x reference window size, or default max
        int maxPending = Math.min(config.getReferenceWindowSize() * 2, DEFAULT_MAX_PENDING);
        this.pendingRecords = new ArrayBlockingQueue<>(maxPending);

        logger.info("DriftStatistics initialized with reference window {}, comparison window {}, max pending {} (bounded queue)",
                config.getReferenceWindowSize(), config.getComparisonWindowSize(), maxPending);
    }

    /**
     * Records a prediction for drift analysis.
     *
     * <p><b>Non-blocking operation.</b> Uses bounded queue with graceful degradation:
     * if the queue is full (drain not keeping up), predictions are dropped rather
     * than blocking the inference hot path. Dropped predictions are counted for monitoring.
     *
     * @param record the prediction to record
     */
    public void recordForDrift(PredictionLogRecord record) {
        if (pendingRecords.offer(record)) {
            pendingCount.incrementAndGet();
        } else {
            // Queue full - drop record to avoid blocking hot path
            long dropped = droppedCount.incrementAndGet();
            if (dropped % 1000 == 1) {
                logger.warn("Drift statistics queue full, dropping records. Total dropped: {}", dropped);
            }
        }
    }

    /**
     * Returns the number of records dropped due to queue overflow.
     */
    public long getDroppedCount() {
        return droppedCount.get();
    }

    /**
     * Drains pending records into the windowed structures.
     *
     * <p>Called by {@link DriftDetector} before computing drift metrics.
     * This is the only place that modifies the reference/comparison windows,
     * ensuring thread safety without blocking the inference hot path.
     *
     * @return number of records drained
     */
    public int drainPendingRecords() {
        int drained = 0;
        PredictionLogRecord record;

        windowLock.writeLock().lock();
        try {
            while ((record = pendingRecords.poll()) != null) {
                pendingCount.decrementAndGet();
                drained++;

                if (state == State.COLLECTING_REFERENCE) {
                    referenceWindow.addLast(record);

                    if (referenceWindow.size() >= config.getReferenceWindowSize()) {
                        state = State.MONITORING;
                        logger.info("Reference window complete with {} predictions. Drift monitoring active.",
                                referenceWindow.size());
                    }
                } else {
                    // In MONITORING state, maintain sliding comparison window
                    if (comparisonWindow.size() >= config.getComparisonWindowSize()) {
                        comparisonWindow.removeFirst();
                    }
                    comparisonWindow.addLast(record);
                }
            }
        } finally {
            windowLock.writeLock().unlock();
        }

        if (drained > 0) {
            logger.debug("Drained {} pending records into drift windows", drained);
        }

        return drained;
    }

    /**
     * Returns the number of pending records not yet drained.
     */
    public int getPendingCount() {
        return pendingCount.get();
    }

    /**
     * Returns whether drift detection is ready (reference window complete).
     * Thread-safe: acquires read lock to ensure consistent view of window state.
     */
    public boolean isReadyForComparison() {
        if (state != State.MONITORING) {
            return false;  // Volatile read, no lock needed
        }
        windowLock.readLock().lock();
        try {
            return comparisonWindow.size() >= config.getComparisonWindowSize() / 2;
        } finally {
            windowLock.readLock().unlock();
        }
    }

    /**
     * Returns the current state.
     */
    public State getState() {
        return state;
    }

    /**
     * Returns the size of the reference window.
     */
    public int getReferenceWindowSize() {
        windowLock.readLock().lock();
        try {
            return referenceWindow.size();
        } finally {
            windowLock.readLock().unlock();
        }
    }

    /**
     * Returns the size of the comparison window.
     */
    public int getComparisonWindowSize() {
        windowLock.readLock().lock();
        try {
            return comparisonWindow.size();
        } finally {
            windowLock.readLock().unlock();
        }
    }

    /**
     * Computes label distribution for the reference window.
     *
     * @return map of label to proportion (values sum to 1.0)
     */
    public Map<String, Double> getReferenceLabelDistribution() {
        windowLock.readLock().lock();
        try {
            return computeLabelDistribution(referenceWindow);
        } finally {
            windowLock.readLock().unlock();
        }
    }

    /**
     * Computes label distribution for the comparison window.
     *
     * @return map of label to proportion (values sum to 1.0)
     */
    public Map<String, Double> getComparisonLabelDistribution() {
        windowLock.readLock().lock();
        try {
            return computeLabelDistribution(comparisonWindow);
        } finally {
            windowLock.readLock().unlock();
        }
    }

    /**
     * Computes confidence histogram for the reference window.
     *
     * @return histogram bucket proportions
     */
    public double[] getReferenceConfidenceHistogram() {
        windowLock.readLock().lock();
        try {
            return computeConfidenceHistogram(referenceWindow);
        } finally {
            windowLock.readLock().unlock();
        }
    }

    /**
     * Computes confidence histogram for the comparison window.
     *
     * @return histogram bucket proportions
     */
    public double[] getComparisonConfidenceHistogram() {
        windowLock.readLock().lock();
        try {
            return computeConfidenceHistogram(comparisonWindow);
        } finally {
            windowLock.readLock().unlock();
        }
    }

    /**
     * Resets the reference window and restarts collection.
     * Use this after model updates or when baseline needs refreshing.
     */
    public void resetReferenceWindow() {
        windowLock.writeLock().lock();
        try {
            // Clear pending queue
            pendingRecords.clear();
            pendingCount.set(0);

            // Clear windows
            referenceWindow.clear();
            comparisonWindow.clear();
            state = State.COLLECTING_REFERENCE;
            logger.info("Reference window reset. Collecting new baseline.");
        } finally {
            windowLock.writeLock().unlock();
        }
    }

    private Map<String, Double> computeLabelDistribution(Deque<PredictionLogRecord> window) {
        Map<String, Integer> counts = new HashMap<>();
        int total = 0;

        for (PredictionLogRecord record : window) {
            counts.merge(record.predictedLabel().toLowerCase(), 1, Integer::sum);
            total++;
        }

        Map<String, Double> distribution = new HashMap<>();
        if (total > 0) {
            for (Map.Entry<String, Integer> entry : counts.entrySet()) {
                distribution.put(entry.getKey(), entry.getValue() / (double) total);
            }
        }

        // Ensure both positive and negative are present
        distribution.putIfAbsent("positive", 0.0);
        distribution.putIfAbsent("negative", 0.0);

        return distribution;
    }

    private double[] computeConfidenceHistogram(Deque<PredictionLogRecord> window) {
        int numBuckets = config.getConfidenceHistogramBuckets();
        int[] counts = new int[numBuckets];
        int total = 0;

        for (PredictionLogRecord record : window) {
            // Map confidence [0, 1] to bucket index [0, numBuckets-1]
            int bucket = (int) (record.confidence() * numBuckets);
            if (bucket >= numBuckets) {
                bucket = numBuckets - 1; // Handle confidence = 1.0
            }
            counts[bucket]++;
            total++;
        }

        double[] histogram = new double[numBuckets];
        if (total > 0) {
            for (int i = 0; i < numBuckets; i++) {
                histogram[i] = counts[i] / (double) total;
            }
        } else {
            // Uniform distribution if empty
            double uniform = 1.0 / numBuckets;
            for (int i = 0; i < numBuckets; i++) {
                histogram[i] = uniform;
            }
        }

        return histogram;
    }
}
