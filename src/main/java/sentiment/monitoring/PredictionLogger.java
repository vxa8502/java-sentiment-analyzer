package sentiment.monitoring;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import sentiment.config.DriftDetectionProperties;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Stream;

/**
 * Asynchronous prediction logger using a bounded queue and dedicated writer thread.
 *
 * <p>Implements graceful degradation: drops predictions if queue is full rather than
 * blocking the inference thread. Dropped predictions are counted and logged.
 *
 * <p>Prediction logs are written in JSON Lines format with daily rotation.
 * Old log files are automatically cleaned up based on retention configuration.
 */
@Component
@ConditionalOnProperty(name = "sentiment.drift.enabled", havingValue = "true", matchIfMissing = true)
public class PredictionLogger implements DisposableBean {

    private static final Logger logger = LoggerFactory.getLogger(PredictionLogger.class);
    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd");

    private final DriftDetectionProperties config;
    private final BlockingQueue<PredictionLogRecord> queue;
    private final ExecutorService writerExecutor;
    private final AtomicBoolean running = new AtomicBoolean(true);
    private final Counter loggedCounter;
    private final Counter droppedCounter;

    private volatile BufferedWriter currentWriter;
    private volatile LocalDate currentFileDate;
    private volatile Path currentFilePath;

    // Batch flush configuration for performance (avoid flush per record)
    private static final int FLUSH_BATCH_SIZE = 20;
    private int unflushedCount = 0;

    public PredictionLogger(DriftDetectionProperties config, MeterRegistry meterRegistry) {
        this.config = config;
        this.queue = new ArrayBlockingQueue<>(config.getMaxQueueSize());

        // Register metrics
        this.loggedCounter = Counter.builder("sentiment.predictions.logged")
                .description("Total predictions logged to file")
                .register(meterRegistry);

        this.droppedCounter = Counter.builder("sentiment.predictions.dropped")
                .description("Predictions dropped due to queue full")
                .register(meterRegistry);

        // Initialize log directory
        initializeLogDirectory();

        // Start background writer thread
        this.writerExecutor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "prediction-logger");
            t.setDaemon(true);
            return t;
        });
        this.writerExecutor.submit(this::writerLoop);

        logger.info("PredictionLogger initialized with queue size {} and log path {}",
                config.getMaxQueueSize(), config.getLogPath());
    }

    /**
     * Logs a prediction asynchronously.
     *
     * <p>This method is non-blocking. If the queue is full, the prediction is dropped
     * and a counter is incremented.
     *
     * @param record the prediction to log
     */
    public void logPrediction(PredictionLogRecord record) {
        if (!running.get()) {
            return;
        }

        if (!queue.offer(record)) {
            droppedCounter.increment();
            if (droppedCounter.count() % 100 == 1) {
                // Log periodically to avoid log spam
                logger.warn("Prediction log queue full, dropping records. Total dropped: {}",
                        (long) droppedCounter.count());
            }
        }
    }

    /**
     * Returns the current queue size for monitoring.
     */
    public int getQueueSize() {
        return queue.size();
    }

    /**
     * Returns total predictions logged.
     */
    public long getLoggedCount() {
        return (long) loggedCounter.count();
    }

    /**
     * Returns total predictions dropped.
     */
    public long getDroppedCount() {
        return (long) droppedCounter.count();
    }

    @Override
    public void destroy() {
        logger.info("Shutting down PredictionLogger...");
        running.set(false);

        // Drain remaining items from queue
        writerExecutor.shutdown();
        try {
            if (!writerExecutor.awaitTermination(5, TimeUnit.SECONDS)) {
                writerExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            writerExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }

        closeCurrentWriter();
        logger.info("PredictionLogger shutdown complete. Logged: {}, Dropped: {}",
                (long) loggedCounter.count(), (long) droppedCounter.count());
    }

    private void writerLoop() {
        while (running.get() || !queue.isEmpty()) {
            try {
                PredictionLogRecord record = queue.poll(100, TimeUnit.MILLISECONDS);
                if (record != null) {
                    writeRecord(record);
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            } catch (Exception e) {
                logger.error("Error writing prediction log", e);
            }
        }

        // Drain any remaining records after shutdown signal
        PredictionLogRecord record;
        while ((record = queue.poll()) != null) {
            try {
                writeRecord(record);
            } catch (Exception e) {
                logger.error("Error writing prediction log during shutdown", e);
            }
        }
    }

    private void writeRecord(PredictionLogRecord record) throws IOException {
        rotateFileIfNeeded();

        if (currentWriter == null) {
            openNewFile();
        }

        currentWriter.write(record.toJsonLine());
        currentWriter.newLine();
        unflushedCount++;

        // Batch flush for performance: flush every N records instead of every record
        // This reduces syscall overhead by ~20x while maintaining reasonable durability
        if (unflushedCount >= FLUSH_BATCH_SIZE) {
            currentWriter.flush();
            unflushedCount = 0;
        }
        loggedCounter.increment();
    }

    /**
     * Rotates log file at date boundary.
     *
     * <p>Synchronized to prevent race conditions during file rotation, even though
     * we currently use a single-threaded executor. This provides defense-in-depth
     * against future changes that might introduce concurrent access, and ensures
     * safe interaction with the destroy() method.
     */
    private synchronized void rotateFileIfNeeded() throws IOException {
        LocalDate today = LocalDate.now();
        if (currentFileDate == null || !currentFileDate.equals(today)) {
            closeCurrentWriter();
            currentFileDate = today;
            openNewFile();

            // Cleanup old files in background
            cleanupOldFilesAsync();
        }
    }

    private void openNewFile() throws IOException {
        Path logDir = Path.of(config.getLogPath());
        String filename = String.format("predictions-%s.jsonl", DATE_FORMATTER.format(currentFileDate));
        currentFilePath = logDir.resolve(filename);

        currentWriter = Files.newBufferedWriter(
                currentFilePath,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND
        );

        logger.info("Opened prediction log file: {}", currentFilePath);
    }

    /**
     * Closes the current writer safely, flushing any remaining records.
     * Synchronized to coordinate with rotateFileIfNeeded() and destroy().
     */
    private synchronized void closeCurrentWriter() {
        if (currentWriter != null) {
            try {
                // Flush any remaining unflushed records before closing
                if (unflushedCount > 0) {
                    currentWriter.flush();
                    unflushedCount = 0;
                }
                currentWriter.close();
            } catch (IOException e) {
                logger.error("Error closing prediction log file", e);
            }
            currentWriter = null;
        }
    }

    private void initializeLogDirectory() {
        try {
            Path logDir = Path.of(config.getLogPath());
            if (!Files.exists(logDir)) {
                Files.createDirectories(logDir);
                logger.info("Created prediction log directory: {}", logDir);
            }
        } catch (IOException e) {
            logger.error("Failed to create prediction log directory: {}", config.getLogPath(), e);
        }
    }

    private void cleanupOldFilesAsync() {
        // Run cleanup in a daemon thread to avoid blocking the writer
        Thread cleanupThread = new Thread(() -> {
            try {
                cleanupOldFiles();
            } catch (Exception e) {
                logger.error("Error cleaning up old prediction logs", e);
            }
        }, "prediction-log-cleanup");
        cleanupThread.setDaemon(true);
        cleanupThread.start();
    }

    private void cleanupOldFiles() throws IOException {
        Path logDir = Path.of(config.getLogPath());
        if (!Files.exists(logDir)) {
            return;
        }

        LocalDate cutoff = LocalDate.now().minusDays(config.getLogRetention().toDays());

        try (Stream<Path> files = Files.list(logDir)) {
            files.filter(p -> p.getFileName().toString().startsWith("predictions-"))
                    .filter(p -> p.getFileName().toString().endsWith(".jsonl"))
                    .filter(p -> isOlderThan(p, cutoff))
                    .forEach(this::deleteFile);
        }
    }

    private boolean isOlderThan(Path file, LocalDate cutoff) {
        String filename = file.getFileName().toString();
        // Extract date from "predictions-YYYY-MM-DD.jsonl"
        if (filename.length() < 22) {
            return false;
        }
        try {
            String dateStr = filename.substring(12, 22);
            LocalDate fileDate = LocalDate.parse(dateStr, DATE_FORMATTER);
            return fileDate.isBefore(cutoff);
        } catch (Exception e) {
            return false;
        }
    }

    private void deleteFile(Path file) {
        try {
            Files.delete(file);
            logger.info("Deleted old prediction log: {}", file);
        } catch (IOException e) {
            logger.warn("Failed to delete old prediction log: {}", file, e);
        }
    }
}
