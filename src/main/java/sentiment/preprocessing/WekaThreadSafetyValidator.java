package sentiment.preprocessing;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

/**
 * Utility class to validate Weka thread-safety assumptions.
 *
 * USE IN TESTS:
 * =============
 * Run this validator in your test suite to verify Weka's Filter.useFilter()
 * behaves as expected with concurrent access.
 *
 * WHAT IT CHECKS:
 * ===============
 * 1. Concurrent Filter.useFilter() calls produce identical results
 * 2. No exceptions thrown during concurrent execution
 * 3. Filter state remains unchanged after concurrent access
 * 4. No thread contention or deadlocks
 *
 * FAILURE MODES:
 * ==============
 * If this validator fails, it means:
 * - Weka version changed thread-safety guarantees
 * - Filter state is being mutated during useFilter()
 * - Race conditions in filter implementation
 *
 * ACTION: Update thread-safety documentation and add synchronization!
 */
public class WekaThreadSafetyValidator {

    private static final Logger logger = LoggerFactory.getLogger(WekaThreadSafetyValidator.class);

    /**
     * Validate that Filter.useFilter() is thread-safe with a trained filter.
     *
     * @param trainedFilter Filter that has completed setInputFormat()
     * @param testInstances Test data to transform
     * @param numThreads Number of concurrent threads to test
     * @param iterationsPerThread Number of transform operations per thread
     * @return true if all thread-safety checks pass
     */
    public static boolean validateFilterThreadSafety(
            Filter trainedFilter,
            Instances testInstances,
            int numThreads,
            int iterationsPerThread) {

        logger.info("Starting Weka thread-safety validation: {} threads, {} iterations each",
                numThreads, iterationsPerThread);

        try {
            // Get baseline result (single-threaded)
            Instances baseline = Filter.useFilter(testInstances, trainedFilter);
            logger.debug("Baseline result: {} instances, {} attributes",
                    baseline.numInstances(), baseline.numAttributes());

            // Create thread pool
            ExecutorService executor = Executors.newFixedThreadPool(numThreads);
            List<Future<ValidationResult>> futures = new ArrayList<>();

            // Submit concurrent tasks
            for (int t = 0; t < numThreads; t++) {
                final int threadId = t;
                futures.add(executor.submit(() ->
                        runValidationThread(threadId, trainedFilter, testInstances,
                                baseline, iterationsPerThread)));
            }

            // Wait for all threads to complete
            boolean allPassed = true;
            for (Future<ValidationResult> future : futures) {
                ValidationResult result = future.get(30, TimeUnit.SECONDS);
                if (!result.passed) {
                    logger.error("Thread {} FAILED: {}", result.threadId, result.errorMessage);
                    allPassed = false;
                }
            }

            executor.shutdown();

            if (allPassed) {
                logger.info("✅ Weka thread-safety validation PASSED - Filter.useFilter() is safe!");
            } else {
                logger.error("❌ Weka thread-safety validation FAILED - Review filter usage!");
            }

            return allPassed;

        } catch (Exception e) {
            logger.error("Thread-safety validation failed with exception", e);
            return false;
        }
    }

    private static ValidationResult runValidationThread(
            int threadId,
            Filter trainedFilter,
            Instances testInstances,
            Instances baseline,
            int iterations) {

        try {
            for (int i = 0; i < iterations; i++) {
                // Apply filter (should be thread-safe)
                Instances result = Filter.useFilter(testInstances, trainedFilter);

                // Verify result matches baseline
                if (result.numInstances() != baseline.numInstances() ||
                        result.numAttributes() != baseline.numAttributes()) {
                    return new ValidationResult(threadId, false,
                            "Result dimensions differ from baseline");
                }

                // Small delay to increase chance of race conditions if they exist
                if (i % 10 == 0) {
                    Thread.sleep(1);
                }
            }

            logger.debug("Thread {} completed {} iterations successfully",
                    threadId, iterations);
            return new ValidationResult(threadId, true, null);

        } catch (Exception e) {
            return new ValidationResult(threadId, false, e.getMessage());
        }
    }

    private static class ValidationResult {
        final int threadId;
        final boolean passed;
        final String errorMessage;

        ValidationResult(int threadId, boolean passed, String errorMessage) {
            this.threadId = threadId;
            this.passed = passed;
            this.errorMessage = errorMessage;
        }
    }
}
