package sentiment.api;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PatchMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import sentiment.monitoring.DriftDetector;
import sentiment.monitoring.DriftResult;
import sentiment.monitoring.DriftStatistics;
import sentiment.monitoring.PredictionLogger;

/**
 * REST endpoint for drift detection status and management.
 *
 * <p>Provides endpoints to:
 * <ul>
 *   <li>Query current drift status and PSI values</li>
 *   <li>Trigger on-demand drift detection</li>
 *   <li>Reset the reference window (after model updates)</li>
 * </ul>
 */
@RestController
@RequestMapping("/api/v1/drift")
@ConditionalOnProperty(name = "sentiment.drift.enabled", havingValue = "true", matchIfMissing = true)
public class DriftController {

    private final DriftDetector driftDetector;
    private final DriftStatistics statistics;
    private final PredictionLogger logger;

    public DriftController(
            DriftDetector driftDetector,
            DriftStatistics statistics,
            PredictionLogger logger) {
        this.driftDetector = driftDetector;
        this.statistics = statistics;
        this.logger = logger;
    }

    /**
     * Returns the current drift detection status.
     *
     * <p>Example response:
     * <pre>
     * {
     *   "status": "HEALTHY",
     *   "predictionPsi": 0.023,
     *   "confidencePsi": 0.015,
     *   "maxPsi": 0.023,
     *   "referenceWindowSize": 1000,
     *   "comparisonWindowSize": 500,
     *   "state": "MONITORING",
     *   "ready": true,
     *   "computedAt": "2026-06-21T12:00:00Z",
     *   "logging": {
     *     "queueSize": 5,
     *     "logged": 15432,
     *     "dropped": 0
     *   }
     * }
     * </pre>
     */
    @GetMapping("/status")
    public ResponseEntity<DriftStatusResponse> getDriftStatus() {
        DriftResult result = driftDetector.getLatestResult();

        return ResponseEntity.ok(new DriftStatusResponse(
                result.status().name(),
                result.predictionPsi(),
                result.confidencePsi(),
                result.maxPsi(),
                statistics.getReferenceWindowSize(),
                statistics.getComparisonWindowSize(),
                statistics.getPendingCount(),
                statistics.getState().name(),
                statistics.isReadyForComparison(),
                result.computedAt() != null ? result.computedAt().toString() : null,
                new DriftStatusResponse.LoggingStatus(
                        logger.getQueueSize(),
                        logger.getLoggedCount(),
                        logger.getDroppedCount()
                )
        ));
    }

    /**
     * Triggers on-demand drift detection.
     *
     * <p>Useful for immediate checks after deployments or when investigating issues.
     */
    @PostMapping("/detect")
    public ResponseEntity<DriftStatusResponse> detectNow() {
        DriftResult result = driftDetector.detectDriftNow();

        return ResponseEntity.ok(new DriftStatusResponse(
                result.status().name(),
                result.predictionPsi(),
                result.confidencePsi(),
                result.maxPsi(),
                result.referenceSize(),
                result.comparisonSize(),
                statistics.getPendingCount(),
                statistics.getState().name(),
                statistics.isReadyForComparison(),
                result.computedAt() != null ? result.computedAt().toString() : null,
                new DriftStatusResponse.LoggingStatus(
                        logger.getQueueSize(),
                        logger.getLoggedCount(),
                        logger.getDroppedCount()
                )
        ));
    }

    /**
     * Resets the reference window and starts collecting a new baseline.
     *
     * <p>Use this endpoint after:
     * <ul>
     *   <li>Model updates or retraining</li>
     *   <li>Known distribution shifts (e.g., new product category)</li>
     *   <li>Investigation reveals the current baseline is inappropriate</li>
     * </ul>
     *
     * <p>Uses PATCH because this modifies the existing reference window state
     * rather than creating a new resource or triggering a side-effect action.
     */
    @PatchMapping("/reference")
    public ResponseEntity<DriftResetResponse> resetReferenceWindow() {
        statistics.resetReferenceWindow();

        return ResponseEntity.ok(new DriftResetResponse(
                "Reference window reset. Collecting new baseline.",
                statistics.getState().name(),
                statistics.getReferenceWindowSize()
        ));
    }
}
