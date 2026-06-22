package sentiment.api;

/**
 * Response DTO for drift detection reference window reset endpoint.
 *
 * @param message Human-readable status message
 * @param state New state after reset (typically COLLECTING_REFERENCE)
 * @param referenceWindowSize Target size for the new reference window
 */
public record DriftResetResponse(
        String message,
        String state,
        int referenceWindowSize
) {}
