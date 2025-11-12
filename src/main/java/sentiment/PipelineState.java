package sentiment;

import java.util.ArrayList;
import java.util.List;

/**
 * Unified lifecycle state for all pipeline components (preprocessing, models, etc.).
 * Eliminates duplication between FilterState and ClassifierState.
 *
 * Provides STRICT state management with enforced transitions.
 *
 * CRITICAL: All state changes MUST go through validateTransition()
 * to enforce the state machine.
 *
 * STATE MACHINE:
 * UNINITIALIZED --train()/fit()--> TRAINING --success--> READY
 *      ^                              |                    |
 *      |                           failure              reset()
 *      |                              |                    |
 *      +--------<-----------------ERROR---------<---------+
 */
public enum PipelineState {
    /**
     * Initial state - component has not been configured or trained
     */
    UNINITIALIZED("Component not initialized"),

    /**
     * Transient state - training is in progress
     */
    TRAINING("Training in progress"),

    /**
     * Ready state - component is trained and ready for inference
     */
    READY("Trained and ready"),

    /**
     * Error state - training failed
     */
    ERROR("Training failed");

    private final String description;

    PipelineState(String description) {
        this.description = description;
    }

    public String getDescription() {
        return description;
    }

    /**
     * Check if component is ready for inference
     */
    public boolean isReady() {
        return this == READY;
    }

    /**
     * Check if component can accept training
     */
    public boolean canStartTraining() {
        return this == UNINITIALIZED || this == ERROR;
    }

    /**
     * Check if state is in error
     */
    public boolean isError() {
        return this == ERROR;
    }

    /**
     * Check if transition to new state is valid.
     *
     * Valid transitions:
     * - UNINITIALIZED → TRAINING, ERROR
     * - TRAINING → READY, ERROR
     * - READY → UNINITIALIZED, ERROR (for reset)
     * - ERROR → UNINITIALIZED (for recovery)
     */
    public boolean canTransitionTo(PipelineState newState) {
        return switch (this) {
            case UNINITIALIZED -> newState == TRAINING || newState == ERROR;
            case TRAINING -> newState == READY || newState == ERROR;
            case READY -> newState == UNINITIALIZED || newState == ERROR;
            case ERROR -> newState == UNINITIALIZED;
        };
    }

    /**
     * ENFORCED state transition validation.
     * Throws IllegalStateException if transition is not allowed.
     *
     * Use this method for ALL state transitions to enforce the state machine.
     *
     * @param newState Target state
     * @throws IllegalStateException if transition is invalid
     */
    public void validateTransition(PipelineState newState) {
        if (!canTransitionTo(newState)) {
            throw new IllegalStateException(
                    String.format("Invalid state transition: %s -> %s. Valid transitions from %s: %s",
                            this, newState, this, getValidTransitions())
            );
        }
    }

    /**
     * Get list of valid target states from current state.
     * Derives valid transitions programmatically from canTransitionTo().
     */
    private String getValidTransitions() {
        List<String> validTransitions = new ArrayList<>();

        for (PipelineState targetState : values()) {
            if (canTransitionTo(targetState)) {
                validTransitions.add(targetState.name());
            }
        }

        return String.join(", ", validTransitions);
    }

    @Override
    public String toString() {
        return name() + ": " + description;
    }
}