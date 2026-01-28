package sentiment;

import java.util.ArrayList;
import java.util.List;

/**
 * Lifecycle state for pipeline components.
 *
 * <p>State transitions:
 * <pre>
 * UNINITIALIZED ─→ TRAINING ─→ READY
 *       ↑ │            ↓        ↓ │
 *       │ └──────→ ERROR ←──────┘ │
 *       └──────────────┴──────────┘
 * </pre>
 */
public enum PipelineState {
    /** Component not initialized */
    UNINITIALIZED("Component not initialized"),

    /** Training in progress */
    TRAINING("Training in progress"),

    /** Trained and ready for inference */
    READY("Trained and ready"),

    /** Training failed */
    ERROR("Training failed");

    private final String description;

    PipelineState(String description) {
        this.description = description;
    }

    public String getDescription() {
        return description;
    }

    /** @return true if component is ready for inference */
    public boolean isReady() {
        return this == READY;
    }

    /** @return true if component can accept training */
    public boolean canStartTraining() {
        return this == UNINITIALIZED || this == ERROR;
    }

    /** @return true if state is in error */
    public boolean isError() {
        return this == ERROR;
    }

    /** @return true if transition to new state is valid */
    public boolean canTransitionTo(PipelineState newState) {
        return switch (this) {
            case UNINITIALIZED -> newState == TRAINING || newState == ERROR;
            case TRAINING -> newState == READY || newState == ERROR;
            case READY -> newState == UNINITIALIZED || newState == ERROR;
            case ERROR -> newState == UNINITIALIZED;
        };
    }

    /**
     * Validates state transition.
     *
     * @param newState target state
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

    /** @return comma-separated list of valid target states */
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