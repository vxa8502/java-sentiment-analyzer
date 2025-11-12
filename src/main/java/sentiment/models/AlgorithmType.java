package sentiment.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Type-safe enumeration of supported sentiment analysis algorithms.
 * Provides consistent algorithm identification across the system.
 */
public enum AlgorithmType {
    SVM("SVM (SMO)", "Support Vector Machine with Sequential Minimal Optimization"),
    NAIVE_BAYES("Naive Bayes", "Probabilistic classifier based on Bayes' theorem"),
    RANDOM_FOREST("Random Forest", "Ensemble learning method using decision trees"),
    LOGISTIC_REGRESSION("Logistic Regression", "Linear model for binary classification"),
    NEURAL_NETWORK("Neural Network", "Multi-layer perceptron classifier"),
    UNKNOWN("Unknown", "Algorithm type not specified");

    private final String displayName;
    private final String description;

    AlgorithmType(String displayName, String description) {
        this.displayName = displayName;
        this.description = description;
    }

    /**
     * Get human-readable algorithm name for UI display and logging
     */
    public String getDisplayName() {
        return displayName;
    }

    /**
     * Get detailed description of the algorithm
     */
    public String getDescription() {
        return description;
    }

    /**
     * Parse algorithm type from string with priority-based matching.
     *
     * Matching priority:
     * 1. Exact enum name match (e.g., "NAIVE_BAYES")
     * 2. Exact display name match (e.g., "Naive Bayes")
     * 3. Display name starts with input (e.g., "naive" → "Naive Bayes")
     * 4. Display name contains input (only if unambiguous)
     *
     * @param algorithmName Algorithm name to parse
     * @return Matching AlgorithmType
     * @throws IllegalArgumentException if name is invalid or ambiguous
     */
    public static AlgorithmType fromString(String algorithmName) {
        if (algorithmName == null || algorithmName.trim().isEmpty()) {
            throw new IllegalArgumentException("Algorithm name cannot be null or empty");
        }

        String normalized = algorithmName.trim().toLowerCase();

        // Priority 1: Exact enum name match
        for (AlgorithmType type : values()) {
            if (type.name().toLowerCase().equals(normalized)) {
                return type;
            }
        }

        // Priority 2: Exact display name match
        for (AlgorithmType type : values()) {
            if (type.displayName.toLowerCase().equals(normalized)) {
                return type;
            }
        }

        // Priority 3: Display name starts with input
        List<AlgorithmType> startsWithMatches = new ArrayList<>();
        for (AlgorithmType type : values()) {
            if (type.displayName.toLowerCase().startsWith(normalized)) {
                startsWithMatches.add(type);
            }
        }
        if (startsWithMatches.size() == 1) {
            return startsWithMatches.get(0);
        }

        // Priority 4: Display name contains input (only if unambiguous)
        List<AlgorithmType> containsMatches = new ArrayList<>();
        for (AlgorithmType type : values()) {
            if (type.displayName.toLowerCase().contains(normalized)) {
                containsMatches.add(type);
            }
        }

        if (containsMatches.size() == 1) {
            return containsMatches.get(0);
        } else if (containsMatches.size() > 1) {
            String matchingTypes = containsMatches.stream()
                    .map(t -> t.displayName)
                    .collect(Collectors.joining(", "));
            throw new IllegalArgumentException(
                    "Ambiguous algorithm name '" + algorithmName + "' matches multiple types: " +
                            matchingTypes + ". Use exact name for clarity."
            );
        }

        // No matches found
        String availableTypes = Arrays.stream(values())
                .map(t -> t.displayName)
                .collect(Collectors.joining(", "));
        throw new IllegalArgumentException(
                "Unknown algorithm: '" + algorithmName + "'. Available: " + availableTypes
        );
    }

    @Override
    public String toString() {
        return displayName;
    }
}