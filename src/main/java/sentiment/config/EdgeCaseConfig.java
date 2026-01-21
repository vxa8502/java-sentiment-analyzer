package sentiment.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/**
 * Configuration for edge case evaluation pipeline.
 * Loaded from config/edge-case-evaluation.json.
 * Shared format with Python scripts for reproducibility.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class EdgeCaseConfig {

    private static final String DEFAULT_CONFIG_PATH = "config/edge-case-evaluation.json";
    private static final String ENV_CONFIG_PATH = "EDGE_CASE_CONFIG";

    @JsonProperty("_version")
    private String version;

    @JsonProperty("directories")
    private Directories directories;

    @JsonProperty("edge_case_definition")
    private EdgeCaseDefinition edgeCaseDefinition;

    @JsonProperty("sampling")
    private Sampling sampling;

    @JsonProperty("filtering")
    private Filtering filtering;

    @JsonProperty("evaluation")
    private Evaluation evaluation;

    @JsonProperty("categories")
    private List<Category> categories;

    @JsonProperty("algorithms")
    private List<String> algorithms;

    @JsonProperty("domains")
    private List<String> domains;

    // Nested config classes

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Directories {
        @JsonProperty("edge_cases_dir")
        private String edgeCasesDir = "data/raw/edge_cases";

        @JsonProperty("candidates_dir")
        private String candidatesDir = "data/raw/edge_cases/candidates";

        @JsonProperty("models_dir")
        private String modelsDir = "models";

        public String getEdgeCasesDir() { return edgeCasesDir; }
        public String getCandidatesDir() { return candidatesDir; }
        public String getModelsDir() { return modelsDir; }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class EdgeCaseDefinition {
        @JsonProperty("min_models_failed")
        private int minModelsFailed = 3;

        @JsonProperty("rationale")
        private String rationale = "Errors where 3+ algorithms failed indicate systematic difficulty";

        public int getMinModelsFailed() { return minModelsFailed; }
        public String getRationale() { return rationale; }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Sampling {
        @JsonProperty("min_samples_per_category")
        private int minSamplesPerCategory = 40;

        @JsonProperty("target_margin_of_error")
        private double targetMarginOfError = 0.15;

        @JsonProperty("distribution_buffer")
        private double distributionBuffer = 1.25;

        @JsonProperty("random_seed")
        private long randomSeed = 42L;

        public int getMinSamplesPerCategory() { return minSamplesPerCategory; }
        public double getTargetMarginOfError() { return targetMarginOfError; }
        public double getDistributionBuffer() { return distributionBuffer; }
        public long getRandomSeed() { return randomSeed; }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Filtering {
        @JsonProperty("label_error_threshold")
        private double labelErrorThreshold = 0.25;

        @JsonProperty("high_signal_min_models")
        private int highSignalMinModels = 3;

        @JsonProperty("very_conservative_min_models")
        private int veryConservativeMinModels = 4;

        public double getLabelErrorThreshold() { return labelErrorThreshold; }
        public int getHighSignalMinModels() { return highSignalMinModels; }
        public int getVeryConservativeMinModels() { return veryConservativeMinModels; }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Evaluation {
        @JsonProperty("confidence_level")
        private double confidenceLevel = 0.95;

        @JsonProperty("ci_method")
        private String ciMethod = "wilson";

        @JsonProperty("min_samples_for_inclusion")
        private int minSamplesForInclusion = 30;

        public double getConfidenceLevel() { return confidenceLevel; }
        public String getCiMethod() { return ciMethod; }
        public int getMinSamplesForInclusion() { return minSamplesForInclusion; }
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class Category {
        @JsonProperty("id")
        private String id;

        @JsonProperty("name")
        private String name;

        @JsonProperty("description")
        private String description;

        public String getId() { return id; }
        public String getName() { return name; }
        public String getDescription() { return description; }
    }

    // Static loaders

    /**
     * Load config from default path or environment variable override.
     */
    public static EdgeCaseConfig load() throws IOException {
        String configPath = System.getenv(ENV_CONFIG_PATH);
        if (configPath == null || configPath.isEmpty()) {
            configPath = DEFAULT_CONFIG_PATH;
        }
        return load(Paths.get(configPath));
    }

    /**
     * Load config from specified path.
     */
    public static EdgeCaseConfig load(Path path) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        return mapper.readValue(path.toFile(), EdgeCaseConfig.class);
    }

    /**
     * Load config from classpath resource.
     */
    public static EdgeCaseConfig loadFromClasspath(String resourcePath) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        try (InputStream is = EdgeCaseConfig.class.getClassLoader().getResourceAsStream(resourcePath)) {
            if (is == null) {
                throw new IOException("Resource not found: " + resourcePath);
            }
            return mapper.readValue(is, EdgeCaseConfig.class);
        }
    }

    /**
     * Create default config (useful for testing).
     */
    public static EdgeCaseConfig defaults() {
        EdgeCaseConfig config = new EdgeCaseConfig();
        config.version = "1.1.0";
        config.directories = new Directories();
        config.edgeCaseDefinition = new EdgeCaseDefinition();
        config.sampling = new Sampling();
        config.filtering = new Filtering();
        config.evaluation = new Evaluation();
        config.algorithms = List.of("svm", "naive_bayes", "random_forest", "logistic_regression");
        config.domains = List.of("imdb_50k", "amazon_polarity", "yelp");
        config.categories = List.of(); // Empty, would need to populate
        return config;
    }

    // Getters

    public String getVersion() { return version; }
    public Directories getDirectories() { return directories; }
    public EdgeCaseDefinition getEdgeCaseDefinition() {
        return edgeCaseDefinition != null ? edgeCaseDefinition : new EdgeCaseDefinition();
    }
    public Sampling getSampling() { return sampling; }
    public Filtering getFiltering() { return filtering; }
    public Evaluation getEvaluation() { return evaluation; }
    public List<Category> getCategories() { return categories; }
    public List<String> getAlgorithms() { return algorithms; }
    public List<String> getDomains() { return domains; }

    /**
     * Convenience accessor for edge case threshold.
     * An edge case requires this many models to have failed on the same text.
     */
    public int getMinModelsForEdgeCase() {
        return getEdgeCaseDefinition().getMinModelsFailed();
    }

    /**
     * Compute categorization sample size from config parameters.
     * Formula: min_samples_per_category * num_categories * distribution_buffer
     */
    public int getCategorizationSampleSize() {
        int numCategories = categories != null ? categories.size() : 4;
        return (int) (getSampling().getMinSamplesPerCategory()
                    * numCategories
                    * getSampling().getDistributionBuffer());
    }

    /**
     * Get category IDs as array (convenience for iteration).
     */
    public String[] getCategoryIds() {
        if (categories == null) return new String[0];
        return categories.stream().map(Category::getId).toArray(String[]::new);
    }

    /**
     * Validate config has required fields.
     */
    public void validate() {
        if (directories == null) throw new IllegalStateException("Missing 'directories' in config");
        if (sampling == null) throw new IllegalStateException("Missing 'sampling' in config");
        if (evaluation == null) throw new IllegalStateException("Missing 'evaluation' in config");
        if (categories == null || categories.isEmpty()) {
            throw new IllegalStateException("Missing or empty 'categories' in config");
        }
        if (algorithms == null || algorithms.isEmpty()) {
            throw new IllegalStateException("Missing or empty 'algorithms' in config");
        }
        if (domains == null || domains.isEmpty()) {
            throw new IllegalStateException("Missing or empty 'domains' in config");
        }
    }

    @Override
    public String toString() {
        return String.format("EdgeCaseConfig[version=%s, edge_case_min_models=%d, categories=%d, algorithms=%d, domains=%d]",
                version,
                getMinModelsForEdgeCase(),
                categories != null ? categories.size() : 0,
                algorithms != null ? algorithms.size() : 0,
                domains != null ? domains.size() : 0);
    }

    /**
     * CLI entry point for testing config loading.
     */
    public static void main(String[] args) {
        try {
            EdgeCaseConfig config = EdgeCaseConfig.load();
            config.validate();
            System.out.println("Config loaded successfully: " + config);
            System.out.println("  Edge cases dir: " + config.getDirectories().getEdgeCasesDir());
            System.out.println("  Edge case threshold: " + config.getMinModelsForEdgeCase() + "+ models failed");
            System.out.println("  Edge case rationale: " + config.getEdgeCaseDefinition().getRationale());
            System.out.println("  Min samples/category: " + config.getSampling().getMinSamplesPerCategory());
            System.out.println("  Distribution buffer: " + config.getSampling().getDistributionBuffer());
            System.out.println("  Categorization sample size: " + config.getCategorizationSampleSize());
            System.out.println("  Confidence level: " + config.getEvaluation().getConfidenceLevel());
            System.out.println("  Categories: " + String.join(", ", config.getCategoryIds()));
            System.out.println("  Algorithms: " + String.join(", ", config.getAlgorithms()));
            System.out.println("  Domains: " + String.join(", ", config.getDomains()));
        } catch (IOException e) {
            System.err.println("Failed to load config: " + e.getMessage());
            System.exit(1);
        }
    }
}
