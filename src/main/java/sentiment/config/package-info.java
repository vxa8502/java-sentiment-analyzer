/**
 * Spring configuration classes for the Sentiment Analyzer.
 *
 * <h2>Package Contents</h2>
 * <ul>
 *   <li><b>Configuration</b>: Bean definitions ({@code *Configuration})</li>
 *   <li><b>Properties</b>: Externalized config ({@code *Properties})</li>
 * </ul>
 *
 * <h2>Properties Classes</h2>
 * <p>All {@code *Properties} classes use:
 * <ul>
 *   <li>JavaBean-style accessors: {@code getFieldName()}, {@code setFieldName()}</li>
 *   <li>{@code @ConfigurationProperties} for type-safe binding</li>
 *   <li>Sensible defaults that can be overridden via environment variables</li>
 * </ul>
 *
 * <h3>Property Naming Convention</h3>
 * <p>Properties follow the pattern: {@code sentiment.<module>.<setting>}
 * <ul>
 *   <li>{@code sentiment.api.*} - API configuration ({@link ApiProperties})</li>
 *   <li>{@code sentiment.models.*} - Model paths ({@link ModelPathProperties})</li>
 *   <li>{@code sentiment.features.*} - Feature extraction ({@link FeatureExtractionProperties})</li>
 *   <li>{@code sentiment.drift.*} - Drift detection ({@link DriftDetectionProperties})</li>
 * </ul>
 *
 * @see SentimentConfiguration Main application configuration
 * @see ApiProperties API-specific settings
 */
package sentiment.config;
