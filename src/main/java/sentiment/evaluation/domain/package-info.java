/**
 * Domain value objects for feature importance analysis.
 *
 * <p>This package contains immutable data types shared across all layers:
 * <ul>
 * <li>{@link sentiment.evaluation.domain.FeatureWeight} - Individual feature importance</li>
 * <li>{@link sentiment.evaluation.domain.FeatureStatistics} - Statistical summary</li>
 * <li>{@link sentiment.evaluation.domain.FeatureImportanceResult} - Complete analysis result</li>
 * </ul>
 *
 * <p><b>Design Principles:</b>
 * <ul>
 * <li>Immutability - All types are Java records (immutable by default)</li>
 * <li>Serialization-ready - Jackson annotations for JSON persistence</li>
 * <li>Single source of truth - No duplicate type definitions</li>
 * <li>Layer-independent - Used by domain, persistence, and API layers</li>
 * </ul>
 *
 * <p><b>Usage Example:</b>
 * <pre>{@code
 * FeatureImportanceAnalyzer analyzer = FeatureImportanceAnalyzer.forClassifier(classifier);
 * FeatureImportanceResult result = analyzer.analyze(instances, classifier, 100);
 *
 * // Use directly in API responses
 * return FeatureImportanceResponse.fromDomain(result);
 *
 * // Serialize to JSON
 * FeatureImportancePersistence.save(result, outputPath);
 * }</pre>
 */
package sentiment.evaluation.domain;
