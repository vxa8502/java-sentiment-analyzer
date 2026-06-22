package sentiment.api;

import io.github.resilience4j.ratelimiter.annotation.RateLimiter;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.beans.factory.annotation.Qualifier;
import sentiment.config.ApiProperties;
import sentiment.monitoring.DriftDetector;
import sentiment.monitoring.DriftResult;
import sentiment.monitoring.PredictionMetrics;
import sentiment.models.SentimentClassifier;

import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * REST API controller for sentiment analysis operations.
 */
@RestController
@RequestMapping("/api/v1")
@org.springframework.context.annotation.Profile("!training")
public class SentimentController {

    private static final Logger logger = LoggerFactory.getLogger(SentimentController.class);

    // Pre-compiled patterns for IP validation (avoids regex compilation per request)
    private static final Pattern IPV6_PATTERN = Pattern.compile("^[0-9a-fA-F:.]+$");
    private static final Pattern IPV4_PATTERN = Pattern.compile("^\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}$");

    /**
     * Default trusted proxy CIDRs for common cloud providers and local development.
     * These are only used when trust-proxy-headers is enabled.
     */
    private static final Set<String> DEFAULT_TRUSTED_PROXIES = Set.of(
            "127.0.0.1",      // localhost
            "10.",            // Private Class A (prefix match)
            "172.16.",        // Private Class B (prefix match)
            "172.17.",        // Docker default
            "172.18.",
            "172.19.",
            "172.20.",
            "172.21.",
            "172.22.",
            "172.23.",
            "172.24.",
            "172.25.",
            "172.26.",
            "172.27.",
            "172.28.",
            "172.29.",
            "172.30.",
            "172.31.",
            "192.168."        // Private Class C (prefix match)
    );

    private final SentimentService sentimentService;
    private final FeatureImportanceService featureImportanceService;
    private final ApiProperties apiProperties;
    private final String loadedModelPath;
    private final long startTime;
    private final boolean trustProxyHeaders;
    private final Set<String> trustedProxies;

    // Performance: Cache drift detector reference to avoid ObjectProvider lookup per request
    private final DriftDetector driftDetector;

    public SentimentController(SentimentService sentimentService,
                               FeatureImportanceService featureImportanceService,
                               ApiProperties apiProperties,
                               @Qualifier("loadedModelPath") String loadedModelPath,
                               ObjectProvider<DriftDetector> driftDetectorProvider) {
        this.sentimentService = sentimentService;
        this.featureImportanceService = featureImportanceService;
        this.apiProperties = apiProperties;
        this.loadedModelPath = loadedModelPath;
        this.trustProxyHeaders = apiProperties.isTrustProxyHeaders();
        this.trustedProxies = parseTrustedProxies(apiProperties.getTrustedProxies());
        this.startTime = System.currentTimeMillis();

        // Performance: Cache drift detector once at startup (availability doesn't change at runtime)
        this.driftDetector = driftDetectorProvider.getIfAvailable();

        SentimentClassifier classifier = sentimentService.getClassifier();
        String algorithmName = Optional.ofNullable(classifier)
                .map(SentimentClassifier::getAlgorithmName)
                .orElse("unknown");
        logger.info("SentimentController initialized with {} classifier (model: {}), production metrics, and drift detection={}",
                   algorithmName, loadedModelPath, driftDetector != null);
    }

    /**
     * Analyzes sentiment for a single text input.
     */
    @PostMapping("/sentiment/analyze")
    @RateLimiter(name = "sentimentApi")
    public ResponseEntity<SentimentResponse> analyzeSentiment(
            @Valid @RequestBody SentimentRequest request,
            HttpServletRequest httpRequest) {

        // Performance: Only extract IP and log if debug enabled (avoids work on hot path)
        if (logger.isDebugEnabled()) {
            String clientIp = extractClientIp(httpRequest);
            logger.debug("REQUEST: client={}, textLength={}", clientIp, request.text().length());
        }

        ResponseEntity<SentimentResponse> response = sentimentService.classifyText(
            request.text(), request.confidenceThreshold());

        // Performance: Only log response details if debug enabled
        if (logger.isDebugEnabled()) {
            SentimentResponse body = response.getBody();
            if (body != null) {
                String clientIp = extractClientIp(httpRequest);
                logger.debug("RESPONSE: client={}, sentiment={}, time={}ms",
                            clientIp, body.sentiment(), body.processingTimeMs());
            }
        }

        return response;
    }


    /**
     * Analyzes sentiment for multiple texts in batch using parallel processing.
     * Results maintain input order despite concurrent execution.
     *
     * <p>Performance optimizations:
     * <ul>
     *   <li>Pre-allocates result array for O(1) index access (avoids List.get() overhead)</li>
     *   <li>Uses parallel streams only for batches larger than threshold</li>
     *   <li>Stores results directly in array slots (avoids post-processing sort)</li>
     * </ul>
     */
    @PostMapping("/sentiment/batch")
    @RateLimiter(name = "batchApi")
    public ResponseEntity<BatchResponse> analyzeBatch(
            @Valid @RequestBody BatchRequest request) {

        long batchStartTime = System.currentTimeMillis();
        List<String> texts = request.texts();
        int batchSize = texts.size();

        if (logger.isDebugEnabled()) {
            logger.debug("Batch classification request: {} texts", batchSize);
        }

        // Pre-allocate array for O(1) indexed access (avoids List.get() overhead for LinkedList)
        // and direct slot assignment (avoids post-processing sort)
        SentimentResponse[] resultsArray = new SentimentResponse[batchSize];
        Double confidenceThreshold = request.confidenceThreshold();

        // Use parallel processing only for larger batches (avoids thread pool overhead for small batches)
        java.util.stream.IntStream indexStream = java.util.stream.IntStream.range(0, batchSize);
        if (batchSize >= apiProperties.getParallelBatchThreshold()) {
            indexStream = indexStream.parallel();
        }

        // Process and store results directly in array slots (order preserved without sort)
        indexStream.forEach(i -> {
            ResponseEntity<SentimentResponse> result = sentimentService.classifyText(texts.get(i), confidenceThreshold);
            resultsArray[i] = result.getBody();
        });

        // Convert array to list for response
        List<SentimentResponse> results = java.util.Arrays.asList(resultsArray);

        long totalProcessingTime = System.currentTimeMillis() - batchStartTime;
        BatchResponse response = BatchResponse.fromResults(results, totalProcessingTime);

        if (logger.isDebugEnabled()) {
            logger.debug("Batch completed: {} success, {} errors in {}ms",
                       response.successCount(), response.errorCount(), totalProcessingTime);
        }

        return ResponseEntity.ok(response);
    }


    /**
     * Returns feature importance analysis.
     * For SVM models, extracts coefficients directly from the trained model.
     * Results are cached automatically via Spring's @Cacheable.
     *
     * @param topFeatures Number of top features to return (default: 30, max: 1000)
     */
    @GetMapping("/model/feature-importance")
    public ResponseEntity<FeatureImportanceResponse> getFeatureImportance(
            @RequestParam(defaultValue = "30") int topFeatures) {

        logger.info("Feature importance request: topFeatures={}", topFeatures);

        // Validate topFeatures parameter (with upper bound to prevent resource exhaustion)
        int maxFeatures = apiProperties.getMaxTopFeatures();
        if (topFeatures < 1 || topFeatures > maxFeatures) {
            return ResponseEntity.badRequest()
                    .body(FeatureImportanceResponse.error(
                            String.format("Invalid topFeatures value: must be between 1 and %d", maxFeatures)));
        }

        SentimentClassifier classifier = sentimentService.getClassifier();
        if (classifier == null || !classifier.isTrained()) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(FeatureImportanceResponse.error(
                            "Model not trained yet. Check /api/v1/health for status."));
        }

        // Use cached service - caching handled by @Cacheable
        return featureImportanceService.getFeatureImportance()
                .map(response -> ResponseEntity.ok(response.withTopFeatures(topFeatures)))
                .orElseGet(() -> ResponseEntity.status(HttpStatus.NOT_FOUND)
                        .body(FeatureImportanceResponse.error(
                                "Feature importance data not found and could not be extracted from model. " +
                                "Please re-train the model with --show-feature-importance flag.")));
    }

    /**
     * Health check endpoint returning service status, model information, and production metrics.
     */
    @GetMapping("/health")
    public ResponseEntity<HealthResponse> health() {
        long uptime = System.currentTimeMillis() - startTime;

        SentimentClassifier classifier = sentimentService.getClassifier();
        PredictionMetrics metrics = sentimentService.getMetrics();

        // Get production metrics snapshot
        PredictionMetrics.MetricsSnapshot snapshot = metrics.getSnapshot();

        // Get supported labels from classifier (may be subset of all possible labels)
        List<String> supportedLabels = classifier.isTrained()
                ? List.of(classifier.getSupportedClasses())
                : List.of();

        HealthResponse.ProductionMetrics productionMetrics = new HealthResponse.ProductionMetrics(
                snapshot.totalPredictions(),
                new HealthResponse.LabelDistribution(
                        snapshot.positivePredictions(),
                        snapshot.negativePredictions(),
                        snapshot.uncertainPredictions()
                ),
                snapshot.averageConfidence(),
                snapshot.lowConfidenceRatePercent(),
                new HealthResponse.LatencyStats(
                        snapshot.meanLatencyMs(),
                        snapshot.p95LatencyMs(),
                        snapshot.p99LatencyMs()
                )
        );

        // Get drift info if available (using cached reference - no ObjectProvider lookup)
        HealthResponse.DriftInfo driftInfo = null;
        if (driftDetector != null) {
            DriftResult driftResult = driftDetector.getLatestResult();
            driftInfo = new HealthResponse.DriftInfo(
                    driftResult.status().name(),
                    driftResult.maxPsi(),
                    driftResult.status() != DriftResult.DriftStatus.NOT_READY
            );
        }

        HealthResponse response = HealthResponse.withMetricsAndDrift(
            "1.0.0",
            classifier.isTrained(),
            classifier.isTrained() ? classifier.getAlgorithmName() : "Not loaded",
            supportedLabels,
            uptime,
            productionMetrics,
            driftInfo
        );

        logger.debug("Health check: model loaded={}, uptime={}ms, predictions={}",
                    classifier.isTrained(), uptime, snapshot.totalPredictions());

        return ResponseEntity.ok(response);
    }

    /**
     * Extracts client IP from request with security validation.
     *
     * <p>Only trusts proxy headers (X-Forwarded-For, X-Real-IP) when:
     * <ul>
     *   <li>trust-proxy-headers is explicitly enabled in configuration</li>
     *   <li>The direct connection (remoteAddr) comes from a trusted proxy</li>
     * </ul>
     *
     * <p>This prevents IP spoofing attacks where clients send fake X-Forwarded-For
     * headers to bypass rate limiting or impersonate other users.
     *
     * @param request the HTTP request
     * @return the client IP address (or "unknown" if not determinable)
     */
    private String extractClientIp(HttpServletRequest request) {
        String remoteAddr = request.getRemoteAddr();

        // Only trust proxy headers if explicitly enabled AND request comes from trusted proxy
        if (trustProxyHeaders && isTrustedProxy(remoteAddr)) {
            // Check X-Forwarded-For first (standard header)
            String xForwardedFor = request.getHeader("X-Forwarded-For");
            if (xForwardedFor != null && !xForwardedFor.isEmpty() && !"unknown".equalsIgnoreCase(xForwardedFor)) {
                // X-Forwarded-For format: client, proxy1, proxy2, ...
                // The leftmost IP is the original client
                // Performance: Use indexOf instead of split() to avoid array allocation
                int commaIndex = xForwardedFor.indexOf(',');
                String clientIp = (commaIndex > 0 ? xForwardedFor.substring(0, commaIndex) : xForwardedFor).trim();
                if (isValidIpAddress(clientIp)) {
                    return clientIp;
                }
            }

            // Fall back to X-Real-IP (nginx convention)
            String xRealIp = request.getHeader("X-Real-IP");
            if (xRealIp != null && !xRealIp.isEmpty() && !"unknown".equalsIgnoreCase(xRealIp)) {
                // Performance: Only call trim() once, store result
                String trimmedIp = xRealIp.trim();
                if (isValidIpAddress(trimmedIp)) {
                    return trimmedIp;
                }
            }
        }

        // Default: use direct connection address (cannot be spoofed)
        return remoteAddr != null ? remoteAddr : "unknown";
    }

    /**
     * Checks if the given IP address belongs to a trusted proxy.
     */
    private boolean isTrustedProxy(String ip) {
        if (ip == null) {
            return false;
        }

        // Check custom trusted proxies first
        if (trustedProxies.contains(ip)) {
            return true;
        }

        // Check default trusted proxy prefixes (private networks)
        for (String prefix : DEFAULT_TRUSTED_PROXIES) {
            if (ip.startsWith(prefix)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Validates that a string is a valid IP address format (IPv4 or IPv6).
     * Prevents injection of malicious values via proxy headers.
     *
     * <p>SECURITY: Validates IPv4 octet ranges (0-255) to prevent invalid IPs
     * like "999.999.999.999" from passing validation.
     *
     * <p>Performance: Uses pre-compiled patterns for format check, then validates octets.
     */
    private boolean isValidIpAddress(String ip) {
        if (ip == null || ip.isEmpty()) {
            return false;
        }

        // Basic validation: IPv4 format (x.x.x.x) or IPv6 format (contains :)
        // More strict validation could use InetAddress.getByName() but that does DNS lookup
        if (ip.contains(":")) {
            // IPv6: must only contain hex digits, colons, and possibly dots for mapped IPv4
            return ip.length() <= 45 && IPV6_PATTERN.matcher(ip).matches();
        } else {
            // IPv4: validate format AND octet ranges (0-255)
            return isValidIpv4Address(ip);
        }
    }

    /**
     * Validates IPv4 address format and octet ranges.
     *
     * <p>SECURITY: Each octet must be 0-255. Rejects invalid IPs like "999.999.999.999".
     *
     * @param ip the IP address string to validate
     * @return true if valid IPv4 address with valid octet ranges
     */
    private boolean isValidIpv4Address(String ip) {
        // First check format matches x.x.x.x pattern
        if (!IPV4_PATTERN.matcher(ip).matches()) {
            return false;
        }

        // Validate each octet is in range 0-255
        int start = 0;
        int octetCount = 0;

        for (int i = 0; i <= ip.length(); i++) {
            if (i == ip.length() || ip.charAt(i) == '.') {
                if (i == start) {
                    return false; // Empty octet
                }

                // Parse octet value
                int octet = 0;
                for (int j = start; j < i; j++) {
                    char c = ip.charAt(j);
                    if (c < '0' || c > '9') {
                        return false; // Non-digit character
                    }
                    octet = octet * 10 + (c - '0');
                    if (octet > 255) {
                        return false; // Octet exceeds 255
                    }
                }

                // Reject leading zeros (e.g., "01.02.03.04") except for "0" itself
                if (i - start > 1 && ip.charAt(start) == '0') {
                    return false;
                }

                octetCount++;
                start = i + 1;
            }
        }

        return octetCount == 4;
    }

    /**
     * Parses comma-separated trusted proxy configuration into a Set.
     */
    private static Set<String> parseTrustedProxies(String config) {
        if (config == null || config.isBlank()) {
            return Set.of();
        }
        return Set.of(config.split(","))
                .stream()
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .collect(java.util.stream.Collectors.toSet());
    }

}
