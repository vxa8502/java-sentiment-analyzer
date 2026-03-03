package sentiment.api;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * Root endpoint providing API landing page for browser visitors.
 */
@RestController
@org.springframework.context.annotation.Profile("!training")
public class RootController {

    private record LandingResponse(
        String service,
        String status,
        String version,
        String description,
        String documentation
    ) {}

    @GetMapping("/")
    public ResponseEntity<LandingResponse> root() {
        return ResponseEntity.ok(new LandingResponse(
            "Java Sentiment Analyzer",
            "live",
            "1.0.0",
            "ML-powered sentiment analysis API with negation handling",
            "/api/v1/health"
        ));
    }
}
