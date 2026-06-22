package sentiment.config;

import org.springframework.cache.CacheManager;
import org.springframework.cache.annotation.EnableCaching;
import org.springframework.cache.concurrent.ConcurrentMapCacheManager;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Cache configuration for the sentiment analysis application.
 * <p>
 * Uses a simple in-memory cache for feature importance results.
 * The cache is cleared on model reload or application restart.
 */
@Configuration
@EnableCaching
public class CacheConfiguration {

    /**
     * Cache names used in the application.
     */
    public static final String FEATURE_IMPORTANCE_CACHE = "featureImportance";

    @Bean
    public CacheManager cacheManager() {
        return new ConcurrentMapCacheManager(FEATURE_IMPORTANCE_CACHE);
    }
}
