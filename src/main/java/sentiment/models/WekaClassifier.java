package sentiment.models;

import weka.classifiers.Classifier;
import java.util.concurrent.Callable;

/**
 * Extended interface for Weka-based sentiment classifiers.
 * Provides access to underlying Weka classifier for feature importance analysis
 * and thread-safe inference execution.
 */
public interface WekaClassifier extends SentimentClassifier {

    /** Gets underlying Weka classifier (for feature importance analysis). */
    Classifier getWekaClassifier();

    /**
     * Executes inference task with thread-safe read lock.
     *
     * @param task inference task
     * @param <R> return type
     * @return task result
     */
    <R> R executeInference(Callable<R> task) throws Exception;
}
