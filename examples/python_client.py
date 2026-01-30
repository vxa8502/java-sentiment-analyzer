#!/usr/bin/env python3
"""
Java Sentiment Analyzer - Python Client Example

This script demonstrates how to interact with the Sentiment Analysis API
using Python's requests library.

Requirements:
    pip install requests

Usage:
    python python_client.py
"""

import os
from typing import Any, Dict, List

import requests


class SentimentAnalyzerClient:
    """Client for Java Sentiment Analyzer API"""

    def __init__(self, base_url: str = "https://java-sentiment-api.onrender.com/api/v1"):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the API (default: https://java-sentiment-api.onrender.com/api/v1)
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health_check(self) -> Dict[str, Any]:
        """
        Check API and model health.

        Returns:
            Dictionary with status, version, modelLoaded, modelType,
            supportedLabels, uptimeMs, and productionMetrics
        """
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def analyze(self, text: str, confidence_threshold: float = None) -> Dict[str, Any]:
        """
        Analyze sentiment for a single text.

        Args:
            text: Text to analyze (1-10000 characters)
            confidence_threshold: Optional minimum confidence (0.0-1.0).
                                  Below threshold returns 'uncertain'.

        Returns:
            Dictionary with sentiment, confidence, text, processingTimeMs,
            and optional warning field for low-confidence predictions
        """
        payload = {"text": text}
        if confidence_threshold is not None:
            payload["confidenceThreshold"] = confidence_threshold

        response = self.session.post(f"{self.base_url}/sentiment/analyze", json=payload)
        response.raise_for_status()
        return response.json()

    def analyze_batch(
        self, texts: List[str], confidence_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Analyze sentiment for multiple texts in parallel.

        Args:
            texts: List of texts to analyze (1-100 texts, 1-10000 chars each)
            confidence_threshold: Optional minimum confidence (0.0-1.0)

        Returns:
            Dictionary with results array, totalProcessed, successCount,
            errorCount, and totalProcessingTimeMs
        """
        payload = {"texts": texts}
        if confidence_threshold is not None:
            payload["confidenceThreshold"] = confidence_threshold

        response = self.session.post(f"{self.base_url}/sentiment/batch", json=payload)
        response.raise_for_status()
        return response.json()

    def get_feature_importance(self, top_features: int = 30) -> Dict[str, Any]:
        """
        Get feature importance analysis from the model.

        Args:
            top_features: Number of top features to return (default: 30)

        Returns:
            Dictionary with modelType, totalFeatures, topFeatures array,
            statistics, analysisTimeMs, and note
        """
        response = self.session.get(
            f"{self.base_url}/model/feature-importance",
            params={"topFeatures": top_features}
        )
        response.raise_for_status()
        return response.json()


def print_result(result: Dict[str, Any], indent: str = "") -> None:
    """Print a sentiment result, including warning if present."""
    print(f"{indent}Sentiment: {result['sentiment']}")
    print(f"{indent}Confidence: {result['confidence']:.2%}")
    print(f"{indent}Processing Time: {result['processingTimeMs']}ms")
    if result.get("warning"):
        print(f"{indent}Warning: {result['warning']}")


def main():
    """Main function demonstrating API usage"""

    base_url = os.getenv("SENTIMENT_API_URL", "https://java-sentiment-api.onrender.com/api/v1")
    client = SentimentAnalyzerClient(base_url=base_url)

    print("=" * 60)
    print("Java Sentiment Analyzer - Python Client Example")
    print("=" * 60)
    print()

    # 1. Health check
    print("1. Health Check")
    print("-" * 60)
    try:
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Version: {health['version']}")
        print(f"Model Loaded: {health['modelLoaded']}")
        print(f"Model Type: {health['modelType']}")
        print(f"Supported Labels: {health['supportedLabels']}")
        print(f"Uptime: {health['uptimeMs']}ms")
        if health.get("productionMetrics"):
            metrics = health["productionMetrics"]
            print(f"Total Predictions: {metrics['totalPredictions']}")
        print()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        print("Make sure the API is running on port 8080")
        return

    # 2. Positive sentiment
    print("2. Positive Sentiment Analysis")
    print("-" * 60)
    result = client.analyze("This product is absolutely amazing! Best purchase ever!")
    print(f"Text: {result['text']}")
    print_result(result)
    print()

    # 3. Negative sentiment
    print("3. Negative Sentiment Analysis")
    print("-" * 60)
    result = client.analyze("Terrible quality. Complete waste of money. Very disappointed.")
    print(f"Text: {result['text']}")
    print_result(result)
    print()

    # 4. Low confidence prediction (demonstrates warning field)
    print("4. Low Confidence Prediction")
    print("-" * 60)
    result = client.analyze("It arrived.")
    print(f"Text: {result['text']}")
    print_result(result)
    print()

    # 5. Custom confidence threshold
    print("5. Custom Confidence Threshold (0.9)")
    print("-" * 60)
    result = client.analyze("Pretty good product overall.", confidence_threshold=0.9)
    print(f"Text: {result['text']}")
    print_result(result)
    print()

    # 6. Batch analysis
    print("6. Batch Analysis")
    print("-" * 60)
    texts = [
        "Amazing product, highly recommend!",
        "Terrible experience, very disappointed.",
        "It works as expected, nothing special.",
        "Best purchase this year!",
        "Complete waste of money.",
    ]
    batch_result = client.analyze_batch(texts)

    print(f"Total Processed: {batch_result['totalProcessed']}")
    print(f"Success: {batch_result['successCount']}")
    print(f"Errors: {batch_result['errorCount']}")
    print(f"Total Time: {batch_result['totalProcessingTimeMs']}ms")

    for i, result in enumerate(batch_result["results"], 1):
        print(f"\n  [{i}] {result['text'][:40]}...")
        print_result(result, indent="      ")
    print()

    # 7. Feature importance
    print("7. Feature Importance (Top 10)")
    print("-" * 60)
    try:
        features = client.get_feature_importance(top_features=10)
        print(f"Model Type: {features['modelType']}")
        print(f"Total Features: {features['totalFeatures']}")
        print(f"Analysis Time: {features['analysisTimeMs']}ms")
        print()
        print("Top Features:")
        for f in features["topFeatures"]:
            direction = "+" if f["direction"] == "positive" else "-"
            print(f"  {direction} {f['feature']}: {f['weight']:.4f}")
        print()
        if features.get("statistics"):
            stats = features["statistics"]
            print(f"Statistics: mean={stats['mean']:.4f}, stdDev={stats['stdDev']:.4f}")
    except requests.exceptions.HTTPError as e:
        print(f"Feature importance unavailable: {e.response.status_code}")
        if e.response.status_code == 404:
            print("(Model may not support feature extraction)")
    print()

    # 8. Error handling
    print("8. Error Handling (Invalid Input)")
    print("-" * 60)
    try:
        client.analyze("")
    except requests.exceptions.HTTPError as e:
        print(f"Status: {e.response.status_code}")
        error_data = e.response.json()
        print(f"Error: {error_data.get('error')}")
        if error_data.get("message"):
            print(f"Message: {error_data['message']}")
        if error_data.get("details"):
            print("Details:")
            for field, message in error_data["details"].items():
                print(f"  - {field}: {message}")
    print()

    # 9. Sentiment distribution
    print("9. Sentiment Distribution Analysis")
    print("-" * 60)
    reviews = [
        "Outstanding quality and service!",
        "Product broke after one day.",
        "Decent product for the price.",
        "Absolutely love it! Five stars!",
        "Worst purchase ever made.",
        "Good value, works as advertised.",
        "Terrible customer service.",
        "Highly recommend to everyone!",
        "Not worth the money.",
        "Exceeded all my expectations!",
    ]

    batch_result = client.analyze_batch(reviews)

    # Defensive counting - handles any sentiment value
    sentiments: Dict[str, int] = {}
    total_confidence = 0.0

    for result in batch_result["results"]:
        label = result.get("sentiment", "unknown")
        sentiments[label] = sentiments.get(label, 0) + 1
        total_confidence += result.get("confidence", 0)

    count = len(reviews)
    print(f"Total Analyzed: {count}")
    for label, num in sorted(sentiments.items()):
        print(f"  {label.capitalize()}: {num} ({num/count:.1%})")
    print(f"Average Confidence: {total_confidence/count:.2%}")
    print(f"Processing Time: {batch_result['totalProcessingTimeMs']}ms")
    print()

    print("=" * 60)
    print("Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
