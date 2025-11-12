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

import requests
import json
import os
from typing import List, Dict, Optional


class SentimentAnalyzerClient:
    """Client for Java Sentiment Analyzer API"""

    def __init__(self, base_url: str = "http://localhost:8080/api/v1"):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the API (default: http://localhost:8080/api/v1)
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def analyze(
        self, text: str, confidence_threshold: float = 0.7
    ) -> Dict[str, any]:
        """
        Analyze sentiment for a single text.

        Args:
            text: Text to analyze (1-10000 characters)
            confidence_threshold: Minimum confidence for classification (0.0-1.0)

        Returns:
            Dictionary with sentiment, confidence, text, and processing time

        Raises:
            requests.HTTPError: If API returns an error status
        """
        endpoint = f"{self.base_url}/sentiment/analyze"
        payload = {"text": text, "confidenceThreshold": confidence_threshold}

        response = self.session.post(endpoint, json=payload)
        response.raise_for_status()

        return response.json()

    def analyze_batch(
        self, texts: List[str], confidence_threshold: float = 0.7
    ) -> Dict[str, any]:
        """
        Analyze sentiment for multiple texts in parallel.

        Args:
            texts: List of texts to analyze (max 100 texts, 1-10000 characters each)
            confidence_threshold: Minimum confidence for classification (0.0-1.0)

        Returns:
            Dictionary with results array and total processing time

        Raises:
            requests.HTTPError: If API returns an error status
        """
        endpoint = f"{self.base_url}/sentiment/batch"
        payload = {"texts": texts, "confidenceThreshold": confidence_threshold}

        response = self.session.post(endpoint, json=payload)
        response.raise_for_status()

        return response.json()

    def health_check(self) -> Dict[str, any]:
        """
        Check API and model health.

        Returns:
            Dictionary with status, modelLoaded, algorithmType, timestamp, version

        Raises:
            requests.HTTPError: If API returns an error status
        """
        endpoint = f"{self.base_url}/health"
        response = self.session.get(endpoint)
        response.raise_for_status()

        return response.json()


def main():
    """Main function demonstrating API usage"""

    # Initialize client
    base_url = os.getenv("SENTIMENT_API_URL", "http://localhost:8080/api/v1")
    client = SentimentAnalyzerClient(base_url=base_url)

    print("=" * 60)
    print("Java Sentiment Analyzer - Python Client Example")
    print("=" * 60)
    print()

    # Example 1: Health check
    print("1. Health Check")
    print("-" * 60)
    try:
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Model Loaded: {health['modelLoaded']}")
        print(f"Algorithm: {health['algorithmType']}")
        print(f"Version: {health['version']}")
        print()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        print("Make sure the API is running on port 8080")
        return

    # Example 2: Positive sentiment
    print("2. Positive Sentiment Analysis")
    print("-" * 60)
    text = "This product is absolutely amazing! Best purchase ever!"
    result = client.analyze(text)
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Processing Time: {result['processingTimeMs']}ms")
    print()

    # Example 3: Negative sentiment
    print("3. Negative Sentiment Analysis")
    print("-" * 60)
    text = "Terrible quality. Complete waste of money. Very disappointed."
    result = client.analyze(text)
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Processing Time: {result['processingTimeMs']}ms")
    print()

    # Example 4: Custom confidence threshold
    print("4. Custom Confidence Threshold (0.9)")
    print("-" * 60)
    text = "Pretty good product overall."
    result = client.analyze(text, confidence_threshold=0.9)
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Processing Time: {result['processingTimeMs']}ms")
    print()

    # Example 5: Batch analysis
    print("5. Batch Analysis")
    print("-" * 60)
    texts = [
        "Amazing product, highly recommend!",
        "Terrible experience, very disappointed.",
        "It works as expected, nothing special.",
        "Best purchase this year!",
        "Complete waste of money.",
    ]
    batch_result = client.analyze_batch(texts)

    for i, result in enumerate(batch_result["results"], 1):
        print(f"\nText {i}: {result['text']}")
        print(f"  Sentiment: {result['sentiment']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Processing Time: {result['processingTimeMs']}ms")

    print(f"\nTotal Processing Time: {batch_result['totalProcessingTimeMs']}ms")
    print()

    # Example 6: Error handling
    print("6. Error Handling (Invalid Input)")
    print("-" * 60)
    try:
        # This should fail validation (empty text)
        result = client.analyze("")
    except requests.exceptions.HTTPError as e:
        print(f"Expected error caught: {e.response.status_code}")
        error_data = e.response.json()
        print(f"Error message: {error_data.get('message')}")
        if "validationErrors" in error_data:
            for error in error_data["validationErrors"]:
                print(f"  - Field '{error['field']}': {error['message']}")
    print()

    # Example 7: Sentiment distribution
    print("7. Sentiment Distribution Analysis")
    print("-" * 60)
    sample_reviews = [
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

    batch_result = client.analyze_batch(sample_reviews)

    sentiments = {"positive": 0, "negative": 0, "uncertain": 0}
    total_confidence = 0

    for result in batch_result["results"]:
        sentiments[result["sentiment"]] += 1
        total_confidence += result["confidence"]

    print(f"Total reviews analyzed: {len(sample_reviews)}")
    print(f"Positive: {sentiments['positive']} ({sentiments['positive']/len(sample_reviews):.1%})")
    print(f"Negative: {sentiments['negative']} ({sentiments['negative']/len(sample_reviews):.1%})")
    print(f"Uncertain: {sentiments['uncertain']} ({sentiments['uncertain']/len(sample_reviews):.1%})")
    print(f"Average confidence: {total_confidence/len(sample_reviews):.2%}")
    print(f"Total processing time: {batch_result['totalProcessingTimeMs']}ms")
    print()

    print("=" * 60)
    print("Examples Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
