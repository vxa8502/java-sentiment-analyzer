#!/bin/bash

# Java Sentiment Analyzer - API Examples
# This script demonstrates various API endpoints and usage patterns

# Color output for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Base URL (modify if deployed elsewhere)
BASE_URL="${SENTIMENT_API_URL:-http://localhost:8080/api/v1}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Java Sentiment Analyzer - API Examples${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if API is running
echo -e "${YELLOW}0. Health Check${NC}"
echo "curl $BASE_URL/health"
curl -s $BASE_URL/health | jq '.'
echo ""
read -p "Press Enter to continue..."
echo ""

# Example 1: Positive sentiment
echo -e "${GREEN}1. Positive Sentiment Example${NC}"
echo "curl -X POST $BASE_URL/sentiment/analyze \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"text\":\"This product is absolutely amazing! Best purchase ever!\"}'"
echo ""
curl -X POST $BASE_URL/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"This product is absolutely amazing! Best purchase ever!"}' | jq '.'
echo ""
read -p "Press Enter to continue..."
echo ""

# Example 2: Negative sentiment
echo -e "${RED}2. Negative Sentiment Example${NC}"
echo "curl -X POST $BASE_URL/sentiment/analyze \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"text\":\"Terrible quality. Complete waste of money. Very disappointed.\"}'"
echo ""
curl -X POST $BASE_URL/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Terrible quality. Complete waste of money. Very disappointed."}' | jq '.'
echo ""
read -p "Press Enter to continue..."
echo ""

# Example 3: Neutral/Uncertain sentiment
echo -e "${YELLOW}3. Neutral/Uncertain Sentiment Example${NC}"
echo "curl -X POST $BASE_URL/sentiment/analyze \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"text\":\"It arrived on time.\"}'"
echo ""
curl -X POST $BASE_URL/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"It arrived on time."}' | jq '.'
echo ""
read -p "Press Enter to continue..."
echo ""

# Example 4: Custom confidence threshold
echo -e "${BLUE}4. Custom Confidence Threshold (0.9)${NC}"
echo "curl -X POST $BASE_URL/sentiment/analyze \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"text\":\"Pretty good product overall.\",\"confidenceThreshold\":0.9}'"
echo ""
curl -X POST $BASE_URL/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Pretty good product overall.","confidenceThreshold":0.9}' | jq '.'
echo ""
read -p "Press Enter to continue..."
echo ""

# Example 5: Batch analysis
echo -e "${GREEN}5. Batch Analysis (Multiple Texts)${NC}"
echo "curl -X POST $BASE_URL/sentiment/batch \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{...}'"
echo ""
curl -X POST $BASE_URL/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Amazing product, highly recommend!",
      "Terrible experience, very disappointed.",
      "It works as expected, nothing special.",
      "Best purchase this year!",
      "Complete waste of money."
    ]
  }' | jq '.'
echo ""
read -p "Press Enter to continue..."
echo ""

# Example 6: Long review text
echo -e "${BLUE}6. Long Product Review${NC}"
echo "curl -X POST $BASE_URL/sentiment/analyze \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"text\":\"<long review text>\"}'"
echo ""
curl -X POST $BASE_URL/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I recently purchased this product and I have to say, it exceeded all my expectations. The build quality is outstanding, the customer service was responsive and helpful, and the shipping was incredibly fast. I have been using it daily for the past month and have had zero issues. The features work exactly as advertised and the user interface is intuitive. I would highly recommend this to anyone looking for a reliable and high-quality product. Five stars!"
  }' | jq '.'
echo ""
read -p "Press Enter to continue..."
echo ""

# Example 7: Mixed sentiment
echo -e "${YELLOW}7. Mixed Sentiment (Positive and Negative Elements)${NC}"
echo "curl -X POST $BASE_URL/sentiment/analyze \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"text\":\"The product quality is great, but shipping took forever.\"}'"
echo ""
curl -X POST $BASE_URL/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"The product quality is great, but shipping took forever."}' | jq '.'
echo ""
read -p "Press Enter to continue..."
echo ""

# Example 8: Edge case - very short text
echo -e "${BLUE}8. Edge Case - Very Short Text${NC}"
echo "curl -X POST $BASE_URL/sentiment/analyze \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"text\":\"Good!\"}'"
echo ""
curl -X POST $BASE_URL/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"Good!"}' | jq '.'
echo ""
read -p "Press Enter to continue..."
echo ""

# Example 9: Validation error (missing text)
echo -e "${RED}9. Validation Error Example (Missing Text)${NC}"
echo "curl -X POST $BASE_URL/sentiment/analyze \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"confidenceThreshold\":0.7}'"
echo ""
curl -X POST $BASE_URL/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"confidenceThreshold":0.7}' | jq '.'
echo ""
read -p "Press Enter to continue..."
echo ""

# Example 10: Batch with different sentiments
echo -e "${GREEN}10. Batch Analysis - Diverse Sentiments${NC}"
echo "curl -X POST $BASE_URL/sentiment/batch \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{...}'"
echo ""
curl -X POST $BASE_URL/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Absolutely love it! Will buy again.",
      "Worst purchase ever. Do not buy.",
      "Meh, it is okay I guess.",
      "Fantastic quality and fast shipping!",
      "Product broke after two days, very upset.",
      "Decent product for the price.",
      "Exceeded my expectations in every way!",
      "Disappointing quality, expected better.",
      "Works fine, nothing to complain about.",
      "Horrible customer service experience."
    ],
    "confidenceThreshold": 0.75
  }' | jq '.'
echo ""

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Examples Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
