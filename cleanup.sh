#!/bin/bash
# Cleanup Script - removes logs, temp files, and other junk that mvn clean doesn't handle
# For model artifacts, use ./scripts/reset.sh instead

set -e

echo "Cleaning up..."

# Kill any running instances
echo "  -> Stopping running instances..."
pkill -f "java.*sentiment" 2>/dev/null && echo "    - Killed running app" || echo "    - No running instances"

# Clean build artifacts
echo "  -> Cleaning Maven build artifacts..."
rm -rf target/
echo "    - Removed target/"

# Clean logs
echo "  -> Cleaning logs..."
find . -type f \( -name "*.log" -o -name "*.log.*" \) -delete
if [ -d "logs" ]; then
    rm -rf logs/*
    echo "    - Emptied logs/"
fi

# Clean Mac OS junk
echo "  -> Cleaning Mac OS junk..."
find . -name ".DS_Store" -delete

# Clean temp files
echo "  -> Cleaning temporary files..."
find . -type f \( -name "*.tmp" -o -name "*.bak" -o -name "*~" \) -delete

# Clean test JSON files
echo "  -> Cleaning test request files..."
rm -f comparison-*.json imdb-comparison-results.json

# Clean compiled classes (if any leaked outside target)
echo "  -> Cleaning leaked .class files..."
find src -name "*.class" -delete 2>/dev/null || true

echo ""
echo "Done."
