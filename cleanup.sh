#!/bin/bash
# Cleanup Script - removes logs, temp files, and other junk that mvn clean doesn't handle
# For model artifacts, use ./scripts/reset.sh instead

set -e

echo "Cleaning up..."

# Kill any running instances
echo "  -> Stopping running instances..."
pkill -f "java.*sentiment" 2>/dev/null && echo "    - Killed running app" || echo "    - No running instances"

# Clean build artifacts
echo "  -> Removing target/..."
rm -rf target/

# Clean logs
echo "  -> Emptying logs/..."
rm -rf logs/* 2>/dev/null || true

# Clean Python cache
echo "  -> Removing __pycache__/..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Clean OS and editor junk
echo "  -> Removing .DS_Store, temp, backup files..."
find . \( -name ".DS_Store" -o -name "*.tmp" -o -name "*.bak" -o -name "*~" \) -delete 2>/dev/null || true

echo ""
echo "Done."
