#!/bin/bash
# 🧹 SVMSentimentAnalyzer Cleanup Script
# Clean all build artifacts, logs, and temporary files

set -e

echo "🧹 Cleaning SVMSentimentAnalyzer..."

# Kill any running instances
echo "  → Stopping running instances..."
pkill -f "sentiment-analyzer" 2>/dev/null && echo "    ✓ Killed running app" || echo "    ✓ No running instances"

# Clean build artifacts
echo "  → Cleaning Maven build artifacts..."
# Remove all directories starting with "target"
for dir in target*; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"
        echo "    ✓ Removed $dir/"
    fi
done

# Clean logs
echo "  → Cleaning logs..."
find . -type f \( -name "*.log" -o -name "*.log.*" \) -delete 2>/dev/null
[ -d "logs" ] && rm -rf logs/*.log 2>/dev/null
echo "    ✓ Removed all .log files"

# Clean Mac OS junk
echo "  → Cleaning Mac OS junk..."
find . -name ".DS_Store" -delete 2>/dev/null && echo "    ✓ Removed .DS_Store files"

# Clean temp files
echo "  → Cleaning temporary files..."
find . -type f \( -name "*.tmp" -o -name "*.bak" -o -name "*~" \) -delete 2>/dev/null
echo "    ✓ Removed temp files"

# Clean test JSON files
echo "  → Cleaning test request files..."
rm -f comparison-*.json imdb-comparison-results.json 2>/dev/null && echo "    ✓ Removed test JSON files"

# Clean IDE files (optional)
echo "  → Cleaning IDE artifacts..."
# rm -rf .idea .vscode *.iml 2>/dev/null && echo "    ✓ Removed IDE files"

# Clean compiled classes (if any leaked outside target)
echo "  → Cleaning leaked .class files..."
find src -name "*.class" -delete 2>/dev/null && echo "    ✓ Removed .class files"

# Summary
echo ""
echo "✨ Cleanup complete!"
echo "📊 Disk space reclaimed:"
du -sh . 2>/dev/null || echo "  (calculation skipped)"
