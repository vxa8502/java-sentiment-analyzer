#!/bin/bash
# Load Test Script for Java Sentiment Analyzer
# Verifies latency targets: P50 <= 1ms, P95 <= 2ms, P99 < 3ms
#
# Requirements:
#   - Apache Benchmark (ab) installed
#   - Application running on specified port
#
# Usage: ./run_load_tests.sh [port] [output_dir]

set -euo pipefail

PORT="${1:-8090}"
OUTPUT_DIR="${2:-../../results/load-tests}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_URL="http://localhost:${PORT}"

# Latency targets (in milliseconds)
TARGET_P50_MS=1
TARGET_P95_MS=2
TARGET_P99_MS=3

# Test payloads (<=512 characters)
# Note: Avoid exclamation marks to prevent shell escaping issues with printf
SHORT_TEXT='{"text":"This product is amazing and wonderful"}'
MEDIUM_TEXT='{"text":"I recently purchased this item and I have to say I am extremely satisfied with the quality. The build is solid, the features work as advertised, and the customer service was excellent. Would definitely recommend to anyone looking for a reliable product in this category."}'
LONG_TEXT='{"text":"After using this product for several weeks, I can confidently say it exceeds all my expectations. The attention to detail in the design is remarkable, and every feature has been thoughtfully implemented. The performance is consistently excellent, and I have not encountered any issues whatsoever. The value for money is outstanding, and I would not hesitate to purchase from this brand again. This is exactly what I was looking for and I am thrilled with my decision."}'

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Java Sentiment Analyzer - Load Test Suite"
echo "=========================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Target: ${BASE_URL}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check if server is running
echo "Checking server health..."
if ! curl -s "${BASE_URL}/api/v1/health" > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Server not responding at ${BASE_URL}${NC}"
    exit 1
fi
echo -e "${GREEN}Server is healthy${NC}"
echo ""

# Function to run ab test and extract percentiles
run_test() {
    local name="$1"
    local requests="$2"
    local concurrency="$3"
    local payload="$4"
    local output_file="${OUTPUT_DIR}/${TIMESTAMP}_${name}.txt"

    echo "Running: ${name}"
    echo "  Requests: ${requests}, Concurrency: ${concurrency}"

    # Write payload to temp file (use printf to avoid trailing newline)
    local payload_file=$(mktemp)
    printf '%s' "${payload}" > "${payload_file}"

    # Run Apache Benchmark
    ab -n "${requests}" \
       -c "${concurrency}" \
       -p "${payload_file}" \
       -T "application/json" \
       -H "Accept: application/json" \
       "${BASE_URL}/api/v1/sentiment/analyze" 2>&1 > "${output_file}"

    rm -f "${payload_file}"

    # Extract metrics
    local p50=$(grep "50%" "${output_file}" | awk '{print $2}' || echo "N/A")
    local p95=$(grep "95%" "${output_file}" | awk '{print $2}' || echo "N/A")
    local p99=$(grep "99%" "${output_file}" | awk '{print $2}' || echo "N/A")
    local mean=$(grep "Time per request:" "${output_file}" | head -1 | awk '{print $4}' || echo "N/A")
    local rps=$(grep "Requests per second:" "${output_file}" | awk '{print $4}' || echo "N/A")
    local failed=$(grep "Failed requests:" "${output_file}" | awk '{print $3}' || echo "0")

    echo "  Results: P50=${p50}ms, P95=${p95}ms, P99=${p99}ms, Mean=${mean}ms, RPS=${rps}"

    # Check compliance
    local pass=true
    if [[ "${p50}" != "N/A" ]] && (( $(echo "${p50} > ${TARGET_P50_MS}" | bc -l) )); then
        echo -e "  ${RED}FAIL: P50 (${p50}ms) exceeds target (${TARGET_P50_MS}ms)${NC}"
        pass=false
    fi
    if [[ "${p95}" != "N/A" ]] && (( $(echo "${p95} > ${TARGET_P95_MS}" | bc -l) )); then
        echo -e "  ${RED}FAIL: P95 (${p95}ms) exceeds target (${TARGET_P95_MS}ms)${NC}"
        pass=false
    fi
    if [[ "${p99}" != "N/A" ]] && (( $(echo "${p99} > ${TARGET_P99_MS}" | bc -l) )); then
        echo -e "  ${RED}FAIL: P99 (${p99}ms) exceeds target (${TARGET_P99_MS}ms)${NC}"
        pass=false
    fi
    if [[ "${failed}" != "0" ]]; then
        echo -e "  ${YELLOW}WARNING: ${failed} failed requests${NC}"
    fi
    if [[ "${pass}" == "true" ]]; then
        echo -e "  ${GREEN}PASS: All latency targets met${NC}"
    fi
    echo ""

    # Return results as JSON-like format for archival
    echo "{\"test\":\"${name}\",\"requests\":${requests},\"concurrency\":${concurrency},\"p50_ms\":${p50:-null},\"p95_ms\":${p95:-null},\"p99_ms\":${p99:-null},\"mean_ms\":${mean:-null},\"rps\":${rps:-null},\"failed\":${failed}}" >> "${OUTPUT_DIR}/${TIMESTAMP}_summary.jsonl"
}

# Warmup
echo "=== Warmup Phase ==="
run_test "warmup" 100 1 "${SHORT_TEXT}"

# Test Suite
echo "=== Single Request Baseline ==="
run_test "baseline_c1" 1000 1 "${SHORT_TEXT}"

echo "=== Low Concurrency (10 threads) ==="
run_test "low_concurrency_short" 1000 10 "${SHORT_TEXT}"
run_test "low_concurrency_medium" 1000 10 "${MEDIUM_TEXT}"
run_test "low_concurrency_long" 1000 10 "${LONG_TEXT}"

echo "=== Medium Concurrency (50 threads) ==="
run_test "medium_concurrency_short" 2000 50 "${SHORT_TEXT}"
run_test "medium_concurrency_medium" 2000 50 "${MEDIUM_TEXT}"

echo "=== High Concurrency (100 threads) ==="
run_test "high_concurrency_short" 5000 100 "${SHORT_TEXT}"

echo "=== Sustained Load (30 seconds) ==="
run_test "sustained_load" 10000 20 "${MEDIUM_TEXT}"

# Generate summary report
echo "=========================================="
echo "Test Complete"
echo "=========================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR}/${TIMESTAMP}_*.txt"
echo "Summary: ${OUTPUT_DIR}/${TIMESTAMP}_summary.jsonl"
echo ""
echo "Latency Targets:"
echo "  P50: <= ${TARGET_P50_MS}ms"
echo "  P95: <= ${TARGET_P95_MS}ms"
echo "  P99: < ${TARGET_P99_MS}ms"
echo ""

# Create markdown report
REPORT_FILE="${OUTPUT_DIR}/${TIMESTAMP}_report.md"
cat > "${REPORT_FILE}" << EOF
# Load Test Report - Java Sentiment Analyzer

**Date:** $(date -Iseconds)
**Server:** ${BASE_URL}
**Tool:** Apache Benchmark (ab)

## Latency Targets

| Metric | Target |
|--------|--------|
| P50 Latency | <= 1 ms |
| P95 Latency | <= 2 ms |
| P99 Latency | < 3 ms |

## Test Results

$(cat "${OUTPUT_DIR}/${TIMESTAMP}_summary.jsonl" | while read line; do
    test=$(echo "$line" | sed 's/.*"test":"\([^"]*\)".*/\1/')
    p50=$(echo "$line" | sed 's/.*"p50_ms":\([^,}]*\).*/\1/')
    p95=$(echo "$line" | sed 's/.*"p95_ms":\([^,}]*\).*/\1/')
    p99=$(echo "$line" | sed 's/.*"p99_ms":\([^,}]*\).*/\1/')
    rps=$(echo "$line" | sed 's/.*"rps":\([^,}]*\).*/\1/')
    failed=$(echo "$line" | sed 's/.*"failed":\([^,}]*\).*/\1/')
    echo "| ${test} | ${p50} | ${p95} | ${p99} | ${rps} | ${failed} |"
done)

## Raw Data

Individual test results are saved in:
- \`${TIMESTAMP}_*.txt\` - Full ab output per test
- \`${TIMESTAMP}_summary.jsonl\` - Machine-readable summary

## Environment

- Java Version: $(java -version 2>&1 | head -1)
- OS: $(uname -sr)
- CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
EOF

echo "Report: ${REPORT_FILE}"
echo ""
