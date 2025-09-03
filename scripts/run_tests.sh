#!/bin/bash

echo "🧪 Running Pinecone Update Tests..."

# Change to project root
cd "$(dirname "$0")/.."

echo "=== Running Basic Environment Tests ==="
python3 -m pytest tests/test_environment.py -v

echo ""
echo "=== Running Configuration Tests ==="
python3 -m pytest tests/test_config.py -v

echo ""
echo "=== Running Environment Detection Scripts ==="
./scripts/test_basic.sh

echo ""
echo "✅ All tests completed!"
