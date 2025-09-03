#!/bin/bash

echo "Testing pinecone_update environment detection..."

# Test 1: Local environment (should use .env file)
echo "=== Test 1: Local Environment ==="
python3 -c "
from config import get_config
config = get_config()
print(f'Config type: {config.__class__.__name__}')
print(f'DB path: {config.get_db_path(\"test.db\")}')
print(f'Log path: {config.get_log_path()}')
print(f'Cache path: {config.get_cache_path()}')
"

echo ""

# Test 2: AWS environment simulation
echo "=== Test 2: AWS Environment (simulated) ==="
AWS_EXECUTION_ENV=AWS_ECS_FARGATE python3 -c "
from config import get_config
config = get_config()
print(f'Config type: {config.__class__.__name__}')
print(f'DB path: {config.get_db_path(\"test.db\")}')
print(f'Log path: {config.get_log_path()}')
print(f'Cache path: {config.get_cache_path()}')
"

echo ""
echo "Environment detection test completed!"
