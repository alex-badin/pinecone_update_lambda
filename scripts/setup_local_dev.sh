#!/bin/bash

# Setup script for local development environment

echo "🚀 Setting up Pinecone Update local development environment..."

# Check if Python 3.11+ is available
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
if [[ $(echo "$python_version >= 3.11" | bc -l) -eq 1 ]]; then
    echo "✅ Python $python_version is compatible"
else
    echo "❌ Python 3.11+ required, found $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env template..."
    cat > .env << 'EOF'
# Telegram API credentials
TG_API_ID=your_telegram_api_id
TG_API_HASH=your_telegram_api_hash
TG_SESSION_STRING=your_telegram_session_string

# AI Service API keys
COHERE_KEY=your_cohere_api_key
PINE_KEY=your_pinecone_api_key
GEMINI_API_KEY=your_gemini_api_key

# Pinecone configuration
PINE_INDEX=your_pinecone_index_name

# Optional configuration
DAYS_TO_PARSE=2
DB_BATCH_SIZE=1000

# Airtable (optional)
AIRTABLE_API_TOKEN=your_airtable_token
EOF
    echo "✅ .env template created - please edit with your API keys"
else
    echo "✅ .env file already exists"
fi

# Create directories
mkdir -p logs
mkdir -p data/databases
mkdir -p data/cache  
mkdir -p data/exports
echo "✅ Directory structure created"

# Test environment detection
echo "Testing environment detection..."
if ./scripts/test_basic.sh > /dev/null 2>&1; then
    echo "✅ Environment detection working"
else
    echo "⚠️  Environment detection test failed (this is expected without dependencies)"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python main.py"
echo ""
echo "For Docker development: docker-compose up"
