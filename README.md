# Pinecone Update

A Telegram news processing service that downloads messages, creates summaries, generates embeddings, and stores them in Pinecone vector database. Compatible with both local development and AWS ECS deployment.

## Features

- **Telegram Integration**: Downloads messages from multiple channels
- **AI Processing**: Summarizes content using Gemini AI
- **Vector Storage**: Creates embeddings with Cohere and stores in Pinecone
- **Dual Environment**: Runs locally or on AWS ECS with automatic detection
- **Persistent Storage**: SQLite databases with environment-appropriate paths
- **Caching**: Intelligent summary caching to reduce API costs

## Quick Start

### Local Development
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Run the application
python main.py
```

### Docker Development
```bash
# Run with Docker Compose
docker-compose up
```

### AWS Deployment
```bash
# Deploy to AWS ECS
cd deployment/aws && ./deploy.sh
```

## Project Structure

```
pinecone_update/
├── main.py                      # Main application
├── config.py                    # Environment configuration
├── src/                         # Source modules
├── data/                        # Data files (local development)
│   ├── databases/               # Database files
│   ├── cache/                   # Cache files
│   └── exports/                 # Export files
├── logs/                        # Log files
├── docs/                        # Documentation
├── scripts/                     # Utility scripts
├── deployment/                  # Deployment configs
└── tests/                       # Test files
```

## Environment Support

- **Local**: Uses `.env` file, organized data directory structure
- **AWS ECS**: Uses AWS Secrets Manager, EBS-mounted storage
- **Automatic Detection**: No code changes needed between environments

### Data Organization
| Resource | Local | AWS ECS |
|----------|-------|---------|
| Databases | `data/databases/` | `/data/databases/` |
| Cache | `data/cache/` | `/data/cache/` |
| Exports | `data/exports/` | `/data/exports/` |
| Logs | `logs/` | `/data/logs/` |

## Documentation

- [AWS Deployment Guide](docs/README_AWS.md)
- [Implementation Details](docs/IMPLEMENTATION_SUMMARY.md)

## Requirements

- Python 3.11+
- API keys for: Telegram, Cohere, Pinecone, Gemini AI
- For AWS: ECS cluster, EBS volume, Secrets Manager

## License

Private project - not for redistribution.
