# Pinecone Update - AWS ECS Compatible

This project has been updated to run both locally and on AWS ECS with automatic environment detection.

## Changes Made

### 1. New Configuration System (`config.py`)
- **Environment Detection**: Automatically detects if running locally or on AWS ECS
- **Local Config**: Uses `.env` file for secrets, local paths for databases/logs
- **AWS Config**: Uses AWS Secrets Manager for secrets, EBS-mounted paths for persistence
- **Simple Interface**: Clean abstraction that hides environment complexity

### 2. Updated Main Script (`main.py`)
- Simplified initialization using the new config system
- Environment-aware database and cache paths
- Cleaner logging setup with environment context

### 3. Docker Support
- **Multi-stage Dockerfile**: Development and production builds
- **Development stage**: Includes all dependencies for local testing
- **Production stage**: Optimized for AWS deployment with EBS mounts
- **Docker Compose**: For easy local development and testing

### 4. AWS Deployment Files (`aws/`)
- **Task Definition**: ECS task configuration with EBS volume mounting
- **Deployment Script**: Automated deployment to AWS with ECR push
- **IAM Policies**: Required permissions for ECS execution

## Usage

### Local Development
```bash
# Run directly
python main.py

# Or with Docker
docker-compose up

# Test environment detection
./test_env.sh
```

### AWS Deployment
```bash
# 1. Set up your AWS credentials and update aws/deploy.sh with your account details
# 2. Create secrets in AWS Secrets Manager
# 3. Deploy to AWS
cd aws && ./deploy.sh
```

## Environment Detection

The system automatically detects the environment:
- **Local**: When `AWS_EXECUTION_ENV` or `ECS_CONTAINER_METADATA_URI` are not set
- **AWS**: When running in ECS containers

## Configuration

### Local Environment
- Uses `.env` file for API keys
- Stores databases in current directory
- Logs to `./logs/` directory
- Cache in current directory

### AWS Environment
- Retrieves secrets from AWS Secrets Manager (`pinecone-update/api-keys`)
- Stores databases in `/data/db/` (EBS mounted)
- Logs to `/data/logs/` (EBS mounted)
- Cache in `/data/cache/` (EBS mounted)

## Required AWS Resources

1. **ECR Repository**: For container images
2. **ECS Cluster**: To run the tasks
3. **EBS Volume**: For persistent database storage
4. **Secrets Manager**: For API keys storage
5. **CloudWatch**: For logging and monitoring
6. **EventBridge**: For scheduled execution
7. **IAM Roles**: For ECS execution and task permissions

## Secret Format (AWS Secrets Manager)

The secret `pinecone-update/api-keys` should contain:
```json
{
  "TG_API_ID": "your_telegram_api_id",
  "TG_API_HASH": "your_telegram_api_hash", 
  "TG_SESSION_STRING": "your_telegram_session_string",
  "COHERE_KEY": "your_cohere_key",
  "PINE_KEY": "your_pinecone_key",
  "GEMINI_API_KEY": "your_gemini_key",
  "PINE_INDEX": "your_pinecone_index_name"
}
```

## Benefits

- **Single Codebase**: Same code runs in both environments
- **Environment Isolation**: Clean separation of concerns
- **Easy Testing**: Local development mirrors production
- **Cost Effective**: Optimized containers for AWS
- **Persistent Storage**: EBS ensures data survives container restarts
- **Monitoring**: CloudWatch integration for logs and metrics
