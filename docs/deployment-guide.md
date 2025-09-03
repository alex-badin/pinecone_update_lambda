# Deployment Guide

## Local Development Deployment

### Quick Setup
```bash
# Run setup script
./scripts/setup_local_dev.sh

# Edit .env file with your API keys
nano .env

# Run the application
python main.py
```

### Manual Setup
```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your credentials

# 4. Run
python main.py
```

### Docker Development
```bash
# Using Docker Compose
docker-compose up

# Using Docker directly
docker build -t pinecone-update .
docker run --env-file .env -v ./logs:/app/logs pinecone-update
```

## AWS ECS Deployment

### Prerequisites
- AWS CLI configured with appropriate permissions
- Docker installed
- Your API keys ready for AWS Secrets Manager

### Step 1: Configure Deployment
```bash
# Edit deployment script with your details
nano deployment/aws/deploy.sh

# Update these variables:
ACCOUNT_ID="123456789012"
REGION="eu-central-1"  
REPOSITORY_NAME="ask-media/pinecone-update"
```

### Step 2: Create AWS Secrets
```bash
# Create secret in AWS Secrets Manager
aws secretsmanager create-secret \
  --name "pinecone-update/api-keys" \
  --description "API keys for pinecone update service" \
  --secret-string '{
    "TG_API_ID": "your_telegram_api_id",
    "TG_API_HASH": "your_telegram_api_hash",
    "TG_SESSION_STRING": "your_telegram_session_string",
    "COHERE_KEY": "your_cohere_key",
    "PINE_KEY": "your_pinecone_key",
    "GEMINI_API_KEY": "your_gemini_key",
    "PINE_INDEX": "your_pinecone_index_name"
  }'
```

### Step 3: Deploy to AWS
```bash
# Run deployment script
cd deployment/aws
./deploy.sh
```

### Step 4: Set Up EBS Volume
```bash
# Create EBS volume for persistent storage
aws ec2 create-volume \
  --size 50 \
  --volume-type gp3 \
  --iops 3000 \
  --encrypted \
  --availability-zone eu-central-1a \
  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=pinecone-update-storage}]'

# Note the volume ID and attach to your ECS instances
```

### Step 5: Set Up Scheduled Execution
```bash
# Create EventBridge rule for hourly execution
aws events put-rule \
  --name pinecone-update-schedule \
  --schedule-expression "rate(1 hour)" \
  --description "Hourly trigger for pinecone update task"

# Add ECS target to the rule
aws events put-targets \
  --rule pinecone-update-schedule \
  --targets "Id=1,Arn=arn:aws:ecs:eu-central-1:ACCOUNT:cluster/pinecone-update-cluster,RoleArn=arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole,EcsParameters={TaskDefinitionArn=arn:aws:ecs:eu-central-1:ACCOUNT:task-definition/pinecone-update,LaunchType=EC2}"
```

## Environment Variables

### Local Environment (.env file)
```bash
# Required
TG_API_ID=your_telegram_api_id
TG_API_HASH=your_telegram_api_hash
TG_SESSION_STRING=your_telegram_session_string
COHERE_KEY=your_cohere_api_key
PINE_KEY=your_pinecone_api_key
GEMINI_API_KEY=your_gemini_api_key
PINE_INDEX=your_pinecone_index_name

# Optional
DAYS_TO_PARSE=2
DB_BATCH_SIZE=1000
```

### AWS Environment
- Secrets stored in AWS Secrets Manager
- Environment variables set in ECS task definition
- Persistent storage mounted to `/data/`

## Monitoring

### Local Monitoring
- Logs: `./logs/pinecone_update.log`
- Databases: `./data/databases/`
- Cache: `./data/cache/`
- Exports: `./data/exports/`

### AWS Monitoring
- CloudWatch Logs: `/ecs/pinecone-update`
- EBS Storage: `/data/databases/`, `/data/logs/`, `/data/cache/`, `/data/exports/`
- Metrics: CPU, memory, task success/failure rates

## Troubleshooting

### Common Issues

1. **Missing API Keys**
   - Local: Check `.env` file
   - AWS: Verify AWS Secrets Manager

2. **Database Locked**
   - Local: Check if another instance is running
   - AWS: Ensure EBS volume is properly mounted

3. **Memory Issues**
   - Increase ECS task memory allocation
   - Monitor CloudWatch metrics

4. **Network Issues**
   - Check security groups for ECS tasks
   - Verify VPC configuration

### Testing

```bash
# Test environment detection
./scripts/test_basic.sh

# Run unit tests
./scripts/run_tests.sh

# Test configuration
python3 tests/test_config.py
```
