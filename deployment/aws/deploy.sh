#!/bin/bash

# AWS Deployment Script for Pinecone Update
# Replace these variables with your actual values
ACCOUNT_ID="YOUR_ACCOUNT_ID"
REGION="eu-central-1"
REPOSITORY_NAME="ask-media/pinecone-update"

echo "=== AWS Deployment Script for Pinecone Update ==="
echo "Account ID: $ACCOUNT_ID"
echo "Region: $REGION"
echo "Repository: $REPOSITORY_NAME"

# Step 1: Create ECR repository if it doesn't exist
echo "Creating ECR repository..."
aws ecr create-repository --repository-name $REPOSITORY_NAME --region $REGION 2>/dev/null || echo "Repository already exists"

# Step 2: Build and push Docker image
echo "Building Docker image..."
docker build --target production -t pinecone-update .

echo "Tagging image for ECR..."
docker tag pinecone-update:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:latest

echo "Logging into ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

echo "Pushing image to ECR..."
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPOSITORY_NAME:latest

# Step 3: Create CloudWatch Log Group
echo "Creating CloudWatch log group..."
aws logs create-log-group --log-group-name /ecs/pinecone-update --region $REGION 2>/dev/null || echo "Log group already exists"

# Step 4: Create ECS cluster
echo "Creating ECS cluster..."
aws ecs create-cluster --cluster-name pinecone-update-cluster --region $REGION 2>/dev/null || echo "Cluster already exists"

# Step 5: Update task definition template
echo "Updating task definition..."
sed "s/{ACCOUNT_ID}/$ACCOUNT_ID/g; s/{REGION}/$REGION/g" aws/task-definition.json > aws/task-definition-updated.json

# Step 6: Register task definition
echo "Registering task definition..."
aws ecs register-task-definition --cli-input-json file://aws/task-definition-updated.json --region $REGION

echo "=== Deployment completed! ==="
echo ""
echo "Next steps:"
echo "1. Create AWS Secrets Manager secret 'pinecone-update/api-keys' with your API keys"
echo "2. Create EBS volume and attach to EC2 instances"
echo "3. Set up EventBridge rule to trigger the task hourly"
echo "4. Configure ECS service with the task definition"
