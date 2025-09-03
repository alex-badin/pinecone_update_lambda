import os
import json
import logging
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)


class ConfigBase(ABC):
    """Abstract base class for configuration management"""
    
    @abstractmethod
    def load_secrets(self):
        """Load API keys and secrets"""
        pass
    
    @abstractmethod
    def get_db_path(self, db_name):
        """Get database file path"""
        pass
    
    @abstractmethod
    def get_log_path(self):
        """Get log directory path"""
        pass
    
    @abstractmethod
    def get_export_path(self, filename):
        """Get export file path"""
        pass


class LocalConfig(ConfigBase):
    """Configuration for local development environment"""
    
    def __init__(self):
        # Load environment variables from .env file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        env_path = os.path.join(script_dir, '.env')
        load_dotenv(env_path)
        logger.info("Loaded configuration for LOCAL environment")
    
    def load_secrets(self):
        """Load secrets from environment variables"""
        required_vars = [
            "TG_API_ID", "TG_API_HASH", "TG_SESSION_STRING",
            "COHERE_KEY", "PINE_KEY", "GEMINI_API_KEY", "PINE_INDEX"
        ]
        
        secrets = {}
        missing = []
        
        for var in required_vars:
            value = os.getenv(var)
            if value:
                secrets[var] = value
            else:
                missing.append(var)
        
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
        
        return secrets
    
    def get_db_path(self, db_name):
        """Get local database path"""
        return os.path.join("data", "databases", db_name)
    
    def get_log_path(self):
        """Get local log directory"""
        return "logs"
    
    def get_cache_path(self):
        """Get local cache path"""
        return os.path.join("data", "cache", "summary_cache.db")
    
    def get_export_path(self, filename):
        """Get local export file path"""
        return os.path.join("data", "exports", filename)


class AWSConfig(ConfigBase):
    """Configuration for AWS ECS environment"""
    
    def __init__(self):
        self.region = os.environ.get('AWS_REGION', 'eu-central-1')
        self.secret_name = 'pinecone-update/api-keys'
        logger.info("Loaded configuration for AWS environment")
    
    def load_secrets(self):
        """Load secrets from AWS Secrets Manager"""
        try:
            secrets_manager = boto3.client('secretsmanager', region_name=self.region)
            response = secrets_manager.get_secret_value(SecretId=self.secret_name)
            
            if 'SecretString' in response:
                secrets = json.loads(response['SecretString'])
            else:
                secrets = json.loads(response['SecretBinary'])
            
            logger.info("Successfully loaded secrets from AWS Secrets Manager")
            return secrets
            
        except ClientError as e:
            logger.error(f"Error retrieving secrets from AWS: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing secrets JSON: {str(e)}")
            raise
    
    def get_db_path(self, db_name):
        """Get EBS-mounted database path"""
        return f"/data/databases/{db_name}"
    
    def get_log_path(self):
        """Get EBS-mounted log directory"""
        return "/data/logs"
    
    def get_cache_path(self):
        """Get EBS-mounted cache path"""
        return "/data/cache/summary_cache.db"
    
    def get_export_path(self, filename):
        """Get EBS-mounted export file path"""
        return f"/data/exports/{filename}"


def get_config():
    """Factory function to get appropriate configuration based on environment"""
    if os.environ.get('AWS_EXECUTION_ENV') or os.environ.get('ECS_CONTAINER_METADATA_URI'):
        return AWSConfig()
    else:
        return LocalConfig()


def setup_environment():
    """Initialize environment and return configuration and secrets"""
    config = get_config()
    secrets = config.load_secrets()
    
    # Set environment variables from secrets for backward compatibility
    for key, value in secrets.items():
        os.environ[key] = str(value)
    
    # Ensure required directories exist
    log_dir = config.get_log_path()
    os.makedirs(log_dir, exist_ok=True)
    
    # Ensure data directories exist
    if isinstance(config, AWSConfig):
        # For AWS, ensure EBS mount directories exist
        os.makedirs("/data/databases", exist_ok=True)
        os.makedirs("/data/cache", exist_ok=True)
        os.makedirs("/data/exports", exist_ok=True)
    else:
        # For local, ensure data directories exist
        os.makedirs("data/databases", exist_ok=True)
        os.makedirs("data/cache", exist_ok=True)
        os.makedirs("data/exports", exist_ok=True)
    
    return config, secrets
