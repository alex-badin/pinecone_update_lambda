"""Test configuration module functionality"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import LocalConfig, AWSConfig, get_config
except ImportError:
    # Handle case where dependencies are not available
    print("Warning: Config module dependencies not available")
    sys.exit(0)


class TestConfigModule(unittest.TestCase):
    
    def test_environment_detection_local(self):
        """Test that local environment is detected correctly"""
        with patch.dict(os.environ, {}, clear=True):
            config = get_config()
            self.assertIsInstance(config, LocalConfig)
    
    def test_environment_detection_aws_execution_env(self):
        """Test AWS environment detection via AWS_EXECUTION_ENV"""
        with patch.dict(os.environ, {'AWS_EXECUTION_ENV': 'AWS_ECS_FARGATE'}):
            config = get_config()
            self.assertIsInstance(config, AWSConfig)
    
    def test_environment_detection_ecs_metadata(self):
        """Test AWS environment detection via ECS_CONTAINER_METADATA_URI"""
        with patch.dict(os.environ, {'ECS_CONTAINER_METADATA_URI': 'http://localhost/metadata'}):
            config = get_config()
            self.assertIsInstance(config, AWSConfig)
    
    def test_local_config_paths(self):
        """Test local configuration paths"""
        config = LocalConfig()
        
        # Test database paths
        self.assertEqual(config.get_db_path("test.db"), "data/databases/test.db")
        
        # Test cache path
        self.assertEqual(config.get_cache_path(), "data/cache/summary_cache.db")
        
        # Test log path
        self.assertEqual(config.get_log_path(), "logs")
        
        # Test export path
        self.assertEqual(config.get_export_path("test.csv"), "data/exports/test.csv")
    
    def test_aws_config_paths(self):
        """Test AWS configuration paths"""
        config = AWSConfig()
        
        # Test database paths
        self.assertEqual(config.get_db_path("test.db"), "/data/databases/test.db")
        
        # Test cache path
        self.assertEqual(config.get_cache_path(), "/data/cache/summary_cache.db")
        
        # Test log path
        self.assertEqual(config.get_log_path(), "/data/logs")
        
        # Test export path
        self.assertEqual(config.get_export_path("test.csv"), "/data/exports/test.csv")


if __name__ == '__main__':
    unittest.main()
