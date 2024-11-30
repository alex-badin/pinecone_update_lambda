import os
import json
import boto3
from botocore.exceptions import ClientError

def get_secret(secret_name):
    secrets_manager = boto3.client('secretsmanager', region_name=os.environ.get('AWS_REGION', 'eu-central-1'))
    try:
        get_secret_value_response = secrets_manager.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        raise e
    else:
        if 'SecretString' in get_secret_value_response:
            return json.loads(get_secret_value_response['SecretString'])
        else:
            return json.loads(get_secret_value_response['SecretBinary'])

# Load all configuration
def load_config():
    secrets = get_secret(os.environ['SECRET_NAME'])
    
    return {
        'TG_API_ID': secrets['tg_api_id'],
        'TG_API_HASH': secrets['tg_api_hash'],
        'TG_SESSION_STRING': secrets['tg_session_string'],
        'COHERE_KEY': secrets['cohere_key'],
        'PINE_KEY': secrets['pine_key'],
        'AIRTABLE_API_TOKEN': secrets['airtable_api_token'],
        'GEMINI_KEY': secrets['gemini_key'],
        'PINE_INDEX': os.environ['PINE_INDEX'],
        'AIRTABLE_BASE_ID': os.environ['AIRTABLE_BASE_ID'],
        'AIRTABLE_TABLE_NAME': os.environ['AIRTABLE_TABLE_NAME'],
        'AIRTABLE_LOG_TABLE': os.environ['AIRTABLE_LOG_TABLE']
    } 