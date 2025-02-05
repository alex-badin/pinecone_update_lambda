print('start pinecone update lambda')

import os
from dotenv import load_dotenv
import asyncio
import random
import json
import uuid
from datetime import datetime, timedelta, timezone
import time

import boto3
from botocore.exceptions import ClientError
from pyairtable import Api
from telethon import TelegramClient
from telethon.sessions import StringSession
import cohere
from pinecone import Pinecone
import re
import unicodedata
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from src.summarizer import Summarizer

# Function to retrieve secrets
def get_secret(secret_name):
    # Initialize AWS clients
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

# Retrieve secrets
try:
    # secrets = get_secret(os.environ['SECRET_NAME'])
    # TG_API_ID = secrets['tg_api_id']
    # TG_API_HASH = secrets['tg_api_hash']
    # TG_SESSION_STRING = secrets['tg_session_string']
    # COHERE_KEY = secrets['cohere_key']
    # PINE_KEY = secrets['pine_key']
    # AIRTABLE_API_TOKEN = secrets['airtable_api_token']
    # GEMINI_API_KEY = secrets['gemini_api_key']

    # Load environment variables
    load_dotenv()
    TG_API_ID = os.getenv('TG_API_ID')
    TG_API_HASH = os.getenv('TG_API_HASH')
    TG_SESSION_STRING = os.getenv('TG_SESSION_STRING')
    COHERE_KEY = os.getenv('COHERE_KEY')
    PINE_KEY = os.getenv('PINE_KEY')
    AIRTABLE_API_TOKEN = os.getenv('AIRTABLE_API_TOKEN')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

    # Verify all required variables are present
    required_vars = {
    "TG_API_ID": os.getenv("TG_API_ID"),
    "TG_API_HASH": os.getenv("TG_API_HASH"),
    "TG_SESSION_STRING": os.getenv("TG_SESSION_STRING"),
    "COHERE_KEY": os.getenv("COHERE_KEY"),
    "PINE_KEY": os.getenv("PINE_KEY"),
    "AIRTABLE_API_TOKEN": os.getenv("AIRTABLE_API_TOKEN"),
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
    }

    missing = [var_name for var_name, value in required_vars.items() if not value]

    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
except Exception as e:
    print(f"Error retrieving secrets: {str(e)}")
    raise

# Environment variables
PINE_INDEX = os.environ['PINE_INDEX']
AIRTABLE_BASE_ID = os.environ['AIRTABLE_BASE_ID']
AIRTABLE_TABLE_NAME = os.environ['AIRTABLE_TABLE_NAME']
AIRTABLE_LOG_TABLE = os.environ['AIRTABLE_LOG_TABLE']

# Initialize clients
co = cohere.Client(COHERE_KEY)
pc = Pinecone(PINE_KEY)
pine_index = pc.Index(PINE_INDEX)
airtable_api = Api(AIRTABLE_API_TOKEN)
table = airtable_api.table(AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)
log_table = airtable_api.table(AIRTABLE_BASE_ID, AIRTABLE_LOG_TABLE)

# Initialize summarizer with other clients
summarizer = Summarizer(GEMINI_API_KEY)

async def get_new_messages(channel, last_id, start_date, channel_index, total_channels):
    async with TelegramClient(StringSession(TG_SESSION_STRING), TG_API_ID, TG_API_HASH,
                              system_version="4.16.30-vxCUSTOM") as client:
        data = []
        try:
            offset_id = int(last_id) if last_id else 0  # Use None instead of 0
        except ValueError:
            offset_id = 0
        
        # Make start_date timezone-aware
        start_date = start_date.replace(tzinfo=timezone.utc)
        print(f"Parsing channel (#{channel_index} out of {total_channels}): {channel}, start date: {start_date}, last id: {last_id}, offset id: {offset_id}")
        # we start from the offset_id message, but store only messages after start_date
        async for message in client.iter_messages(channel, reverse=True, offset_id=offset_id):
            # print(f"Message date: {message.date}, message id: {message.id}")
            if message.date < start_date:
                continue
            data.append(message.to_dict())

    print(f"Channel: {channel}, N of new messages: {len(data)}")
    return data

def clean_text(text):
    # Unicode range for emojis
    emoji_pattern = re.compile("["
                               "\U0001F600-\U0001F64F"  # Emoticons
                               "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                               "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                               "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
                               "]+", flags=re.UNICODE)

    # Remove emojis
    text = emoji_pattern.sub(r'', str(text))
    # Regular expression for URLs
    url_pattern = re.compile(r"http\S+|www\S+")
    # Remove URLs
    text = url_pattern.sub(r'', str(text))
    # remove /n
    text = text.replace('\n', ' ')
    # Remove any remaining variation selectors
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')

    #Remove Foreign Agent text
    pattern = re.compile(r'[А-ЯЁ18+]{3,}\s[А-ЯЁ()]{5,}[^\n]*ИНОСТРАННОГО АГЕНТА')
    text = pattern.sub('', text)
    name1 = 'ПИВОВАРОВА АЛЕКСЕЯ ВЛАДИМИРОВИЧА'
    text = text.replace(name1, '')

    return text

def process_new_messages(messages, channel, stance):
    processed_messages = []
    empty_message_count = 0  # Counter for empty messages
    for message in messages:
        if 'message' not in message:
            empty_message_count += 1
            continue
        cleaned_message = clean_text(message['message'])
        if len(cleaned_message) > 30:
            summary, _ = summarizer.summarize(cleaned_message, max_summary_length=500, length_threshold=750)
            processed_messages.append({
                'id': message['id'],
                'channel': channel,
                'stance': stance,
                'cleaned_message': cleaned_message,
                'summary': summary,
                'date': message['date'],
                'views': message.get('views', 0)
            })
    if empty_message_count > 0:
        print(f"Number of empty messages: {empty_message_count}")
    return processed_messages

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
def get_embeddings_with_retry(texts, model="embed-multilingual-v3.0"):
    print(f"Getting embeddings for {len(texts)} texts")
    return co.embed(texts=texts, model=model, input_type="clustering").embeddings

def get_embeddings(messages, text_col='summary', model="embed-multilingual-v3.0"):
    texts = [msg[text_col] for msg in messages if text_col in msg]
    if not texts:
        print(f"Warning: No '{text_col}' found in messages. Available keys: {messages[0].keys() if messages else 'No messages'}")
        return messages
    try:
        embeddings = get_embeddings_with_retry(texts, model)
        for msg, embedding in zip(messages, embeddings):
            msg['embeddings'] = embedding
    except Exception as e:
        print(f"Error getting embeddings: {str(e)}")
        # Return messages without embeddings if there's an error
        return messages
    return messages

def upsert_to_pinecone(messages, index, batch_size=100):
    vectors = []
    for msg in messages:
        vector = {
            'id': f"{msg['channel']}_{msg['id']}",
            'values': msg['embeddings'],
            'metadata': {
                'cleaned_message': msg['cleaned_message'],
                'summary': msg['summary'],
                'stance': msg['stance'],
                'channel': msg['channel'],
                'date': int(time.mktime(msg['date'].timetuple())),
                'views': msg['views']
            }
        }
        vectors.append(vector)
    
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch)
    print(f"Upserted {len(vectors)} records. Last id: {vectors[-1]['id']}")

def update_airtable_last_id(table, channel, last_id):
    matching_records = table.all(formula=f"{{channel_name}}='{channel}'")
    if matching_records:
        record_id = matching_records[0]['id']
        table.update(record_id, {'last_id': int(last_id)})
    else:
        print(f"No matching record found for channel {channel}")

def log_summary_to_airtable(parsed_channels, missed_channels, new_channels):
    execution_date = datetime.now()
    primary_key = f"{execution_date.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    summary = {
        'id': primary_key,
        'execution_date': execution_date.isoformat(),
        'parsed_channels_count': len(parsed_channels),
        'parsed_channels': ', '.join(parsed_channels),
        'missed_channels_count': len(missed_channels),
        'missed_channels': ', '.join(missed_channels),
        'new_channels_count': len(new_channels),
        'new_channels': ', '.join(new_channels)
    }
    
    log_table.create(summary)
    print(f"Execution summary logged with ID: {primary_key}")

async def main():
    start_date = datetime.now(timezone.utc) - timedelta(days=4)
    print(f"Start date for messages: {start_date}")
    parsed_channels = []
    missed_channels = []
    new_channels = []

    records = table.all()
    df_channels = [record['fields'] for record in records]
    df_channels = [record for record in df_channels if record['status'] == 'Active']

    channels = [(record['channel_name'], record['last_id'], record['stance']) for record in df_channels]
    random.shuffle(channels)

    print(f"Channels to parse: {len(channels)}: {channels}")
    print(f"Start date for messages: {start_date}")
    total_channels = len(channels)
    for index, (channel, last_id, stance) in enumerate(channels, 1):
        try:
            messages = await get_new_messages(channel, last_id, start_date, index, total_channels)
            if not messages:
                print(f"No new messages for channel: {channel}")
                continue

            processed_messages = process_new_messages(messages, channel, stance)
            if not processed_messages:
                print(f"No processed messages for channel: {channel}")
                continue

            messages_with_embeddings = get_embeddings(processed_messages, text_col='cleaned_message', model="embed-multilingual-v3.0")

            if messages_with_embeddings:
                upsert_to_pinecone(messages_with_embeddings, pine_index)

                new_last_id = max(msg['id'] for msg in messages_with_embeddings)
                update_airtable_last_id(table, channel, new_last_id)

                parsed_channels.append(channel)
                if not last_id:
                    new_channels.append(channel)

                print(f"Successfully processed channel: {channel}")
            else:
                print(f"No messages with embeddings for channel: {channel}")

        except Exception as e:
            print(f"Error processing channel {channel}: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e.args}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            missed_channels.append(channel)

    log_summary_to_airtable(parsed_channels, missed_channels, new_channels)

async def async_handler(event, context):
    try:
        await main()
        return {
            'statusCode': 200,
            'body': 'Pinecone database updated successfully'
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': f'Error updating Pinecone database: {str(e)}'
        }
    finally:
        print('end pinecone update lambda')

def lambda_handler(event, context):
    return asyncio.get_event_loop().run_until_complete(async_handler(event, context))

if __name__ == "__main__":
    print("Starting local execution...")
    asyncio.run(main())
    print("Execution completed!")