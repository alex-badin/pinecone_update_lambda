print('start pinecone update lambda')

import os
from dotenv import load_dotenv
import asyncio
import random
import json
import uuid
from datetime import datetime, timedelta, timezone
import time
import sqlite3

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

# Function to retrieve secrets (if you use AWS Secrets Manager)
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

# Load environment variables
try:
    load_dotenv()
    TG_API_ID = os.getenv('TG_API_ID')
    TG_API_HASH = os.getenv('TG_API_HASH')
    TG_SESSION_STRING = os.getenv('TG_SESSION_STRING')
    COHERE_KEY = os.getenv('COHERE_KEY')
    PINE_KEY = os.getenv('PINE_KEY')
    AIRTABLE_API_TOKEN = os.getenv('AIRTABLE_API_TOKEN')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    required_vars = {
        "TG_API_ID": TG_API_ID,
        "TG_API_HASH": TG_API_HASH,
        "TG_SESSION_STRING": TG_SESSION_STRING,
        "COHERE_KEY": COHERE_KEY,
        "PINE_KEY": PINE_KEY,
        "AIRTABLE_API_TOKEN": AIRTABLE_API_TOKEN,
        "GEMINI_API_KEY": GEMINI_API_KEY,
    }
    missing = [var_name for var_name, value in required_vars.items() if not value]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
except Exception as e:
    print(f"Error retrieving secrets: {str(e)}")
    raise

# Environment variables for Airtable and Pinecone
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

# Initialize your summarizer
summarizer = Summarizer(GEMINI_API_KEY)

async def get_new_messages(channel, last_id, start_date, channel_index, total_channels):
    async with TelegramClient(StringSession(TG_SESSION_STRING), TG_API_ID, TG_API_HASH,
                              system_version="4.16.30-vxCUSTOM") as client:
        data = []
        try:
            offset_id = int(last_id) if last_id else 0
        except ValueError:
            offset_id = 0
        
        # Make start_date timezone-aware
        start_date = start_date.replace(tzinfo=timezone.utc)
        print(f"Parsing channel (#{channel_index} of {total_channels}): {channel}, start date: {start_date}, last id: {last_id}, offset id: {offset_id}")
        async for message in client.iter_messages(channel, reverse=True, offset_id=offset_id):
            if message.date < start_date:
                continue
            data.append(message.to_dict())
    print(f"Channel: {channel}, number of new messages: {len(data)}")
    return data

def clean_text(text):
    # Remove emojis using a Unicode pattern
    emoji_pattern = re.compile("["
                               "\U0001F600-\U0001F64F"
                               "\U0001F300-\U0001F5FF"
                               "\U0001F680-\U0001F6FF"
                               "\U0001F1E0-\U0001F1FF"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', str(text))
    # Remove URLs
    url_pattern = re.compile(r"http\S+|www\S+")
    text = url_pattern.sub(r'', str(text))
    # Replace newline characters with a space
    text = text.replace('\n', ' ')
    # Remove remaining variation selectors
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    # Remove specific unwanted patterns
    pattern = re.compile(r'[А-ЯЁ18+]{3,}\s[А-ЯЁ()]{5,}[^\n]*ИНОСТРАННОГО АГЕНТА')
    text = pattern.sub('', text)
    name1 = 'ПИВОВАРОВА АЛЕКСЕЯ ВЛАДИМИРОВИЧА'
    text = text.replace(name1, '')
    return text

def process_new_messages(messages, channel, stance):
    processed_messages = []
    empty_message_count = 0
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
                'raw_message': message,  # Save full raw data
                'cleaned_message': cleaned_message,
                'summary': summary,
                'date': message['date'],
                'views': message.get('views', 0)
            })
    if empty_message_count > 0:
        print(f"Number of empty messages skipped: {empty_message_count}")
    return processed_messages

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
def get_embeddings_with_retry(texts, model="embed-multilingual-v3.0"):
    print(f"Getting embeddings for {len(texts)} texts")
    return co.embed(texts=texts, model=model, input_type="clustering").embeddings

def get_embeddings(messages, text_col='cleaned_message', model="embed-multilingual-v3.0"):
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
    print(f"Upserted {len(vectors)} records to Pinecone. Last id: {vectors[-1]['id']}")

def update_airtable_last_id(table, channel, last_id):
    matching_records = table.all(formula=f"{{channel_name}}='{channel}'")
    if matching_records:
        record_id = matching_records[0]['id']
        table.update(record_id, {'last_id': int(last_id)})
    else:
        print(f"No matching Airtable record found for channel {channel}")

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

# --- SQLite storage functions ---
@contextmanager
def get_db_connection(db_path="collected_messages.db", max_attempts=3):
    """Context manager for database connections with retry logic"""
    attempt = 0
    while attempt < max_attempts:
        try:
            conn = sqlite3.connect(db_path, timeout=20.0)  # Add timeout parameter
            yield conn
            return
        except sqlite3.OperationalError as e:
            attempt += 1
            if attempt == max_attempts:
                raise
            print(f"Database locked, retrying... (attempt {attempt}/{max_attempts})")
            time.sleep(1)  # Wait before retrying
        finally:
            try:
                conn.close()
            except:
                pass

def create_sqlite_db(db_path="collected_messages.db"):
    """Connects to (or creates) the SQLite database and ensures the table exists."""
    with get_db_connection(db_path) as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                channel TEXT,
                stance TEXT,
                raw_message TEXT,
                cleaned_message TEXT,
                summary TEXT,
                date TEXT,
                views INTEGER,
                embeddings TEXT
            )
        ''')
        conn.commit()

def store_in_sqlite(messages, db_path="collected_messages.db"):
    """Stores a list of processed messages in the SQLite database."""
    with get_db_connection(db_path) as conn:
        c = conn.cursor()
        to_insert = []
        for msg in messages:
            # Create a unique key combining channel and message ID
            pk = f"{msg['channel']}_{msg['id']}"
            # Convert raw_message and embeddings to JSON strings for storage
            raw_message_json = json.dumps(msg['raw_message'], ensure_ascii=False, default=str)
            embeddings_json = json.dumps(msg.get('embeddings', []), ensure_ascii=False) if 'embeddings' in msg else ""
            # Convert date to ISO string if necessary
            if isinstance(msg['date'], str):
                date_str = msg['date']
            else:
                date_str = msg['date'].isoformat() if hasattr(msg['date'], 'isoformat') else str(msg['date'])
            to_insert.append((
                pk,
                msg['channel'],
                msg['stance'],
                raw_message_json,
                msg['cleaned_message'],
                msg['summary'],
                date_str,
                msg.get('views', 0),
                embeddings_json
            ))
        try:
            c.executemany('''
                INSERT OR REPLACE INTO messages
                (id, channel, stance, raw_message, cleaned_message, summary, date, views, embeddings)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', to_insert)
            conn.commit()
            print(f"Stored {len(to_insert)} messages in SQLite database at {db_path}")
        except Exception as e:
            print("Error storing messages in SQLite:", str(e))
            raise

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

                # Incrementally store the collected messages in the SQLite database
                store_in_sqlite(messages_with_embeddings)
            else:
                print(f"No messages with embeddings for channel: {channel}")

        except Exception as e:
            print(f"Error processing channel {channel}: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            missed_channels.append(channel)

    log_summary_to_airtable(parsed_channels, missed_channels, new_channels)

async def async_handler(event, context):
    try:
        await main()
        return {
            'statusCode': 200,
            'body': 'Pinecone database updated and messages stored in SQLite successfully'
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