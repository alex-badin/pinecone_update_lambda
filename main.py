print('start pinecone update')

import os
import sys
import traceback
from dotenv import load_dotenv
import asyncio
import random
import json
import uuid
from datetime import datetime, timedelta, timezone
import time
import sqlite3
import logging
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager

# At the top of the file with other constants
CHANNELS_DB = "channels.db"
MESSAGES_DB = "collected_messages.db"

# Set up logging configuration
try:
    print('Setting up logging...')
    # Get absolute path for the log directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_directory = os.path.join(script_dir, 'logs')
    os.makedirs(log_directory, exist_ok=True)
    log_file = os.path.join(log_directory, 'pinecone_update.log')
    
    # Add immediate file logging for debugging
    with open(os.path.join(log_directory, 'startup_debug.log'), 'a') as f:
        f.write(f'\n--- New execution at {datetime.now()} ---\n')
        f.write(f'Script directory: {script_dir}\n')
        f.write(f'Python executable: {sys.executable}\n')
        f.write(f'Working directory: {os.getcwd()}\n')

    # Configure logging
    logger = logging.getLogger('PineconeUpdate')
    logger.setLevel(logging.DEBUG)  # Temporarily set to DEBUG level

    # Create handlers
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    console_handler = logging.StreamHandler()

    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    console_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('Logging setup completed')
    
except Exception as e:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'crash_log.txt'), 'a') as f:
        f.write(f'\n--- Error at {datetime.now()} ---\n')
        f.write(f'Error: {str(e)}\n')
        f.write(f'Traceback: {traceback.format_exc()}\n')
    raise

import boto3
from botocore.exceptions import ClientError
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
from tqdm import tqdm  # Import tqdm for progress bars

from src.summarizer import Summarizer

# Function to retrieve secrets (if AWS Secrets Manager is used)
def get_secret(secret_name):
    secrets_manager = boto3.client('secretsmanager', region_name=os.environ.get('AWS_REGION', 'eu-central-1'))
    try:
        get_secret_value_response = secrets_manager.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        logger.error(f"Error retrieving secret {secret_name}: {str(e)}")
        raise e
    else:
        if 'SecretString' in get_secret_value_response:
            return json.loads(get_secret_value_response['SecretString'])
        else:
            return json.loads(get_secret_value_response['SecretBinary'])

# Load environment variables
# Modify the load_dotenv() call to use absolute path
try:
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct absolute path to .env file
    env_path = os.path.join(script_dir, '.env')
    # Load environment variables with explicit path
    load_dotenv(env_path)
    
    TG_API_ID = os.getenv('TG_API_ID')
    TG_API_HASH = os.getenv('TG_API_HASH')
    TG_SESSION_STRING = os.getenv('TG_SESSION_STRING')
    COHERE_KEY = os.getenv('COHERE_KEY')
    PINE_KEY = os.getenv('PINE_KEY')
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    required_vars = {
        "TG_API_ID": TG_API_ID,
        "TG_API_HASH": TG_API_HASH,
        "TG_SESSION_STRING": TG_SESSION_STRING,
        "COHERE_KEY": COHERE_KEY,
        "PINE_KEY": PINE_KEY,
        "GEMINI_API_KEY": GEMINI_API_KEY,
    }
    missing = [var_name for var_name, value in required_vars.items() if not value]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
except Exception as e:
    logger.error(f"Error retrieving secrets: {str(e)}")
    raise

# Environment variables for Pinecone
PINE_INDEX = os.environ['PINE_INDEX']

# Initialize clients
co = cohere.Client(COHERE_KEY)
pc = Pinecone(PINE_KEY)
pine_index = pc.Index(PINE_INDEX)

# Initialize your summarizer
summarizer = Summarizer(GEMINI_API_KEY)

async def get_new_messages(channel, last_id, start_date, channel_index, total_channels):
    async with TelegramClient(StringSession(TG_SESSION_STRING), TG_API_ID, TG_API_HASH,
                              system_version="4.16.30-vxCUSTOM") as client:
        data = []
        
        # Check if we should ignore last_id based on environment variable
        ignore_last_id = os.getenv('IGNORE_LAST_ID', 'False').lower() in ('true', '1', 't', 'yes')
        
        try:
            # If IGNORE_LAST_ID is True, always start from the beginning (offset_id = 0)
            if ignore_last_id:
                offset_id = 0
                logger.info(f"IGNORE_LAST_ID is set to True, ignoring last_id and starting from the beginning")
            else:
                offset_id = int(last_id) if last_id else 0
        except ValueError:
            offset_id = 0
        
        start_date = start_date.replace(tzinfo=timezone.utc) # Ensure timezone-aware
        logger.info(f"Parsing channel (#{channel_index} of {total_channels}): {channel}, start date: {start_date}, last id: {last_id}, offset_id: {offset_id}")
        async for message in client.iter_messages(channel, reverse=True, offset_id=offset_id):
            if message.date < start_date: #for new channels with no last_id - not earlier than 4 days ago
                continue
            data.append(message.to_dict())
    logger.info(f"Channel: {channel}, number of new messages: {len(data)}")
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

# Update the process_new_messages function to use batch processing
# Update the process_new_messages function to store is_digest
async def process_new_messages(messages, channel, stance):
    processed_messages = []
    empty_message_count = 0
    
    # Create a progress bar for message processing
    total_messages = len(messages)
    logger.info(f"Processing {total_messages} messages from channel: {channel}")
    
    # Extract texts for batch processing
    texts_to_summarize = []
    valid_messages = []
    
    for message in messages:
        if 'message' not in message:
            empty_message_count += 1
            continue
        cleaned_message = clean_text(message['message'])
        if len(cleaned_message) > 30:
            texts_to_summarize.append(cleaned_message)
            valid_messages.append({
                'id': message['id'],
                'channel': channel,
                'stance': stance,
                'raw_message': message,  # Save full raw data
                'cleaned_message': cleaned_message,
                'date': message['date'],
                'views': message.get('views', 0)
            })
    
    # Batch summarize all texts
    if texts_to_summarize:
        # Calculate the number of batches
        batch_size = summarizer.batch_size
        num_batches = (len(texts_to_summarize) + batch_size - 1) // batch_size
        
        # Add rate limiting - ensure we don't exceed Gemini's 2000 RPM limit
        # Assuming each batch has batch_size requests, calculate max batches per minute
        max_batches_per_minute = 2000 // batch_size
        # Add a safety margin
        safe_batches_per_minute = max(1, int(max_batches_per_minute * 0.8))
        # Calculate delay between batches in seconds
        batch_delay = 60 / safe_batches_per_minute if safe_batches_per_minute > 0 else 0
        
        logger.info(f"Rate limiting: Processing max {safe_batches_per_minute} batches/minute with {batch_delay:.2f}s delay between batches")
        
        with tqdm(total=num_batches, desc=f"Summarizing {channel}", unit="batch") as pbar:
            # Pass the progress bar to the summarize_batch method
            summaries = await summarizer.summarize_batch(
                texts_to_summarize, 
                progress_callback=pbar.update,
                batch_delay=batch_delay  # Pass the delay to the summarize_batch method
            )
        
        # Add summaries to processed messages
        for msg, (summary, is_digest) in zip(valid_messages, summaries):
            msg['summary'] = summary
            msg['is_digest'] = is_digest  # Store the is_digest flag
            processed_messages.append(msg)
    
    if empty_message_count > 0:
        logger.warning(f"Number of empty messages skipped for channel {channel}: {empty_message_count}")
    
    logger.info(f"Successfully processed {len(processed_messages)} of {total_messages} messages from {channel}")
    return processed_messages

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
def get_embeddings_with_retry(texts, model="embed-multilingual-v3.0"):
    logger.info(f"Getting embeddings for {len(texts)} texts")
    return co.embed(texts=texts, model=model, input_type="clustering").embeddings

def get_embeddings(messages, text_col='cleaned_message', model="embed-multilingual-v3.0"):
    texts = [msg[text_col] for msg in messages if text_col in msg]
    if not texts:
        logger.warning(f"No '{text_col}' found in messages. Available keys: {messages[0].keys() if messages else 'No messages'}")  
        return messages
    try:
        logger.info(f"Getting embeddings for {len(texts)} texts")
        # Add progress bar for embedding generation
        with tqdm(total=1, desc="Generating embeddings", unit="batch") as pbar:
            embeddings = get_embeddings_with_retry(texts, model)
            pbar.update(1)
        
        for msg, embedding in zip(messages, embeddings):
            msg['embeddings'] = embedding
    except Exception as e:
        logger.error(f"Error getting embeddings: {str(e)}")
        return messages
    return messages

# Update the upsert_to_pinecone function to include is_digest in metadata
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
                'views': msg['views'],
                'is_digest': msg.get('is_digest', False)  # Include is_digest in metadata
            }
        }
        vectors.append(vector)
    
    total_batches = (len(vectors) + batch_size - 1) // batch_size  # Calculate total number of batches
    with tqdm(total=total_batches, desc="Upserting to Pinecone", unit="batch") as pbar:
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            index.upsert(vectors=batch)
            pbar.update(1)
    
    logger.info(f"Upserted {len(vectors)} records to Pinecone. Last id: {vectors[-1]['id'] if vectors else 'No vectors'}")

def log_summary_to_db(parsed_channels, missed_channels, new_channels):
    execution_date = datetime.now()
    primary_key = f"{execution_date.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    with get_db_connection(CHANNELS_DB) as conn:
        c = conn.cursor()
        c.execute('''
            INSERT INTO execution_logs
            (id, execution_date, parsed_channels_count, parsed_channels,
             missed_channels_count, missed_channels,
             new_channels_count, new_channels)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            primary_key,
            execution_date.isoformat(),
            len(parsed_channels),
            ', '.join(parsed_channels),
            len(missed_channels),
            ', '.join(missed_channels),
            len(new_channels),
            ', '.join(new_channels)
        ))
        conn.commit()
    logger.info(f"Execution summary logged with ID: {primary_key}")

# --- SQLite storage functions ---
@contextmanager # for proper closing of sqlite connection
def get_db_connection(db_path, max_attempts=3):
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
            logger.warning(f"Database locked, retrying... (attempt {attempt}/{max_attempts})")
            time.sleep(1)  # Wait before retrying
        finally:
            try:
                conn.close()
            except:
                pass

def create_channels_db():
    """Creates the channels and execution_logs tables in the channels database."""
    with get_db_connection(CHANNELS_DB) as conn:
        c = conn.cursor()
        # Channels table
        c.execute('''
            CREATE TABLE IF NOT EXISTS channels (
                channel_name TEXT PRIMARY KEY,
                last_id TEXT,
                stance TEXT,
                status TEXT DEFAULT 'Active',
                last_sync TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Execution_logs table
        c.execute('''
            CREATE TABLE IF NOT EXISTS execution_logs (
                id TEXT PRIMARY KEY,
                execution_date TIMESTAMP,
                parsed_channels_count INTEGER,
                parsed_channels TEXT,
                missed_channels_count INTEGER,
                missed_channels TEXT,
                new_channels_count INTEGER,
                new_channels TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def create_messages_db():
    """Creates the messages table in the messages database."""
    with get_db_connection(MESSAGES_DB) as conn:
        c = conn.cursor()
        # Messages table
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
                embeddings TEXT,
                is_digest BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

# Update the store_in_sqlite function to include is_digest
def store_in_sqlite(messages):
    """Stores a list of processed messages in the SQLite database."""
    with get_db_connection(MESSAGES_DB) as conn:
        c = conn.cursor()
        to_insert = []
        current_timestamp = datetime.now().isoformat()
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
            
            # Get is_digest value (default to 0 if not present)
            is_digest = 1 if msg.get('is_digest', False) else 0
            
            to_insert.append((
                pk,
                msg['channel'],
                msg['stance'],
                raw_message_json,
                msg['cleaned_message'],
                msg['summary'],
                date_str,
                msg.get('views', 0),
                embeddings_json,
                is_digest,
                current_timestamp  # Add current timestamp for created_at
            ))
        try:
            c.executemany('''
                INSERT OR REPLACE INTO messages
                (id, channel, stance, raw_message, cleaned_message, summary, date, views, embeddings, is_digest, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', to_insert)
            conn.commit()
            logger.info(f"Stored {len(to_insert)} messages in SQLite database at {MESSAGES_DB}")
        except Exception as e:
            logger.error(f"Error storing messages in SQLite: {str(e)}")
            raise

def get_active_channels():
    """Retrieve active channels from SQLite."""
    logger.info("Retrieving active channels from SQLite...")
    with get_db_connection(CHANNELS_DB) as conn:
        c = conn.cursor()
        c.execute("SELECT channel_name, last_id, stance FROM channels WHERE status = 'Active'")
        return c.fetchall()

def update_channel_last_id(channel, last_id):
    """Update the last_id for a channel in SQLite."""
    with get_db_connection(CHANNELS_DB) as conn:
        c = conn.cursor()
        c.execute("""
            UPDATE channels 
            SET last_id = ?, last_sync = CURRENT_TIMESTAMP 
            WHERE channel_name = ?
        """, (str(last_id), channel))
        conn.commit()

async def main():
    try:
        # Ensure databases exist with the correct tables
        create_channels_db()
        create_messages_db()
        
        # Get DAYS_TO_PARSE from environment variable, default to 2 if not set
        start_date = datetime.now(timezone.utc) - timedelta(days=int(os.getenv('DAYS_TO_PARSE', 2)))
        logger.info(f"Starting main execution with start date: {start_date}")
        
        # Log if we're ignoring last_id values
        ignore_last_id = os.getenv('IGNORE_LAST_ID', 'False').lower() in ('true', '1', 't', 'yes')
        if ignore_last_id:
            logger.info("IGNORE_LAST_ID is set to True - will collect messages from the beginning regardless of last_id values")
        
        # Check if we should skip upserting to Pinecone
        skip_pinecone = os.getenv('SKIP_PINECONE', 'False').lower() in ('true', '1', 't', 'yes')
        if skip_pinecone:
            logger.info("SKIP_PINECONE is set to True - will only store data locally without upserting to Pinecone")
        
        parsed_channels = []
        missed_channels = []
        new_channels = []

        # Get channels from SQLite
        channels = get_active_channels()
        random.shuffle(channels)
        total_channels = len(channels)

        logger.info(f"Processing {len(channels)} channels: {channels}")

        # Initialize your summarizer with batch size from environment variable
        batch_size = int(os.getenv('SUMMARIZER_BATCH_SIZE', '5'))
        cache_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'summary_cache.db')
        global summarizer
        summarizer = Summarizer(GEMINI_API_KEY, cache_db=cache_db, batch_size=batch_size)

        for index, (channel, last_id, stance) in enumerate(channels, 1):
            try:
                messages = await get_new_messages(channel, last_id, start_date, index, total_channels)
                if not messages:
                    logger.info(f"No new messages for channel: {channel}")
                    continue

                processed_messages = await process_new_messages(messages, channel, stance)
                if not processed_messages:
                    logger.warning(f"No processed messages for channel: {channel}")
                    continue

                messages_with_embeddings = get_embeddings(processed_messages, text_col='cleaned_message', model="embed-multilingual-v3.0")

                if messages_with_embeddings:
                    # Only upsert to Pinecone if not skipping
                    if not skip_pinecone:
                        upsert_to_pinecone(messages_with_embeddings, pine_index)
                    else:
                        logger.info(f"Skipping Pinecone upsert for {len(messages_with_embeddings)} messages from {channel}")
                    
                    new_last_id = max(msg['id'] for msg in messages_with_embeddings)
                    # Update only SQLite
                    update_channel_last_id(channel, new_last_id)
                    parsed_channels.append(channel)
                    if not last_id:
                        new_channels.append(channel)
                    logger.info(f"Successfully processed channel: {channel}")

                    # Incrementally store the collected messages in the SQLite database
                    store_in_sqlite(messages_with_embeddings)
                else:
                    logger.warning(f"No messages with embeddings for channel: {channel}")

            except Exception as e:
                logger.error(f"Error processing channel {channel}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                missed_channels.append(channel)
        
        # With this:
        log_summary_to_db(parsed_channels, missed_channels, new_channels)
    finally:
        # Cleanup connections
        try:
            co.close()  # Close Cohere client if it has a close method
        except:
            pass
        
        try:
            pc.close()  # Close Pinecone client if it has a close method
        except:
            pass
        
        # Allow time for gRPC connections to close gracefully
        await asyncio.sleep(1)

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
        print('end pinecone update')

# def lambda_handler(event, context):
#     return asyncio.get_event_loop().run_until_complete(async_handler(event, context))

if __name__ == "__main__":
    logger.info("Starting local execution...")
    asyncio.run(main())
    # Add small delay to allow gRPC connections to close
    time.sleep(1)
    logger.info("Execution completed!")