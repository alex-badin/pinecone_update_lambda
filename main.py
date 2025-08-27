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

# Step 5: Connection pool for better connection reuse
_db_connections = {}

def get_pooled_connection(db_path):
    """Get or create a reusable connection for the database path"""
    if db_path not in _db_connections:
        conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
        
        # Apply performance pragmas
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
        
        _db_connections[db_path] = conn
        logger.info(f"Created new pooled connection for {db_path}")
    
    return _db_connections[db_path]

def close_all_connections():
    """Close all pooled connections"""
    for db_path, conn in _db_connections.items():
        try:
            conn.close()
            logger.info(f"Closed connection for {db_path}")
        except:
            pass
    _db_connections.clear()

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
        try:
            offset_id = int(last_id) if last_id else 0
        except ValueError:
            offset_id = 0
        
        start_date = start_date.replace(tzinfo=timezone.utc) # Ensure timezone-aware
        logger.info(f"Parsing channel (#{channel_index} of {total_channels}): {channel}, start date: {start_date}, last id: {last_id}, offset_id: {offset_id}")
        
        # Add progress tracking
        message_count = 0
        last_log_time = time.time()
        log_interval = 30  # Log progress every 30 seconds
        
        async for message in client.iter_messages(channel, reverse=True, offset_id=offset_id):
            if message.date < start_date: # for new channels with no last_id or if IGNORE_LAST_ID=True
                continue
            data.append(message.to_dict())
            
            # Update progress counter and log periodically
            message_count += 1
            current_time = time.time()
            if current_time - last_log_time > log_interval:
                logger.info(f"Channel {channel}: Downloaded {message_count} messages so far...")
                last_log_time = current_time
                
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
        logger.warning(f"Number of empty messages skipped for channel {channel}: {empty_message_count}")
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
        embeddings = get_embeddings_with_retry(texts, model)
        for msg, embedding in zip(messages, embeddings):
            msg['embeddings'] = embedding
    except Exception as e:
        logger.error(f"Error getting embeddings: {str(e)}")
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
    logger.info(f"Upserted {len(vectors)} records to Pinecone. Last id: {vectors[-1]['id']}")

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
    """Context manager for database connections with retry logic and performance optimization"""
    attempt = 0
    while attempt < max_attempts:
        try:
            conn = sqlite3.connect(db_path, timeout=30.0)  # Increased timeout
            
            # Step 2: Enable WAL mode and performance pragmas
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
            
            yield conn
            return
        except sqlite3.OperationalError as e:
            attempt += 1
            if attempt == max_attempts:
                raise
            logger.warning(f"Database locked, retrying... (attempt {attempt}/{max_attempts})")
            time.sleep(2)  # Increased wait time
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
    """Creates the messages table in the messages database with performance indexes."""
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
                embeddings TEXT
            )
        ''')
        
        # Create indexes for better query performance (Step 1)
        c.execute('CREATE INDEX IF NOT EXISTS idx_channel ON messages(channel)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_date ON messages(date)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_channel_date ON messages(channel, date)')
        
        conn.commit()

def store_in_sqlite(messages):
    """Stores a list of processed messages in the SQLite database with optimized insertions."""
    if not messages:
        return
        
    # Step 5: Use pooled connection for better performance
    conn = get_pooled_connection(MESSAGES_DB)
    c = conn.cursor()
    
    try:
        # Step 3: Check for existing messages to avoid duplicates (faster than INSERT OR REPLACE)
        message_ids = [f"{msg['channel']}_{msg['id']}" for msg in messages]
        placeholders = ','.join('?' * len(message_ids))
        c.execute(f"SELECT id FROM messages WHERE id IN ({placeholders})", message_ids)
        existing_ids = {row[0] for row in c.fetchall()}
        
        # Filter out existing messages
        new_messages = [msg for msg in messages if f"{msg['channel']}_{msg['id']}" not in existing_ids]
        
        if not new_messages:
            logger.info("No new messages to store (all already exist)")
            return
        
        to_insert = []
        for msg in new_messages:
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
        
        # Step 3: Use regular INSERT (much faster than INSERT OR REPLACE)
        c.executemany('''
            INSERT INTO messages
            (id, channel, stance, raw_message, cleaned_message, summary, date, views, embeddings)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', to_insert)
        conn.commit()
        logger.info(f"Stored {len(to_insert)} new messages in SQLite database at {MESSAGES_DB} (skipped {len(existing_ids)} existing)")
    except Exception as e:
        logger.error(f"Error storing messages in SQLite: {str(e)}")
        conn.rollback()
        raise

def get_active_channels():
    """Retrieve active channels from SQLite."""
    logger.info("Retrieving active channels from SQLite...")
    # Step 5: Use pooled connection
    conn = get_pooled_connection(CHANNELS_DB)
    c = conn.cursor()
    c.execute("SELECT channel_name, last_id, stance FROM channels WHERE status = 'Active'")
    return c.fetchall()

def update_channel_last_id(channel, last_id):
    """Update the last_id for a channel in SQLite."""
    # Step 5: Use pooled connection
    conn = get_pooled_connection(CHANNELS_DB)
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
        
        start_date = datetime.now(timezone.utc) - timedelta(days=int(os.getenv('DAYS_TO_PARSE', 2)))
        logger.info(f"Starting main execution with start date: {start_date}")
        
        parsed_channels = []
        missed_channels = []
        new_channels = []

        # Get channels from SQLite
        channels = get_active_channels()
        random.shuffle(channels)
        total_channels = len(channels)

        logger.info(f"Processing {len(channels)} channels: {channels}")

        # Step 4: Collect messages in larger batches for more efficient database operations
        all_messages_batch = []
        batch_size = int(os.getenv('DB_BATCH_SIZE', 1000))  # Configurable batch size
        
        for index, (channel, last_id, stance) in enumerate(channels, 1):
            try:
                messages = await get_new_messages(channel, last_id, start_date, index, total_channels)
                if not messages:
                    logger.info(f"No new messages for channel: {channel}")
                    continue

                processed_messages = process_new_messages(messages, channel, stance)
                if not processed_messages:
                    logger.warning(f"No processed messages for channel: {channel}")
                    continue

                messages_with_embeddings = get_embeddings(processed_messages, text_col='cleaned_message', model="embed-multilingual-v3.0")

                if messages_with_embeddings:
                    upsert_to_pinecone(messages_with_embeddings, pine_index)
                    new_last_id = max(msg['id'] for msg in messages_with_embeddings)
                    # Update only SQLite
                    update_channel_last_id(channel, new_last_id)
                    parsed_channels.append(channel)
                    if not last_id:
                        new_channels.append(channel)
                    logger.info(f"Successfully processed channel: {channel}")

                    # Step 4: Add to batch instead of immediate storage
                    all_messages_batch.extend(messages_with_embeddings)
                    
                    # Store batch when it reaches the configured size
                    if len(all_messages_batch) >= batch_size:
                        store_in_sqlite(all_messages_batch)
                        all_messages_batch = []  # Clear the batch
                        
                else:
                    logger.warning(f"No messages with embeddings for channel: {channel}")

            except Exception as e:
                logger.error(f"Error processing channel {channel}: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                missed_channels.append(channel)
        
        # Step 4: Store any remaining messages in the final batch
        if all_messages_batch:
            store_in_sqlite(all_messages_batch)
        
        # Log execution summary
        log_summary_to_db(parsed_channels, missed_channels, new_channels)
    finally:
        # Step 5: Close all pooled database connections
        close_all_connections()
        
        # Cleanup connections
        try:
            co.close()  # Close Cohere client if it has a close method
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