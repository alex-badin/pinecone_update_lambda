#!/usr/bin/env python3
print('Starting historic data download')

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
import logging
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import execute_values
import argparse

# Set up logging configuration
try:
    print('Setting up logging...')
    # Get absolute path for the log directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_directory = os.path.join(script_dir, 'logs')
    os.makedirs(log_directory, exist_ok=True)
    log_file = os.path.join(log_directory, 'historic_download.log')
    
    # Add immediate file logging for debugging
    with open(os.path.join(log_directory, 'startup_debug.log'), 'a') as f:
        f.write(f'\n--- New execution at {datetime.now()} ---\n')
        f.write(f'Script directory: {script_dir}\n')
        f.write(f'Python executable: {sys.executable}\n')
        f.write(f'Working directory: {os.getcwd()}\n')

    # Configure logging
    logger = logging.getLogger('HistoricDownload')
    logger.setLevel(logging.DEBUG)

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
    logging.getLogger('telethon').setLevel(logging.WARNING)  # Set telethon logging to WARNING
    
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
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    
    # PostgreSQL connection details
    PG_HOST = os.getenv('PG_HOST', 'localhost')
    PG_PORT = os.getenv('PG_PORT', '5432')
    PG_USER = os.getenv('PG_USER', 'postgres')
    PG_PASSWORD = os.getenv('PG_PASSWORD', '')
    PG_DATABASE = os.getenv('PG_DATABASE', 'telegram_data')
    
    required_vars = {
        "TG_API_ID": TG_API_ID,
        "TG_API_HASH": TG_API_HASH,
        "TG_SESSION_STRING": TG_SESSION_STRING,
        "COHERE_KEY": COHERE_KEY,
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "PG_HOST": PG_HOST,
        "PG_USER": PG_USER,
        "PG_PASSWORD": PG_PASSWORD,
        "PG_DATABASE": PG_DATABASE
    }
    missing = [var_name for var_name, value in required_vars.items() if not value]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
except Exception as e:
    logger.error(f"Error retrieving secrets: {str(e)}")
    raise

# Initialize clients
co = cohere.Client(COHERE_KEY)

# Initialize your summarizer
summarizer = Summarizer(GEMINI_API_KEY)

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

def process_messages(messages, channel, stance):
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

def get_embeddings(messages, text_col='cleaned_message', model="embed-multilingual-v3.0", batch_size=50):
    """Get embeddings for messages in batches to handle large datasets."""
    if not messages:
        return []
    
    all_messages = messages.copy()
    
    # Process in batches
    for i in range(0, len(all_messages), batch_size):
        batch = all_messages[i:i+batch_size]
        texts = [msg[text_col] for msg in batch if text_col in msg]
        
        if not texts:
            logger.warning(f"No '{text_col}' found in batch {i//batch_size + 1}")
            continue
            
        try:
            embeddings = get_embeddings_with_retry(texts, model)
            for msg, embedding in zip(batch, embeddings):
                msg['embeddings'] = embedding
            logger.info(f"Processed embeddings batch {i//batch_size + 1}/{(len(all_messages) + batch_size - 1)//batch_size}")
        except Exception as e:
            logger.error(f"Error getting embeddings for batch {i//batch_size + 1}: {str(e)}")
            # Continue with the next batch instead of failing completely
            for msg in batch:
                msg['embeddings'] = []
    
    return all_messages

@contextmanager
def get_pg_connection():
    """Context manager for PostgreSQL database connections."""
    conn = None
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            user=PG_USER,
            password=PG_PASSWORD,
            database=PG_DATABASE
        )
        yield conn
    finally:
        if conn:
            conn.close()

def create_postgres_tables():
    """Creates the necessary tables in PostgreSQL if they don't exist."""
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            # Check if pgvector extension exists
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
            if not cur.fetchone():
                try:
                    # Create pgvector extension if it doesn't exist
                    cur.execute("CREATE EXTENSION vector")
                    logger.info("Created pgvector extension")
                except Exception as e:
                    logger.warning(f"Could not create pgvector extension: {e}. You may need to install it manually.")
            
            # Check if tables exist
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('channels', 'messages', 'execution_logs')
            """)
            existing_tables = [row[0] for row in cur.fetchall()]
            
            # Create tables only if they don't exist
            if 'channels' not in existing_tables:
                logger.info("Creating channels table...")
                cur.execute('''
                    CREATE TABLE channels (
                        channel_name TEXT PRIMARY KEY,
                        last_id TEXT,
                        stance TEXT,
                        status TEXT DEFAULT 'Active',
                        last_sync TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
            
            if 'messages' not in existing_tables:
                logger.info("Creating messages table with vector support...")
                cur.execute('''
                    CREATE TABLE messages (
                        id TEXT PRIMARY KEY,
                        channel TEXT,
                        stance TEXT,
                        raw_message JSONB,
                        cleaned_message TEXT,
                        summary TEXT,
                        message_date TIMESTAMP,
                        views INTEGER,
                        embedding vector(1024)  -- Using pgvector type
                    )
                ''')
                
                # Add indexes
                cur.execute('''
                    CREATE INDEX IF NOT EXISTS idx_messages_date ON messages(message_date)
                ''')
                
                # Add index on channel for better filtering
                cur.execute('''
                    CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel)
                ''')
                
                # Add vector index for similarity search
                cur.execute('''
                    CREATE INDEX IF NOT EXISTS idx_messages_embedding ON messages USING ivfflat (embedding vector_l2_ops)
                ''')
            
            if 'execution_logs' not in existing_tables:
                logger.info("Creating execution_logs table...")
                cur.execute('''
                    CREATE TABLE execution_logs (
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
            logger.info("PostgreSQL tables check/creation completed")

def store_in_postgres(messages, batch_size=100):
    """Stores a list of processed messages in PostgreSQL using pgvector."""
    if not messages:
        logger.warning("No messages to store in PostgreSQL")
        return
        
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            for i in range(0, len(messages), batch_size):
                batch = messages[i:i+batch_size]
                values = []
                
                for msg in batch:
                    # Create a unique key combining channel and message ID
                    pk = f"{msg['channel']}_{msg['id']}"
                    
                    # Convert raw_message to JSON string
                    raw_message_json = json.dumps(msg['raw_message'], ensure_ascii=False, default=str)
                    
                    # Convert date to datetime object if it's a string
                    if isinstance(msg['date'], str):
                        date_obj = datetime.fromisoformat(msg['date'].replace('Z', '+00:00'))
                    else:
                        date_obj = msg['date']
                    
                    # Get embeddings as a list or empty list if not present
                    embedding = msg.get('embeddings', [])
                    
                    values.append((
                        pk,
                        msg['channel'],
                        msg['stance'],
                        raw_message_json,
                        msg['cleaned_message'],
                        msg['summary'],
                        date_obj,
                        msg.get('views', 0),
                        embedding
                    ))
                
                # Use execute_values for efficient batch insertion
                execute_values(
                    cur,
                    """
                    INSERT INTO messages
                    (id, channel, stance, raw_message, cleaned_message, summary, message_date, views, embedding)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        cleaned_message = EXCLUDED.cleaned_message,
                        summary = EXCLUDED.summary,
                        views = EXCLUDED.views,
                        embedding = EXCLUDED.embedding
                    """,
                    values
                )
                
                conn.commit()
                logger.info(f"Stored batch of {len(batch)} messages in PostgreSQL (total progress: {i+len(batch)}/{len(messages)})")

def get_active_channels():
    """Retrieve active channels from PostgreSQL."""
    logger.info("Retrieving active channels from PostgreSQL...")
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT channel_name, last_id, stance FROM channels WHERE status = 'Active'")
            return cur.fetchall()

def log_summary_to_db(parsed_channels, missed_channels, new_channels):
    """Log execution summary to PostgreSQL."""
    execution_date = datetime.now()
    primary_key = f"{execution_date.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('''
                INSERT INTO execution_logs
                (id, execution_date, parsed_channels_count, parsed_channels,
                 missed_channels_count, missed_channels,
                 new_channels_count, new_channels)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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

async def process_channel(channel, last_id, stance, start_date, end_date, index, total_channels, batch_size, skip_embeddings):
    """Process a channel's messages in a complete pipeline with streaming processing."""
    total_processed = 0
    
    # Get already processed message IDs for this channel in the date range
    existing_message_ids = set()
    with get_pg_connection() as conn:
        with conn.cursor() as cur:
            # Ensure timezone-aware dates for the query
            aware_start_date = start_date.replace(tzinfo=timezone.utc)
            aware_end_date = end_date.replace(tzinfo=timezone.utc)
            
            cur.execute("""
                SELECT SPLIT_PART(id, '_', 2)::TEXT 
                FROM messages 
                WHERE channel = %s 
                AND message_date BETWEEN %s AND %s
            """, (channel, aware_start_date, aware_end_date))
            
            existing_message_ids = {row[0] for row in cur.fetchall()}
            
    logger.info(f"Found {len(existing_message_ids)} already processed messages for channel {channel}")
    
    async with TelegramClient(StringSession(TG_SESSION_STRING), TG_API_ID, TG_API_HASH,
                            system_version="4.16.30-vxCUSTOM") as client:
        
        # Ensure timezone-aware dates
        start_date = start_date.replace(tzinfo=timezone.utc)
        end_date = end_date.replace(tzinfo=timezone.utc)
        
        logger.info(f"Parsing channel (#{index} of {total_channels}): {channel}, date range: {start_date} to {end_date}")
        
        # Process with streaming approach
        last_message_id = 0
        current_batch = []
        
        async for message in client.iter_messages(channel, limit=None, offset_id=last_message_id):
            if message.date < start_date:
                # We've gone past our start date, stop collecting
                break
                
            if message.date <= end_date:
                # Check if this message is already processed
                if str(message.id) not in existing_message_ids:
                    # Add to current batch
                    current_batch.append(message.to_dict())
                    
                    # Process batch when it reaches the batch size
                    if len(current_batch) >= batch_size:
                        # Process and store the batch
                        processed_count = await process_and_store_batch(
                            current_batch, channel, stance, skip_embeddings
                        )
                        total_processed += processed_count
                        
                        # Clear the batch for next round
                        current_batch = []
                        
                        # Small delay to avoid rate limiting
                        await asyncio.sleep(0.5)
                
                # Update last_message_id for potential pagination
                last_message_id = message.id
        
        # Process any remaining messages in the final batch
        if current_batch:
            processed_count = await process_and_store_batch(
                current_batch, channel, stance, skip_embeddings
            )
            total_processed += processed_count
    
    return total_processed

async def process_and_store_batch(messages_batch, channel, stance, skip_embeddings):
    """Helper function to process and store a batch of messages."""
    # Process batch
    processed_batch = process_messages(messages_batch, channel, stance)
    if not processed_batch:
        return 0
        
    # Get embeddings if not skipped
    if not skip_embeddings:
        processed_batch = get_embeddings(
            processed_batch, 
            text_col='cleaned_message',
            model="embed-multilingual-v3.0",
            batch_size=min(50, len(processed_batch))
        )
    else:
        for msg in processed_batch:
            msg['embeddings'] = []
    
    # Store batch in PostgreSQL
    store_in_postgres(processed_batch, batch_size=min(100, len(processed_batch)))
    
    logger.info(f"Channel {channel}: Processed and stored batch of {len(processed_batch)} messages")
    return len(processed_batch)

async def main():
    """Main function to download historic data."""
    parser = argparse.ArgumentParser(description='Download historic Telegram data')
    parser.add_argument('--start-date', type=str, help='Start date in YYYY-MM-DD format', required=True)
    parser.add_argument('--end-date', type=str, help='End date in YYYY-MM-DD format', default=None)
    parser.add_argument('--channels', type=str, help='Comma-separated list of channels to process (default: all active channels)', default=None)
    parser.add_argument('--batch-size', type=int, help='Batch size for processing messages', default=100)
    parser.add_argument('--skip-embeddings', action='store_true', help='Skip generating embeddings')
    
    args = parser.parse_args()
    
    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return
    
    logger.info(f"Starting historic data download from {start_date} to {end_date}")
    
    try:
        # Create PostgreSQL tables
        create_postgres_tables()
        
        parsed_channels = []
        missed_channels = []
        new_channels = []

        # Get channels from PostgreSQL or use specified channels
        if args.channels:
            # If channels are specified, use them
            channel_list = [ch.strip() for ch in args.channels.split(',')]
            channels = []
            for channel in channel_list:
                # Default stance to 'unknown' for manually specified channels
                channels.append((channel, None, 'unknown'))
        else:
            # Otherwise get from database
            channels = get_active_channels()
        
        # Shuffle to distribute load
        random.shuffle(channels)
        total_channels = len(channels)

        logger.info(f"Processing {total_channels} channels")

        for index, (channel, last_id, stance) in enumerate(channels, 1):
            try:
                logger.info(f"Processing channel {channel} ({index}/{total_channels})")
                
                # Process channel with complete pipeline
                total_processed = await process_channel(
                    channel, 
                    last_id, 
                    stance, 
                    start_date, 
                    end_date, 
                    index, 
                    total_channels,
                    args.batch_size,
                    args.skip_embeddings
                )
                
                if total_processed > 0:
                    parsed_channels.append(channel)
                    logger.info(f"Successfully processed channel: {channel}, stored {total_processed} messages")
                else:
                    logger.info(f"No messages found/processed for channel: {channel}")
                
                # Small delay between channels to avoid rate limiting
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error processing channel {channel}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                missed_channels.append(channel)
        
        # Log summary
        log_summary_to_db(parsed_channels, missed_channels, new_channels)
        logger.info(f"Historic data download completed. Processed {len(parsed_channels)} channels, missed {len(missed_channels)} channels.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        # Cleanup connections
        try:
            co.close()  # Close Cohere client if it has a close method
        except:
            pass
        
        # Allow time for connections to close gracefully
        await asyncio.sleep(1)

if __name__ == "__main__":
    logger.info("Starting historic data download...")
    asyncio.run(main())
    # Add small delay to allow connections to close
    time.sleep(1)
    logger.info("Historic data download completed!")