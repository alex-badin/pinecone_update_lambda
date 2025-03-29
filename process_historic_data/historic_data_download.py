"""
args:
    --start-date: start date in YYYY-MM-DD format
    --end-date: end date in YYYY-MM-DD format
    --channels: comma-separated list of channels to process
    --channels-file: path to text file with channels (one per line)
    --output-dir: directory to save output JSON files
"""


#!/usr/bin/env python3
print('Starting historic data download')

import os
import sys
import traceback
from dotenv import load_dotenv
import asyncio
import json
from datetime import datetime, timezone
import time
import logging
from logging.handlers import RotatingFileHandler
import argparse

# Set up logging configuration
try:
    print('Setting up logging...')
    # Get absolute path for the log directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_directory = os.path.join(script_dir, 'logs')
    os.makedirs(log_directory, exist_ok=True)
    log_file = os.path.join(log_directory, 'historic_download.log')
    
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

from telethon import TelegramClient
from telethon.sessions import StringSession

# Load environment variables
try:
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct absolute path to .env file (one level up)
    env_path = os.path.join(os.path.dirname(script_dir), '.env')
    # Load environment variables with explicit path
    load_dotenv(env_path)
    
    TG_API_ID = os.getenv('TG_API_ID')
    TG_API_HASH = os.getenv('TG_API_HASH')
    TG_SESSION_STRING = os.getenv('TG_SESSION_STRING')
    
    required_vars = {
        "TG_API_ID": TG_API_ID,
        "TG_API_HASH": TG_API_HASH,
        "TG_SESSION_STRING": TG_SESSION_STRING
    }
    missing = [var_name for var_name, value in required_vars.items() if not value]
    if missing:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
except Exception as e:
    logger.error(f"Error retrieving environment variables: {str(e)}")
    raise

async def download_channel_messages(client, channel, start_date, end_date):
    """Download messages from a Telegram channel within the specified date range."""
    messages = []
    
    # Ensure timezone-aware dates
    start_date = start_date.replace(tzinfo=timezone.utc)
    end_date = end_date.replace(tzinfo=timezone.utc)
    
    logger.info(f"Downloading messages from channel: {channel}, date range: {start_date} to {end_date}")
    
    try:
        async for message in client.iter_messages(channel, limit=None):
            if message.date < start_date:
                # We've gone past our start date, stop collecting
                break
                
            if message.date <= end_date:
                # Add message to our collection
                messages.append(message.to_dict())
                
                # Log progress periodically
                if len(messages) % 100 == 0:
                    logger.info(f"Downloaded {len(messages)} messages from {channel}")
    except Exception as e:
        logger.error(f"Error downloading messages from {channel}: {str(e)}")
        logger.error(traceback.format_exc())
    
    logger.info(f"Completed download of {len(messages)} messages from {channel}")
    return messages

def save_to_json(data, filename):
    """Save data to a JSON file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Data saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {str(e)}")
        return False

async def main():
    """Main function to download historic data."""
    parser = argparse.ArgumentParser(description='Download historic Telegram data')
    parser.add_argument('--start-date', type=str, help='Start date in YYYY-MM-DD format', required=True)
    parser.add_argument('--end-date', type=str, help='End date in YYYY-MM-DD format', default=None)
    
    # Create a mutually exclusive group for channel specification
    channel_group = parser.add_mutually_exclusive_group(required=True)
    channel_group.add_argument('--channels', type=str, help='Comma-separated list of channels to process')
    channel_group.add_argument('--channels-file', type=str, help='Path to text file with channels (one per line)')
    
    parser.add_argument('--output-dir', type=str, help='Directory to save output JSON files', default='output')
    
    args = parser.parse_args()
    
    # Parse dates
    try:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else datetime.now()
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return
    
    logger.info(f"Starting historic data download from {start_date} to {end_date}")
    
    # Create output directory
    output_dir = os.path.join(script_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get channels from either command line or file
    if args.channels:
        channels = [ch.strip() for ch in args.channels.split(',')]
        logger.info(f"Using {len(channels)} channels from command line argument")
    else:
        try:
            with open(args.channels_file, 'r', encoding='utf-8') as f:
                channels = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(channels)} channels from {args.channels_file}")
        except Exception as e:
            logger.error(f"Error reading channels file: {str(e)}")
            return
    
    # Create a single JSON structure to hold all data
    all_data = {}
    
    try:
        # Initialize Telegram client
        async with TelegramClient(StringSession(TG_SESSION_STRING), TG_API_ID, TG_API_HASH,
                                system_version="4.16.30-vxCUSTOM") as client:
            
            for channel in channels:
                try:
                    logger.info(f"Processing channel: {channel}")
                    
                    # Check if channel file already exists
                    channel_filename = os.path.join(output_dir, f"{channel.replace('/', '_')}.json")
                    if os.path.exists(channel_filename):
                        logger.info(f"File already exists for channel {channel}. Skipping download.")
                        
                        # Load existing data for the combined file
                        try:
                            with open(channel_filename, 'r', encoding='utf-8') as f:
                                existing_data = json.load(f)
                                all_data[channel] = existing_data.get(channel, [])
                            logger.info(f"Loaded existing data for channel {channel}")
                        except Exception as e:
                            logger.error(f"Error loading existing data for {channel}: {str(e)}")
                        
                        continue
                    
                    # Download messages
                    messages = await download_channel_messages(client, channel, start_date, end_date)
                    
                    if messages:
                        # Add to the combined data structure
                        all_data[channel] = messages
                        
                        # Also save individual channel data
                        channel_data = {channel: messages}
                        save_to_json(channel_data, channel_filename)
                    
                    # Small delay between channels to avoid rate limiting
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error processing channel {channel}: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # Save all data to a combined file
            combined_filename = os.path.join(output_dir, f"all_channels_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            save_to_json(all_data, combined_filename)
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger.info("Starting historic data download...")
    asyncio.run(main())
    logger.info("Historic data download completed!")