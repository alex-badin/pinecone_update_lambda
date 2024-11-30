print('start pinecone update lambda')

import asyncio
from datetime import datetime, timezone, timedelta
import random
import traceback

from src.config import load_config
from src.clients import initialize_clients
from src.telegram import get_new_messages
from src.prompt_manager import PromptManager
from src.text_processing import TextProcessor
from src.embeddings import get_embeddings
from src.storage import (
    upsert_to_pinecone,
    update_airtable_last_id,
    get_active_channels
)
from src.logging import log_summary_to_airtable

async def main():
    # Load configuration and initialize clients
    config = load_config()
    clients = initialize_clients(config)
    
    # Initialize prompt manager and text processor
    prompt_manager = PromptManager()
    text_processor = TextProcessor(prompt_manager)
    
    start_date = datetime.now(timezone.utc) - timedelta(days=4)
    print(f"Start date for messages: {start_date}")
    
    parsed_channels = []
    missed_channels = []
    new_channels = []

    # Get active channels from Airtable
    channels = get_active_channels(
        clients['airtable'],
        config['AIRTABLE_BASE_ID'],
        config['AIRTABLE_TABLE_NAME']
    )
    random.shuffle(channels)

    print(f"Channels to parse: {len(channels)}: {channels}")
    for channel, last_id, stance in channels:
        try:
            messages = await get_new_messages(channel, last_id, start_date, config)
            if not messages:
                print(f"No new messages for channel: {channel}")
                continue

            processed_messages = text_processor.process_new_messages(
                messages, 
                channel, 
                stance, 
                clients['gemini']
            )
            if not processed_messages:
                print(f"No processed messages for channel: {channel}")
                continue

            messages_with_embeddings = get_embeddings(
                processed_messages,
                clients['cohere'],
                text_col='cleaned_message'
            )

            if messages_with_embeddings:
                upsert_to_pinecone(messages_with_embeddings, clients['pine_index'])

                new_last_id = max(msg['id'] for msg in messages_with_embeddings)
                table = clients['airtable'].table(config['AIRTABLE_BASE_ID'], config['AIRTABLE_TABLE_NAME'])
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
            print(f"Traceback: {traceback.format_exc()}")
            missed_channels.append(channel)

    # Log summary
    log_table = clients['airtable'].table(config['AIRTABLE_BASE_ID'], config['AIRTABLE_LOG_TABLE'])
    log_summary_to_airtable(log_table, parsed_channels, missed_channels, new_channels)

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