print('start pinecone update lambda')
import os
import asyncio
import random
import json
import uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

import boto3
from botocore.exceptions import ClientError
import pandas as pd
from pyairtable import Api
from telethon import TelegramClient
from telethon.sessions import StringSession
import cohere
from pinecone import Pinecone
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import re
import unicodedata
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

# Initialize AWS clients
secrets_manager = boto3.client('secretsmanager', region_name=os.environ.get('AWS_REGION', 'eu-central-1'))

# Function to retrieve secrets
def get_secret(secret_name):
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
    secrets = get_secret(os.environ['SECRET_NAME'])
    API_ID = secrets['api_id']
    API_HASH = secrets['api_hash']
    SESSION_STRING = secrets['session_string']
    COHERE_KEY = secrets['cohere_key']
    PINE_KEY = secrets['pine_key']
    AIRTABLE_API_TOKEN = secrets['airtable_api_token']
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

async def get_new_messages(channel, last_id, start_date):
    async with TelegramClient(StringSession(SESSION_STRING), API_ID, API_HASH,
                              system_version="4.16.30-vxCUSTOM") as client:
        data = []
        try:
            offset_id = int(last_id) if last_id else 0
        except ValueError:
            offset_id = 0
        
        async for message in client.iter_messages(channel, reverse=True,
                                                  offset_id=offset_id,
                                                  offset_date=start_date):
            data.append(message.to_dict())

    print(f"Channel: {channel}, N of new messages: {len(data)}")
    return pd.DataFrame(data) if data else None

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

def summarize(text, language="russian", sentences_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join([str(sentence) for sentence in summary])

def process_new_messages(df, channel, stance):
    # add channel name & stance
    df.loc[:, 'channel'] = channel
    df.loc[:, 'stance'] = stance
    df.loc[:, 'cleaned_message'] = df['message'].apply(clean_text) #remove emojis, urls, foreign agent text
    df.drop_duplicates(subset=['id'], inplace = True) # remove duplicates
    df = df[~df.cleaned_message.str.len().between(0, 30)].copy() #remove empty or too short messages
    # summarize cleaned_messages: 3 sentences if length > 750, 4 sentences if length > 1500
    df.loc[:, 'summary'] = df['cleaned_message'].apply(lambda x: summarize(x, sentences_count=3) if len(x) > 750 else summarize(x, sentences_count=4) if len(x) > 500 else x)
    return df

def get_embeddings_df(df, text_col='summary', model="embed-multilingual-v3.0"):
    df['embeddings'] = co.embed(texts=df[text_col].tolist(), model=model, input_type="clustering").embeddings
    return df

@retry(stop=stop_after_attempt(4), wait=wait_random_exponential(multiplier=1, max=10))
def upsert_to_pinecone(df, index, batch_size=100):
    # create df for pinecone
    meta_col = ['cleaned_message', 'summary', 'stance', 'channel', 'date', 'views']
    #rename embeddings to values
    df4pinecone = df[meta_col+['id', 'embeddings']].copy()
    df4pinecone = df4pinecone.rename(columns={'embeddings': 'values'})
    # convert date to integer (as pinecone doesn't support datetime)
    df4pinecone['date'] = df4pinecone['date'].apply(lambda x: int(time.mktime(x.timetuple())))
    # id as channel_id + message_id (to avoid duplication and easier identification)
    df4pinecone['id'] = df4pinecone['channel'] + '_' + df4pinecone['id'].astype(str)
    # convert to pinecone format
    df4pinecone['metadata'] = df4pinecone[meta_col].to_dict('records')
    df4pinecone = df4pinecone[['id', 'values', 'metadata']]
    if df4pinecone.empty:
        print("DataFrame is empty. No records to upsert.")
        return
    for i in range(0, df4pinecone.shape[0], batch_size):
        index.upsert(vectors=df4pinecone.iloc[i:i+batch_size].to_dict('records'))
    print(f"Upserted {df4pinecone.shape[0]} records. Last id: {df4pinecone.iloc[-1]['id']}")

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
    start_date = datetime.now() - timedelta(days=4)
    parsed_channels = []
    missed_channels = []
    new_channels = []

    records = table.all()
    df_channels = pd.DataFrame([record['fields'] for record in records])
    df_channels = df_channels[df_channels['status'] == 'Active']

    channels = df_channels[['channel_name', 'last_id', 'stance']].values.tolist()
    random.shuffle(channels)

    for channel, last_id, stance in channels:
        try:
            df = await get_new_messages(channel, last_id, start_date)
            if df is None or df.empty:
                continue

            df = process_new_messages(df, channel, stance)
            df = get_embeddings_df(df, text_col='message', model="embed-multilingual-v3.0")

            upsert_to_pinecone(df, pine_index)

            new_last_id = df['id'].max()
            update_airtable_last_id(table, channel, new_last_id)

            parsed_channels.append(channel)
            if not last_id:
                new_channels.append(channel)

            print(f"Successfully processed channel: {channel}")

        except Exception as e:
            print(f"Error processing channel {channel}: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error details: {e.args}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            missed_channels.append(channel)

    log_summary_to_airtable(parsed_channels, missed_channels, new_channels)

async def lambda_handler(event, context):
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

print('end pinecone update lambda')