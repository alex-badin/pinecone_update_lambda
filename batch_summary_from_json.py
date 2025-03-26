"""
Arguments:
    --import-only: Only import messages from JSON files
    --summarize-only: Only summarize messages already in the database
Example usage:
    python batch_summary_from_json.py --import-only
    python batch_summary_from_json.py --summarize-only
"""


import json
import pandas as pd
import os
import asyncio
import time
import sqlite3
from tqdm import tqdm
import sys
import glob
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the src directory to the path so we can import the summarizer
sys.path.append('.')
from src.summarizer import Summarizer

# Database setup - using relative paths
results_db_path = os.path.join('temp', 'summarized_messages.db')

def init_results_db():
    """Initialize the SQLite database for storing messages and summaries"""
    os.makedirs(os.path.dirname(results_db_path), exist_ok=True)
    conn = sqlite3.connect(results_db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS message_summaries (
            message_id TEXT,
            source TEXT,
            original_message TEXT,
            summary TEXT,
            is_digest BOOLEAN,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            date TEXT,
            views INTEGER,
            forwards INTEGER,
            PRIMARY KEY (source, message_id)
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Results database initialized at {results_db_path}")

def save_batch_to_db(message_ids, sources, messages, summaries_with_digest):
    """Save a batch of summarized messages to the database"""
    conn = sqlite3.connect(results_db_path)
    cursor = conn.cursor()
    
    for i, (message_id, source, message) in enumerate(zip(message_ids, sources, messages)):
        summary, is_digest = summaries_with_digest[i]
        cursor.execute(
            "UPDATE message_summaries SET summary = ?, is_digest = ?, processed_at = CURRENT_TIMESTAMP WHERE source = ? AND message_id = ?",
            (summary, is_digest, source, message_id)
        )
    
    conn.commit()
    conn.close()

def import_json_files():
    """Import messages from JSON and pickle files in the data_historic folder"""
    data_dir = os.path.join('data_historic')
    json_files = glob.glob(os.path.join(data_dir, '*.json'))
    pickle_files = glob.glob(os.path.join(data_dir, '*.pkl'))
    all_files = json_files + pickle_files
    
    if not all_files:
        print(f"No JSON or pickle files found in {data_dir}")
        return 0
    
    print(f"Found {len(all_files)} files to process ({len(json_files)} JSON, {len(pickle_files)} pickle)")
    
    conn = sqlite3.connect(results_db_path)
    cursor = conn.cursor()
    
    total_imported = 0
    total_skipped = 0
    
    for file_path in tqdm(all_files, desc="Importing files"):
        try:
            # Load data based on file extension
            if file_path.endswith('.json'):
                with open(file_path, 'r') as file:
                    data = json.load(file)
            elif file_path.endswith('.pkl'):
                import pickle
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
            
            file_imported = 0
            file_skipped = 0
            
            # Process each source and its messages in the file
            for source, messages in data.items():
                # Add a progress bar for each source's messages
                for item in tqdm(messages, desc=f"Processing {source}", leave=False):
                    if 'message' in item and item['message'] and 'id' in item:
                        message_id = item['id']
                        message = item['message']
                        
                        # Extract additional fields with defaults if not present
                        date = item.get('date', None)
                        views = item.get('views', 0)
                        forwards = item.get('forwards', 0)
                        
                        # Check if this source+message_id combination already exists
                        cursor.execute("SELECT 1 FROM message_summaries WHERE source = ? AND message_id = ?", 
                                      (source, message_id))
                        if cursor.fetchone():
                            file_skipped += 1
                            continue
                        
                        # Insert new message with NULL summary and is_digest
                        cursor.execute(
                            """INSERT INTO message_summaries 
                               (message_id, source, original_message, summary, is_digest, date, views, forwards) 
                               VALUES (?, ?, ?, NULL, NULL, ?, ?, ?)""",
                            (message_id, source, message, date, views, forwards)
                        )
                        file_imported += 1
            
            conn.commit()
            total_imported += file_imported
            total_skipped += file_skipped
            print(f"Processed {file_path}: imported {file_imported}, skipped {file_skipped}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    conn.close()
    print(f"Total messages imported: {total_imported}")
    print(f"Total messages skipped (already in database): {total_skipped}")
    return total_imported

async def summarize_messages():
    """Summarize messages that don't have summaries yet"""
    # Get your API key from environment variable or set it directly
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        api_key = input("Please enter your Gemini API key: ")

    # Initialize the summarizer with relative path
    cache_db_path = os.path.join('temp', 'summary_cache.db')
    summarizer = Summarizer(api_key=api_key, cache_db=cache_db_path, batch_size=100)
    
    # Get messages that need summarization
    conn = sqlite3.connect(results_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT message_id, source, original_message FROM message_summaries WHERE summary IS NULL")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("No messages need summarization")
        return 0
    
    message_ids = [row[0] for row in rows]
    sources = [row[1] for row in rows]
    messages = [row[2] for row in rows]
    
    print(f"Found {len(messages)} messages to summarize")
    
    # Create a progress bar
    pbar = tqdm(total=len(messages), desc="Summarizing messages")
    
    # Callback function to update progress bar
    def update_progress(n=1):
        pbar.update(len(batch_messages))
    
    # Process messages in batches with incremental saving
    batch_size = summarizer.batch_size
    total_batches = (len(messages) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(messages))
        
        batch_messages = messages[start_idx:end_idx]
        batch_ids = message_ids[start_idx:end_idx]
        batch_sources = sources[start_idx:end_idx]
        
        # Process this batch
        batch_results = await summarizer.summarize_batch(
            batch_messages, 
            max_summary_length=500,
            length_threshold=750,
            progress_callback=update_progress,
            batch_delay=0.5
        )
        
        # Save this batch to the database
        save_batch_to_db(batch_ids, batch_sources, batch_messages, batch_results)
        
        # Add delay between batches if needed
        if batch_idx < total_batches - 1:
            await asyncio.sleep(1)  # 1 second delay between saving batches
    
    pbar.close()
    
    # Get statistics on summarized messages
    conn = sqlite3.connect(results_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM message_summaries WHERE summary IS NOT NULL")
    total_summarized = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM message_summaries WHERE is_digest = 1")
    total_digests = cursor.fetchone()[0]
    conn.close()
    
    print(f"\nTotal messages summarized: {total_summarized}")
    print(f"Messages identified as digests: {total_digests}")
    
    return len(rows)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process and summarize messages from JSON files')
    parser.add_argument('--import-only', action='store_true', help='Only import messages from JSON files')
    parser.add_argument('--summarize-only', action='store_true', help='Only summarize messages already in the database')
    return parser.parse_args()

def main():
    """Main function to run steps based on command line arguments"""
    args = parse_arguments()
    
    # Initialize the database
    init_results_db()
    
    # Determine which steps to run
    run_import = not args.summarize_only
    run_summarize = not args.import_only
    
    # Step 1: Import JSON files
    imported = 0
    if run_import:
        print("\n=== STEP 1: IMPORTING JSON FILES ===")
        imported = import_json_files()
    
    # Step 2: Summarize messages
    if run_summarize:
        print("\n=== STEP 2: SUMMARIZING MESSAGES ===")
        if imported > 0 or args.summarize_only or input("No new messages imported. Proceed with summarization anyway? (y/n): ").lower() == 'y':
            start_time = time.time()
            asyncio.run(summarize_messages())
            end_time = time.time()
            print(f"\nSummarization completed in {end_time - start_time:.2f} seconds")
    
    # Display a sample of the results
    conn = sqlite3.connect(results_db_path)
    sample_df = pd.read_sql_query("SELECT * FROM message_summaries LIMIT 5", conn)
    conn.close()
    print("\nSample of results:")
    print(sample_df)

if __name__ == "__main__":
    main()