"""
Arguments:
    --import-only: Only import messages from JSON files
Example usage:
    python batch_summary_from_json.py
"""

import json
import pandas as pd
import os
import sqlite3
from tqdm import tqdm
import sys
import glob
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database setup - using relative paths
results_db_path = os.path.join('json_files', 'summarized_messages.db')

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

def import_json_files():
    """Import messages from JSON files in the data_historic folder"""
    data_dir = os.path.join('data_historic')
    json_files = glob.glob(os.path.join(data_dir, '*.json'))
    
    if not json_files:
        print(f"No JSON files found in {data_dir}")
        return 0
    
    print(f"Found {len(json_files)} JSON files to process")
    
    conn = sqlite3.connect(results_db_path)
    cursor = conn.cursor()
    
    total_imported = 0
    total_skipped = 0
    
    for json_file in tqdm(json_files, desc="Importing JSON files"):
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
            
            file_imported = 0
            file_skipped = 0
            
            # Process each source and its messages in the JSON file
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
            print(f"Processed {json_file}: imported {file_imported}, skipped {file_skipped}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    conn.close()
    print(f"Total messages imported: {total_imported}")
    print(f"Total messages skipped (already in database): {total_skipped}")
    return total_imported

def main():
    """Main function to run the import process"""
    # Initialize the database
    init_results_db()
    
    print("\n=== IMPORTING JSON FILES ===")
    imported = import_json_files()
    
    # Display a sample of the results
    if imported > 0:
        conn = sqlite3.connect(results_db_path)
        sample_df = pd.read_sql_query("SELECT * FROM message_summaries LIMIT 5", conn)
        conn.close()
        print("\nSample of results:")
        print(sample_df)
    else:
        print("\nNo new messages were imported.")

if __name__ == "__main__":
    main()