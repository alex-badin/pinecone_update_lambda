import os
from dotenv import load_dotenv
from pyairtable import Api
import sqlite3
from datetime import datetime
from contextlib import contextmanager

@contextmanager
def get_db_connection(db_path="collected_messages.db"):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

def sync_channels_with_airtable():
    # Load environment variables
    load_dotenv()
    
    # Initialize Airtable client
    airtable_api = Api(os.getenv('AIRTABLE_API_TOKEN'))
    table = airtable_api.table(os.getenv('AIRTABLE_BASE_ID'), os.getenv('AIRTABLE_TABLE_NAME'))
    
    # Get channels from Airtable
    airtable_records = table.all()
    airtable_channels = {
        record['fields']['channel_name']: {
            'stance': record['fields'].get('stance', ''),
            'status': record['fields'].get('status', 'Active')
        }
        for record in airtable_records
    }
    
    with get_db_connection() as conn:
        c = conn.cursor()
        
        # Get current channels from SQLite
        c.execute("SELECT channel_name, stance, status FROM channels")
        sqlite_channels = {row[0]: {'stance': row[1], 'status': row[2]} for row in c.fetchall()}
        
        # Update or insert channels from Airtable
        for channel, data in airtable_channels.items():
            if channel not in sqlite_channels:
                # New channel
                c.execute("""
                    INSERT INTO channels (channel_name, stance, status, last_sync)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (channel, data['stance'], data['status']))
            elif (sqlite_channels[channel]['stance'] != data['stance'] or 
                  sqlite_channels[channel]['status'] != data['status']):
                # Update existing channel
                c.execute("""
                    UPDATE channels 
                    SET stance = ?, status = ?, last_sync = CURRENT_TIMESTAMP
                    WHERE channel_name = ?
                """, (data['stance'], data['status'], channel))
        
        # Mark channels as inactive if they're not in Airtable anymore
        for channel in sqlite_channels:
            if channel not in airtable_channels:
                c.execute("""
                    UPDATE channels 
                    SET status = 'Inactive', last_sync = CURRENT_TIMESTAMP
                    WHERE channel_name = ?
                """, (channel,))
        
        conn.commit()

if __name__ == "__main__":
    sync_channels_with_airtable()