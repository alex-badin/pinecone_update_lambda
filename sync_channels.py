import os
import argparse
from dotenv import load_dotenv
from pyairtable import Api
import sqlite3
import psycopg  # Changed from psycopg2 to psycopg
from datetime import datetime
from contextlib import contextmanager

@contextmanager
def get_db_connection(db_path="collected_messages.db"):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def get_pg_connection():
    """Context manager for PostgreSQL database connections."""
    conn = None
    try:
        conn = psycopg.connect(  # Changed from psycopg2 to psycopg
            host=os.getenv('PG_HOST', 'localhost'),
            port=os.getenv('PG_PORT', '5432'),
            user=os.getenv('PG_USER', 'postgres'),
            password=os.getenv('PG_PASSWORD', ''),
            database=os.getenv('PG_DATABASE', 'telegram_data')
        )
        yield conn
    finally:
        if conn:
            conn.close()

def sync_channels_with_airtable(use_sqlite=True, use_postgres=True):
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
    
    # Sync with SQLite if enabled
    if use_sqlite:
        sync_with_sqlite(airtable_channels)
    
    # Sync with PostgreSQL if enabled
    if use_postgres:
        sync_with_postgres(airtable_channels)

def sync_with_sqlite(airtable_channels):
    print("Syncing channels with SQLite...")
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
    print("SQLite sync completed.")

def sync_with_postgres(airtable_channels):
    print("Syncing channels with PostgreSQL...")
    try:
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                # Ensure the channels table exists
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS channels (
                        channel_name TEXT PRIMARY KEY,
                        last_id TEXT,
                        stance TEXT,
                        status TEXT DEFAULT 'Active',
                        last_sync TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
                
                # Get current channels from PostgreSQL
                cur.execute("SELECT channel_name, stance, status FROM channels")
                pg_channels = {row[0]: {'stance': row[1], 'status': row[2]} for row in cur.fetchall()}
                
                # Update or insert channels from Airtable
                for channel, data in airtable_channels.items():
                    if channel not in pg_channels:
                        # New channel
                        cur.execute("""
                            INSERT INTO channels (channel_name, stance, status, last_sync)
                            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                        """, (channel, data['stance'], data['status']))
                    elif (pg_channels[channel]['stance'] != data['stance'] or 
                          pg_channels[channel]['status'] != data['status']):
                        # Update existing channel
                        cur.execute("""
                            UPDATE channels 
                            SET stance = %s, status = %s, last_sync = CURRENT_TIMESTAMP
                            WHERE channel_name = %s
                        """, (data['stance'], data['status'], channel))
                
                # Mark channels as inactive if they're not in Airtable anymore
                for channel in pg_channels:
                    if channel not in airtable_channels:
                        cur.execute("""
                            UPDATE channels 
                            SET status = 'Inactive', last_sync = CURRENT_TIMESTAMP
                            WHERE channel_name = %s
                        """, (channel,))
                
                conn.commit()
        print("PostgreSQL sync completed.")
    except Exception as e:
        print(f"Error syncing with PostgreSQL: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sync channels from Airtable to databases')
    parser.add_argument('--sqlite', action='store_true', help='Sync with SQLite database')
    parser.add_argument('--postgres', action='store_true', help='Sync with PostgreSQL database')
    
    args = parser.parse_args()
    
    # If no specific database is selected, sync with both
    if not args.sqlite and not args.postgres:
        sync_channels_with_airtable(use_sqlite=True, use_postgres=True)
    else:
        sync_channels_with_airtable(use_sqlite=args.sqlite, use_postgres=args.postgres)