import time
from datetime import datetime
import uuid

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
    print(f"Upserted {len(vectors)} records. Last id: {vectors[-1]['id']}")

def update_airtable_last_id(table, channel, last_id):
    matching_records = table.all(formula=f"{{channel_name}}='{channel}'")
    if matching_records:
        record_id = matching_records[0]['id']
        table.update(record_id, {'last_id': int(last_id)})
    else:
        print(f"No matching record found for channel {channel}")

def get_active_channels(airtable_api, base_id, table_name):
    table = airtable_api.table(base_id, table_name)
    records = table.all()
    df_channels = [record['fields'] for record in records]
    df_channels = [record for record in df_channels if record['status'] == 'Active']
    return [(record['channel_name'], record['last_id'], record['stance']) for record in df_channels] 