from datetime import datetime
import uuid

def log_summary_to_airtable(log_table, parsed_channels, missed_channels, new_channels):
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