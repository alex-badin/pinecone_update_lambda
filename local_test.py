import asyncio
import os
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

from src.config import load_config
from src.clients import initialize_clients
from src.prompt_manager import PromptManager
from src.text_processing import TextProcessor
from src.telegram import get_new_messages

async def test_single_channel():
    # Load environment variables from .env file
    load_dotenv()
    
    # Load configuration
    config = load_config()
    
    # Initialize clients
    clients = initialize_clients(config)
    
    # Initialize prompt manager and text processor
    prompt_manager = PromptManager()
    text_processor = TextProcessor(prompt_manager)
    
    # Test parameters
    test_channel = "actual_channel_name"  # Replace with your test channel
    start_date = datetime.now(timezone.utc) - timedelta(hours=1)  # Last hour only
    
    try:
        # Get messages
        print(f"Fetching messages from {test_channel}")
        messages = await get_new_messages(test_channel, None, start_date, config)
        print(f"Found {len(messages)} messages")
        
        if not messages:
            print("No messages found")
            return
            
        # Process messages
        print("Processing messages...")
        processed_messages = text_processor.process_new_messages(
            messages,
            test_channel,
            "neutral",  # Test stance
            clients['gemini']
        )
        print(f"Processed {len(processed_messages)} messages")
        
        # Print results
        for msg in processed_messages:
            print("\nMessage:")
            print(f"ID: {msg['id']}")
            print(f"Is Digest: {msg['is_digest']}")
            print(f"Summary: {msg['summary']}")
            print(f"Named Entities: {msg['named_entities']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())

def test_prompt_manager():
    prompt_manager = PromptManager()
    prompt = prompt_manager.get_prompt(
        'summarize',
        text="This is a test message",
        language="english"
    )
    print("Test prompt:")
    print(prompt)

# Run specific test
if __name__ == "__main__":
    # test_prompt_manager()
    asyncio.run(test_single_channel())  # Uncomment this line