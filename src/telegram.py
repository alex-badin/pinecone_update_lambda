from telethon import TelegramClient
from telethon.sessions import StringSession

async def get_new_messages(channel, last_id, start_date, config):
    async with TelegramClient(
        StringSession(config['TG_SESSION_STRING']), 
        config['TG_API_ID'], 
        config['TG_API_HASH'],
        system_version="4.16.30-vxCUSTOM"
    ) as client:
        data = []
        try:
            offset_id = int(last_id) if last_id else 0
        except ValueError:
            offset_id = 0
        
        print(f"Parsing channel: {channel}, start date: {start_date}, last id: {last_id}")
        async for message in client.iter_messages(channel, reverse=True, offset_id=offset_id):
            if message.date < start_date:
                continue
            data.append(message.to_dict())

    print(f"Channel: {channel}, N of new messages: {len(data)}")
    return data 