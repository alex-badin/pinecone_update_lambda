from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=60))
def get_embeddings_with_retry(cohere_client, texts, model="embed-multilingual-v3.0"):
    print(f"Getting embeddings for {len(texts)} texts")
    return cohere_client.embed(texts=texts, model=model, input_type="clustering").embeddings

def get_embeddings(messages, cohere_client, text_col='summary', model="embed-multilingual-v3.0"):
    texts = [msg[text_col] for msg in messages if text_col in msg]
    if not texts:
        print(f"Warning: No '{text_col}' found in messages. Available keys: {messages[0].keys() if messages else 'No messages'}")
        return messages
    try:
        embeddings = get_embeddings_with_retry(cohere_client, texts, model)
        for msg, embedding in zip(messages, embeddings):
            msg['embeddings'] = embedding
    except Exception as e:
        print(f"Error getting embeddings: {str(e)}")
        return messages
    return messages 