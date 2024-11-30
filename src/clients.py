import cohere
from pinecone import Pinecone
from pyairtable import Api
import google.generative.ai as genai

def initialize_clients(config):
    # Initialize all clients
    co = cohere.Client(config['COHERE_KEY'])
    pc = Pinecone(config['PINE_KEY'])
    pine_index = pc.Index(config['PINE_INDEX'])
    airtable_api = Api(config['AIRTABLE_API_TOKEN'])
    
    genai.configure(api_key=config['GEMINI_KEY'])
    model = genai.GenerativeModel('gemini-pro')
    
    return {
        'cohere': co,
        'pinecone': pc,
        'pine_index': pine_index,
        'airtable': airtable_api,
        'gemini': model
    } 