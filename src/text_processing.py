import re
import unicodedata
import json
import google.generative.ai as genai
from typing import Tuple, List
from .prompt_manager import PromptManager

class TextProcessor:
    def __init__(self, prompt_manager: PromptManager):
        self.prompt_manager = prompt_manager

    def clean_text(self, text):
        # Unicode range for emojis
        emoji_pattern = re.compile("["
                               "\U0001F600-\U0001F64F"  # Emoticons
                               "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                               "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                               "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
                               "]+", flags=re.UNICODE)

        # Remove emojis
        text = emoji_pattern.sub(r'', str(text))
        # Remove URLs
        url_pattern = re.compile(r"http\S+|www\S+")
        text = url_pattern.sub(r'', str(text))
        # Remove newlines
        text = text.replace('\n', ' ')
        # Remove variation selectors
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')

        # Remove Foreign Agent text
        pattern = re.compile(r'[А-ЯЁ18+]{3,}\s[А-ЯЁ()]{5,}[^\n]*ИНОСТРАННОГО АГЕНТА')
        text = pattern.sub('', text)
        name1 = 'ПИВОВАРОВА АЛЕКСЕЯ ВЛАДИМИРОВИЧА'
        text = text.replace(name1, '')

        return text

    def summarize(self, text: str, model: genai.GenerativeModel, language: str = "russian") -> Tuple[bool, str, List[str]]:
        """
        Analyzes text using Gemini to detect if it's a digest, create a summary, and extract NERs.
        """
        prompt = self.prompt_manager.get_prompt(
            'summarize',
            text=text,
            language=language
        )

        response = model.generate_content(prompt)
        result = json.loads(response.text)
        
        return (
            result.get('is_digest', False),
            result.get('summary', ''),
            result.get('named_entities', [])
        )

    def process_new_messages(self, messages, channel, stance, gemini_model):
        processed_messages = []
        empty_message_count = 0
        failed_message_count = 0
        
        for message in messages:
            if 'message' not in message:
                empty_message_count += 1
                continue
            
            cleaned_message = self.clean_text(message['message'])
            if len(cleaned_message) > 100:
                try:
                    is_digest, summary, named_entities = self.summarize(cleaned_message, gemini_model)
                    processed_messages.append({
                        'id': message['id'],
                        'channel': channel,
                        'stance': stance,
                        'cleaned_message': cleaned_message,
                        'summary': summary,
                        'is_digest': is_digest,
                        'named_entities': named_entities,
                        'date': message['date'],
                        'views': message.get('views', 0)
                    })
                except Exception as e:
                    print(f"Failed to process message {message['id']}: {str(e)}")
                    failed_message_count += 1
                    continue
                
        if empty_message_count > 0:
            print(f"Number of empty messages: {empty_message_count}")
        if failed_message_count > 0:
            print(f"Number of failed messages: {failed_message_count}")
        
        return processed_messages 