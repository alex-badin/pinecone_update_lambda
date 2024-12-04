import google.generativeai as genai
from typing import Optional, Tuple

class Summarizer:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def summarize(self, text: str, max_symbols: Optional[int] = 500) -> Tuple[str, bool]:
        prompt = f"""
        Тебе будет представлена новостная заметка.
        Твоя задача:
        1. Сделать его краткое изложение в {max_symbols} символов или меньше.
        Сохрани наиболее важную информацию, исходный смысл и стиль. Саммари должно быть только на русском языке.
        2. Определить, является ли текст дайджестом новостей (True) или нет (False). Дайджест - это несколько новостей про разные темы.
        
        Формат ответа:
        SUMMARY: <краткое изложение>
        IS_DIGEST: <True/False>
        
        Текст для анализа: {text}
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse response
            parts = response_text.split('IS_DIGEST:')
            summary = parts[0].replace('SUMMARY:', '').strip()
            is_digest = parts[1].strip().lower() == 'true'
            
            return summary, is_digest
        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            # Return a truncated version of original text as fallback
            words = text.split()
            return ' '.join(words[:50]) + ('...' if len(words) > 50 else ''), False