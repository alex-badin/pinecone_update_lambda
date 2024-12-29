import google.generativeai as genai
from typing import Optional, Tuple
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

class Summarizer:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True  # optional, but helps to see final error
    )
    def _generate_content(self, prompt: str, _retry_state=None) -> str:
        try:
            return self.model.generate_content(prompt)
        except Exception as e:
            if _retry_state and _retry_state.next_action:
                wait_time = _retry_state.next_action.sleep
                attempt_num = _retry_state.attempt_number
                print(f"API call failed. Attempt {attempt_num}")
                print(f"Error: {str(e)}")
                print(f"Waiting {wait_time:.1f} seconds before next attempt...")
            else:
                print("API call failed. Attempt unknown")
                print(f"Error: {str(e)}")

            raise  # re-raise to trigger Tenacity’s retry
    
    def summarize(self, text: str, max_summary_length: Optional[int] = 500, length_threshold: int = 750) -> Tuple[str, bool]:
        try:
            if len(text) > length_threshold:
                prompt = f"""
                Тебе будет представлена новостная заметка.
                Твоя задача:
                1. Сделать его краткое изложение в {max_summary_length} символов или меньше.
                При сокращении:
                   - Максимально сохраняй оригинальные формулировки и цитаты
                   - Не перефразируй текст, а удаляй менее важные части
                   - Оставляй ключевые факты и детали в исходном виде
                   - Сохраняй авторский стиль и тон повествования
                Саммари должно быть только на русском языке.
                
                2. Определить, является ли текст дайджестом новостей (True) или нет (False). Дайджест - это несколько новостей про разные темы.
                
                Формат ответа:
                SUMMARY: <краткое изложение>
                IS_DIGEST: <True/False>
                
                Текст для анализа: {text}
                """
                response = self._generate_content(prompt)
                response_text = response.text.strip()
                parts = response_text.split('IS_DIGEST:')
                return parts[0].replace('SUMMARY:', '').strip(), parts[1].strip().lower() == 'true'
            
            # For shorter texts, just clean technical parts
            prompt = f"""
            Тебе будет представлена новостная заметка.
            Твоя задача:
            1. Удалить из текста все технические части, не относящиеся к основному сообщению, такие как:
               - призывы подписаться на канал/группу
               - уведомления об иностранных агентах
               - призывы читать подробности по ссылке
               - рекламные вставки
               - и подобные технические элементы
            2. Оставить только информативную часть новости, сохранив исходный текст и стиль.
            
            Формат ответа:
            SUMMARY: <очищенный текст>
            
            Текст для анализа: {text}
            """
            response = self._generate_content(prompt)
            return response.text.replace('SUMMARY:', '').strip(), False

        except Exception as e:
            print(f"Error in summarization after retries: {str(e)}")
            # Return a truncated version of original text as fallback
            words = text.split()
            return ' '.join(words[:100]) + ('...' if len(words) > 50 else ''), False