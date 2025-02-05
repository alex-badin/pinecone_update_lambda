# summarizer.py

import logging
import google.generativeai as genai
from typing import Optional, Tuple
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

# Configure a logger so Tenacity can log the wait times
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Summarizer:
    def __init__(self, api_key: str):
        # Configure your Generative AI client
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.INFO)
    )
    def _generate_content(self, prompt):
        # print(f"Calling generate_content for prompt {prompt[:30]!r}...")
        return self.model.generate_content(prompt)

    def summarize(self, text: str, max_summary_length: Optional[int] = 500, length_threshold: int = 750) -> Tuple[str, bool]:
        """
        Summarizes the text and returns (summary, is_digest).
        If text is short, it just cleans out 'technical' elements.
        """
        try:
            # For longer texts:
            if len(text) > length_threshold:
                prompt = f"""
                Тебе будет представлена новостная заметка.
                Твоя задача:
                1. Сделать его краткое изложение в {max_summary_length} символов или меньше.
                   - Максимально сохраняй оригинальные формулировки и цитаты
                   - Не перефразируй текст, а удаляй менее важные части
                   - Оставляй ключевые факты и детали
                   - Сохраняй авторский стиль и тон повествования
                Саммари должно быть только на русском языке.
                
                2. Определить, является ли текст дайджестом новостей (True) или нет (False).
                   Дайджест — это несколько новостей про разные темы.
                
                Формат ответа:
                SUMMARY: <краткое изложение>
                IS_DIGEST: <True/False>
                
                Текст для анализа: {text}
                """
                response = self._generate_content(prompt)
                # Once Tenacity succeeds (or raises final error),
                # we parse the response:
                response_text = response.text.strip()
                parts = response_text.split('IS_DIGEST:')
                summary = parts[0].replace('SUMMARY:', '').strip()
                is_digest = parts[1].strip().lower() == 'true'
                return summary, is_digest
            
            # For shorter texts, just clean out technical elements
            prompt = f"""
            Тебе будет представлена новостная заметка.
            Твоя задача:
            1. Удалить из текста все технические части, не относящиеся к основному сообщению
               (призывы подписаться, иностранные агенты и т.п.)
            2. Сохранить исходный стиль и текст.

            Формат ответа:
            SUMMARY: <очищенный текст>
            
            Текст для анализа: {text}
            """
            response = self._generate_content(prompt)
            summary_clean = response.text.replace('SUMMARY:', '').strip()
            return summary_clean, False

        except Exception as e:
            # If Tenacity exhausts all retries, we'll end up here.
            print(f"Error in summarization after retries: {str(e)}")
            # Fallback: Return a truncated version of original text
            words = text.split()
            truncated = ' '.join(words[:100]) + ('...' if len(words) > 100 else '')
            return truncated, False