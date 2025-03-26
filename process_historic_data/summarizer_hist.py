import logging
import google.generativeai as genai
import google.api_core.exceptions
import time
import asyncio
import hashlib
import sqlite3
from typing import Optional, Tuple, List, Union
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Summarizer:
    def __init__(
        self,
        api_key: str,
        cache_db: str = "summary_cache.db",
        batch_size: int = 5
    ):
        """
        Summarizer class for generating summaries (and determining if digest)
        using Google Generative AI with caching and optional batch processing.

        :param api_key: Generative AI API Key
        :param cache_db: Path to the SQLite database for caching
        :param batch_size: Number of texts to summarize concurrently in batch
        """
        # Configure the Generative AI client
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

        # Cache settings
        self.cache_db = cache_db or None  # If empty, caching is disabled
        self.batch_size = batch_size
        if self.cache_db:
            self._init_cache()

    def _init_cache(self):
        """
        Initialize the SQLite cache for summaries.
        """
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS summary_cache (
                        text_hash TEXT PRIMARY KEY,
                        summary TEXT,
                        is_digest BOOLEAN,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
            logger.info(f"Summary cache initialized at {self.cache_db}")
        except Exception as e:
            logger.error(f"Error initializing cache: {e}")

    def _get_text_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def _get_from_cache(self, text: str) -> Optional[Tuple[str, bool]]:
        """
        Retrieve a cached result if available.
        Returns (summary, is_digest) or None if not in cache.
        """
        if not self.cache_db:
            return None

        text_hash = self._get_text_hash(text)
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT summary, is_digest FROM summary_cache WHERE text_hash = ?",
                    (text_hash,)
                )
                row = cursor.fetchone()

            if row:
                return row[0], bool(row[1])
            return None
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
            return None

    def _save_to_cache(self, text: str, summary: str, is_digest: bool):
        """
        Save or update a summary in the cache.
        """
        if not self.cache_db:
            return

        text_hash = self._get_text_hash(text)
        try:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO summary_cache (text_hash, summary, is_digest) "
                    "VALUES (?, ?, ?)",
                    (text_hash, summary, is_digest)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")

    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.INFO),
        retry=retry_if_exception_type((
            google.api_core.exceptions.ResourceExhausted,
            google.api_core.exceptions.ServiceUnavailable
        ))
    )
    def _generate_content_sync(self, prompt: str):
        """
        Synchronous content generation with Tenacity-based retry/backoff.
        """
        try:
            return self.model.generate_content(prompt)
        except Exception as e:
            # Detect rate-limiting
            msg_lower = str(e).lower()
            if any(k in msg_lower for k in ["quota", "rate limit", "resource exhausted"]):
                logger.warning(f"Rate limit hit: {e}. Retrying with backoff...")
                time.sleep(5)
                raise google.api_core.exceptions.ResourceExhausted(str(e))
            logger.error(f"Error generating content: {e}")
            raise

    async def _generate_content_async(self, prompt: str):
        """
        Asynchronous wrapper around the synchronous generator function.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._generate_content_sync(prompt))

    def _create_prompt(
        self, 
        text: str, 
        max_summary_length: int, 
        length_threshold: int
    ) -> Tuple[str, bool]:
        """
        Build the prompt string based on whether text is "long" or "short."

        Returns:
            (prompt, is_long_text)
            is_long_text is True if the text is above threshold and we want the 
            special format with SUMMARY/IS_DIGEST.
        """
        if len(text) > length_threshold:
            # Long text prompt
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
            return prompt.strip(), True
        else:
            # Short text prompt
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
            return prompt.strip(), False

    def _parse_response(
        self,
        original_text: str,
        response_text: str,
        is_long_text: bool
    ) -> Tuple[str, bool]:
        """
        Parse the text returned from the model.
        Returns (summary, is_digest).
        """
        try:
            response_text = response_text.strip()
            if is_long_text:
                # Expecting SUMMARY: ... IS_DIGEST: ...
                parts = response_text.split("IS_DIGEST:")
                summary_part = parts[0].replace("SUMMARY:", "").strip()
                digest_str = parts[1].strip().lower()
                is_digest = digest_str == "true"
                return summary_part, is_digest
            else:
                # Expecting SUMMARY: ...
                summary_part = response_text.replace("SUMMARY:", "").strip()
                return summary_part, False
        except Exception as e:
            logger.error(f"Error parsing response: {e}. Returning truncated text.")
            # Fallback: truncated original text
            words = original_text.split()
            truncated = " ".join(words[:100]) + ("..." if len(words) > 100 else "")
            return truncated, False

    def summarize(
        self,
        text: str,
        max_summary_length: int = 500,
        length_threshold: int = 750
    ) -> Tuple[str, bool]:
        """
        Synchronous single-text summarization.
        Returns (summary, is_digest).
        """
        # Check cache
        cached = self._get_from_cache(text)
        if cached:
            return cached

        prompt, is_long_text = self._create_prompt(text, max_summary_length, length_threshold)

        try:
            response = self._generate_content_sync(prompt)
            summary, is_digest = self._parse_response(text, response.text, is_long_text)
            self._save_to_cache(text, summary, is_digest)
            return summary, is_digest
        except Exception as e:
            logger.error(f"Summarization failed. Fallback to truncation: {e}")
            # Fallback
            words = text.split()
            truncated = " ".join(words[:100]) + ("..." if len(words) > 100 else "")
            return truncated, False

    async def summarize_async(
        self,
        text: str,
        max_summary_length: int = 500,
        length_threshold: int = 750
    ) -> Tuple[str, bool]:
        """
        Asynchronous single-text summarization.
        Returns (summary, is_digest).
        """
        cached = self._get_from_cache(text)
        if cached:
            return cached

        prompt, is_long_text = self._create_prompt(text, max_summary_length, length_threshold)

        try:
            response = await self._generate_content_async(prompt)
            summary, is_digest = self._parse_response(text, response.text, is_long_text)
            self._save_to_cache(text, summary, is_digest)
            return summary, is_digest
        except Exception as e:
            logger.error(f"Async summarization failed. Fallback to truncation: {e}")
            # Fallback
            words = text.split()
            truncated = " ".join(words[:100]) + ("..." if len(words) > 100 else "")
            return truncated, False

    async def summarize_batch(
        self,
        texts: List[str],
        max_summary_length: int = 500,
        length_threshold: int = 750,
        progress_callback=None,
        batch_delay: float = 0.0
    ) -> List[Tuple[str, bool]]:
        """
        Summarize a list of texts asynchronously, optionally in batches.
        Returns a list of (summary, is_digest) in the same order.

        :param texts: List of texts to process
        :param max_summary_length: Max length of the summary for long texts
        :param length_threshold: Texts longer than this get "long text" prompt
        :param progress_callback: Optional function to track batch progress
        :param batch_delay: Delay (in seconds) between batches (to help rate-limit)
        """
        if not texts:
            return []

        results = [None] * len(texts)   # Pre-allocate result list

        # First pass: fill from cache
        to_process = []
        to_process_indices = []
        for i, text in enumerate(texts):
            cached = self._get_from_cache(text)
            if cached:
                results[i] = cached
            else:
                to_process.append(text)
                to_process_indices.append(i)

        if not to_process:
            return results

        logger.info(f"Processing {len(to_process)} texts in batches of {self.batch_size}...")

        # Split into batches
        batches = [
            to_process[i : i + self.batch_size]
            for i in range(0, len(to_process), self.batch_size)
        ]

        # Process each batch concurrently
        for batch_idx, batch_texts in enumerate(batches):
            if batch_idx > 0 and batch_delay > 0:
                await asyncio.sleep(batch_delay)

            # Create tasks for each text in this batch
            tasks = [
                self.summarize_async(text, max_summary_length, length_threshold)
                for text in batch_texts
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Place batch results back in correct positions
            start_idx = batch_idx * self.batch_size
            for offset, result in enumerate(batch_results):
                index_in_results = to_process_indices[start_idx + offset]
                if isinstance(result, Exception):
                    logger.error(f"Error in batch item: {result}")
                    # Fallback truncated text
                    text = to_process[start_idx + offset]
                    words = text.split()
                    truncated = " ".join(words[:100]) + ("..." if len(words) > 100 else "")
                    results[index_in_results] = (truncated, False)
                else:
                    results[index_in_results] = result

            # Update progress if callback is provided
            if progress_callback:
                progress_callback(1)

        return results