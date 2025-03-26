# summarizer.py

import logging
import google.generativeai as genai
import google.api_core.exceptions  # Add this import for the exceptions
import time  # Add this for the sleep function
import asyncio
import hashlib
import sqlite3
import os
from typing import Optional, Tuple, List, Dict, Any
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
    def __init__(self, api_key: str, cache_db: str = "summary_cache.db", batch_size: int = 5):
        # Configure your Generative AI client
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.cache_db = cache_db
        self.batch_size = batch_size
        if self.cache_db:  # Only initialize cache if cache_db is provided
            self._init_cache()
        
    def _init_cache(self):
        """Initialize the cache database"""
        try:
            conn = sqlite3.connect(self.cache_db)
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
            conn.close()
            logger.info(f"Summary cache initialized at {self.cache_db}")
        except Exception as e:
            logger.error(f"Error initializing cache: {str(e)}")
    
    def _get_from_cache(self, text: str) -> Optional[Tuple[str, bool]]:
        """Try to get a summary from cache"""
        if not self.cache_db:  # Skip if caching is disabled
            return None
            
        text_hash = hashlib.md5(text.encode()).hexdigest()
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            cursor.execute("SELECT summary, is_digest FROM summary_cache WHERE text_hash = ?", (text_hash,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return result[0], bool(result[1])
            return None
        except Exception as e:
            logger.error(f"Error accessing cache: {str(e)}")
            return None
    
    def _save_to_cache(self, text: str, summary: str, is_digest: bool):
        """Save a summary to cache"""
        if not self.cache_db:  # Skip if caching is disabled
            return
            
        text_hash = hashlib.md5(text.encode()).hexdigest()
        try:
            conn = sqlite3.connect(self.cache_db)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO summary_cache (text_hash, summary, is_digest) VALUES (?, ?, ?)",
                (text_hash, summary, is_digest)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")
    
    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.INFO),
        retry=retry_if_exception_type((google.api_core.exceptions.ResourceExhausted, 
                                      google.api_core.exceptions.ServiceUnavailable))
    )
    def _generate_content(self, prompt):
        try:
            return self.model.generate_content(prompt)
        except Exception as e:
            # Check if it's a rate limit error
            if "quota" in str(e).lower() or "rate limit" in str(e).lower() or "resource exhausted" in str(e).lower():
                logger.warning(f"Rate limit hit: {str(e)}. Retrying with backoff...")
                # Sleep for a bit longer on rate limits to allow quota to reset
                time.sleep(5)
                raise google.api_core.exceptions.ResourceExhausted(str(e))
            else:
                logger.error(f"Error generating content: {str(e)}")
                raise
    
    async def _generate_content_async(self, prompt):
        """Async wrapper for _generate_content"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._generate_content(prompt))
    
    async def _process_batch(self, batch_texts: List[str], max_summary_length: int, length_threshold: int) -> List[Tuple[str, bool]]:
        """Process a batch of texts concurrently"""
        tasks = []
        for text in batch_texts:
            if len(text) > length_threshold:
                prompt = self._create_long_text_prompt(text, max_summary_length)
            else:
                prompt = self._create_short_text_prompt(text)
            tasks.append(self._generate_content_async(prompt))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        summaries = []
        
        for i, result in enumerate(results):
            text = batch_texts[i]
            if isinstance(result, Exception):
                logger.error(f"Error in batch processing: {str(result)}")
                # Fallback for errors
                words = text.split()
                truncated = ' '.join(words[:100]) + ('...' if len(words) > 100 else '')
                summaries.append((truncated, False))
            else:
                try:
                    response_text = result.text.strip()
                    if len(text) > length_threshold:
                        parts = response_text.split('IS_DIGEST:')
                        summary = parts[0].replace('SUMMARY:', '').strip()
                        is_digest = parts[1].strip().lower() == 'true'
                    else:
                        summary = response_text.replace('SUMMARY:', '').strip()
                        is_digest = False
                    
                    # Save to cache
                    self._save_to_cache(text, summary, is_digest)
                    summaries.append((summary, is_digest))
                except Exception as e:
                    logger.error(f"Error parsing response: {str(e)}")
                    words = text.split()
                    truncated = ' '.join(words[:100]) + ('...' if len(words) > 100 else '')
                    summaries.append((truncated, False))
        
        return summaries
    
    def _create_long_text_prompt(self, text: str, max_summary_length: int) -> str:
        """Create prompt for longer texts"""
        return f"""
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
    
    def _create_short_text_prompt(self, text: str) -> str:
        """Create prompt for shorter texts"""
        return f"""
        Тебе будет представлена новостная заметка.
        Твоя задача:
        1. Удалить из текста все технические части, не относящиеся к основному сообщению
           (призывы подписаться, иностранные агенты и т.п.)
        2. Сохранить исходный стиль и текст.

        Формат ответа:
        SUMMARY: <очищенный текст>
        
        Текст для анализа: {text}
        """
    
    def summarize(self, text: str, max_summary_length: Optional[int] = 500, length_threshold: int = 750) -> Tuple[str, bool]:
        """
        Summarizes the text and returns (summary, is_digest).
        If text is short, it just cleans out 'technical' elements.
        Checks cache first before making API call.
        """
        try:
            # Check cache first (if enabled)
            cached_result = self._get_from_cache(text)
            if cached_result:
                logger.debug("Cache hit for text")
                return cached_result
            
            # For longer texts:
            if len(text) > length_threshold:
                prompt = self._create_long_text_prompt(text, max_summary_length)
                response = self._generate_content(prompt)
                response_text = response.text.strip()
                parts = response_text.split('IS_DIGEST:')
                summary = parts[0].replace('SUMMARY:', '').strip()
                is_digest = parts[1].strip().lower() == 'true'
            else:
                # For shorter texts, just clean out technical elements
                prompt = self._create_short_text_prompt(text)
                response = self._generate_content(prompt)
                summary = response.text.replace('SUMMARY:', '').strip()
                is_digest = False
            
            # Save to cache (if enabled)
            self._save_to_cache(text, summary, is_digest)
            return summary, is_digest

        except Exception as e:
            # If Tenacity exhausts all retries, we'll end up here.
            logger.error(f"Error in summarization after retries: {str(e)}")
            # Fallback: Return a truncated version of original text
            words = text.split()
            truncated = ' '.join(words[:100]) + ('...' if len(words) > 100 else '')
            return truncated, False
    
    async def summarize_batch(self, texts: List[str], max_summary_length: Optional[int] = 500, 
                             length_threshold: int = 750, progress_callback=None, batch_delay: float = 0) -> List[Tuple[str, bool]]:
        """
        Summarize multiple texts in parallel batches.
        Returns a list of (summary, is_digest) tuples in the same order as input texts.
        
        Args:
            texts: List of texts to summarize
            max_summary_length: Maximum length of summaries
            length_threshold: Threshold to determine if text needs summarization
            progress_callback: Callback function to update progress
            batch_delay: Delay between batches in seconds (for rate limiting)
        """
        if not texts:
            return []
        
        results = []
        to_process = []
        to_process_indices = []
        
        # First check cache for all texts (if caching is enabled)
        for i, text in enumerate(texts):
            cached_result = self._get_from_cache(text)
            if cached_result:
                # Insert cached result
                results.append(cached_result)
            else:
                # Need to process this text
                to_process.append(text)
                to_process_indices.append(i)
                # Insert None as placeholder to maintain order
                results.append(None)
        
        # Process texts not found in cache in batches
        if to_process:
            logger.info(f"Processing {len(to_process)} texts in batches of {self.batch_size}")
            batches = [to_process[i:i+self.batch_size] for i in range(0, len(to_process), self.batch_size)]
            
            all_batch_results = []
            for batch_idx, batch in enumerate(batches):
                # Add delay between batches for rate limiting
                if batch_idx > 0 and batch_delay > 0:
                    await asyncio.sleep(batch_delay)
                    
                batch_results = await self._process_batch(batch, max_summary_length, length_threshold)
                all_batch_results.extend(batch_results)
                # Update progress if callback is provided
                if progress_callback:
                    progress_callback(1)
            
            # Place batch results back in the correct positions
            for i, result in zip(to_process_indices, all_batch_results):
                results[i] = result
        
        return results