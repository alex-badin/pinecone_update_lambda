import google.generativeai as genai
from typing import Optional

class Summarizer:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def summarize(self, text: str, max_sentences: Optional[int] = 2) -> str:
        prompt = f"""
        Summarize the following text in {max_sentences} sentences or less. 
        Keep the most important information and maintain the original meaning.
        
        Text to summarize: {text}
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error in summarization: {str(e)}")
            # Return a truncated version of original text as fallback
            words = text.split()
            return ' '.join(words[:50]) + ('...' if len(words) > 50 else '') 