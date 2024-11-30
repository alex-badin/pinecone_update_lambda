import os
from src.summarizer import Summarizer

def test_summarizer():
    api_key = os.environ.get('GEMINI_API_KEY')
    summarizer = Summarizer(api_key)
    
    test_text = """
    This is a long piece of text that needs to be summarized.
    It contains multiple sentences and important information.
    We want to make sure the summarizer captures the key points
    while maintaining the original meaning of the text.
    """
    
    summary = summarizer.summarize(test_text, max_sentences=2)
    print(f"Original text: {test_text}")
    print(f"Summary: {summary}")
    
    assert len(summary.split('.')) <= 2, "Summary should not exceed 2 sentences"
    assert len(summary) < len(test_text), "Summary should be shorter than original text"

if __name__ == "__main__":
    test_summarizer() 