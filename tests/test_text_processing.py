import unittest
from src.prompt_manager import PromptManager
from src.text_processing import TextProcessor

class TestTextProcessing(unittest.TestCase):
    def setUp(self):
        self.prompt_manager = PromptManager()
        self.text_processor = TextProcessor(self.prompt_manager)

    def test_clean_text(self):
        # Test emoji removal
        text_with_emoji = "Hello 👋 World 🌍"
        cleaned = self.text_processor.clean_text(text_with_emoji)
        self.assertEqual(cleaned, "Hello  World ")

        # Test URL removal
        text_with_url = "Check this https://example.com and www.test.com"
        cleaned = self.text_processor.clean_text(text_with_url)
        self.assertEqual(cleaned, "Check this  and ")

        # Test newline removal
        text_with_newlines = "Hello\nWorld\n!"
        cleaned = self.text_processor.clean_text(text_with_newlines)
        self.assertEqual(cleaned, "Hello World !")

    def test_prompt_manager(self):
        prompt = self.prompt_manager.get_prompt(
            'summarize',
            text="Sample text",
            language="english"
        )
        self.assertIn("Sample text", prompt)
        self.assertIn("english", prompt)

if __name__ == '__main__':
    unittest.main() 