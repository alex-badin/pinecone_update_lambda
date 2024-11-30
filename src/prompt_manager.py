import yaml

class PromptManager:
    def __init__(self, prompt_file='src/prompts.yaml'):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            self.prompts = yaml.safe_load(f)
    
    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        Get a prompt template and fill it with the provided kwargs
        
        Args:
            prompt_name: Name of the prompt in the YAML file
            **kwargs: Keywords arguments to fill the template
        
        Returns:
            Formatted prompt string
        """
        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt '{prompt_name}' not found")
            
        prompt_template = self.prompts[prompt_name]['template']
        return prompt_template.format(**kwargs) 