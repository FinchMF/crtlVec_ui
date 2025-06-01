import json
import os
from pathlib import Path

class ConfigManager:
    def __init__(self, config_dir='config'):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
    def load_prompt_sets(self, model_name):
        """Load model-specific prompt sets"""
        config_path = self.config_dir / f"{model_name}_prompts.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}
        
    def validate_prompt_set(self, prompt_set):
        """Validate prompt set structure"""
        if not isinstance(prompt_set, dict):
            return False
            
        for category in prompt_set.values():
            if not isinstance(category, dict) or 'contrasts' not in category:
                return False
            if not isinstance(category['contrasts'], list):
                return False
            for contrast in category['contrasts']:
                if not isinstance(contrast, dict) or len(contrast) != 2:
                    return False
                for examples in contrast.values():
                    if not isinstance(examples, list) or not examples:
                        return False
        return True

    def save_prompt_sets(self, model_name, prompt_sets):
        """Save model-specific prompt sets with validation"""
        if self.validate_prompt_set(prompt_sets):
            config_path = self.config_dir / f"{model_name}_prompts.json"
            with open(config_path, 'w') as f:
                json.dump(prompt_sets, f, indent=2)
            
    def get_model_config(self, model_name):
        """Load model-specific configuration"""
        config_path = self.config_dir / f"{model_name}_config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return {}
