from controllers.model_controller import ModelController
from ui.tabs import ControlVectorUI
from utils.storage import ConfigManager
import os

def main():
    # Ensure config directory exists
    os.makedirs('config', exist_ok=True)
    
    # Initialize controllers and configuration
    config_manager = ConfigManager()
    model_controller = ModelController(config_manager)
    
    # Load predefined prompt sets from config
    gpt2_prompts = config_manager.load_prompt_sets('gpt2')
    bert_prompts = config_manager.load_prompt_sets('bert')
    
    if not gpt2_prompts:  # Set default prompts if none exist
        gpt2_prompts = {
            "Sentiment": {
                "contrasts": [
                    {"positive": ["The movie was fantastic.", "I loved the experience."],
                     "negative": ["The movie was terrible.", "I hated the experience."]}
                ]
            },
            "Formality": {
                "contrasts": [
                    {"formal": ["I am writing to inform you.", "Please find the report attached."],
                     "informal": ["Hey! Just letting you know.", "Check this out!"]}
                ]
            },
            "Tense": {
                "contrasts": [
                    {"present": ["He walks to school.", "She eats lunch."],
                     "past": ["He walked to school.", "She ate lunch."]}
                ]
            }
        }
        config_manager.save_prompt_sets('gpt2', gpt2_prompts)
    else:
        # Validate existing prompt sets
        for category in list(gpt2_prompts.keys()):
            if not isinstance(gpt2_prompts[category], dict) or \
               'contrasts' not in gpt2_prompts[category] or \
               not isinstance(gpt2_prompts[category]['contrasts'], list):
                del gpt2_prompts[category]
        config_manager.save_prompt_sets('gpt2', gpt2_prompts)
    
    if not bert_prompts:  # Set default prompts if none exist
        bert_prompts = {
            "Sentiment": {
                "contrasts": [
                    {"positive": ["This is wonderful.", "I am very happy."],
                     "negative": ["This is terrible.", "I am very sad."]}
                ]
            },
            "Formality": {
                "contrasts": [
                    {"formal": ["Dear Sir/Madam,", "As per our discussion,"],
                     "informal": ["Hey there!", "What's up!"]}
                ]
            }
        }
        config_manager.save_prompt_sets('bert', bert_prompts)
    else:
        # Validate existing bert prompt sets
        for category in list(bert_prompts.keys()):
            if not isinstance(bert_prompts[category], dict) or \
               'contrasts' not in bert_prompts[category] or \
               not isinstance(bert_prompts[category]['contrasts'], list):
                # Convert old format to new format if needed
                if isinstance(bert_prompts[category], dict) and \
                   all(k in bert_prompts[category] for k in ('positive', 'negative')):
                    bert_prompts[category] = {
                        "contrasts": [
                            {"positive": bert_prompts[category]["positive"],
                             "negative": bert_prompts[category]["negative"]}
                        ]
                    }
                else:
                    del bert_prompts[category]
        config_manager.save_prompt_sets('bert', bert_prompts)
    
    # Initialize and launch UI
    ui = ControlVectorUI(model_controller, config_manager)
    demo = ui.create_interface()
    demo.launch(server_port=7860)

if __name__ == "__main__":
    main()