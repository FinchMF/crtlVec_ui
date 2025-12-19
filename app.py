from controllers.model_controller import ModelController
from ui.tabs import ControlVectorUI
from utils.storage import ConfigManager
import os
import warnings

# Suppress matplotlib and torch warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)

def main():
    """
    Control Vector Laboratory - Main Application Entry Point

    This application provides an educational interface for exploring
    how control vectors can steer language model behavior.

    Features:
    - GPT-2 and BERT text generation with control vectors
    - Layer-by-layer activation visualization
    - Entropy and prediction analysis
    - Dictionary learning (sparse autoencoders)
    - Interactive vector editing
    - Educational demonstrations
    """

    # Ensure config directory exists
    os.makedirs('config', exist_ok=True)

    print("🚀 Starting Control Vector Laboratory...")
    print("=" * 50)

    # Initialize controllers and configuration
    config_manager = ConfigManager()
    print("📁 Configuration manager initialized")

    model_controller = ModelController(config_manager)
    print("🤖 Models loaded (GPT-2, BERT)")

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

    print("📊 Prompt sets loaded/initialized")
    print("=" * 50)

    # Initialize and launch UI
    ui = ControlVectorUI(model_controller, config_manager)
    demo = ui.create_interface()

    print("\n🎛️  Control Vector Laboratory")
    print("=" * 50)
    print("📚 Available tabs:")
    print("   • GPT-2 Generation - Text generation with control")
    print("   • BERT Prediction - Masked token prediction")
    print("   • Layer Analysis - Activation visualization")
    print("   • Entropy & Predictions - Model certainty analysis")
    print("   • Dictionary Learning - Sparse autoencoder features")
    print("   • Vector Editor - Interactive vector modification")
    print("   • Educational Demos - Pre-configured examples")
    print("   • Create Custom Vector - Build your own vectors")
    print("=" * 50)
    print("\n🌐 Launching server...")

    demo.launch(
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )

if __name__ == "__main__":
    main()
