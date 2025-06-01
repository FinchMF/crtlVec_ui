from models.gpt2_model import GPT2ControlModel
from models.bert_model import BertControlModel
from controllers.vector_controller import VectorController

class ModelController:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.models = {
            'gpt2': GPT2ControlModel(),
            'bert': BertControlModel()
        }
        self.vector_controllers = {
            'gpt2': VectorController('gpt2'),
            'bert': VectorController('bert')
        }

    def get_available_vectors(self, model_name):
        """Get list of available vectors for a model"""
        prompt_sets = self.config_manager.load_prompt_sets(model_name)
        custom_vectors = list(self.vector_controllers[model_name].vectors.keys())
        return list(prompt_sets.keys()) + custom_vectors

    def get_vector_from_prompt_set(self, model, prompt_set, name):
        """Create vectors from prompt set with debug logging"""
        try:
            category = prompt_set[name]
            print(f"Processing category {name}: {category}")
            
            if 'contrasts' not in category or not category['contrasts']:
                print(f"No contrasts found in {name}, attempting legacy format conversion")
                # Try legacy format conversion
                if isinstance(category, dict) and all(k in category for k in ('positive', 'negative')):
                    contrast = {"positive": category["positive"], "negative": category["negative"]}
                    print(f"Converted to contrast format: {contrast}")
                else:
                    print(f"Invalid category format: {category}")
                    return None
            else:
                contrast = category['contrasts'][0]
            
            contrast_keys = list(contrast.keys())
            if len(contrast_keys) != 2:
                print(f"Invalid contrast keys: {contrast_keys}")
                return None
                
            first_examples = contrast[contrast_keys[0]]
            second_examples = contrast[contrast_keys[1]]
            
            print(f"Creating vector from {len(first_examples)} {contrast_keys[0]} examples vs "
                  f"{len(second_examples)} {contrast_keys[1]} examples")
            
            return model.make_control_vector(first_examples, second_examples)
            
        except Exception as e:
            print(f"Error in vector creation for {name}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def generate_text(self, model_name, prompt, vector_names, scale=1.0):
        """Generate text with improved error handling"""
        print(f"Received vector_names: {vector_names}")  # Debug
        
        if not vector_names or (isinstance(vector_names, list) and not vector_names):
            print("No vectors selected")  # Debug
            return "Please select at least one control vector"
            
        if isinstance(vector_names, str):
            vector_names = [vector_names]
        
        model = self.models[model_name]
        vector_controller = self.vector_controllers[model_name]
        prompt_sets = self.config_manager.load_prompt_sets(model_name)
        
        print(f"Generating text with vectors: {vector_names}")  # Debug
        print(f"Available prompt sets: {list(prompt_sets.keys())}")  # Debug
        
        vectors = []
        for name in vector_names:
            if name in prompt_sets:
                vector = self.get_vector_from_prompt_set(model, prompt_sets, name)
                if vector is not None:
                    vectors.append(vector)
                    print(f"Successfully created vector for {name}")
                else:
                    print(f"Failed to create vector for {name}")
            elif name in vector_controller.vectors:
                vectors.append(vector_controller.vectors[name])
                print(f"Using custom vector: {name}")
        
        if not vectors:
            return "No valid control vectors found. Available vectors: " + \
                   ", ".join(prompt_sets.keys())
        
        combined_vector = sum(vectors)
        return model.generate_with_control(prompt, combined_vector, scale)

    def create_control_vector(self, model_name, vector_name, pos_examples, neg_examples):
        """Create and save a new control vector"""
        model = self.models[model_name]
        vector = model.make_control_vector(pos_examples, neg_examples)
        self.vector_controllers[model_name].add_vector(vector_name, vector)
        return f"Created control vector: {vector_name}"
