import gradio as gr

class ModelTab:
    def __init__(self, model_controller, model_name, default_prompt, prompt_sets):
        self.controller = model_controller
        self.model_name = model_name
        self.default_prompt = default_prompt
        self.prompt_sets = prompt_sets
        
    def get_available_vectors(self):
        """Get all available vectors including both prompt sets and custom vectors"""
        print(f"Getting available vectors for {self.model_name}...")  # Debug
        default_vectors = list(self.prompt_sets.keys())
        custom_vectors = list(self.controller.vector_controllers[self.model_name].vectors.keys())
        all_vectors = default_vectors + custom_vectors
        print(f"Found vectors: {all_vectors}")  # Debug
        return all_vectors
        
    def create_interface(self):
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", value=self.default_prompt)
            strength = gr.Slider(
                label="Control Strength",
                minimum=-2,
                maximum=2,
                step=0.1,
                value=1.0
            )
        
        available_vectors = self.get_available_vectors()
        default_vector = available_vectors[0] if available_vectors else None
        
        vector_type = gr.CheckboxGroup(
            choices=available_vectors,
            label="Control Vectors",
            value=[default_vector] if default_vector else [],
            interactive=True
        )
        
        # Fix update method
        def update_choices():
            new_choices = self.get_available_vectors()
            return gr.update(choices=new_choices)
        
        # Create refresh button instead of using change event
        refresh_btn = gr.Button("🔄 Refresh Vectors")
        refresh_btn.click(fn=update_choices, inputs=[], outputs=[vector_type])
        
        with gr.Row():
            output = gr.Textbox(label="Generated Output")
            pca_plot = gr.Image(label="PCA Visualization")
            
        return prompt, strength, vector_type, output, pca_plot

class CustomVectorTab:
    def __init__(self, model_controller):
        self.controller = model_controller
        
    def create_interface(self):
        model_select = gr.Dropdown(
            choices=list(self.controller.models.keys()),
            label="Select Model",
            value="gpt2"
        )
        vector_name = gr.Textbox(label="Custom Vector Name")
        pos_prompts = gr.Textbox(label="Positive Examples (one per line)")
        neg_prompts = gr.Textbox(label="Negative Examples (one per line)")
        save_output = gr.Textbox(label="Status")
        
        return model_select, vector_name, pos_prompts, neg_prompts, save_output
