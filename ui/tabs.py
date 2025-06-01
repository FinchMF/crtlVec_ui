import gradio as gr
from .components import ModelTab, CustomVectorTab

class ControlVectorUI:
    def __init__(self, model_controller, config_manager):
        self.controller = model_controller
        self.config = config_manager
        
        # Get prompt sets for each model
        self.gpt2_prompts = self.config.load_prompt_sets('gpt2')
        self.bert_prompts = self.config.load_prompt_sets('bert')
        
        # Ensure prompt sets are loaded before creating tabs
        if not self.gpt2_prompts or not self.bert_prompts:
            print("Warning: Empty prompt sets detected")
            
        self.gpt2_tab = ModelTab(model_controller, "gpt2", "The food was", self.gpt2_prompts)
        self.bert_tab = ModelTab(model_controller, "bert", "This movie is [MASK].", self.bert_prompts)
        self.custom_tab = CustomVectorTab(model_controller)
    
    def model_generate(self, model_name, prompt, strength, vectors):
        """Validate inputs before generation and return both text and visualization"""
        if not vectors or not isinstance(vectors, list):
            return "Please select at least one control vector", None
            
        available_vectors = self.gpt2_prompts.keys() if model_name == "gpt2" else self.bert_prompts.keys()
        if not any(v in available_vectors for v in vectors):
            return f"Selected vectors not found in available choices: {list(available_vectors)}", None
            
        # Get the prompt sets for visualization
        prompt_sets = self.gpt2_prompts if model_name == "gpt2" else self.bert_prompts
        selected_sets = {v: prompt_sets[v] for v in vectors if v in prompt_sets}
        
        output = self.controller.generate_text(model_name, prompt, vectors, strength)
        
        # Generate visualization if we have valid sets
        if selected_sets:
            from utils.visualization import LatentSpaceVisualizer
            visualizer = LatentSpaceVisualizer(self.controller.models[model_name])
            plot = visualizer.create_pca_plot(selected_sets, None)
            return output, plot
        
        return output, None
        
    def create_interface(self):
        with gr.Blocks() as demo:
            gr.Markdown("# 🎛️ Latent Control Vector Generator with Custom Vector Support")
            
            with gr.Tab("GPT-2"):
                prompt, strength, vector_type, output, pca_plot = self.gpt2_tab.create_interface()
                run_button = gr.Button("Generate with Control")
                run_button.click(
                    fn=lambda p, s, v: self.model_generate("gpt2", p, s, v),
                    inputs=[prompt, strength, vector_type],
                    outputs=[output, pca_plot]
                )
                
            with gr.Tab("BERT"):
                prompt, strength, vector_type, output, pca_plot = self.bert_tab.create_interface()
                run_button = gr.Button("Generate with Control")
                run_button.click(
                    fn=lambda p, s, v: self.model_generate("bert", p, s, v),
                    inputs=[prompt, strength, vector_type],
                    outputs=[output, pca_plot]
                )
                
            with gr.Tab("Create Custom Vector"):
                model, name, pos, neg, status = self.custom_tab.create_interface()
                save_button = gr.Button("Save Custom Vector")
                save_button.click(
                    fn=self.controller.create_control_vector,
                    inputs=[model, name, pos, neg],
                    outputs=[status]
                )
                
        return demo
