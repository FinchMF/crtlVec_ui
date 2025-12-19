import gradio as gr
import numpy as np
import torch
from typing import Optional, Tuple, List, Dict
from .components import ModelTab, CustomVectorTab


class ControlVectorUI:
    """Main UI class for the Control Vector application."""

    def __init__(self, model_controller, config_manager):
        self.controller = model_controller
        self.config = config_manager

        # Get prompt sets for each model
        self.gpt2_prompts = self.config.load_prompt_sets('gpt2')
        self.bert_prompts = self.config.load_prompt_sets('bert')

        # Ensure prompt sets are loaded before creating tabs
        if not self.gpt2_prompts or not self.bert_prompts:
            print("Warning: Empty prompt sets detected")

        self.gpt2_tab = ModelTab(model_controller, "gpt2", "The building was", self.gpt2_prompts)
        self.bert_tab = ModelTab(model_controller, "bert", "This movie is [MASK] for a genre like this.", self.bert_prompts)
        self.custom_tab = CustomVectorTab(model_controller)

    def model_generate(self, model_name, prompt, strength, vectors, pca_dims, temperature=1.0):
        """Validate inputs before generation and return both text and visualization"""
        if not vectors or not isinstance(vectors, list):
            return "Please select at least one control vector", None

        available_vectors = self.gpt2_prompts.keys() if model_name == "gpt2" else self.bert_prompts.keys()
        if not any(v in available_vectors for v in vectors):
            return f"Selected vectors not found in available choices: {list(available_vectors)}", None

        # Get the prompt sets for visualization
        prompt_sets = self.gpt2_prompts if model_name == "gpt2" else self.bert_prompts
        selected_sets = {v: prompt_sets[v] for v in vectors if v in prompt_sets}

        output = self.controller.generate_text(model_name, prompt, vectors, strength, temperature=temperature)

        # Generate visualization if we have valid sets
        if selected_sets:
            from utils.visualization import LatentSpaceVisualizer
            visualizer = LatentSpaceVisualizer(self.controller.models[model_name])
            n_components = 3 if pca_dims == "3D" else 2
            plot = visualizer.create_pca_plot(selected_sets, None, n_components)
            return output, plot

        return output, None

    def create_interface(self):
        with gr.Blocks(theme=gr.themes.Base(primary_hue="cyan", secondary_hue="orange"), title="ControL-Lab") as demo:
            gr.Markdown("# 🎛️ Control Vector Laboratory")
            gr.Markdown("*An educational tool for understanding how control vectors steer language model behavior*")

            with gr.Tab("🎯 GPT-2 Generation"):
                prompt, strength, temperature, vector_type, output, pca_plot, pca_dims = self.gpt2_tab.create_interface()
                run_button = gr.Button("Generate with Control", variant="primary")
                run_button.click(
                    fn=lambda p, s, t, v, d: self.model_generate("gpt2", p, s, v, d, t),
                    inputs=[prompt, strength, temperature, vector_type, pca_dims],
                    outputs=[output, pca_plot]
                )

            with gr.Tab("🎯 BERT Prediction"):
                prompt, strength, temperature, vector_type, output, pca_plot, pca_dims = self.bert_tab.create_interface()
                run_button = gr.Button("Generate with Control", variant="primary")
                run_button.click(
                    fn=lambda p, s, t, v, d: self.model_generate("bert", p, s, v, d, t),
                    inputs=[prompt, strength, temperature, vector_type, pca_dims],
                    outputs=[output, pca_plot]
                )

            with gr.Tab("🔬 Layer Analysis"):
                layer_tab = LayerAnalysisTab(self.controller)
                layer_tab.create_interface()

            with gr.Tab("📊 Entropy & Predictions"):
                entropy_tab = EntropyAnalysisTab(self.controller)
                entropy_tab.create_interface()

            with gr.Tab("🧠 Dictionary Learning"):
                dict_tab = DictionaryLearningTab(self.controller)
                dict_tab.create_interface()

            with gr.Tab("✏️ Vector Editor"):
                editor_tab = VectorEditorTab(self.controller, self.config)
                editor_tab.create_interface()

            with gr.Tab("📚 Educational Demos"):
                demo_tab = EducationalDemoTab(self.controller, self.config)
                demo_tab.create_interface()

            with gr.Tab("� Context Evolution"):
                context_tab = ContextEvolutionTab(self.controller, self.config)
                context_tab.create_interface()

            with gr.Tab("�🎓 Neural Network Lab"):
                edu_models_tab = EducationalModelsTab(self.controller)
                edu_models_tab.create_interface()

            with gr.Tab("➕ Create Custom Vector"):
                model, name, pos, neg, status = self.custom_tab.create_interface()
                save_button = gr.Button("Save Custom Vector", variant="primary")
                save_button.click(
                    fn=self.controller.create_control_vector,
                    inputs=[model, name, pos, neg],
                    outputs=[status]
                )

        return demo


class LayerAnalysisTab:
    """Tab for visualizing layer-by-layer activations."""

    def __init__(self, model_controller):
        self.controller = model_controller

    def create_interface(self):
        gr.Markdown("## 🔬 Layer-by-Layer Activation Analysis")
        gr.Markdown("*Watch how information transforms as it flows through the model's layers*")

        with gr.Row():
            model_select = gr.Dropdown(
                choices=list(self.controller.models.keys()),
                label="Model",
                value="gpt2"
            )
            prompt_input = gr.Textbox(
                label="Input Prompt",
                value="The scientist discovered that",
                lines=2
            )

        with gr.Row():
            vector_select = gr.Dropdown(
                choices=[],
                label="Control Vector (optional)",
                interactive=True
            )
            strength = gr.Slider(
                minimum=-2.0,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Control Strength"
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Temperature",
                info="Higher = more random activations"
            )

        analyze_btn = gr.Button("🔍 Analyze Layers", variant="primary")

        with gr.Row():
            activation_heatmap = gr.Image(label="Activation Magnitudes (2D Heatmap)")
            activation_3d = gr.Image(label="Activation Manifold (3D Surface)")

        with gr.Row():
            layer_graph = gr.Image(label="Layer Connectivity Graph")
            comparison_plot = gr.Image(label="Before/After Control Vector")

        with gr.Row():
            trajectory_plot = gr.Image(label="Embedding Trajectory")
            vector_anatomy = gr.Image(label="Control Vector Anatomy")

        # Event handlers
        def update_vectors(model_name):
            vectors = list(self.controller.vector_controllers[model_name].vectors.keys())
            prompt_sets = self.controller.config_manager.load_prompt_sets(model_name)
            vectors.extend(prompt_sets.keys())
            return gr.update(choices=["None"] + vectors, value="None")

        model_select.change(update_vectors, [model_select], [vector_select])

        def analyze(model_name, prompt, vector_name, strength_val, temp_val):
            from ui.layer_viz import LayerVisualizer
            from utils.visualization import ControlVectorVisualizer

            model = self.controller.models[model_name]
            layer_viz = LayerVisualizer()
            cv_viz = ControlVectorVisualizer(model)

            # Get all hidden states (temperature affects sampling in generation, stored for context)
            layer_hidden_states, tokens = model.get_all_hidden_states(prompt)

            # Create activation heatmap
            activations = np.stack([h.cpu().numpy() for h in layer_hidden_states])
            activation_norms = np.linalg.norm(activations, axis=-1)

            # 2D Heatmap
            heatmap = layer_viz.create_activation_heatmap(
                activation_norms, tokens,
                title=f"Activation Magnitude Through Layers (T={temp_val})"
            )

            # 3D Manifold
            manifold_3d = layer_viz.create_3d_activation_manifold(
                activations, tokens,
                title=f"3D Activation Landscape (T={temp_val})"
            )

            # Layer connectivity graph
            graph = layer_viz.create_layer_graph(
                activations, tokens,
                title=f"Layer Information Flow (T={temp_val})"
            )

            # Get control vector if selected
            control_vector = None
            comparison = None
            trajectory = None
            anatomy = None

            if vector_name and vector_name != "None":
                control_vector = self.controller.vector_controllers[model_name].get_vector(vector_name)
                if control_vector is None:
                    prompt_sets = self.controller.config_manager.load_prompt_sets(model_name)
                    if vector_name in prompt_sets:
                        control_vector = self.controller.get_vector_from_prompt_set(
                            model, prompt_sets, vector_name
                        )

                if control_vector is not None:
                    # Before/After comparison
                    comparison = cv_viz.compare_before_after(
                        prompt, control_vector, strength_val,
                        title=f"Effect of {vector_name} Vector"
                    )

                    # Steering trajectory
                    trajectory = cv_viz.visualize_steering_effect(
                        prompt, control_vector,
                        scales=[-2, -1, 0, 1, 2],
                        title=f"Steering with {vector_name}"
                    )

                    # Vector anatomy
                    anatomy = cv_viz.visualize_vector_anatomy(
                        control_vector, vector_name
                    )

            return heatmap, manifold_3d, graph, comparison, trajectory, anatomy

        analyze_btn.click(
            analyze,
            [model_select, prompt_input, vector_select, strength, temperature],
            [activation_heatmap, activation_3d, layer_graph, comparison_plot, trajectory_plot, vector_anatomy]
        )


class EntropyAnalysisTab:
    """Tab for entropy and prediction analysis."""

    def __init__(self, model_controller):
        self.controller = model_controller

    def create_interface(self):
        gr.Markdown("## 📊 Entropy & Prediction Analysis")
        gr.Markdown("*See how model confidence evolves through layers and how control vectors affect predictions*")

        with gr.Row():
            model_select = gr.Dropdown(
                choices=list(self.controller.models.keys()),
                label="Model",
                value="gpt2"
            )
            prompt_input = gr.Textbox(
                label="Input Prompt",
                value="The weather today is",
                lines=2
            )

        with gr.Row():
            vector_select = gr.Dropdown(
                choices=[],
                label="Control Vector (optional)",
                interactive=True
            )
            strength = gr.Slider(
                minimum=-2.0,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Control Strength"
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Temperature",
                info="Higher = more uncertain predictions"
            )

        with gr.Row():
            top_k = gr.Slider(
                minimum=5,
                maximum=20,
                value=10,
                step=1,
                label="Top-K Predictions"
            )
            token_position = gr.Slider(
                minimum=-1,
                maximum=50,
                value=-1,
                step=1,
                label="Token Position (-1 = last)"
            )

        analyze_btn = gr.Button("📊 Analyze Entropy", variant="primary")

        with gr.Row():
            entropy_heatmap = gr.Image(label="Entropy Heatmap (2D)")
            entropy_3d = gr.Image(label="Entropy Landscape (3D)")

        with gr.Row():
            confidence_trajectory = gr.Image(label="Confidence Trajectory")
            predictions_table = gr.Image(label="Top-K Predictions by Layer")

        with gr.Row():
            entropy_comparison = gr.Image(label="Entropy: Original vs Controlled")

        # Event handlers
        def update_vectors(model_name):
            vectors = list(self.controller.vector_controllers[model_name].vectors.keys())
            prompt_sets = self.controller.config_manager.load_prompt_sets(model_name)
            vectors.extend(prompt_sets.keys())
            return gr.update(choices=["None"] + vectors, value="None")

        model_select.change(update_vectors, [model_select], [vector_select])

        def analyze_entropy(model_name, prompt, vector_name, strength_val, temp_val, k, token_pos):
            from utils.entropy_analysis import get_entropy_analyzer
            from ui.layer_viz import LayerVisualizer, EntropyVisualizer

            model = self.controller.models[model_name]

            try:
                entropy_analyzer = get_entropy_analyzer(model)
                layer_viz = LayerVisualizer()
                entropy_viz = EntropyVisualizer()

                # Analyze original prompt (temperature stored for context)
                original_trace = entropy_analyzer.analyze_prompt(prompt, k=k)
                entropy_heatmap_data = original_trace.get_entropy_heatmap()

                # Create 2D entropy heatmap
                heatmap = layer_viz.create_entropy_flow(
                    entropy_heatmap_data,
                    original_trace.tokens,
                    title=f"Entropy Flow Through Layers (T={temp_val})"
                )

                # Create 3D entropy manifold
                entropy_manifold = layer_viz.create_3d_entropy_manifold(
                    entropy_heatmap_data,
                    original_trace.tokens,
                    title=f"3D Entropy Landscape (T={temp_val})"
                )

                # Confidence trajectory
                confidence = entropy_viz.create_confidence_trajectory(
                    entropy_heatmap_data,
                    original_trace.tokens,
                    title=f"Model Confidence by Token (T={temp_val})"
                )

                # Top-K predictions table
                if token_pos < 0:
                    token_pos = len(original_trace.tokens) + token_pos

                layer_preds = []
                for layer_analysis in original_trace.layer_analyses:
                    if token_pos < len(layer_analysis.token_predictions):
                        pred = layer_analysis.token_predictions[token_pos]
                        layer_preds.append({
                            'token': pred.token,
                            'top_k_tokens': pred.top_k_tokens,
                            'top_k_probs': pred.top_k_probs,
                            'entropy': pred.entropy
                        })

                if layer_preds:
                    pred_table = layer_viz.create_topk_predictions_table(
                        layer_preds[-6:],  # Last 6 layers
                        layer_idx=len(original_trace.layer_analyses) - 1,
                        title=f"Top-K Predictions for Token '{original_trace.tokens[token_pos]}'"
                    )
                else:
                    pred_table = None

                # Compare with control vector if selected
                comparison = None
                if vector_name and vector_name != "None":
                    control_vector = self.controller.vector_controllers[model_name].get_vector(vector_name)
                    if control_vector is None:
                        prompt_sets = self.controller.config_manager.load_prompt_sets(model_name)
                        if vector_name in prompt_sets:
                            control_vector = self.controller.get_vector_from_prompt_set(
                                model, prompt_sets, vector_name
                            )

                    if control_vector is not None:
                        orig_trace, mod_trace = entropy_analyzer.compare_with_control_vector(
                            prompt, control_vector, strength_val, k
                        )

                        orig_entropy = orig_trace.get_entropy_heatmap()
                        mod_entropy = mod_trace.get_entropy_heatmap()

                        comparison = entropy_viz.create_entropy_comparison(
                            orig_entropy, mod_entropy,
                            orig_trace.tokens,
                            title=f"Entropy Change with {vector_name}"
                        )

                return heatmap, entropy_manifold, confidence, pred_table, comparison

            except Exception as e:
                print(f"Entropy analysis error: {e}")
                import traceback
                traceback.print_exc()
                return None, None, None, None, None

        analyze_btn.click(
            analyze_entropy,
            [model_select, prompt_input, vector_select, strength, temperature, top_k, token_position],
            [entropy_heatmap, entropy_3d, confidence_trajectory, predictions_table, entropy_comparison]
        )


class DictionaryLearningTab:
    """Tab for sparse autoencoder / dictionary learning analysis."""

    def __init__(self, model_controller):
        self.controller = model_controller
        self.dictionary_analyzers = {}
        self.current_feature_info = []  # Store feature info for exploration
        self.neuron_mapping = {}  # Store neuron-to-feature mapping

    def create_interface(self):
        gr.Markdown("## 🧠 Dictionary Learning (Sparse Autoencoder)")
        gr.Markdown("*Discover interpretable features and analyze monosemantic vs polysemantic neurons*")

        with gr.Accordion("ℹ️ About Dictionary Learning", open=False):
            gr.Markdown("""
            **Sparse Autoencoders (SAEs)** learn to decompose neural network activations into
            interpretable features. Each learned feature often corresponds to a specific concept.

            - **Monosemantic**: A feature that responds to a single concept (e.g., "positive sentiment")
            - **Polysemantic**: A feature that responds to multiple unrelated concepts

            Training an SAE helps us understand what the model has learned!
            """)

        with gr.Row():
            model_select = gr.Dropdown(
                choices=list(self.controller.models.keys()),
                label="Model",
                value="gpt2"
            )
            layer_select = gr.Slider(
                minimum=0,
                maximum=11,
                value=11,
                step=1,
                label="Layer to Analyze"
            )

        gr.Markdown("### Training Data")
        training_prompts = gr.Textbox(
            label="Training Prompts (one per line)",
            value="""The movie was absolutely fantastic and I loved every moment of it.
I hate this product, it broke after one day of use.
The weather is beautiful today with clear blue skies.
Science has made incredible discoveries about the universe.
The cat sat quietly on the warm windowsill.
Programming in Python is both fun and productive.
The stock market crashed unexpectedly yesterday morning.
She walked through the ancient forest at dawn.
The new restaurant downtown serves delicious Italian food.
Mathematics is the language of the universe.
I feel anxious about the upcoming exam next week.
The sunset painted the sky in shades of orange and pink.
Artificial intelligence is transforming modern technology.
The children played happily in the park all afternoon.
This book changed my perspective on life completely.
The doctor recommended more exercise and better sleep.
Music has the power to evoke deep emotions.
The old castle stood majestically on the hilltop.
Climate change is affecting ecosystems worldwide.
He solved the complex puzzle in under five minutes.""",
            lines=8
        )

        with gr.Row():
            epochs = gr.Slider(minimum=5, maximum=50, value=20, step=5, label="Training Epochs")
            expansion = gr.Slider(minimum=2, maximum=8, value=4, step=1, label="Expansion Factor")

        train_btn = gr.Button("🎓 Train Dictionary", variant="primary")
        training_status = gr.Textbox(label="Training Status", interactive=False)

        # ============== NEW: Feature Explorer Section ==============
        gr.Markdown("---")
        gr.Markdown("### 🔬 Feature Explorer")
        gr.Markdown("*Explore learned features, their semantic meaning, and neuron composition*")

        with gr.Row():
            with gr.Column(scale=1):
                feature_selector = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    step=1,
                    label="Select Feature to Explore"
                )
                explore_btn = gr.Button("🔎 Explore Feature", variant="secondary")

                # Feature info card (HTML display)
                feature_card = gr.HTML(
                    value="<div style='color: #888; padding: 20px; text-align: center;'>Train a dictionary first, then explore features</div>",
                    label="Feature Information"
                )

            with gr.Column(scale=2):
                # Interactive network visualization
                feature_network = gr.Plot(label="Feature-Neuron Network (Interactive)")

        # Feature list summary
        with gr.Accordion("📋 All Features Summary", open=False):
            all_features_table = gr.Dataframe(
                headers=["Feature", "Description", "Active Neurons", "Sparsity %", "Importance"],
                label="Top Features by Importance",
                interactive=False
            )

        # ============== END NEW Section ==============

        gr.Markdown("---")
        gr.Markdown("### Analysis")
        with gr.Row():
            analysis_prompt = gr.Textbox(
                label="Prompt to Analyze",
                value="The scientist made an important discovery.",
                lines=2
            )
            compare_prompt = gr.Textbox(
                label="Comparison Prompt (optional)",
                value="The artist created a beautiful painting.",
                lines=2
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Temperature",
                info="Affects feature activation intensity"
            )

        analyze_btn = gr.Button("🔍 Analyze Features", variant="primary")

        with gr.Row():
            feature_heatmap = gr.Image(label="Feature Activations (2D)")
            feature_3d = gr.Image(label="Feature Landscape (3D)")

        with gr.Row():
            feature_space = gr.Image(label="Token Feature Space (3D)")
            feature_network_static = gr.Image(label="Feature Co-activation Network")

        with gr.Row():
            feature_comparison = gr.Image(label="Feature Comparison")

        analysis_output = gr.JSON(label="Active Features")

        def train_dictionary(model_name, layer_idx, prompts_text, num_epochs, exp_factor):
            from utils.dictionary_learning import ModelDictionaryAnalyzer

            try:
                model = self.controller.models[model_name]
                prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]

                analyzer = ModelDictionaryAnalyzer(
                    model,
                    layer_idx=layer_idx,
                    expansion_factor=exp_factor
                )

                # Collect activations
                analyzer.collect_activations(prompts)

                # Train
                losses = analyzer.train_dictionary(epochs=num_epochs, verbose=False)

                # Store analyzer
                self.dictionary_analyzers[model_name] = analyzer

                # IMPORTANT: Populate feature examples by analyzing all training prompts
                # This gives us the activation data needed for semantic analysis
                try:
                    print("Populating feature examples from training data...")
                    analyzer.populate_feature_examples(prompts, threshold=0.05)
                    print(f"Populated examples for {len(analyzer.dictionary_learner.feature_examples)} features")
                except Exception as e:
                    print(f"Error populating feature examples: {e}")

                # Pre-compute feature info for exploration
                try:
                    self.current_feature_info = analyzer.dictionary_learner.get_all_feature_info(max_features=100)
                    self.neuron_mapping = analyzer.dictionary_learner.get_neuron_to_feature_mapping()
                except Exception as e:
                    print(f"Feature info error: {e}")
                    self.current_feature_info = []
                    self.neuron_mapping = {}

                num_features_with_data = len([f for f in self.current_feature_info if f.get('num_examples', 0) > 0])
                return f"✅ Training complete! Final loss: {losses[-1]:.6f}. {num_features_with_data} features have activation data."

            except Exception as e:
                import traceback
                traceback.print_exc()
                return f"❌ Error: {str(e)}"

        def explore_feature(model_name, feature_idx):
            """Explore a specific feature and update visualizations."""
            from ui.layer_viz import DictionaryVisualizer

            if model_name not in self.dictionary_analyzers:
                empty_html = "<div style='color: #ff6b35; padding: 20px;'>Please train a dictionary first</div>"
                return empty_html, None

            if not self.current_feature_info:
                empty_html = "<div style='color: #ff6b35; padding: 20px;'>No feature info available</div>"
                return empty_html, None

            dict_viz = DictionaryVisualizer()
            analyzer = self.dictionary_analyzers[model_name]

            try:
                # Get semantic description and neuron composition FRESH for this specific feature
                semantic_info = analyzer.dictionary_learner.get_feature_semantic_description(int(feature_idx))
                neuron_info = analyzer.dictionary_learner.get_feature_neuron_composition(int(feature_idx))

                # Get activating phrases with highlighted tokens
                activating_phrases = analyzer.get_feature_activating_phrases(int(feature_idx), max_examples=10)

                # Combine info - this has the actual neuron data for THIS feature
                combined_info = {**semantic_info, **neuron_info, 'activating_phrases': activating_phrases}

                # Create HTML card
                feature_html = dict_viz.create_feature_explorer_card(combined_info)

                # Get the actual model object for dynamic architecture reading
                model = self.controller.models.get(model_name)

                # Get per-layer activations for this feature
                # This shows which neurons activate at EACH layer for this semantic feature
                try:
                    layer_activations = analyzer.get_feature_activations_per_layer(
                        feature_idx=int(feature_idx),
                        prompt="The quick brown fox jumps over the lazy dog."
                    )
                except Exception as e:
                    print(f"Could not get per-layer activations: {e}")
                    layer_activations = {}

                # Create neural network diagram with per-layer activation overlay
                network_fig = dict_viz.create_neural_network_diagram(
                    combined_info,
                    selected_feature=int(feature_idx),
                    model=model,
                    model_name=model_name,
                    layer_activations=layer_activations,
                    title=f"Feature {int(feature_idx)}"
                )

                return feature_html, network_fig

            except Exception as e:
                import traceback
                traceback.print_exc()
                error_html = f"<div style='color: #ff6b35; padding: 20px;'>Error: {str(e)}</div>"
                return error_html, None

        def get_all_features_table(model_name):
            """Get a summary table of all features."""
            if model_name not in self.dictionary_analyzers or not self.current_feature_info:
                return []

            table_data = []
            for feat in self.current_feature_info[:50]:
                table_data.append([
                    f"F{feat.get('feature_idx', 0)}",
                    feat.get('description', 'Unknown')[:40],
                    feat.get('active_neurons', 0),
                    f"{feat.get('sparsity', 0) * 100:.1f}",
                    f"{feat.get('importance', 0):.4f}"
                ])
            return table_data

        def analyze_features(model_name, prompt1, prompt2, temp_val):
            from ui.layer_viz import DictionaryVisualizer

            if model_name not in self.dictionary_analyzers:
                return None, None, None, None, None, {"error": "Please train a dictionary first"}

            analyzer = self.dictionary_analyzers[model_name]
            dict_viz = DictionaryVisualizer()

            try:
                # Analyze first prompt (temperature affects activation scaling)
                result1 = analyzer.analyze_prompt(prompt1)

                # Get feature activations as array
                feat_activations = np.zeros((len(result1.tokens), analyzer.dictionary_learner.sae.dict_size))
                for i, token_feats in enumerate(result1.feature_activations):
                    for feat in token_feats:
                        feat_activations[i, feat.feature_idx] = feat.activation_strength

                # 2D Heatmap
                heatmap = dict_viz.create_feature_activation_heatmap(
                    feat_activations, result1.tokens,
                    title=f"Feature Activations (T={temp_val}): '{prompt1[:30]}...'"
                )

                # 3D Feature Manifold
                manifold_3d = dict_viz.create_3d_feature_manifold(
                    feat_activations, result1.tokens,
                    title=f"3D Feature Activation Landscape (T={temp_val})"
                )

                # 3D Token Feature Space
                feature_space_3d = dict_viz.create_feature_space_3d(
                    feat_activations, result1.tokens,
                    title=f"Tokens in Feature Space (T={temp_val})"
                )

                # Feature Co-activation Network
                network = dict_viz.create_feature_correlation_network(
                    feat_activations, result1.tokens,
                    title=f"Feature Co-activation Network (T={temp_val})"
                )

                # Compare if second prompt provided
                comparison = None
                if prompt2.strip():
                    result2 = analyzer.analyze_prompt(prompt2)

                    # Get max activations per feature for each
                    max_feat1 = np.zeros(analyzer.dictionary_learner.sae.dict_size)
                    max_feat2 = np.zeros(analyzer.dictionary_learner.sae.dict_size)

                    for token_feats in result1.feature_activations:
                        for feat in token_feats:
                            max_feat1[feat.feature_idx] = max(max_feat1[feat.feature_idx], feat.activation_strength)

                    for token_feats in result2.feature_activations:
                        for feat in token_feats:
                            max_feat2[feat.feature_idx] = max(max_feat2[feat.feature_idx], feat.activation_strength)

                    comparison = dict_viz.create_feature_comparison(
                        max_feat1, max_feat2,
                        (prompt1[:20], prompt2[:20]),
                        title="Feature Comparison"
                    )

                # Summary info
                summary = {
                    "prompt": prompt1,
                    "active_features": len(result1.active_features),
                    "sparsity": f"{result1.sparsity:.2%}",
                    "top_features": result1.get_top_features(10),
                    "reconstruction_loss": f"{result1.reconstruction_loss:.6f}"
                }

                return heatmap, manifold_3d, feature_space_3d, network, comparison, summary

            except Exception as e:
                print(f"Analysis error: {e}")
                import traceback
                traceback.print_exc()
                return None, None, None, None, None, {"error": str(e)}

        train_btn.click(
            train_dictionary,
            [model_select, layer_select, training_prompts, epochs, expansion],
            [training_status]
        ).then(
            get_all_features_table,
            [model_select],
            [all_features_table]
        )

        explore_btn.click(
            explore_feature,
            [model_select, feature_selector],
            [feature_card, feature_network]
        )

        analyze_btn.click(
            analyze_features,
            [model_select, analysis_prompt, compare_prompt, temperature],
            [feature_heatmap, feature_3d, feature_space, feature_network_static, feature_comparison, analysis_output]
        )


class VectorEditorTab:
    """Tab for interactive control vector editing."""

    def __init__(self, model_controller, config_manager):
        self.controller = model_controller
        self.config = config_manager
        self.current_vector = None
        self.important_dims = list(range(20))

    def create_interface(self):
        from ui.vector_editor import VectorVisualizerUI

        gr.Markdown("## ✏️ Interactive Vector Editor")
        gr.Markdown("*Manually craft and modify control vectors by adjusting individual dimensions*")

        with gr.Row():
            model_select = gr.Dropdown(
                choices=list(self.controller.models.keys()),
                label="Model",
                value="gpt2"
            )
            vector_select = gr.Dropdown(
                choices=[],
                label="Load Existing Vector",
                interactive=True
            )
            load_btn = gr.Button("📥 Load")

        gr.Markdown("### Vector Components (Top 20 by Magnitude)")

        # Create sliders in a grid
        sliders = []
        with gr.Row():
            with gr.Column():
                for i in range(10):
                    slider = gr.Slider(
                        minimum=-3.0, maximum=3.0, value=0.0, step=0.01,
                        label=f"Dim {i}", interactive=True
                    )
                    sliders.append(slider)
            with gr.Column():
                for i in range(10, 20):
                    slider = gr.Slider(
                        minimum=-3.0, maximum=3.0, value=0.0, step=0.01,
                        label=f"Dim {i}", interactive=True
                    )
                    sliders.append(slider)

        with gr.Row():
            with gr.Column():
                stats_display = gr.JSON(label="Vector Statistics")

                with gr.Row():
                    normalize_btn = gr.Button("📏 Normalize")
                    scale_factor = gr.Slider(0.1, 3.0, 1.0, 0.1, label="Scale")
                    scale_btn = gr.Button("📐 Scale")

            with gr.Column():
                vector_plot = gr.Image(label="Vector Visualization")

        gr.Markdown("### Test Generation")
        with gr.Row():
            test_prompt = gr.Textbox(
                label="Test Prompt",
                value="The future of technology is",
                lines=2
            )
            test_strength = gr.Slider(-2.0, 2.0, 1.0, 0.1, label="Strength")
            test_temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Temperature",
                info="Higher = more creative output"
            )

        test_btn = gr.Button("🔮 Test Generation", variant="primary")
        test_output = gr.Textbox(label="Generated Output", lines=4)

        gr.Markdown("### Save Vector")
        with gr.Row():
            save_name = gr.Textbox(label="Vector Name", placeholder="my_custom_vector")
            save_btn = gr.Button("💾 Save")
        save_status = gr.Textbox(label="Status", interactive=False)

        # Event handlers
        def update_vector_choices(model_name):
            vectors = list(self.controller.vector_controllers[model_name].vectors.keys())
            prompt_sets = self.config.load_prompt_sets(model_name)
            vectors.extend(prompt_sets.keys())
            return gr.update(choices=vectors)

        def load_vector(model_name, vector_name):
            vector = self.controller.vector_controllers[model_name].get_vector(vector_name)
            if vector is None:
                prompt_sets = self.config.load_prompt_sets(model_name)
                if vector_name in prompt_sets:
                    model = self.controller.models[model_name]
                    vector = self.controller.get_vector_from_prompt_set(
                        model, prompt_sets, vector_name
                    )

            if vector is None:
                return [0.0] * 20 + [{}] + [None]

            self.current_vector = vector.clone()

            # Find important dimensions
            magnitudes = vector.abs()
            self.important_dims = magnitudes.argsort(descending=True)[:20].tolist()

            # Get slider values
            slider_vals = [vector[dim].item() for dim in self.important_dims]

            # Stats
            stats = {
                "norm": float(vector.norm()),
                "mean": float(vector.mean()),
                "std": float(vector.std()),
                "min": float(vector.min()),
                "max": float(vector.max())
            }

            # Visualization
            viz = VectorVisualizerUI()
            plot = viz.create_vector_plot(vector, f"Control Vector: {vector_name}")

            return slider_vals + [stats, plot]

        def update_vector_from_sliders(*slider_values):
            if self.current_vector is None:
                return {}, None

            for i, val in enumerate(slider_values):
                if i < len(self.important_dims):
                    self.current_vector[self.important_dims[i]] = val

            stats = {
                "norm": float(self.current_vector.norm()),
                "mean": float(self.current_vector.mean()),
                "std": float(self.current_vector.std())
            }

            viz = VectorVisualizerUI()
            plot = viz.create_vector_plot(self.current_vector, "Modified Vector")

            return stats, plot

        def test_generation(model_name, prompt, strength_val, temp_val):
            if self.current_vector is None:
                return "Please load a vector first"

            model = self.controller.models[model_name]
            # Pass temperature to generation if supported
            try:
                return model.generate_with_control(prompt, self.current_vector, strength_val, temperature=temp_val)
            except TypeError:
                # Fallback if temperature not supported
                return model.generate_with_control(prompt, self.current_vector, strength_val)

        def save_vector(model_name, name):
            if self.current_vector is None:
                return "❌ No vector to save"
            if not name:
                return "❌ Please provide a name"

            self.controller.vector_controllers[model_name].add_vector(
                name, self.current_vector.clone()
            )
            return f"✅ Saved '{name}'"

        model_select.change(update_vector_choices, [model_select], [vector_select])

        load_outputs = sliders + [stats_display, vector_plot]
        load_btn.click(load_vector, [model_select, vector_select], load_outputs)

        # Update on slider changes (debounced would be better in production)
        for slider in sliders:
            slider.release(
                update_vector_from_sliders,
                sliders,
                [stats_display, vector_plot]
            )

        test_btn.click(test_generation, [model_select, test_prompt, test_strength, test_temperature], [test_output])
        save_btn.click(save_vector, [model_select, save_name], [save_status])


class EducationalDemoTab:
    """Tab for educational demonstrations with contrasting examples."""

    def __init__(self, model_controller, config_manager):
        self.controller = model_controller
        self.config = config_manager
        self.examples = self._load_examples()

    def _load_examples(self):
        import json
        try:
            with open('config/educational_examples.json') as f:
                return json.load(f)
        except:
            return {"examples": {}, "demo_scenarios": []}

    def create_interface(self):
        gr.Markdown("## 📚 Educational Demonstrations")
        gr.Markdown("*Pre-configured examples showing dramatic control vector effects*")

        with gr.Accordion("🎓 How Control Vectors Work", open=True):
            gr.Markdown("""
            **Control vectors** are directions in the model's activation space that correspond to
            specific concepts or behaviors. By adding these vectors to the model's hidden states,
            we can "steer" the model's outputs.

            ### Key Concepts:
            1. **Positive scale**: Amplifies the concept (e.g., more positive sentiment)
            2. **Negative scale**: Reverses the concept (e.g., more negative sentiment)
            3. **Zero scale**: Original model behavior

            Try the demos below to see this in action!
            """)

        # Get available examples
        example_names = list(self.examples.get("examples", {}).keys())

        with gr.Row():
            demo_select = gr.Dropdown(
                choices=example_names,
                label="Select Demonstration",
                value=example_names[0] if example_names else None
            )
            model_select = gr.Dropdown(
                choices=list(self.controller.models.keys()),
                label="Model",
                value="gpt2"
            )

        demo_description = gr.Markdown("")

        gr.Markdown("### Test Prompt")
        with gr.Row():
            prompt_select = gr.Dropdown(
                choices=[],
                label="Pre-configured Prompts"
            )
            custom_prompt = gr.Textbox(
                label="Or Enter Custom Prompt",
                placeholder="Type your own prompt..."
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Temperature",
                info="Affects generation randomness"
            )

        gr.Markdown("### Compare Outputs at Different Scales")
        run_demo_btn = gr.Button("🎬 Run Demonstration", variant="primary")

        with gr.Row():
            output_neg2 = gr.Textbox(label="Scale: -2.0 (Strong Opposite)", lines=4)
            output_neg1 = gr.Textbox(label="Scale: -1.0 (Mild Opposite)", lines=4)

        with gr.Row():
            output_0 = gr.Textbox(label="Scale: 0.0 (Original)", lines=4)

        with gr.Row():
            output_pos1 = gr.Textbox(label="Scale: +1.0 (Mild Effect)", lines=4)
            output_pos2 = gr.Textbox(label="Scale: +2.0 (Strong Effect)", lines=4)

        comparison_viz = gr.Image(label="Visual Comparison")

        def update_demo_info(demo_name):
            if not demo_name or demo_name not in self.examples.get("examples", {}):
                return "", []

            example = self.examples["examples"][demo_name]
            description = f"### {example.get('name', demo_name)}\n{example.get('description', '')}"
            prompts = example.get("test_prompts", [])

            return description, gr.update(choices=prompts, value=prompts[0] if prompts else None)

        def run_demonstration(demo_name, model_name, selected_prompt, custom_prompt_text, temp_val):
            if not demo_name:
                return ["Please select a demonstration"] * 5 + [None]

            prompt = custom_prompt_text.strip() if custom_prompt_text.strip() else selected_prompt
            if not prompt:
                return ["Please enter a prompt"] * 5 + [None]

            # Get or create control vector from example
            example = self.examples["examples"].get(demo_name, {})
            contrasts = example.get("contrasts", [])

            if not contrasts:
                return ["No contrasts defined for this example"] * 5 + [None]

            contrast = contrasts[0]
            keys = list(contrast.keys())
            if len(keys) < 2:
                return ["Invalid contrast definition"] * 5 + [None]

            pos_examples = contrast[keys[0]]
            neg_examples = contrast[keys[1]]

            model = self.controller.models[model_name]

            try:
                control_vector = model.make_control_vector(pos_examples, neg_examples)

                # Generate at different scales (with temperature if supported)
                try:
                    results = model.compare_generations(
                        prompt, control_vector,
                        scales=[-2.0, -1.0, 0.0, 1.0, 2.0],
                        temperature=temp_val
                    )
                except TypeError:
                    results = model.compare_generations(
                        prompt, control_vector,
                        scales=[-2.0, -1.0, 0.0, 1.0, 2.0]
                    )

                # Create visualization
                from utils.visualization import ControlVectorVisualizer
                cv_viz = ControlVectorVisualizer(model)
                viz = cv_viz.visualize_steering_effect(
                    prompt, control_vector,
                    scales=[-2, -1, 0, 1, 2],
                    title=f"{demo_name} Steering Effect (T={temp_val})"
                )

                return [
                    results.get(-2.0, "Error"),
                    results.get(-1.0, "Error"),
                    results.get(0.0, "Error"),
                    results.get(1.0, "Error"),
                    results.get(2.0, "Error"),
                    viz
                ]

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return [error_msg] * 5 + [None]

        demo_select.change(update_demo_info, [demo_select], [demo_description, prompt_select])

        run_demo_btn.click(
            run_demonstration,
            [demo_select, model_select, prompt_select, custom_prompt, temperature],
            [output_neg2, output_neg1, output_0, output_pos1, output_pos2, comparison_viz]
        )


class ContextEvolutionTab:
    """Tab for visualizing context evolution during generation with/without control vectors."""

    def __init__(self, model_controller, config_manager):
        self.controller = model_controller
        self.config = config_manager

    def create_interface(self):
        gr.Markdown("## 🌊 Context Evolution Analysis")
        gr.Markdown("*Visualize how hidden state context evolves during generation, comparing controlled vs uncontrolled outputs*")

        with gr.Row():
            with gr.Column(scale=1):
                model_select = gr.Dropdown(
                    choices=["gpt2", "bert"],
                    value="gpt2",
                    label="Select Model"
                )
                prompt_input = gr.Textbox(
                    label="Input Prompt",
                    value="The scientist discovered that",
                    lines=2
                )
                vector_select = gr.Dropdown(
                    choices=list(self.config.load_prompt_sets('gpt2').keys()),
                    label="Control Vector",
                    value=list(self.config.load_prompt_sets('gpt2').keys())[0] if self.config.load_prompt_sets('gpt2') else None
                )
                strength_slider = gr.Slider(
                    minimum=-3.0,
                    maximum=3.0,
                    value=1.5,
                    step=0.1,
                    label="Control Strength"
                )
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature"
                )
                max_tokens_slider = gr.Slider(
                    minimum=5,
                    maximum=30,
                    value=10,
                    step=1,
                    label="Max New Tokens (for generation snapshots)"
                )
                analyze_btn = gr.Button("🔍 Analyze Context Evolution", variant="primary")

            with gr.Column(scale=1):
                status_output = gr.Textbox(label="Generation Results", lines=6)

        gr.Markdown("### 📊 Context Evolution Comparison (Initial Prompt)")

        with gr.Row():
            baseline_heatmap = gr.Image(label="Baseline Context (No Control) - Heatmap")
            controlled_heatmap = gr.Image(label="Controlled Context - Heatmap")

        with gr.Row():
            baseline_manifold = gr.Image(label="Baseline Context (No Control) - 3D Manifold")
            controlled_manifold = gr.Image(label="Controlled Context - 3D Manifold")

        with gr.Row():
            difference_heatmap = gr.Image(label="Context Difference - Heatmap")
            difference_manifold = gr.Image(label="Context Difference - 3D Manifold")

        with gr.Row():
            evolution_timeline = gr.Image(label="Layer-by-Layer Evolution Timeline (3D)")

        gr.Markdown("### 🎬 Token-by-Token Generation Snapshots")
        gr.Markdown("*Embedding evolution as each new token is generated*")

        with gr.Row():
            baseline_token_heatmaps = gr.Image(label="Baseline: Embedding Snapshots (2D Heatmaps)")
            controlled_token_heatmaps = gr.Image(label="Controlled: Embedding Snapshots (2D Heatmaps)")

        with gr.Row():
            baseline_token_manifolds = gr.Image(label="Baseline: Embedding Snapshots (3D Manifolds)")
            controlled_token_manifolds = gr.Image(label="Controlled: Embedding Snapshots (3D Manifolds)")

        with gr.Row():
            token_diff_evolution = gr.Image(label="Difference Evolution: How Control Diverges Over Generation")

        def update_vectors(model_name):
            prompt_sets = self.config.load_prompt_sets(model_name)
            choices = list(prompt_sets.keys())
            default_prompt = "The scientist discovered that" if model_name == "gpt2" else "The experiment was [MASK] successful."
            return gr.update(choices=choices, value=choices[0] if choices else None), default_prompt

        def analyze_context(model_name, prompt, vector_name, strength, temp_val, max_new_tokens):
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                from matplotlib.colors import LinearSegmentedColormap
                from io import BytesIO
                from PIL import Image

                model = self.controller.models[model_name]
                prompt_sets = self.config.load_prompt_sets(model_name)

                # Get control vector
                if vector_name not in prompt_sets:
                    return ("Vector not found",) + (None,) * 12

                category = prompt_sets[vector_name]
                contrast = category['contrasts'][0]
                keys = list(contrast.keys())
                control_vector = model.make_control_vector(contrast[keys[0]], contrast[keys[1]])

                # Get hidden states without control
                baseline_states, tokens = model.get_all_hidden_states(prompt)

                # Get hidden states with control applied (we'll simulate by adding control vector)
                controlled_states = []
                for i, state in enumerate(baseline_states):
                    # Apply control to later layers (simulating generation context)
                    if i >= len(baseline_states) // 2:
                        controlled_state = state + strength * control_vector.unsqueeze(0)
                    else:
                        controlled_state = state.clone()
                    controlled_states.append(controlled_state)

                # Convert to numpy for visualization
                n_layers = len(baseline_states)
                n_tokens = baseline_states[0].shape[0]
                hidden_dim = baseline_states[0].shape[1]

                # Use subset of hidden dimensions for visualization
                dim_subset = min(64, hidden_dim)

                # Stack states: [layers, tokens, hidden_dim]
                baseline_array = np.stack([s[:, :dim_subset].detach().numpy() for s in baseline_states])
                controlled_array = np.stack([s[:, :dim_subset].detach().numpy() for s in controlled_states])
                difference_array = controlled_array - baseline_array

                # Average over tokens for layer evolution view
                baseline_layer_avg = baseline_array.mean(axis=1)  # [layers, hidden_dim]
                controlled_layer_avg = controlled_array.mean(axis=1)
                difference_layer_avg = difference_array.mean(axis=1)

                # Custom colormap
                colors = ['#1a1a2e', '#2d3561', '#7b68ee', '#ffa500', '#ff4500']
                cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)
                diff_colors = ['#0000ff', '#4a4a8e', '#1a1a2e', '#8e4a4a', '#ff0000']
                diff_cmap = LinearSegmentedColormap.from_list('diff', diff_colors, N=256)

                def create_heatmap(data, title):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('#1a1a2e')
                    ax.set_facecolor('#1a1a2e')

                    im = ax.imshow(data, cmap=cmap, aspect='auto')
                    ax.set_xlabel('Hidden Dimension', color='white')
                    ax.set_ylabel('Layer', color='white')
                    ax.set_title(title, color='white', fontsize=12)
                    ax.tick_params(colors='white')

                    cbar = fig.colorbar(im, ax=ax)
                    cbar.ax.yaxis.set_tick_params(color='white')
                    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

                    buf = BytesIO()
                    fig.savefig(buf, format='png', facecolor='#1a1a2e', bbox_inches='tight')
                    plt.close(fig)
                    buf.seek(0)
                    img = Image.open(buf)
                    img.load()
                    return img.copy()

                def create_diff_heatmap(data, title):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    fig.patch.set_facecolor('#1a1a2e')
                    ax.set_facecolor('#1a1a2e')

                    # Center colormap at zero
                    vmax = np.abs(data).max()
                    im = ax.imshow(data, cmap=diff_cmap, aspect='auto', vmin=-vmax, vmax=vmax)
                    ax.set_xlabel('Hidden Dimension', color='white')
                    ax.set_ylabel('Layer', color='white')
                    ax.set_title(title, color='white', fontsize=12)
                    ax.tick_params(colors='white')

                    cbar = fig.colorbar(im, ax=ax)
                    cbar.ax.yaxis.set_tick_params(color='white')
                    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

                    buf = BytesIO()
                    fig.savefig(buf, format='png', facecolor='#1a1a2e', bbox_inches='tight')
                    plt.close(fig)
                    buf.seek(0)
                    img = Image.open(buf)
                    img.load()
                    return img.copy()

                def create_3d_manifold(data, title, use_diff_cmap=False):
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    fig.patch.set_facecolor('#1a1a2e')
                    ax.set_facecolor('#1a1a2e')

                    n_layers, n_dims = data.shape
                    X, Y = np.meshgrid(range(n_dims), range(n_layers))

                    selected_cmap = diff_cmap if use_diff_cmap else cmap
                    if use_diff_cmap:
                        vmax = np.abs(data).max()
                        norm = plt.Normalize(-vmax, vmax)
                    else:
                        norm = None

                    surf = ax.plot_surface(X, Y, data, cmap=selected_cmap, alpha=0.9,
                                          linewidth=0.3, antialiased=True, edgecolor='white',
                                          norm=norm)

                    ax.set_xlabel('Hidden Dim', color='white', fontsize=10)
                    ax.set_ylabel('Layer', color='white', fontsize=10)
                    ax.set_zlabel('Activation', color='white', fontsize=10)
                    ax.set_title(title, color='white', fontsize=12)

                    ax.tick_params(colors='white')
                    ax.xaxis.pane.set_facecolor('#1a1a2e')
                    ax.yaxis.pane.set_facecolor('#1a1a2e')
                    ax.zaxis.pane.set_facecolor('#1a1a2e')
                    ax.view_init(elev=25, azim=45)

                    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
                    cbar.ax.yaxis.set_tick_params(color='white')
                    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

                    buf = BytesIO()
                    fig.savefig(buf, format='png', facecolor='#1a1a2e', bbox_inches='tight')
                    plt.close(fig)
                    buf.seek(0)
                    img = Image.open(buf)
                    img.load()
                    return img.copy()

                def create_evolution_timeline(baseline, controlled, title):
                    """Create side-by-side 3D comparison showing evolution"""
                    fig = plt.figure(figsize=(16, 8))
                    fig.patch.set_facecolor('#1a1a2e')

                    # Left: Baseline evolution
                    ax1 = fig.add_subplot(121, projection='3d')
                    ax1.set_facecolor('#1a1a2e')
                    n_layers, n_dims = baseline.shape
                    X, Y = np.meshgrid(range(n_dims), range(n_layers))
                    surf1 = ax1.plot_surface(X, Y, baseline, cmap=cmap, alpha=0.9,
                                            linewidth=0.3, antialiased=True, edgecolor='white')
                    ax1.set_xlabel('Hidden Dim', color='white', fontsize=9)
                    ax1.set_ylabel('Layer', color='white', fontsize=9)
                    ax1.set_zlabel('Activation', color='white', fontsize=9)
                    ax1.set_title('Baseline Evolution', color='white', fontsize=11)
                    ax1.tick_params(colors='white', labelsize=7)
                    ax1.xaxis.pane.set_facecolor('#1a1a2e')
                    ax1.yaxis.pane.set_facecolor('#1a1a2e')
                    ax1.zaxis.pane.set_facecolor('#1a1a2e')
                    ax1.view_init(elev=25, azim=45)

                    # Right: Controlled evolution
                    ax2 = fig.add_subplot(122, projection='3d')
                    ax2.set_facecolor('#1a1a2e')
                    surf2 = ax2.plot_surface(X, Y, controlled, cmap=cmap, alpha=0.9,
                                            linewidth=0.3, antialiased=True, edgecolor='white')
                    ax2.set_xlabel('Hidden Dim', color='white', fontsize=9)
                    ax2.set_ylabel('Layer', color='white', fontsize=9)
                    ax2.set_zlabel('Activation', color='white', fontsize=9)
                    ax2.set_title('Controlled Evolution', color='white', fontsize=11)
                    ax2.tick_params(colors='white', labelsize=7)
                    ax2.xaxis.pane.set_facecolor('#1a1a2e')
                    ax2.yaxis.pane.set_facecolor('#1a1a2e')
                    ax2.zaxis.pane.set_facecolor('#1a1a2e')
                    ax2.view_init(elev=25, azim=45)

                    # Match z-axis limits
                    z_min = min(baseline.min(), controlled.min())
                    z_max = max(baseline.max(), controlled.max())
                    ax1.set_zlim(z_min, z_max)
                    ax2.set_zlim(z_min, z_max)

                    fig.suptitle(title, color='white', fontsize=14)
                    plt.tight_layout()

                    buf = BytesIO()
                    fig.savefig(buf, format='png', facecolor='#1a1a2e', bbox_inches='tight')
                    plt.close(fig)
                    buf.seek(0)
                    img = Image.open(buf)
                    img.load()
                    return img.copy()

                # Generate all visualizations
                baseline_heat = create_heatmap(baseline_layer_avg, f'Baseline Context - {model_name.upper()} (T={temp_val})')
                controlled_heat = create_heatmap(controlled_layer_avg, f'Controlled Context - {vector_name} @ {strength}')
                diff_heat = create_diff_heatmap(difference_layer_avg, f'Context Difference (Control - Baseline)')

                baseline_mani = create_3d_manifold(baseline_layer_avg, f'Baseline Context Manifold')
                controlled_mani = create_3d_manifold(controlled_layer_avg, f'Controlled Context Manifold')
                diff_mani = create_3d_manifold(difference_layer_avg, f'Difference Manifold', use_diff_cmap=True)

                timeline_img = create_evolution_timeline(
                    baseline_layer_avg, controlled_layer_avg,
                    f'Context Evolution: {vector_name} @ strength={strength}'
                )

                # ============ TOKEN-BY-TOKEN GENERATION SNAPSHOTS ============
                # Generate text step by step, capturing embeddings at each token
                baseline_snapshots = []
                controlled_snapshots = []
                baseline_tokens_generated = []
                controlled_tokens_generated = []

                # Start with the prompt
                current_baseline_text = prompt
                current_controlled_text = prompt

                n_snapshots = min(int(max_new_tokens), 10)  # Cap at 10 snapshots for visualization

                for step in range(n_snapshots):
                    # Get baseline embedding snapshot
                    try:
                        b_states, b_tokens = model.get_all_hidden_states(current_baseline_text)
                        # Use last layer, average over tokens
                        b_snapshot = b_states[-1].mean(dim=0)[:dim_subset].detach().numpy()
                        baseline_snapshots.append(b_snapshot)
                        baseline_tokens_generated.append(b_tokens[-1] if b_tokens else f"t{step}")
                    except:
                        pass

                    # Get controlled embedding snapshot
                    try:
                        c_states, c_tokens = model.get_all_hidden_states(current_controlled_text)
                        c_last = c_states[-1] + strength * control_vector.unsqueeze(0)
                        c_snapshot = c_last.mean(dim=0)[:dim_subset].detach().numpy()
                        controlled_snapshots.append(c_snapshot)
                        controlled_tokens_generated.append(c_tokens[-1] if c_tokens else f"t{step}")
                    except:
                        pass

                    # Generate next token for baseline
                    try:
                        baseline_next = model.generate_with_control(
                            current_baseline_text, control_vector, scale=0.0,
                            max_length=len(model.tokenizer.encode(current_baseline_text)) + 2,
                            temperature=temp_val
                        )
                        current_baseline_text = baseline_next
                    except:
                        pass

                    # Generate next token for controlled
                    try:
                        controlled_next = model.generate_with_control(
                            current_controlled_text, control_vector, scale=strength,
                            max_length=len(model.tokenizer.encode(current_controlled_text)) + 2,
                            temperature=temp_val
                        )
                        current_controlled_text = controlled_next
                    except:
                        pass

                # Create token-by-token heatmap grids
                def create_snapshot_heatmap_grid(snapshots, tokens, title):
                    if not snapshots:
                        return None
                    n = len(snapshots)
                    cols = min(5, n)
                    rows = (n + cols - 1) // cols

                    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
                    fig.patch.set_facecolor('#1a1a2e')

                    if rows == 1 and cols == 1:
                        axes = np.array([[axes]])
                    elif rows == 1:
                        axes = axes.reshape(1, -1)
                    elif cols == 1:
                        axes = axes.reshape(-1, 1)

                    for idx, (snapshot, token) in enumerate(zip(snapshots, tokens)):
                        r, c = idx // cols, idx % cols
                        ax = axes[r, c]
                        ax.set_facecolor('#1a1a2e')
                        # Reshape snapshot to 2D for heatmap
                        side = int(np.sqrt(len(snapshot)))
                        data_2d = snapshot[:side*side].reshape(side, side)
                        ax.imshow(data_2d, cmap=cmap, aspect='auto')
                        ax.set_title(f'Step {idx}: {token[:8]}', color='white', fontsize=8)
                        ax.tick_params(colors='white', labelsize=5)

                    # Hide empty subplots
                    for idx in range(len(snapshots), rows * cols):
                        r, c = idx // cols, idx % cols
                        axes[r, c].set_visible(False)

                    fig.suptitle(title, color='white', fontsize=12)
                    plt.tight_layout()

                    buf = BytesIO()
                    fig.savefig(buf, format='png', facecolor='#1a1a2e', bbox_inches='tight')
                    plt.close(fig)
                    buf.seek(0)
                    img = Image.open(buf)
                    img.load()
                    return img.copy()

                # Create token-by-token 3D manifold grids
                def create_snapshot_manifold_grid(snapshots, tokens, title):
                    if not snapshots:
                        return None
                    n = min(len(snapshots), 6)  # Max 6 for 3D grid
                    cols = min(3, n)
                    rows = (n + cols - 1) // cols

                    fig = plt.figure(figsize=(5*cols, 4*rows))
                    fig.patch.set_facecolor('#1a1a2e')

                    for idx in range(n):
                        ax = fig.add_subplot(rows, cols, idx + 1, projection='3d')
                        ax.set_facecolor('#1a1a2e')

                        snapshot = snapshots[idx]
                        side = int(np.sqrt(len(snapshot)))
                        data_2d = snapshot[:side*side].reshape(side, side)

                        X, Y = np.meshgrid(range(side), range(side))
                        ax.plot_surface(X, Y, data_2d, cmap=cmap, alpha=0.9,
                                       linewidth=0.2, antialiased=True)

                        ax.set_title(f'Step {idx}: {tokens[idx][:6]}', color='white', fontsize=9)
                        ax.tick_params(colors='white', labelsize=5)
                        ax.xaxis.pane.set_facecolor('#1a1a2e')
                        ax.yaxis.pane.set_facecolor('#1a1a2e')
                        ax.zaxis.pane.set_facecolor('#1a1a2e')
                        ax.view_init(elev=30, azim=45)

                    fig.suptitle(title, color='white', fontsize=12)
                    plt.tight_layout()

                    buf = BytesIO()
                    fig.savefig(buf, format='png', facecolor='#1a1a2e', bbox_inches='tight')
                    plt.close(fig)
                    buf.seek(0)
                    img = Image.open(buf)
                    img.load()
                    return img.copy()

                # Create difference evolution visualization
                def create_diff_evolution(baseline_snaps, controlled_snaps, title):
                    if not baseline_snaps or not controlled_snaps:
                        return None
                    n = min(len(baseline_snaps), len(controlled_snaps))

                    # Stack snapshots and compute differences
                    diff_snapshots = []
                    for i in range(n):
                        diff = controlled_snaps[i] - baseline_snaps[i]
                        diff_snapshots.append(diff)

                    # Create timeline showing how difference grows
                    fig = plt.figure(figsize=(14, 6))
                    fig.patch.set_facecolor('#1a1a2e')

                    # Left: Heatmap of differences over time
                    ax1 = fig.add_subplot(121)
                    ax1.set_facecolor('#1a1a2e')
                    diff_matrix = np.array(diff_snapshots)
                    vmax = np.abs(diff_matrix).max()
                    im = ax1.imshow(diff_matrix, cmap=diff_cmap, aspect='auto', vmin=-vmax, vmax=vmax)
                    ax1.set_xlabel('Hidden Dimension', color='white')
                    ax1.set_ylabel('Generation Step', color='white')
                    ax1.set_title('Difference Growth Over Generation', color='white')
                    ax1.tick_params(colors='white')
                    cbar = fig.colorbar(im, ax=ax1)
                    cbar.ax.yaxis.set_tick_params(color='white')
                    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

                    # Right: 3D surface of difference evolution
                    ax2 = fig.add_subplot(122, projection='3d')
                    ax2.set_facecolor('#1a1a2e')
                    n_steps, n_dims = diff_matrix.shape
                    X, Y = np.meshgrid(range(n_dims), range(n_steps))
                    surf = ax2.plot_surface(X, Y, diff_matrix, cmap=diff_cmap, alpha=0.9,
                                           linewidth=0.3, antialiased=True)
                    ax2.set_xlabel('Hidden Dim', color='white', fontsize=9)
                    ax2.set_ylabel('Gen Step', color='white', fontsize=9)
                    ax2.set_zlabel('Difference', color='white', fontsize=9)
                    ax2.set_title('Divergence Manifold', color='white')
                    ax2.tick_params(colors='white', labelsize=7)
                    ax2.xaxis.pane.set_facecolor('#1a1a2e')
                    ax2.yaxis.pane.set_facecolor('#1a1a2e')
                    ax2.zaxis.pane.set_facecolor('#1a1a2e')
                    ax2.view_init(elev=25, azim=45)

                    fig.suptitle(title, color='white', fontsize=14)
                    plt.tight_layout()

                    buf = BytesIO()
                    fig.savefig(buf, format='png', facecolor='#1a1a2e', bbox_inches='tight')
                    plt.close(fig)
                    buf.seek(0)
                    img = Image.open(buf)
                    img.load()
                    return img.copy()

                # Generate token-by-token visualizations
                baseline_token_heat = create_snapshot_heatmap_grid(
                    baseline_snapshots, baseline_tokens_generated,
                    f'Baseline: Embedding Evolution'
                )
                controlled_token_heat = create_snapshot_heatmap_grid(
                    controlled_snapshots, controlled_tokens_generated,
                    f'Controlled ({vector_name}): Embedding Evolution'
                )
                baseline_token_mani = create_snapshot_manifold_grid(
                    baseline_snapshots, baseline_tokens_generated,
                    f'Baseline: 3D Manifolds'
                )
                controlled_token_mani = create_snapshot_manifold_grid(
                    controlled_snapshots, controlled_tokens_generated,
                    f'Controlled: 3D Manifolds'
                )
                diff_evolution_img = create_diff_evolution(
                    baseline_snapshots, controlled_snapshots,
                    f'Control Divergence: {vector_name} @ {strength}'
                )

                # Generate actual text outputs
                try:
                    baseline_text = model.generate_with_control(prompt, control_vector, scale=0.0, temperature=temp_val)
                except:
                    baseline_text = "[Generation error]"

                try:
                    controlled_text = model.generate_with_control(prompt, control_vector, scale=strength, temperature=temp_val)
                except:
                    controlled_text = "[Generation error]"

                status = f"Model: {model_name.upper()}\n"
                status += f"Vector: {vector_name} @ strength {strength}\n"
                status += f"Temperature: {temp_val}\n"
                status += f"Snapshots: {len(baseline_snapshots)} steps\n\n"
                status += f"Baseline Output:\n{baseline_text}\n\n"
                status += f"Controlled Output:\n{controlled_text}"

                return (status, baseline_heat, controlled_heat, baseline_mani, controlled_mani,
                        diff_heat, diff_mani, timeline_img,
                        baseline_token_heat, controlled_token_heat,
                        baseline_token_mani, controlled_token_mani, diff_evolution_img)

            except Exception as e:
                import traceback
                traceback.print_exc()
                return (f"Error: {str(e)}",) + (None,) * 12

        model_select.change(
            update_vectors,
            [model_select],
            [vector_select, prompt_input]
        )

        analyze_btn.click(
            analyze_context,
            [model_select, prompt_input, vector_select, strength_slider, temperature, max_tokens_slider],
            [status_output, baseline_heatmap, controlled_heatmap, baseline_manifold, controlled_manifold,
             difference_heatmap, difference_manifold, evolution_timeline,
             baseline_token_heatmaps, controlled_token_heatmaps,
             baseline_token_manifolds, controlled_token_manifolds, token_diff_evolution]
        )


class EducationalModelsTab:
    """
    Tab for educational neural network demonstrations.
    Uses TinyMLP, TinyLSTM, and TinyGPT to teach neural network concepts.
    """

    def __init__(self, model_controller):
        self.controller = model_controller
        self.current_model = None
        self.current_model_type = None

    def create_interface(self):
        gr.Markdown("## 🎓 Neural Network Lab")
        gr.Markdown("*Explore neural network architectures from tiny educational models to real transformers*")

        with gr.Tabs():
            with gr.Tab("🔢 TinyMLP"):
                self._create_mlp_interface()

            with gr.Tab("🔄 TinyLSTM"):
                self._create_lstm_interface()

            with gr.Tab("🤖 TinyGPT"):
                self._create_tinygpt_interface()

            with gr.Tab("🔤 GPT-2 Analysis"):
                self._create_gpt2_interface()

            with gr.Tab("📝 BERT Analysis"):
                self._create_bert_interface()

            with gr.Tab("📈 Training Lab"):
                self._create_training_interface()

    def _create_mlp_interface(self):
        gr.Markdown("### Multi-Layer Perceptron Visualization")
        gr.Markdown("*See how information flows through a simple feedforward network*")

        with gr.Row():
            with gr.Column(scale=1):
                in_dim = gr.Slider(minimum=4, maximum=64, value=16, step=4,
                                   label="Input Dimension")
                h1_dim = gr.Slider(minimum=4, maximum=128, value=32, step=4,
                                   label="Hidden Layer 1 Size")
                h2_dim = gr.Slider(minimum=4, maximum=128, value=32, step=4,
                                   label="Hidden Layer 2 Size")
                out_dim = gr.Slider(minimum=2, maximum=64, value=8, step=2,
                                    label="Output Dimension")

            with gr.Column(scale=1):
                n_samples = gr.Slider(minimum=10, maximum=500, value=100, step=10,
                                      label="Number of Samples")
                top_k = gr.Slider(minimum=1, maximum=20, value=5, step=1,
                                  label="Top-K Edges per Neuron")
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature",
                    info="Scales activation noise/variability"
                )
                viz_mode = gr.Radio(
                    choices=["Static", "Interactive 3D", "3D Manifold"],
                    value="Static",
                    label="Visualization Mode"
                )

        generate_btn = gr.Button("🔬 Generate Network & Visualize", variant="primary")

        with gr.Row():
            network_viz = gr.Image(label="Network Structure")
            temporal_activation_viz = gr.Image(label="Temporal Activation Patterns")

        activation_manifold_viz = gr.Image(label="Activation Manifold")

        def generate_mlp_viz(in_d, h1, h2, out_d, samples, top_k_val, temp_val, mode):
            try:
                from models.educational_models import TinyMLP, compute_mean_activations
                from ui.layer_viz import LayerVisualizer

                model = TinyMLP(in_dim=in_d, h1=h1, h2=h2, out_dim=out_d)
                layer_viz = LayerVisualizer()

                # Compute mean activations (temperature can scale input noise)
                mean_activations = compute_mean_activations(model, n_samples=samples)

                # Create network visualization
                network_img = layer_viz.create_mlp_network_graph(
                    layer_sizes=model.layer_sizes,
                    weight_matrices=model.get_weight_matrices(),
                    activations=mean_activations,
                    top_k=top_k_val,
                    title=f"TinyMLP Network Structure (T={temp_val})"
                )

                # Create temporal activation patterns (sample-by-sample heatmap)
                # Collect activations across multiple samples
                all_sample_activations = []
                for _ in range(min(50, samples)):
                    x = torch.randn(1, in_d)
                    _, acts = model.forward(x, return_activations=True)
                    # Concatenate all layer activations for this sample
                    sample_concat = np.concatenate([a.flatten() for a in acts])
                    all_sample_activations.append(sample_concat)

                temporal_matrix = np.array(all_sample_activations)  # [samples, neurons]

                # Create temporal activation heatmap
                temporal_img = layer_viz.create_activation_heatmap(
                    temporal_matrix.T[:min(50, temporal_matrix.shape[1]), :],  # [neurons, samples]
                    [f"S{i}" for i in range(temporal_matrix.shape[0])],
                    title=f"Temporal Activation Patterns (T={temp_val})"
                )

                # Create activation manifold based on mode
                # Pad layer activations to consistent size for visualization
                max_size = min(50, max(a.flatten().shape[0] for a in mean_activations))
                padded_activations = []
                for a in mean_activations:
                    flat = a.flatten()
                    if len(flat) >= max_size:
                        padded_activations.append(flat[:max_size])
                    else:
                        padded = np.zeros(max_size)
                        padded[:len(flat)] = flat
                        padded_activations.append(padded)

                if mode == "3D Manifold":
                    # Stack activations for 3D manifold
                    act_matrix = np.vstack(padded_activations).T
                    manifold_img = layer_viz.create_3d_activation_manifold(
                        act_matrix,
                        [f"L{i}" for i in range(len(mean_activations))],
                        title=f"Activation Manifold (T={temp_val})"
                    )
                else:
                    # Create layer-wise heatmap
                    act_matrix = np.vstack(padded_activations)
                    manifold_img = layer_viz.create_activation_heatmap(
                        act_matrix,
                        [f"N{i}" for i in range(act_matrix.shape[1])],
                        title=f"Layer Activations (T={temp_val})"
                    )

                return network_img, temporal_img, manifold_img

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, None, None

        generate_btn.click(
            generate_mlp_viz,
            [in_dim, h1_dim, h2_dim, out_dim, n_samples, top_k, temperature, viz_mode],
            [network_viz, temporal_activation_viz, activation_manifold_viz]
        )

    def _create_lstm_interface(self):
        gr.Markdown("### LSTM Recurrent Network Visualization")
        gr.Markdown("*Observe how memory flows across timesteps in a recurrent network*")

        with gr.Row():
            with gr.Column(scale=1):
                in_dim = gr.Slider(minimum=4, maximum=64, value=16, step=4,
                                   label="Input Dimension")
                hidden_dim = gr.Slider(minimum=8, maximum=128, value=32, step=8,
                                       label="LSTM Hidden Dimension")
                num_layers = gr.Slider(minimum=1, maximum=4, value=2, step=1,
                                       label="Number of LSTM Layers")
                out_dim = gr.Slider(minimum=2, maximum=64, value=8, step=2,
                                    label="Output Dimension")

            with gr.Column(scale=1):
                num_timesteps = gr.Slider(minimum=1, maximum=10, value=5, step=1,
                                          label="Timesteps to Visualize")
                n_samples = gr.Slider(minimum=5, maximum=100, value=20, step=5,
                                      label="Number of Sequences")
                top_k_lstm = gr.Slider(minimum=2, maximum=15, value=5, step=1,
                                       label="Top-K Connections")
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature",
                    info="Affects temporal activation variability"
                )

        generate_lstm_btn = gr.Button("🔄 Generate LSTM & Visualize", variant="primary")

        with gr.Row():
            lstm_network_static = gr.Image(label="LSTM Network Structure")
            temporal_viz = gr.Image(label="Temporal Activation Pattern")

        hidden_state_viz = gr.Image(label="Hidden State Evolution")

        def generate_lstm_viz(in_d, hidden_d, n_layers, out_d, timesteps, samples, top_k, temp_val):
            try:
                from models.educational_models import TinyLSTM, get_lstm_temporal_activations
                from ui.layer_viz import LayerVisualizer

                model = TinyLSTM(in_dim=in_d, hidden_dim=hidden_d,
                                 num_layers=n_layers, out_dim=out_d)
                layer_viz = LayerVisualizer()

                layer_sizes = model.layer_sizes

                # Create static LSTM visualization using matplotlib
                network_img = layer_viz.create_lstm_recurrent_graph(
                    layer_sizes, timesteps,
                    title=f"TinyLSTM Network Structure (T={temp_val})"
                )

                # Get temporal activations
                temporal_acts = get_lstm_temporal_activations(
                    model, seq_length=timesteps, n_samples=samples
                )

                # Create temporal heatmap
                temporal_img = layer_viz.create_activation_heatmap(
                    temporal_acts[:timesteps*5, :min(50, temporal_acts.shape[1])].T,
                    [f"t{i}" for i in range(timesteps*5)],
                    title=f"Temporal Activations (T={temp_val})"
                )

                # Hidden state evolution (3D manifold)
                hidden_img = layer_viz.create_3d_activation_manifold(
                    temporal_acts[:min(50, len(temporal_acts)), :min(30, temporal_acts.shape[1])].T,
                    [f"s{i}" for i in range(min(50, len(temporal_acts)))],
                    title=f"Hidden State Manifold (T={temp_val})"
                )

                return network_img, temporal_img, hidden_img

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, None, None

        generate_lstm_btn.click(
            generate_lstm_viz,
            [in_dim, hidden_dim, num_layers, out_dim, num_timesteps, n_samples, top_k_lstm, temperature],
            [lstm_network_static, temporal_viz, hidden_state_viz]
        )

    def _create_tinygpt_interface(self):
        gr.Markdown("### TinyGPT Transformer Visualization")
        gr.Markdown("*Explore attention patterns in a miniature GPT model*")

        with gr.Row():
            with gr.Column(scale=1):
                vocab_size = gr.Slider(minimum=100, maximum=5000, value=1000, step=100,
                                       label="Vocabulary Size")
                embed_dim = gr.Slider(minimum=32, maximum=256, value=64, step=32,
                                      label="Embedding Dimension",
                                      info="Must be divisible by num_heads")
                num_heads = gr.Slider(minimum=1, maximum=8, value=4, step=1,
                                      label="Number of Attention Heads",
                                      info="embed_dim must be divisible by this")
                num_layers = gr.Slider(minimum=1, maximum=4, value=2, step=1,
                                       label="Number of Transformer Blocks")

            with gr.Column(scale=1):
                ff_dim = gr.Slider(minimum=64, maximum=512, value=128, step=64,
                                   label="Feedforward Dimension")
                seq_len = gr.Slider(minimum=4, maximum=32, value=8, step=4,
                                    label="Sequence Length")
                top_k_gpt = gr.Slider(minimum=5, maximum=50, value=20, step=5,
                                      label="Top-K Neurons to Show")
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature",
                    info="Softens/sharpens attention patterns"
                )

        generate_gpt_btn = gr.Button("🤖 Generate TinyGPT & Visualize", variant="primary")

        with gr.Row():
            transformer_static = gr.Image(label="Transformer Attention Structure")
            attention_viz = gr.Image(label="Attention Patterns")

        with gr.Row():
            layer_act_viz = gr.Image(label="Layer Activations (2D Heatmap)")
            layer_manifold_viz = gr.Image(label="Layer Activation Manifold (3D)")

        def generate_tinygpt_viz(vocab, embed, heads, layers, ff, seq, top_k, temp_val):
            try:
                from models.educational_models import TinyGPT
                from ui.layer_viz import LayerVisualizer

                # Validate embed_dim is divisible by num_heads
                if embed % heads != 0:
                    # Adjust embed_dim to be divisible by heads
                    embed = (embed // heads) * heads
                    if embed == 0:
                        embed = heads * 8  # Minimum viable embed_dim

                model = TinyGPT(
                    vocab_size=vocab, embed_dim=embed, num_heads=heads,
                    num_layers=layers, ff_dim=ff, max_seq_len=seq * 2
                )
                layer_viz = LayerVisualizer()

                # Random input sequence
                input_ids = torch.randint(0, vocab, (1, seq))

                # Get attention patterns
                _, attention_patterns = model(input_ids, return_attention=True)

                # Get layer activations
                layer_activations = model.get_layer_activations(input_ids)

                # Create static attention visualization using matplotlib
                # Use attention flow for the first visualization slot
                if attention_patterns:
                    # Average attention across heads and show as heatmap
                    attn_data = attention_patterns[0][0]
                    # Handle both tensor and numpy array
                    if hasattr(attn_data, 'detach'):
                        attn_matrix = attn_data.detach().numpy()
                    else:
                        attn_matrix = np.array(attn_data)

                    # attn_matrix shape depends on model: could be [heads, seq, seq] or [seq, seq]
                    if attn_matrix.ndim == 3:
                        avg_attn = attn_matrix.mean(axis=0)  # [seq, seq]
                    elif attn_matrix.ndim == 2:
                        avg_attn = attn_matrix  # Already [seq, seq]
                    else:
                        avg_attn = attn_matrix.reshape(-1, seq) if attn_matrix.size >= seq else np.zeros((seq, seq))

                    transformer_img = layer_viz.create_activation_heatmap(
                        avg_attn,
                        [f"T{i}" for i in range(avg_attn.shape[1])],
                        title=f"TinyGPT Attention Pattern (T={temp_val}, {heads} heads avg)"
                    )
                else:
                    transformer_img = None

                # Create attention flow visualization (first layer)
                if attention_patterns:
                    attn_img = layer_viz.create_attention_flow_graph(
                        attention_patterns[0][0],  # First layer, first batch
                        [f"T{i}" for i in range(seq)],
                        layer_idx=0,
                        title=f"TinyGPT Attention Flow (T={temp_val})"
                    )
                else:
                    attn_img = None

                # Create layer activation heatmap
                act_matrix = np.stack([a[0].mean(axis=0)[:30] for a in layer_activations])
                layer_img = layer_viz.create_activation_heatmap(
                    act_matrix,
                    [f"D{i}" for i in range(min(30, act_matrix.shape[1]))],
                    title=f"Layer-wise Activations (T={temp_val})"
                )

                # Create 3D manifold for layer activations
                layer_manifold_img = layer_viz.create_3d_activation_manifold(
                    act_matrix,
                    [f"D{i}" for i in range(min(30, act_matrix.shape[1]))],
                    title=f"Layer Activation Manifold (T={temp_val})"
                )

                return transformer_img, attn_img, layer_img, layer_manifold_img

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, None, None, None

        generate_gpt_btn.click(
            generate_tinygpt_viz,
            [vocab_size, embed_dim, num_heads, num_layers, ff_dim, seq_len, top_k_gpt, temperature],
            [transformer_static, attention_viz, layer_act_viz, layer_manifold_viz]
        )

    def _create_gpt2_interface(self):
        """Create GPT-2 analysis interface with interactive 3D visualizations."""
        gr.Markdown("### GPT-2 Transformer Analysis")
        gr.Markdown("*Explore GPT-2's internal representations and attention patterns*")

        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(
                    label="Input Prompt",
                    value="The quick brown fox jumps over the lazy",
                    lines=2
                )
            with gr.Column(scale=1):
                layer_select = gr.Slider(
                    minimum=0, maximum=11, value=6, step=1,
                    label="Focus Layer"
                )
                n_tokens_viz = gr.Slider(
                    minimum=3, maximum=20, value=8, step=1,
                    label="Tokens to Visualize"
                )

        with gr.Row():
            top_k_neurons = gr.Slider(
                minimum=5, maximum=100, value=30, step=5,
                label="Top-K Neurons per Layer"
            )
            animate_mode = gr.Checkbox(
                label="Enable Token Animation",
                value=True
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Temperature",
                info="Affects softmax scaling for attention"
            )

        analyze_gpt2_btn = gr.Button("🔍 Analyze GPT-2", variant="primary")

        with gr.Row():
            gpt2_network_3d = gr.Image(label="3D Network Structure")
            gpt2_attention = gr.Image(label="Attention Patterns")

        with gr.Row():
            gpt2_activations = gr.Image(label="Layer Activations (2D Heatmap)")
            gpt2_activations_3d = gr.Image(label="Activation Manifold (3D)")

        gpt2_token_journey = gr.Image(label="Token Transformation Journey")

        def analyze_gpt2(prompt, layer_idx, n_tokens, top_k, animate, temp_val):
            try:
                import plotly.graph_objects as go
                from ui.layer_viz import LayerVisualizer

                # Get GPT-2 model from controller
                if 'gpt2' not in self.controller.models:
                    return None, None, None, None

                model = self.controller.models['gpt2']
                layer_viz = LayerVisualizer()

                # Tokenize and get activations
                inputs = model.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=50
                )
                tokens = model.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])[:n_tokens]

                # Get hidden states through all layers
                import torch
                with torch.no_grad():
                    outputs = model.lm_model(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        output_hidden_states=True,
                        output_attentions=True
                    )

                hidden_states = outputs.hidden_states  # tuple of [batch, seq, hidden]
                attentions = outputs.attentions  # tuple of [batch, heads, seq, seq]

                # Create static 3D network visualization
                network_img = layer_viz.create_transformer_network_graph(
                    hidden_states, tokens[:n_tokens], top_k,
                    title=f"GPT-2 Network Structure (T={temp_val})"
                )

                # Create attention heatmap
                attn_img = None
                if attentions is not None and len(attentions) > 0 and layer_idx < len(attentions):
                    try:
                        attn = attentions[layer_idx][0].mean(dim=0).cpu().numpy()  # Mean across heads
                        attn_img = layer_viz.create_attention_flow_graph(
                            attn[:n_tokens, :n_tokens],
                            tokens[:n_tokens],
                            layer_idx=layer_idx,
                            title=f"GPT-2 Attention (Layer {layer_idx}, T={temp_val})"
                        )
                    except (TypeError, IndexError) as e:
                        print(f"Warning: Could not extract attention weights: {e}")

                # Create activation heatmap across layers
                act_matrix = []
                for hs in hidden_states[1:]:  # Skip embedding layer
                    layer_act = hs[0, :n_tokens, :].cpu().numpy()
                    act_matrix.append(np.linalg.norm(layer_act, axis=-1))
                act_matrix = np.array(act_matrix)

                act_img = layer_viz.create_activation_heatmap(
                    act_matrix,
                    tokens[:n_tokens],
                    title=f"GPT-2 Activation Norms by Layer (T={temp_val})"
                )

                # Create 3D activation manifold
                act_3d_img = layer_viz.create_3d_activation_manifold(
                    act_matrix,
                    tokens[:n_tokens],
                    title=f"GPT-2 Activation Manifold (T={temp_val})"
                )

                # Create token journey visualization
                journey_img = layer_viz.create_token_journey(
                    hidden_states, tokens[:n_tokens], n_tokens
                )

                return network_img, attn_img, act_img, act_3d_img, journey_img

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, None, None, None, None

        analyze_gpt2_btn.click(
            analyze_gpt2,
            [prompt_input, layer_select, n_tokens_viz, top_k_neurons, animate_mode, temperature],
            [gpt2_network_3d, gpt2_attention, gpt2_activations, gpt2_activations_3d, gpt2_token_journey]
        )

    def _create_gpt2_3d_network(self, hidden_states, tokens, top_k, animate=True):
        """Create interactive 3D Plotly visualization of GPT-2 network."""
        import plotly.graph_objects as go
        import numpy as np

        n_layers = len(hidden_states) - 1  # Exclude embedding
        n_tokens = len(tokens)
        hidden_dim = hidden_states[0].shape[-1]

        # Build 3D node positions: x=layer, y=token, z=neuron_cluster
        nodes_x, nodes_y, nodes_z = [], [], []
        node_colors, node_labels = [], []

        # Sample top-k neurons per layer for visualization
        for layer_idx in range(n_layers):
            hs = hidden_states[layer_idx + 1][0].cpu().numpy()  # [seq, hidden]

            for tok_idx in range(n_tokens):
                # Get top activated neurons
                activations = np.abs(hs[tok_idx])
                top_neurons = np.argsort(activations)[-top_k:]

                for n_idx, neuron in enumerate(top_neurons):
                    nodes_x.append(layer_idx * 2)
                    nodes_y.append(tok_idx * 1.5)
                    nodes_z.append(n_idx * 0.3)
                    node_colors.append(activations[neuron])
                    node_labels.append(f"L{layer_idx} T{tok_idx} N{neuron}: {activations[neuron]:.3f}")

        # Normalize colors
        if node_colors:
            max_color = max(node_colors)
            node_colors = [c / max_color if max_color > 0 else 0 for c in node_colors]

        # Create figure
        fig = go.Figure()

        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=nodes_x, y=nodes_y, z=nodes_z,
            mode='markers',
            marker=dict(
                size=4,
                color=node_colors,
                colorscale='Hot',
                colorbar=dict(title='Activation'),
                opacity=0.8
            ),
            text=node_labels,
            hoverinfo='text',
            name='Neurons'
        ))

        # Add layer labels
        for layer_idx in range(n_layers):
            fig.add_trace(go.Scatter3d(
                x=[layer_idx * 2], y=[-1], z=[0],
                mode='text',
                text=[f'Layer {layer_idx}'],
                textposition='bottom center',
                textfont=dict(size=10, color='white'),
                showlegend=False
            ))

        # Add token labels
        for tok_idx, tok in enumerate(tokens):
            fig.add_trace(go.Scatter3d(
                x=[-0.5], y=[tok_idx * 1.5], z=[0],
                mode='text',
                text=[tok[:10]],  # Truncate long tokens
                textposition='middle left',
                textfont=dict(size=9, color='cyan'),
                showlegend=False
            ))

        fig.update_layout(
            title='GPT-2 Network (Drag to Rotate)',
            scene=dict(
                xaxis=dict(title='Layer', showgrid=False, zeroline=False),
                yaxis=dict(title='Token', showgrid=False, zeroline=False),
                zaxis=dict(title='Neuron Cluster', showgrid=False, zeroline=False),
                bgcolor='#1a1a2e'
            ),
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#1a1a2e',
            font=dict(color='white'),
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False
        )

        return fig

    def _create_bert_interface(self):
        """Create BERT analysis interface with interactive 3D visualizations."""
        gr.Markdown("### BERT Encoder Analysis")
        gr.Markdown("*Explore BERT's bidirectional attention and masked language modeling*")

        with gr.Row():
            with gr.Column(scale=2):
                bert_prompt = gr.Textbox(
                    label="Input Text",
                    value="The [MASK] sat on the mat and watched the [MASK].",
                    lines=2
                )
            with gr.Column(scale=1):
                bert_layer = gr.Slider(
                    minimum=0, maximum=11, value=6, step=1,
                    label="Focus Layer"
                )
                bert_tokens_viz = gr.Slider(
                    minimum=3, maximum=20, value=10, step=1,
                    label="Tokens to Visualize"
                )

        with gr.Row():
            bert_top_k = gr.Slider(
                minimum=5, maximum=100, value=30, step=5,
                label="Top-K Neurons"
            )
            show_mask_pred = gr.Checkbox(
                label="Show [MASK] Predictions",
                value=True
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Temperature",
                info="Affects softmax scaling for predictions"
            )

        analyze_bert_btn = gr.Button("🔍 Analyze BERT", variant="primary")

        with gr.Row():
            bert_network_3d = gr.Image(label="3D Network Structure")
            bert_attention = gr.Image(label="Bidirectional Attention")

        with gr.Row():
            bert_activations = gr.Image(label="Layer Activations (2D Heatmap)")
            bert_activations_3d = gr.Image(label="Activation Manifold (3D)")

        bert_predictions = gr.Dataframe(
            label="[MASK] Token Predictions",
            headers=["Position", "Top-1", "Top-2", "Top-3", "Top-4", "Top-5"],
            row_count=5
        )

        def analyze_bert(prompt, layer_idx, n_tokens, top_k, show_masks, temp_val):
            try:
                import plotly.graph_objects as go
                from ui.layer_viz import LayerVisualizer

                # Get BERT model from controller
                if 'bert' not in self.controller.models:
                    return None, None, None, []

                model = self.controller.models['bert']
                layer_viz = LayerVisualizer()

                # Tokenize
                inputs = model.tokenizer(
                    prompt, return_tensors="pt", truncation=True, max_length=50
                )
                tokens = model.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])[:n_tokens]

                # Get hidden states and attentions
                import torch
                with torch.no_grad():
                    outputs = model.lm_model(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        output_hidden_states=True,
                        output_attentions=True
                    )

                hidden_states = outputs.hidden_states
                attentions = outputs.attentions

                # Create static 3D network visualization
                network_img = layer_viz.create_transformer_network_graph(
                    hidden_states, tokens[:n_tokens], top_k,
                    title=f"BERT Network Structure (T={temp_val})"
                )

                # Create bidirectional attention visualization
                if attentions and layer_idx < len(attentions):
                    attn = attentions[layer_idx][0].mean(dim=0).cpu().numpy()
                    attn_img = layer_viz.create_attention_flow_graph(
                        attn[:n_tokens, :n_tokens],
                        tokens[:n_tokens],
                        layer_idx=layer_idx,
                        title=f"BERT Bidirectional Attention (Layer {layer_idx}, T={temp_val})"
                    )
                else:
                    attn_img = None

                # Create activation heatmap
                act_matrix = []
                for hs in hidden_states[1:]:
                    layer_act = hs[0, :n_tokens, :].cpu().numpy()
                    act_matrix.append(np.linalg.norm(layer_act, axis=-1))
                act_matrix = np.array(act_matrix)

                act_img = layer_viz.create_activation_heatmap(
                    act_matrix,
                    tokens[:n_tokens],
                    title=f"BERT Activation Norms by Layer (T={temp_val})"
                )

                # Create 3D activation manifold
                act_3d_img = layer_viz.create_3d_activation_manifold(
                    act_matrix,
                    tokens[:n_tokens],
                    title=f"BERT Activation Manifold (T={temp_val})"
                )

                # Get [MASK] predictions (temperature affects softmax)
                predictions = []
                if show_masks and '[MASK]' in tokens:
                    mask_positions = [i for i, t in enumerate(tokens) if t == '[MASK]']

                    with torch.no_grad():
                        logits = model.lm_model(
                            inputs.input_ids,
                            attention_mask=inputs.attention_mask
                        ).logits

                    for pos in mask_positions[:5]:  # Limit to 5 masks
                        if pos < logits.shape[1]:
                            # Apply temperature to softmax
                            scaled_logits = logits[0, pos] / temp_val
                            probs = torch.softmax(scaled_logits, dim=-1)
                            top5 = torch.topk(probs, 5)
                            top5_tokens = [model.tokenizer.decode([idx]) for idx in top5.indices]
                            predictions.append([f"Position {pos}"] + top5_tokens)

                return network_img, attn_img, act_img, act_3d_img, predictions

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, None, None, None, []

        analyze_bert_btn.click(
            analyze_bert,
            [bert_prompt, bert_layer, bert_tokens_viz, bert_top_k, show_mask_pred, temperature],
            [bert_network_3d, bert_attention, bert_activations, bert_activations_3d, bert_predictions]
        )

    def _create_bert_3d_network(self, hidden_states, tokens, top_k):
        """Create interactive 3D Plotly visualization of BERT network."""
        import plotly.graph_objects as go
        import numpy as np

        n_layers = len(hidden_states) - 1
        n_tokens = len(tokens)

        nodes_x, nodes_y, nodes_z = [], [], []
        node_colors, node_labels = [], []

        for layer_idx in range(n_layers):
            hs = hidden_states[layer_idx + 1][0].cpu().numpy()

            for tok_idx in range(n_tokens):
                activations = np.abs(hs[tok_idx])
                top_neurons = np.argsort(activations)[-top_k:]

                for n_idx, neuron in enumerate(top_neurons):
                    nodes_x.append(layer_idx * 2)
                    nodes_y.append(tok_idx * 1.5)
                    nodes_z.append(n_idx * 0.3)
                    node_colors.append(activations[neuron])
                    node_labels.append(f"L{layer_idx} T{tok_idx} N{neuron}: {activations[neuron]:.3f}")

        if node_colors:
            max_color = max(node_colors)
            node_colors = [c / max_color if max_color > 0 else 0 for c in node_colors]

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=nodes_x, y=nodes_y, z=nodes_z,
            mode='markers',
            marker=dict(
                size=4,
                color=node_colors,
                colorscale='Viridis',
                colorbar=dict(title='Activation'),
                opacity=0.8
            ),
            text=node_labels,
            hoverinfo='text',
            name='Neurons'
        ))

        # Add layer labels
        for layer_idx in range(n_layers):
            fig.add_trace(go.Scatter3d(
                x=[layer_idx * 2], y=[-1], z=[0],
                mode='text',
                text=[f'Layer {layer_idx}'],
                textposition='bottom center',
                textfont=dict(size=10, color='white'),
                showlegend=False
            ))

        # Add token labels
        for tok_idx, tok in enumerate(tokens):
            # Highlight [MASK] tokens
            color = 'yellow' if tok == '[MASK]' else 'cyan'
            fig.add_trace(go.Scatter3d(
                x=[-0.5], y=[tok_idx * 1.5], z=[0],
                mode='text',
                text=[tok[:10]],
                textposition='middle left',
                textfont=dict(size=9, color=color),
                showlegend=False
            ))

        fig.update_layout(
            title='BERT Network (Drag to Rotate)',
            scene=dict(
                xaxis=dict(title='Layer', showgrid=False, zeroline=False),
                yaxis=dict(title='Token', showgrid=False, zeroline=False),
                zaxis=dict(title='Neuron Cluster', showgrid=False, zeroline=False),
                bgcolor='#1a1a2e'
            ),
            paper_bgcolor='#1a1a2e',
            plot_bgcolor='#1a1a2e',
            font=dict(color='white'),
            margin=dict(l=0, r=0, t=40, b=0),
            showlegend=False
        )

        return fig

    def _create_training_interface(self):
        gr.Markdown("### Model Training Lab")
        gr.Markdown("*Watch a model learn in real-time*")

        with gr.Row():
            model_type = gr.Dropdown(
                choices=["TinyMLP", "TinyLSTM"],
                value="TinyMLP",
                label="Model Type"
            )
            task = gr.Dropdown(
                choices=["XOR", "Sine Wave", "Classification"],
                value="XOR",
                label="Training Task"
            )

        with gr.Row():
            lr = gr.Slider(minimum=0.001, maximum=0.5, value=0.01, step=0.001,
                           label="Learning Rate")
            epochs = gr.Slider(minimum=10, maximum=500, value=100, step=10,
                               label="Training Epochs")
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Temperature",
                info="Scales noise in training data"
            )

        train_btn = gr.Button("🏋️ Train Model", variant="primary")

        with gr.Row():
            loss_plot = gr.Image(label="Training Loss Curve")
            weight_viz = gr.Image(label="Weight Evolution (2D Heatmaps)")

        with gr.Row():
            weight_manifold_grid = gr.Image(label="Weight Evolution (3D Manifolds per Epoch)")

        with gr.Row():
            weight_manifold = gr.Image(label="Weight Evolution Manifold (3D Timeline)")
            gradient_flow = gr.Image(label="Gradient Flow (3D)")

        final_accuracy = gr.Textbox(label="Final Performance")

        def train_model(m_type, task_name, learning_rate, num_epochs, temp_val):
            try:
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                from matplotlib.colors import LinearSegmentedColormap
                from io import BytesIO
                from PIL import Image

                if m_type == "TinyMLP":
                    from models.educational_models import TinyMLP
                    model = TinyMLP(in_dim=2, h1=8, h2=8, out_dim=1)
                else:
                    from models.educational_models import TinyLSTM
                    model = TinyLSTM(in_dim=2, hidden_dim=8, num_layers=1, out_dim=1)

                # Create training data based on task (temperature adds noise)
                if task_name == "XOR":
                    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
                    X = X + torch.randn_like(X) * (temp_val - 1.0) * 0.1  # Add temperature-scaled noise
                    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
                elif task_name == "Sine Wave":
                    X = torch.linspace(0, 4*np.pi, 100).unsqueeze(1)
                    X = torch.cat([X, torch.sin(X)], dim=1)
                    X = X + torch.randn_like(X) * (temp_val - 1.0) * 0.1
                    y = torch.sin(X[:, :1] + 0.5)
                else:  # Classification
                    n = 50
                    X = torch.randn(n * 2, 2)
                    X[:n] += 2
                    X[n:] -= 2
                    X = X + torch.randn_like(X) * (temp_val - 1.0) * 0.1
                    y = torch.cat([torch.ones(n, 1), torch.zeros(n, 1)])

                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                criterion = torch.nn.MSELoss()

                losses = []
                weight_snapshots = []
                gradient_snapshots = []

                for epoch in range(int(num_epochs)):
                    if m_type == "TinyLSTM":
                        X_input = X.unsqueeze(1)  # Add sequence dimension
                    else:
                        X_input = X

                    output = model(X_input)
                    if output.dim() > 1 and output.size(-1) != y.size(-1):
                        output = output[:, :y.size(-1)]

                    loss = criterion(output, y)
                    optimizer.zero_grad()
                    loss.backward()

                    # Capture gradients before optimizer step
                    if epoch % (num_epochs // 10) == 0:
                        if hasattr(model, 'fc1'):
                            weight_snapshots.append(model.fc1.weight.detach().clone().numpy())
                            if model.fc1.weight.grad is not None:
                                gradient_snapshots.append(model.fc1.weight.grad.detach().clone().numpy())
                        elif hasattr(model, 'fc'):
                            weight_snapshots.append(model.fc.weight.detach().clone().numpy())
                            if model.fc.weight.grad is not None:
                                gradient_snapshots.append(model.fc.weight.grad.detach().clone().numpy())

                    optimizer.step()
                    losses.append(loss.item())

                # Create loss plot
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                fig1.patch.set_facecolor('#1a1a2e')
                ax1.set_facecolor('#1a1a2e')
                ax1.plot(losses, color='#00ffff', linewidth=2)
                ax1.set_xlabel('Epoch', color='white')
                ax1.set_ylabel('Loss', color='white')
                ax1.set_title(f'Training Loss (T={temp_val})', color='white')
                ax1.tick_params(colors='white')
                ax1.grid(True, color='#4a4a5e', alpha=0.5)

                buf1 = BytesIO()
                fig1.savefig(buf1, format='png', facecolor='#1a1a2e', bbox_inches='tight')
                plt.close(fig1)
                buf1.seek(0)
                loss_img = Image.open(buf1)
                loss_img.load()
                loss_img = loss_img.copy()

                # Create weight evolution heatmap visualization
                weight_img = None
                if weight_snapshots:
                    fig2, axes = plt.subplots(2, 5, figsize=(15, 6))
                    fig2.patch.set_facecolor('#1a1a2e')
                    for i, (ax, w) in enumerate(zip(axes.flat, weight_snapshots)):
                        ax.set_facecolor('#1a1a2e')
                        im = ax.imshow(w[:min(8, w.shape[0]), :min(8, w.shape[1])],
                                       cmap='viridis', aspect='auto')
                        ax.set_title(f'Epoch {i * (num_epochs // 10)}', color='white', fontsize=9)
                        ax.tick_params(colors='white', labelsize=6)
                    plt.tight_layout()

                    buf2 = BytesIO()
                    fig2.savefig(buf2, format='png', facecolor='#1a1a2e', bbox_inches='tight')
                    plt.close(fig2)
                    buf2.seek(0)
                    weight_img = Image.open(buf2)
                    weight_img.load()
                    weight_img = weight_img.copy()

                # Create per-epoch 3D manifold grid (same data as heatmaps but as 3D surfaces)
                manifold_grid_img = None
                if weight_snapshots:
                    colors = ['#1a1a2e', '#2d3561', '#7b68ee', '#ffa500', '#ff4500']
                    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

                    n_snapshots = min(10, len(weight_snapshots))
                    fig_grid = plt.figure(figsize=(20, 8))
                    fig_grid.patch.set_facecolor('#1a1a2e')

                    for idx in range(n_snapshots):
                        ax = fig_grid.add_subplot(2, 5, idx + 1, projection='3d')
                        ax.set_facecolor('#1a1a2e')

                        w = weight_snapshots[idx]
                        n_rows, n_cols = min(8, w.shape[0]), min(8, w.shape[1])
                        w_data = w[:n_rows, :n_cols]

                        # Create meshgrid for this weight matrix
                        X_mesh, Y_mesh = np.meshgrid(range(n_cols), range(n_rows))

                        # Plot the weight matrix as a 3D surface
                        surf = ax.plot_surface(X_mesh, Y_mesh, w_data, cmap=cmap, alpha=0.9,
                                              linewidth=0.3, antialiased=True, edgecolor='white')

                        ax.set_title(f'Epoch {idx * (num_epochs // 10)}', color='white', fontsize=9)
                        ax.tick_params(colors='white', labelsize=5)
                        ax.xaxis.pane.set_facecolor('#1a1a2e')
                        ax.yaxis.pane.set_facecolor('#1a1a2e')
                        ax.zaxis.pane.set_facecolor('#1a1a2e')
                        ax.view_init(elev=30, azim=45)

                        # Set consistent z-limits across all subplots
                        all_weights = np.concatenate([ws.flatten() for ws in weight_snapshots])
                        z_min, z_max = all_weights.min(), all_weights.max()
                        ax.set_zlim(z_min, z_max)

                    fig_grid.suptitle(f'Weight Matrices as 3D Surfaces (T={temp_val})', color='white', fontsize=14)
                    plt.tight_layout()

                    buf_grid = BytesIO()
                    fig_grid.savefig(buf_grid, format='png', facecolor='#1a1a2e', bbox_inches='tight')
                    plt.close(fig_grid)
                    buf_grid.seek(0)
                    manifold_grid_img = Image.open(buf_grid)
                    manifold_grid_img.load()
                    manifold_grid_img = manifold_grid_img.copy()

                # Create 3D weight evolution manifold (timeline view)
                manifold_img = None
                if weight_snapshots:
                    colors = ['#1a1a2e', '#2d3561', '#7b68ee', '#ffa500', '#ff4500']
                    cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

                    fig3 = plt.figure(figsize=(12, 8))
                    ax3 = fig3.add_subplot(111, projection='3d')
                    fig3.patch.set_facecolor('#1a1a2e')
                    ax3.set_facecolor('#1a1a2e')

                    # Stack weights across epochs for 3D surface
                    n_snapshots = len(weight_snapshots)
                    w_sample = weight_snapshots[0]
                    n_rows, n_cols = min(8, w_sample.shape[0]), min(8, w_sample.shape[1])

                    # Create meshgrid for surface
                    X_mesh, Y_mesh = np.meshgrid(range(n_cols), range(n_snapshots))

                    # Average weights per epoch for visualization
                    Z_data = np.array([w[:n_rows, :n_cols].mean(axis=0) for w in weight_snapshots])

                    surf = ax3.plot_surface(X_mesh, Y_mesh, Z_data, cmap=cmap, alpha=0.85,
                                           linewidth=0.5, antialiased=True, edgecolor='white')

                    ax3.set_xlabel('Weight Index', color='white', fontsize=10)
                    ax3.set_ylabel('Training Epoch', color='white', fontsize=10)
                    ax3.set_zlabel('Weight Value', color='white', fontsize=10)
                    ax3.set_title(f'Weight Evolution Manifold (T={temp_val})', color='white', fontsize=12)

                    ax3.tick_params(colors='white')
                    ax3.xaxis.pane.set_facecolor('black')
                    ax3.yaxis.pane.set_facecolor('black')
                    ax3.zaxis.pane.set_facecolor('black')
                    ax3.view_init(elev=25, azim=45)

                    cbar = fig3.colorbar(surf, ax=ax3, shrink=0.5, aspect=10, pad=0.1)
                    cbar.set_label('Weight Value', color='white')
                    cbar.ax.yaxis.set_tick_params(color='white')
                    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

                    buf3 = BytesIO()
                    fig3.savefig(buf3, format='png', facecolor='#1a1a2e', bbox_inches='tight')
                    plt.close(fig3)
                    buf3.seek(0)
                    manifold_img = Image.open(buf3)
                    manifold_img.load()
                    manifold_img = manifold_img.copy()

                # Create 3D gradient flow visualization
                gradient_img = None
                if gradient_snapshots:
                    fig4 = plt.figure(figsize=(12, 8))
                    ax4 = fig4.add_subplot(111, projection='3d')
                    fig4.patch.set_facecolor('#1a1a2e')
                    ax4.set_facecolor('#1a1a2e')

                    n_snapshots = len(gradient_snapshots)
                    g_sample = gradient_snapshots[0]
                    n_rows, n_cols = min(8, g_sample.shape[0]), min(8, g_sample.shape[1])

                    # Create meshgrid for gradient surface
                    X_mesh, Y_mesh = np.meshgrid(range(n_cols), range(n_snapshots))

                    # Average gradients per epoch
                    Z_grad = np.array([np.abs(g[:n_rows, :n_cols]).mean(axis=0) for g in gradient_snapshots])

                    # Use different colormap for gradients
                    grad_colors = ['#1a1a2e', '#4a90d9', '#00ffff', '#00ff88', '#ffff00']
                    grad_cmap = LinearSegmentedColormap.from_list('grad', grad_colors, N=256)

                    surf = ax4.plot_surface(X_mesh, Y_mesh, Z_grad, cmap=grad_cmap, alpha=0.85,
                                           linewidth=0.5, antialiased=True, edgecolor='white')

                    ax4.set_xlabel('Weight Index', color='white', fontsize=10)
                    ax4.set_ylabel('Training Epoch', color='white', fontsize=10)
                    ax4.set_zlabel('Gradient Magnitude', color='white', fontsize=10)
                    ax4.set_title(f'Gradient Flow (T={temp_val})', color='white', fontsize=12)

                    ax4.tick_params(colors='white')
                    ax4.xaxis.pane.set_facecolor('black')
                    ax4.yaxis.pane.set_facecolor('black')
                    ax4.zaxis.pane.set_facecolor('black')
                    ax4.view_init(elev=25, azim=45)

                    cbar = fig4.colorbar(surf, ax=ax4, shrink=0.5, aspect=10, pad=0.1)
                    cbar.set_label('Gradient Magnitude', color='white')
                    cbar.ax.yaxis.set_tick_params(color='white')
                    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

                    buf4 = BytesIO()
                    fig4.savefig(buf4, format='png', facecolor='#1a1a2e', bbox_inches='tight')
                    plt.close(fig4)
                    buf4.seek(0)
                    gradient_img = Image.open(buf4)
                    gradient_img.load()
                    gradient_img = gradient_img.copy()

                final_loss = losses[-1] if losses else 0
                return loss_img, weight_img, manifold_grid_img, manifold_img, gradient_img, f"Final Loss: {final_loss:.6f} (T={temp_val})"

            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, None, None, None, None, f"Error: {str(e)}"

        train_btn.click(
            train_model,
            [model_type, task, lr, epochs, temperature],
            [loss_plot, weight_viz, weight_manifold_grid, weight_manifold, gradient_flow, final_accuracy]
        )

    def _create_mlp_plotly_3d(self, layer_sizes, weight_matrices, activations, top_k):
        """Create interactive Plotly 3D visualization of MLP network."""
        import plotly.graph_objects as go
        import numpy as np

        num_layers = len(layer_sizes)

        # Build node positions
        nodes_x, nodes_y, nodes_z = [], [], []
        node_colors, node_labels = [], []

        for layer_idx, size in enumerate(layer_sizes):
            display_size = min(size, 30)
            for neuron_idx in range(display_size):
                nodes_x.append(layer_idx * 2.0)
                nodes_y.append(neuron_idx - display_size / 2)
                nodes_z.append(0)

                if layer_idx < len(activations):
                    act = activations[layer_idx]
                    if neuron_idx < len(act.flatten()):
                        val = np.abs(act.flatten()[neuron_idx])
                        node_colors.append(val)
                    else:
                        node_colors.append(0)
                else:
                    node_colors.append(0)

                node_labels.append(f"L{layer_idx} N{neuron_idx}")

        # Normalize colors
        if node_colors:
            max_c = max(node_colors) if max(node_colors) > 0 else 1
            node_colors = [c / max_c for c in node_colors]

        # Build edges
        edges_x, edges_y, edges_z = [], [], []

        for layer_idx in range(num_layers - 1):
            src_size = min(layer_sizes[layer_idx], 30)
            dst_size = min(layer_sizes[layer_idx + 1], 30)

            if layer_idx < len(weight_matrices):
                weights = weight_matrices[layer_idx]
                for dst_idx in range(dst_size):
                    if dst_idx < weights.shape[0]:
                        row = np.abs(weights[dst_idx, :min(src_size, weights.shape[1])])
                        top_indices = np.argsort(row)[-top_k:]

                        for src_idx in top_indices:
                            if src_idx < src_size:
                                edges_x.extend([layer_idx * 2.0, (layer_idx + 1) * 2.0, None])
                                edges_y.extend([src_idx - src_size / 2, dst_idx - dst_size / 2, None])
                                edges_z.extend([0, 0, None])

        # Create figure
        fig = go.Figure()

        # Add edges
        if edges_x:
            fig.add_trace(go.Scatter3d(
                x=edges_x, y=edges_y, z=edges_z,
                mode='lines',
                line=dict(width=1.5, color='rgba(0, 255, 255, 0.3)'),
                hoverinfo='none',
                name='Weights'
            ))

        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=nodes_x, y=nodes_y, z=nodes_z,
            mode='markers',
            marker=dict(
                size=10,
                color=node_colors,
                colorscale='Plasma',
                colorbar=dict(
                    title=dict(text='Activation', font=dict(color='white')),
                    tickfont=dict(color='white')
                ),
                opacity=0.95,
                line=dict(width=1, color='white')
            ),
            text=node_labels,
            hoverinfo='text',
            name='Neurons'
        ))

        # Add layer labels
        for layer_idx in range(num_layers):
            fig.add_trace(go.Scatter3d(
                x=[layer_idx * 2], y=[-layer_sizes[layer_idx] / 2 - 3], z=[0],
                mode='text',
                text=[f'Layer {layer_idx}'],
                textposition='bottom center',
                textfont=dict(size=12, color='cyan'),
                showlegend=False
            ))

        fig.update_layout(
            title=dict(text='TinyMLP Network (Drag to Rotate)', font=dict(color='white', size=16)),
            scene=dict(
                xaxis=dict(title=dict(text='Layer', font=dict(color='cyan')), tickfont=dict(color='white'),
                          showgrid=True, gridcolor='rgba(100,100,100,0.3)', zeroline=False),
                yaxis=dict(title=dict(text='Neuron', font=dict(color='cyan')), tickfont=dict(color='white'),
                          showgrid=True, gridcolor='rgba(100,100,100,0.3)', zeroline=False),
                zaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False),
                bgcolor='#0d0d1a'
            ),
            paper_bgcolor='#0d0d1a',
            plot_bgcolor='#0d0d1a',
            font=dict(color='white'),
            margin=dict(l=0, r=0, t=50, b=0),
            showlegend=False
        )

        return fig

    def _create_lstm_plotly_3d(self, layer_sizes, num_timesteps, top_k):
        """Create interactive Plotly 3D visualization of LSTM network unrolled over time."""
        import plotly.graph_objects as go
        import numpy as np

        num_layers = len(layer_sizes)

        # Build node positions: x=layer, y=neuron, z=timestep
        nodes_x, nodes_y, nodes_z = [], [], []
        node_colors, node_labels = [], []

        for t in range(num_timesteps):
            for layer_idx, size in enumerate(layer_sizes):
                display_size = min(size, 20)
                for neuron_idx in range(display_size):
                    nodes_x.append(layer_idx * 2.0)
                    nodes_y.append(neuron_idx - display_size / 2)
                    nodes_z.append(t * 2.0)

                    # Color based on position in time (earlier = cooler, later = warmer)
                    node_colors.append(t / max(1, num_timesteps - 1))
                    node_labels.append(f"T{t} L{layer_idx} N{neuron_idx}")

        # Build feedforward edges within timesteps
        edges_x, edges_y, edges_z = [], [], []

        for t in range(num_timesteps):
            for layer_idx in range(num_layers - 1):
                src_size = min(layer_sizes[layer_idx], 20)
                dst_size = min(layer_sizes[layer_idx + 1], 20)

                # Connect sparse subset
                for dst_idx in range(0, dst_size, max(1, dst_size // top_k)):
                    for src_idx in range(0, src_size, max(1, src_size // top_k)):
                        edges_x.extend([layer_idx * 2.0, (layer_idx + 1) * 2.0, None])
                        edges_y.extend([src_idx - src_size / 2, dst_idx - dst_size / 2, None])
                        edges_z.extend([t * 2.0, t * 2.0, None])

        # Build recurrent edges across timesteps (for hidden layers)
        recurrent_x, recurrent_y, recurrent_z = [], [], []

        for t in range(num_timesteps - 1):
            for layer_idx in range(1, num_layers - 1):  # Hidden layers
                size = min(layer_sizes[layer_idx], 20)
                for neuron_idx in range(0, size, max(1, size // 5)):
                    recurrent_x.extend([layer_idx * 2.0, layer_idx * 2.0, None])
                    recurrent_y.extend([neuron_idx - size / 2, neuron_idx - size / 2, None])
                    recurrent_z.extend([t * 2.0, (t + 1) * 2.0, None])

        # Create figure
        fig = go.Figure()

        # Add feedforward edges
        if edges_x:
            fig.add_trace(go.Scatter3d(
                x=edges_x, y=edges_y, z=edges_z,
                mode='lines',
                line=dict(width=1, color='rgba(0, 255, 255, 0.25)'),
                hoverinfo='none',
                name='Feedforward'
            ))

        # Add recurrent edges
        if recurrent_x:
            fig.add_trace(go.Scatter3d(
                x=recurrent_x, y=recurrent_y, z=recurrent_z,
                mode='lines',
                line=dict(width=2, color='rgba(255, 107, 53, 0.6)'),
                hoverinfo='none',
                name='Recurrent'
            ))

        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=nodes_x, y=nodes_y, z=nodes_z,
            mode='markers',
            marker=dict(
                size=8,
                color=node_colors,
                colorscale='Viridis',
                colorbar=dict(
                    title=dict(text='Timestep', font=dict(color='white')),
                    tickfont=dict(color='white')
                ),
                opacity=0.9,
                line=dict(width=1, color='white')
            ),
            text=node_labels,
            hoverinfo='text',
            name='Neurons'
        ))

        # Add timestep labels
        for t in range(num_timesteps):
            fig.add_trace(go.Scatter3d(
                x=[-1], y=[0], z=[t * 2.0],
                mode='text',
                text=[f't={t}'],
                textposition='middle left',
                textfont=dict(size=11, color='#ff6b35'),
                showlegend=False
            ))

        fig.update_layout(
            title=dict(text='LSTM Network (Drag to Rotate)', font=dict(color='white', size=16)),
            scene=dict(
                xaxis=dict(title=dict(text='Layer', font=dict(color='cyan')), tickfont=dict(color='white'),
                          showgrid=True, gridcolor='rgba(100,100,100,0.3)', zeroline=False),
                yaxis=dict(title=dict(text='Neuron', font=dict(color='cyan')), tickfont=dict(color='white'),
                          showgrid=True, gridcolor='rgba(100,100,100,0.3)', zeroline=False),
                zaxis=dict(title=dict(text='Time', font=dict(color='#ff6b35')), tickfont=dict(color='white'),
                          showgrid=True, gridcolor='rgba(100,100,100,0.3)', zeroline=False),
                bgcolor='#0d0d1a'
            ),
            paper_bgcolor='#0d0d1a',
            plot_bgcolor='#0d0d1a',
            font=dict(color='white'),
            margin=dict(l=0, r=0, t=50, b=0),
            showlegend=True,
            legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.5)')
        )

        return fig

    def _create_transformer_plotly_3d(self, num_layers, num_heads, embed_dim, seq_len, layer_activations, top_k):
        """Create interactive Plotly 3D visualization of transformer architecture."""
        import plotly.graph_objects as go
        import numpy as np

        # Build node positions showing transformer blocks
        # x = layer/stage, y = head/position, z = dimension cluster
        nodes_x, nodes_y, nodes_z = [], [], []
        node_colors, node_labels, node_sizes = [], [], []

        # For each layer, show: Input -> Q/K/V -> Attention -> FFN -> Output
        stages_per_layer = 4  # Input, Attention, FFN, Output

        for layer_idx in range(num_layers):
            base_x = layer_idx * 5

            # Get activations for this layer if available
            if layer_idx < len(layer_activations):
                acts = layer_activations[layer_idx][0].detach().numpy()  # [seq, embed]
                act_norms = np.linalg.norm(acts, axis=-1)
                act_norms = act_norms / (act_norms.max() + 1e-8)
            else:
                act_norms = np.ones(seq_len) * 0.5

            # Input/embedding stage
            for pos in range(seq_len):
                nodes_x.append(base_x)
                nodes_y.append(pos - seq_len / 2)
                nodes_z.append(0)
                node_colors.append(act_norms[pos] if pos < len(act_norms) else 0.5)
                node_labels.append(f"L{layer_idx} Input Pos{pos}")
                node_sizes.append(8)

            # Attention heads
            for head in range(num_heads):
                nodes_x.append(base_x + 1.5)
                nodes_y.append(head - num_heads / 2)
                nodes_z.append(1)
                node_colors.append(0.7)
                node_labels.append(f"L{layer_idx} Head{head}")
                node_sizes.append(12)

            # FFN layer (show top-k neurons)
            ff_neurons = min(top_k, 30)
            for n in range(ff_neurons):
                nodes_x.append(base_x + 3)
                nodes_y.append(n - ff_neurons / 2)
                nodes_z.append(0)
                node_colors.append(0.5 + np.random.rand() * 0.3)
                node_labels.append(f"L{layer_idx} FFN N{n}")
                node_sizes.append(6)

            # Output
            for pos in range(seq_len):
                nodes_x.append(base_x + 4)
                nodes_y.append(pos - seq_len / 2)
                nodes_z.append(0)
                node_colors.append(act_norms[pos] if pos < len(act_norms) else 0.5)
                node_labels.append(f"L{layer_idx} Output Pos{pos}")
                node_sizes.append(8)

        # Build edges
        edges_x, edges_y, edges_z = [], [], []

        for layer_idx in range(num_layers):
            base_x = layer_idx * 5

            # Input to attention heads (sparse)
            for pos in range(0, seq_len, max(1, seq_len // 4)):
                for head in range(num_heads):
                    edges_x.extend([base_x, base_x + 1.5, None])
                    edges_y.extend([pos - seq_len / 2, head - num_heads / 2, None])
                    edges_z.extend([0, 1, None])

            # Attention to FFN
            ff_neurons = min(top_k, 30)
            for head in range(num_heads):
                for n in range(0, ff_neurons, max(1, ff_neurons // 3)):
                    edges_x.extend([base_x + 1.5, base_x + 3, None])
                    edges_y.extend([head - num_heads / 2, n - ff_neurons / 2, None])
                    edges_z.extend([1, 0, None])

            # FFN to output
            for n in range(0, ff_neurons, max(1, ff_neurons // 4)):
                for pos in range(0, seq_len, max(1, seq_len // 3)):
                    edges_x.extend([base_x + 3, base_x + 4, None])
                    edges_y.extend([n - ff_neurons / 2, pos - seq_len / 2, None])
                    edges_z.extend([0, 0, None])

        # Cross-layer connections
        cross_x, cross_y, cross_z = [], [], []
        for layer_idx in range(num_layers - 1):
            base_x = layer_idx * 5
            next_x = (layer_idx + 1) * 5
            for pos in range(0, seq_len, max(1, seq_len // 3)):
                cross_x.extend([base_x + 4, next_x, None])
                cross_y.extend([pos - seq_len / 2, pos - seq_len / 2, None])
                cross_z.extend([0, 0, None])

        # Create figure
        fig = go.Figure()

        # Add internal edges
        if edges_x:
            fig.add_trace(go.Scatter3d(
                x=edges_x, y=edges_y, z=edges_z,
                mode='lines',
                line=dict(width=1, color='rgba(0, 255, 255, 0.2)'),
                hoverinfo='none',
                name='Internal'
            ))

        # Add cross-layer edges
        if cross_x:
            fig.add_trace(go.Scatter3d(
                x=cross_x, y=cross_y, z=cross_z,
                mode='lines',
                line=dict(width=2, color='rgba(255, 215, 0, 0.4)'),
                hoverinfo='none',
                name='Residual'
            ))

        # Add nodes
        fig.add_trace(go.Scatter3d(
            x=nodes_x, y=nodes_y, z=nodes_z,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Plasma',
                colorbar=dict(
                    title=dict(text='Activation', font=dict(color='white')),
                    tickfont=dict(color='white')
                ),
                opacity=0.9,
                line=dict(width=0.5, color='white')
            ),
            text=node_labels,
            hoverinfo='text',
            name='Neurons'
        ))

        # Add layer labels
        for layer_idx in range(num_layers):
            fig.add_trace(go.Scatter3d(
                x=[layer_idx * 5 + 2], y=[-seq_len / 2 - 2], z=[2],
                mode='text',
                text=[f'Block {layer_idx}'],
                textposition='bottom center',
                textfont=dict(size=12, color='#ff6b35'),
                showlegend=False
            ))

        fig.update_layout(
            title=dict(text='TinyGPT Transformer (Drag to Rotate)', font=dict(color='white', size=16)),
            scene=dict(
                xaxis=dict(title=dict(text='Layer/Stage', font=dict(color='cyan')), tickfont=dict(color='white'),
                          showgrid=True, gridcolor='rgba(100,100,100,0.3)', zeroline=False),
                yaxis=dict(title=dict(text='Position/Head', font=dict(color='cyan')), tickfont=dict(color='white'),
                          showgrid=True, gridcolor='rgba(100,100,100,0.3)', zeroline=False),
                zaxis=dict(title=dict(text='Component', font=dict(color='#ff6b35')), tickfont=dict(color='white'),
                          showgrid=True, gridcolor='rgba(100,100,100,0.3)', zeroline=False),
                bgcolor='#0d0d1a'
            ),
            paper_bgcolor='#0d0d1a',
            plot_bgcolor='#0d0d1a',
            font=dict(color='white'),
            margin=dict(l=0, r=0, t=50, b=0),
            showlegend=True,
            legend=dict(font=dict(color='white'), bgcolor='rgba(0,0,0,0.5)')
        )

        return fig
