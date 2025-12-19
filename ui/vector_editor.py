"""
Interactive Vector Editor Module
Provides UI components for manual creation and modification of control vectors
using sliders, real-time previews, and vector interpolation.
"""

import gradio as gr
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class VectorComponent:
    """Represents a single dimension of a control vector."""
    index: int
    value: float
    importance: float = 0.0  # How much this dimension contributes
    label: Optional[str] = None


class VectorEditor:
    """
    Interactive editor for control vectors.
    Allows visualization and modification of vector components.
    """

    def __init__(self, hidden_dim: int = 768, num_visible_dims: int = 50):
        """
        Initialize vector editor.

        Args:
            hidden_dim: Full dimension of the control vector
            num_visible_dims: Number of dimensions to show in UI
        """
        self.hidden_dim = hidden_dim
        self.num_visible_dims = num_visible_dims
        self.current_vector: Optional[torch.Tensor] = None
        self.vector_history: List[torch.Tensor] = []
        self.important_dims: List[int] = list(range(num_visible_dims))

    def load_vector(self, vector: torch.Tensor) -> Dict:
        """
        Load a vector for editing.

        Args:
            vector: Control vector tensor

        Returns:
            Dict with component values for UI
        """
        self.current_vector = vector.clone()
        self.vector_history.append(vector.clone())

        # Find most important dimensions (highest magnitude)
        magnitudes = vector.abs()
        self.important_dims = magnitudes.argsort(descending=True)[:self.num_visible_dims].tolist()

        return self._get_ui_values()

    def _get_ui_values(self) -> Dict[int, float]:
        """Get values for UI sliders."""
        if self.current_vector is None:
            return {}

        return {
            dim: self.current_vector[dim].item()
            for dim in self.important_dims
        }

    def update_dimension(self, dim_idx: int, value: float) -> torch.Tensor:
        """
        Update a single dimension of the vector.

        Args:
            dim_idx: Dimension index
            value: New value

        Returns:
            Updated vector
        """
        if self.current_vector is None:
            raise ValueError("No vector loaded")

        self.current_vector[dim_idx] = value
        return self.current_vector.clone()

    def update_multiple(self, updates: Dict[int, float]) -> torch.Tensor:
        """
        Update multiple dimensions at once.

        Args:
            updates: Dict mapping dimension index to new value

        Returns:
            Updated vector
        """
        if self.current_vector is None:
            raise ValueError("No vector loaded")

        for dim_idx, value in updates.items():
            self.current_vector[dim_idx] = value

        return self.current_vector.clone()

    def scale_vector(self, scale: float) -> torch.Tensor:
        """Scale the entire vector."""
        if self.current_vector is None:
            raise ValueError("No vector loaded")

        self.current_vector = self.current_vector * scale
        return self.current_vector.clone()

    def normalize_vector(self) -> torch.Tensor:
        """Normalize vector to unit length."""
        if self.current_vector is None:
            raise ValueError("No vector loaded")

        norm = self.current_vector.norm()
        if norm > 0:
            self.current_vector = self.current_vector / norm

        return self.current_vector.clone()

    def reset_to_original(self) -> torch.Tensor:
        """Reset to the originally loaded vector."""
        if not self.vector_history:
            raise ValueError("No vector history")

        self.current_vector = self.vector_history[0].clone()
        return self.current_vector.clone()

    def undo(self) -> torch.Tensor:
        """Undo last change."""
        if len(self.vector_history) < 2:
            return self.current_vector.clone() if self.current_vector else torch.zeros(self.hidden_dim)

        self.vector_history.pop()
        self.current_vector = self.vector_history[-1].clone()
        return self.current_vector.clone()

    def save_checkpoint(self):
        """Save current state to history."""
        if self.current_vector is not None:
            self.vector_history.append(self.current_vector.clone())

    def interpolate_vectors(
        self,
        vector_a: torch.Tensor,
        vector_b: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """
        Interpolate between two vectors.

        Args:
            vector_a: First vector
            vector_b: Second vector
            alpha: Interpolation factor (0 = vector_a, 1 = vector_b)

        Returns:
            Interpolated vector
        """
        return (1 - alpha) * vector_a + alpha * vector_b

    def add_vectors(
        self,
        vector_a: torch.Tensor,
        vector_b: torch.Tensor,
        weight_a: float = 1.0,
        weight_b: float = 1.0
    ) -> torch.Tensor:
        """
        Add two vectors with weights.

        Args:
            vector_a: First vector
            vector_b: Second vector
            weight_a: Weight for first vector
            weight_b: Weight for second vector

        Returns:
            Combined vector
        """
        return weight_a * vector_a + weight_b * vector_b

    def project_out_direction(
        self,
        vector: torch.Tensor,
        direction: torch.Tensor
    ) -> torch.Tensor:
        """
        Project a direction out of the vector.
        Useful for removing unwanted components.

        Args:
            vector: Vector to modify
            direction: Direction to project out

        Returns:
            Modified vector
        """
        direction = direction / direction.norm()
        projection = torch.dot(vector, direction) * direction
        return vector - projection

    def get_statistics(self) -> Dict:
        """Get statistics about the current vector."""
        if self.current_vector is None:
            return {}

        return {
            'norm': self.current_vector.norm().item(),
            'mean': self.current_vector.mean().item(),
            'std': self.current_vector.std().item(),
            'min': self.current_vector.min().item(),
            'max': self.current_vector.max().item(),
            'sparsity': (self.current_vector.abs() < 0.01).float().mean().item(),
            'num_positive': (self.current_vector > 0).sum().item(),
            'num_negative': (self.current_vector < 0).sum().item()
        }


class VectorEditorUI:
    """Gradio UI components for vector editing."""

    def __init__(self, model_controller, hidden_dim: int = 768):
        """
        Initialize UI.

        Args:
            model_controller: ModelController instance
            hidden_dim: Hidden dimension of the model
        """
        self.controller = model_controller
        self.hidden_dim = hidden_dim
        self.editor = VectorEditor(hidden_dim)
        self.current_model = 'gpt2'
        self.loaded_vector_name: Optional[str] = None

    def create_interface(self) -> Tuple:
        """Create the Gradio interface components."""

        with gr.Column():
            gr.Markdown("## 🎛️ Interactive Control Vector Editor")
            gr.Markdown("*Directly modify vector dimensions to craft custom steering behaviors*")

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
                load_btn = gr.Button("📥 Load Vector")

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Vector Components (Top 20 by magnitude)")

                    # Create sliders for top dimensions
                    sliders = []
                    for i in range(20):
                        slider = gr.Slider(
                            minimum=-5.0,
                            maximum=5.0,
                            step=0.01,
                            value=0.0,
                            label=f"Dim {i}",
                            interactive=True
                        )
                        sliders.append(slider)

                with gr.Column(scale=1):
                    gr.Markdown("### Vector Statistics")
                    stats_display = gr.JSON(label="Current Stats")

                    gr.Markdown("### Quick Actions")
                    with gr.Row():
                        normalize_btn = gr.Button("📏 Normalize")
                        reset_btn = gr.Button("🔄 Reset")

                    scale_slider = gr.Slider(
                        minimum=0.1,
                        maximum=3.0,
                        step=0.1,
                        value=1.0,
                        label="Scale Factor"
                    )
                    scale_btn = gr.Button("📐 Apply Scale")

            gr.Markdown("### Vector Combination")
            with gr.Row():
                vector_a_select = gr.Dropdown(
                    choices=[],
                    label="Vector A"
                )
                vector_b_select = gr.Dropdown(
                    choices=[],
                    label="Vector B"
                )
                interpolation_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.5,
                    label="Interpolation (0=A, 1=B)"
                )
                interpolate_btn = gr.Button("🔀 Interpolate")

            with gr.Row():
                weight_a = gr.Slider(
                    minimum=-2.0,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    label="Weight A"
                )
                weight_b = gr.Slider(
                    minimum=-2.0,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    label="Weight B"
                )
                add_btn = gr.Button("➕ Add Vectors")

            gr.Markdown("### Save Custom Vector")
            with gr.Row():
                save_name = gr.Textbox(label="Vector Name", placeholder="my_custom_vector")
                save_btn = gr.Button("💾 Save Vector")
                save_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("### Preview Generation")
            with gr.Row():
                preview_prompt = gr.Textbox(
                    label="Test Prompt",
                    value="The scientist discovered that",
                    lines=2
                )
                preview_strength = gr.Slider(
                    minimum=-2.0,
                    maximum=2.0,
                    step=0.1,
                    value=1.0,
                    label="Strength"
                )

            preview_btn = gr.Button("🔮 Preview Generation")
            preview_output = gr.Textbox(label="Generated Output", lines=4)

        # Return all components for event wiring
        return (
            model_select, vector_select, load_btn,
            sliders, stats_display,
            normalize_btn, reset_btn, scale_slider, scale_btn,
            vector_a_select, vector_b_select, interpolation_slider, interpolate_btn,
            weight_a, weight_b, add_btn,
            save_name, save_btn, save_status,
            preview_prompt, preview_strength, preview_btn, preview_output
        )

    def get_available_vectors(self, model_name: str) -> List[str]:
        """Get list of available vectors for a model."""
        vectors = list(self.controller.vector_controllers[model_name].vectors.keys())
        # Also add prompt set names
        prompt_sets = self.controller.config_manager.load_prompt_sets(model_name)
        vectors.extend(prompt_sets.keys())
        return vectors

    def load_vector(self, model_name: str, vector_name: str) -> Tuple[Dict, List[float]]:
        """Load a vector and return UI values."""
        self.current_model = model_name
        self.loaded_vector_name = vector_name

        # Try to get from saved vectors
        vector = self.controller.vector_controllers[model_name].get_vector(vector_name)

        if vector is None:
            # Try to create from prompt sets
            prompt_sets = self.controller.config_manager.load_prompt_sets(model_name)
            if vector_name in prompt_sets:
                model = self.controller.models[model_name]
                vector = self.controller.get_vector_from_prompt_set(model, prompt_sets, vector_name)

        if vector is None:
            return {}, [0.0] * 20

        ui_values = self.editor.load_vector(vector)
        stats = self.editor.get_statistics()

        # Return values for top 20 sliders
        slider_values = [
            self.editor.current_vector[dim].item()
            for dim in self.editor.important_dims[:20]
        ]

        return stats, slider_values

    def update_from_sliders(self, *slider_values) -> Dict:
        """Update vector from slider values."""
        if self.editor.current_vector is None:
            return {}

        updates = {
            self.editor.important_dims[i]: val
            for i, val in enumerate(slider_values)
            if i < len(self.editor.important_dims)
        }

        self.editor.update_multiple(updates)
        return self.editor.get_statistics()

    def normalize(self) -> Tuple[Dict, List[float]]:
        """Normalize the current vector."""
        if self.editor.current_vector is None:
            return {}, [0.0] * 20

        self.editor.normalize_vector()
        stats = self.editor.get_statistics()
        slider_values = [
            self.editor.current_vector[dim].item()
            for dim in self.editor.important_dims[:20]
        ]
        return stats, slider_values

    def scale(self, scale_factor: float) -> Tuple[Dict, List[float]]:
        """Scale the current vector."""
        if self.editor.current_vector is None:
            return {}, [0.0] * 20

        self.editor.scale_vector(scale_factor)
        stats = self.editor.get_statistics()
        slider_values = [
            self.editor.current_vector[dim].item()
            for dim in self.editor.important_dims[:20]
        ]
        return stats, slider_values

    def reset(self) -> Tuple[Dict, List[float]]:
        """Reset to original vector."""
        if self.editor.current_vector is None:
            return {}, [0.0] * 20

        self.editor.reset_to_original()
        stats = self.editor.get_statistics()
        slider_values = [
            self.editor.current_vector[dim].item()
            for dim in self.editor.important_dims[:20]
        ]
        return stats, slider_values

    def interpolate(
        self,
        model_name: str,
        vector_a_name: str,
        vector_b_name: str,
        alpha: float
    ) -> Tuple[Dict, List[float]]:
        """Interpolate between two vectors."""
        vector_a = self._get_vector(model_name, vector_a_name)
        vector_b = self._get_vector(model_name, vector_b_name)

        if vector_a is None or vector_b is None:
            return {}, [0.0] * 20

        interpolated = self.editor.interpolate_vectors(vector_a, vector_b, alpha)
        self.editor.load_vector(interpolated)

        stats = self.editor.get_statistics()
        slider_values = [
            self.editor.current_vector[dim].item()
            for dim in self.editor.important_dims[:20]
        ]
        return stats, slider_values

    def add_vectors(
        self,
        model_name: str,
        vector_a_name: str,
        vector_b_name: str,
        weight_a: float,
        weight_b: float
    ) -> Tuple[Dict, List[float]]:
        """Add two vectors with weights."""
        vector_a = self._get_vector(model_name, vector_a_name)
        vector_b = self._get_vector(model_name, vector_b_name)

        if vector_a is None or vector_b is None:
            return {}, [0.0] * 20

        combined = self.editor.add_vectors(vector_a, vector_b, weight_a, weight_b)
        self.editor.load_vector(combined)

        stats = self.editor.get_statistics()
        slider_values = [
            self.editor.current_vector[dim].item()
            for dim in self.editor.important_dims[:20]
        ]
        return stats, slider_values

    def _get_vector(self, model_name: str, vector_name: str) -> Optional[torch.Tensor]:
        """Helper to get a vector by name."""
        vector = self.controller.vector_controllers[model_name].get_vector(vector_name)

        if vector is None:
            prompt_sets = self.controller.config_manager.load_prompt_sets(model_name)
            if vector_name in prompt_sets:
                model = self.controller.models[model_name]
                vector = self.controller.get_vector_from_prompt_set(model, prompt_sets, vector_name)

        return vector

    def save_vector(self, model_name: str, name: str) -> str:
        """Save the current vector."""
        if self.editor.current_vector is None:
            return "❌ No vector to save"

        if not name:
            return "❌ Please provide a name"

        self.controller.vector_controllers[model_name].add_vector(
            name,
            self.editor.current_vector.clone()
        )

        return f"✅ Saved vector '{name}'"

    def preview_generation(
        self,
        model_name: str,
        prompt: str,
        strength: float
    ) -> str:
        """Generate text with the current vector."""
        if self.editor.current_vector is None:
            return "No vector loaded"

        model = self.controller.models[model_name]

        try:
            output = model.generate_with_control(
                prompt,
                self.editor.current_vector,
                scale=strength
            )
            return output
        except Exception as e:
            return f"Error: {str(e)}"


class VectorVisualizerUI:
    """UI for visualizing control vectors."""

    def __init__(self, hidden_dim: int = 768):
        self.hidden_dim = hidden_dim

    def create_vector_plot(
        self,
        vector: torch.Tensor,
        title: str = "Control Vector Components"
    ):
        """Create a visualization of vector components."""
        import matplotlib.pyplot as plt
        import io
        from PIL import Image

        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor('#1a1a2e')

        vector_np = vector.cpu().numpy()

        # Plot 1: Component values (sorted by magnitude)
        ax = axes[0, 0]
        ax.set_facecolor('#1a1a2e')
        sorted_indices = np.argsort(np.abs(vector_np))[::-1]
        top_n = 50
        ax.bar(range(top_n), vector_np[sorted_indices[:top_n]],
               color=['#00ffff' if v > 0 else '#ff6b35' for v in vector_np[sorted_indices[:top_n]]],
               alpha=0.8)
        ax.set_xlabel('Dimension (sorted by magnitude)', color='#e0e0ff')
        ax.set_ylabel('Value', color='#e0e0ff')
        ax.set_title('Top 50 Components', color='#e0e0ff')
        ax.tick_params(colors='#e0e0ff')
        ax.axhline(y=0, color='#4a4a5e', linestyle='-', linewidth=0.5)

        # Plot 2: Distribution histogram
        ax = axes[0, 1]
        ax.set_facecolor('#1a1a2e')
        ax.hist(vector_np, bins=50, color='#ffd700', alpha=0.7, edgecolor='#1a1a2e')
        ax.set_xlabel('Value', color='#e0e0ff')
        ax.set_ylabel('Count', color='#e0e0ff')
        ax.set_title('Value Distribution', color='#e0e0ff')
        ax.tick_params(colors='#e0e0ff')

        # Plot 3: Cumulative magnitude
        ax = axes[1, 0]
        ax.set_facecolor('#1a1a2e')
        sorted_magnitudes = np.sort(np.abs(vector_np))[::-1]
        cumulative = np.cumsum(sorted_magnitudes) / np.sum(sorted_magnitudes)
        ax.plot(cumulative, color='#00ffff', linewidth=2)
        ax.set_xlabel('Number of Dimensions', color='#e0e0ff')
        ax.set_ylabel('Cumulative Magnitude Fraction', color='#e0e0ff')
        ax.set_title('Magnitude Concentration', color='#e0e0ff')
        ax.tick_params(colors='#e0e0ff')
        ax.grid(True, color='#4a4a5e', alpha=0.3)
        ax.axhline(y=0.9, color='#ff6b35', linestyle='--', alpha=0.5, label='90%')
        ax.legend(facecolor='#1a1a2e', labelcolor='#e0e0ff')

        # Plot 4: Heatmap view
        ax = axes[1, 1]
        ax.set_facecolor('#1a1a2e')
        # Reshape vector into a 2D grid for visualization
        side = int(np.ceil(np.sqrt(len(vector_np))))
        padded = np.zeros(side * side)
        padded[:len(vector_np)] = vector_np
        grid = padded.reshape(side, side)

        im = ax.imshow(grid, cmap='RdBu_r', aspect='auto')
        ax.set_title('Vector as 2D Grid', color='#e0e0ff')
        ax.tick_params(colors='#e0e0ff')
        plt.colorbar(im, ax=ax)

        fig.suptitle(title, color='#e0e0ff', fontsize=14, y=1.02)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1a1a2e', dpi=150)
        plt.close(fig)
        buf.seek(0)

        return Image.open(buf)
