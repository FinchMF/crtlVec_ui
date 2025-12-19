"""
Layer Visualization Module
Creates visualizations for model activations, entropy flow, and control vector effects.
Includes heatmaps, trajectory plots, 3D manifolds, layer graphs, and before/after comparisons.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import io
from PIL import Image
from dataclasses import dataclass

# Import our analysis modules
import sys
from pathlib import Path


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    figure_bg_color: str = 'black'
    text_color: str = '#e0e0ff'
    grid_color: str = '#444444'
    accent_colors: List[str] = None

    def __post_init__(self):
        if self.accent_colors is None:
            self.accent_colors = [
                '#FF6B35', '#FF1493', '#FFD700', '#00FFFF',
                '#32CD32', '#FF4500', '#DA70D6', '#00FA9A'
            ]


class LayerVisualizer:
    """Visualizes layer-by-layer activations and transformations."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        plt.style.use('dark_background')

    def _setup_figure(self, figsize: Tuple[int, int] = (12, 8)) -> Tuple[plt.Figure, plt.Axes]:
        """Create a styled figure."""
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)
        return fig, ax

    def _fig_to_image(self, fig: plt.Figure) -> Image.Image:
        """Convert matplotlib figure to PIL Image."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight',
                   facecolor=self.config.figure_bg_color, dpi=150)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img.load()  # Force load image data into memory
        return img.copy()  # Return a copy that doesn't depend on the buffer

    def create_activation_heatmap(
        self,
        activations: np.ndarray,
        tokens: List[str],
        title: str = "Layer Activations",
        layer_labels: Optional[List[str]] = None
    ) -> Image.Image:
        """
        Create heatmap of activations across layers and tokens.

        Args:
            activations: Array of shape [num_layers, seq_len] or [num_layers, seq_len, hidden_dim]
            tokens: List of token strings
            title: Plot title
            layer_labels: Optional labels for layers

        Returns:
            PIL Image of the heatmap
        """
        # If activations have hidden dim, take mean or norm
        if activations.ndim == 3:
            activations = np.linalg.norm(activations, axis=-1)

        num_layers, seq_len = activations.shape

        fig, ax = plt.subplots(figsize=(max(12, seq_len * 0.8), max(8, num_layers * 0.4)))
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        # Create custom colormap (dark blue to bright orange)
        colors = ['#1a1a2e', '#2d3561', '#4a4a8a', '#7b68ee', '#ffa500', '#ff4500']
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

        im = ax.imshow(activations, aspect='auto', cmap=cmap)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Activation Magnitude', color=self.config.text_color)
        cbar.ax.yaxis.set_tick_params(color=self.config.text_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.config.text_color)

        # Set labels
        ax.set_xlabel('Token Position', color=self.config.text_color, fontsize=12)
        ax.set_ylabel('Layer', color=self.config.text_color, fontsize=12)
        ax.set_title(title, color=self.config.text_color, fontsize=14, pad=20)

        # Token labels on x-axis
        if len(tokens) <= 20:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
        else:
            # Show subset of tokens
            step = max(1, len(tokens) // 20)
            positions = list(range(0, len(tokens), step))
            ax.set_xticks(positions)
            ax.set_xticklabels([tokens[i] for i in positions], rotation=45, ha='right', fontsize=9)

        # Layer labels on y-axis
        if layer_labels:
            ax.set_yticks(range(len(layer_labels)))
            ax.set_yticklabels(layer_labels)
        else:
            ax.set_yticks(range(num_layers))
            ax.set_yticklabels([f'L{i}' for i in range(num_layers)])

        ax.tick_params(colors=self.config.text_color)

        return self._fig_to_image(fig)

    def create_entropy_flow(
        self,
        entropy_values: np.ndarray,
        tokens: List[str],
        title: str = "Entropy Flow Through Layers"
    ) -> Image.Image:
        """
        Visualize how entropy changes as information flows through layers.

        Args:
            entropy_values: Array of shape [num_layers, seq_len]
            tokens: List of token strings
            title: Plot title

        Returns:
            PIL Image of entropy flow
        """
        num_layers, seq_len = entropy_values.shape

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        # Create heatmap
        colors = ['#1a1a2e', '#2e4057', '#4a6fa5', '#ffd700', '#ff6b35', '#ff1493']
        cmap = LinearSegmentedColormap.from_list('entropy', colors, N=256)

        im = ax.imshow(entropy_values, aspect='auto', cmap=cmap)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Entropy (bits)', color=self.config.text_color)
        cbar.ax.yaxis.set_tick_params(color=self.config.text_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.config.text_color)

        # Labels
        ax.set_xlabel('Token', color=self.config.text_color, fontsize=12)
        ax.set_ylabel('Layer', color=self.config.text_color, fontsize=12)
        ax.set_title(title, color=self.config.text_color, fontsize=14, pad=20)

        # Token labels
        if len(tokens) <= 15:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)

        ax.set_yticks(range(num_layers))
        ax.set_yticklabels([f'Layer {i}' for i in range(num_layers)], fontsize=9)
        ax.tick_params(colors=self.config.text_color)

        # Add mean entropy per layer as side plot
        mean_entropy = entropy_values.mean(axis=1)

        # Create inset axes for the line plot
        ax2 = fig.add_axes([0.92, 0.15, 0.06, 0.7])
        ax2.set_facecolor(self.config.figure_bg_color)
        ax2.plot(mean_entropy, range(num_layers), color='#00ffff', linewidth=2, marker='o', markersize=4)
        ax2.set_ylim(-0.5, num_layers - 0.5)
        ax2.set_xlabel('Mean', color=self.config.text_color, fontsize=9)
        ax2.invert_yaxis()
        ax2.set_yticks([])
        ax2.tick_params(colors=self.config.text_color, labelsize=8)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_color(self.config.grid_color)

        return self._fig_to_image(fig)

    def create_topk_predictions_table(
        self,
        predictions: List[Dict],
        layer_idx: int,
        title: str = "Top-K Predictions"
    ) -> Image.Image:
        """
        Create a visual table of top-k predictions for each token.

        Args:
            predictions: List of dicts with 'token', 'top_k_tokens', 'top_k_probs', 'entropy'
            layer_idx: Layer index for title
            title: Plot title

        Returns:
            PIL Image of predictions table
        """
        num_tokens = len(predictions)
        k = len(predictions[0]['top_k_tokens']) if predictions else 5

        fig, ax = plt.subplots(figsize=(max(14, num_tokens * 1.5), k * 0.8 + 2))
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)
        ax.axis('off')

        # Create table data
        cell_text = []
        for i in range(k):
            row = []
            for pred in predictions:
                if i < len(pred['top_k_tokens']):
                    token = pred['top_k_tokens'][i]
                    prob = pred['top_k_probs'][i]
                    row.append(f"{token}\n({prob:.2%})")
                else:
                    row.append("")
            cell_text.append(row)

        # Column headers (input tokens)
        col_labels = [f"{p['token']}\nH={p['entropy']:.2f}" for p in predictions]
        row_labels = [f"#{i+1}" for i in range(k)]

        # Create table
        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )

        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)

        # Color cells based on probability
        for i in range(k):
            for j in range(num_tokens):
                cell = table[(i + 1, j)]
                if i < len(predictions[j]['top_k_probs']):
                    prob = predictions[j]['top_k_probs'][i]
                    # Color intensity based on probability
                    intensity = min(prob * 2, 1.0)
                    cell.set_facecolor(plt.cm.YlOrRd(intensity * 0.7))
                    cell.set_text_props(color='white' if intensity > 0.5 else 'black')
                else:
                    cell.set_facecolor(self.config.figure_bg_color)

        # Style header row
        for j in range(num_tokens):
            cell = table[(0, j)]
            cell.set_facecolor('#2d3561')
            cell.set_text_props(color=self.config.text_color, fontweight='bold')

        # Style row labels
        for i in range(k):
            cell = table[(i + 1, -1)]
            cell.set_facecolor('#2d3561')
            cell.set_text_props(color=self.config.text_color)

        ax.set_title(f"{title} (Layer {layer_idx})",
                    color=self.config.text_color, fontsize=14, pad=20)

        return self._fig_to_image(fig)

    def create_control_vector_comparison(
        self,
        original_activations: np.ndarray,
        modified_activations: np.ndarray,
        tokens: List[str],
        title: str = "Control Vector Effect"
    ) -> Image.Image:
        """
        Compare activations before and after control vector application.

        Args:
            original_activations: Original hidden states [num_layers, seq_len, hidden_dim]
            modified_activations: Modified hidden states [num_layers, seq_len, hidden_dim]
            tokens: Token strings
            title: Plot title

        Returns:
            PIL Image showing comparison
        """
        # Validate inputs - handle empty or mismatched arrays
        if original_activations.size == 0 or modified_activations.size == 0:
            # Return a placeholder image with error message
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor(self.config.figure_bg_color)
            ax.set_facecolor(self.config.figure_bg_color)
            ax.text(0.5, 0.5, "Insufficient data for comparison\n(activation arrays are empty)",
                   ha='center', va='center', fontsize=14, color=self.config.text_color,
                   transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return self._fig_to_image(fig)

        # Compute norms for visualization
        if original_activations.ndim == 3:
            orig_norms = np.linalg.norm(original_activations, axis=-1)
            mod_norms = np.linalg.norm(modified_activations, axis=-1)
        else:
            orig_norms = original_activations
            mod_norms = modified_activations

        # Check shape compatibility
        if orig_norms.shape != mod_norms.shape:
            # Return a placeholder image with error message
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor(self.config.figure_bg_color)
            ax.set_facecolor(self.config.figure_bg_color)
            ax.text(0.5, 0.5, f"Shape mismatch: original {orig_norms.shape} vs modified {mod_norms.shape}\nCannot create comparison",
                   ha='center', va='center', fontsize=14, color=self.config.text_color,
                   transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return self._fig_to_image(fig)

        # Compute difference
        diff = mod_norms - orig_norms

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor(self.config.figure_bg_color)

        titles = ['Original', 'With Control Vector', 'Difference (Δ)']
        data = [orig_norms, mod_norms, diff]
        cmaps = ['viridis', 'viridis', 'RdBu_r']

        for ax, d, t, cmap in zip(axes, data, titles, cmaps):
            ax.set_facecolor(self.config.figure_bg_color)

            if t == 'Difference (Δ)':
                # Center colormap at 0 for difference
                vmax = max(abs(d.min()), abs(d.max()))
                im = ax.imshow(d, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax)
            else:
                im = ax.imshow(d, aspect='auto', cmap=cmap)

            plt.colorbar(im, ax=ax, pad=0.02)
            ax.set_title(t, color=self.config.text_color, fontsize=12)
            ax.set_xlabel('Token', color=self.config.text_color)
            ax.set_ylabel('Layer', color=self.config.text_color)
            ax.tick_params(colors=self.config.text_color)

            if len(tokens) <= 10:
                ax.set_xticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)

        fig.suptitle(title, color=self.config.text_color, fontsize=14, y=1.02)
        plt.tight_layout()

        return self._fig_to_image(fig)

    def create_embedding_trajectory(
        self,
        embeddings_original: np.ndarray,
        embeddings_modified: np.ndarray,
        control_vector: np.ndarray,
        title: str = "Embedding Space Trajectory"
    ) -> Image.Image:
        """
        Visualize the trajectory of embeddings in 2D PCA space.
        Shows original, modified, and the control vector direction.

        Args:
            embeddings_original: Original embeddings [seq_len, hidden_dim]
            embeddings_modified: Modified embeddings [seq_len, hidden_dim]
            control_vector: The control vector [hidden_dim]
            title: Plot title

        Returns:
            PIL Image of trajectory visualization
        """
        from sklearn.decomposition import PCA

        # Combine all points for PCA
        all_points = np.vstack([
            embeddings_original,
            embeddings_modified,
            control_vector.reshape(1, -1)
        ])

        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_points)

        n = len(embeddings_original)
        orig_2d = all_2d[:n]
        mod_2d = all_2d[n:2*n]
        cv_2d = all_2d[-1]

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        # Plot original trajectory
        ax.plot(orig_2d[:, 0], orig_2d[:, 1], 'o-', color='#00ffff',
               linewidth=2, markersize=8, label='Original', alpha=0.8)

        # Plot modified trajectory
        ax.plot(mod_2d[:, 0], mod_2d[:, 1], 's-', color='#ff6b35',
               linewidth=2, markersize=8, label='With Control Vector', alpha=0.8)

        # Draw arrows from original to modified
        for i in range(n):
            ax.annotate('', xy=(mod_2d[i, 0], mod_2d[i, 1]),
                       xytext=(orig_2d[i, 0], orig_2d[i, 1]),
                       arrowprops=dict(arrowstyle='->', color='#ffd700',
                                      alpha=0.5, lw=1.5))

        # Plot control vector direction from origin
        origin = np.mean(orig_2d, axis=0)
        cv_scale = np.linalg.norm(mod_2d - orig_2d, axis=1).mean() * 2
        cv_direction = cv_2d / np.linalg.norm(cv_2d) * cv_scale

        ax.annotate('', xy=(origin[0] + cv_direction[0], origin[1] + cv_direction[1]),
                   xytext=(origin[0], origin[1]),
                   arrowprops=dict(arrowstyle='->', color='#ff1493',
                                  lw=3, alpha=0.9))
        ax.scatter([origin[0]], [origin[1]], marker='*', s=200,
                  color='#ff1493', label='Control Vector Direction', zorder=5)

        # Add token indices
        for i in range(n):
            ax.annotate(str(i), (orig_2d[i, 0], orig_2d[i, 1]),
                       fontsize=8, color=self.config.text_color,
                       xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('PC1', color=self.config.text_color, fontsize=12)
        ax.set_ylabel('PC2', color=self.config.text_color, fontsize=12)
        ax.set_title(title, color=self.config.text_color, fontsize=14, pad=20)
        ax.tick_params(colors=self.config.text_color)
        ax.legend(facecolor=self.config.figure_bg_color,
                 edgecolor=self.config.grid_color,
                 labelcolor=self.config.text_color)
        ax.grid(True, color=self.config.grid_color, alpha=0.3, linestyle='--')

        return self._fig_to_image(fig)

    def create_3d_activation_manifold(
        self,
        activations: np.ndarray,
        tokens: List[str],
        title: str = "3D Activation Manifold"
    ) -> Image.Image:
        """
        Create 3D manifold representation of activations across layers.

        Args:
            activations: Array of shape [num_layers, seq_len] or [num_layers, seq_len, hidden_dim]
            tokens: List of token strings
            title: Plot title

        Returns:
            PIL Image of 3D manifold
        """
        # If activations have hidden dim, take norm
        if activations.ndim == 3:
            activations = np.linalg.norm(activations, axis=-1)

        num_layers, seq_len = activations.shape

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        # Create meshgrid for surface
        X, Y = np.meshgrid(range(seq_len), range(num_layers))
        Z = activations

        # Create custom colormap
        colors = ['#1a1a2e', '#2d3561', '#4a4a8a', '#7b68ee', '#ffa500', '#ff4500']
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85,
                               linewidth=0.5, antialiased=True,
                               edgecolor='white')

        # Add wireframe for structure
        ax.plot_wireframe(X, Y, Z, color='white', alpha=0.1, linewidth=0.3)

        # Add contour projections on walls
        ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap=cmap, alpha=0.5)
        ax.contour(X, Y, Z, zdir='x', offset=-1, cmap=cmap, alpha=0.3)
        ax.contour(X, Y, Z, zdir='y', offset=num_layers, cmap=cmap, alpha=0.3)

        # Colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label('Activation Magnitude', color=self.config.text_color)
        cbar.ax.yaxis.set_tick_params(color=self.config.text_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.config.text_color)

        # Labels
        ax.set_xlabel('Token Position', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_ylabel('Layer', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_zlabel('Activation', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_title(title, color=self.config.text_color, fontsize=14, pad=20)

        # Set viewing angle
        ax.view_init(elev=25, azim=45)

        # Tick labels
        if len(tokens) <= 10:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, fontsize=8)
        ax.set_yticks(range(0, num_layers, max(1, num_layers // 6)))

        ax.tick_params(colors=self.config.text_color)
        ax.xaxis.pane.set_facecolor('black')
        ax.yaxis.pane.set_facecolor('black')
        ax.zaxis.pane.set_facecolor('black')
        ax.xaxis.pane.set_edgecolor(self.config.grid_color)
        ax.yaxis.pane.set_edgecolor(self.config.grid_color)
        ax.zaxis.pane.set_edgecolor(self.config.grid_color)

        return self._fig_to_image(fig)

    def create_3d_entropy_manifold(
        self,
        entropy_values: np.ndarray,
        tokens: List[str],
        title: str = "3D Entropy Landscape"
    ) -> Image.Image:
        """
        Create 3D manifold representation of entropy across layers.
        Shows the 'certainty landscape' of the model.

        Args:
            entropy_values: Array of shape [num_layers, seq_len]
            tokens: List of token strings
            title: Plot title

        Returns:
            PIL Image of 3D entropy manifold
        """
        num_layers, seq_len = entropy_values.shape

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        # Create meshgrid
        X, Y = np.meshgrid(range(seq_len), range(num_layers))
        Z = entropy_values

        # Entropy colormap (low entropy = confident = cool, high = uncertain = hot)
        colors = ['#00ff88', '#00ffff', '#ffd700', '#ff6b35', '#ff1493']
        cmap = LinearSegmentedColormap.from_list('entropy3d', colors, N=256)

        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85,
                               linewidth=0.5, antialiased=True,
                               edgecolor='white')

        # Add scatter points at peaks (high uncertainty)
        threshold = np.percentile(Z, 90)
        high_entropy_mask = Z > threshold
        if high_entropy_mask.any():
            high_x = X[high_entropy_mask]
            high_y = Y[high_entropy_mask]
            high_z = Z[high_entropy_mask]
            ax.scatter(high_x, high_y, high_z, c='#ff1493', s=50,
                      marker='^', alpha=0.9, label='High Uncertainty')

        # Colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label('Entropy (bits)', color=self.config.text_color)
        cbar.ax.yaxis.set_tick_params(color=self.config.text_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.config.text_color)

        # Labels
        ax.set_xlabel('Token Position', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_ylabel('Layer', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_zlabel('Entropy', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_title(title, color=self.config.text_color, fontsize=14, pad=20)

        ax.view_init(elev=30, azim=135)

        if len(tokens) <= 10:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, fontsize=8)

        ax.tick_params(colors=self.config.text_color)
        ax.xaxis.pane.set_facecolor('black')
        ax.yaxis.pane.set_facecolor('black')
        ax.zaxis.pane.set_facecolor('black')
        ax.xaxis.pane.set_edgecolor(self.config.grid_color)
        ax.yaxis.pane.set_edgecolor(self.config.grid_color)
        ax.zaxis.pane.set_edgecolor(self.config.grid_color)

        if high_entropy_mask.any():
            ax.legend(facecolor=self.config.figure_bg_color,
                     labelcolor=self.config.text_color, loc='upper right')

        return self._fig_to_image(fig)

    def create_layer_graph(
        self,
        activations: np.ndarray,
        tokens: List[str],
        title: str = "Layer Connectivity Graph",
        connection_threshold: float = 0.7
    ) -> Image.Image:
        """
        Visualize layers as connected nodes showing information flow.

        Args:
            activations: Array of shape [num_layers, seq_len] or [num_layers, seq_len, hidden_dim]
            tokens: List of token strings
            title: Plot title
            connection_threshold: Correlation threshold for drawing connections

        Returns:
            PIL Image of layer graph
        """
        if activations.ndim == 3:
            # Flatten to [num_layers, seq_len * hidden_dim] for correlation
            activations_flat = activations.reshape(activations.shape[0], -1)
            activations_norm = np.linalg.norm(activations, axis=-1)
        else:
            activations_flat = activations
            activations_norm = activations

        num_layers = activations.shape[0]
        seq_len = activations.shape[1]

        # Compute layer correlations
        correlations = np.corrcoef(activations_flat)

        fig, ax = plt.subplots(figsize=(16, 12))
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        # Layout: layers vertically, tokens horizontally
        layer_y = np.linspace(0.9, 0.1, num_layers)
        token_x = np.linspace(0.1, 0.9, seq_len)

        # Draw connections between layers (based on correlation)
        for i in range(num_layers - 1):
            corr = correlations[i, i + 1]
            if corr > connection_threshold:
                # Strong connection
                linewidth = 3 * corr
                alpha = 0.8
                color = '#00ffff'
            elif corr > 0.5:
                linewidth = 2 * corr
                alpha = 0.5
                color = '#ffd700'
            else:
                linewidth = 1
                alpha = 0.3
                color = '#4a4a5e'

            # Draw connection lines for each token path
            for j in range(seq_len):
                x_offset = (np.random.random() - 0.5) * 0.02  # Slight jitter
                ax.plot([token_x[j], token_x[j] + x_offset],
                       [layer_y[i], layer_y[i + 1]],
                       color=color, linewidth=linewidth, alpha=alpha)

        # Draw layer nodes
        for i in range(num_layers):
            layer_acts = activations_norm[i]
            normalized = (layer_acts - layer_acts.min()) / (layer_acts.max() - layer_acts.min() + 1e-8)

            for j in range(seq_len):
                # Node size based on activation magnitude
                size = 100 + 400 * normalized[j]

                # Color based on activation
                color_idx = int(normalized[j] * (len(self.config.accent_colors) - 1))
                color = self.config.accent_colors[min(color_idx, len(self.config.accent_colors) - 1)]

                ax.scatter(token_x[j], layer_y[i], s=size, c=color,
                          alpha=0.8, edgecolors='white', linewidths=0.5, zorder=5)

        # Add layer labels
        for i in range(num_layers):
            ax.text(-0.02, layer_y[i], f'L{i}', color=self.config.text_color,
                   fontsize=10, ha='right', va='center', fontweight='bold')

        # Add token labels
        if len(tokens) <= 15:
            for j, token in enumerate(tokens):
                ax.text(token_x[j], 0.02, token, color=self.config.text_color,
                       fontsize=9, ha='center', va='top', rotation=45)

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='#00ffff', linewidth=3, label='Strong correlation (>0.7)'),
            plt.Line2D([0], [0], color='#ffd700', linewidth=2, label='Medium correlation (0.5-0.7)'),
            plt.Line2D([0], [0], color='#4a4a5e', linewidth=1, label='Weak correlation (<0.5)')
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                 facecolor=self.config.figure_bg_color,
                 edgecolor=self.config.grid_color,
                 labelcolor=self.config.text_color)

        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 1)
        ax.set_title(title, color=self.config.text_color, fontsize=14, pad=20)
        ax.axis('off')

        return self._fig_to_image(fig)

    def create_attention_flow_graph(
        self,
        attention_weights: np.ndarray,
        tokens: List[str],
        layer_idx: int = -1,
        title: str = "Attention Flow Graph"
    ) -> Image.Image:
        """
        Visualize attention as a flow graph between tokens.

        Args:
            attention_weights: Attention matrix [num_heads, seq_len, seq_len] or [seq_len, seq_len]
            tokens: List of token strings
            layer_idx: Which layer (for title)
            title: Plot title

        Returns:
            PIL Image of attention flow graph
        """
        # Average over heads if needed
        if attention_weights.ndim == 3:
            attn = attention_weights.mean(axis=0)
        else:
            attn = attention_weights

        seq_len = len(tokens)

        fig, ax = plt.subplots(figsize=(14, 14))
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        # Circular layout for tokens
        angles = np.linspace(0, 2 * np.pi, seq_len, endpoint=False)
        radius = 0.4
        center = (0.5, 0.5)

        node_x = center[0] + radius * np.cos(angles - np.pi/2)
        node_y = center[1] + radius * np.sin(angles - np.pi/2)

        # Draw attention edges
        for i in range(seq_len):
            for j in range(seq_len):
                weight = attn[i, j]
                if weight > 0.05:  # Only draw significant attention
                    alpha = min(weight * 2, 1.0)
                    linewidth = weight * 5

                    # Color based on weight
                    if weight > 0.3:
                        color = '#ff1493'
                    elif weight > 0.15:
                        color = '#ffd700'
                    else:
                        color = '#00ffff'

                    # Draw curved edge
                    if i != j:
                        # Control point for curve
                        mid_x = (node_x[i] + node_x[j]) / 2
                        mid_y = (node_y[i] + node_y[j]) / 2
                        ctrl_x = center[0] + 0.7 * (mid_x - center[0])
                        ctrl_y = center[1] + 0.7 * (mid_y - center[1])

                        ax.annotate('', xy=(node_x[j], node_y[j]),
                                   xytext=(node_x[i], node_y[i]),
                                   arrowprops=dict(arrowstyle='->', color=color,
                                                  alpha=alpha, lw=linewidth,
                                                  connectionstyle=f'arc3,rad=0.2'))
                    else:
                        # Self-attention loop
                        circle = plt.Circle((node_x[i], node_y[i] + 0.05), 0.03,
                                           fill=False, color=color, alpha=alpha,
                                           linewidth=linewidth)
                        ax.add_patch(circle)

        # Draw token nodes
        for i, token in enumerate(tokens):
            ax.scatter(node_x[i], node_y[i], s=800, c='#2d3561',
                      edgecolors='#00ffff', linewidths=2, zorder=10)

            # Token label
            ax.text(node_x[i], node_y[i], token[:8], color=self.config.text_color,
                   fontsize=9, ha='center', va='center', fontweight='bold', zorder=11)

            # Position label outside
            label_x = center[0] + (radius + 0.08) * np.cos(angles[i] - np.pi/2)
            label_y = center[1] + (radius + 0.08) * np.sin(angles[i] - np.pi/2)
            ax.text(label_x, label_y, str(i), color=self.config.text_color,
                   fontsize=8, ha='center', va='center', alpha=0.7)

        layer_str = f" (Layer {layer_idx})" if layer_idx >= 0 else ""
        ax.set_title(f"{title}{layer_str}", color=self.config.text_color, fontsize=14, pad=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='#ff1493', linewidth=3, label='Strong (>0.3)'),
            plt.Line2D([0], [0], color='#ffd700', linewidth=2, label='Medium (0.15-0.3)'),
            plt.Line2D([0], [0], color='#00ffff', linewidth=1, label='Weak (0.05-0.15)')
        ]
        ax.legend(handles=legend_elements, loc='lower right',
                 facecolor=self.config.figure_bg_color,
                 edgecolor=self.config.grid_color,
                 labelcolor=self.config.text_color)

        return self._fig_to_image(fig)

    def create_attention_head_3d(
        self,
        layer_name: str,
        weight_dict: Dict[str, np.ndarray],
        num_heads: int = 4,
        head_dim: int = 16,
        title: str = "Attention Head Structure 3D"
    ) -> Image.Image:
        """
        Create 3D visualization of attention head structure showing Q/K/V/O projections.

        Args:
            layer_name: Name of the attention layer
            weight_dict: Dict with 'q', 'k', 'v', 'o' weight matrices
            num_heads: Number of attention heads
            head_dim: Dimension per head
            title: Plot title

        Returns:
            PIL Image of 3D attention head visualization
        """
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        stages = ['Input', 'Q/K/V', 'Attention', 'Output']
        stage_colors = ['#00ffff', '#ff6b35', '#ffd700', '#00ff88']

        input_dim = num_heads * head_dim

        # Stage 0: Input layer nodes
        input_x = np.zeros(input_dim)
        input_y = np.arange(input_dim)
        input_z = np.zeros(input_dim)

        ax.scatter(input_x, input_y, input_z, c=stage_colors[0], s=20,
                   alpha=0.7, label='Input')

        # Stage 1: Q, K, V projection nodes (grouped by head)
        qkv_x, qkv_y, qkv_z, qkv_colors = [], [], [], []
        proj_colors = ['#ff1493', '#ffd700', '#00ff88']  # Q, K, V

        for head in range(num_heads):
            for proj_idx, proj in enumerate(['Q', 'K', 'V']):
                for d in range(head_dim):
                    qkv_x.append(1)
                    qkv_y.append(head * head_dim * 3 + proj_idx * head_dim + d)
                    qkv_z.append(head)
                    qkv_colors.append(proj_colors[proj_idx])

        ax.scatter(qkv_x, qkv_y, qkv_z, c=qkv_colors, s=25, alpha=0.8, label='Q/K/V')

        # Stage 2: Attention output nodes
        attn_x = np.full(num_heads * head_dim, 2)
        attn_y = np.arange(num_heads * head_dim)
        attn_z = np.array([h // head_dim for h in range(num_heads * head_dim)])

        ax.scatter(attn_x, attn_y, attn_z, c=stage_colors[2], s=25,
                   alpha=0.8, label='Attention')

        # Stage 3: Output nodes
        output_x = np.full(input_dim, 3)
        output_y = np.arange(input_dim)
        output_z = np.zeros(input_dim)

        ax.scatter(output_x, output_y, output_z, c=stage_colors[3], s=20,
                   alpha=0.7, label='Output')

        # Draw sample connections (sparse for clarity)
        for head in range(num_heads):
            # Input to Q/K/V
            for i in range(0, input_dim, input_dim // 4):
                for proj_idx in range(3):
                    q_y = head * head_dim * 3 + proj_idx * head_dim + (i % head_dim)
                    ax.plot([0, 1], [i, q_y], [0, head],
                           color='gray', alpha=0.2, linewidth=0.5)

            # Q/K/V to Attention
            for d in range(0, head_dim, head_dim // 4):
                attn_y_pos = head * head_dim + d
                for proj_idx in range(3):
                    q_y = head * head_dim * 3 + proj_idx * head_dim + d
                    ax.plot([1, 2], [q_y, attn_y_pos], [head, head],
                           color='gray', alpha=0.2, linewidth=0.5)

            # Attention to Output
            for d in range(0, head_dim, head_dim // 4):
                attn_y_pos = head * head_dim + d
                ax.plot([2, 3], [attn_y_pos, attn_y_pos], [head, 0],
                       color='gray', alpha=0.2, linewidth=0.5)

        # Labels and styling
        ax.set_xlabel('Stage', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_ylabel('Dimension', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_zlabel('Head', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_title(f'{title}\n{layer_name}', color=self.config.text_color, fontsize=14, pad=20)

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(stages, fontsize=9)

        ax.tick_params(colors=self.config.text_color)
        ax.xaxis.pane.set_facecolor('black')
        ax.yaxis.pane.set_facecolor('black')
        ax.zaxis.pane.set_facecolor('black')
        ax.xaxis.pane.set_edgecolor(self.config.grid_color)
        ax.yaxis.pane.set_edgecolor(self.config.grid_color)
        ax.zaxis.pane.set_edgecolor(self.config.grid_color)

        ax.legend(facecolor=self.config.figure_bg_color,
                  labelcolor=self.config.text_color, loc='upper right')

        ax.view_init(elev=20, azim=45)

        return self._fig_to_image(fig)

    def create_transformer_network_graph(
        self,
        hidden_states: tuple,
        tokens: List[str],
        top_k: int = 10,
        title: str = "Transformer Network Structure"
    ) -> Image.Image:
        """
        Create static 3D visualization of transformer network structure.

        Args:
            hidden_states: Tuple of hidden states per layer [batch, seq, hidden]
            tokens: List of token strings
            top_k: Number of top neurons to highlight per token
            title: Plot title

        Returns:
            PIL Image of transformer network graph
        """
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        n_layers = len(hidden_states) - 1  # Exclude embedding
        n_tokens = min(len(tokens), 15)  # Cap for visibility

        # Collect node data
        nodes_x, nodes_y, nodes_z = [], [], []
        node_colors, node_sizes = [], []

        for layer_idx in range(n_layers):
            hs = hidden_states[layer_idx + 1][0].cpu().numpy()  # [seq, hidden]

            for tok_idx in range(n_tokens):
                # Get top activated neurons
                activations = np.abs(hs[tok_idx])
                top_neurons = np.argsort(activations)[-top_k:]

                for n_idx, neuron in enumerate(top_neurons):
                    nodes_x.append(layer_idx * 2.0)
                    nodes_y.append(tok_idx * 1.5)
                    nodes_z.append(n_idx * 0.4)
                    node_colors.append(activations[neuron])
                    node_sizes.append(30 + activations[neuron] * 20)

        # Normalize colors
        if node_colors:
            max_color = max(node_colors) if max(node_colors) > 0 else 1
            node_colors_norm = [c / max_color for c in node_colors]
        else:
            node_colors_norm = node_colors

        # Create colormap
        colors = ['#1a1a2e', '#2d3561', '#7b68ee', '#ffa500', '#ff4500']
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

        # Plot nodes
        scatter = ax.scatter(
            nodes_x, nodes_y, nodes_z,
            c=node_colors_norm, cmap=cmap,
            s=node_sizes, alpha=0.85,
            edgecolors='white', linewidths=0.3
        )

        # Draw layer connections (vertical lines showing information flow)
        for layer_idx in range(n_layers - 1):
            for tok_idx in range(n_tokens):
                # Draw connection from this layer to next
                ax.plot(
                    [layer_idx * 2.0, (layer_idx + 1) * 2.0],
                    [tok_idx * 1.5, tok_idx * 1.5],
                    [top_k * 0.2, top_k * 0.2],
                    color='#4a90d9', alpha=0.3, linewidth=1
                )

        # Draw attention-like connections between tokens at each layer
        for layer_idx in range(0, n_layers, 2):  # Every other layer for clarity
            for src_tok in range(0, n_tokens, 2):
                for dst_tok in range(0, n_tokens, 2):
                    if src_tok != dst_tok:
                        ax.plot(
                            [layer_idx * 2.0, layer_idx * 2.0],
                            [src_tok * 1.5, dst_tok * 1.5],
                            [0, 0],
                            color='#ff6b35', alpha=0.15, linewidth=0.5
                        )

        # Labels
        ax.set_xlabel('Layer', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_ylabel('Token', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_zlabel('Neuron Cluster', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_title(title, color=self.config.text_color, fontsize=14, pad=20)

        # Token labels on y-axis
        ax.set_yticks([i * 1.5 for i in range(n_tokens)])
        ax.set_yticklabels(tokens[:n_tokens], fontsize=8)

        # Colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label('Activation', color=self.config.text_color)
        cbar.ax.yaxis.set_tick_params(color=self.config.text_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.config.text_color)

        ax.tick_params(colors=self.config.text_color)
        ax.xaxis.pane.set_facecolor('black')
        ax.yaxis.pane.set_facecolor('black')
        ax.zaxis.pane.set_facecolor('black')
        ax.xaxis.pane.set_edgecolor(self.config.grid_color)
        ax.yaxis.pane.set_edgecolor(self.config.grid_color)
        ax.zaxis.pane.set_edgecolor(self.config.grid_color)

        ax.view_init(elev=25, azim=45)

        return self._fig_to_image(fig)

    def create_lstm_recurrent_graph(
        self,
        layer_sizes: List[int],
        num_timesteps: int = 3,
        activations: Optional[List[np.ndarray]] = None,
        title: str = "LSTM Recurrent Graph"
    ) -> Image.Image:
        """
        Visualize LSTM network unrolled across timesteps showing recurrent connections.

        Args:
            layer_sizes: Number of neurons per layer [input, hidden..., output]
            num_timesteps: Number of timesteps to unroll
            activations: Optional list of activation values per timestep
            title: Plot title

        Returns:
            PIL Image of LSTM recurrent graph
        """
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        num_layers = len(layer_sizes)

        # Node positions: x = layer, y = neuron, z = timestep
        all_nodes_x, all_nodes_y, all_nodes_z = [], [], []
        all_colors = []

        for t in range(num_timesteps):
            for layer_idx, size in enumerate(layer_sizes):
                for neuron_idx in range(min(size, 20)):  # Cap at 20 neurons per layer for visibility
                    all_nodes_x.append(layer_idx * 2.0)
                    all_nodes_y.append(neuron_idx - min(size, 20) / 2)
                    all_nodes_z.append(t * 3.0)

                    # Color based on layer type
                    if layer_idx == 0:
                        all_colors.append('#00ffff')  # Input
                    elif layer_idx == len(layer_sizes) - 1:
                        all_colors.append('#00ff88')  # Output
                    else:
                        all_colors.append('#ff6b35')  # Hidden

        # Plot nodes
        ax.scatter(all_nodes_x, all_nodes_y, all_nodes_z,
                   c=all_colors, s=50, alpha=0.8, edgecolors='white', linewidths=0.5)

        # Draw feedforward connections (within timestep)
        for t in range(num_timesteps):
            for layer_idx in range(num_layers - 1):
                src_size = min(layer_sizes[layer_idx], 20)
                dst_size = min(layer_sizes[layer_idx + 1], 20)

                # Sparse connections for clarity
                for src in range(0, src_size, max(1, src_size // 3)):
                    for dst in range(0, dst_size, max(1, dst_size // 3)):
                        ax.plot(
                            [layer_idx * 2.0, (layer_idx + 1) * 2.0],
                            [src - src_size / 2, dst - dst_size / 2],
                            [t * 3.0, t * 3.0],
                            color='#4a90d9', alpha=0.4, linewidth=0.8
                        )

        # Draw recurrent connections (between timesteps) - cyan, dashed
        for t in range(num_timesteps - 1):
            for layer_idx in range(1, num_layers - 1):  # Hidden layers only
                size = min(layer_sizes[layer_idx], 20)
                for neuron in range(0, size, max(1, size // 4)):
                    ax.plot(
                        [layer_idx * 2.0, layer_idx * 2.0],
                        [neuron - size / 2, neuron - size / 2],
                        [t * 3.0, (t + 1) * 3.0],
                        color='#00d4ff', alpha=0.6, linewidth=2, linestyle='--'
                    )

        # Labels and styling
        ax.set_xlabel('Layer', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_ylabel('Neuron', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_zlabel('Timestep', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_title(title, color=self.config.text_color, fontsize=14, pad=20)

        ax.tick_params(colors=self.config.text_color)
        ax.xaxis.pane.set_facecolor('black')
        ax.yaxis.pane.set_facecolor('black')
        ax.zaxis.pane.set_facecolor('black')
        ax.xaxis.pane.set_edgecolor(self.config.grid_color)
        ax.yaxis.pane.set_edgecolor(self.config.grid_color)
        ax.zaxis.pane.set_edgecolor(self.config.grid_color)

        # Custom legend
        legend_elements = [
            plt.Line2D([0], [0], color='#4a90d9', linewidth=2, label='Feedforward'),
            plt.Line2D([0], [0], color='#00d4ff', linewidth=2, linestyle='--', label='Recurrent'),
            plt.scatter([], [], c='#00ffff', s=50, label='Input'),
            plt.scatter([], [], c='#ff6b35', s=50, label='Hidden'),
            plt.scatter([], [], c='#00ff88', s=50, label='Output')
        ]
        ax.legend(handles=legend_elements[:2], facecolor=self.config.figure_bg_color,
                  labelcolor=self.config.text_color, loc='upper right')

        ax.view_init(elev=25, azim=45)

        return self._fig_to_image(fig)

    def create_mlp_network_graph(
        self,
        layer_sizes: List[int],
        weight_matrices: Optional[List[np.ndarray]] = None,
        activations: Optional[List[np.ndarray]] = None,
        top_k: int = 5,
        title: str = "MLP Network Graph"
    ) -> Image.Image:
        """
        Create 3D visualization of MLP network structure.

        Args:
            layer_sizes: Number of neurons per layer
            weight_matrices: Optional weight matrices for edge coloring
            activations: Optional activations for node coloring
            top_k: Number of top edges to show per neuron
            title: Plot title

        Returns:
            PIL Image of MLP network graph
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        num_layers = len(layer_sizes)

        # Node positions
        all_x, all_y, all_z = [], [], []
        all_colors = []

        for layer_idx, size in enumerate(layer_sizes):
            display_size = min(size, 30)  # Cap for visibility
            for neuron_idx in range(display_size):
                all_x.append(layer_idx * 2.0)
                all_y.append(neuron_idx - display_size / 2)
                all_z.append(0)

                # Color by activation if provided
                if activations and layer_idx < len(activations):
                    act = activations[layer_idx]
                    if neuron_idx < len(act.flatten()):
                        val = np.abs(act.flatten()[neuron_idx])
                        all_colors.append(val)
                    else:
                        all_colors.append(0)
                else:
                    # Default: color by layer
                    all_colors.append(layer_idx / (num_layers - 1))

        # Normalize colors
        if all_colors:
            max_c = max(all_colors) if max(all_colors) > 0 else 1
            all_colors = [c / max_c for c in all_colors]

        # Plot nodes
        colors = ['#1a1a2e', '#2d3561', '#7b68ee', '#ffa500', '#ff4500']
        cmap = LinearSegmentedColormap.from_list('custom', colors, N=256)

        scatter = ax.scatter(all_x, all_y, all_z, c=all_colors, cmap=cmap,
                            s=80, alpha=0.9, edgecolors='white', linewidths=0.5)

        # Draw edges
        for layer_idx in range(num_layers - 1):
            src_size = min(layer_sizes[layer_idx], 30)
            dst_size = min(layer_sizes[layer_idx + 1], 30)

            if weight_matrices and layer_idx < len(weight_matrices):
                weights = weight_matrices[layer_idx]

                # For each destination, find top-k sources
                for dst_idx in range(dst_size):
                    if dst_idx < weights.shape[0]:
                        row = np.abs(weights[dst_idx, :min(src_size, weights.shape[1])])
                        top_indices = np.argsort(row)[-top_k:]

                        for src_idx in top_indices:
                            if src_idx < src_size:
                                weight_val = row[src_idx]
                                alpha = min(0.9, 0.3 + weight_val / (row.max() + 1e-8) * 0.6)

                                ax.plot(
                                    [layer_idx * 2.0, (layer_idx + 1) * 2.0],
                                    [src_idx - src_size / 2, dst_idx - dst_size / 2],
                                    [0, 0],
                                    color='#4a90d9', alpha=alpha, linewidth=0.8 + alpha
                                )
            else:
                # Sparse random connections
                for src in range(0, src_size, max(1, src_size // 5)):
                    for dst in range(0, dst_size, max(1, dst_size // 5)):
                        ax.plot(
                            [layer_idx * 2.0, (layer_idx + 1) * 2.0],
                            [src - src_size / 2, dst - dst_size / 2],
                            [0, 0],
                            color='#4a90d9', alpha=0.3, linewidth=0.6
                        )

        # Labels
        ax.set_xlabel('Layer', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_ylabel('Neuron', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_zlabel('', color=self.config.text_color)
        ax.set_title(title, color=self.config.text_color, fontsize=14, pad=20)

        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label('Activation', color=self.config.text_color)
        cbar.ax.yaxis.set_tick_params(color=self.config.text_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.config.text_color)

        ax.tick_params(colors=self.config.text_color)
        ax.xaxis.pane.set_facecolor('black')
        ax.yaxis.pane.set_facecolor('black')
        ax.zaxis.pane.set_facecolor('black')
        ax.xaxis.pane.set_edgecolor(self.config.grid_color)
        ax.yaxis.pane.set_edgecolor(self.config.grid_color)
        ax.zaxis.pane.set_edgecolor(self.config.grid_color)

        ax.view_init(elev=20, azim=45)

        return self._fig_to_image(fig)

    def create_token_journey(
        self,
        hidden_states: tuple,
        tokens: List[str],
        n_tokens: int,
        title: str = "Token Transformation Journey"
    ) -> Image.Image:
        """
        Visualize how token representations transform through layers.

        Args:
            hidden_states: Tuple of hidden states per layer [batch, seq, hidden]
            tokens: List of token strings
            n_tokens: Number of tokens to visualize
            title: Plot title

        Returns:
            PIL Image showing token transformation
        """
        import numpy as np
        from sklearn.decomposition import PCA

        n_layers = len(hidden_states) - 1  # Skip embedding
        n_tokens = min(n_tokens, len(tokens))

        # Collect embeddings for each token across layers
        token_trajectories = []
        for tok_idx in range(n_tokens):
            trajectory = []
            for layer_idx in range(n_layers):
                hs = hidden_states[layer_idx + 1][0, tok_idx, :].cpu().numpy()
                trajectory.append(hs)
            token_trajectories.append(np.array(trajectory))

        # Flatten all embeddings for PCA
        all_embeddings = np.vstack([t for t in token_trajectories])

        # Reduce to 2D
        pca = PCA(n_components=2)
        all_2d = pca.fit_transform(all_embeddings)

        # Split back by token
        trajectories_2d = []
        idx = 0
        for traj in token_trajectories:
            trajectories_2d.append(all_2d[idx:idx + len(traj)])
            idx += len(traj)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        # Color palette for tokens
        colors = plt.cm.rainbow(np.linspace(0, 1, n_tokens))

        for tok_idx, (traj_2d, color) in enumerate(zip(trajectories_2d, colors)):
            # Plot trajectory line
            ax.plot(traj_2d[:, 0], traj_2d[:, 1], '-', color=color,
                   linewidth=2, alpha=0.7, label=tokens[tok_idx][:15])

            # Plot layer points
            for layer_idx in range(n_layers):
                marker_size = 50 + layer_idx * 20  # Bigger for later layers
                ax.scatter(traj_2d[layer_idx, 0], traj_2d[layer_idx, 1],
                          c=[color], s=marker_size, alpha=0.9,
                          edgecolors='white', linewidths=0.5, zorder=5)

            # Start marker (triangle)
            ax.scatter(traj_2d[0, 0], traj_2d[0, 1], marker='^',
                      c=[color], s=150, edgecolors='white', linewidths=2, zorder=10)

            # End marker (star)
            ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], marker='*',
                      c=[color], s=200, edgecolors='white', linewidths=1, zorder=10)

        ax.set_xlabel('PC1', color=self.config.text_color, fontsize=12)
        ax.set_ylabel('PC2', color=self.config.text_color, fontsize=12)
        ax.set_title(f'{title}\n(△=Layer 0, ★=Final Layer)',
                    color=self.config.text_color, fontsize=14, pad=20)
        ax.tick_params(colors=self.config.text_color)
        ax.grid(True, color=self.config.grid_color, alpha=0.3, linestyle='--')

        ax.legend(facecolor=self.config.figure_bg_color,
                 edgecolor=self.config.grid_color,
                 labelcolor=self.config.text_color,
                 loc='upper right', fontsize=8)

        return self._fig_to_image(fig)


class EntropyVisualizer:
    """Specialized visualizer for entropy and prediction analysis."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.layer_viz = LayerVisualizer(config)
        plt.style.use('dark_background')

    def create_entropy_comparison(
        self,
        original_entropy: np.ndarray,
        modified_entropy: np.ndarray,
        tokens: List[str],
        title: str = "Entropy: Original vs With Control Vector"
    ) -> Image.Image:
        """
        Compare entropy before and after control vector.

        Args:
            original_entropy: Shape [num_layers, seq_len]
            modified_entropy: Shape [num_layers, seq_len]
            tokens: Token strings
            title: Plot title

        Returns:
            PIL Image of comparison
        """
        return self.layer_viz.create_control_vector_comparison(
            original_entropy,
            modified_entropy,
            tokens,
            title
        )

    def create_confidence_trajectory(
        self,
        entropy_values: np.ndarray,
        tokens: List[str],
        highlight_positions: Optional[List[int]] = None,
        title: str = "Model Confidence Through Layers"
    ) -> Image.Image:
        """
        Show how model confidence (inverse entropy) changes through layers.

        Args:
            entropy_values: Shape [num_layers, seq_len]
            tokens: Token strings
            highlight_positions: Token positions to highlight
            title: Plot title

        Returns:
            PIL Image of confidence trajectory
        """
        num_layers, seq_len = entropy_values.shape

        # Convert entropy to confidence (normalized)
        max_entropy = np.log(50257)  # GPT-2 vocab size
        confidence = 1 - (entropy_values / max_entropy)

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        # Plot each token's confidence trajectory
        colors = plt.cm.rainbow(np.linspace(0, 1, seq_len))

        for i in range(seq_len):
            alpha = 0.9 if highlight_positions is None or i in highlight_positions else 0.3
            linewidth = 2.5 if highlight_positions is None or i in highlight_positions else 1

            ax.plot(range(num_layers), confidence[:, i],
                   color=colors[i], alpha=alpha, linewidth=linewidth,
                   label=tokens[i] if i < len(tokens) else f'pos_{i}')

        ax.set_xlabel('Layer', color=self.config.text_color, fontsize=12)
        ax.set_ylabel('Confidence', color=self.config.text_color, fontsize=12)
        ax.set_title(title, color=self.config.text_color, fontsize=14, pad=20)
        ax.tick_params(colors=self.config.text_color)
        ax.grid(True, color=self.config.grid_color, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1)

        # Legend (only show first 10 tokens to avoid clutter)
        if seq_len <= 10:
            ax.legend(facecolor=self.config.figure_bg_color,
                     edgecolor=self.config.grid_color,
                     labelcolor=self.config.text_color,
                     loc='upper left', bbox_to_anchor=(1.02, 1))

        plt.tight_layout()
        return self.layer_viz._fig_to_image(fig)

    def create_prediction_evolution(
        self,
        layer_predictions: List[List[Dict]],
        token_idx: int,
        title: str = "Top Predictions Evolution"
    ) -> Image.Image:
        """
        Show how top predictions for a token evolve through layers.

        Args:
            layer_predictions: List of predictions per layer, each containing token predictions
            token_idx: Which token position to visualize
            title: Plot title

        Returns:
            PIL Image showing prediction evolution
        """
        num_layers = len(layer_predictions)
        k = 5  # Show top 5 predictions

        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        # Collect top predictions across layers
        all_top_tokens = set()
        layer_data = []

        for layer_idx, layer_preds in enumerate(layer_predictions):
            if token_idx < len(layer_preds):
                pred = layer_preds[token_idx]
                top_tokens = pred['top_k_tokens'][:k]
                top_probs = pred['top_k_probs'][:k]
                all_top_tokens.update(top_tokens)
                layer_data.append({'tokens': top_tokens, 'probs': top_probs})
            else:
                layer_data.append({'tokens': [], 'probs': []})

        # Track probability of each token across layers
        token_trajectories = {token: [] for token in all_top_tokens}

        for layer_idx, data in enumerate(layer_data):
            token_to_prob = dict(zip(data['tokens'], data['probs']))
            for token in all_top_tokens:
                token_trajectories[token].append(token_to_prob.get(token, 0))

        # Plot trajectories for most common tokens
        colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(all_top_tokens))))

        for i, (token, probs) in enumerate(sorted(
            token_trajectories.items(),
            key=lambda x: max(x[1]),
            reverse=True
        )[:10]):
            ax.plot(range(num_layers), probs, 'o-',
                   color=colors[i], linewidth=2, markersize=6,
                   label=f'"{token}"', alpha=0.8)

        ax.set_xlabel('Layer', color=self.config.text_color, fontsize=12)
        ax.set_ylabel('Probability', color=self.config.text_color, fontsize=12)
        ax.set_title(f"{title} (Token Position {token_idx})",
                    color=self.config.text_color, fontsize=14, pad=20)
        ax.tick_params(colors=self.config.text_color)
        ax.grid(True, color=self.config.grid_color, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1)

        ax.legend(facecolor=self.config.figure_bg_color,
                 edgecolor=self.config.grid_color,
                 labelcolor=self.config.text_color,
                 loc='upper left', bbox_to_anchor=(1.02, 1))

        plt.tight_layout()
        return self.layer_viz._fig_to_image(fig)


class DictionaryVisualizer:
    """Visualizer for dictionary learning / sparse autoencoder analysis."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        plt.style.use('dark_background')

    def _fig_to_image(self, fig: plt.Figure) -> Image.Image:
        """Convert figure to PIL Image."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight',
                   facecolor=self.config.figure_bg_color, dpi=150)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img.load()  # Force load image data into memory
        return img.copy()  # Return a copy that doesn't depend on the buffer

    def create_feature_activation_heatmap(
        self,
        feature_activations: np.ndarray,
        tokens: List[str],
        feature_labels: Optional[List[str]] = None,
        title: str = "Feature Activations"
    ) -> Image.Image:
        """
        Create heatmap of which features activate for which tokens.

        Args:
            feature_activations: Shape [seq_len, num_features] or sparse representation
            tokens: Token strings
            feature_labels: Optional labels for features
            title: Plot title

        Returns:
            PIL Image of heatmap
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        # Show only top active features
        max_activations = feature_activations.max(axis=0)
        top_feature_indices = np.argsort(max_activations)[-50:]  # Top 50 features

        data = feature_activations[:, top_feature_indices].T

        colors = ['#1a1a2e', '#3d5a80', '#98c1d9', '#ee6c4d', '#f7b538']
        cmap = LinearSegmentedColormap.from_list('features', colors, N=256)

        im = ax.imshow(data, aspect='auto', cmap=cmap)

        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Activation', color=self.config.text_color)
        cbar.ax.yaxis.set_tick_params(color=self.config.text_color)

        ax.set_xlabel('Token', color=self.config.text_color, fontsize=12)
        ax.set_ylabel('Feature Index', color=self.config.text_color, fontsize=12)
        ax.set_title(title, color=self.config.text_color, fontsize=14, pad=20)

        if len(tokens) <= 15:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')

        ax.tick_params(colors=self.config.text_color)

        return self._fig_to_image(fig)

    def create_feature_comparison(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
        labels: Tuple[str, str],
        title: str = "Feature Activation Comparison"
    ) -> Image.Image:
        """
        Compare feature activations between two inputs.

        Args:
            features1: Feature activations for first input [num_features]
            features2: Feature activations for second input [num_features]
            labels: Labels for the two inputs
            title: Plot title

        Returns:
            PIL Image of comparison
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.patch.set_facecolor(self.config.figure_bg_color)

        # Find top features from both
        combined = np.maximum(features1, features2)
        top_indices = np.argsort(combined)[-30:]

        f1 = features1[top_indices]
        f2 = features2[top_indices]

        x = np.arange(len(top_indices))
        width = 0.35

        # Plot 1: Side by side bars
        ax = axes[0]
        ax.set_facecolor(self.config.figure_bg_color)
        ax.bar(x - width/2, f1, width, label=labels[0], color='#00ffff', alpha=0.8)
        ax.bar(x + width/2, f2, width, label=labels[1], color='#ff6b35', alpha=0.8)
        ax.set_xlabel('Feature Index', color=self.config.text_color)
        ax.set_ylabel('Activation', color=self.config.text_color)
        ax.set_title('Feature Activations', color=self.config.text_color)
        ax.legend(facecolor=self.config.figure_bg_color, labelcolor=self.config.text_color)
        ax.tick_params(colors=self.config.text_color)

        # Plot 2: Difference
        ax = axes[1]
        ax.set_facecolor(self.config.figure_bg_color)
        diff = f2 - f1
        colors = ['#ff6b35' if d > 0 else '#00ffff' for d in diff]
        ax.bar(x, diff, color=colors, alpha=0.8)
        ax.axhline(y=0, color=self.config.text_color, linestyle='-', linewidth=0.5)
        ax.set_xlabel('Feature Index', color=self.config.text_color)
        ax.set_ylabel('Δ Activation', color=self.config.text_color)
        ax.set_title(f'Difference ({labels[1]} - {labels[0]})', color=self.config.text_color)
        ax.tick_params(colors=self.config.text_color)

        # Plot 3: Scatter
        ax = axes[2]
        ax.set_facecolor(self.config.figure_bg_color)
        ax.scatter(features1, features2, alpha=0.5, color='#ffd700', s=20)
        max_val = max(features1.max(), features2.max())
        ax.plot([0, max_val], [0, max_val], '--', color=self.config.text_color, alpha=0.5)
        ax.set_xlabel(f'{labels[0]} Activation', color=self.config.text_color)
        ax.set_ylabel(f'{labels[1]} Activation', color=self.config.text_color)
        ax.set_title('Activation Correlation', color=self.config.text_color)
        ax.tick_params(colors=self.config.text_color)

        fig.suptitle(title, color=self.config.text_color, fontsize=14, y=1.02)
        plt.tight_layout()

        return self._fig_to_image(fig)

    def create_3d_feature_manifold(
        self,
        feature_activations: np.ndarray,
        tokens: List[str],
        title: str = "3D Feature Activation Landscape"
    ) -> Image.Image:
        """
        Create 3D surface plot of feature activations.

        Args:
            feature_activations: Shape [seq_len, num_features]
            tokens: Token strings
            title: Plot title

        Returns:
            PIL Image of 3D manifold
        """
        # Get top active features for visualization
        max_activations = feature_activations.max(axis=0)
        top_feature_indices = np.argsort(max_activations)[-40:]  # Top 40 features

        data = feature_activations[:, top_feature_indices]
        seq_len, num_features = data.shape

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        # Create meshgrid
        X, Y = np.meshgrid(range(num_features), range(seq_len))
        Z = data

        # Custom colormap for features
        colors = ['#1a1a2e', '#3d5a80', '#98c1d9', '#ee6c4d', '#f7b538', '#ff1493']
        cmap = LinearSegmentedColormap.from_list('features3d', colors, N=256)

        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85,
                               linewidth=0.3, antialiased=True,
                               edgecolor='white')

        # Add contour on the bottom
        ax.contour(X, Y, Z, zdir='z', offset=Z.min() - 0.1, cmap=cmap, alpha=0.5)

        # Highlight peaks (highly active features)
        threshold = np.percentile(Z, 95)
        peak_mask = Z > threshold
        if peak_mask.any():
            peak_x = X[peak_mask]
            peak_y = Y[peak_mask]
            peak_z = Z[peak_mask]
            ax.scatter(peak_x, peak_y, peak_z, c='#ff1493', s=50,
                      marker='*', alpha=0.9, label='High Activation')

        # Colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label('Feature Activation', color=self.config.text_color)
        cbar.ax.yaxis.set_tick_params(color=self.config.text_color)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.config.text_color)

        # Labels
        ax.set_xlabel('Feature Index', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_ylabel('Token Position', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_zlabel('Activation', color=self.config.text_color, fontsize=11, labelpad=10)
        ax.set_title(title, color=self.config.text_color, fontsize=14, pad=20)

        # Set viewing angle
        ax.view_init(elev=25, azim=45)

        # Token labels on y-axis if few tokens
        if len(tokens) <= 10:
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=8)

        ax.tick_params(colors=self.config.text_color)
        ax.xaxis.pane.set_facecolor('black')
        ax.yaxis.pane.set_facecolor('black')
        ax.zaxis.pane.set_facecolor('black')
        ax.xaxis.pane.set_edgecolor(self.config.grid_color)
        ax.yaxis.pane.set_edgecolor(self.config.grid_color)
        ax.zaxis.pane.set_edgecolor(self.config.grid_color)

        if peak_mask.any():
            ax.legend(facecolor=self.config.figure_bg_color,
                     labelcolor=self.config.text_color, loc='upper right')

        return self._fig_to_image(fig)

    def create_feature_space_3d(
        self,
        feature_activations: np.ndarray,
        tokens: List[str],
        title: str = "Token Embeddings in Feature Space"
    ) -> Image.Image:
        """
        Project tokens into 3D feature space using PCA on sparse features.

        Args:
            feature_activations: Shape [seq_len, num_features]
            tokens: Token strings
            title: Plot title

        Returns:
            PIL Image of 3D scatter
        """
        from sklearn.decomposition import PCA

        # PCA to 3D
        pca = PCA(n_components=3)
        tokens_3d = pca.fit_transform(feature_activations)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        # Color by position in sequence
        colors = plt.cm.rainbow(np.linspace(0, 1, len(tokens)))

        # Plot points
        for i, (point, token, color) in enumerate(zip(tokens_3d, tokens, colors)):
            ax.scatter(point[0], point[1], point[2], c=[color], s=150,
                      alpha=0.8, edgecolors='white', linewidths=1)
            ax.text(point[0], point[1], point[2] + 0.1, token[:8],
                   color=self.config.text_color, fontsize=9, ha='center')

        # Connect sequential tokens with lines
        for i in range(len(tokens_3d) - 1):
            ax.plot([tokens_3d[i, 0], tokens_3d[i+1, 0]],
                   [tokens_3d[i, 1], tokens_3d[i+1, 1]],
                   [tokens_3d[i, 2], tokens_3d[i+1, 2]],
                   color='#ffd700', alpha=0.5, linewidth=1.5)

        # Labels
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                     color=self.config.text_color, fontsize=11)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                     color=self.config.text_color, fontsize=11)
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})',
                     color=self.config.text_color, fontsize=11)
        ax.set_title(title, color=self.config.text_color, fontsize=14, pad=20)

        ax.view_init(elev=20, azim=45)
        ax.tick_params(colors=self.config.text_color)
        ax.xaxis.pane.set_facecolor('black')
        ax.yaxis.pane.set_facecolor('black')
        ax.zaxis.pane.set_facecolor('black')
        ax.xaxis.pane.set_edgecolor(self.config.grid_color)
        ax.yaxis.pane.set_edgecolor(self.config.grid_color)
        ax.zaxis.pane.set_edgecolor(self.config.grid_color)

        return self._fig_to_image(fig)

    def create_feature_correlation_network(
        self,
        feature_activations: np.ndarray,
        tokens: List[str],
        title: str = "Feature Co-activation Network"
    ) -> Image.Image:
        """
        Visualize which features tend to activate together as a network graph.

        Args:
            feature_activations: Shape [seq_len, num_features]
            tokens: Token strings
            title: Plot title

        Returns:
            PIL Image of network
        """
        # Get top active features
        max_activations = feature_activations.max(axis=0)
        top_indices = np.argsort(max_activations)[-20:]  # Top 20 features

        top_activations = feature_activations[:, top_indices]

        # Compute correlation matrix
        correlations = np.corrcoef(top_activations.T)

        fig, ax = plt.subplots(figsize=(14, 14))
        fig.patch.set_facecolor(self.config.figure_bg_color)
        ax.set_facecolor(self.config.figure_bg_color)

        num_features = len(top_indices)

        # Circular layout
        angles = np.linspace(0, 2 * np.pi, num_features, endpoint=False)
        radius = 0.4
        center = (0.5, 0.5)

        node_x = center[0] + radius * np.cos(angles - np.pi/2)
        node_y = center[1] + radius * np.sin(angles - np.pi/2)

        # Draw edges based on correlation
        for i in range(num_features):
            for j in range(i + 1, num_features):
                corr = correlations[i, j]
                if abs(corr) > 0.3:  # Only significant correlations
                    alpha = min(abs(corr), 1.0)
                    linewidth = abs(corr) * 4
                    color = '#00ffff' if corr > 0 else '#ff6b35'

                    ax.plot([node_x[i], node_x[j]], [node_y[i], node_y[j]],
                           color=color, alpha=alpha * 0.7, linewidth=linewidth)

        # Draw nodes
        for i, feat_idx in enumerate(top_indices):
            # Node size based on max activation
            size = 200 + 800 * (max_activations[feat_idx] / max_activations.max())

            ax.scatter(node_x[i], node_y[i], s=size, c='#2d3561',
                      edgecolors='#00ffff', linewidths=2, zorder=10)
            ax.text(node_x[i], node_y[i], f'F{feat_idx}', color=self.config.text_color,
                   fontsize=8, ha='center', va='center', fontweight='bold', zorder=11)

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], color='#00ffff', linewidth=3, label='Positive correlation'),
            plt.Line2D([0], [0], color='#ff6b35', linewidth=3, label='Negative correlation')
        ]
        ax.legend(handles=legend_elements, loc='lower right',
                 facecolor=self.config.figure_bg_color,
                 edgecolor=self.config.grid_color,
                 labelcolor=self.config.text_color)

        ax.set_title(title, color=self.config.text_color, fontsize=14, pad=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        return self._fig_to_image(fig)

    def create_monosemantic_analysis(
        self,
        feature_examples: Dict[int, List[Tuple[str, float]]],
        title: str = "Feature Semantic Analysis"
    ) -> Image.Image:
        """
        Visualize which concepts each feature responds to.

        Args:
            feature_examples: Dict mapping feature_idx to list of (text, activation) tuples
            title: Plot title

        Returns:
            PIL Image of analysis
        """
        num_features = min(10, len(feature_examples))

        fig, axes = plt.subplots(num_features, 1, figsize=(14, num_features * 1.5))
        fig.patch.set_facecolor(self.config.figure_bg_color)

        if num_features == 1:
            axes = [axes]

        sorted_features = sorted(
            feature_examples.items(),
            key=lambda x: max(a for _, a in x[1]) if x[1] else 0,
            reverse=True
        )[:num_features]

        for ax, (feat_idx, examples) in zip(axes, sorted_features):
            ax.set_facecolor(self.config.figure_bg_color)

            if examples:
                texts = [t[:30] + '...' if len(t) > 30 else t for t, _ in examples[:5]]
                activations = [a for _, a in examples[:5]]

                y_pos = np.arange(len(texts))
                ax.barh(y_pos, activations, color='#00ffff', alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(texts, fontsize=8)
                ax.set_xlabel('Activation', color=self.config.text_color, fontsize=9)

            ax.set_title(f'Feature {feat_idx}', color=self.config.text_color, fontsize=10, loc='left')
            ax.tick_params(colors=self.config.text_color)

        fig.suptitle(title, color=self.config.text_color, fontsize=14, y=1.02)
        plt.tight_layout()

        return self._fig_to_image(fig)

    def create_interactive_feature_network(
        self,
        feature_info: Dict,
        neuron_mapping: Dict[int, List[tuple]],
        selected_feature: int = 0,
        title: str = "Feature-Neuron Network",
        hidden_dim: int = 768,
        num_layers: int = 4
    ):
        """
        Create an interactive Plotly network graph showing the path from input to output,
        highlighting which neurons are active for the selected feature at each layer.

        Args:
            feature_info: Dict with feature semantic descriptions and neuron compositions
            neuron_mapping: Dict mapping neuron_idx to list of (feature_idx, weight) tuples
            selected_feature: The feature index to highlight
            title: Plot title
            hidden_dim: Hidden dimension of the model
            num_layers: Number of conceptual layers to display

        Returns:
            Plotly Figure object for interactive display
        """
        import plotly.graph_objects as go

        # Get feature data
        if isinstance(feature_info, list):
            features_data = feature_info
        else:
            features_data = [feature_info]

        # Find selected feature data
        selected_feat_data = None
        for feat in features_data:
            if feat.get('feature_idx', -1) == selected_feature:
                selected_feat_data = feat
                break

        if not selected_feat_data:
            selected_feat_data = features_data[0] if features_data else {}

        # Get top neurons for this feature
        top_neuron_indices = selected_feat_data.get('top_neuron_indices', [])[:20]
        top_neuron_weights = selected_feat_data.get('top_neuron_weights', [])[:20]

        # Create layer structure - distribute neurons across conceptual layers
        # Layer 0: Input representation
        # Layers 1-2: Hidden processing
        # Layer 3: Output/Feature
        neurons_per_layer = max(8, len(top_neuron_indices) // num_layers + 1)

        # Create figure
        fig = go.Figure()

        # Track all node positions for connections
        layer_nodes = {i: [] for i in range(num_layers + 1)}  # +1 for feature output

        # Layer colors
        layer_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

        # Position neurons in each layer
        layer_x_positions = [0, 1.5, 3, 4.5, 6]  # X positions for each layer
        layer_names = ['Input', 'Early Hidden', 'Mid Hidden', 'Late Hidden', 'Feature Output']

        # Distribute top neurons across hidden layers based on their index
        # (simulating different parts of the model)
        for layer_idx in range(num_layers):
            layer_x = layer_x_positions[layer_idx]

            # Calculate which neurons go in this layer
            start_idx = layer_idx * neurons_per_layer
            end_idx = min(start_idx + neurons_per_layer, len(top_neuron_indices))
            layer_neuron_subset = list(range(start_idx, end_idx))

            # Add neurons for this layer
            for i, neuron_list_idx in enumerate(layer_neuron_subset):
                if neuron_list_idx >= len(top_neuron_indices):
                    continue

                neuron_idx = top_neuron_indices[neuron_list_idx]
                weight = top_neuron_weights[neuron_list_idx] if neuron_list_idx < len(top_neuron_weights) else 0

                y_pos = (i - len(layer_neuron_subset) / 2) * 1.2

                # Determine color based on weight magnitude
                weight_magnitude = abs(weight) if weight else 0
                max_weight = max(abs(w) for w in top_neuron_weights) if top_neuron_weights else 1

                # Size based on weight
                node_size = 15 + 25 * (weight_magnitude / max_weight)

                # Color based on weight sign and magnitude
                if weight > 0:
                    color_intensity = min(255, int(100 + 155 * (weight_magnitude / max_weight)))
                    node_color = f'rgb(0, {color_intensity}, 100)'  # Green tones
                else:
                    color_intensity = min(255, int(100 + 155 * (weight_magnitude / max_weight)))
                    node_color = f'rgb({color_intensity}, 50, 50)'  # Red tones

                layer_nodes[layer_idx].append({
                    'x': layer_x,
                    'y': y_pos,
                    'neuron_idx': neuron_idx,
                    'weight': weight,
                    'size': node_size,
                    'color': node_color
                })

        # Add the feature output node
        feature_y = 0
        layer_nodes[num_layers].append({
            'x': layer_x_positions[num_layers],
            'y': feature_y,
            'neuron_idx': f'F{selected_feature}',
            'weight': 1.0,
            'size': 40,
            'color': '#ff1493'  # Hot pink for the feature
        })

        # Draw connections between layers (the "path")
        for layer_idx in range(num_layers):
            current_layer = layer_nodes[layer_idx]
            next_layer = layer_nodes[layer_idx + 1]

            for curr_node in current_layer:
                for next_node in next_layer:
                    # Connection strength based on combined weights
                    weight1 = abs(curr_node.get('weight', 0.5))
                    weight2 = abs(next_node.get('weight', 0.5))
                    conn_strength = (weight1 + weight2) / 2

                    # Line width and opacity based on connection strength
                    max_weight = max(abs(w) for w in top_neuron_weights) if top_neuron_weights else 1
                    normalized_strength = conn_strength / max_weight if max_weight > 0 else 0.5

                    fig.add_trace(go.Scatter(
                        x=[curr_node['x'], next_node['x']],
                        y=[curr_node['y'], next_node['y']],
                        mode='lines',
                        line=dict(
                            width=1 + 3 * normalized_strength,
                            color=f'rgba(255, 215, 0, {0.2 + 0.5 * normalized_strength})'  # Gold connections
                        ),
                        hoverinfo='none',
                        showlegend=False
                    ))

        # Draw nodes for each layer
        for layer_idx in range(num_layers + 1):
            layer = layer_nodes[layer_idx]
            if not layer:
                continue

            x_vals = [n['x'] for n in layer]
            y_vals = [n['y'] for n in layer]
            sizes = [n['size'] for n in layer]
            colors = [n['color'] for n in layer]
            labels = [f"N{n['neuron_idx']}<br>W: {n['weight']:.3f}" if isinstance(n['neuron_idx'], int)
                     else f"{n['neuron_idx']}" for n in layer]

            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers+text',
                marker=dict(
                    size=sizes,
                    color=colors,
                    line=dict(width=2, color='white'),
                    symbol='circle' if layer_idx < num_layers else 'diamond'
                ),
                text=[str(n['neuron_idx']) if isinstance(n['neuron_idx'], int) else n['neuron_idx'] for n in layer],
                textposition='middle center',
                textfont=dict(color='white', size=8),
                hovertext=labels,
                hoverinfo='text',
                name=layer_names[layer_idx],
                showlegend=True
            ))

        # Add layer labels at the top
        for layer_idx, (x_pos, name) in enumerate(zip(layer_x_positions, layer_names)):
            fig.add_annotation(
                x=x_pos,
                y=max(n['y'] for layer in layer_nodes.values() for n in layer) + 1.5 if any(layer_nodes.values()) else 2,
                text=f"<b>{name}</b>",
                showarrow=False,
                font=dict(color=layer_colors[layer_idx % len(layer_colors)], size=12)
            )

        # Add flow arrows
        for i in range(len(layer_x_positions) - 1):
            mid_x = (layer_x_positions[i] + layer_x_positions[i + 1]) / 2
            fig.add_annotation(
                x=mid_x,
                y=max(n['y'] for layer in layer_nodes.values() for n in layer) + 2.5 if any(layer_nodes.values()) else 3,
                text="→",
                showarrow=False,
                font=dict(color='#ffd700', size=20)
            )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{title}<br><sub>Feature {selected_feature} - Neuron activation path through model</sub>",
                font=dict(color='white', size=16)
            ),
            plot_bgcolor='#0d0d1a',
            paper_bgcolor='#0d0d1a',
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-0.5, 7]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False
            ),
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.15,
                xanchor='center',
                x=0.5,
                font=dict(color='white', size=10),
                bgcolor='rgba(0,0,0,0.5)'
            ),
            hovermode='closest',
            margin=dict(l=20, r=20, t=80, b=60)
        )

        return fig

    def create_neural_network_diagram(
        self,
        feature_info: Dict,
        selected_feature: int = 0,
        model=None,
        model_name: str = "gpt2",
        layer_activations: Dict = None,
        title: str = "Neural Network Architecture"
    ):
        """
        Create a dynamic neural network diagram showing which neurons activate
        at EACH layer for the selected semantic feature.

        Args:
            feature_info: Dict with feature data including top_neuron_indices and weights
            selected_feature: The feature index being displayed
            model: The actual model object (GPT2ControlModel, BertControlModel, etc.)
            model_name: Name of the model ("gpt2", "bert", etc.)
            layer_activations: Dict from get_feature_activations_per_layer() mapping
                              layer_idx -> {top_neurons, feature_activation, ...}
            title: Plot title

        Returns:
            Plotly Figure object for interactive display
        """
        import plotly.graph_objects as go

        # =====================================================================
        # DYNAMICALLY READ MODEL ARCHITECTURE
        # =====================================================================
        if model is not None:
            num_layers = getattr(model, 'num_layers', 12)
            hidden_dim = getattr(model, 'hidden_size', 768)
            vocab_size = getattr(model, 'vocab_size', 50257)
        else:
            num_layers = 12
            hidden_dim = 768
            vocab_size = 50257

        # Handle feature_info
        if isinstance(feature_info, list):
            selected_feat_data = None
            for feat in feature_info:
                if feat.get('feature_idx', -1) == selected_feature:
                    selected_feat_data = feat
                    break
            if not selected_feat_data:
                selected_feat_data = feature_info[0] if feature_info else {}
        else:
            selected_feat_data = feature_info

        # Create figure
        fig = go.Figure()

        # =====================================================================
        # LAYOUT: Show all layers with per-layer activations
        # =====================================================================
        n_display_neurons = 48  # Neurons shown per layer
        layer_spacing = max(0.6, 12.0 / (num_layers + 3))
        neuron_spacing = 0.1

        # Build layer structure: [Emb] -> [L0] -> ... -> [L{n-1}] -> [Head] -> [Feature]
        layers = []
        layers.append({'name': 'Emb', 'type': 'embed', 'neurons': 12})
        for i in range(num_layers):
            layers.append({
                'name': f'L{i}',
                'type': 'transformer',
                'layer_idx': i,
                'neurons': n_display_neurons
            })
        layers.append({'name': 'Out', 'type': 'head', 'neurons': 8})
        layers.append({'name': f'F{selected_feature}', 'type': 'feature', 'neurons': 1})

        layer_x = [i * layer_spacing for i in range(len(layers))]

        # Batch data for efficient rendering
        all_x, all_y, all_colors, all_sizes, all_hovers = [], [], [], [], []

        # Track active neuron positions for connection drawing
        active_positions = []

        # =====================================================================
        # GET PER-LAYER ACTIVATION DATA
        # =====================================================================
        if layer_activations is None:
            layer_activations = {}

        # Find global max activation for normalization
        global_max = 0.001
        for layer_idx, layer_data in layer_activations.items():
            for neuron_idx, activation in layer_data.get('top_neurons', []):
                global_max = max(global_max, abs(activation))

        # =====================================================================
        # DRAW EACH LAYER with per-layer activations
        # =====================================================================

        for display_idx, layer_info in enumerate(layers):
            x_pos = layer_x[display_idx]
            layer_type = layer_info['type']
            n_neurons = layer_info['neurons']

            if layer_type == 'embed':
                for i in range(n_neurons):
                    y_pos = (i - n_neurons / 2) * neuron_spacing * 2
                    all_x.append(x_pos)
                    all_y.append(y_pos)
                    all_colors.append('#3498db')
                    all_sizes.append(5)
                    all_hovers.append(f'Embedding dim {i}')

            elif layer_type == 'head':
                for i in range(n_neurons):
                    y_pos = (i - n_neurons / 2) * neuron_spacing * 2
                    all_x.append(x_pos)
                    all_y.append(y_pos)
                    all_colors.append('#f39c12')
                    all_sizes.append(4)
                    all_hovers.append(f'Output dim {i}')

            elif layer_type == 'feature':
                y_pos = 0
                all_x.append(x_pos)
                all_y.append(y_pos)
                all_colors.append('#ff1493')
                all_sizes.append(18)
                desc = selected_feat_data.get('description', '')[:35]
                all_hovers.append(f'<b>Feature {selected_feature}</b><br>{desc}')

            elif layer_type == 'transformer':
                trans_layer_idx = layer_info['layer_idx']

                # Get activation data for this layer
                layer_data = layer_activations.get(trans_layer_idx, {})
                top_neurons = layer_data.get('top_neurons', [])
                active_neuron_set = set(idx for idx, _ in top_neurons[:15])
                neuron_weights = {idx: w for idx, w in top_neurons}
                feature_activation = layer_data.get('feature_activation', 0)

                for i in range(n_neurons):
                    y_pos = (i - n_neurons / 2) * neuron_spacing

                    # Map display position to actual neuron index (0-767)
                    actual_neuron_idx = int(i * hidden_dim / n_neurons)

                    # Check if this neuron is active AT THIS LAYER
                    is_active = actual_neuron_idx in active_neuron_set
                    weight = neuron_weights.get(actual_neuron_idx, 0)

                    if is_active and abs(weight) > 0.001:
                        # ACTIVE neuron at this layer - VIVID colors
                        norm_weight = min(1.0, abs(weight) / global_max)
                        if weight > 0:
                            # Bright neon green with full opacity
                            color = f'rgba(0, 255, 100, {0.75 + 0.25 * norm_weight})'
                        else:
                            # Bright neon red/magenta with full opacity
                            color = f'rgba(255, 50, 100, {0.75 + 0.25 * norm_weight})'
                        size = 8 + 12 * norm_weight  # Larger sizes for visibility
                        hover = f'<b>🔥 N{actual_neuron_idx}</b><br>L{trans_layer_idx}<br>Act: {weight:.4f}'

                        # Track for connection drawing
                        active_positions.append((x_pos, y_pos, weight, trans_layer_idx))
                    else:
                        # Inactive neuron
                        # Slightly highlight layers with high feature activation
                        base_brightness = 0.08 + 0.05 * min(1, feature_activation)
                        color = f'rgba(40, 40, 60, {base_brightness})'
                        size = 2
                        hover = f'N{actual_neuron_idx} L{trans_layer_idx}'

                    all_x.append(x_pos)
                    all_y.append(y_pos)
                    all_colors.append(color)
                    all_sizes.append(size)
                    all_hovers.append(hover)

        # =====================================================================
        # DRAW ALL NEURONS
        # =====================================================================
        fig.add_trace(go.Scatter(
            x=all_x, y=all_y,
            mode='markers',
            marker=dict(size=all_sizes, color=all_colors, line=dict(width=0)),
            hovertext=all_hovers,
            hoverinfo='text',
            showlegend=False
        ))

        # =====================================================================
        # DRAW CONNECTIONS from active neurons to feature
        # =====================================================================
        feature_x = layer_x[-1]
        feature_y = 0

        for (x, y, weight, layer_idx) in active_positions:
            norm_w = min(1.0, abs(weight) / global_max)
            if weight > 0:
                # Vivid neon green lines
                line_color = f'rgba(0, 255, 100, {0.4 + 0.5 * norm_w})'
            else:
                # Vivid neon red/magenta lines
                line_color = f'rgba(255, 50, 100, {0.4 + 0.5 * norm_w})'

            fig.add_trace(go.Scatter(
                x=[x, feature_x], y=[y, feature_y],
                mode='lines',
                line=dict(width=1.5 + 2.5 * norm_w, color=line_color),
                hoverinfo='none',
                showlegend=False
            ))

        # =====================================================================
        # LAYER LABELS
        # =====================================================================
        max_y = n_display_neurons * neuron_spacing / 2 + 0.5
        for display_idx, layer_info in enumerate(layers):
            x_pos = layer_x[display_idx]
            name = layer_info['name']
            ltype = layer_info['type']

            # Color and highlight based on feature activation at this layer
            if ltype == 'embed':
                color = '#3498db'
            elif ltype == 'head':
                color = '#f39c12'
            elif ltype == 'feature':
                color = '#ff1493'
            else:
                # Color label by feature activation strength at this layer
                layer_data = layer_activations.get(layer_info['layer_idx'], {})
                feat_act = layer_data.get('feature_activation', 0)
                if feat_act > 0.5:
                    color = '#00ff88'
                elif feat_act > 0.1:
                    color = '#88aa88'
                else:
                    color = '#555'

            fig.add_annotation(
                x=x_pos, y=max_y,
                text=f"<b>{name}</b>",
                showarrow=False,
                font=dict(color=color, size=7)
            )

        # =====================================================================
        # LEGEND
        # =====================================================================
        legend_y = -n_display_neurons * neuron_spacing / 2 - 0.5
        legend_items = [
            ('+Activation', '#00ff64'),
            ('-Activation', '#ff3264'),
            ('Inactive', '#282840'),
        ]

        for i, (name, color) in enumerate(legend_items):
            fig.add_trace(go.Scatter(
                x=[i * 2.0], y=[legend_y],
                mode='markers+text',
                marker=dict(size=7, color=color),
                text=[f' {name}'],
                textposition='middle right',
                textfont=dict(color='#666', size=7),
                hoverinfo='none',
                showlegend=False
            ))

        # =====================================================================
        # LAYOUT
        # =====================================================================
        desc = selected_feat_data.get('description', '')[:45]
        model_label = model_name.upper()
        n_active_layers = sum(1 for l in layer_activations.values() if l.get('feature_activation', 0) > 0.1)

        fig.update_layout(
            title=dict(
                text=f"<b>{model_label} - {title}</b><br>"
                     f"<sub>{desc} | Active in {n_active_layers}/{num_layers} layers</sub>",
                font=dict(color='white', size=12),
                x=0.5
            ),
            plot_bgcolor='#08080f',
            paper_bgcolor='#08080f',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                      range=[-0.3, layer_x[-1] + 0.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                      range=[legend_y - 0.3, max_y + 0.4]),
            showlegend=False,
            hovermode='closest',
            margin=dict(l=10, r=10, t=70, b=25),
            height=500
        )

        return fig

    def create_feature_explorer_card(
        self,
        feature_info: Dict
    ) -> str:
        """
        Create an HTML card showing feature information with clustered phrases.

        Args:
            feature_info: Dict with feature semantic description and neuron composition

        Returns:
            HTML string for display
        """
        feat_idx = feature_info.get('feature_idx', 0)
        description = feature_info.get('description', 'Unknown')
        confidence = feature_info.get('confidence', 0) * 100
        pattern = feature_info.get('activation_pattern', 'sparse')
        active_neurons = feature_info.get('active_neurons', 0)
        total_neurons = feature_info.get('total_neurons', 0)
        sparsity = feature_info.get('sparsity', 0) * 100
        clustered_phrases = feature_info.get('clustered_phrases', [])
        activating_phrases = feature_info.get('activating_phrases', [])

        # Pattern badge color
        pattern_colors = {
            'strongly_active': '#00ff00',
            'moderately_active': '#ffd700',
            'sparse': '#ff6b35'
        }
        pattern_color = pattern_colors.get(pattern, '#888888')

        html = f"""
        <div style="background: #1a1a2e; border-radius: 10px; padding: 15px; margin: 10px 0; border: 1px solid #4a4a5e;">
            <h3 style="color: #00ffff; margin: 0 0 10px 0;">Feature {feat_idx}</h3>
            <p style="color: #e0e0ff; font-size: 14px; margin: 5px 0;">
                <strong>Semantic Meaning:</strong> {description}
            </p>
            <div style="display: flex; gap: 20px; margin: 10px 0;">
                <span style="color: {pattern_color}; font-size: 12px; padding: 2px 8px; border-radius: 4px; background: rgba(255,255,255,0.1);">
                    {pattern.replace('_', ' ').title()}
                </span>
                <span style="color: #ffd700; font-size: 12px;">
                    Confidence: {confidence:.1f}%
                </span>
            </div>
            <hr style="border: none; border-top: 1px solid #4a4a5e; margin: 10px 0;">
            <p style="color: #ff6b35; font-size: 13px; margin: 5px 0;">
                <strong>Neuron Composition:</strong> {active_neurons} / {total_neurons} neurons active
            </p>
            <div style="background: #2d2d4e; border-radius: 4px; height: 8px; margin: 5px 0;">
                <div style="background: linear-gradient(90deg, #00ffff, #ff1493); width: {100-sparsity}%; height: 100%; border-radius: 4px;"></div>
            </div>
            <p style="color: #888; font-size: 11px;">Sparsity: {sparsity:.1f}%</p>
        """

        # Add activating phrases with highlighted tokens
        if activating_phrases:
            html += """
            <hr style="border: none; border-top: 1px solid #4a4a5e; margin: 10px 0;">
            <p style="color: #e0e0ff; font-size: 13px; margin: 5px 0;"><strong>📝 Activating Phrases:</strong></p>
            <p style="color: #888; font-size: 10px; margin: 2px 0;">Most activating token <span style="background: #ff1493; color: white; padding: 1px 4px; border-radius: 3px;">highlighted</span></p>
            """

            for phrase_info in activating_phrases[:6]:
                phrase = phrase_info.get('phrase', '')
                token = phrase_info.get('highlighted_token', '')
                activation = phrase_info.get('activation', 0)

                # Escape HTML
                safe_phrase = phrase.replace('<', '&lt;').replace('>', '&gt;')
                safe_token = token.replace('<', '&lt;').replace('>', '&gt;')

                # Highlight the token in the phrase
                if safe_token and safe_token in safe_phrase:
                    highlighted_phrase = safe_phrase.replace(
                        safe_token,
                        f'<span style="background: #ff1493; color: white; padding: 1px 4px; border-radius: 3px; font-weight: bold;">{safe_token}</span>',
                        1  # Replace only first occurrence
                    )
                else:
                    highlighted_phrase = safe_phrase

                html += f"""
                <div style="background: #252545; padding: 8px 12px; margin: 6px 0; border-radius: 6px; border-left: 3px solid #ff1493;">
                    <p style="color: #ddd; font-size: 12px; margin: 0; line-height: 1.5;">{highlighted_phrase}</p>
                    <div style="display: flex; justify-content: space-between; margin-top: 4px;">
                        <span style="color: #ff1493; font-size: 10px;">Token: "{safe_token}"</span>
                        <span style="color: #ffd700; font-size: 10px;">Activation: {activation:.4f}</span>
                    </div>
                </div>
                """

        # Add clustered phrases section
        if clustered_phrases:
            # Cluster theme colors
            cluster_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

            html += """
            <hr style="border: none; border-top: 1px solid #4a4a5e; margin: 10px 0;">
            <p style="color: #e0e0ff; font-size: 13px; margin: 5px 0;"><strong>🏷️ Semantic Clusters:</strong></p>
            """

            for i, cluster in enumerate(clustered_phrases[:4]):
                theme = cluster.get('theme', 'Unknown')
                phrases = cluster.get('phrases', [])
                avg_activation = cluster.get('avg_activation', 0)
                count = cluster.get('count', len(phrases))
                color = cluster_colors[i % len(cluster_colors)]

                html += f"""
                <div style="background: #252545; border-left: 3px solid {color}; padding: 8px 12px; margin: 8px 0; border-radius: 0 6px 6px 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="color: {color}; font-weight: bold; font-size: 12px;">{theme}</span>
                        <span style="color: #888; font-size: 10px;">{count} examples | Avg: {avg_activation:.3f}</span>
                    </div>
                    <ul style="color: #aaa; font-size: 11px; margin: 5px 0 0 0; padding-left: 15px;">
                """

                for phrase in phrases[:3]:
                    # Escape HTML in phrase
                    safe_phrase = phrase.replace('<', '&lt;').replace('>', '&gt;')
                    html += f"<li style='margin: 2px 0;'>{safe_phrase}</li>"

                html += "</ul></div>"

        html += "</div>"
        return html
