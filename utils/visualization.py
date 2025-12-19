import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import io
from PIL import Image
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple
import torch


class VisualizationConfig:
    """Shared configuration for all visualizations."""
    FIGURE_BG = '#1a1a2e'
    TEXT_COLOR = '#e0e0ff'
    GRID_COLOR = '#4a4a5e'
    ACCENT_COLORS = ['#FF6B35', '#FF1493', '#FFD700', '#00FFFF', '#32CD32', '#FF4500', '#DA70D6', '#00FA9A']


class LatentSpaceVisualizer:
    """Visualizes embeddings in latent space with PCA."""

    def __init__(self, model):
        self.model = model
        self.config = VisualizationConfig()

    def create_pca_plot(self, prompt_groups, labels_dict=None, n_components=2):
        # Set dark theme and figure
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 6))

        # Configure background and axes
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor(self.config.FIGURE_BG)
            # Update 3D pane colors using current API
            ax.xaxis.pane.fill = True
            ax.yaxis.pane.fill = True
            ax.zaxis.pane.fill = True

            ax.xaxis.pane.set_color((0.1, 0.1, 0.2, 0.8))
            ax.yaxis.pane.set_color((0.1, 0.1, 0.2, 0.8))
            ax.zaxis.pane.set_color((0.1, 0.1, 0.2, 0.8))

            # Set grid colors for 3D
            ax.grid(True, color=self.config.GRID_COLOR, linestyle='--', alpha=0.3)
            ax.xaxis.set_pane_color((0.1, 0.1, 0.2, 0.8))
            ax.yaxis.set_pane_color((0.1, 0.1, 0.2, 0.8))
            ax.zaxis.set_pane_color((0.1, 0.1, 0.2, 0.8))

        fig.patch.set_facecolor(self.config.FIGURE_BG)

        # Use bright contrasting colors
        colors = [self.config.ACCENT_COLORS[i % len(self.config.ACCENT_COLORS)]
                  for i in range(len(prompt_groups))]

        all_components = []
        for group_idx, (group_name, category) in enumerate(prompt_groups.items()):
            vectors, labels = [], []

            # Handle new contrast structure
            for contrast in category['contrasts']:
                contrast_keys = list(contrast.keys())
                for idx, (label, examples) in enumerate(contrast.items()):
                    for prompt in examples:
                        hidden = self.model.get_hidden_state(prompt)
                        vector_np = hidden.cpu().detach().numpy()
                        vectors.append(vector_np)
                        labels.append(f"{group_name}-{label}")

            if vectors:
                pca = PCA(n_components=n_components)
                components = pca.fit_transform(vectors)
                all_components.extend(components)

                unique_labels = list(set(labels))
                for label_idx, label in enumerate(unique_labels):
                    indices = [i for i, l in enumerate(labels) if l == label]
                    if n_components == 2:
                        # Plot connecting lines with neon effect
                        plt.plot(
                            components[indices, 0],
                            components[indices, 1],
                            color=colors[group_idx],
                            linestyle='--',
                            alpha=0.4,
                            zorder=1,
                            path_effects=[pe.Stroke(linewidth=2, foreground='white', alpha=0.2)]
                        )
                        plt.scatter(
                            components[indices, 0],
                            components[indices, 1],
                            label=label,
                            color=colors[group_idx],
                            alpha=0.9,
                            marker=['o', 's'][label_idx % 2],
                            zorder=2,
                            edgecolor='white'
                        )
                    else:
                        # 3D plotting with similar styling
                        ax.plot(
                            components[indices, 0],
                            components[indices, 1],
                            components[indices, 2],
                            color=colors[group_idx],
                            linestyle='--',
                            alpha=0.4,
                            zorder=1
                        )
                        ax.scatter(
                            components[indices, 0],
                            components[indices, 1],
                            components[indices, 2],
                            label=label,
                            color=colors[group_idx],
                            alpha=0.9,
                            marker=['o', 's'][label_idx % 2],
                            zorder=2,
                            edgecolor='white'
                        )

        # Style grid and text
        plt.grid(True, color=self.config.GRID_COLOR, linestyle='--', alpha=0.3)
        plt.rcParams.update({
            'axes.labelcolor': self.config.TEXT_COLOR,
            'text.color': self.config.TEXT_COLOR,
            'xtick.color': self.config.TEXT_COLOR,
            'ytick.color': self.config.TEXT_COLOR,
            'axes.edgecolor': self.config.GRID_COLOR
        })

        # Style legend and title
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                  facecolor=self.config.FIGURE_BG, edgecolor=self.config.GRID_COLOR)
        plt.title(f"{n_components}D PCA of Prompt Embeddings\n({', '.join(prompt_groups.keys())})",
                 color=self.config.TEXT_COLOR, pad=20)

        if all_components:
            all_components = np.vstack(all_components)
            if n_components == 2:
                plt.xlim(all_components[:, 0].min() * 1.2, all_components[:, 0].max() * 1.2)
                plt.ylim(all_components[:, 1].min() * 1.2, all_components[:, 1].max() * 1.2)
            else:
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_zlabel('PC3')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=self.config.FIGURE_BG, dpi=150)
        plt.close()
        buf.seek(0)
        return Image.open(buf)


class ControlVectorVisualizer:
    """Visualizes control vector effects on embeddings."""

    def __init__(self, model):
        self.model = model
        self.config = VisualizationConfig()

    def _fig_to_image(self, fig: plt.Figure) -> Image.Image:
        """Convert matplotlib figure to PIL Image."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight',
                   facecolor=self.config.FIGURE_BG, dpi=150)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img.load()  # Force load image data into memory
        return img.copy()  # Return a copy that doesn't depend on the buffer

    def visualize_steering_effect(
        self,
        prompt: str,
        control_vector: torch.Tensor,
        scales: List[float] = [-2, -1, 0, 1, 2],
        title: str = "Control Vector Steering Effect"
    ) -> Image.Image:
        """
        Visualize how different strengths of control vector affect the embedding.
        Shows trajectory from negative to positive scaling.

        Args:
            prompt: Input text
            control_vector: The control vector
            scales: List of scale factors to visualize
            title: Plot title

        Returns:
            PIL Image of the visualization
        """
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(self.config.FIGURE_BG)
        ax.set_facecolor(self.config.FIGURE_BG)

        # Get base embedding
        base_hidden = self.model.get_hidden_state(prompt)

        # Generate embeddings at different scales
        embeddings = []
        for scale in scales:
            modified = base_hidden + scale * control_vector
            embeddings.append(modified.cpu().detach().numpy())

        embeddings = np.array(embeddings)

        # Add control vector direction for visualization
        cv_np = control_vector.cpu().detach().numpy()
        all_points = np.vstack([embeddings, cv_np.reshape(1, -1)])

        # PCA to 2D
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(all_points)

        embeddings_2d = points_2d[:-1]
        cv_2d = points_2d[-1]

        # Create colormap from red (negative) to blue (positive)
        cmap = LinearSegmentedColormap.from_list('diverging', ['#FF6B35', '#FFD700', '#00FFFF'])
        norm_scales = [(s - min(scales)) / (max(scales) - min(scales)) for s in scales]

        # Plot trajectory
        for i in range(len(scales) - 1):
            ax.annotate(
                '',
                xy=(embeddings_2d[i+1, 0], embeddings_2d[i+1, 1]),
                xytext=(embeddings_2d[i, 0], embeddings_2d[i, 1]),
                arrowprops=dict(
                    arrowstyle='->',
                    color=cmap(norm_scales[i]),
                    lw=2,
                    alpha=0.7
                )
            )

        # Plot points with scale labels
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=norm_scales,
            cmap=cmap,
            s=150,
            edgecolor='white',
            linewidth=2,
            zorder=5
        )

        # Add scale labels
        for i, scale in enumerate(scales):
            ax.annotate(
                f'{scale:+.1f}',
                (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                color=self.config.TEXT_COLOR,
                fontweight='bold'
            )

        # Draw control vector direction from center
        center = embeddings_2d[len(scales)//2]
        cv_direction = cv_2d / np.linalg.norm(cv_2d) * np.linalg.norm(embeddings_2d[-1] - embeddings_2d[0]) * 0.3

        ax.annotate(
            '',
            xy=(center[0] + cv_direction[0], center[1] + cv_direction[1]),
            xytext=(center[0], center[1]),
            arrowprops=dict(
                arrowstyle='->',
                color='#FF1493',
                lw=3
            )
        )
        ax.annotate(
            'CV Direction',
            (center[0] + cv_direction[0], center[1] + cv_direction[1]),
            xytext=(10, 5),
            textcoords='offset points',
            fontsize=9,
            color='#FF1493'
        )

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Scale Factor', color=self.config.TEXT_COLOR)
        cbar.ax.yaxis.set_tick_params(color=self.config.TEXT_COLOR)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.config.TEXT_COLOR)

        ax.set_xlabel('PC1', color=self.config.TEXT_COLOR, fontsize=12)
        ax.set_ylabel('PC2', color=self.config.TEXT_COLOR, fontsize=12)
        ax.set_title(f"{title}\nPrompt: '{prompt[:50]}...'",
                    color=self.config.TEXT_COLOR, fontsize=14, pad=20)
        ax.tick_params(colors=self.config.TEXT_COLOR)
        ax.grid(True, color=self.config.GRID_COLOR, alpha=0.3, linestyle='--')

        return self._fig_to_image(fig)

    def compare_before_after(
        self,
        prompt: str,
        control_vector: torch.Tensor,
        scale: float = 1.0,
        title: str = "Before vs After Control Vector"
    ) -> Image.Image:
        """
        Side-by-side comparison of embeddings before and after control vector.

        Args:
            prompt: Input text
            control_vector: Control vector to apply
            scale: Scale factor
            title: Plot title

        Returns:
            PIL Image of comparison
        """
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.patch.set_facecolor(self.config.FIGURE_BG)

        # Get embeddings
        base_hidden = self.model.get_hidden_state(prompt)
        modified_hidden = base_hidden + scale * control_vector

        base_np = base_hidden.cpu().detach().numpy()
        modified_np = modified_hidden.cpu().detach().numpy()
        cv_np = control_vector.cpu().detach().numpy()

        # Plot 1: Original embedding components
        ax = axes[0]
        ax.set_facecolor(self.config.FIGURE_BG)
        top_indices = np.argsort(np.abs(base_np))[-30:]
        ax.bar(range(30), base_np[top_indices], color='#00FFFF', alpha=0.8)
        ax.set_title('Original Embedding\n(Top 30 dims)', color=self.config.TEXT_COLOR)
        ax.set_xlabel('Dimension Index', color=self.config.TEXT_COLOR)
        ax.set_ylabel('Value', color=self.config.TEXT_COLOR)
        ax.tick_params(colors=self.config.TEXT_COLOR)

        # Plot 2: Modified embedding components
        ax = axes[1]
        ax.set_facecolor(self.config.FIGURE_BG)
        ax.bar(range(30), modified_np[top_indices], color='#FF6B35', alpha=0.8)
        ax.set_title(f'With Control Vector\n(scale={scale})', color=self.config.TEXT_COLOR)
        ax.set_xlabel('Dimension Index', color=self.config.TEXT_COLOR)
        ax.set_ylabel('Value', color=self.config.TEXT_COLOR)
        ax.tick_params(colors=self.config.TEXT_COLOR)

        # Plot 3: Difference
        ax = axes[2]
        ax.set_facecolor(self.config.FIGURE_BG)
        diff = modified_np[top_indices] - base_np[top_indices]
        colors = ['#32CD32' if d > 0 else '#FF1493' for d in diff]
        ax.bar(range(30), diff, color=colors, alpha=0.8)
        ax.axhline(y=0, color=self.config.TEXT_COLOR, linestyle='-', linewidth=0.5)
        ax.set_title('Difference (Δ)', color=self.config.TEXT_COLOR)
        ax.set_xlabel('Dimension Index', color=self.config.TEXT_COLOR)
        ax.set_ylabel('Change', color=self.config.TEXT_COLOR)
        ax.tick_params(colors=self.config.TEXT_COLOR)

        fig.suptitle(title, color=self.config.TEXT_COLOR, fontsize=14, y=1.02)
        plt.tight_layout()

        return self._fig_to_image(fig)

    def visualize_vector_anatomy(
        self,
        control_vector: torch.Tensor,
        name: str = "Control Vector",
        title: str = "Control Vector Anatomy"
    ) -> Image.Image:
        """
        Detailed visualization of a control vector's structure.

        Args:
            control_vector: The control vector to visualize
            name: Name of the vector
            title: Plot title

        Returns:
            PIL Image of the visualization
        """
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.patch.set_facecolor(self.config.FIGURE_BG)

        cv_np = control_vector.cpu().detach().numpy()

        # Plot 1: Top components by magnitude
        ax = axes[0, 0]
        ax.set_facecolor(self.config.FIGURE_BG)
        sorted_indices = np.argsort(np.abs(cv_np))[::-1]
        top_n = 40
        values = cv_np[sorted_indices[:top_n]]
        colors = ['#00FFFF' if v > 0 else '#FF6B35' for v in values]
        ax.bar(range(top_n), values, color=colors, alpha=0.8)
        ax.axhline(y=0, color=self.config.TEXT_COLOR, linestyle='-', linewidth=0.5)
        ax.set_title('Top 40 Components by Magnitude', color=self.config.TEXT_COLOR)
        ax.set_xlabel('Rank', color=self.config.TEXT_COLOR)
        ax.set_ylabel('Value', color=self.config.TEXT_COLOR)
        ax.tick_params(colors=self.config.TEXT_COLOR)

        # Plot 2: Distribution histogram
        ax = axes[0, 1]
        ax.set_facecolor(self.config.FIGURE_BG)
        ax.hist(cv_np, bins=50, color='#FFD700', alpha=0.7, edgecolor=self.config.FIGURE_BG)
        ax.axvline(x=0, color='#FF1493', linestyle='--', linewidth=2)
        ax.set_title('Value Distribution', color=self.config.TEXT_COLOR)
        ax.set_xlabel('Value', color=self.config.TEXT_COLOR)
        ax.set_ylabel('Count', color=self.config.TEXT_COLOR)
        ax.tick_params(colors=self.config.TEXT_COLOR)

        # Add statistics text
        stats_text = f"Mean: {cv_np.mean():.4f}\nStd: {cv_np.std():.4f}\nNorm: {np.linalg.norm(cv_np):.4f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right',
               color=self.config.TEXT_COLOR,
               bbox=dict(boxstyle='round', facecolor=self.config.FIGURE_BG,
                        edgecolor=self.config.GRID_COLOR))

        # Plot 3: Cumulative magnitude
        ax = axes[1, 0]
        ax.set_facecolor(self.config.FIGURE_BG)
        sorted_magnitudes = np.sort(np.abs(cv_np))[::-1]
        cumulative = np.cumsum(sorted_magnitudes) / np.sum(sorted_magnitudes)
        ax.plot(cumulative, color='#00FFFF', linewidth=2)
        ax.axhline(y=0.9, color='#FF6B35', linestyle='--', alpha=0.7, label='90%')
        ax.axhline(y=0.95, color='#FF1493', linestyle='--', alpha=0.7, label='95%')
        ax.set_title('Cumulative Magnitude', color=self.config.TEXT_COLOR)
        ax.set_xlabel('Number of Dimensions', color=self.config.TEXT_COLOR)
        ax.set_ylabel('Fraction of Total', color=self.config.TEXT_COLOR)
        ax.tick_params(colors=self.config.TEXT_COLOR)
        ax.legend(facecolor=self.config.FIGURE_BG, labelcolor=self.config.TEXT_COLOR)
        ax.grid(True, color=self.config.GRID_COLOR, alpha=0.3, linestyle='--')

        # Find 90% and 95% thresholds
        dims_90 = np.searchsorted(cumulative, 0.9)
        dims_95 = np.searchsorted(cumulative, 0.95)
        ax.axvline(x=dims_90, color='#FF6B35', linestyle=':', alpha=0.5)
        ax.axvline(x=dims_95, color='#FF1493', linestyle=':', alpha=0.5)

        # Plot 4: 2D grid view
        ax = axes[1, 1]
        ax.set_facecolor(self.config.FIGURE_BG)
        side = int(np.ceil(np.sqrt(len(cv_np))))
        padded = np.zeros(side * side)
        padded[:len(cv_np)] = cv_np
        grid = padded.reshape(side, side)

        im = ax.imshow(grid, cmap='RdBu_r', aspect='auto')
        ax.set_title('Vector as 2D Grid', color=self.config.TEXT_COLOR)
        ax.tick_params(colors=self.config.TEXT_COLOR)
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color=self.config.TEXT_COLOR)
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=self.config.TEXT_COLOR)

        fig.suptitle(f"{title}: {name}", color=self.config.TEXT_COLOR, fontsize=14, y=1.02)
        plt.tight_layout()

        return self._fig_to_image(fig)


class GenerationComparisonVisualizer:
    """Visualizes generation differences with and without control vectors."""

    def __init__(self):
        self.config = VisualizationConfig()

    def _fig_to_image(self, fig: plt.Figure) -> Image.Image:
        """Convert matplotlib figure to PIL Image."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight',
                   facecolor=self.config.FIGURE_BG, dpi=150)
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)
        img.load()  # Force load image data into memory
        return img.copy()  # Return a copy that doesn't depend on the buffer

    def create_generation_comparison_card(
        self,
        prompt: str,
        original_output: str,
        controlled_outputs: Dict[str, str],
        title: str = "Generation Comparison"
    ) -> Image.Image:
        """
        Create a visual comparison card showing different generations.

        Args:
            prompt: The input prompt
            original_output: Output without control vector
            controlled_outputs: Dict mapping vector_name -> output
            title: Card title

        Returns:
            PIL Image of the comparison
        """
        plt.style.use('dark_background')

        num_outputs = 1 + len(controlled_outputs)
        fig, axes = plt.subplots(num_outputs, 1, figsize=(14, 3 * num_outputs))
        fig.patch.set_facecolor(self.config.FIGURE_BG)

        if num_outputs == 1:
            axes = [axes]

        # Original output
        ax = axes[0]
        ax.set_facecolor('#2d3561')
        ax.text(0.02, 0.5, f"Original:\n{original_output[:200]}...",
               transform=ax.transAxes, fontsize=11,
               color=self.config.TEXT_COLOR,
               verticalalignment='center',
               wrap=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Without Control Vector', color='#00FFFF',
                    fontsize=12, loc='left', pad=10)

        # Controlled outputs
        for i, (vector_name, output) in enumerate(controlled_outputs.items()):
            ax = axes[i + 1]
            ax.set_facecolor('#3d2561')
            ax.text(0.02, 0.5, f"With {vector_name}:\n{output[:200]}...",
                   transform=ax.transAxes, fontsize=11,
                   color=self.config.TEXT_COLOR,
                   verticalalignment='center',
                   wrap=True)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title(f'With {vector_name} Vector',
                        color='#FF6B35', fontsize=12, loc='left', pad=10)

        fig.suptitle(f"{title}\nPrompt: '{prompt}'",
                    color=self.config.TEXT_COLOR, fontsize=14, y=1.02)
        plt.tight_layout()

        return self._fig_to_image(fig)

