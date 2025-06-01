import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import io
from PIL import Image
import matplotlib.patheffects as pe

class LatentSpaceVisualizer:
    def __init__(self, model):
        self.model = model
        
    def create_pca_plot(self, prompt_groups, labels_dict=None, n_components=2):
        # Set dark theme and figure
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 6))
        
        # Configure background and axes
        if n_components == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('#1a1a2e')
            # Update 3D pane colors using current API
            ax.xaxis.pane.fill = True
            ax.yaxis.pane.fill = True
            ax.zaxis.pane.fill = True
            
            ax.xaxis.pane.set_color((0.1, 0.1, 0.2, 0.8))
            ax.yaxis.pane.set_color((0.1, 0.1, 0.2, 0.8))
            ax.zaxis.pane.set_color((0.1, 0.1, 0.2, 0.8))
            
            # Set grid colors for 3D
            ax.grid(True, color='#4a4a5e', linestyle='--', alpha=0.3)
            ax.xaxis.set_pane_color((0.1, 0.1, 0.2, 0.8))
            ax.yaxis.set_pane_color((0.1, 0.1, 0.2, 0.8))
            ax.zaxis.set_pane_color((0.1, 0.1, 0.2, 0.8))
            
        fig.patch.set_facecolor('#1a1a2e')
        
        # Use cool blue color palette
        colors = plt.cm.winter(np.linspace(0.2, 0.8, len(prompt_groups)))
        
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
        plt.grid(True, color='#4a4a5e', linestyle='--', alpha=0.3)
        plt.rcParams.update({
            'axes.labelcolor': '#e0e0ff',
            'text.color': '#e0e0ff',
            'xtick.color': '#e0e0ff',
            'ytick.color': '#e0e0ff',
            'axes.edgecolor': '#4a4a5e'
        })
        
        # Style legend and title
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                  facecolor='#1a1a2e', edgecolor='#4a4a5e')
        plt.title(f"{n_components}D PCA of Prompt Embeddings\n({', '.join(prompt_groups.keys())})", 
                 color='#e0e0ff', pad=20)
        
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
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return Image.open(buf)
