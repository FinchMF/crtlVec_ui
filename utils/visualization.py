import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import io
from PIL import Image

class LatentSpaceVisualizer:
    def __init__(self, model):
        self.model = model
        
    def create_pca_plot(self, prompt_groups, labels_dict=None):
        plt.figure(figsize=(7, 5))
        colors = plt.cm.Set3(np.linspace(0, 1, len(prompt_groups)))
        
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
                pca = PCA(n_components=2)
                components = pca.fit_transform(vectors)
                all_components.extend(components)
                
                unique_labels = list(set(labels))
                for label_idx, label in enumerate(unique_labels):
                    indices = [i for i, l in enumerate(labels) if l == label]
                    plt.scatter(
                        components[indices, 0], 
                        components[indices, 1],
                        label=label,
                        color=colors[group_idx],
                        alpha=0.7,
                        marker=['o', 's'][label_idx % 2]  # Alternate markers for contrasting pairs
                    )
        
        if all_components:
            all_components = np.vstack(all_components)
            plt.xlim(all_components[:, 0].min() * 1.2, all_components[:, 0].max() * 1.2)
            plt.ylim(all_components[:, 1].min() * 1.2, all_components[:, 1].max() * 1.2)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"PCA of Prompt Embeddings\n({', '.join(prompt_groups.keys())})")
        plt.grid(True)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        return Image.open(buf)
