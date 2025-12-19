"""
Animated Visualization Module

Provides Plotly-based animated 3D visualizations for neural network analysis.
Includes play/pause controls, animation sliders, and multiple animation modes.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from PIL import Image


@dataclass
class NetworkNode:
    """Represents a node in the network visualization."""
    x: float
    y: float
    z: float
    layer: int
    index: int
    label: str


@dataclass
class NetworkEdge:
    """Represents an edge in the network visualization."""
    source: NetworkNode
    target: NetworkNode
    weight: float
    edge_type: str = 'feedforward'  # 'feedforward', 'recurrent', 'attention'


class AnimatedNetworkViz:
    """
    Creates animated 3D visualizations of neural network activations.
    """

    def __init__(self, colorscale: str = 'Viridis'):
        self.colorscale = colorscale
        self.frame_duration = 200  # ms
        self.transition_duration = 100  # ms

    def build_network_graph(
        self,
        layer_sizes: List[int],
        weight_mats: List[np.ndarray],
        top_k: int = 5
    ) -> Tuple[Dict, Dict]:
        """
        Build 3D network graph with nodes and edges.

        Args:
            layer_sizes: Number of neurons per layer
            weight_mats: Weight matrices between layers
            top_k: Number of top edges to show per neuron

        Returns:
            (nodes_dict, edges_dict) for visualization
        """
        nodes = {'x': [], 'y': [], 'z': [], 'labels': [], 'layers': [], 'indices': []}
        edges = {'x': [], 'y': [], 'z': [], 'weights': []}

        # Create nodes for each layer
        for layer_idx, size in enumerate(layer_sizes):
            for neuron_idx in range(size):
                # Position: x = layer, y = neuron index (centered), z = 0
                nodes['x'].append(layer_idx * 2.0)
                nodes['y'].append(neuron_idx - size / 2)
                nodes['z'].append(0)
                nodes['labels'].append(f'L{layer_idx}_N{neuron_idx}')
                nodes['layers'].append(layer_idx)
                nodes['indices'].append(neuron_idx)

        # Create edges based on weight matrices
        node_offset = 0
        for layer_idx, weights in enumerate(weight_mats):
            src_size = layer_sizes[layer_idx]
            dst_size = layer_sizes[layer_idx + 1]

            # For each destination neuron, find top-k source connections
            for dst_idx in range(dst_size):
                row = np.abs(weights[dst_idx])
                top_indices = np.argsort(row)[-top_k:]

                for src_idx in top_indices:
                    if src_idx < src_size:
                        # Source node position
                        src_x = layer_idx * 2.0
                        src_y = src_idx - src_size / 2
                        src_z = 0

                        # Destination node position
                        dst_x = (layer_idx + 1) * 2.0
                        dst_y = dst_idx - dst_size / 2
                        dst_z = 0

                        # Add edge (with None separator for Plotly)
                        edges['x'].extend([src_x, dst_x, None])
                        edges['y'].extend([src_y, dst_y, None])
                        edges['z'].extend([src_z, dst_z, None])
                        edges['weights'].append(weights[dst_idx, src_idx])

            node_offset += src_size

        return nodes, edges

    def build_lstm_graph(
        self,
        layer_sizes: List[int],
        weight_mats: List[np.ndarray],
        top_k: int = 5,
        num_timesteps: int = 3
    ) -> Tuple[Dict, Dict]:
        """
        Build LSTM network graph unrolled across timesteps.

        Args:
            layer_sizes: Number of neurons per layer
            weight_mats: Weight matrices
            top_k: Number of top edges per neuron
            num_timesteps: Number of timesteps to unroll

        Returns:
            (nodes_dict, edges_dict) with temporal structure
        """
        nodes = {'x': [], 'y': [], 'z': [], 'labels': [], 'layers': [], 'timesteps': []}
        edges = {'x': [], 'y': [], 'z': [], 'weights': [], 'types': []}

        # Create nodes for each timestep and layer
        for t in range(num_timesteps):
            for layer_idx, size in enumerate(layer_sizes):
                for neuron_idx in range(size):
                    # Position: x = layer, y = neuron, z = timestep
                    nodes['x'].append(layer_idx * 2.0)
                    nodes['y'].append(neuron_idx - size / 2)
                    nodes['z'].append(t * 3.0)
                    nodes['labels'].append(f'T{t}_L{layer_idx}_N{neuron_idx}')
                    nodes['layers'].append(layer_idx)
                    nodes['timesteps'].append(t)

        # Feedforward edges within each timestep
        for t in range(num_timesteps):
            node_offset = 0
            for layer_idx, weights in enumerate(weight_mats):
                if layer_idx >= len(layer_sizes) - 1:
                    break

                src_size = layer_sizes[layer_idx]
                dst_size = layer_sizes[layer_idx + 1]

                for dst_idx in range(min(dst_size, weights.shape[0])):
                    row = np.abs(weights[dst_idx])
                    top_indices = np.argsort(row.flatten())[-top_k:]

                    for src_idx in top_indices:
                        if src_idx < src_size:
                            src_x = layer_idx * 2.0
                            src_y = src_idx - src_size / 2
                            src_z = t * 3.0

                            dst_x = (layer_idx + 1) * 2.0
                            dst_y = dst_idx - dst_size / 2
                            dst_z = t * 3.0

                            edges['x'].extend([src_x, dst_x, None])
                            edges['y'].extend([src_y, dst_y, None])
                            edges['z'].extend([src_z, dst_z, None])
                            edges['weights'].append(float(weights[dst_idx, src_idx % weights.shape[1]]))
                            edges['types'].append('feedforward')

                node_offset += src_size

        # Recurrent edges between timesteps (for hidden layers)
        for t in range(num_timesteps - 1):
            for layer_idx in range(1, len(layer_sizes) - 1):  # Hidden layers only
                size = layer_sizes[layer_idx]
                for neuron_idx in range(size):
                    # Connect same neuron across timesteps
                    src_x = layer_idx * 2.0
                    src_y = neuron_idx - size / 2
                    src_z = t * 3.0

                    dst_x = layer_idx * 2.0
                    dst_y = neuron_idx - size / 2
                    dst_z = (t + 1) * 3.0

                    edges['x'].extend([src_x, dst_x, None])
                    edges['y'].extend([src_y, dst_y, None])
                    edges['z'].extend([src_z, dst_z, None])
                    edges['weights'].append(1.0)
                    edges['types'].append('recurrent')

        return nodes, edges

    def create_animated_mlp_figure(
        self,
        nodes: Dict,
        edges: Dict,
        layer_sizes: List[int],
        all_activations: List[List[np.ndarray]],
        show_edges: bool = True,
        log_color: bool = False
    ) -> go.Figure:
        """
        Create animated figure showing activation flow through MLP.

        Args:
            nodes: Node positions from build_network_graph
            edges: Edge positions from build_network_graph
            layer_sizes: Neurons per layer
            all_activations: List of activation lists (one per animation frame)
            show_edges: Whether to show edge lines
            log_color: Whether to use log scale for colors

        Returns:
            Plotly Figure with animation controls
        """
        # Create frames for animation
        frames = []

        for frame_idx, activations in enumerate(all_activations):
            # Convert activations to node colors
            node_colors = self._activations_to_colors(activations, layer_sizes, log_color)

            frame_data = [
                go.Scatter3d(
                    x=nodes['x'],
                    y=nodes['y'],
                    z=nodes['z'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=node_colors,
                        colorscale=self.colorscale,
                        colorbar=dict(title='Activation'),
                        opacity=0.9,
                        cmin=0,
                        cmax=1
                    ),
                    text=nodes['labels'],
                    hoverinfo='text+name',
                    name='Neurons'
                )
            ]

            if show_edges:
                frame_data.insert(0, go.Scatter3d(
                    x=edges['x'],
                    y=edges['y'],
                    z=edges['z'],
                    mode='lines',
                    line=dict(width=1.5, color='#4a90d9'),
                    hoverinfo='none',
                    name='Weights'
                ))

            frames.append(go.Frame(data=frame_data, name=f'frame_{frame_idx}'))

        # Initial data (first frame)
        initial_colors = self._activations_to_colors(all_activations[0], layer_sizes, log_color)

        data = []
        if show_edges:
            data.append(go.Scatter3d(
                x=edges['x'],
                y=edges['y'],
                z=edges['z'],
                mode='lines',
                line=dict(width=1.5, color='#4a90d9'),
                hoverinfo='none',
                name='Weights'
            ))

        data.append(go.Scatter3d(
            x=nodes['x'],
            y=nodes['y'],
            z=nodes['z'],
            mode='markers',
            marker=dict(
                size=8,
                color=initial_colors,
                colorscale=self.colorscale,
                colorbar=dict(title='Activation'),
                opacity=0.9,
                cmin=0,
                cmax=1
            ),
            text=nodes['labels'],
            hoverinfo='text+name',
            name='Neurons'
        ))

        # Create figure with animation controls
        fig = go.Figure(
            data=data,
            frames=frames,
            layout=go.Layout(
                title='MLP Activation Flow Animation',
                showlegend=True,
                paper_bgcolor='black',
                plot_bgcolor='black',
                scene=dict(
                    bgcolor='black',
                    xaxis=dict(title='Layer', showgrid=True, gridcolor='#444444', zeroline=False, color='#888888'),
                    yaxis=dict(title='Neuron Index', showgrid=True, gridcolor='#444444', zeroline=False, color='#888888'),
                    zaxis=dict(title='Z', showgrid=True, gridcolor='#444444', zeroline=False, color='#888888'),
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                updatemenus=[
                    dict(
                        type='buttons',
                        showactive=False,
                        y=0,
                        x=0.1,
                        xanchor='right',
                        yanchor='top',
                        buttons=[
                            dict(
                                label='▶ Play',
                                method='animate',
                                args=[
                                    None,
                                    dict(
                                        frame=dict(duration=self.frame_duration, redraw=True),
                                        fromcurrent=True,
                                        transition=dict(duration=self.transition_duration)
                                    )
                                ]
                            ),
                            dict(
                                label='⏸ Pause',
                                method='animate',
                                args=[
                                    [None],
                                    dict(
                                        frame=dict(duration=0, redraw=False),
                                        mode='immediate',
                                        transition=dict(duration=0)
                                    )
                                ]
                            )
                        ]
                    )
                ],
                sliders=[
                    dict(
                        active=0,
                        yanchor='top',
                        xanchor='left',
                        currentvalue=dict(
                            font=dict(size=12),
                            prefix='Frame: ',
                            visible=True,
                            xanchor='right'
                        ),
                        transition=dict(duration=self.transition_duration),
                        pad=dict(b=10, t=50),
                        len=0.9,
                        x=0.1,
                        y=0,
                        steps=[
                            dict(
                                args=[[f'frame_{i}'],
                                      dict(frame=dict(duration=self.frame_duration, redraw=True),
                                           mode='immediate',
                                           transition=dict(duration=self.transition_duration))],
                                label=str(i),
                                method='animate'
                            )
                            for i in range(len(all_activations))
                        ]
                    )
                ]
            )
        )

        return fig

    def create_animated_lstm_figure(
        self,
        nodes: Dict,
        edges: Dict,
        layer_sizes: List[int],
        timestep_activations: List[Dict],
        show_edges: bool = True
    ) -> go.Figure:
        """
        Create animated figure showing LSTM temporal dynamics.

        Args:
            nodes: Node positions from build_lstm_graph
            edges: Edge positions from build_lstm_graph
            layer_sizes: Neurons per layer
            timestep_activations: List of activation dicts per timestep
            show_edges: Whether to show edges

        Returns:
            Plotly Figure with animation
        """
        num_timesteps = len(timestep_activations)
        frames = []

        for t, step_acts in enumerate(timestep_activations):
            # Build activation values for all nodes up to timestep t
            node_colors = []

            for node_idx in range(len(nodes['x'])):
                node_t = nodes['timesteps'][node_idx]
                node_layer = nodes['layers'][node_idx]

                if node_t <= t:
                    # Get activation for this node
                    if node_layer == 0:  # Input layer
                        act = np.abs(step_acts['input']).mean()
                    elif node_layer == len(layer_sizes) - 1:  # Output layer
                        act = np.abs(step_acts['output']).mean()
                    else:  # Hidden layers
                        hidden_idx = node_layer - 1
                        if hidden_idx < len(step_acts['hidden_states']):
                            act = np.abs(step_acts['hidden_states'][hidden_idx]).mean()
                        else:
                            act = 0.0
                    node_colors.append(min(1.0, act))
                else:
                    node_colors.append(0.0)  # Future nodes are dark

            # Normalize colors
            max_color = max(node_colors) if max(node_colors) > 0 else 1.0
            node_colors = [c / max_color for c in node_colors]

            frame_data = []

            if show_edges:
                # Color edges based on type
                edge_colors = []
                for i, edge_type in enumerate(edges.get('types', ['feedforward'] * (len(edges['x']) // 3))):
                    if edge_type == 'recurrent':
                        edge_colors.extend(['#00d4ff', '#00d4ff', '#00d4ff'])  # Cyan for recurrent
                    else:
                        edge_colors.extend(['#4a90d9', '#4a90d9', '#4a90d9'])  # Blue for feedforward

                frame_data.append(go.Scatter3d(
                    x=edges['x'],
                    y=edges['y'],
                    z=edges['z'],
                    mode='lines',
                    line=dict(width=2, color=edge_colors[:len(edges['x'])]),
                    hoverinfo='none',
                    name='Connections'
                ))

            frame_data.append(go.Scatter3d(
                x=nodes['x'],
                y=nodes['y'],
                z=nodes['z'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=node_colors,
                    colorscale=self.colorscale,
                    colorbar=dict(title='Activation'),
                    opacity=0.9
                ),
                text=nodes['labels'],
                hoverinfo='text+name',
                name='Neurons'
            ))

            frames.append(go.Frame(data=frame_data, name=f'timestep_{t}'))

        # Initial frame
        initial_data = frames[0].data if frames else []

        fig = go.Figure(
            data=list(initial_data),
            frames=frames,
            layout=go.Layout(
                title='LSTM Temporal Activation Flow',
                showlegend=True,
                paper_bgcolor='black',
                plot_bgcolor='black',
                scene=dict(
                    bgcolor='black',
                    xaxis=dict(title='Layer', showgrid=True, gridcolor='#444444', color='#888888'),
                    yaxis=dict(title='Neuron Index', showgrid=True, gridcolor='#444444', color='#888888'),
                    zaxis=dict(title='Timestep', showgrid=True, gridcolor='#444444', color='#888888'),
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                updatemenus=[
                    dict(
                        type='buttons',
                        showactive=False,
                        y=0,
                        x=0.1,
                        buttons=[
                            dict(label='▶ Play', method='animate',
                                 args=[None, dict(frame=dict(duration=500, redraw=True),
                                                  fromcurrent=True)]),
                            dict(label='⏸ Pause', method='animate',
                                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                    mode='immediate')])
                        ]
                    )
                ],
                sliders=[
                    dict(
                        active=0,
                        currentvalue=dict(prefix='Timestep: '),
                        steps=[
                            dict(args=[[f'timestep_{t}'],
                                       dict(frame=dict(duration=500, redraw=True),
                                            mode='immediate')],
                                 label=str(t), method='animate')
                            for t in range(num_timesteps)
                        ]
                    )
                ]
            )
        )

        return fig

    def _activations_to_colors(
        self,
        activations: List[np.ndarray],
        layer_sizes: List[int],
        log_scale: bool = False
    ) -> List[float]:
        """
        Convert layer activations to color values for each node.

        Args:
            activations: List of activation arrays per layer
            layer_sizes: Number of neurons per layer
            log_scale: Whether to apply log scaling

        Returns:
            List of color values (0-1) for each node
        """
        colors = []

        for layer_idx, (size, acts) in enumerate(zip(layer_sizes, activations)):
            acts_flat = np.abs(acts).flatten()

            if log_scale:
                acts_flat = np.log1p(acts_flat)

            # Pad or truncate to match layer size
            if len(acts_flat) < size:
                acts_flat = np.pad(acts_flat, (0, size - len(acts_flat)))
            else:
                acts_flat = acts_flat[:size]

            colors.extend(acts_flat.tolist())

        # Normalize to 0-1
        max_val = max(colors) if colors and max(colors) > 0 else 1.0
        colors = [c / max_val for c in colors]

        return colors


class AttentionHeadViz:
    """
    Visualizes transformer attention head structure in 3D.
    Shows Q, K, V, O projections and attention computation flow.
    """

    def __init__(self, colorscale: str = 'Plasma'):
        self.colorscale = colorscale

    def build_attention_head_graph(
        self,
        layer_name: str,
        weight_dict: Dict[str, np.ndarray],
        num_heads: int = 4,
        head_dim: int = 16
    ) -> go.Figure:
        """
        Build 3D visualization of attention head structure.

        Args:
            layer_name: Name of the attention layer
            weight_dict: Dict with 'q', 'k', 'v', 'o' weight matrices
            num_heads: Number of attention heads
            head_dim: Dimension per head

        Returns:
            Plotly Figure showing attention head structure
        """
        data = []

        # Node positions for each component
        # Stage 0: Input
        # Stage 1: Q, K, V projections
        # Stage 2: Attention scores
        # Stage 3: Output projection

        stages = ['Input', 'Q/K/V', 'Attention', 'Output']

        # Input nodes
        input_dim = num_heads * head_dim
        input_nodes_x = [0] * input_dim
        input_nodes_y = list(range(input_dim))
        input_nodes_z = [0] * input_dim

        # Q, K, V nodes (grouped by head)
        qkv_x, qkv_y, qkv_z = [], [], []
        qkv_colors = []
        qkv_labels = []

        for head in range(num_heads):
            for proj_idx, proj in enumerate(['Q', 'K', 'V']):
                for d in range(head_dim):
                    qkv_x.append(1)  # Stage 1
                    qkv_y.append(head * head_dim * 3 + proj_idx * head_dim + d)
                    qkv_z.append(head)  # Separate heads in Z
                    qkv_colors.append(proj_idx / 2)  # Color by Q/K/V
                    qkv_labels.append(f'Head{head}_{proj}_{d}')

        # Attention output nodes
        attn_x = [2] * (num_heads * head_dim)
        attn_y = list(range(num_heads * head_dim))
        attn_z = [h // head_dim for h in range(num_heads * head_dim)]

        # Final output nodes
        output_x = [3] * input_dim
        output_y = list(range(input_dim))
        output_z = [0] * input_dim

        # Add traces
        # Input layer
        data.append(go.Scatter3d(
            x=input_nodes_x,
            y=input_nodes_y,
            z=input_nodes_z,
            mode='markers',
            marker=dict(size=4, color='blue', opacity=0.7),
            name='Input',
            hoverinfo='text',
            text=[f'Input_{i}' for i in range(input_dim)]
        ))

        # Q/K/V projections
        data.append(go.Scatter3d(
            x=qkv_x,
            y=qkv_y,
            z=qkv_z,
            mode='markers',
            marker=dict(
                size=5,
                color=qkv_colors,
                colorscale='Viridis',
                opacity=0.8
            ),
            name='Q/K/V',
            hoverinfo='text',
            text=qkv_labels
        ))

        # Attention outputs
        data.append(go.Scatter3d(
            x=attn_x,
            y=attn_y,
            z=attn_z,
            mode='markers',
            marker=dict(size=5, color='orange', opacity=0.8),
            name='Attention Output',
            hoverinfo='text',
            text=[f'Attn_{i}' for i in range(len(attn_x))]
        ))

        # Final output
        data.append(go.Scatter3d(
            x=output_x,
            y=output_y,
            z=output_z,
            mode='markers',
            marker=dict(size=4, color='green', opacity=0.7),
            name='Output',
            hoverinfo='text',
            text=[f'Output_{i}' for i in range(input_dim)]
        ))

        # Add connection lines (simplified)
        # Input to Q/K/V
        edge_x, edge_y, edge_z = [], [], []
        for head in range(num_heads):
            # Sample connections from input to Q
            for i in range(0, input_dim, input_dim // 4):
                q_idx = head * head_dim * 3 + (i % head_dim)
                edge_x.extend([0, 1, None])
                edge_y.extend([i, head * head_dim * 3 + (i % head_dim), None])
                edge_z.extend([0, head, None])

        data.insert(0, go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(width=1.5, color='#4a90d9'),
            hoverinfo='none',
            name='Connections'
        ))

        fig = go.Figure(
            data=data,
            layout=go.Layout(
                title=f'Attention Head Structure - {layer_name}',
                showlegend=True,
                paper_bgcolor='black',
                plot_bgcolor='black',
                scene=dict(
                    bgcolor='black',
                    xaxis=dict(
                        title='Stage (Input→Q/K/V→Attn→Output)',
                        tickvals=[0, 1, 2, 3],
                        ticktext=stages,
                        showgrid=True,
                        gridcolor='#444444',
                        color='#888888'
                    ),
                    yaxis=dict(title='Dimension', showgrid=True, gridcolor='#444444', color='#888888'),
                    zaxis=dict(title='Head', showgrid=True, gridcolor='#444444', color='#888888'),
                ),
                margin=dict(l=0, r=0, b=0, t=40),
            )
        )

        return fig


class ManifoldAnimator:
    """
    Creates animated 3D manifold visualizations with multiple animation modes.
    """

    def __init__(self, colorscale: str = 'Viridis'):
        self.colorscale = colorscale

    def create_animated_manifold(
        self,
        activation_matrix: np.ndarray,
        layer_sizes: Optional[List[int]] = None,
        animation_mode: str = 'sample_progression',
        frame_duration: int = 200,
        is_temporal: bool = False,
        num_timesteps: int = 1
    ) -> go.Figure:
        """
        Create animated 3D manifold visualization.

        Args:
            activation_matrix: [samples, neurons] activation data
            layer_sizes: Neuron counts per layer (for boundary markers)
            animation_mode: 'sample_progression', 'layer_slicing', 'rotating_camera'
            frame_duration: Milliseconds per frame
            is_temporal: Whether data has temporal structure
            num_timesteps: Number of timesteps (for temporal data)

        Returns:
            Animated Plotly Figure
        """
        # Normalize
        matrix_viz = (activation_matrix - activation_matrix.mean()) / (activation_matrix.std() + 1e-8)

        frames = []

        if animation_mode == 'sample_progression':
            # Build up surface over samples
            step_size = max(1, matrix_viz.shape[0] // 30)

            for i in range(step_size, matrix_viz.shape[0] + 1, step_size):
                Z_frame = matrix_viz[:i, :]
                X_frame, Y_frame = np.meshgrid(
                    np.arange(matrix_viz.shape[1]),
                    np.arange(i)
                )

                frame_data = [go.Surface(
                    x=X_frame,
                    y=Y_frame,
                    z=Z_frame,
                    colorscale=self.colorscale,
                    colorbar=dict(title='Activation'),
                    showscale=True,
                    cmin=matrix_viz.min(),
                    cmax=matrix_viz.max()
                )]

                frames.append(go.Frame(data=frame_data, name=f'sample_{i}'))

            # Initial frame
            X, Y = np.meshgrid(
                np.arange(matrix_viz.shape[1]),
                np.arange(step_size)
            )
            Z = matrix_viz[:step_size, :]

        elif animation_mode == 'layer_slicing' and layer_sizes:
            # Animate through layers
            layer_boundaries = [0]
            cumsum = 0
            for size in layer_sizes:
                cumsum += size
                layer_boundaries.append(cumsum)

            for l_idx in range(len(layer_sizes)):
                end = layer_boundaries[l_idx + 1]

                Z_frame = matrix_viz[:, :end]
                X_frame, Y_frame = np.meshgrid(
                    np.arange(end),
                    np.arange(matrix_viz.shape[0])
                )

                frame_data = [go.Surface(
                    x=X_frame,
                    y=Y_frame,
                    z=Z_frame,
                    colorscale=self.colorscale,
                    colorbar=dict(title='Activation'),
                    showscale=True,
                    cmin=matrix_viz.min(),
                    cmax=matrix_viz.max()
                )]

                frames.append(go.Frame(data=frame_data, name=f'layer_{l_idx}'))

            # Initial frame
            X, Y = np.meshgrid(
                np.arange(layer_boundaries[1]),
                np.arange(matrix_viz.shape[0])
            )
            Z = matrix_viz[:, :layer_boundaries[1]]

        elif animation_mode == 'rotating_camera':
            # Rotate camera around manifold
            X, Y = np.meshgrid(
                np.arange(matrix_viz.shape[1]),
                np.arange(matrix_viz.shape[0])
            )
            Z = matrix_viz

            n_angles = 60
            for i in range(n_angles):
                angle = i * (360 / n_angles)
                eye_x = 2.5 * np.cos(np.radians(angle))
                eye_y = 2.5 * np.sin(np.radians(angle))

                frame = go.Frame(
                    data=[go.Surface(
                        x=X, y=Y, z=Z,
                        colorscale=self.colorscale,
                        colorbar=dict(title='Activation'),
                        showscale=True
                    )],
                    layout=go.Layout(
                        scene_camera=dict(
                            eye=dict(x=eye_x, y=eye_y, z=1.5)
                        )
                    ),
                    name=f'angle_{angle:.0f}'
                )
                frames.append(frame)
        else:
            # Default: full static
            X, Y = np.meshgrid(
                np.arange(matrix_viz.shape[1]),
                np.arange(matrix_viz.shape[0])
            )
            Z = matrix_viz

        # Build figure
        fig = go.Figure(
            data=[go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale=self.colorscale,
                colorbar=dict(title='Activation'),
                hovertemplate='Neuron: %{x}<br>Sample: %{y}<br>Activation: %{z:.3f}<extra></extra>',
                lighting=dict(ambient=0.4, diffuse=0.5, specular=0.6, roughness=0.5),
                contours=dict(z=dict(show=True, usecolormap=True, highlightcolor='white', project=dict(z=True)))
            )],
            frames=frames,
            layout=go.Layout(
                title='Activation Manifold Animation',
                paper_bgcolor='black',
                plot_bgcolor='black',
                scene=dict(
                    bgcolor='black',
                    xaxis=dict(title='Neuron Index', showgrid=True, gridcolor='#444444', color='#888888'),
                    yaxis=dict(title='Sample', showgrid=True, gridcolor='#444444', color='#888888'),
                    zaxis=dict(title='Activation Value', showgrid=True, gridcolor='#444444', color='#888888'),
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                updatemenus=[
                    dict(
                        type='buttons',
                        showactive=False,
                        y=0,
                        x=0.1,
                        buttons=[
                            dict(label='▶ Play', method='animate',
                                 args=[None, dict(frame=dict(duration=frame_duration, redraw=True),
                                                  fromcurrent=True)]),
                            dict(label='⏸ Pause', method='animate',
                                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                    mode='immediate')])
                        ]
                    )
                ],
                sliders=[
                    dict(
                        active=0,
                        currentvalue=dict(prefix='Frame: '),
                        steps=[
                            dict(args=[[f.name], dict(frame=dict(duration=frame_duration, redraw=True),
                                                       mode='immediate')],
                                 label=f.name.split('_')[-1], method='animate')
                            for f in frames
                        ] if frames else []
                    )
                ] if frames else []
            )
        )

        return fig


def figure_to_image(fig: go.Figure, width: int = 800, height: int = 600) -> Image.Image:
    """
    Convert Plotly figure to PIL Image for Gradio display.

    Args:
        fig: Plotly Figure
        width: Image width
        height: Image height

    Returns:
        PIL Image
    """
    try:
        img_bytes = fig.to_image(format='png', width=width, height=height)
        return Image.open(io.BytesIO(img_bytes))
    except Exception:
        # Fallback if kaleido not available
        # Return a placeholder image
        img = Image.new('RGB', (width, height), color='white')
        return img
