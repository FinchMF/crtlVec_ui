 app.py
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objs as go
import time
import logging
from dataclasses import dataclass, field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from typing import List, Tuple, Dict, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
)

try:
    from sklearn.decomposition import SparseCoder, DictionaryLearning
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ---------------------------
# Dataclasses for unpickling experiment results
# ---------------------------

@dataclass
class TokenJourney:
    """Track a single token's complete journey through the model."""
    token_id: int
    token_text: str
    position: int
    layer_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    layer_norms: Dict[str, float] = field(default_factory=dict)
    layer_cosine_to_prev: Dict[str, float] = field(default_factory=dict)

@dataclass
class CompactTokenJourney:
    """Lightweight token journey with compressed data."""
    token_id: int
    token_text: str
    position: int
    layer_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    layer_norms: Dict[str, float] = field(default_factory=dict)
    layer_cosine_to_prev: Dict[str, float] = field(default_factory=dict)

@dataclass
class WeightSnapshot:
    """Snapshot of important weights at a layer."""
    layer_idx: int
    layer_name: str
    mlp_up_norm: float = 0.0
    mlp_down_norm: Optional[float] = None
    attn_q_norm: Optional[float] = None
    attn_k_norm: Optional[float] = None
    attn_v_norm: Optional[float] = None
    attn_o_norm: Optional[float] = None

@dataclass
class SampleResult:
    """Basic sample result."""
    condition: str
    prompt: str
    category: str
    seed: int
    generated_text: str
    token_ids: List[int]
    hidden_states: Dict[str, np.ndarray]

@dataclass
class EnhancedSampleResult(SampleResult):
    """Extended result with token journeys and weight snapshots."""
    token_journeys: Optional[List[TokenJourney]] = None
    weight_snapshots: Optional[List[WeightSnapshot]] = None
    attention_patterns: Optional[List[np.ndarray]] = None

@dataclass
class CompactSampleResult:
    """Simplified sample result with minimal data."""
    condition: str
    prompt: str
    output: str
    prompt_length: int
    output_length: int
    token_journeys: Optional[List[CompactTokenJourney]] = None
    attention_patterns: Optional[List[np.ndarray]] = None

# =====================================================
# 1. Tiny demo MLP (full multi-layer network mode)
# =====================================================

class TinyMLP(nn.Module):
    def __init__(self, in_dim=16, h1=32, h2=32, out_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_dim)
        self.act = nn.ReLU()

    def forward(self, x, return_hidden: bool = False):
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        out = self.fc3(h2)
        if return_hidden:
            return out, [x, h1, h2, out]
        return out


class TinyLSTM(nn.Module):
    """LSTM network with multiple layers for sequence processing."""
    def __init__(self, in_dim=16, hidden_dim=32, num_layers=2, out_dim=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, return_hidden: bool = False):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, in_dim) or (batch, in_dim)
            return_hidden: If True, return layer-wise hidden states
        """
        # Handle single vector input
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (batch, 1, in_dim)

        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the last timestep output
        last_output = lstm_out[:, -1, :]  # (batch, hidden_dim)

        # Final linear layer
        out = self.fc(last_output)

        if return_hidden:
            # Return layer-wise hidden states
            hidden_states = [x.squeeze(1)]  # Input

            # Extract hidden states from each LSTM layer
            for layer_idx in range(self.num_layers):
                hidden_states.append(h_n[layer_idx])  # (batch, hidden_dim)

            hidden_states.append(out)  # Output
            return out, hidden_states

        return out


# ====================
# Training Models for Lab
# ====================

class LSTMLanguageModel(nn.Module):
    """LSTM language model for next-word prediction."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))  # (batch, seq_len, embedding_dim)
        output, hidden = self.lstm(embedded, hidden)  # output: (batch, seq_len, hidden_dim)
        output = self.dropout(output)
        logits = self.fc(output)  # (batch, seq_len, vocab_size)
        return logits, hidden


class TinyGPT(nn.Module):
    """Small GPT-style decoder-only transformer."""
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, context_length, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(context_length, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.context_length = context_length

    def forward(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape

        # Create causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        # Embeddings
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        embedded = self.dropout(self.embedding(x) + self.pos_embedding(positions))

        # Transformer with causal mask
        output = self.transformer(embedded, mask=mask, is_causal=True)
        logits = self.fc(output)  # (batch, seq_len, vocab_size)
        return logits


class MaskedGPT(nn.Module):
    """GPT-style encoder with masked language modeling (like BERT)."""
    def __init__(self, vocab_size, embedding_dim, num_heads, num_layers, context_length, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(context_length, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape

        # Embeddings (no causal mask - bidirectional)
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        embedded = self.dropout(self.embedding(x) + self.pos_embedding(positions))

        # Transformer without causal mask (bidirectional attention)
        output = self.transformer(embedded)
        logits = self.fc(output)  # (batch, seq_len, vocab_size)
        return logits


def build_lstm_graph(
    layer_sizes: List[int],
    weight_mats: List[np.ndarray],
    top_k: int = 5,
    num_timesteps: int = 3,
):
    """
    Build graph for LSTM showing recurrent connections across time.
    Shows the unrolled LSTM across multiple timesteps to visualize recurrence.

    layer_sizes: [in_dim, hidden_dim_1, ..., hidden_dim_n, out_dim]
    weight_mats: list of weight matrices
    num_timesteps: number of time steps to unroll
    """
    node_x, node_y, node_z = [], [], []
    node_layer_idx, node_neuron_idx, node_timestep = [], [], []
    node_labels = []
    layer_offsets = []

    # Create nodes across timesteps and layers
    for t in range(num_timesteps):
        for l, size in enumerate(layer_sizes):
            layer_offsets.append(len(node_x))
            for i in range(size):
                x = float(t * len(layer_sizes)) + float(l)  # Timestep offset + layer
                y = float(i - size / 2.0)
                z = float(t) * 2.0  # Spread timesteps in Z dimension
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)
                node_layer_idx.append(l)
                node_neuron_idx.append(i)
                node_timestep.append(t)
                node_labels.append(f"t={t}, Layer {l}, neuron {i}")

    nodes = {
        "x": np.array(node_x),
        "y": np.array(node_y),
        "z": np.array(node_z),
        "layer_idx": np.array(node_layer_idx),
        "neuron_idx": np.array(node_neuron_idx),
        "timestep": np.array(node_timestep),
        "labels": np.array(node_labels),
        "layer_offsets": layer_offsets,
        "num_timesteps": num_timesteps,
        "is_lstm": True,
    }

    # Build edges: feedforward + recurrent
    edge_x, edge_y, edge_z = [], [], []
    edge_weights = []

    for t in range(num_timesteps):
        # Feedforward edges within timestep
        for l_idx, W in enumerate(weight_mats):
            in_size = layer_sizes[l_idx]
            out_size = layer_sizes[l_idx + 1]

            W_abs = np.abs(W)

            for out_i in range(out_size):
                top_indices = np.argsort(W_abs[out_i])[-top_k:]

                for in_i in top_indices:
                    # Calculate node indices
                    in_node_idx = t * sum(layer_sizes) + sum(layer_sizes[:l_idx]) + in_i
                    out_node_idx = t * sum(layer_sizes) + sum(layer_sizes[:l_idx+1]) + out_i

                    if in_node_idx < len(nodes["x"]) and out_node_idx < len(nodes["x"]):
                        edge_x.extend([nodes["x"][in_node_idx], nodes["x"][out_node_idx], None])
                        edge_y.extend([nodes["y"][in_node_idx], nodes["y"][out_node_idx], None])
                        edge_z.extend([nodes["z"][in_node_idx], nodes["z"][out_node_idx], None])
                        edge_weights.append(float(W[out_i, in_i]))

        # Recurrent edges from t to t+1 for LSTM hidden layers
        if t < num_timesteps - 1:
            for l_idx in range(1, len(layer_sizes) - 1):  # Only hidden layers
                size = layer_sizes[l_idx]
                # Connect hidden state at time t to same position at time t+1
                for i in range(min(size, 10)):  # Limit for visualization
                    curr_node_idx = t * sum(layer_sizes) + sum(layer_sizes[:l_idx]) + i
                    next_node_idx = (t + 1) * sum(layer_sizes) + sum(layer_sizes[:l_idx]) + i

                    if curr_node_idx < len(nodes["x"]) and next_node_idx < len(nodes["x"]):
                        edge_x.extend([nodes["x"][curr_node_idx], nodes["x"][next_node_idx], None])
                        edge_y.extend([nodes["y"][curr_node_idx], nodes["y"][next_node_idx], None])
                        edge_z.extend([nodes["z"][curr_node_idx], nodes["z"][next_node_idx], None])
                        edge_weights.append(0.5)  # Recurrent connection indicator

    edges = {
        "x": np.array(edge_x),
        "y": np.array(edge_y),
        "z": np.array(edge_z),
        "weights": np.array(edge_weights),
    }

    return nodes, edges


def build_network_graph(
    layer_sizes: List[int],
    weight_mats: List[np.ndarray],
    top_k: int = 5,
):
    """
    Multi-layer graph for a chain of linear layers defined by weight matrices.
    layer_sizes: e.g., [in_dim, h1, h2, out_dim]
    weight_mats: list of W[l] with shape [out_dim_l, in_dim_l]
    """
    assert len(weight_mats) == len(layer_sizes) - 1

    node_x, node_y, node_z = [], [], []
    node_layer_idx, node_neuron_idx = [], []
    node_labels = []
    layer_offsets = []

    for l, size in enumerate(layer_sizes):
        layer_offsets.append(len(node_x))
        for i in range(size):
            x = float(l)
            y = float(i - size / 2.0)
            z = 0.0
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_layer_idx.append(l)
            node_neuron_idx.append(i)
            node_labels.append(f"Layer {l}, neuron {i}")

    nodes = {
        "x": np.array(node_x),
        "y": np.array(node_y),
        "z": np.array(node_z),
        "layer_idx": np.array(node_layer_idx),
        "neuron_idx": np.array(node_neuron_idx),
        "labels": np.array(node_labels),
        "layer_offsets": layer_offsets,
        "is_lstm": False,
    }

    edge_x, edge_y, edge_z = [], [], []

    for l, W in enumerate(weight_mats):
        out_size, in_size = W.shape
        base_src = layer_offsets[l]
        base_dst = layer_offsets[l + 1]

        for j in range(out_size):
            w_row = W[j]
            if top_k > 0 and top_k < len(w_row):
                idxs = np.argpartition(np.abs(w_row), -top_k)[-top_k:]
            else:
                idxs = np.arange(len(w_row))
            for i in idxs:
                src = base_src + i
                dst = base_dst + j
                edge_x += [node_x[src], node_x[dst], None]
                edge_y += [node_y[src], node_y[dst], None]
                edge_z += [node_z[src], node_z[dst], None]

    edges = {
        "x": np.array(edge_x, dtype=float),
        "y": np.array(edge_y, dtype=float),
        "z": np.array(edge_z, dtype=float),
    }

    return nodes, edges


def compute_mean_activations_tinymlp(
    model: TinyMLP,
    in_dim: int,
    n_samples: int,
) -> List[np.ndarray]:
    model.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, in_dim)
        _, acts = model(x, return_hidden=True)
    means = [a.abs().mean(dim=0).cpu().numpy() for a in acts]
    return means


def compute_single_input_activations_tinymlp(
    model: TinyMLP,
    x: torch.Tensor,
) -> List[np.ndarray]:
    """Compute activations for a single input (for animation)."""
    model.eval()
    with torch.no_grad():
        _, acts = model(x, return_hidden=True)
    return [a.abs().cpu().numpy() for a in acts]


def create_animated_tinymlp_figure(
    nodes,
    edges,
    layer_sizes: List[int],
    all_activations: List[List[np.ndarray]],
    show_edges: bool = True,
    log_color: bool = False,
):
    """Create animated 3D figure showing tokens passing through TinyMLP."""
    frames = []

    for frame_idx, layer_acts in enumerate(all_activations):
        # Convert single-input activations to mean format (squeeze batch dimension)
        layer_means = [act.squeeze() if act.ndim > 1 else act for act in layer_acts]
        node_colors = activations_to_node_colors_full(nodes, layer_sizes, layer_means)

        if log_color:
            node_colors = np.log1p(node_colors)

        if node_colors.max() > 0:
            node_colors_norm = node_colors / node_colors.max()
        else:
            node_colors_norm = node_colors

        # Build frame data with both edges and nodes
        frame_data = []

        if show_edges and len(edges["x"]) > 0:
            frame_data.append(
                go.Scatter3d(
                    x=edges["x"],
                    y=edges["y"],
                    z=edges["z"],
                    mode="lines",
                    line=dict(width=0.5, color="rgba(100,100,100,0.3)"),
                    hoverinfo="none",
                    name="Weights",
                )
            )

        frame_data.append(
            go.Scatter3d(
                x=nodes["x"],
                y=nodes["y"],
                z=nodes["z"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=node_colors_norm,
                    colorscale="Hot",
                    cmin=0,
                    cmax=1,
                    colorbar=dict(title="Activation"),
                    opacity=0.9,
                ),
                text=nodes["labels"],
                hoverinfo="text",
                name="Neurons",
            )
        )

        frame = go.Frame(
            data=frame_data,
            name=f"frame_{frame_idx}",
            layout=go.Layout(
                title=f"Token Flow - Input {frame_idx + 1}/{len(all_activations)}"
            )
        )
        frames.append(frame)

    # Initial frame
    initial_layer_means = [act.squeeze() if act.ndim > 1 else act for act in all_activations[0]]
    initial_colors = activations_to_node_colors_full(nodes, layer_sizes, initial_layer_means)
    if log_color:
        initial_colors = np.log1p(initial_colors)
    if initial_colors.max() > 0:
        initial_colors = initial_colors / initial_colors.max()

    data = []
    if show_edges and len(edges["x"]) > 0:
        edge_trace = go.Scatter3d(
            x=edges["x"],
            y=edges["y"],
            z=edges["z"],
            mode="lines",
            line=dict(width=0.5, color="rgba(100,100,100,0.3)"),
            hoverinfo="none",
            name="Weights",
        )
        data.append(edge_trace)

    node_trace = go.Scatter3d(
        x=nodes["x"],
        y=nodes["y"],
        z=nodes["z"],
        mode="markers",
        marker=dict(
            size=8,
            color=initial_colors,
            colorscale="Hot",
            cmin=0,
            cmax=1,
            colorbar=dict(title="Activation"),
            opacity=0.9,
        ),
        text=nodes["labels"],
        hoverinfo="text",
        name="Neurons",
    )
    data.append(node_trace)

    fig = go.Figure(
        data=data,
        frames=frames,
        layout=go.Layout(
            title="TinyMLP Token Flow Animation",
            showlegend=True,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=True,
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=300, redraw=True),
                                    fromcurrent=True,
                                    mode="immediate",
                                    transition=dict(duration=200, easing="linear"),
                                )
                            ],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                )
                            ],
                        ),
                    ],
                    x=0.1,
                    y=1.15,
                ),
            ],
            sliders=[
                dict(
                    active=0,
                    steps=[
                        dict(
                            args=[
                                [f"frame_{i}"],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                )
                            ],
                            label=f"Input {i+1}",
                            method="animate",
                        )
                        for i in range(len(all_activations))
                    ],
                    x=0.1,
                    y=0,
                    len=0.9,
                    currentvalue=dict(prefix="Frame: ", visible=True),
                )
            ],
            scene=dict(
                xaxis=dict(title="Layer", showgrid=False, zeroline=False),
                yaxis=dict(title="Neuron index", showgrid=False, zeroline=False),
                zaxis=dict(title="Z", showgrid=False, zeroline=False),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        ),
    )

    return fig


def activations_to_node_colors_full(
    nodes,
    layer_sizes: List[int],
    layer_means: List[np.ndarray],
) -> np.ndarray:
    colors = np.zeros(len(nodes["x"]), dtype=float)
    offsets = nodes["layer_offsets"]
    for l, size in enumerate(layer_sizes):
        start = offsets[l]
        end = start + size
        values = layer_means[l]
        if len(values) > size:
            values = values[:size]
        elif len(values) < size:
            pad = np.zeros(size)
            pad[: len(values)] = values
            values = pad
        colors[start:end] = values
    return colors


# =====================================================
# 2. Hugging Face models (multi-block bipartite mode)
# =====================================================

HF_MODELS = {
    "Qwen2.5-0.5B": {
        "hf_id": "Qwen/Qwen2.5-0.5B",
        "task": "causal",
        "layer_pattern": "mlp.up_proj",
    },
    "TinyLlama-1.1B-Chat": {
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "task": "causal",
        "layer_pattern": "mlp.up_proj",
    },
    "BERT-base-uncased": {
        "hf_id": "google-bert/bert-base-uncased",
        "task": "encoder",
        "layer_pattern": "intermediate.dense",
    },
    "GPT2-small": {
        "hf_id": "openai-community/gpt2",
        "task": "causal",
        "layer_pattern": "mlp.c_fc",
    },
}


def load_hf_model_and_tokenizer(model_key: str):
    cfg = HF_MODELS[model_key]
    hf_id = cfg["hf_id"]
    task = cfg["task"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if task == "causal":
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.float32,
            device_map=None,
        )
    else:
        model = AutoModel.from_pretrained(
            hf_id,
            torch_dtype=torch.float32,
            device_map=None,
        )

    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model.to(device)
    model.eval()

    return model, tokenizer, device


def get_candidate_linear_layers(model: nn.Module, pattern: str):
    """
    Return list of (name, module) where name contains pattern and module is Linear.
    """
    candidates: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and pattern in name:
            candidates.append((name, module))
    return candidates


def build_multi_block_graph(
    weights_list: List[np.ndarray],
    max_in: int,
    max_out: int,
    top_k: int,
):
    """
    Build a multi-block bipartite graph.

    For block b:
      - inputs at x = 2*b
      - outputs at x = 2*b + 1
    Returns:
      nodes: dict with node positions and metadata
      edges: dict with edge positions
      block_meta: list of dicts with per-block index ranges
      trunc_weights: list of truncated W for each block
    """
    node_x, node_y, node_z = [], [], []
    node_labels = []
    node_block_idx = []
    node_io_role = []  # 0 = input side, 1 = output side

    block_meta = []
    trunc_weights = []

    edge_x, edge_y, edge_z = [], [], []

    current_node_idx = 0

    for b, W in enumerate(weights_list):
        out_dim, in_dim = W.shape

        in_dim_trunc = min(in_dim, max_in)
        out_dim_trunc = min(out_dim, max_out)
        W_trunc = W[:out_dim_trunc, :in_dim_trunc]
        trunc_weights.append(W_trunc)

        # Input nodes for this block
        in_start_idx = current_node_idx
        for i in range(in_dim_trunc):
            x = float(2 * b)
            y = float(i - in_dim_trunc / 2.0)
            z = 0.0
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_labels.append(f"block {b} in {i}")
            node_block_idx.append(b)
            node_io_role.append(0)
            current_node_idx += 1

        # Output nodes for this block
        out_start_idx = current_node_idx
        for j in range(out_dim_trunc):
            x = float(2 * b + 1)
            y = float(j - out_dim_trunc / 2.0)
            z = 0.0
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            node_labels.append(f"block {b} out {j}")
            node_block_idx.append(b)
            node_io_role.append(1)
            current_node_idx += 1

        block_meta.append(
            dict(
                block_idx=b,
                in_start=in_start_idx,
                in_dim=in_dim_trunc,
                out_start=out_start_idx,
                out_dim=out_dim_trunc,
            )
        )

        # Edges for this block: top-k per output neuron
        for j in range(out_dim_trunc):
            w_row = W_trunc[j]  # [in_dim_trunc]
            if top_k > 0 and top_k < len(w_row):
                idxs = np.argpartition(np.abs(w_row), -top_k)[-top_k:]
            else:
                idxs = np.arange(len(w_row))
            for i in idxs:
                src = in_start_idx + i
                dst = out_start_idx + j
                edge_x += [node_x[src], node_x[dst], None]
                edge_y += [node_y[src], node_y[dst], None]
                edge_z += [node_z[src], node_z[dst], None]

    nodes = {
        "x": np.array(node_x),
        "y": np.array(node_y),
        "z": np.array(node_z),
        "labels": np.array(node_labels),
        "block_idx": np.array(node_block_idx),
        "io_role": np.array(node_io_role),
    }
    edges = {
        "x": np.array(edge_x),
        "y": np.array(edge_y),
        "z": np.array(edge_z),
    }

    return nodes, edges, block_meta, trunc_weights


def compute_hf_multi_layer_activations(
    model: nn.Module,
    tokenizer,
    device,
    modules: List[nn.Linear],
    prompts: List[str],
    max_length: int = 64,
) -> List[np.ndarray]:
    """
    Hook multiple linear layers at once and compute mean |activation| per output neuron
    for each module. Returns list of arrays [out_dim].
    """
    storages: List[List[torch.Tensor]] = [[] for _ in modules]
    handles = []

    def make_hook(idx):
        def hook(module, inp, out):
            with torch.no_grad():
                storages[idx].append(out.detach().cpu())
        return hook

    for i, m in enumerate(modules):
        handles.append(m.register_forward_hook(make_hook(i)))

    try:
        if len(prompts) == 0:
            prompts = ["Hello world"]

        enc = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            _ = model(**enc)

        results = []
        for s in storages:
            if len(s) == 0:
                results.append(None)
            else:
                acts = torch.cat(
                    [a.reshape(-1, a.shape[-1]) for a in s],
                    dim=0,
                )
                means = acts.abs().mean(dim=0).cpu().numpy()
                results.append(means)
        return results
    finally:
        for h in handles:
            h.remove()


def compute_hf_token_by_token_activations(
    model: nn.Module,
    tokenizer,
    device,
    modules: List[nn.Linear],
    prompt: str,
    max_length: int = 64,
) -> Tuple[List[List[np.ndarray]], List[str]]:
    """
    Compute activations token-by-token for animation.
    Returns: (list of snapshots, token texts)
    """
    tokens = tokenizer.encode(prompt, add_special_tokens=True, truncation=True, max_length=max_length)
    token_texts = [tokenizer.decode([t]) for t in tokens]

    all_snapshots = []

    for i in range(1, len(tokens) + 1):
        token_slice = tokens[:i]

        storages: List[List[torch.Tensor]] = [[] for _ in modules]
        handles = []

        def make_hook(idx):
            def hook(module, inp, out):
                with torch.no_grad():
                    storages[idx].append(out.detach().cpu())
            return hook

        for j, m in enumerate(modules):
            handles.append(m.register_forward_hook(make_hook(j)))

        try:
            input_ids = torch.tensor([token_slice]).to(device)
            with torch.no_grad():
                _ = model(input_ids=input_ids)

            snapshot = []
            for s in storages:
                if len(s) == 0:
                    snapshot.append(None)
                else:
                    # Take the last token's activation
                    acts = s[0][:, -1, :]  # [batch, seq, hidden] -> [batch, hidden]
                    snapshot.append(acts.abs().mean(dim=0).cpu().numpy())

            all_snapshots.append(snapshot)
        finally:
            for h in handles:
                h.remove()

    return all_snapshots, token_texts


def learn_neuron_dictionary(
    activation_samples: List[np.ndarray],
    n_components: int = 32,
    alpha: float = 1.0,
    max_iter: int = 100,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Learn a sparse dictionary to represent neuron activations.

    Args:
        activation_samples: List of activation vectors [n_samples, n_neurons]
        n_components: Number of dictionary atoms (semantic features)
        alpha: Sparsity parameter (higher = sparser)
        max_iter: Maximum iterations for dictionary learning
        random_state: Random seed for reproducibility

    Returns:
        dictionary: Learned dictionary [n_components, n_neurons]
        codes: Sparse codes for each sample [n_samples, n_components]
    """
    if not SKLEARN_AVAILABLE:
        return None, None

    X = np.array(activation_samples)

    # Normalize to improve learning
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X_normalized = (X - X_mean) / X_std

    # Learn dictionary
    dict_learner = DictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        max_iter=max_iter,
        fit_algorithm='cd',  # Use coordinate descent which supports positive constraints
        transform_algorithm='lasso_cd',
        random_state=random_state,
        positive_code=True,  # Force positive activations
        positive_dict=False,
    )

    codes = dict_learner.fit_transform(X_normalized)
    dictionary = dict_learner.components_

    return dictionary, codes


def interpret_dictionary_atoms(
    dictionary: np.ndarray,
    codes: np.ndarray,
    activation_samples: List[np.ndarray],
    top_k: int = 5,
    tokens: List[str] = None,
) -> List[Dict]:
    """
    Interpret what each dictionary atom (semantic feature) represents.

    Args:
        dictionary: [n_components, n_neurons]
        codes: [n_samples, n_components]
        activation_samples: Original activation data
        top_k: Number of top neurons per atom
        tokens: Optional list of token/prompt strings for each sample

    Returns:
        List of dicts with interpretation for each atom
    """
    n_components = dictionary.shape[0]
    interpretations = []

    for atom_idx in range(n_components):
        atom_vector = dictionary[atom_idx]

        # Find neurons with highest weights in this atom
        top_neuron_indices = np.argsort(np.abs(atom_vector))[-top_k:][::-1]
        top_neuron_weights = atom_vector[top_neuron_indices]

        # Compute how often this atom is activated
        atom_codes = codes[:, atom_idx]
        activation_frequency = (atom_codes > 0.1).mean()
        avg_activation = atom_codes[atom_codes > 0.1].mean() if activation_frequency > 0 else 0

        # Get top activating tokens if available
        top_tokens = []
        if tokens is not None and len(tokens) == len(atom_codes):
            # Find samples where this feature is most active
            top_sample_indices = np.argsort(atom_codes)[-15:][::-1]  # Top 15 samples
            top_tokens = [
                {
                    'token': tokens[idx],
                    'activation': float(atom_codes[idx])
                }
                for idx in top_sample_indices
                if atom_codes[idx] > 0.1  # Only include if meaningfully active
            ]
        elif tokens is not None:
            # Debug: length mismatch
            print(f"Warning: tokens length ({len(tokens)}) != codes length ({len(atom_codes)})")

        interpretations.append({
            'atom_idx': atom_idx,
            'top_neurons': top_neuron_indices.tolist(),
            'top_weights': top_neuron_weights.tolist(),
            'activation_frequency': float(activation_frequency),
            'avg_activation': float(avg_activation),
            'l2_norm': float(np.linalg.norm(atom_vector)),
            'top_tokens': top_tokens,
        })

    return interpretations


def get_semantic_features_for_neurons(
    neuron_activations: np.ndarray,
    dictionary: np.ndarray,
    top_k: int = 3,
) -> List[Tuple[int, float]]:
    """
    Find which semantic features (dictionary atoms) are most active for given neuron activations.

    Args:
        neuron_activations: Single activation vector [n_neurons]
        dictionary: Learned dictionary [n_components, n_neurons]
        top_k: Number of top features to return

    Returns:
        List of (feature_idx, activation_strength) tuples
    """
    if not SKLEARN_AVAILABLE or dictionary is None:
        return []

    # Encode the activation using the dictionary
    coder = SparseCoder(dictionary, transform_algorithm='lasso_lars', transform_alpha=1.0, positive_code=True)
    code = coder.transform(neuron_activations.reshape(1, -1))

    # Find top activated features
    feature_activations = code[0]
    top_feature_indices = np.argsort(feature_activations)[-top_k:][::-1]

    result = []
    for idx in top_feature_indices:
        if feature_activations[idx] > 0.01:  # Threshold
            result.append((int(idx), float(feature_activations[idx])))

    return result


def create_animated_hf_figure(
    nodes,
    edges,
    block_meta,
    trunc_weights,
    all_activations: List[List[np.ndarray]],
    token_texts: List[str],
):
    """Create animated figure showing tokens passing through transformer blocks."""
    frames = []

    for frame_idx, acts_list in enumerate(all_activations):
        node_colors = np.zeros(len(nodes["x"]), dtype=float)

        for b, meta in enumerate(block_meta):
            in_start = meta["in_start"]
            in_dim = meta["in_dim"]
            out_start = meta["out_start"]
            out_dim = meta["out_dim"]

            W_trunc = trunc_weights[b]
            out_acts = acts_list[b]

            if out_acts is None:
                in_c = np.abs(W_trunc).mean(axis=0)
                out_c = np.abs(W_trunc).mean(axis=1)
            else:
                in_c, out_c = activations_to_bipartite_colors(
                    W=W_trunc,
                    out_acts=out_acts,
                )

            in_c = np.maximum(in_c, 0)
            out_c = np.maximum(out_c, 0)

            if in_c.max() > 0:
                in_c = in_c / in_c.max()
            if out_c.max() > 0:
                out_c = out_c / out_c.max()

            node_colors[in_start : in_start + in_dim] = in_c[:in_dim]
            node_colors[out_start : out_start + out_dim] = out_c[:out_dim]

        token_preview = " ".join(token_texts[:frame_idx + 1])

        # Build frame data with both edges and nodes
        frame_data = []

        if len(edges["x"]) > 0:
            frame_data.append(
                go.Scatter3d(
                    x=edges["x"],
                    y=edges["y"],
                    z=edges["z"],
                    mode="lines",
                    line=dict(width=0.5, color="rgba(100,100,100,0.2)"),
                    hoverinfo="none",
                    name="Weights",
                )
            )

        frame_data.append(
            go.Scatter3d(
                x=nodes["x"],
                y=nodes["y"],
                z=nodes["z"],
                mode="markers",
                marker=dict(
                    size=6,
                    color=node_colors,
                    colorscale="Hot",
                    cmin=0,
                    cmax=1,
                    colorbar=dict(title="Activation"),
                    opacity=0.9,
                ),
                text=nodes["labels"],
                hoverinfo="text",
                name="Neurons",
            )
        )

        frame = go.Frame(
            data=frame_data,
            name=f"frame_{frame_idx}",
            layout=go.Layout(
                title=f"Token {frame_idx + 1}/{len(all_activations)}: '{token_preview[:50]}...'"
            )
        )
        frames.append(frame)

    # Initial frame
    initial_colors = np.zeros(len(nodes["x"]), dtype=float)
    for b, meta in enumerate(block_meta):
        in_start = meta["in_start"]
        in_dim = meta["in_dim"]
        out_start = meta["out_start"]
        out_dim = meta["out_dim"]

        W_trunc = trunc_weights[b]
        out_acts = all_activations[0][b]

        if out_acts is None:
            in_c = np.abs(W_trunc).mean(axis=0)
            out_c = np.abs(W_trunc).mean(axis=1)
        else:
            in_c, out_c = activations_to_bipartite_colors(W=W_trunc, out_acts=out_acts)

        in_c = np.maximum(in_c, 0)
        out_c = np.maximum(out_c, 0)
        if in_c.max() > 0:
            in_c = in_c / in_c.max()
        if out_c.max() > 0:
            out_c = out_c / out_c.max()

        initial_colors[in_start : in_start + in_dim] = in_c[:in_dim]
        initial_colors[out_start : out_start + out_dim] = out_c[:out_dim]

    data = []
    if len(edges["x"]) > 0:
        edge_trace = go.Scatter3d(
            x=edges["x"],
            y=edges["y"],
            z=edges["z"],
            mode="lines",
            line=dict(width=0.5, color="rgba(100,100,100,0.2)"),
            hoverinfo="none",
            name="Weights",
        )
        data.append(edge_trace)

    node_trace = go.Scatter3d(
        x=nodes["x"],
        y=nodes["y"],
        z=nodes["z"],
        mode="markers",
        marker=dict(
            size=6,
            color=initial_colors,
            colorscale="Hot",
            cmin=0,
            cmax=1,
            colorbar=dict(title="Activation"),
            opacity=0.9,
        ),
        text=nodes["labels"],
        hoverinfo="text",
        name="Neurons",
    )
    data.append(node_trace)

    fig = go.Figure(
        data=data,
        frames=frames,
        layout=go.Layout(
            title="Transformer Token Flow Animation",
            showlegend=True,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=True,
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=500, redraw=True),
                                    fromcurrent=True,
                                    mode="immediate",
                                    transition=dict(duration=300, easing="cubic-in-out"),
                                )
                            ],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                )
                            ],
                        ),
                    ],
                    x=0.1,
                    y=1.15,
                ),
            ],
            sliders=[
                dict(
                    active=0,
                    steps=[
                        dict(
                            args=[
                                [f"frame_{i}"],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                )
                            ],
                            label=token_texts[i] if i < len(token_texts) else f"Token {i+1}",
                            method="animate",
                        )
                        for i in range(len(all_activations))
                    ],
                    x=0.1,
                    y=0,
                    len=0.9,
                    currentvalue=dict(prefix="Token: ", visible=True),
                )
            ],
            scene=dict(
                xaxis=dict(
                    title="2*b (inputs) / 2*b+1 (outputs)",
                    showgrid=False,
                    zeroline=False,
                ),
                yaxis=dict(title="Neuron index", showgrid=False, zeroline=False),
                zaxis=dict(title="Z", showgrid=False, zeroline=False),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        ),
    )

    return fig


def activations_to_bipartite_colors(W: np.ndarray, out_acts: np.ndarray):
    """
    W: [out_dim, in_dim] (truncated)
    out_acts: [out_dim]
    Returns:
        in_colors: [in_dim]
        out_colors: [out_dim]
    """
    out_dim, in_dim = W.shape
    out_acts = out_acts[:out_dim]

    out_colors = out_acts
    in_colors = np.abs(W).T @ out_acts  # [in_dim]

    return in_colors, out_colors


# =====================================================
# 2b. Attention Head Visualization
# =====================================================

def get_attention_layers(model: nn.Module, pattern: str = "attn"):
    """
    Extract attention layers based on pattern.
    Returns list of (layer_name, q_proj, k_proj, v_proj, o_proj) tuples.
    """
    attention_modules = []

    # Get all transformer layers
    for name, module in model.named_modules():
        if "self_attn" in name or "attention" in name.lower():
            # Try to find Q, K, V, O projections
            q_proj = None
            k_proj = None
            v_proj = None
            o_proj = None

            for subname, submodule in module.named_modules():
                if isinstance(submodule, nn.Linear):
                    if "q_proj" in subname or subname == "query":
                        q_proj = submodule
                    elif "k_proj" in subname or subname == "key":
                        k_proj = submodule
                    elif "v_proj" in subname or subname == "value":
                        v_proj = submodule
                    elif "o_proj" in subname or "out_proj" in subname or subname == "dense":
                        o_proj = submodule

            if q_proj is not None and k_proj is not None and v_proj is not None:
                attention_modules.append((name, q_proj, k_proj, v_proj, o_proj))

    return attention_modules


def build_attention_head_graph(
    q_weight: np.ndarray,
    k_weight: np.ndarray,
    v_weight: np.ndarray,
    o_weight: np.ndarray,
    num_heads: int,
    head_dim: int,
    max_input_neurons: int = 64,
    top_k: int = 5,
):
    """
    Build 3D graph for attention mechanism showing:
    - Input embeddings (left)
    - Q, K, V projections (middle) organized by head
    - Output projection (right)

    Args:
        q_weight: [hidden_dim, embed_dim]
        k_weight: [hidden_dim, embed_dim]
        v_weight: [hidden_dim, embed_dim]
        o_weight: [embed_dim, hidden_dim] or None
        num_heads: number of attention heads
        head_dim: dimension per head
    """
    embed_dim = q_weight.shape[1]
    hidden_dim = q_weight.shape[0]

    # Truncate input dimension if needed
    embed_dim_trunc = min(embed_dim, max_input_neurons)

    node_x, node_y, node_z = [], [], []
    node_labels = []
    node_colors_default = []

    # Layer 0: Input embeddings (x=0)
    input_start = len(node_x)
    for i in range(embed_dim_trunc):
        node_x.append(0)
        node_y.append(i - embed_dim_trunc / 2)
        node_z.append(0)
        node_labels.append(f"Input {i}")
        node_colors_default.append(0.3)

    # Layer 1: Q, K, V heads (x=2, 3, 4)
    qkv_start = len(node_x)

    # Q heads at x=2
    q_start = len(node_x)
    for h in range(num_heads):
        for d in range(head_dim):
            node_x.append(2)
            node_y.append(d - head_dim / 2)
            node_z.append(h - num_heads / 2)
            node_labels.append(f"Q_head{h}_{d}")
            node_colors_default.append(0.5)

    # K heads at x=3
    k_start = len(node_x)
    for h in range(num_heads):
        for d in range(head_dim):
            node_x.append(3)
            node_y.append(d - head_dim / 2)
            node_z.append(h - num_heads / 2)
            node_labels.append(f"K_head{h}_{d}")
            node_colors_default.append(0.5)

    # V heads at x=4
    v_start = len(node_x)
    for h in range(num_heads):
        for d in range(head_dim):
            node_x.append(4)
            node_y.append(d - head_dim / 2)
            node_z.append(h - num_heads / 2)
            node_labels.append(f"V_head{h}_{d}")
            node_colors_default.append(0.5)

    # Layer 2: Output (x=6) if o_weight exists
    output_start = len(node_x)
    if o_weight is not None:
        out_dim = min(o_weight.shape[0], max_input_neurons)
        for i in range(out_dim):
            node_x.append(6)
            node_y.append(i - out_dim / 2)
            node_z.append(0)
            node_labels.append(f"Output {i}")
            node_colors_default.append(0.3)

    nodes = {
        "x": np.array(node_x),
        "y": np.array(node_y),
        "z": np.array(node_z),
        "labels": node_labels,
        "colors": np.array(node_colors_default),
        "input_start": input_start,
        "q_start": q_start,
        "k_start": k_start,
        "v_start": v_start,
        "output_start": output_start,
        "num_heads": num_heads,
        "head_dim": head_dim,
    }

    # Build edges
    edge_x, edge_y, edge_z = [], [], []

    # Connect inputs to Q heads (top-k strongest connections per Q neuron)
    q_weight_trunc = q_weight[:, :embed_dim_trunc]
    for h in range(num_heads):
        for d in range(head_dim):
            q_idx = h * head_dim + d
            if q_idx < q_weight_trunc.shape[0]:
                w_row = q_weight_trunc[q_idx]
                if top_k > 0 and top_k < len(w_row):
                    idxs = np.argpartition(np.abs(w_row), -top_k)[-top_k:]
                else:
                    idxs = np.arange(len(w_row))

                for i in idxs:
                    src = input_start + i
                    dst = q_start + h * head_dim + d
                    edge_x += [node_x[src], node_x[dst], None]
                    edge_y += [node_y[src], node_y[dst], None]
                    edge_z += [node_z[src], node_z[dst], None]

    # Connect inputs to K heads
    k_weight_trunc = k_weight[:, :embed_dim_trunc]
    for h in range(num_heads):
        for d in range(head_dim):
            k_idx = h * head_dim + d
            if k_idx < k_weight_trunc.shape[0]:
                w_row = k_weight_trunc[k_idx]
                if top_k > 0 and top_k < len(w_row):
                    idxs = np.argpartition(np.abs(w_row), -top_k)[-top_k:]
                else:
                    idxs = np.arange(len(w_row))

                for i in idxs:
                    src = input_start + i
                    dst = k_start + h * head_dim + d
                    edge_x += [node_x[src], node_x[dst], None]
                    edge_y += [node_y[src], node_y[dst], None]
                    edge_z += [node_z[src], node_z[dst], None]

    # Connect inputs to V heads
    v_weight_trunc = v_weight[:, :embed_dim_trunc]
    for h in range(num_heads):
        for d in range(head_dim):
            v_idx = h * head_dim + d
            if v_idx < v_weight_trunc.shape[0]:
                w_row = v_weight_trunc[v_idx]
                if top_k > 0 and top_k < len(w_row):
                    idxs = np.argpartition(np.abs(w_row), -top_k)[-top_k:]
                else:
                    idxs = np.arange(len(w_row))

                for i in idxs:
                    src = input_start + i
                    dst = v_start + h * head_dim + d
                    edge_x += [node_x[src], node_x[dst], None]
                    edge_y += [node_y[src], node_y[dst], None]
                    edge_z += [node_z[src], node_z[dst], None]

    # Connect V heads to output
    if o_weight is not None:
        o_weight_trunc = o_weight[:output_start - len(node_x) if output_start > 0 else o_weight.shape[0], :]
        out_dim = output_start - len(node_x) + len(node_labels) - output_start
        for i in range(min(out_dim, o_weight_trunc.shape[0])):
            w_col = o_weight_trunc[i, :]
            if top_k > 0 and top_k < len(w_col):
                idxs = np.argpartition(np.abs(w_col), -top_k)[-top_k:]
            else:
                idxs = np.arange(min(len(w_col), num_heads * head_dim))

            for v_idx in idxs:
                if v_idx < num_heads * head_dim:
                    src = v_start + v_idx
                    dst = output_start + i
                    edge_x += [node_x[src], node_x[dst], None]
                    edge_y += [node_y[src], node_y[dst], None]
                    edge_z += [node_z[src], node_z[dst], None]

    edges = {
        "x": np.array(edge_x),
        "y": np.array(edge_y),
        "z": np.array(edge_z),
    }

    return nodes, edges


def create_attention_head_figure(nodes, edges, layer_name: str, show_edges: bool = True):
    """Create 3D visualization of attention heads."""
    data = []

    if show_edges and len(edges["x"]) > 0:
        edge_trace = go.Scatter3d(
            x=edges["x"],
            y=edges["y"],
            z=edges["z"],
            mode="lines",
            line=dict(width=0.5, color="rgba(100,100,200,0.2)"),
            hoverinfo="none",
            name="Connections",
        )
        data.append(edge_trace)

    node_trace = go.Scatter3d(
        x=nodes["x"],
        y=nodes["y"],
        z=nodes["z"],
        mode="markers",
        marker=dict(
            size=4,
            color=nodes["colors"],
            colorscale="Viridis",
            colorbar=dict(title="Activation"),
            opacity=0.8,
        ),
        text=nodes["labels"],
        hoverinfo="text",
        name="Neurons",
    )
    data.append(node_trace)

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title=f"Attention Heads 3D View - {layer_name}",
            showlegend=True,
            scene=dict(
                xaxis=dict(
                    title="Stage (Input→Q/K/V→Output)",
                    showgrid=False,
                    zeroline=False,
                ),
                yaxis=dict(title="Dimension", showgrid=False, zeroline=False),
                zaxis=dict(title="Head", showgrid=False, zeroline=False),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
        ),
    )

    return fig


# =====================================================
# 3. Streamlit UI
# =====================================================

st.set_page_config(page_title="3D Model Network Viewer", layout="wide")

st.title("3D Network Visualization of Models")

mode = st.sidebar.radio(
    "Mode",
    [
        "TinyMLP full network",
        "HF transformer blocks (Qwen/TinyLlama/BERT/GPT2)",
        "Attention Heads 3D",
        "Token Journey Viewer",
        "🔬 Experimental Suite",
        "🏋️ Model Training Lab"
    ],
)

# -----------------------
# Mode 1: TinyMLP full network
# -----------------------
if mode == "TinyMLP full network":
    st.sidebar.header("Network Settings")

    network_type = st.sidebar.selectbox(
        "Network Type",
        ["MLP", "LSTM"],
        help="Choose between Multi-Layer Perceptron or LSTM network"
    )

    in_dim = st.sidebar.slider("Input dimension", 4, 64, 16, step=4)

    if network_type == "MLP":
        h1 = st.sidebar.slider("Hidden layer 1 size", 4, 128, 32, step=4)
        h2 = st.sidebar.slider("Hidden layer 2 size", 4, 128, 32, step=4)
    else:  # LSTM
        hidden_dim = st.sidebar.slider("LSTM hidden dimension", 8, 128, 32, step=8)
        num_lstm_layers = st.sidebar.slider("Number of LSTM layers", 1, 4, 2, step=1)
        num_timesteps = st.sidebar.slider("Timesteps to unroll", 1, 5, 3, step=1,
                                          help="Show recurrent connections across time")

    out_dim = st.sidebar.slider("Output dimension", 2, 64, 8, step=2)

    top_k = st.sidebar.slider("Top-k edges per neuron", 1, 20, 5, step=1)

    viz_mode = st.sidebar.radio(
        "Visualization Mode",
        ["Static", "Animated", "3D Manifold"],
        index=0,
        help="Static: average over multiple inputs. Animated: watch activations flow. 3D Manifold: surface rendering of activation patterns"
    )

    if viz_mode == "Static":
        n_samples = st.sidebar.slider("Samples for activation stats", 10, 1000, 200, step=10)
    elif viz_mode == "Animated":
        n_inputs = st.sidebar.slider("Number of animation frames", 1, 20, 5, step=1)
    else:  # 3D Manifold
        n_samples_manifold = st.sidebar.slider("Number of input samples", 10, 200, 50, step=10)

    log_color = st.sidebar.checkbox("Log-scale activations", value=False)
    show_edges = st.sidebar.checkbox("Show edges", value=True)

    if st.button(f"Generate {network_type} 3D Network ({viz_mode})"):
        # Create model based on network type
        if network_type == "MLP":
            model = TinyMLP(in_dim=in_dim, h1=h1, h2=h2, out_dim=out_dim)
            weight_mats = [
                model.fc1.weight.detach().cpu().numpy(),
                model.fc2.weight.detach().cpu().numpy(),
                model.fc3.weight.detach().cpu().numpy(),
            ]
            layer_sizes = [in_dim, h1, h2, out_dim]
        else:  # LSTM
            model = TinyLSTM(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_lstm_layers, out_dim=out_dim)
            # For LSTM, extract weight matrices from each layer
            weight_mats = []

            # Input to first LSTM layer (input to hidden)
            weight_mats.append(model.lstm.weight_ih_l0.detach().cpu().numpy()[:hidden_dim])  # First gate slice

            # LSTM layer-to-layer connections
            for i in range(num_lstm_layers - 1):
                weight_name = f'weight_hh_l{i}'
                if hasattr(model.lstm, weight_name):
                    weight_mats.append(getattr(model.lstm, weight_name).detach().cpu().numpy()[:hidden_dim])

            # Final linear layer
            weight_mats.append(model.fc.weight.detach().cpu().numpy())

            # Layer sizes: input -> LSTM layers -> output
            layer_sizes = [in_dim] + [hidden_dim] * num_lstm_layers + [out_dim]

        # Build graph based on network type
        if network_type == "LSTM":
            nodes, edges = build_lstm_graph(
                layer_sizes=layer_sizes,
                weight_mats=weight_mats,
                top_k=top_k,
                num_timesteps=num_timesteps,
            )
        else:
            nodes, edges = build_network_graph(
                layer_sizes=layer_sizes,
                weight_mats=weight_mats,
                top_k=top_k,
            )

        if viz_mode == "3D Manifold":
            # Generate 3D manifold visualization
            st.subheader("Activation Manifold Landscape")

            if network_type == "LSTM":
                # For LSTM: process sequences and show temporal dynamics
                st.write(f"**LSTM Temporal Manifold**: Processing sequences of length {num_timesteps}")

                all_temporal_activations = []  # [sample][timestep][layer] -> activation

                for _ in range(n_samples_manifold):
                    # Generate a random sequence
                    seq = torch.randn(1, num_timesteps, in_dim)  # (batch, seq_len, in_dim)

                    # Process sequence through LSTM
                    with torch.no_grad():
                        _, hidden_states = model(seq, return_hidden=True)

                    # Extract activations at each timestep
                    # For LSTM, we get hidden states at the final timestep, but we can
                    # run it step-by-step to get temporal activations
                    timestep_activations = []
                    h_state = None
                    c_state = None

                    for t in range(num_timesteps):
                        x_t = seq[:, t:t+1, :]  # Single timestep
                        lstm_out, (h_t, c_t) = model.lstm(x_t, (h_state, c_state) if h_state is not None else None)

                        # Collect layer activations at this timestep
                        layer_acts = [x_t.squeeze().detach().cpu().numpy()]
                        for layer_idx in range(num_lstm_layers):
                            layer_acts.append(h_t[layer_idx].squeeze().detach().cpu().numpy())

                        # Final output at this timestep
                        out_t = model.fc(lstm_out.squeeze())
                        layer_acts.append(out_t.detach().cpu().numpy())

                        timestep_activations.append(layer_acts)
                        h_state, c_state = h_t, c_t

                    all_temporal_activations.append(timestep_activations)

                # Build activation matrix: rows = (sample × timestep), columns = concatenated layers
                activation_matrix = []
                for sample_temporal_acts in all_temporal_activations:
                    for timestep_acts in sample_temporal_acts:
                        concat_acts = np.concatenate([act.flatten() for act in timestep_acts])
                        activation_matrix.append(concat_acts)

                activation_matrix = np.array(activation_matrix)  # [samples*timesteps, total_neurons]

                # Store in session state with temporal info
                st.session_state['manifold_data'] = {
                    'activation_matrix': activation_matrix,
                    'layer_sizes': layer_sizes,
                    'is_temporal': True,
                    'num_timesteps': num_timesteps,
                    'samples_per_timestep': n_samples_manifold
                }
            else:
                # For MLP: standard feedforward processing
                all_layer_activations = []  # [sample][layer] -> activation vector
                for _ in range(n_samples_manifold):
                    x = torch.randn(1, in_dim)
                    acts = compute_single_input_activations_tinymlp(model, x)
                    # Squeeze batch dimension
                    acts = [act.squeeze() if act.ndim > 1 else act for act in acts]
                    all_layer_activations.append(acts)

                # Build activation matrix: rows = samples, columns = concatenated neurons across layers
                activation_matrix = []
                for sample_acts in all_layer_activations:
                    # Concatenate all layers into one vector
                    concat_acts = np.concatenate([act.flatten() for act in sample_acts])
                    activation_matrix.append(concat_acts)

                activation_matrix = np.array(activation_matrix)  # [samples, total_neurons]

                # Store in session state
                st.session_state['manifold_data'] = {
                    'activation_matrix': activation_matrix,
                    'layer_sizes': layer_sizes,
                    'is_temporal': False
                }

        elif viz_mode == "Animated":
            # Generate animated visualization
            all_activations = []
            for _ in range(n_inputs):
                x = torch.randn(1, in_dim)
                acts = compute_single_input_activations_tinymlp(model, x)
                all_activations.append(acts)

            fig = create_animated_tinymlp_figure(
                nodes=nodes,
                edges=edges,
                layer_sizes=layer_sizes,
                all_activations=all_activations,
                show_edges=show_edges,
                log_color=log_color,
            )

            st.plotly_chart(fig, width='stretch')
            st.info(f"Animated {n_inputs} random inputs flowing through the network. Use play/pause controls and slider.")
        else:
            # Generate static visualization
            layer_means = compute_mean_activations_tinymlp(
                model=model,
                in_dim=in_dim,
                n_samples=n_samples,
            )
            node_colors = activations_to_node_colors_full(nodes, layer_sizes, layer_means)

            if log_color:
                node_colors = np.log1p(node_colors)

            if node_colors.max() > 0:
                node_colors_norm = node_colors / node_colors.max()
            else:
                node_colors_norm = node_colors

            node_trace = go.Scatter3d(
                x=nodes["x"],
                y=nodes["y"],
                z=nodes["z"],
                mode="markers",
                marker=dict(
                    size=6,
                    color=node_colors_norm,
                    colorscale="Viridis",
                    colorbar=dict(title="Activation"),
                    opacity=0.9,
                ),
                text=nodes["labels"],
                hoverinfo="text",
                name="Neurons",
            )

            data = []
            if show_edges and len(edges["x"]) > 0:
                # For LSTM, separate feedforward and recurrent edges
                if network_type == "LSTM" and "weights" in edges:
                    # Identify recurrent vs feedforward edges
                    feedforward_x, feedforward_y, feedforward_z = [], [], []
                    recurrent_x, recurrent_y, recurrent_z = [], [], []

                    for i in range(0, len(edges["x"]), 3):  # Groups of 3 (start, end, None)
                        if i + 2 < len(edges["x"]):
                            # Check if this is a recurrent edge (z changes significantly)
                            if edges["z"][i] is not None and edges["z"][i+1] is not None:
                                z_diff = abs(edges["z"][i+1] - edges["z"][i])
                                if z_diff > 1.0:  # Recurrent (crosses timesteps)
                                    recurrent_x.extend([edges["x"][i], edges["x"][i+1], None])
                                    recurrent_y.extend([edges["y"][i], edges["y"][i+1], None])
                                    recurrent_z.extend([edges["z"][i], edges["z"][i+1], None])
                                else:  # Feedforward
                                    feedforward_x.extend([edges["x"][i], edges["x"][i+1], None])
                                    feedforward_y.extend([edges["y"][i], edges["y"][i+1], None])
                                    feedforward_z.extend([edges["z"][i], edges["z"][i+1], None])

                    # Add feedforward edges (blue)
                    if feedforward_x:
                        ff_trace = go.Scatter3d(
                            x=feedforward_x,
                            y=feedforward_y,
                            z=feedforward_z,
                            mode="lines",
                            line=dict(width=1, color='lightblue'),
                            hoverinfo="none",
                            name="Feedforward",
                        )
                        data.append(ff_trace)

                    # Add recurrent edges (red, thicker)
                    if recurrent_x:
                        rec_trace = go.Scatter3d(
                            x=recurrent_x,
                            y=recurrent_y,
                            z=recurrent_z,
                            mode="lines",
                            line=dict(width=3, color='red', dash='dash'),
                            hoverinfo="none",
                            name="Recurrent",
                        )
                        data.append(rec_trace)
                else:
                    # Standard edges for MLP
                    edge_trace = go.Scatter3d(
                        x=edges["x"],
                        y=edges["y"],
                        z=edges["z"],
                        mode="lines",
                        line=dict(width=1),
                        hoverinfo="none",
                        name="Weights",
                    )
                    data.append(edge_trace)

            data.append(node_trace)

            # Set title and axis labels based on network type
            if network_type == "LSTM":
                title = f"LSTM 3D Network (Unrolled {num_timesteps} timesteps)"
                xaxis_title = "Time × Layer"
                zaxis_title = "Timestep"
            else:
                title = "MLP 3D Network"
                xaxis_title = "Layer"
                zaxis_title = "Z"

            fig = go.Figure(
                data=data,
                layout=go.Layout(
                    title=title,
                    showlegend=True,
                    scene=dict(
                        xaxis=dict(title=xaxis_title, showgrid=False, zeroline=False),
                        yaxis=dict(title="Neuron index", showgrid=False, zeroline=False),
                        zaxis=dict(title=zaxis_title, showgrid=False, zeroline=False),
                    ),
                    margin=dict(l=0, r=0, b=0, t=40),
                ),
            )

            st.plotly_chart(fig, width='stretch')

            st.subheader("Layer activation stats")
            for l, (size, means) in enumerate(zip(layer_sizes, layer_means)):
                st.write(
                    f"Layer {l}: size={size}, mean(|act|)≈{float(means.mean()):.4f}"
                )

    # Manifold visualization controls (outside button to preserve state)
    if viz_mode == "3D Manifold" and 'manifold_data' in st.session_state:
        st.subheader("Activation Manifold Landscape")

        activation_matrix = st.session_state['manifold_data']['activation_matrix']
        layer_sizes = st.session_state['manifold_data']['layer_sizes']
        is_temporal = st.session_state['manifold_data'].get('is_temporal', False)

        if is_temporal:
            num_timesteps = st.session_state['manifold_data']['num_timesteps']
            samples_per_timestep = st.session_state['manifold_data']['samples_per_timestep']
            st.info(f"📊 **LSTM Temporal Manifold**: {samples_per_timestep} samples × {num_timesteps} timesteps = {activation_matrix.shape[0]} total trajectories")

        # Visualization controls
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            manifold_type = st.selectbox(
                "Surface type",
                ["Surface", "Wireframe", "Contour"],
                help="How to render the activation landscape",
                key='manifold_type_select'
            )
        with col2:
            if manifold_type == "Wireframe":
                animate_manifold = False
                st.info("Animation not available for Wireframe")
            else:
                animate_manifold = st.checkbox("Animate", value=False, key='manifold_animate',
                                              help="Create animated visualization")
        with col3:
            normalize_manifold = st.checkbox("Normalize values", value=True, key='manifold_normalize')
        with col4:
            if is_temporal:
                show_temporal_boundaries = st.checkbox("Show timestep boundaries", value=True, key='manifold_temporal_boundaries')
            else:
                show_layer_boundaries = st.checkbox("Show layer boundaries", value=True, key='manifold_boundaries')

        # Animation options
        animation_mode = None
        frame_duration = 200
        transition_duration = 100

        if animate_manifold:
            anim_col1, anim_col2, anim_col3 = st.columns(3)
            with anim_col1:
                if is_temporal:
                    animation_mode = st.selectbox(
                        "Animation mode",
                        ["Timestep Evolution", "Sample Progression", "Rotating Camera"],
                        help="How to animate the manifold",
                        key='manifold_anim_mode'
                    )
                else:
                    animation_mode = st.selectbox(
                        "Animation mode",
                        ["Sample Progression", "Layer Slicing", "Rotating Camera"],
                        help="How to animate the manifold",
                        key='manifold_anim_mode_mlp'
                    )
            with anim_col2:
                frame_duration = st.slider("Frame duration (ms)", 50, 1000, 200, step=50, key='manifold_frame_dur')
            with anim_col3:
                transition_duration = st.slider("Transition (ms)", 0, 500, 100, step=50, key='manifold_trans_dur')

        # Apply normalization
        if normalize_manifold:
            matrix_viz = (activation_matrix - activation_matrix.mean()) / (activation_matrix.std() + 1e-8)
        else:
            matrix_viz = activation_matrix.copy()

        # Create figure
        fig_manifold = go.Figure()

        # Handle animation
        if animate_manifold:
            frames = []

            if is_temporal and animation_mode == "Timestep Evolution":
                # Animate through timesteps, showing activation evolution
                for t in range(num_timesteps):
                    start_idx = t * samples_per_timestep
                    end_idx = (t + 1) * samples_per_timestep

                    # Show only samples up to this timestep
                    Z_frame = matrix_viz[:end_idx, :]
                    X_frame, Y_frame = np.meshgrid(
                        np.arange(matrix_viz.shape[1]),
                        np.arange(end_idx)
                    )

                    frame_data = [go.Surface(
                        x=X_frame,
                        y=Y_frame,
                        z=Z_frame,
                        colorscale='Viridis',
                        colorbar=dict(title="Activation"),
                        showscale=True,
                        cmin=matrix_viz.min(),
                        cmax=matrix_viz.max()
                    )]

                    frames.append(go.Frame(data=frame_data, name=f"t={t}"))

                # Initial frame
                X, Y = np.meshgrid(
                    np.arange(matrix_viz.shape[1]),
                    np.arange(samples_per_timestep)
                )
                Z = matrix_viz[:samples_per_timestep, :]

            elif animation_mode == "Sample Progression":
                # Animate through samples, building up the surface
                step_size = max(1, matrix_viz.shape[0] // 30)  # ~30 frames
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
                        colorscale='Viridis',
                        colorbar=dict(title="Activation"),
                        showscale=True,
                        cmin=matrix_viz.min(),
                        cmax=matrix_viz.max()
                    )]

                    frames.append(go.Frame(data=frame_data, name=f"sample_{i}"))

                # Initial frame
                X, Y = np.meshgrid(
                    np.arange(matrix_viz.shape[1]),
                    np.arange(step_size)
                )
                Z = matrix_viz[:step_size, :]

            elif animation_mode == "Layer Slicing":
                # Animate through layers, showing neuron groups
                layer_boundaries = [0]
                cumsum = 0
                for size in layer_sizes:
                    cumsum += size
                    layer_boundaries.append(cumsum)

                for l_idx in range(len(layer_sizes)):
                    start = layer_boundaries[l_idx]
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
                        colorscale='Viridis',
                        colorbar=dict(title="Activation"),
                        showscale=True,
                        cmin=matrix_viz.min(),
                        cmax=matrix_viz.max()
                    )]

                    frames.append(go.Frame(data=frame_data, name=f"layer_{l_idx}"))

                # Initial frame
                X, Y = np.meshgrid(
                    np.arange(layer_boundaries[1]),
                    np.arange(matrix_viz.shape[0])
                )
                Z = matrix_viz[:, :layer_boundaries[1]]

            elif animation_mode == "Rotating Camera":
                # Animate camera rotation around the manifold
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
                            colorscale='Viridis',
                            colorbar=dict(title="Activation"),
                            showscale=True
                        )],
                        layout=go.Layout(
                            scene_camera=dict(
                                eye=dict(x=eye_x, y=eye_y, z=1.5)
                            )
                        ),
                        name=f"angle_{angle:.0f}"
                    )
                    frames.append(frame)
            else:
                # Default: no animation, full data
                X, Y = np.meshgrid(
                    np.arange(matrix_viz.shape[1]),
                    np.arange(matrix_viz.shape[0])
                )
                Z = matrix_viz

            fig_manifold.frames = frames

            # Add initial trace for animation (first frame's data)
            if manifold_type == "Surface":
                fig_manifold.add_trace(go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    colorscale='Viridis',
                    colorbar=dict(title="Activation"),
                    hovertemplate='Neuron: %{x}<br>Sample: %{y}<br>Activation: %{z:.3f}<extra></extra>',
                    lighting=dict(
                        ambient=0.4,
                        diffuse=0.5,
                        specular=0.6,
                        roughness=0.5
                    ),
                    contours=dict(
                        z=dict(show=True, usecolormap=True, highlightcolor="white", project=dict(z=True))
                    )
                ))
            elif manifold_type == "Contour":
                fig_manifold.add_trace(go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    colorscale='Viridis',
                    colorbar=dict(title="Activation"),
                    hovertemplate='Neuron: %{x}<br>Sample: %{y}<br>Activation: %{z:.3f}<extra></extra>',
                    contours=dict(
                        z=dict(show=True, usecolormap=True, highlightcolor="lime", project=dict(z=True), width=2),
                        x=dict(show=True, usecolormap=False, color="white", highlightcolor="yellow"),
                        y=dict(show=True, usecolormap=False, color="white", highlightcolor="yellow")
                    )
                ))
        else:
            # No animation: full static visualization
            X, Y = np.meshgrid(
                np.arange(matrix_viz.shape[1]),  # Neurons
                np.arange(matrix_viz.shape[0])   # Samples
            )
            Z = matrix_viz

        # Only add traces if not animated (animated already has frames with data)
        if not animate_manifold and manifold_type == "Surface":
            fig_manifold.add_trace(go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale='Viridis',
                colorbar=dict(title="Activation"),
                hovertemplate='Neuron: %{x}<br>Sample: %{y}<br>Activation: %{z:.3f}<extra></extra>',
                lighting=dict(
                    ambient=0.4,
                    diffuse=0.5,
                    specular=0.6,
                    roughness=0.5
                ),
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="white", project=dict(z=True))
                )
            ))
        elif not animate_manifold and manifold_type == "Wireframe":
            # Create wireframe
            # Sample rows
            for i in range(0, matrix_viz.shape[0], max(1, matrix_viz.shape[0]//20)):
                fig_manifold.add_trace(go.Scatter3d(
                    x=X[i, :],
                    y=[i] * len(X[i, :]),
                    z=Z[i, :],
                    mode='lines',
                    line=dict(color='cyan', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            # Neuron columns
            for j in range(0, matrix_viz.shape[1], max(1, matrix_viz.shape[1]//30)):
                fig_manifold.add_trace(go.Scatter3d(
                    x=[j] * matrix_viz.shape[0],
                    y=np.arange(matrix_viz.shape[0]),
                    z=Z[:, j],
                    mode='lines',
                    line=dict(color='magenta', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        elif not animate_manifold and manifold_type == "Contour":
            fig_manifold.add_trace(go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale='Viridis',
                colorbar=dict(title="Activation"),
                hovertemplate='Neuron: %{x}<br>Sample: %{y}<br>Activation: %{z:.3f}<extra></extra>',
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="lime", project=dict(z=True), width=2),
                    x=dict(show=True, usecolormap=False, color="white", highlightcolor="yellow"),
                    y=dict(show=True, usecolormap=False, color="white", highlightcolor="yellow")
                )
            ))

        # Add boundary markers if requested (only for non-animated)
        if not animate_manifold and manifold_type != "Wireframe":
            if is_temporal and show_temporal_boundaries:
                # Show timestep boundaries (horizontal planes in Y)
                for t in range(1, num_timesteps):
                    boundary_y = t * samples_per_timestep

                    x_range = np.arange(matrix_viz.shape[1])
                    z_range = np.linspace(Z.min(), Z.max(), 20)
                    X_plane, Z_plane = np.meshgrid(x_range, z_range)
                    Y_plane = np.full_like(X_plane, boundary_y)

                    fig_manifold.add_trace(go.Surface(
                        x=X_plane,
                        y=Y_plane,
                        z=Z_plane,
                        opacity=0.4,
                        colorscale=[[0, 'rgba(255,100,100,0.4)'], [1, 'rgba(255,100,100,0.4)']],
                        showscale=False,
                        hoverinfo='skip',
                        name=f"Timestep {t}"
                    ))

                # Also show layer boundaries (vertical planes in X)
                layer_boundaries = [0]
                cumsum = 0
                for size in layer_sizes:
                    cumsum += size
                    layer_boundaries.append(cumsum)

                for b_idx, boundary in enumerate(layer_boundaries[1:-1], 1):
                    y_range = np.arange(matrix_viz.shape[0])
                    z_range = np.linspace(Z.min(), Z.max(), 20)
                    Y_plane, Z_plane = np.meshgrid(y_range, z_range)
                    X_plane = np.full_like(Y_plane, boundary)

                    fig_manifold.add_trace(go.Surface(
                        x=X_plane,
                        y=Y_plane,
                        z=Z_plane,
                        opacity=0.2,
                        colorscale=[[0, 'rgba(100,100,255,0.2)'], [1, 'rgba(100,100,255,0.2)']],
                        showscale=False,
                        hoverinfo='skip',
                        name=f"Layer {b_idx}"
                    ))
            elif not is_temporal and show_layer_boundaries:
                # Standard layer boundaries for non-temporal (MLP)
                layer_boundaries = [0]
                cumsum = 0
                for size in layer_sizes:
                    cumsum += size
                    layer_boundaries.append(cumsum)

                # Add vertical planes at boundaries
                for b_idx, boundary in enumerate(layer_boundaries[:-1]):
                    y_range = np.arange(matrix_viz.shape[0])
                    z_range = np.linspace(Z.min(), Z.max(), 20)
                    Y_plane, Z_plane = np.meshgrid(y_range, z_range)
                    X_plane = np.full_like(Y_plane, boundary)

                    fig_manifold.add_trace(go.Surface(
                        x=X_plane,
                        y=Y_plane,
                        z=Z_plane,
                        opacity=0.3,
                        colorscale=[[0, 'rgba(255,255,255,0.3)'], [1, 'rgba(255,255,255,0.3)']],
                        showscale=False,
                        hoverinfo='skip',
                        name=f"Layer {b_idx}"
                    ))

        # Calculate neuron position labels
        layer_labels = []
        cumsum = 0
        for l_idx, size in enumerate(layer_sizes):
            mid = cumsum + size // 2
            layer_labels.append((mid, f"L{l_idx}"))
            cumsum += size

        # Set title and labels based on temporal vs non-temporal
        if is_temporal:
            title = f"LSTM Temporal Manifold - {samples_per_timestep} samples × {num_timesteps} timesteps"
            yaxis_title = "Sample × Timestep (temporal sequence)"
        else:
            title = f"MLP Activation Manifold - {matrix_viz.shape[0]} samples"
            yaxis_title = "Input Sample"

        # Add animation controls if animated
        if animate_manifold and len(fig_manifold.frames) > 0:
            fig_manifold.update_layout(
                title=title + " (Animated)",
                scene=dict(
                    xaxis_title="Neuron Index (concatenated layers)",
                    yaxis_title=yaxis_title,
                    zaxis_title="Activation Value",
