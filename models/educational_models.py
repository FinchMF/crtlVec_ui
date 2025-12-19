"""
Educational Neural Network Models

Small, interpretable neural network models designed for educational purposes.
These allow users to visualize and understand neural network concepts
before moving to larger transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class ActivationSnapshot:
    """Snapshot of activations at each layer for a single input."""
    input_data: np.ndarray
    layer_activations: List[np.ndarray]
    layer_names: List[str]
    output: np.ndarray


class TinyMLP(nn.Module):
    """
    A small Multi-Layer Perceptron for educational demonstrations.

    Architecture: input -> fc1 -> ReLU -> fc2 -> ReLU -> fc3 -> output

    This model is designed to be small enough to visualize every neuron
    and connection, making it ideal for teaching feedforward concepts.
    """

    def __init__(self, in_dim: int = 16, h1: int = 32, h2: int = 32, out_dim: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.h1 = h1
        self.h2 = h2
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_dim)

        self.layer_sizes = [in_dim, h1, h2, out_dim]
        self.layer_names = ['input', 'hidden1', 'hidden2', 'output']

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        """
        Forward pass with optional activation capture.

        Args:
            x: Input tensor of shape [batch, in_dim]
            return_activations: If True, return list of activations at each layer

        Returns:
            output tensor, or (output, activations) if return_activations=True
        """
        activations = [x.detach().cpu().numpy()] if return_activations else None

        x = F.relu(self.fc1(x))
        if return_activations:
            activations.append(x.detach().cpu().numpy())

        x = F.relu(self.fc2(x))
        if return_activations:
            activations.append(x.detach().cpu().numpy())

        x = self.fc3(x)
        if return_activations:
            activations.append(x.detach().cpu().numpy())

        if return_activations:
            return x, activations
        return x

    def get_weight_matrices(self) -> List[np.ndarray]:
        """Get weight matrices for visualization."""
        return [
            self.fc1.weight.detach().cpu().numpy(),
            self.fc2.weight.detach().cpu().numpy(),
            self.fc3.weight.detach().cpu().numpy(),
        ]

    def compute_activations(self, x: torch.Tensor) -> ActivationSnapshot:
        """Compute and return activation snapshot for input."""
        output, activations = self.forward(x, return_activations=True)
        return ActivationSnapshot(
            input_data=x.detach().cpu().numpy(),
            layer_activations=activations,
            layer_names=self.layer_names,
            output=output.detach().cpu().numpy()
        )


class TinyLSTM(nn.Module):
    """
    A small LSTM network for educational demonstrations.

    This model demonstrates recurrent connections and how information
    flows across timesteps. Designed for visualizing memory dynamics.
    """

    def __init__(self, in_dim: int = 16, hidden_dim: int = 32,
                 num_layers: int = 2, out_dim: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, out_dim)

        self.layer_sizes = [in_dim] + [hidden_dim] * num_layers + [out_dim]

    def forward(self, x: torch.Tensor, return_hidden: bool = False,
                initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass with optional hidden state return.

        Args:
            x: Input tensor of shape [batch, seq_len, in_dim]
            return_hidden: If True, return hidden states
            initial_state: Optional (h_0, c_0) initial hidden state

        Returns:
            output tensor, or (output, (h_n, c_n)) if return_hidden=True
        """
        if initial_state is not None:
            lstm_out, (h_n, c_n) = self.lstm(x, initial_state)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last timestep output
        output = self.fc(lstm_out[:, -1, :])

        if return_hidden:
            return output, (h_n, c_n)
        return output

    def forward_stepwise(self, x: torch.Tensor) -> List[Dict[str, np.ndarray]]:
        """
        Process input one timestep at a time, returning activations at each step.

        Args:
            x: Input tensor of shape [batch, seq_len, in_dim]

        Returns:
            List of dicts containing activations at each timestep
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Initialize hidden states
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        timestep_activations = []

        for t in range(seq_len):
            x_t = x[:, t:t+1, :]  # [batch, 1, in_dim]
            lstm_out, (h, c) = self.lstm(x_t, (h, c))

            step_acts = {
                'timestep': t,
                'input': x_t.detach().cpu().numpy(),
                'hidden_states': [h[i].detach().cpu().numpy() for i in range(self.num_layers)],
                'cell_states': [c[i].detach().cpu().numpy() for i in range(self.num_layers)],
                'lstm_output': lstm_out.detach().cpu().numpy(),
                'output': self.fc(lstm_out[:, -1, :]).detach().cpu().numpy()
            }
            timestep_activations.append(step_acts)

        return timestep_activations

    def get_weight_matrices(self) -> Dict[str, np.ndarray]:
        """Get LSTM weight matrices for visualization."""
        weights = {}

        # Input-hidden weights for each layer
        for i in range(self.num_layers):
            weights[f'weight_ih_l{i}'] = getattr(self.lstm, f'weight_ih_l{i}').detach().cpu().numpy()
            weights[f'weight_hh_l{i}'] = getattr(self.lstm, f'weight_hh_l{i}').detach().cpu().numpy()
            weights[f'bias_ih_l{i}'] = getattr(self.lstm, f'bias_ih_l{i}').detach().cpu().numpy()
            weights[f'bias_hh_l{i}'] = getattr(self.lstm, f'bias_hh_l{i}').detach().cpu().numpy()

        weights['fc_weight'] = self.fc.weight.detach().cpu().numpy()
        weights['fc_bias'] = self.fc.bias.detach().cpu().numpy()

        return weights


class TinyGPT(nn.Module):
    """
    A minimal GPT-style transformer for educational purposes.

    This model implements the core transformer decoder architecture
    with enough simplicity to visualize attention patterns and
    information flow through layers.
    """

    def __init__(self, vocab_size: int = 1000, embed_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2,
                 ff_dim: int = 128, max_seq_len: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

        # Output projection
        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Forward pass with optional attention pattern return.

        Args:
            x: Input token IDs of shape [batch, seq_len]
            return_attention: If True, return attention patterns per layer

        Returns:
            logits, or (logits, attention_patterns) if return_attention=True
        """
        batch_size, seq_len = x.shape

        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = tok_emb + pos_emb

        # Track attention patterns
        attention_patterns = [] if return_attention else None

        # Transformer blocks
        for block in self.blocks:
            x, attn = block(x, return_attention=True)
            if return_attention:
                attention_patterns.append(attn.detach().cpu().numpy())

        # Output
        x = self.ln_final(x)
        logits = self.head(x)

        if return_attention:
            return logits, attention_patterns
        return logits

    def get_attention_patterns(self, x: torch.Tensor) -> List[np.ndarray]:
        """Get attention patterns for visualization."""
        _, patterns = self.forward(x, return_attention=True)
        return patterns

    def get_layer_activations(self, x: torch.Tensor) -> List[np.ndarray]:
        """Get activations at each layer."""
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        h = tok_emb + pos_emb

        activations = [h.detach().cpu().numpy()]

        for block in self.blocks:
            h, _ = block(h, return_attention=True)
            activations.append(h.detach().cpu().numpy())

        return activations


class TransformerBlock(nn.Module):
    """Single transformer decoder block."""

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Forward pass with optional attention return.

        Args:
            x: Input tensor of shape [batch, seq_len, embed_dim]
            return_attention: If True, return attention weights

        Returns:
            output tensor, or (output, attention_weights) if return_attention=True
        """
        # Self-attention with residual
        x_norm = self.ln1(x)

        # Create causal mask
        seq_len = x.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device) * float('-inf'),
            diagonal=1
        )

        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask,
            need_weights=True
        )
        x = x + attn_out

        # Feedforward with residual
        x = x + self.ff(self.ln2(x))

        if return_attention:
            return x, attn_weights
        return x


class MaskedGPT(TinyGPT):
    """
    TinyGPT variant that can use masked language modeling (BERT-style).
    Useful for comparing autoregressive vs masked language modeling.
    """

    def __init__(self, *args, mask_token_id: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_token_id = mask_token_id

    def forward(self, x: torch.Tensor, return_attention: bool = False,
                use_causal_mask: bool = True):
        """
        Forward pass with configurable masking.

        Args:
            x: Input token IDs
            return_attention: Return attention patterns
            use_causal_mask: If True, use causal (autoregressive) mask
                            If False, use bidirectional (BERT-style)
        """
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        h = tok_emb + pos_emb

        attention_patterns = [] if return_attention else None

        for block in self.blocks:
            if use_causal_mask:
                h, attn = block(h, return_attention=True)
            else:
                # Bidirectional attention (no causal mask)
                h_norm = block.ln1(h)
                attn_out, attn = block.attn(h_norm, h_norm, h_norm, need_weights=True)
                h = h + attn_out
                h = h + block.ff(block.ln2(h))

            if return_attention:
                attention_patterns.append(attn.detach().cpu().numpy())

        h = self.ln_final(h)
        logits = self.head(h)

        if return_attention:
            return logits, attention_patterns
        return logits


# Utility functions for model analysis

def compute_mean_activations(model: TinyMLP, n_samples: int = 200) -> List[np.ndarray]:
    """
    Compute mean activation magnitudes across random samples.

    Args:
        model: TinyMLP model
        n_samples: Number of random inputs to average over

    Returns:
        List of mean activation magnitudes per layer
    """
    model.eval()
    all_activations = []

    with torch.no_grad():
        for _ in range(n_samples):
            x = torch.randn(1, model.in_dim)
            _, acts = model.forward(x, return_activations=True)
            all_activations.append(acts)

    # Average across samples
    num_layers = len(all_activations[0])
    mean_activations = []

    for layer_idx in range(num_layers):
        layer_acts = np.array([acts[layer_idx] for acts in all_activations])
        mean_acts = np.mean(np.abs(layer_acts), axis=0).squeeze()
        mean_activations.append(mean_acts)

    return mean_activations


def compute_single_activations(model: TinyMLP, x: torch.Tensor) -> List[np.ndarray]:
    """
    Compute activations for a single input.

    Args:
        model: TinyMLP model
        x: Input tensor

    Returns:
        List of activations per layer
    """
    model.eval()
    with torch.no_grad():
        _, acts = model.forward(x, return_activations=True)
    return acts


def get_lstm_temporal_activations(model: TinyLSTM, seq_length: int = 5,
                                   n_samples: int = 50) -> np.ndarray:
    """
    Get temporal activation patterns from LSTM.

    Args:
        model: TinyLSTM model
        seq_length: Length of input sequences
        n_samples: Number of samples

    Returns:
        Array of shape [n_samples * seq_length, total_neurons]
    """
    model.eval()
    all_activations = []

    with torch.no_grad():
        for _ in range(n_samples):
            x = torch.randn(1, seq_length, model.in_dim)
            timestep_acts = model.forward_stepwise(x)

            for step_acts in timestep_acts:
                # Concatenate all hidden states
                concat = np.concatenate(
                    [step_acts['input'].flatten()] +
                    [h.flatten() for h in step_acts['hidden_states']] +
                    [step_acts['output'].flatten()]
                )
                all_activations.append(concat)

    return np.array(all_activations)
