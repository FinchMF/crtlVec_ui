"""
Token Journey Tracking Module

Provides dataclasses for tracking how tokens transform as they pass through
model layers, including activations, attention weights, and entropy measurements.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import torch


@dataclass
class TokenJourney:
    """
    Comprehensive tracking of a single token's journey through model layers.

    Attributes:
        token: The string representation of the token
        token_id: The vocabulary ID of the token
        position: Position in the input sequence
        layer_activations: Dict mapping layer names to activation vectors
        attention_weights: Optional dict mapping layer names to attention weights
        entropy_per_layer: Optional dict mapping layer names to entropy values
        hidden_states: Optional list of hidden state vectors per layer
        residual_stream: Optional tracking of residual stream contributions
    """
    token: str
    token_id: int
    position: int
    layer_activations: Dict[str, np.ndarray] = field(default_factory=dict)
    attention_weights: Optional[Dict[str, np.ndarray]] = None
    entropy_per_layer: Optional[Dict[str, float]] = None
    hidden_states: Optional[List[np.ndarray]] = None
    residual_stream: Optional[Dict[str, np.ndarray]] = None

    def get_activation_norms(self) -> Dict[str, float]:
        """Get L2 norms of activations per layer."""
        return {
            layer: float(np.linalg.norm(act))
            for layer, act in self.layer_activations.items()
        }

    def get_trajectory_matrix(self) -> np.ndarray:
        """
        Get activation trajectory as a 2D matrix.
        Returns: [num_layers, max_dim] matrix of activations
        """
        if not self.layer_activations:
            return np.array([])

        layers = sorted(self.layer_activations.keys())
        max_dim = max(act.shape[-1] if act.ndim > 0 else 1
                      for act in self.layer_activations.values())

        trajectory = np.zeros((len(layers), max_dim))
        for i, layer in enumerate(layers):
            act = self.layer_activations[layer].flatten()
            trajectory[i, :len(act)] = act

        return trajectory

    def to_compact(self) -> 'CompactTokenJourney':
        """Convert to compact representation for memory efficiency."""
        norms = list(self.get_activation_norms().values())
        entropy = list(self.entropy_per_layer.values()) if self.entropy_per_layer else []
        return CompactTokenJourney(
            token=self.token,
            position=self.position,
            layer_norms=norms,
            entropy=entropy
        )


@dataclass
class CompactTokenJourney:
    """
    Memory-efficient representation of token journey.
    Stores only summary statistics rather than full activations.

    Attributes:
        token: The string representation of the token
        position: Position in the input sequence
        layer_norms: L2 norms of activations per layer
        entropy: Entropy values per layer
        attention_entropy: Optional attention entropy per layer
    """
    token: str
    position: int
    layer_norms: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    attention_entropy: Optional[List[float]] = None

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert to numpy arrays for visualization."""
        return {
            'norms': np.array(self.layer_norms),
            'entropy': np.array(self.entropy),
            'attention_entropy': np.array(self.attention_entropy) if self.attention_entropy else None
        }


@dataclass
class WeightSnapshot:
    """
    Snapshot of model weights at a specific training step.
    Used for visualizing weight evolution during training.

    Attributes:
        epoch: Training epoch number
        step: Global training step
        layer_name: Name of the layer
        weights: Weight matrix as numpy array
        gradients: Optional gradient values
        loss: Optional loss value at this step
    """
    epoch: int
    step: int
    layer_name: str
    weights: np.ndarray
    gradients: Optional[np.ndarray] = None
    loss: Optional[float] = None

    def get_weight_stats(self) -> Dict[str, float]:
        """Get summary statistics of weights."""
        return {
            'mean': float(np.mean(self.weights)),
            'std': float(np.std(self.weights)),
            'min': float(np.min(self.weights)),
            'max': float(np.max(self.weights)),
            'l2_norm': float(np.linalg.norm(self.weights)),
            'sparsity': float(np.mean(np.abs(self.weights) < 1e-6))
        }


@dataclass
class SequenceJourney:
    """
    Tracking of an entire sequence's journey through the model.
    Contains TokenJourney for each token plus sequence-level metrics.

    Attributes:
        tokens: List of TokenJourney objects
        input_text: Original input text
        model_name: Name of the model used
        total_layers: Number of layers in the model
        sequence_entropy: Optional sequence-level entropy curve
        cross_token_attention: Optional attention patterns between tokens
    """
    tokens: List[TokenJourney] = field(default_factory=list)
    input_text: str = ""
    model_name: str = ""
    total_layers: int = 0
    sequence_entropy: Optional[np.ndarray] = None
    cross_token_attention: Optional[Dict[str, np.ndarray]] = None

    def get_activation_matrix(self, layer_name: str) -> np.ndarray:
        """
        Get activation matrix for a specific layer.
        Returns: [num_tokens, hidden_dim] matrix
        """
        acts = []
        for tj in self.tokens:
            if layer_name in tj.layer_activations:
                acts.append(tj.layer_activations[layer_name].flatten())

        if not acts:
            return np.array([])

        max_dim = max(a.shape[0] for a in acts)
        matrix = np.zeros((len(acts), max_dim))
        for i, act in enumerate(acts):
            matrix[i, :len(act)] = act

        return matrix

    def get_entropy_matrix(self) -> np.ndarray:
        """
        Get entropy matrix across all tokens and layers.
        Returns: [num_tokens, num_layers] matrix
        """
        if not self.tokens or not self.tokens[0].entropy_per_layer:
            return np.array([])

        num_tokens = len(self.tokens)
        num_layers = len(self.tokens[0].entropy_per_layer)

        matrix = np.zeros((num_tokens, num_layers))
        for i, tj in enumerate(self.tokens):
            if tj.entropy_per_layer:
                for j, (layer, ent) in enumerate(sorted(tj.entropy_per_layer.items())):
                    matrix[i, j] = ent

        return matrix

    def to_compact(self) -> List[CompactTokenJourney]:
        """Convert all tokens to compact representation."""
        return [tj.to_compact() for tj in self.tokens]


class TokenJourneyTracker:
    """
    Utility class for tracking token journeys through a model.
    Attaches hooks to capture activations and compute metrics.
    """

    def __init__(self, model: torch.nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        self.layer_outputs = {}

    def _hook_fn(self, layer_name: str):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.layer_outputs[layer_name] = output.detach().cpu()
        return hook

    def attach_hooks(self, layer_names: Optional[List[str]] = None):
        """Attach hooks to model layers."""
        self.clear_hooks()

        for name, module in self.model.named_modules():
            if layer_names is None or name in layer_names:
                if hasattr(module, 'weight') or 'layer' in name.lower():
                    hook = module.register_forward_hook(self._hook_fn(name))
                    self.hooks.append(hook)

    def clear_hooks(self):
        """Remove all attached hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.layer_outputs = {}

    def track_sequence(self, text: str, compute_entropy: bool = True) -> SequenceJourney:
        """
        Track a sequence through the model.

        Args:
            text: Input text to process
            compute_entropy: Whether to compute per-layer entropy

        Returns:
            SequenceJourney containing TokenJourney for each token
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt")
        token_ids = inputs['input_ids'][0].tolist()
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        # Clear previous outputs
        self.layer_outputs = {}

        # Forward pass
        with torch.no_grad():
            self.model(**inputs)

        # Build token journeys
        sequence = SequenceJourney(
            input_text=text,
            model_name=type(self.model).__name__,
            total_layers=len(self.layer_outputs)
        )

        for pos, (token, token_id) in enumerate(zip(tokens, token_ids)):
            journey = TokenJourney(
                token=token,
                token_id=token_id,
                position=pos
            )

            # Extract activations for this token position
            for layer_name, output in self.layer_outputs.items():
                if output.ndim >= 3:  # [batch, seq, hidden]
                    act = output[0, pos, :].numpy()
                else:
                    act = output[0].numpy() if output.ndim == 2 else output.numpy()
                journey.layer_activations[layer_name] = act

            # Compute entropy if requested
            if compute_entropy:
                journey.entropy_per_layer = {}
                for layer_name, act in journey.layer_activations.items():
                    # Normalize to probability distribution
                    act_abs = np.abs(act) + 1e-10
                    probs = act_abs / act_abs.sum()
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    journey.entropy_per_layer[layer_name] = float(entropy)

            sequence.tokens.append(journey)

        return sequence

    def __del__(self):
        """Clean up hooks on deletion."""
        self.clear_hooks()
