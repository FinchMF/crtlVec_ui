"""
Entropy Analysis Module
Computes token entropy and top-k predictions at each layer to measure
model certainty and understand prediction possibilities during inference.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class TokenPrediction:
    """Stores prediction information for a single token position."""
    token_idx: int
    token: str
    entropy: float
    top_k_tokens: List[str]
    top_k_probs: List[float]
    top_k_ids: List[int]

    @property
    def confidence(self) -> float:
        """Return confidence as probability of top prediction."""
        return self.top_k_probs[0] if self.top_k_probs else 0.0

    @property
    def uncertainty(self) -> float:
        """Normalized uncertainty (0 = certain, 1 = maximum uncertainty)."""
        # Normalize entropy by log(vocab_size) for interpretability
        max_entropy = np.log(50257)  # GPT-2 vocab size
        return min(self.entropy / max_entropy, 1.0)


@dataclass
class LayerEntropyAnalysis:
    """Entropy analysis for a single layer."""
    layer_idx: int
    token_predictions: List[TokenPrediction]
    mean_entropy: float
    max_entropy: float
    min_entropy: float

    @property
    def entropy_trajectory(self) -> List[float]:
        """Get entropy values for all tokens."""
        return [tp.entropy for tp in self.token_predictions]

    def get_high_entropy_positions(self, threshold: float = 2.0) -> List[int]:
        """Find token positions with high uncertainty."""
        return [tp.token_idx for tp in self.token_predictions if tp.entropy > threshold]


@dataclass
class EntropyTrace:
    """Complete entropy analysis across all layers."""
    tokens: List[str]
    layer_analyses: List[LayerEntropyAnalysis] = field(default_factory=list)

    @property
    def num_layers(self) -> int:
        return len(self.layer_analyses)

    @property
    def sequence_length(self) -> int:
        return len(self.tokens)

    def get_entropy_heatmap(self) -> np.ndarray:
        """Get entropy values as [num_layers, seq_len] array for visualization."""
        if not self.layer_analyses:
            return np.array([])

        num_layers = len(self.layer_analyses)
        seq_len = len(self.layer_analyses[0].token_predictions)

        heatmap = np.zeros((num_layers, seq_len))
        for layer_analysis in self.layer_analyses:
            for pred in layer_analysis.token_predictions:
                heatmap[layer_analysis.layer_idx, pred.token_idx] = pred.entropy

        return heatmap

    def get_confidence_trajectory(self, token_idx: int) -> List[float]:
        """Get confidence trajectory for a specific token across layers."""
        return [
            layer.token_predictions[token_idx].confidence
            for layer in self.layer_analyses
            if token_idx < len(layer.token_predictions)
        ]

    def get_layer_mean_entropy(self) -> List[float]:
        """Get mean entropy for each layer."""
        return [layer.mean_entropy for layer in self.layer_analyses]

    def get_entropy_reduction(self) -> float:
        """Calculate entropy reduction from first to last layer."""
        if len(self.layer_analyses) < 2:
            return 0.0
        first = self.layer_analyses[0].mean_entropy
        last = self.layer_analyses[-1].mean_entropy
        return first - last


class EntropyAnalyzer:
    """Analyzes entropy and prediction confidence through model layers."""

    def __init__(self, model, tokenizer, unembedding_matrix: torch.Tensor):
        """
        Initialize entropy analyzer.

        Args:
            model: The language model
            tokenizer: The tokenizer
            unembedding_matrix: The output embedding/unembedding matrix (lm_head weights)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.unembedding = unembedding_matrix
        self.vocab_size = unembedding_matrix.shape[0]

    @torch.no_grad()
    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy from logits.

        Args:
            logits: Tensor of shape [batch, seq_len, vocab_size]

        Returns:
            Entropy tensor of shape [batch, seq_len]
        """
        probs = F.softmax(logits, dim=-1)
        # Add small epsilon to avoid log(0)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    @torch.no_grad()
    def get_top_k_predictions(
        self,
        logits: torch.Tensor,
        k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get top-k token predictions from logits.

        Args:
            logits: Tensor of shape [batch, seq_len, vocab_size]
            k: Number of top predictions to return

        Returns:
            Tuple of (top_k_probs, top_k_indices, top_k_tokens)
        """
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
        return top_k_probs, top_k_indices

    @torch.no_grad()
    def analyze_hidden_state(
        self,
        hidden_state: torch.Tensor,
        tokens: List[str],
        layer_idx: int,
        k: int = 10
    ) -> LayerEntropyAnalysis:
        """
        Analyze entropy for a single layer's hidden states.

        Projects hidden states through unembedding to get logits,
        then computes entropy and top-k predictions.

        Args:
            hidden_state: Tensor of shape [batch, seq_len, hidden_size]
            tokens: List of input tokens
            layer_idx: Index of the layer
            k: Number of top predictions

        Returns:
            LayerEntropyAnalysis object
        """
        # Project hidden states to vocabulary space
        # hidden_state: [batch, seq_len, hidden_dim]
        # unembedding: [vocab_size, hidden_dim]
        logits = torch.matmul(hidden_state, self.unembedding.T)

        # Compute entropy
        entropy = self.compute_entropy(logits)  # [batch, seq_len]

        # Get top-k predictions
        top_k_probs, top_k_indices = self.get_top_k_predictions(logits, k)

        # Build token predictions for each position
        token_predictions = []
        batch_idx = 0  # Assuming single batch

        for seq_idx in range(hidden_state.shape[1]):
            top_tokens = [
                self.tokenizer.decode([idx.item()])
                for idx in top_k_indices[batch_idx, seq_idx]
            ]

            prediction = TokenPrediction(
                token_idx=seq_idx,
                token=tokens[seq_idx] if seq_idx < len(tokens) else "<pad>",
                entropy=entropy[batch_idx, seq_idx].item(),
                top_k_tokens=top_tokens,
                top_k_probs=top_k_probs[batch_idx, seq_idx].tolist(),
                top_k_ids=top_k_indices[batch_idx, seq_idx].tolist()
            )
            token_predictions.append(prediction)

        # Compute layer statistics
        entropy_values = [p.entropy for p in token_predictions]

        return LayerEntropyAnalysis(
            layer_idx=layer_idx,
            token_predictions=token_predictions,
            mean_entropy=np.mean(entropy_values),
            max_entropy=np.max(entropy_values),
            min_entropy=np.min(entropy_values)
        )

    def analyze_trace(
        self,
        layer_hidden_states: List[torch.Tensor],
        tokens: List[str],
        k: int = 10
    ) -> EntropyTrace:
        """
        Analyze entropy across all layers from an activation trace.

        Args:
            layer_hidden_states: List of hidden states for each layer
            tokens: Input tokens
            k: Number of top predictions per position

        Returns:
            EntropyTrace with analysis for all layers
        """
        layer_analyses = []

        for layer_idx, hidden_state in enumerate(layer_hidden_states):
            analysis = self.analyze_hidden_state(
                hidden_state,
                tokens,
                layer_idx,
                k
            )
            layer_analyses.append(analysis)

        return EntropyTrace(
            tokens=tokens,
            layer_analyses=layer_analyses
        )


class GPT2EntropyAnalyzer(EntropyAnalyzer):
    """Entropy analyzer specifically for GPT-2 models."""

    def __init__(self, model):
        """
        Initialize GPT-2 entropy analyzer.

        Args:
            model: GPT2ControlModel instance
        """
        # Get the unembedding matrix (lm_head for GPT-2)
        unembedding = model.lm_model.lm_head.weight
        super().__init__(model, model.tokenizer, unembedding)
        self.gpt2_model = model

    @torch.no_grad()
    def analyze_prompt(self, prompt: str, k: int = 10) -> EntropyTrace:
        """
        Analyze entropy for a prompt through all layers.

        Args:
            prompt: Input text
            k: Number of top predictions

        Returns:
            EntropyTrace with full analysis
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        # Get hidden states from all layers
        outputs = self.gpt2_model.lm_model(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True
        )

        # outputs.hidden_states is a tuple of (embedding_output, layer_1, layer_2, ...)
        # Skip the embedding output (index 0)
        layer_hidden_states = list(outputs.hidden_states[1:])

        return self.analyze_trace(layer_hidden_states, tokens, k)

    @torch.no_grad()
    def compare_with_control_vector(
        self,
        prompt: str,
        control_vector: torch.Tensor,
        scale: float = 1.0,
        k: int = 10,
        target_layer: int = -1
    ) -> Tuple[EntropyTrace, EntropyTrace]:
        """
        Compare entropy with and without control vector.

        Args:
            prompt: Input text
            control_vector: Control vector to apply
            scale: Scaling factor for control vector
            k: Number of top predictions
            target_layer: Layer to apply control vector (-1 for last)

        Returns:
            Tuple of (original_trace, modified_trace)
        """
        # Original trace
        original_trace = self.analyze_prompt(prompt, k)

        # Modified trace with control vector
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        num_layers = len(self.gpt2_model.lm_model.transformer.h)
        if target_layer < 0:
            target_layer = num_layers + target_layer

        modified_hidden_states = []

        # Ensure control vector has proper shape for broadcasting
        # Control vector shape: [hidden_dim] -> needs to be [1, 1, hidden_dim] or compatible
        cv = control_vector
        if cv.dim() == 1:
            cv = cv.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
        elif cv.dim() == 2:
            cv = cv.unsqueeze(0)  # [1, seq, hidden_dim]

        def create_capture_hook(layer_idx: int):
            def hook_fn(module, input, output):
                try:
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output

                    if layer_idx == target_layer:
                        # Ensure dimensions match for broadcasting
                        if cv.shape[-1] == hidden_states.shape[-1]:
                            hidden_states = hidden_states + scale * cv

                    modified_hidden_states.append(hidden_states.detach().clone())

                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    return hidden_states
                except Exception as e:
                    # Log error but don't crash
                    print(f"GPT2 hook error at layer {layer_idx}: {e}")
                    if isinstance(output, tuple):
                        modified_hidden_states.append(output[0].detach().clone())
                    else:
                        modified_hidden_states.append(output.detach().clone())
                    return output
            return hook_fn

        hooks = []
        for idx, layer in enumerate(self.gpt2_model.lm_model.transformer.h):
            hook = layer.register_forward_hook(create_capture_hook(idx))
            hooks.append(hook)

        try:
            self.gpt2_model.lm_model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )

            modified_trace = self.analyze_trace(modified_hidden_states, tokens, k)
        finally:
            for hook in hooks:
                hook.remove()

        return original_trace, modified_trace


class BertEntropyAnalyzer(EntropyAnalyzer):
    """Entropy analyzer specifically for BERT models."""

    def __init__(self, model):
        """
        Initialize BERT entropy analyzer.

        Args:
            model: BertControlModel instance
        """
        # BERT's MLM head decoder weights
        unembedding = model.lm_model.cls.predictions.decoder.weight
        super().__init__(model, model.tokenizer, unembedding)
        self.bert_model = model

    @torch.no_grad()
    def analyze_prompt(self, prompt: str, k: int = 10) -> EntropyTrace:
        """Analyze entropy for a prompt through all layers."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        outputs = self.bert_model.lm_model(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True
        )

        layer_hidden_states = list(outputs.hidden_states[1:])

        return self.analyze_trace(layer_hidden_states, tokens, k)

    @torch.no_grad()
    def compare_with_control_vector(
        self,
        prompt: str,
        control_vector: torch.Tensor,
        scale: float = 1.0,
        k: int = 10,
        target_layer: int = -1
    ) -> Tuple[EntropyTrace, EntropyTrace]:
        """Compare entropy with and without control vector."""
        original_trace = self.analyze_prompt(prompt, k)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        num_layers = len(self.bert_model.base_model.encoder.layer)
        if target_layer < 0:
            target_layer = num_layers + target_layer

        modified_hidden_states = []

        # Ensure control vector has proper shape for broadcasting
        cv = control_vector
        if cv.dim() == 1:
            cv = cv.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
        elif cv.dim() == 2:
            cv = cv.unsqueeze(0)  # [1, seq, hidden_dim]

        def create_capture_hook(layer_idx: int):
            def hook_fn(module, input, output):
                try:
                    if isinstance(output, tuple):
                        hidden_states = output[0]
                    else:
                        hidden_states = output

                    if layer_idx == target_layer:
                        # Ensure dimensions match for broadcasting
                        if cv.shape[-1] == hidden_states.shape[-1]:
                            hidden_states = hidden_states + scale * cv

                    modified_hidden_states.append(hidden_states.detach().clone())

                    if isinstance(output, tuple):
                        return (hidden_states,) + output[1:]
                    return hidden_states
                except Exception as e:
                    # Log error but don't crash
                    print(f"BERT hook error at layer {layer_idx}: {e}")
                    if isinstance(output, tuple):
                        modified_hidden_states.append(output[0].detach().clone())
                    else:
                        modified_hidden_states.append(output.detach().clone())
                    return output
            return hook_fn

        hooks = []
        for idx, layer in enumerate(self.bert_model.base_model.encoder.layer):
            hook = layer.register_forward_hook(create_capture_hook(idx))
            hooks.append(hook)

        try:
            self.bert_model.lm_model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )

            modified_trace = self.analyze_trace(modified_hidden_states, tokens, k)
        finally:
            for hook in hooks:
                hook.remove()

        return original_trace, modified_trace


def get_entropy_analyzer(model) -> 'GPT2EntropyAnalyzer | BertEntropyAnalyzer':
    """Factory function to get appropriate entropy analyzer for model type."""
    model_class = model.__class__.__name__
    if 'GPT2' in model_class:
        return GPT2EntropyAnalyzer(model)
    elif 'Bert' in model_class:
        return BertEntropyAnalyzer(model)
    else:
        raise ValueError(f"Unsupported model type: {model_class}")
