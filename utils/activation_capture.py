"""
Activation Capture Module
Provides infrastructure for capturing hidden states, attention patterns,
and logits at each layer of transformer models using forward hooks.
"""

import torch
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np


@dataclass
class LayerActivation:
    """Stores activation data for a single layer."""
    layer_idx: int
    hidden_states: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert tensors to numpy arrays."""
        result = {
            'layer_idx': self.layer_idx,
            'hidden_states': self.hidden_states.cpu().detach().numpy()
        }
        if self.attention_weights is not None:
            result['attention_weights'] = self.attention_weights.cpu().detach().numpy()
        return result


@dataclass
class ActivationTrace:
    """Complete trace of activations through all layers."""
    input_ids: torch.Tensor
    tokens: List[str]
    layer_activations: List[LayerActivation] = field(default_factory=list)
    logits: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None

    @property
    def num_layers(self) -> int:
        return len(self.layer_activations)

    @property
    def hidden_size(self) -> int:
        if self.layer_activations:
            return self.layer_activations[0].hidden_states.shape[-1]
        return 0

    def get_layer_hidden_states(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get hidden states for a specific layer."""
        for activation in self.layer_activations:
            if activation.layer_idx == layer_idx:
                return activation.hidden_states
        return None

    def get_all_hidden_states(self) -> torch.Tensor:
        """Stack all hidden states into a single tensor [num_layers, batch, seq, hidden]."""
        return torch.stack([act.hidden_states for act in self.layer_activations])

    def to_dict(self) -> Dict:
        """Convert trace to dictionary for serialization."""
        return {
            'tokens': self.tokens,
            'layer_activations': [act.to_numpy() for act in self.layer_activations],
            'logits': self.logits.cpu().detach().numpy() if self.logits is not None else None,
            'embeddings': self.embeddings.cpu().detach().numpy() if self.embeddings is not None else None
        }


class ActivationCaptureHook:
    """Manages forward hooks for capturing activations."""

    def __init__(self):
        self.activations: Dict[int, LayerActivation] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.embeddings: Optional[torch.Tensor] = None
        self.logits: Optional[torch.Tensor] = None

    def _create_layer_hook(self, layer_idx: int) -> Callable:
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
                attention = output[1] if len(output) > 1 else None
            else:
                hidden_states = output
                attention = None

            self.activations[layer_idx] = LayerActivation(
                layer_idx=layer_idx,
                hidden_states=hidden_states.detach().clone(),
                attention_weights=attention.detach().clone() if attention is not None else None
            )
        return hook_fn

    def _embedding_hook(self, module, input, output):
        """Hook for capturing input embeddings."""
        self.embeddings = output.detach().clone()

    def _output_hook(self, module, input, output):
        """Hook for capturing final logits."""
        if hasattr(output, 'logits'):
            self.logits = output.logits.detach().clone()
        elif isinstance(output, torch.Tensor):
            self.logits = output.detach().clone()

    def clear(self):
        """Clear stored activations."""
        self.activations.clear()
        self.embeddings = None
        self.logits = None

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_sorted_activations(self) -> List[LayerActivation]:
        """Get activations sorted by layer index."""
        return [self.activations[i] for i in sorted(self.activations.keys())]


class GPT2ActivationCapture:
    """Activation capture specifically for GPT-2 models."""

    def __init__(self, model):
        self.model = model
        self.lm_model = model.lm_model
        self.tokenizer = model.tokenizer
        self.hook_manager = ActivationCaptureHook()

    def _register_hooks(self):
        """Register hooks on all transformer layers."""
        self.hook_manager.clear()
        self.hook_manager.remove_hooks()

        # Hook on embedding layer
        embed_hook = self.lm_model.transformer.wte.register_forward_hook(
            self.hook_manager._embedding_hook
        )
        self.hook_manager.hooks.append(embed_hook)

        # Hook on each transformer layer
        for idx, layer in enumerate(self.lm_model.transformer.h):
            hook = layer.register_forward_hook(
                self.hook_manager._create_layer_hook(idx)
            )
            self.hook_manager.hooks.append(hook)

    @contextmanager
    def capture_context(self):
        """Context manager for capturing activations."""
        self._register_hooks()
        try:
            yield self.hook_manager
        finally:
            self.hook_manager.remove_hooks()

    @torch.no_grad()
    def capture_forward_pass(self, prompt: str) -> ActivationTrace:
        """Capture all activations during a forward pass."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        with self.capture_context():
            outputs = self.lm_model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
                output_attentions=True
            )

            trace = ActivationTrace(
                input_ids=inputs.input_ids,
                tokens=tokens,
                layer_activations=self.hook_manager.get_sorted_activations(),
                logits=outputs.logits,
                embeddings=self.hook_manager.embeddings
            )

        return trace

    @torch.no_grad()
    def capture_with_control_vector(
        self,
        prompt: str,
        control_vector: torch.Tensor,
        scale: float = 1.0,
        target_layers: Optional[List[int]] = None
    ) -> Tuple[ActivationTrace, ActivationTrace]:
        """
        Capture activations both with and without control vector.
        Returns (original_trace, modified_trace).
        """
        # Capture original
        original_trace = self.capture_forward_pass(prompt)

        # Capture with control vector
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        # Determine which layers to apply control vector
        if target_layers is None:
            target_layers = [len(self.lm_model.transformer.h) - 1]  # Last layer by default

        # Create hooks that apply control vector
        modified_hooks = []
        modified_activations = {}

        def create_control_hook(layer_idx: int):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                if layer_idx in target_layers:
                    modified_states = hidden_states + scale * control_vector.unsqueeze(0).unsqueeze(1)
                else:
                    modified_states = hidden_states

                modified_activations[layer_idx] = LayerActivation(
                    layer_idx=layer_idx,
                    hidden_states=modified_states.detach().clone()
                )

                if isinstance(output, tuple):
                    return (modified_states,) + output[1:]
                return modified_states
            return hook_fn

        # Register control hooks
        for idx, layer in enumerate(self.lm_model.transformer.h):
            hook = layer.register_forward_hook(create_control_hook(idx))
            modified_hooks.append(hook)

        try:
            outputs = self.lm_model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )

            modified_trace = ActivationTrace(
                input_ids=inputs.input_ids,
                tokens=tokens,
                layer_activations=[modified_activations[i] for i in sorted(modified_activations.keys())],
                logits=outputs.logits
            )
        finally:
            for hook in modified_hooks:
                hook.remove()

        return original_trace, modified_trace


class BertActivationCapture:
    """Activation capture specifically for BERT models."""

    def __init__(self, model):
        self.model = model
        self.bert_model = model.base_model
        self.lm_model = model.lm_model
        self.tokenizer = model.tokenizer
        self.hook_manager = ActivationCaptureHook()

    def _register_hooks(self):
        """Register hooks on all transformer layers."""
        self.hook_manager.clear()
        self.hook_manager.remove_hooks()

        # Hook on embedding layer
        embed_hook = self.bert_model.embeddings.register_forward_hook(
            self.hook_manager._embedding_hook
        )
        self.hook_manager.hooks.append(embed_hook)

        # Hook on each transformer layer
        for idx, layer in enumerate(self.bert_model.encoder.layer):
            hook = layer.register_forward_hook(
                self.hook_manager._create_layer_hook(idx)
            )
            self.hook_manager.hooks.append(hook)

    @contextmanager
    def capture_context(self):
        """Context manager for capturing activations."""
        self._register_hooks()
        try:
            yield self.hook_manager
        finally:
            self.hook_manager.remove_hooks()

    @torch.no_grad()
    def capture_forward_pass(self, prompt: str) -> ActivationTrace:
        """Capture all activations during a forward pass."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        with self.capture_context():
            outputs = self.lm_model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
                output_attentions=True
            )

            trace = ActivationTrace(
                input_ids=inputs.input_ids,
                tokens=tokens,
                layer_activations=self.hook_manager.get_sorted_activations(),
                logits=outputs.logits,
                embeddings=self.hook_manager.embeddings
            )

        return trace

    @torch.no_grad()
    def capture_with_control_vector(
        self,
        prompt: str,
        control_vector: torch.Tensor,
        scale: float = 1.0,
        target_layers: Optional[List[int]] = None
    ) -> Tuple[ActivationTrace, ActivationTrace]:
        """Capture activations both with and without control vector."""
        original_trace = self.capture_forward_pass(prompt)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        if target_layers is None:
            target_layers = [len(self.bert_model.encoder.layer) - 1]

        modified_hooks = []
        modified_activations = {}

        def create_control_hook(layer_idx: int):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                if layer_idx in target_layers:
                    modified_states = hidden_states + scale * control_vector.unsqueeze(0).unsqueeze(1)
                else:
                    modified_states = hidden_states

                modified_activations[layer_idx] = LayerActivation(
                    layer_idx=layer_idx,
                    hidden_states=modified_states.detach().clone()
                )

                if isinstance(output, tuple):
                    return (modified_states,) + output[1:]
                return modified_states
            return hook_fn

        for idx, layer in enumerate(self.bert_model.encoder.layer):
            hook = layer.register_forward_hook(create_control_hook(idx))
            modified_hooks.append(hook)

        try:
            outputs = self.lm_model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )

            modified_trace = ActivationTrace(
                input_ids=inputs.input_ids,
                tokens=tokens,
                layer_activations=[modified_activations[i] for i in sorted(modified_activations.keys())],
                logits=outputs.logits
            )
        finally:
            for hook in modified_hooks:
                hook.remove()

        return original_trace, modified_trace


def get_activation_capture(model) -> 'GPT2ActivationCapture | BertActivationCapture':
    """Factory function to get appropriate activation capture for model type."""
    model_class = model.__class__.__name__
    if 'GPT2' in model_class:
        return GPT2ActivationCapture(model)
    elif 'Bert' in model_class:
        return BertActivationCapture(model)
    else:
        raise ValueError(f"Unsupported model type: {model_class}")
