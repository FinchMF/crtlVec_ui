import os
from .base_model import LanguageModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
import torch
from typing import List, Optional, Dict, Tuple

GPT2_LOCATION = os.path.expanduser("~/coding/local_models/gpt2-base-cased/")


class GPT2ControlModel(LanguageModel):
    """GPT-2 model with control vector capabilities and activation capture."""

    def __init__(self):
        super().__init__()
        self.base_model = GPT2Model.from_pretrained(GPT2_LOCATION, output_hidden_states=True)
        self.lm_model = GPT2LMHeadModel.from_pretrained(GPT2_LOCATION)
        self.tokenizer = GPT2Tokenizer.from_pretrained(GPT2_LOCATION)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model.eval()
        self.lm_model.eval()

        # Model configuration
        self.hidden_size = self.lm_model.config.hidden_size
        self.num_layers = self.lm_model.config.n_layer
        self.vocab_size = self.lm_model.config.vocab_size

    @torch.no_grad()
    def get_hidden_state(self, prompt: str) -> torch.Tensor:
        """Get the hidden state for a prompt (last token, last layer)."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = self.base_model(**inputs)
        return outputs.hidden_states[-1][0, -1]

    @torch.no_grad()
    def get_all_hidden_states(self, prompt: str) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Get hidden states from all layers.

        Returns:
            Tuple of (list of hidden state tensors per layer, list of tokens)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        outputs = self.lm_model(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True
        )

        # outputs.hidden_states is tuple of (embedding, layer1, layer2, ...)
        # Return all layer outputs (skip embedding at index 0)
        layer_hidden_states = [h.squeeze(0) for h in outputs.hidden_states[1:]]

        return layer_hidden_states, tokens

    @torch.no_grad()
    def get_layer_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project hidden state to vocabulary logits using the LM head."""
        return self.lm_model.lm_head(hidden_state)

    @torch.no_grad()
    def get_all_layer_logits(self, prompt: str) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Get logits from projecting each layer's hidden states through the LM head.
        Useful for analyzing how predictions evolve through layers.

        Returns:
            Tuple of (list of logit tensors per layer, list of tokens)
        """
        layer_hidden_states, tokens = self.get_all_hidden_states(prompt)
        layer_logits = [self.get_layer_logits(h) for h in layer_hidden_states]
        return layer_logits, tokens

    @torch.no_grad()
    def generate_with_control(
        self,
        prompt: str,
        control_vector: torch.Tensor,
        scale: float = 1.0,
        target_layers: Optional[List[int]] = None,
        max_length: int = 80,
        temperature: float = 1.0
    ) -> str:
        """
        Generate text with control vector applied.

        Args:
            prompt: Input text
            control_vector: Control vector to apply
            scale: Scaling factor for control vector
            target_layers: Which layers to apply control vector (default: last layer only)
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)

        Returns:
            Generated text
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        if target_layers is None:
            target_layers = [self.num_layers - 1]  # Last layer by default

        hooks = []

        def create_hook(layer_idx: int):
            def hook_fn(module, input, output):
                if layer_idx in target_layers:
                    hidden_states = output[0]
                    modified_states = hidden_states + scale * control_vector.unsqueeze(0).unsqueeze(1)
                    return (modified_states,) + output[1:] if len(output) > 1 else (modified_states,)
                return output
            return hook_fn

        # Register hooks on target layers
        for idx in target_layers:
            hook = self.lm_model.transformer.h[idx].register_forward_hook(create_hook(idx))
            hooks.append(hook)

        try:
            output_ids = self.lm_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                max_length=max_length,
                do_sample=True,
                top_k=40,
                temperature=temperature
            )
        finally:
            for hook in hooks:
                hook.remove()

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    @torch.no_grad()
    def generate_with_control_and_capture(
        self,
        prompt: str,
        control_vector: torch.Tensor,
        scale: float = 1.0,
        target_layers: Optional[List[int]] = None
    ) -> Tuple[str, Dict]:
        """
        Generate with control vector and capture activation data.

        Returns:
            Tuple of (generated_text, activation_data_dict)
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        if target_layers is None:
            target_layers = [self.num_layers - 1]

        captured_activations = {}
        hooks = []

        def create_capture_hook(layer_idx: int):
            def hook_fn(module, input, output):
                hidden_states = output[0] if isinstance(output, tuple) else output

                if layer_idx in target_layers:
                    modified_states = hidden_states + scale * control_vector.unsqueeze(0).unsqueeze(1)
                    captured_activations[f'layer_{layer_idx}_modified'] = modified_states.detach().clone()

                    if isinstance(output, tuple):
                        return (modified_states,) + output[1:]
                    return modified_states
                else:
                    captured_activations[f'layer_{layer_idx}'] = hidden_states.detach().clone()
                    return output
            return hook_fn

        for idx, layer in enumerate(self.lm_model.transformer.h):
            hook = layer.register_forward_hook(create_capture_hook(idx))
            hooks.append(hook)

        try:
            output_ids = self.lm_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                max_length=80,
                do_sample=True,
                top_k=40
            )
        finally:
            for hook in hooks:
                hook.remove()

        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return generated_text, captured_activations

    @torch.no_grad()
    def compare_generations(
        self,
        prompt: str,
        control_vector: torch.Tensor,
        scales: List[float] = [-1.0, 0.0, 1.0]
    ) -> Dict[float, str]:
        """
        Generate text at different control vector scales for comparison.

        Returns:
            Dict mapping scale -> generated_text
        """
        results = {}
        for scale in scales:
            if scale == 0.0:
                # No control vector
                inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                output_ids = self.lm_model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_length=80,
                    do_sample=True,
                    top_k=40
                )
                results[scale] = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            else:
                results[scale] = self.generate_with_control(prompt, control_vector, scale)

        return results
