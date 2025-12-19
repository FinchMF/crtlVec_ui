import os
from .base_model import LanguageModel
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
from typing import List, Optional, Dict, Tuple

BERT_LOCATION = os.path.expanduser("~/coding/local_models/bert-base-uncased/")


class BertControlModel(LanguageModel):
    """BERT model with control vector capabilities and activation capture."""

    def __init__(self):
        super().__init__()
        self.base_model = BertModel.from_pretrained(BERT_LOCATION, output_hidden_states=True)
        self.lm_model = BertForMaskedLM.from_pretrained(BERT_LOCATION)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_LOCATION)

        self.base_model.eval()
        self.lm_model.eval()

        # Model configuration
        self.hidden_size = self.lm_model.config.hidden_size
        self.num_layers = self.lm_model.config.num_hidden_layers
        self.vocab_size = self.lm_model.config.vocab_size

    @torch.no_grad()
    def get_hidden_state(self, prompt: str) -> torch.Tensor:
        """Get the hidden state for a prompt ([CLS] token, last layer)."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        outputs = self.base_model(**inputs)
        return outputs.hidden_states[-1][0, 0]  # [CLS] token

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
        layer_hidden_states = [h.squeeze(0) for h in outputs.hidden_states[1:]]

        return layer_hidden_states, tokens

    @torch.no_grad()
    def get_layer_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project hidden state to vocabulary logits using the MLM head."""
        # BERT's MLM head has transform, layer_norm, then decoder
        prediction_scores = self.lm_model.cls.predictions(hidden_state)
        return prediction_scores

    @torch.no_grad()
    def get_all_layer_logits(self, prompt: str) -> Tuple[List[torch.Tensor], List[str]]:
        """
        Get logits from projecting each layer's hidden states through the MLM head.

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
        top_k: int = 5,
        temperature: float = 1.0
    ) -> str:
        """
        Fill [MASK] tokens with control vector applied.

        Args:
            prompt: Input text with [MASK] token(s)
            control_vector: Control vector to apply
            scale: Scaling factor for control vector
            target_layers: Which layers to apply control vector (default: last layer only)
            top_k: Number of top predictions to consider
            temperature: Softmax temperature (higher = more random)

        Returns:
            Text with [MASK] replaced by prediction
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        if target_layers is None:
            target_layers = [self.num_layers - 1]

        hooks = []

        def create_hook(layer_idx: int):
            def hook_fn(module, input, output):
                if layer_idx in target_layers:
                    hidden_states = output[0]
                    modified_states = hidden_states + scale * control_vector.unsqueeze(0).unsqueeze(1)
                    return (modified_states,) + output[1:] if len(output) > 1 else (modified_states,)
                return output
            return hook_fn

        for idx in target_layers:
            hook = self.lm_model.bert.encoder.layer[idx].register_forward_hook(create_hook(idx))
            hooks.append(hook)

        try:
            outputs = self.lm_model(**inputs)
        finally:
            for hook in hooks:
                hook.remove()

        mask_token_index = torch.where(inputs.input_ids == self.tokenizer.mask_token_id)[1]
        if len(mask_token_index) == 0:
            return prompt + " [No mask token found]"

        mask_token_logits = outputs.logits[0, mask_token_index, :]
        # Apply temperature scaling before softmax
        scaled_logits = mask_token_logits / temperature
        top_tokens = torch.topk(scaled_logits, top_k, dim=1).indices[0]
        return prompt.replace("[MASK]", self.tokenizer.decode(top_tokens[0]))

    @torch.no_grad()
    def get_mask_predictions(
        self,
        prompt: str,
        control_vector: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        top_k: int = 10,
        target_layers: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Get detailed predictions for [MASK] token(s).

        Returns:
            List of dicts with 'token', 'probability' for each top-k prediction
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        hooks = []

        if control_vector is not None:
            if target_layers is None:
                target_layers = [self.num_layers - 1]

            def create_hook(layer_idx: int):
                def hook_fn(module, input, output):
                    if layer_idx in target_layers:
                        hidden_states = output[0]
                        modified_states = hidden_states + scale * control_vector.unsqueeze(0).unsqueeze(1)
                        return (modified_states,) + output[1:] if len(output) > 1 else (modified_states,)
                    return output
                return hook_fn

            for idx in target_layers:
                hook = self.lm_model.bert.encoder.layer[idx].register_forward_hook(create_hook(idx))
                hooks.append(hook)

        try:
            outputs = self.lm_model(**inputs)
        finally:
            for hook in hooks:
                hook.remove()

        mask_indices = torch.where(inputs.input_ids == self.tokenizer.mask_token_id)[1]

        if len(mask_indices) == 0:
            return []

        results = []
        for mask_idx in mask_indices:
            logits = outputs.logits[0, mask_idx, :]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)

            predictions = [
                {
                    'token': self.tokenizer.decode([idx.item()]),
                    'probability': prob.item(),
                    'token_id': idx.item()
                }
                for prob, idx in zip(top_probs, top_indices)
            ]
            results.append({
                'mask_position': mask_idx.item(),
                'predictions': predictions
            })

        return results

    @torch.no_grad()
    def compare_mask_predictions(
        self,
        prompt: str,
        control_vector: torch.Tensor,
        scales: List[float] = [-1.0, 0.0, 1.0],
        top_k: int = 10
    ) -> Dict[float, List[Dict]]:
        """
        Compare [MASK] predictions at different control vector scales.

        Returns:
            Dict mapping scale -> prediction results
        """
        results = {}
        for scale in scales:
            if scale == 0.0:
                results[scale] = self.get_mask_predictions(prompt, None, 0, top_k)
            else:
                results[scale] = self.get_mask_predictions(prompt, control_vector, scale, top_k)

        return results
