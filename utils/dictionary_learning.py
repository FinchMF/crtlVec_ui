"""
Dictionary Learning Module
Implements Sparse Autoencoder (SAE) for analyzing monosemantic vs polysemantic
features in neural network activations. Enables interpretability analysis
of what concepts individual neurons encode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class FeatureActivation:
    """Stores activation information for a single learned feature."""
    feature_idx: int
    activation_strength: float
    is_active: bool


@dataclass
class FeatureAnalysis:
    """Analysis of a single feature's activation patterns."""
    feature_idx: int
    mean_activation: float
    max_activation: float
    activation_frequency: float  # How often is this feature active?
    top_activating_examples: List[Tuple[str, float]] = field(default_factory=list)
    semantic_label: Optional[str] = None
    is_monosemantic: Optional[bool] = None  # None if not yet determined
    polysemantic_concepts: List[str] = field(default_factory=list)


@dataclass
class DictionaryAnalysisResult:
    """Complete analysis result from dictionary learning."""
    input_text: str
    tokens: List[str]
    feature_activations: List[List[FeatureActivation]]  # [seq_len, num_features]
    active_features: List[int]  # Indices of features that activated
    reconstruction_loss: float
    sparsity: float  # Fraction of features that are active

    def get_top_features(self, k: int = 10) -> List[Tuple[int, float]]:
        """Get top-k most activated features across all tokens."""
        feature_strengths = {}
        for token_features in self.feature_activations:
            for feat in token_features:
                if feat.feature_idx not in feature_strengths:
                    feature_strengths[feat.feature_idx] = 0
                feature_strengths[feat.feature_idx] = max(
                    feature_strengths[feat.feature_idx],
                    feat.activation_strength
                )

        sorted_features = sorted(
            feature_strengths.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:k]


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for learning interpretable features from activations.

    Architecture:
    - Encoder: hidden_dim -> expansion_factor * hidden_dim (with ReLU for sparsity)
    - Decoder: expansion_factor * hidden_dim -> hidden_dim

    The learned features in the expanded space tend to be more interpretable
    and often monosemantic (encoding single concepts).
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_factor: int = 4,
        sparsity_coefficient: float = 0.01,
        device: str = 'cpu'
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.dict_size = hidden_dim * expansion_factor
        self.sparsity_coefficient = sparsity_coefficient
        self.device = device

        # Encoder (with bias)
        self.encoder = nn.Linear(hidden_dim, self.dict_size, bias=True)

        # Decoder (tied weights would be W_enc.T, but we use separate for flexibility)
        self.decoder = nn.Linear(self.dict_size, hidden_dim, bias=True)

        # Initialize weights
        self._init_weights()

        self.to(device)

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse feature activations.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim] or [batch, hidden_dim]

        Returns:
            Sparse feature activations with ReLU applied
        """
        return F.relu(self.encoder(x))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to hidden space.

        Args:
            features: Sparse feature tensor

        Returns:
            Reconstructed hidden states
        """
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.

        Args:
            x: Input tensor

        Returns:
            Tuple of (reconstruction, features, loss)
        """
        features = self.encode(x)
        reconstruction = self.decode(features)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstruction, x)

        # Sparsity loss (L1 on activations)
        sparsity_loss = self.sparsity_coefficient * features.abs().mean()

        total_loss = recon_loss + sparsity_loss

        return reconstruction, features, total_loss

    def get_feature_activations(
        self,
        x: torch.Tensor,
        threshold: float = 0.1
    ) -> List[FeatureActivation]:
        """
        Get list of feature activations for an input.

        Args:
            x: Input tensor of shape [hidden_dim]
            threshold: Minimum activation to consider feature "active"

        Returns:
            List of FeatureActivation objects
        """
        with torch.no_grad():
            if x.dim() == 1:
                x = x.unsqueeze(0)
            features = self.encode(x).squeeze(0)

        activations = []
        for idx in range(self.dict_size):
            strength = features[idx].item()
            activations.append(FeatureActivation(
                feature_idx=idx,
                activation_strength=strength,
                is_active=strength > threshold
            ))

        return activations

    def get_decoder_vectors(self) -> torch.Tensor:
        """Get the decoder weight vectors (learned features)."""
        return self.decoder.weight.data  # [hidden_dim, dict_size]

    def get_feature_vector(self, feature_idx: int) -> torch.Tensor:
        """Get the decoder vector for a specific feature."""
        return self.decoder.weight.data[:, feature_idx]


class DictionaryLearner:
    """
    Manages training and analysis of sparse autoencoders for interpretability.
    """

    def __init__(
        self,
        hidden_dim: int,
        expansion_factor: int = 4,
        sparsity_coefficient: float = 0.01,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        self.hidden_dim = hidden_dim
        self.expansion_factor = expansion_factor
        self.device = device

        self.sae = SparseAutoencoder(
            hidden_dim=hidden_dim,
            expansion_factor=expansion_factor,
            sparsity_coefficient=sparsity_coefficient,
            device=device
        )

        self.optimizer = torch.optim.Adam(
            self.sae.parameters(),
            lr=learning_rate
        )

        # Feature analysis storage
        self.feature_examples: Dict[int, List[Tuple[str, float]]] = {}
        self.feature_stats: Dict[int, Dict] = {}

    def train_step(self, activations: torch.Tensor) -> float:
        """
        Single training step on a batch of activations.

        Args:
            activations: Tensor of shape [batch, hidden_dim] or [batch, seq, hidden_dim]

        Returns:
            Loss value
        """
        self.sae.train()
        self.optimizer.zero_grad()

        # Flatten if needed
        if activations.dim() == 3:
            batch, seq, hidden = activations.shape
            activations = activations.view(-1, hidden)

        activations = activations.to(self.device)

        _, _, loss = self.sae(activations)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_on_activations(
        self,
        activation_batches: List[torch.Tensor],
        epochs: int = 10,
        verbose: bool = True
    ) -> List[float]:
        """
        Train SAE on collected activations.

        Args:
            activation_batches: List of activation tensors
            epochs: Number of training epochs
            verbose: Whether to print progress

        Returns:
            List of loss values per epoch
        """
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            for batch in activation_batches:
                epoch_loss += self.train_step(batch)

            avg_loss = epoch_loss / len(activation_batches)
            losses.append(avg_loss)

            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        return losses

    @torch.no_grad()
    def analyze_activation(
        self,
        activation: torch.Tensor,
        text: str,
        tokens: List[str],
        threshold: float = 0.1
    ) -> DictionaryAnalysisResult:
        """
        Analyze activations using trained SAE.

        Args:
            activation: Hidden state tensor [seq_len, hidden_dim] or [batch, seq, hidden]
            text: Original input text
            tokens: Tokenized input
            threshold: Activation threshold

        Returns:
            DictionaryAnalysisResult with analysis
        """
        self.sae.eval()

        if activation.dim() == 3:
            activation = activation.squeeze(0)  # Remove batch dim

        activation = activation.to(self.device)

        # Get reconstruction and features
        reconstruction, features, loss = self.sae(activation)

        # Analyze per-token features
        all_feature_activations = []
        active_features_set = set()

        for seq_idx in range(activation.shape[0]):
            token_features = self.sae.get_feature_activations(
                activation[seq_idx],
                threshold
            )
            all_feature_activations.append(token_features)

            for feat in token_features:
                if feat.is_active:
                    active_features_set.add(feat.feature_idx)

        # Calculate sparsity
        total_activations = features.numel()
        active_count = (features > threshold).sum().item()
        sparsity = active_count / total_activations

        return DictionaryAnalysisResult(
            input_text=text,
            tokens=tokens,
            feature_activations=all_feature_activations,
            active_features=list(active_features_set),
            reconstruction_loss=loss.item(),
            sparsity=sparsity
        )

    def record_feature_example(
        self,
        feature_idx: int,
        text: str,
        activation_strength: float
    ):
        """Record an example that activated a feature for later analysis."""
        if feature_idx not in self.feature_examples:
            self.feature_examples[feature_idx] = []

        self.feature_examples[feature_idx].append((text, activation_strength))

        # Keep only top 20 examples
        self.feature_examples[feature_idx] = sorted(
            self.feature_examples[feature_idx],
            key=lambda x: x[1],
            reverse=True
        )[:20]

    def analyze_feature_semantics(
        self,
        feature_idx: int,
        examples: Optional[List[Tuple[str, float]]] = None
    ) -> FeatureAnalysis:
        """
        Analyze what semantic concept(s) a feature encodes.

        Args:
            feature_idx: Index of feature to analyze
            examples: Optional list of (text, activation) examples

        Returns:
            FeatureAnalysis with semantic information
        """
        if examples is None:
            examples = self.feature_examples.get(feature_idx, [])

        if not examples:
            return FeatureAnalysis(
                feature_idx=feature_idx,
                mean_activation=0.0,
                max_activation=0.0,
                activation_frequency=0.0
            )

        activations = [a for _, a in examples]

        return FeatureAnalysis(
            feature_idx=feature_idx,
            mean_activation=np.mean(activations),
            max_activation=np.max(activations),
            activation_frequency=len(examples) / 20.0,  # Normalized by max examples
            top_activating_examples=examples[:10]
        )

    def get_feature_similarity(
        self,
        feature_idx_1: int,
        feature_idx_2: int
    ) -> float:
        """Calculate cosine similarity between two feature vectors."""
        vec1 = self.sae.get_feature_vector(feature_idx_1)
        vec2 = self.sae.get_feature_vector(feature_idx_2)

        similarity = F.cosine_similarity(
            vec1.unsqueeze(0),
            vec2.unsqueeze(0)
        ).item()

        return similarity

    def find_similar_features(
        self,
        feature_idx: int,
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """Find k most similar features to a given feature."""
        target_vec = self.sae.get_feature_vector(feature_idx)
        all_vecs = self.sae.get_decoder_vectors().T  # [dict_size, hidden_dim]

        # Compute all similarities
        similarities = F.cosine_similarity(
            target_vec.unsqueeze(0),
            all_vecs
        )

        # Get top-k (excluding self)
        top_k_indices = similarities.argsort(descending=True)[1:k+1]

        return [
            (idx.item(), similarities[idx].item())
            for idx in top_k_indices
        ]

    def save(self, path: str):
        """Save the trained SAE and analysis data."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(self.sae.state_dict(), path / 'sae_weights.pt')

        # Save config
        config = {
            'hidden_dim': self.hidden_dim,
            'expansion_factor': self.expansion_factor
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f)

        # Save feature examples
        with open(path / 'feature_examples.json', 'w') as f:
            json.dump(self.feature_examples, f)

    def load(self, path: str):
        """Load a trained SAE and analysis data."""
        path = Path(path)

        # Load config
        with open(path / 'config.json') as f:
            config = json.load(f)

        # Reinitialize SAE with correct dimensions
        self.hidden_dim = config['hidden_dim']
        self.expansion_factor = config['expansion_factor']
        self.sae = SparseAutoencoder(
            hidden_dim=self.hidden_dim,
            expansion_factor=self.expansion_factor,
            device=self.device
        )

        # Load weights
        self.sae.load_state_dict(torch.load(path / 'sae_weights.pt'))

        # Load feature examples
        if (path / 'feature_examples.json').exists():
            with open(path / 'feature_examples.json') as f:
                self.feature_examples = json.load(f)
                # Convert keys back to int
                self.feature_examples = {
                    int(k): v for k, v in self.feature_examples.items()
                }

    def get_feature_semantic_description(
        self,
        feature_idx: int,
        top_k_examples: int = 10
    ) -> Dict:
        """
        Generate a semantic description of what a feature represents based on
        its activation patterns across training examples.

        Args:
            feature_idx: The feature index to describe
            top_k_examples: Number of top examples to analyze

        Returns:
            Dict with semantic description, confidence, and example tokens
        """
        examples = self.feature_examples.get(feature_idx, [])

        if not examples:
            return {
                'feature_idx': feature_idx,
                'description': 'Unknown (no activation data)',
                'confidence': 0.0,
                'top_tokens': [],
                'activation_pattern': 'sparse',
                'mean_activation': 0.0,
                'num_examples': 0
            }

        # Extract tokens/words that activate this feature
        top_examples = sorted(examples, key=lambda x: x[1], reverse=True)[:top_k_examples]

        # Analyze common patterns in the activating text
        all_words = []
        for text, activation in top_examples:
            # Simple tokenization - extract meaningful words
            words = [w.strip().lower() for w in text.replace('Ġ', ' ').replace('▁', ' ').split()
                    if len(w.strip()) > 2]
            all_words.extend(words)

        # Count word frequency
        from collections import Counter
        word_counts = Counter(all_words)
        common_words = word_counts.most_common(5)

        # Generate semantic description based on patterns
        if common_words:
            main_concepts = [w for w, c in common_words if c >= 2]
            if main_concepts:
                description = f"Responds to: {', '.join(main_concepts[:3])}"
            else:
                description = f"Pattern: {common_words[0][0]}"
        else:
            description = "Mixed/abstract pattern"

        # Calculate confidence based on consistency of activations
        activations = [a for _, a in top_examples]
        if len(activations) >= 2:
            activation_variance = np.var(activations)
            confidence = min(1.0, 1.0 / (1.0 + activation_variance))
        else:
            confidence = 0.5

        # Determine activation pattern
        mean_activation = np.mean(activations) if activations else 0
        if mean_activation > 0.5:
            pattern = 'strongly_active'
        elif mean_activation > 0.2:
            pattern = 'moderately_active'
        else:
            pattern = 'sparse'

        return {
            'feature_idx': feature_idx,
            'description': description,
            'confidence': float(confidence),
            'top_tokens': [text[:50] for text, _ in top_examples[:5]],
            'activation_pattern': pattern,
            'mean_activation': float(mean_activation),
            'num_examples': len(examples),
            'clustered_phrases': self._cluster_activating_phrases(top_examples)
        }

    def _cluster_activating_phrases(
        self,
        examples: List[Tuple[str, float]],
        num_clusters: int = 3
    ) -> List[Dict]:
        """
        Cluster the phrases that activated a feature to find semantic themes.

        Args:
            examples: List of (text, activation) tuples
            num_clusters: Number of clusters to create

        Returns:
            List of cluster dicts with theme and member phrases
        """
        if len(examples) < 3:
            # Not enough examples to cluster
            return [{
                'theme': 'All examples',
                'phrases': [text[:60] for text, _ in examples],
                'avg_activation': np.mean([a for _, a in examples]) if examples else 0
            }]

        # Simple keyword-based clustering
        from collections import defaultdict

        # Define semantic categories for clustering
        category_keywords = {
            'emotion_positive': ['love', 'great', 'amazing', 'beautiful', 'happy', 'fantastic', 'wonderful', 'excellent', 'joy', 'delicious'],
            'emotion_negative': ['hate', 'terrible', 'awful', 'bad', 'angry', 'sad', 'horrible', 'worst', 'anxious', 'fear'],
            'nature': ['weather', 'sky', 'forest', 'tree', 'sun', 'rain', 'ocean', 'mountain', 'flower', 'animal', 'cat', 'dog'],
            'science_tech': ['science', 'technology', 'computer', 'artificial', 'intelligence', 'research', 'discover', 'math', 'data', 'algorithm'],
            'action': ['walked', 'ran', 'played', 'created', 'made', 'solved', 'changed', 'moved', 'built', 'wrote'],
            'time': ['morning', 'evening', 'yesterday', 'today', 'tomorrow', 'ancient', 'new', 'old', 'future', 'past'],
            'people': ['she', 'he', 'they', 'children', 'doctor', 'scientist', 'artist', 'people', 'person', 'family'],
            'abstract': ['life', 'perspective', 'power', 'language', 'universe', 'world', 'idea', 'thought', 'meaning']
        }

        # Assign each example to a category
        categorized = defaultdict(list)
        uncategorized = []

        for text, activation in examples:
            text_lower = text.lower()
            assigned = False

            for category, keywords in category_keywords.items():
                if any(kw in text_lower for kw in keywords):
                    categorized[category].append((text, activation))
                    assigned = True
                    break

            if not assigned:
                uncategorized.append((text, activation))

        # Build clusters from categories
        clusters = []

        # Sort categories by number of examples
        sorted_categories = sorted(categorized.items(), key=lambda x: len(x[1]), reverse=True)

        for category, items in sorted_categories[:num_clusters]:
            if items:
                theme_name = category.replace('_', ' ').title()
                clusters.append({
                    'theme': theme_name,
                    'phrases': [text[:60] for text, _ in items[:5]],
                    'avg_activation': float(np.mean([a for _, a in items])),
                    'count': len(items)
                })

        # Add uncategorized if significant
        if uncategorized and len(clusters) < num_clusters:
            clusters.append({
                'theme': 'Other/Mixed',
                'phrases': [text[:60] for text, _ in uncategorized[:5]],
                'avg_activation': float(np.mean([a for _, a in uncategorized])),
                'count': len(uncategorized)
            })

        # If no clusters found, return single cluster
        if not clusters:
            clusters = [{
                'theme': 'General',
                'phrases': [text[:60] for text, _ in examples[:5]],
                'avg_activation': float(np.mean([a for _, a in examples])),
                'count': len(examples)
            }]

        return clusters

    def get_feature_neuron_composition(
        self,
        feature_idx: int,
        top_k_neurons: int = 20
    ) -> Dict:
        """
        Get which neurons contribute most to a given feature and their weights.

        Args:
            feature_idx: The feature index to analyze
            top_k_neurons: Number of top neurons to return

        Returns:
            Dict with neuron indices, weights, and statistics
        """
        # Get the decoder vector for this feature
        feature_vector = self.sae.get_feature_vector(feature_idx)
        weights = feature_vector.detach().cpu().numpy()

        # Find neurons with highest absolute weights
        abs_weights = np.abs(weights)
        top_neuron_indices = np.argsort(abs_weights)[-top_k_neurons:][::-1]

        # Calculate statistics
        total_neurons = len(weights)
        active_threshold = 0.01 * np.max(abs_weights)  # 1% of max
        active_neurons = np.sum(abs_weights > active_threshold)

        # Get weight distribution
        positive_weight = float(np.sum(weights[weights > 0]))
        negative_weight = float(np.sum(weights[weights < 0]))

        return {
            'feature_idx': feature_idx,
            'total_neurons': total_neurons,
            'active_neurons': int(active_neurons),
            'top_neuron_indices': top_neuron_indices.tolist(),
            'top_neuron_weights': weights[top_neuron_indices].tolist(),
            'positive_weight_sum': positive_weight,
            'negative_weight_sum': negative_weight,
            'weight_l2_norm': float(np.linalg.norm(weights)),
            'sparsity': float(1.0 - active_neurons / total_neurons)
        }

    def get_all_feature_info(
        self,
        max_features: int = 50,
        activation_threshold: float = 0.1
    ) -> List[Dict]:
        """
        Get comprehensive info for all features sorted by importance.

        Args:
            max_features: Maximum number of features to return
            activation_threshold: Minimum mean activation to consider

        Returns:
            List of feature info dictionaries
        """
        feature_infos = []

        for feat_idx in range(self.sae.dict_size):
            semantic = self.get_feature_semantic_description(feat_idx)
            neuron_comp = self.get_feature_neuron_composition(feat_idx)

            feature_infos.append({
                **semantic,
                **neuron_comp,
                'importance': semantic['mean_activation'] * neuron_comp['weight_l2_norm']
            })

        # Sort by importance and return top features
        feature_infos.sort(key=lambda x: x['importance'], reverse=True)
        return feature_infos[:max_features]

    def get_neuron_to_feature_mapping(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Get which features each neuron contributes to.

        Returns:
            Dict mapping neuron_idx to list of (feature_idx, weight) tuples
        """
        decoder_weights = self.sae.get_decoder_vectors().detach().cpu().numpy()
        # decoder_weights shape: [hidden_dim, dict_size]

        neuron_mapping = {}
        for neuron_idx in range(decoder_weights.shape[0]):
            neuron_weights = decoder_weights[neuron_idx, :]

            # Find features where this neuron has significant weight
            significant_features = []
            threshold = 0.01 * np.max(np.abs(neuron_weights))

            for feat_idx in range(decoder_weights.shape[1]):
                if abs(neuron_weights[feat_idx]) > threshold:
                    significant_features.append((feat_idx, float(neuron_weights[feat_idx])))

            # Sort by absolute weight
            significant_features.sort(key=lambda x: abs(x[1]), reverse=True)
            neuron_mapping[neuron_idx] = significant_features[:10]  # Top 10 features per neuron

        return neuron_mapping


class ModelDictionaryAnalyzer:
    """
    High-level interface for dictionary learning on language model activations.
    """

    def __init__(
        self,
        model,
        layer_idx: int = -1,
        expansion_factor: int = 4,
        device: str = 'cpu'
    ):
        """
        Initialize dictionary analyzer for a model.

        Args:
            model: Language model (GPT2ControlModel or BertControlModel)
            layer_idx: Which layer to analyze (-1 for last)
            expansion_factor: SAE expansion factor
            device: Computation device
        """
        self.model = model
        self.layer_idx = layer_idx
        self.device = device

        # Determine hidden dimension
        if hasattr(model, 'lm_model'):
            if hasattr(model.lm_model, 'config'):
                self.hidden_dim = model.lm_model.config.hidden_size
            else:
                # Fallback
                self.hidden_dim = 768
        else:
            self.hidden_dim = 768

        self.dictionary_learner = DictionaryLearner(
            hidden_dim=self.hidden_dim,
            expansion_factor=expansion_factor,
            device=device
        )

        self.collected_activations: List[torch.Tensor] = []

    @torch.no_grad()
    def collect_activations(self, prompts: List[str]):
        """
        Collect activations from a list of prompts for training.

        Args:
            prompts: List of text prompts
        """
        for prompt in prompts:
            inputs = self.model.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            # Get hidden states
            if hasattr(self.model, 'lm_model'):
                outputs = self.model.lm_model(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[self.layer_idx]
            else:
                outputs = self.model.base_model(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[self.layer_idx]

            self.collected_activations.append(hidden_states.detach().cpu())

    def train_dictionary(self, epochs: int = 10, verbose: bool = True) -> List[float]:
        """
        Train the sparse autoencoder on collected activations.

        Args:
            epochs: Number of training epochs
            verbose: Whether to print progress

        Returns:
            List of loss values
        """
        if not self.collected_activations:
            raise ValueError("No activations collected. Call collect_activations first.")

        return self.dictionary_learner.train_on_activations(
            self.collected_activations,
            epochs=epochs,
            verbose=verbose
        )

    @torch.no_grad()
    def analyze_prompt(
        self,
        prompt: str,
        threshold: float = 0.1
    ) -> DictionaryAnalysisResult:
        """
        Analyze a prompt using the trained dictionary.

        Args:
            prompt: Input text
            threshold: Activation threshold

        Returns:
            DictionaryAnalysisResult
        """
        inputs = self.model.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        tokens = self.model.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])

        # Get hidden states
        if hasattr(self.model, 'lm_model'):
            outputs = self.model.lm_model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[self.layer_idx]
        else:
            outputs = self.model.base_model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[self.layer_idx]

        return self.dictionary_learner.analyze_activation(
            hidden_states,
            prompt,
            tokens,
            threshold
        )

    def compare_prompts(
        self,
        prompt1: str,
        prompt2: str,
        threshold: float = 0.1
    ) -> Tuple[DictionaryAnalysisResult, DictionaryAnalysisResult, List[int], List[int]]:
        """
        Compare which features activate for two different prompts.

        Args:
            prompt1: First prompt
            prompt2: Second prompt
            threshold: Activation threshold

        Returns:
            Tuple of (result1, result2, shared_features, unique_features)
        """
        result1 = self.analyze_prompt(prompt1, threshold)
        result2 = self.analyze_prompt(prompt2, threshold)

        set1 = set(result1.active_features)
        set2 = set(result2.active_features)

        shared = list(set1 & set2)
        unique_to_1 = list(set1 - set2)
        unique_to_2 = list(set2 - set1)

        return result1, result2, shared, unique_to_1 + unique_to_2

    def analyze_control_vector_features(
        self,
        control_vector: torch.Tensor
    ) -> List[Tuple[int, float]]:
        """
        Analyze which dictionary features a control vector activates.

        Args:
            control_vector: The control vector to analyze

        Returns:
            List of (feature_idx, activation_strength) tuples
        """
        activations = self.dictionary_learner.sae.get_feature_activations(
            control_vector.to(self.device)
        )

        # Sort by activation strength
        sorted_activations = sorted(
            [(a.feature_idx, a.activation_strength) for a in activations],
            key=lambda x: abs(x[1]),
            reverse=True
        )

        return sorted_activations[:20]  # Top 20 features

    @torch.no_grad()
    def get_feature_activations_per_layer(
        self,
        feature_idx: int,
        prompt: str = "The quick brown fox jumps over the lazy dog."
    ) -> Dict[int, Dict]:
        """
        Get how strongly a feature activates at each layer of the model.

        This shows the "path" of a feature through the network - which neurons
        at each layer contribute to this semantic feature.

        Args:
            feature_idx: The feature to analyze
            prompt: A sample prompt to run through the model

        Returns:
            Dict mapping layer_idx to activation info:
            {
                layer_idx: {
                    'feature_activation': float,  # How strongly feature activates
                    'top_neurons': [(neuron_idx, activation), ...],  # Most active neurons
                    'mean_activation': float,
                    'max_activation': float
                }
            }
        """
        inputs = self.model.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Get hidden states from ALL layers
        if hasattr(self.model, 'lm_model'):
            outputs = self.model.lm_model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )
            # hidden_states is tuple: (embedding, layer0, layer1, ..., layerN)
            all_hidden_states = outputs.hidden_states
        else:
            outputs = self.model.base_model(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )
            all_hidden_states = outputs.hidden_states

        # Get the feature's decoder vector (what pattern it looks for)
        feature_vector = self.dictionary_learner.sae.get_feature_vector(feature_idx)
        feature_vector = feature_vector.detach().cpu().numpy()

        layer_activations = {}

        # For each layer (skip embedding at index 0)
        for layer_idx, hidden_state in enumerate(all_hidden_states[1:]):
            # hidden_state shape: [batch, seq_len, hidden_dim]
            # Average over sequence to get layer representation
            layer_repr = hidden_state.mean(dim=1).squeeze(0).detach().cpu().numpy()

            # Project through the SAE encoder to get feature activation
            hidden_tensor = torch.tensor(layer_repr, dtype=torch.float32).unsqueeze(0)
            features = self.dictionary_learner.sae.encode(hidden_tensor.to(self.device))
            feature_activation = features[0, feature_idx].item()

            # Find which "neurons" (hidden dims) contribute most to this activation
            # by looking at the dot product of hidden state with feature vector
            neuron_contributions = layer_repr * feature_vector
            top_neuron_indices = np.argsort(np.abs(neuron_contributions))[-20:][::-1]
            top_neurons = [
                (int(idx), float(neuron_contributions[idx]))
                for idx in top_neuron_indices
            ]

            layer_activations[layer_idx] = {
                'feature_activation': feature_activation,
                'top_neurons': top_neurons,
                'mean_activation': float(np.mean(np.abs(neuron_contributions))),
                'max_activation': float(np.max(np.abs(neuron_contributions))),
                'active_neuron_indices': [int(idx) for idx in top_neuron_indices[:10]]
            }

        return layer_activations

    def save(self, path: str):
        """Save the analyzer state."""
        self.dictionary_learner.save(path)

    def load(self, path: str):
        """Load a previously trained analyzer."""
        self.dictionary_learner.load(path)

    @torch.no_grad()
    def populate_feature_examples(
        self,
        prompts: List[str],
        threshold: float = 0.1
    ) -> Dict[int, List[Dict]]:
        """
        Analyze prompts and record which features they activate.
        This populates the feature_examples dictionary for semantic analysis.

        Args:
            prompts: List of text prompts to analyze
            threshold: Activation threshold for features

        Returns:
            Dict mapping feature_idx to list of activation examples with token info
        """
        feature_token_examples = {}  # Stores detailed token-level info

        for prompt in prompts:
            try:
                result = self.analyze_prompt(prompt, threshold)
                tokens = result.tokens

                # For each token position, check which features activated
                for token_idx, token_features in enumerate(result.feature_activations):
                    token_text = tokens[token_idx] if token_idx < len(tokens) else "<unk>"

                    for feat in token_features:
                        if feat.is_active and feat.activation_strength > threshold:
                            feat_idx = feat.feature_idx

                            # Record in dictionary_learner for basic semantic analysis
                            self.dictionary_learner.record_feature_example(
                                feat_idx,
                                prompt,
                                feat.activation_strength
                            )

                            # Also store detailed token-level info
                            if feat_idx not in feature_token_examples:
                                feature_token_examples[feat_idx] = []

                            feature_token_examples[feat_idx].append({
                                'prompt': prompt,
                                'token': token_text,
                                'token_idx': token_idx,
                                'activation': feat.activation_strength,
                                'context_tokens': tokens[max(0, token_idx-2):min(len(tokens), token_idx+3)]
                            })
            except Exception as e:
                print(f"Error analyzing prompt '{prompt[:30]}...': {e}")
                continue

        # Sort each feature's examples by activation strength and keep top examples
        for feat_idx in feature_token_examples:
            feature_token_examples[feat_idx] = sorted(
                feature_token_examples[feat_idx],
                key=lambda x: x['activation'],
                reverse=True
            )[:30]  # Keep top 30 examples per feature

        # Store for later access
        self.feature_token_examples = feature_token_examples

        return feature_token_examples

    def get_feature_activating_phrases(
        self,
        feature_idx: int,
        max_examples: int = 10
    ) -> List[Dict]:
        """
        Get phrases that activated a specific feature with the most activating token highlighted.

        Args:
            feature_idx: The feature index
            max_examples: Maximum examples to return

        Returns:
            List of dicts with phrase, highlighted token, and activation info
        """
        if not hasattr(self, 'feature_token_examples'):
            return []

        examples = self.feature_token_examples.get(feature_idx, [])[:max_examples]

        result = []
        for ex in examples:
            # Create highlighted version of the phrase
            prompt = ex['prompt']
            token = ex['token']
            activation = ex['activation']

            # Clean token for display (remove special chars like Ġ)
            clean_token = token.replace('Ġ', ' ').replace('▁', ' ').strip()

            result.append({
                'phrase': prompt,
                'highlighted_token': clean_token,
                'activation': activation,
                'context': ' '.join([t.replace('Ġ', '').replace('▁', '') for t in ex.get('context_tokens', [])])
            })

        return result


# ============================================================================
# sklearn-based Dictionary Learning (alternative approach)
# ============================================================================

class SklearnDictionaryLearner:
    """
    Dictionary Learning using sklearn's implementation.

    This provides an alternative to the neural network-based sparse autoencoder,
    using classical sparse coding algorithms which can be faster and more stable
    for smaller datasets.
    """

    def __init__(
        self,
        n_components: int = 64,
        alpha: float = 1.0,
        max_iter: int = 1000,
        transform_algorithm: str = 'lasso_lars',
        fit_algorithm: str = 'lars',
        random_state: int = 42
    ):
        """
        Initialize sklearn dictionary learner.

        Args:
            n_components: Number of dictionary atoms (features) to learn
            alpha: Sparsity controlling parameter
            max_iter: Maximum number of iterations
            transform_algorithm: Algorithm for sparse coding ('lasso_lars', 'omp', etc.)
            fit_algorithm: Algorithm for dictionary fitting ('lars', 'cd')
            random_state: Random seed for reproducibility
        """
        from sklearn.decomposition import DictionaryLearning, MiniBatchDictionaryLearning

        self.n_components = n_components
        self.alpha = alpha
        self.fitted = False

        self.model = DictionaryLearning(
            n_components=n_components,
            alpha=alpha,
            max_iter=max_iter,
            transform_algorithm=transform_algorithm,
            fit_algorithm=fit_algorithm,
            random_state=random_state,
            positive_code=True  # Non-negative sparse codes like ReLU
        )

        # For online/incremental learning
        self.mini_batch_model = MiniBatchDictionaryLearning(
            n_components=n_components,
            alpha=alpha,
            max_iter=max_iter,
            transform_algorithm=transform_algorithm,
            random_state=random_state,
            positive_code=True
        )

        self.dictionary = None
        self.feature_names = None

    def fit(self, activation_matrix: np.ndarray):
        """
        Fit dictionary to activation data.

        Args:
            activation_matrix: Shape [n_samples, n_features]
        """
        self.model.fit(activation_matrix)
        self.dictionary = self.model.components_  # [n_components, n_features]
        self.fitted = True
        return self

    def fit_incremental(self, activation_matrix: np.ndarray, batch_size: int = 100):
        """
        Fit dictionary incrementally using mini-batches.

        Args:
            activation_matrix: Shape [n_samples, n_features]
            batch_size: Size of mini-batches
        """
        n_samples = activation_matrix.shape[0]
        for i in range(0, n_samples, batch_size):
            batch = activation_matrix[i:i+batch_size]
            self.mini_batch_model.partial_fit(batch)

        self.dictionary = self.mini_batch_model.components_
        self.fitted = True
        return self

    def transform(self, activation_matrix: np.ndarray) -> np.ndarray:
        """
        Transform activations to sparse codes.

        Args:
            activation_matrix: Shape [n_samples, n_features]

        Returns:
            Sparse codes of shape [n_samples, n_components]
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before transform")
        return self.model.transform(activation_matrix)

    def fit_transform(self, activation_matrix: np.ndarray) -> np.ndarray:
        """
        Fit dictionary and return sparse codes.

        Args:
            activation_matrix: Shape [n_samples, n_features]

        Returns:
            Sparse codes of shape [n_samples, n_components]
        """
        self.fit(activation_matrix)
        return self.transform(activation_matrix)

    def get_dictionary(self) -> np.ndarray:
        """Get learned dictionary atoms."""
        return self.dictionary

    def get_atom(self, atom_idx: int) -> np.ndarray:
        """Get a specific dictionary atom."""
        return self.dictionary[atom_idx]

    def interpret_atoms(
        self,
        feature_names: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Interpret dictionary atoms by finding top contributing features.

        Args:
            feature_names: Optional names for input features
            top_k: Number of top features to return per atom

        Returns:
            List of dicts with atom interpretations
        """
        if self.dictionary is None:
            return []

        interpretations = []
        for atom_idx in range(self.n_components):
            atom = self.dictionary[atom_idx]

            # Find top positive and negative contributions
            sorted_indices = np.argsort(np.abs(atom))[::-1][:top_k]

            top_features = []
            for idx in sorted_indices:
                name = feature_names[idx] if feature_names and idx < len(feature_names) else f"feat_{idx}"
                top_features.append({
                    'name': name,
                    'weight': float(atom[idx]),
                    'index': int(idx)
                })

            interpretations.append({
                'atom_idx': atom_idx,
                'top_features': top_features,
                'l2_norm': float(np.linalg.norm(atom)),
                'sparsity': float(np.mean(np.abs(atom) < 0.01))
            })

        return interpretations

    def get_semantic_features(
        self,
        activation_matrix: np.ndarray,
        tokens: List[str],
        threshold: float = 0.1
    ) -> Dict[str, List[int]]:
        """
        Get semantic features that activate for each token.

        Args:
            activation_matrix: Shape [n_tokens, n_features]
            tokens: List of token strings
            threshold: Activation threshold

        Returns:
            Dict mapping tokens to lists of active atom indices
        """
        sparse_codes = self.transform(activation_matrix)

        result = {}
        for i, token in enumerate(tokens):
            active_atoms = np.where(sparse_codes[i] > threshold)[0].tolist()
            result[token] = active_atoms

        return result

    def reconstruction_error(self, activation_matrix: np.ndarray) -> float:
        """
        Compute reconstruction error.

        Args:
            activation_matrix: Shape [n_samples, n_features]

        Returns:
            Mean squared reconstruction error
        """
        sparse_codes = self.transform(activation_matrix)
        reconstruction = sparse_codes @ self.dictionary
        return float(np.mean((activation_matrix - reconstruction) ** 2))

    def analyze_sparsity(self, activation_matrix: np.ndarray) -> Dict[str, float]:
        """
        Analyze sparsity of learned representation.

        Args:
            activation_matrix: Shape [n_samples, n_features]

        Returns:
            Dict with sparsity statistics
        """
        sparse_codes = self.transform(activation_matrix)

        return {
            'mean_active_atoms': float(np.mean(np.sum(sparse_codes > 0, axis=1))),
            'fraction_active': float(np.mean(sparse_codes > 0)),
            'mean_code_l1': float(np.mean(np.abs(sparse_codes))),
            'mean_code_l2': float(np.mean(np.linalg.norm(sparse_codes, axis=1)))
        }


def learn_neuron_dictionary(
    activation_matrix: np.ndarray,
    n_components: int = 20,
    alpha: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Learn dictionary from neuron activations using sklearn.

    Convenience function matching monolith_app.py interface.

    Args:
        activation_matrix: Shape [n_samples, n_features]
        n_components: Number of dictionary atoms
        alpha: Sparsity parameter

    Returns:
        Tuple of (dictionary, sparse_codes)
    """
    learner = SklearnDictionaryLearner(n_components=n_components, alpha=alpha)
    sparse_codes = learner.fit_transform(activation_matrix)
    return learner.get_dictionary(), sparse_codes


def interpret_dictionary_atoms(
    dictionary: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k: int = 5
) -> List[Dict]:
    """
    Interpret dictionary atoms.

    Convenience function matching monolith_app.py interface.

    Args:
        dictionary: Shape [n_atoms, n_features]
        feature_names: Optional feature names
        top_k: Number of top features per atom

    Returns:
        List of interpretation dicts
    """
    interpretations = []
    for atom_idx in range(dictionary.shape[0]):
        atom = dictionary[atom_idx]
        sorted_indices = np.argsort(np.abs(atom))[::-1][:top_k]

        top_features = []
        for idx in sorted_indices:
            name = feature_names[idx] if feature_names and idx < len(feature_names) else f"dim_{idx}"
            top_features.append({
                'name': name,
                'weight': float(atom[idx])
            })

        interpretations.append({
            'atom_idx': atom_idx,
            'top_features': top_features
        })

    return interpretations


def get_semantic_features_for_neurons(
    activation_matrix: np.ndarray,
    dictionary: np.ndarray,
    threshold: float = 0.1
) -> np.ndarray:
    """
    Get sparse semantic features for each sample.

    Convenience function matching monolith_app.py interface.

    Args:
        activation_matrix: Shape [n_samples, n_features]
        dictionary: Shape [n_atoms, n_features]
        threshold: Activation threshold

    Returns:
        Binary mask of active features [n_samples, n_atoms]
    """
    from sklearn.decomposition import SparseCoder

    coder = SparseCoder(
        dictionary=dictionary,
        transform_algorithm='lasso_lars',
        positive_code=True
    )

    sparse_codes = coder.transform(activation_matrix)
    return (sparse_codes > threshold).astype(np.float32)

