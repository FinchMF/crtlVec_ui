"""
Models Module

Provides neural network model wrappers and educational models:
- GPT-2 model wrapper with control vector support
- BERT model wrapper with control vector support
- Educational toy models (TinyMLP, TinyLSTM, TinyGPT)
"""

from .base_model import LanguageModel
from .gpt2_model import GPT2ControlModel
from .bert_model import BertControlModel
from .educational_models import (
    TinyMLP,
    TinyLSTM,
    TinyGPT,
    MaskedGPT,
    TransformerBlock,
    ActivationSnapshot,
    compute_mean_activations,
    compute_single_activations,
    get_lstm_temporal_activations
)

__all__ = [
    # Base
    'LanguageModel',

    # Language models
    'GPT2ControlModel',
    'BertControlModel',

    # Educational models
    'TinyMLP',
    'TinyLSTM',
    'TinyGPT',
    'MaskedGPT',
    'TransformerBlock',
    'ActivationSnapshot',

    # Utility functions
    'compute_mean_activations',
    'compute_single_activations',
    'get_lstm_temporal_activations',
]
