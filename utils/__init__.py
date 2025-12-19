"""
Utils Module

Provides utility classes and functions for:
- Activation capture and analysis
- Dictionary learning (sparse autoencoders)
- Entropy analysis
- Token journey tracking
- Visualization tools
- Configuration storage
"""

from .storage import ConfigManager
from .visualization import LatentSpaceVisualizer, ControlVectorVisualizer
from .activation_capture import (
    ActivationCaptureHook,
    GPT2ActivationCapture,
    BertActivationCapture,
    LayerActivation,
    ActivationTrace
)
from .entropy_analysis import (
    EntropyAnalyzer,
    GPT2EntropyAnalyzer,
    BertEntropyAnalyzer,
    TokenPrediction,
    LayerEntropyAnalysis,
    EntropyTrace
)
from .dictionary_learning import (
    SparseAutoencoder,
    DictionaryLearner,
    ModelDictionaryAnalyzer,
    FeatureActivation,
    FeatureAnalysis,
    DictionaryAnalysisResult,
    SklearnDictionaryLearner,
    learn_neuron_dictionary,
    interpret_dictionary_atoms,
    get_semantic_features_for_neurons
)
from .token_journey import (
    TokenJourney,
    CompactTokenJourney,
    WeightSnapshot,
    SequenceJourney,
    TokenJourneyTracker
)

__all__ = [
    # Storage
    'ConfigManager',

    # Visualization
    'LatentSpaceVisualizer',
    'ControlVectorVisualizer',

    # Activation capture
    'ActivationCaptureHook',
    'GPT2ActivationCapture',
    'BertActivationCapture',
    'LayerActivation',
    'ActivationTrace',

    # Entropy analysis
    'EntropyAnalyzer',
    'GPT2EntropyAnalyzer',
    'BertEntropyAnalyzer',
    'TokenPrediction',
    'LayerEntropyAnalysis',
    'EntropyTrace',

    # Dictionary learning
    'SparseAutoencoder',
    'DictionaryLearner',
    'ModelDictionaryAnalyzer',
    'FeatureActivation',
    'FeatureAnalysis',
    'DictionaryAnalysisResult',
    'SklearnDictionaryLearner',
    'learn_neuron_dictionary',
    'interpret_dictionary_atoms',
    'get_semantic_features_for_neurons',

    # Token journey
    'TokenJourney',
    'CompactTokenJourney',
    'WeightSnapshot',
    'SequenceJourney',
    'TokenJourneyTracker',
]
