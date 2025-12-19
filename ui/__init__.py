"""
UI Module

Provides Gradio UI components and visualization tools:
- Main control vector UI with tabs
- Layer visualization (heatmaps, 3D manifolds, graphs)
- Animated Plotly visualizations
- Interactive vector editor
"""

from .tabs import (
    ControlVectorUI,
    LayerAnalysisTab,
    EntropyAnalysisTab,
    DictionaryLearningTab,
    VectorEditorTab,
    EducationalDemoTab,
    EducationalModelsTab
)
from .components import ModelTab, CustomVectorTab
from .layer_viz import (
    LayerVisualizer,
    EntropyVisualizer,
    DictionaryVisualizer,
    VisualizationConfig
)
from .vector_editor import VectorEditor, VectorEditorUI, VectorVisualizerUI
from .animated_viz import (
    AnimatedNetworkViz,
    AttentionHeadViz,
    ManifoldAnimator,
    NetworkNode,
    NetworkEdge,
    figure_to_image
)

__all__ = [
    # Main UI
    'ControlVectorUI',

    # Tabs
    'LayerAnalysisTab',
    'EntropyAnalysisTab',
    'DictionaryLearningTab',
    'VectorEditorTab',
    'EducationalDemoTab',
    'EducationalModelsTab',

    # Components
    'ModelTab',
    'CustomVectorTab',
    'VectorEditor',
    'VectorEditorUI',
    'VectorVisualizerUI',

    # Visualization
    'LayerVisualizer',
    'EntropyVisualizer',
    'DictionaryVisualizer',
    'VisualizationConfig',

    # Animated visualization
    'AnimatedNetworkViz',
    'AttentionHeadViz',
    'ManifoldAnimator',
    'NetworkNode',
    'NetworkEdge',
    'figure_to_image',
]
