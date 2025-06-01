# Latent Control Vector Generator

A web-based interface for exploring and manipulating language models' latent spaces using control vectors. This application demonstrates controlled text generation by combining multiple semantic directions in transformer model hidden states.

## Overview

This application implements a method for controlling language models (GPT-2 and BERT) through:
- Extraction and manipulation of semantic directions from contrastive examples
- Dynamic combination of multiple control vectors
- Real-time 2D/3D PCA visualization of latent spaces
- Custom vector creation and persistence

## Features

### Model Support
- GPT-2: Autoregressive text generation with control vector injection
- BERT: Masked language modeling with controlled token prediction

### Pre-defined Control Dimensions
- Sentiment (extreme positive/negative contrasts)
- Formality (highly formal/informal language)
- Tense (present/past, GPT-2 only)

### Technical Features
- Dynamic vector creation from contrastive examples
- Multi-vector combination with adjustable strength
- Interactive 2D/3D PCA visualization
- Vector persistence and loading
- Custom vector creation interface

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
./run_app.sh
```

## Development Guide

### Project Structure
```
ControlVectorAnalysis/
├── models/          # Model-specific implementations
├── controllers/     # Business logic and vector management
├── utils/          # Visualization and configuration utilities
├── ui/             # Gradio interface components
├── config/         # Model prompts and configurations
└── app.py          # Application entry point
```

### Key Components
- **Base Model Interface**: Abstract class defining vector operations
- **Vector Controllers**: Manages vector creation and persistence
- **PCA Visualization**: Real-time latent space visualization
- **Config Management**: JSON-based prompt/vector configuration

### Adding New Models
1. Implement `LanguageModel` base class
2. Define vector extraction method
3. Implement control injection mechanism
4. Add model-specific prompt sets
5. Update UI components

### Vector Creation Process
1. Define contrastive example pairs
2. Extract hidden states from model
3. Compute difference vectors
4. Apply during generation via forward hooks

## Usage Examples

### Basic Control
```python
# Generate text with sentiment control
output = model_controller.generate_text(
    "gpt2",
    "The movie was",
    ["Sentiment"],
    strength=1.0
)
```

### Custom Vector Creation
```python
# Create and save custom vector
controller.create_control_vector(
    "gpt2",
    "CustomStyle",
    positive_examples=["..."],
    negative_examples=["..."]
)
```

## Advanced Features

### PCA Visualization
- Toggle between 2D/3D visualizations
- Real-time vector space exploration
- Interactive dimension reduction

### Vector Combination
- Multiple control vector selection
- Adjustable strength (-2 to +2)
- Dynamic vector space updates

## Known Limitations
- Model-specific vector spaces are not interchangeable
- Control strength may need adjustment per vector
- Custom vectors require careful example curation

## License

MIT License
