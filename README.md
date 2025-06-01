# Latent Control Vector Generator

A web-based interface for exploring and manipulating language models' latent spaces using control vectors. This application allows users to influence text generation by combining different semantic directions in the models' hidden states.

## Overview

This application implements a method for controlling language models (GPT-2 and BERT) by:
- Extracting semantic directions from pairs of examples
- Combining multiple control vectors
- Visualizing the latent space using PCA
- Supporting custom control vector creation

## Features

- Support for multiple models:
  - GPT-2 (autoregressive text generation)
  - BERT (masked language modeling)
- Pre-defined control vectors for:
  - Sentiment (positive/negative)
  - Formality (formal/informal)
  - Tense (present/past, GPT-2 only)
- Interactive text generation with adjustable control strength
- Real-time PCA visualization of embedding spaces
- Custom control vector creation from user examples
- Multi-vector combination support

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

## Usage

### GPT-2 Text Generation

1. Open the "GPT-2" tab
2. Enter a prompt (e.g., "The food was")
3. Select one or more control vectors
4. Adjust the control strength (-2 to +2)
5. Click "Generate with Control"

### BERT Masked Prediction

1. Open the "BERT" tab
2. Enter a prompt with [MASK] token (e.g., "This movie is [MASK].")
3. Select one or more control vectors
4. Adjust the control strength (-2 to +2)
5. Click "Generate with Control"

### Creating Custom Control Vectors

1. Open the "Create Custom Vector" tab
2. Enter a name for your control vector
3. Add positive examples (one per line)
4. Add negative examples (one per line)
5. Click "Save Custom Vector"

### Tips

- Combine multiple vectors to achieve more complex effects
- Use the PCA visualization to understand vector relationships
- Experiment with different control strengths
- Keep prompts simple and consistent when creating custom vectors

## Technical Details

The application works by:
1. Extracting hidden states from the models:
   - Last layer for GPT-2
   - [CLS] token embeddings for BERT
2. Computing difference vectors between positive and negative examples
3. Injecting these vectors during generation using forward hooks
4. Visualizing the latent space using PCA dimensionality reduction

## License

MIT License
