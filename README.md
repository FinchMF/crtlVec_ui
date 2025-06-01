# GPT-2 Latent Control Vector Generator

A web-based interface for exploring and manipulating GPT-2's latent space using control vectors. This application allows users to influence text generation by combining different semantic directions in the model's hidden states.

## Overview

This application implements a method for controlling GPT-2's text generation by:
- Extracting semantic directions from pairs of examples
- Combining multiple control vectors
- Visualizing the latent space using PCA
- Supporting custom control vector creation

## Features

- Pre-defined control vectors for:
  - Sentiment (positive/negative)
  - Formality (formal/informal)
  - Tense (present/past)
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

### Basic Text Generation

1. Open the "Generate" tab
2. Enter a prompt (e.g., "The food was")
3. Select one or more control vectors (e.g., "Sentiment")
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
1. Extracting hidden states from the last layer of GPT-2
2. Computing difference vectors between positive and negative examples
3. Injecting these vectors during generation using forward hooks
4. Visualizing the latent space using PCA dimensionality reduction

## License

MIT License
