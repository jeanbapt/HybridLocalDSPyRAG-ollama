# HybridLocalDSPyRAG-ollama

A simple yet powerful French RAG (Retrieval-Augmented Generation) system that combines semantic and lexical search with local LLM inference.

## Features

- Hybrid retrieval (CamemBERT + BM25) for French text
- Local LLM inference using Ollama
- DSPy-powered RAG pipeline
- No cloud dependencies
- Metal (MPS) acceleration support for Mac
- MLX compatibility for optimal performance

## Quick Start

### Prerequisites
- Python 3.11+
- Ollama installed locally
- Mistral model pulled via Ollama
- Mac with Metal support (for MPS acceleration)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/HybridLocalDSPyRAG-ollama.git
   cd HybridLocalDSPyRAG-ollama
   ```

2. Set up the environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   # or
   .\venv\Scripts\activate  # On Windows
   pip install -r requirements.txt
   ```

3. Pull the Mistral model:
   ```bash
   ollama pull mistral
   ```

### Usage

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `notebooks/hybrid_local_dspy_rag.ipynb`

3. Run all cells in sequence

## Project Structure

```
.
├── notebooks/
│   └── hybrid_local_dspy_rag.ipynb  # Main implementation
├── requirements.txt                  # Dependencies
└── README.md                        # This file
```

## Key Components

### 1. Hybrid Retrieval
- CamemBERT for semantic search
- BM25 for lexical search
- Weighted combination of both approaches

### 2. Local LLM
- Mistral model via Ollama
- No API keys required
- Full privacy and control

### 3. DSPy Pipeline
- Modular RAG implementation
- Easy to extend and modify
- Clean integration between components

### 4. Hardware Acceleration
- Metal (MPS) support for Mac
- Automatic device selection
- MLX compatibility for optimal performance

## Hardware Acceleration Options

### Metal (MPS) Support
For Mac users with Metal support, the system automatically utilizes the Metal Performance Shaders (MPS) backend for PyTorch operations. This provides significant speedup for model inference and training.

To enable MPS:
```python
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
```

### MLX Compatibility
The system is compatible with Apple's MLX framework for optimal performance on Apple Silicon. To use MLX:

1. Install MLX:
   ```bash
   pip install mlx
   ```

2. Convert models to MLX format:
   ```python
   import mlx.core as mx
   # Convert PyTorch tensors to MLX arrays
   mlx_tensor = mx.array(pytorch_tensor.numpy())
   ```

## Limitations

- Optimized for French language
- Requires local GPU for best performance
- Limited by Mistral's context window
- Requires sufficient RAM (8GB+ recommended)
- Metal support limited to Mac with Apple Silicon or AMD GPUs

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with:
- CamemBERT for French language understanding
- DSPy for the RAG framework
- Mistral AI for the language model
- CursorAI for development assistance 