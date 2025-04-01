# Hybrid Local DSPy RAG with Ollama Guide

A comprehensive guide for building a French RAG (Retrieval-Augmented Generation) system using hybrid retrieval methods (CamemBERT and BM25), DSPy, and Mistral for language generation.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Implementation Steps](#implementation-steps)
6. [Usage](#usage)
7. [Troubleshooting](#troubleshooting)

## Introduction

This guide demonstrates how to build a hybrid retrieval system for French text using:

- CamemBERT for semantic search
- BM25 for lexical search
- DSPy for orchestration
- Mistral (via Ollama) for language generation
- Metal (MPS) acceleration for Mac
- MLX for optimized computation

## Prerequisites

- Python 3.8+
- Mac with Metal support (for MPS acceleration)
- Ollama installed and running
- Sufficient RAM (at least 8GB recommended)
- Basic understanding of:
  - French language
  - RAG systems
  - Python programming
  - Jupyter notebooks

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/HybridLocalDSPyRAG-ollama.git
   cd HybridLocalDSPyRAG-ollama
   ```

2. Create and activate a virtual environment:

   ```bash
   # Create a new virtual environment with Python 3.11
   python3.11 -m venv venv
   
   # Activate the virtual environment
   source venv/bin/activate  # On Unix/macOS
   # or
   .\venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:

   ```bash
   # Upgrade pip first
   python -m pip install --upgrade pip
   
   # Install core dependencies
   pip install torch torchvision torchaudio
   pip install transformers
   pip install sentencepiece==0.1.99
   pip install nltk
   pip install rank_bm25
   pip install dspy-ai
   pip install ollama
   ```

4. Install MLX and configure Metal support:

   ```bash
   # Install MLX
   pip install mlx
   
   # Install additional MLX packages for vision and audio
   pip install mlx-vision
   pip install mlx-audio
   ```

5. Install Ollama and download Mistral:

   ```bash
   # Install Ollama (if not already installed)
   curl https://ollama.ai/install.sh | sh
   
   # Download Mistral model
   ollama pull mistral
   ```

## Project Structure

```
HybridLocalDSPyRAG-ollama/
├── notebooks/
│   ├── hybrid_local_dspy_rag.ipynb
│   └── hybrid_local_dspy_rag_guide.md
├── requirements.txt
└── README.md
```

## Implementation Steps

### 1. Setting Up the Environment

First, ensure your environment is properly configured:

```python
# Check Metal support
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
```

### 2. Loading CamemBERT

```python
from transformers import CamembertTokenizer, CamembertModel

tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base")
model = CamembertModel.from_pretrained("camembert/camembert-base")
model = model.to(device)
model.eval()
```

### 3. Implementing Hybrid Retrieval

The hybrid retrieval system combines:

1. **Semantic Search (CamemBERT)**
   - Uses contextual embeddings
   - Better for understanding meaning
   - Slower but more accurate

2. **Lexical Search (BM25)**
   - Uses term frequency
   - Faster processing
   - Good for exact matches

### 4. DSPy Integration

```python
import dspy
import ollama

class MistralOllamaLM(dspy.LM):
    def __init__(self, max_retries=3, timeout=30):
        super().__init__(model='mistral')
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = ollama.Client()

    def __call__(self, prompt, **kwargs):
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat(
                    model='mistral',
                    messages=[{"role": "user", "content": prompt}],
                    options={"timeout": self.timeout * 1000}
                )
                return response['message']['content']
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Failed to get response from Ollama after {self.max_retries} attempts: {str(e)}")
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
```

## Usage

1. Start Ollama service:

   ```bash
   ollama serve
   ```

2. Open the notebook:

   ```bash
   jupyter notebook notebooks/hybrid_local_dspy_rag.ipynb
   ```

3. Run cells in sequence

4. Test with a French query:

   ```python
   question = "Quel est l'impact du climat sur la nature ?"
   answer = qa_pipeline(question)
   print(f"Question: {question}\nRéponse: {answer}")
   ```

## Troubleshooting

### Common Issues

1. **Ollama Connection Issues**
   - Ensure Ollama is running
   - Check if Mistral model is downloaded
   - Verify network connectivity

2. **Metal (MPS) Issues**
   - Confirm Metal support on your Mac
   - Check PyTorch installation
   - Verify CUDA/CPU fallback

3. **Memory Issues**
   - Monitor RAM usage
   - Adjust batch sizes
   - Use smaller models if needed

### Performance Optimization

1. **Metal Acceleration**

   ```python
   # Enable Metal support
   import torch
   if torch.backends.mps.is_available():
       device = torch.device("mps")
   ```

2. **MLX Optimization**

   ```python
   import mlx.core as mx
   
   # Convert PyTorch tensors to MLX arrays
   def to_mlx(tensor):
       return mx.array(tensor.cpu().numpy())
   
   # Convert MLX arrays back to PyTorch tensors
   def to_torch(array):
       return torch.from_numpy(array.numpy())
   
   # Example usage with batch processing
   batch_size = 32
   for i in range(0, len(documents), batch_size):
       batch = documents[i:i + batch_size]
       # Convert to MLX for faster computation
       batch_mx = to_mlx(batch)
       # Process with MLX
       result_mx = process_with_mlx(batch_mx)
       # Convert back to PyTorch if needed
       result = to_torch(result_mx)
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 