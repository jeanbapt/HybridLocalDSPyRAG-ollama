# HybridLocalDSPyRAG-ollama

A simple yet powerful French RAG (Retrieval-Augmented Generation) system that combines semantic and lexical search with local LLM inference.

## Features

- Hybrid retrieval (CamemBERT + BM25) for French text
- Local LLM inference using Ollama
- DSPy-powered RAG pipeline
- Metal (MPS) acceleration support for Mac

## Quick Start

### Prerequisites
- Python 3.11+
- Ollama installed locally
- Mac with Metal support (recommended)
- 8GB+ RAM recommended

### Installation

1. Clone and setup:
   ```bash
   git clone https://github.com/yourusername/HybridLocalDSPyRAG-ollama.git
   cd HybridLocalDSPyRAG-ollama
   python -m venv venv
   source venv/bin/activate  # On Unix/macOS
   pip install -r requirements.txt
   ```

2. Pull the Mistral model:
   ```bash
   ollama pull mistral
   ```

### Usage

You can use the system in two ways:

1. **Jupyter Notebook** (interactive exploration):
   ```bash
   jupyter notebook
   # Open notebooks/hybrid_local_dspy_rag.ipynb
   ```

2. **Python Script** (direct usage):
   ```bash
   python hybrid_rag.py
   ```

## Project Structure

```
.
├── hybrid_rag.py               # Standalone Python implementation
├── notebooks/
│   ├── hybrid_local_dspy_rag.ipynb     # Interactive notebook
│   └── hybrid_local_dspy_rag_guide.md  # Detailed guide
├── requirements.txt
└── README.md
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citations and Acknowledgments

This project builds upon several outstanding open-source projects and research papers:

### Models and Papers

- **CamemBERT**: A state-of-the-art language model for French
  ```bibtex
  @inproceedings{martin2020camembert,
    title={CamemBERT: a Tasty French Language Model},
    author={Martin, Louis and Muller, Benjamin and Su{\'a}rez, Pedro Javier Ortiz and Dupont, Yoann and Romary, Laurent and de la Clergerie, {\'E}ric Villemonte and Seddah, Djam{\'e} and Sagot, Beno{\^i}t},
    booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
    year={2020}
  }
  ```
  [Paper](https://arxiv.org/abs/1911.03894) | [HuggingFace Model](https://huggingface.co/camembert/camembert-base)

- **Mistral 7B**: High-performance open-source language model
  ```bibtex
  @misc{mistral2023mistral,
    title={Mistral 7B},
    author={Mistral AI},
    year={2023},
    howpublished={\url{https://mistral.ai/news/announcing-mistral-7b/}},
  }
  ```
  [Blog Post](https://mistral.ai/news/announcing-mistral-7b/)

### Frameworks and Tools

- **DSPy**: Programmatic control over language models
  ```bibtex
  @misc{khattab2023dspy,
    title={DSPy: Programming with Foundation Models},
    author={Omar Khattab and Arnav Singhvi and Paridhi Maheshwari and Christopher Potts and Matei Zaharia and Christopher R{\'e}},
    year={2023},
    eprint={2310.03714},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
  }
  ```
  [Paper](https://arxiv.org/abs/2310.03714) | [GitHub](https://github.com/stanfordnlp/dspy)

- **Ollama**: Local LLM serving and management
  [GitHub](https://github.com/ollama/ollama)

- **BM25**: Probabilistic relevance framework
  ```bibtex
  @article{robertson2009probabilistic,
    title={The probabilistic relevance framework: BM25 and beyond},
    author={Robertson, Stephen and Zaragoza, Hugo},
    journal={Foundations and Trends in Information Retrieval},
    year={2009}
  }
  ```
  [Paper](https://dl.acm.org/doi/10.1561/1500000019) 