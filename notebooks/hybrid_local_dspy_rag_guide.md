# Hybrid Local DSPy RAG with Ollama Guide

A comprehensive guide for building a French RAG (Retrieval-Augmented Generation) system using hybrid retrieval methods.

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Implementation Details](#implementation-details)
4. [Advanced Topics](#advanced-topics)
5. [Troubleshooting](#troubleshooting)

## Introduction

This guide explains how to build a hybrid retrieval system for French text that combines:

- CamemBERT for semantic search (understanding meaning)
- BM25 for lexical search (exact matching)
- DSPy for pipeline orchestration
- Mistral for language generation

## System Architecture

### 1. Hybrid Retrieval System

The system combines two complementary approaches:

1. **Semantic Search (CamemBERT)**
   - Uses contextual embeddings to understand meaning
   - Better for conceptual matches
   - Handles synonyms and related concepts

2. **Lexical Search (BM25)**
   - Uses term frequency for matching
   - Excellent for exact matches
   - Faster processing time

### 2. DSPy Integration

DSPy provides the framework to:
- Combine retrieval methods
- Manage the RAG pipeline
- Handle prompt engineering
- Integrate with Ollama

## Implementation Details

### 1. Setting Up the Models

```python
# Initialize CamemBERT
from transformers import CamembertTokenizer, CamembertModel

tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base")
model = CamembertModel.from_pretrained("camembert/camembert-base")
model = model.to(device)
```

### 2. Ollama Integration

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
                    raise Exception(f"Failed after {self.max_retries} attempts: {str(e)}")
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(1)
```

## Advanced Topics

### 1. Hardware Acceleration

For optimal performance on Mac:

```python
# Enable Metal support
import torch
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
```

### 2. Performance Optimization

Key considerations:
- Batch processing for embeddings
- Caching of embeddings
- Proper chunking of documents
- Index optimization

## Troubleshooting

### Common Issues

1. **Memory Management**
   - Monitor RAM usage
   - Adjust batch sizes
   - Use smaller chunks if needed

2. **Model Performance**
   - Check device utilization
   - Monitor inference times
   - Optimize chunk sizes

3. **Quality Issues**
   - Balance retrieval weights
   - Tune chunk overlap
   - Adjust prompt templates 