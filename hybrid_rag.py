#!/usr/bin/env python3
"""
Hybrid Local DSPy RAG with Ollama
A French RAG system combining CamemBERT and BM25 for retrieval with Mistral for generation.
"""

import os
import time
import argparse
from typing import List, Tuple, Dict, Optional

import torch
import nltk
import dspy
import ollama
from transformers import CamembertTokenizer, CamembertModel
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

# Download required NLTK data
nltk.download('punkt', quiet=True)

class HybridRetriever:
    def __init__(self, device: Optional[str] = None):
        """Initialize the hybrid retrieval system with CamemBERT and BM25."""
        # Set up device
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize CamemBERT
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base")
        self.model = CamembertModel.from_pretrained("camembert/camembert-base")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Initialize storage
        self.documents: List[str] = []
        self.colbert_index = []
        self.bm25 = None
        
    def encode(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text using CamemBERT."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.squeeze(0), inputs["attention_mask"].squeeze(0)
    
    def index_documents(self, documents: List[str]):
        """Index documents for both CamemBERT and BM25."""
        self.documents = documents
        
        # CamemBERT indexing
        self.colbert_index = [(doc, *self.encode(doc)) for doc in documents]
        
        # BM25 indexing
        tokenized_corpus = [word_tokenize(doc.lower()) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def colbert_score(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Get top-k documents using CamemBERT similarity."""
        q_embed, q_mask = self.encode(query)
        scores = []
        for doc, d_embed, d_mask in self.colbert_index:
            sim = torch.einsum('id,jd->ij', q_embed, d_embed)
            maxsim = sim.max(dim=1).values
            score = maxsim.mean().item()
            scores.append((doc, score))
        return sorted(scores, key=lambda x: -x[1])[:k]
    
    def bm25_score(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Get top-k documents using BM25 scoring."""
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:k]
        return [(self.documents[i], scores[i]) for i, _ in ranked]
    
    def hybrid_score(self, query: str, alpha: float = 0.5, k: int = 3) -> List[Tuple[str, float]]:
        """Get top-k documents using hybrid scoring (CamemBERT + BM25)."""
        colbert_results = dict(self.colbert_score(query, k=10))
        bm25_results = dict(self.bm25_score(query, k=10))
        
        combined = {}
        for doc in set(colbert_results) | set(bm25_results):
            c = colbert_results.get(doc, 0)
            b = bm25_results.get(doc, 0)
            combined[doc] = alpha * c + (1 - alpha) * b
        return sorted(combined.items(), key=lambda x: -x[1])[:k]

class MistralOllamaLM(dspy.LM):
    """DSPy language model wrapper for Mistral via Ollama."""
    def __init__(self, max_retries: int = 3, timeout: int = 30):
        super().__init__(model='mistral')
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = ollama.Client()

    def __call__(self, prompt: str, **kwargs) -> str:
        """Generate text using Mistral."""
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

class HybridRAG:
    """Main RAG system combining retrieval and generation."""
    def __init__(self, alpha: float = 0.6, k: int = 3):
        self.retriever = HybridRetriever()
        self.generator = MistralOllamaLM()
        self.alpha = alpha
        self.k = k
        
    def index_documents(self, documents: List[str]):
        """Index documents for retrieval."""
        self.retriever.index_documents(documents)
        
    def answer(self, query: str) -> Tuple[str, List[Tuple[str, float]]]:
        """Answer a query using RAG."""
        # Retrieve relevant documents
        retrieved_docs = self.retriever.hybrid_score(query, alpha=self.alpha, k=self.k)
        
        # Construct prompt
        context = "\n".join(f"- {doc}" for doc, _ in retrieved_docs)
        prompt = f"""En utilisant uniquement les informations fournies ci-dessous, réponds à la question en français.
        Si tu ne peux pas répondre avec ces informations, dis-le honnêtement.

        Contexte:
        {context}

        Question: {query}

        Réponse:"""
        
        # Generate answer
        answer = self.generator(prompt)
        return answer, retrieved_docs

def main():
    """Example usage of the HybridRAG system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hybrid RAG system for French text')
    parser.add_argument('--alpha', type=float, default=0.6,
                      help='Weight for CamemBERT vs BM25 (default: 0.6)')
    parser.add_argument('--k', type=int, default=3,
                      help='Number of documents to retrieve (default: 3)')
    parser.add_argument('--query', type=str,
                      default="Quel est l'impact du climat sur la nature ?",
                      help='Query to answer (default: climate impact question)')
    args = parser.parse_args()
    
    # Sample documents (same as in notebook)
    documents = [
        "Le changement climatique menace la biodiversité mondiale et accélère l'extinction des espèces.",
        "Les océans jouent un rôle crucial dans la régulation du climat et l'absorption du CO2.",
        "La fonte des glaciers arctiques modifie les courants océaniques et affecte le climat global.",
        # Add more documents here...
    ]
    
    # Initialize and index
    rag = HybridRAG(alpha=args.alpha, k=args.k)
    rag.index_documents(documents)
    
    # Process query
    answer, sources = rag.answer(args.query)
    
    # Print results
    print(f"\nQuestion: {args.query}")
    print(f"\nRéponse: {answer}")
    print(f"\nSources utilisées (alpha={args.alpha}):")
    for doc, score in sources:
        print(f"{score:.4f} - {doc}")

if __name__ == "__main__":
    main() 