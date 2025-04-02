from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, ndcg_score
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
import torch
from transformers import CamembertTokenizer, CamembertModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Download required NLTK data
nltk.download('stopwords')
french_stopwords = set(stopwords.words('french'))

@dataclass
class TestCase:
    query: str
    relevant_docs: List[str]
    category: str

# Sample test cases with ground truth
TEST_CASES = [
    TestCase(
        query="Quel est l'impact du changement climatique sur les océans ?",
        relevant_docs=[
            "Les océans jouent un rôle crucial dans la régulation du climat et l'absorption du CO2.",
            "La fonte des glaciers arctiques modifie les courants océaniques et affecte le climat global.",
        ],
        category="climate"
    ),
    TestCase(
        query="Comment l'intelligence artificielle transforme-t-elle la médecine ?",
        relevant_docs=[
            "Le machine learning révolutionne la recherche médicale et le diagnostic des maladies.",
            "La recherche en biotechnologie permet des avancées majeures en médecine personnalisée.",
        ],
        category="ai_health"
    ),
    TestCase(
        query="Quelles sont les traditions culturelles françaises importantes ?",
        relevant_docs=[
            "La gastronomie française est inscrite au patrimoine mondial de l'UNESCO.",
            "Le cinéma français est reconnu pour sa créativité et sa diversité artistique.",
            "Les traditions viticoles françaises se transmettent de génération en génération.",
        ],
        category="culture"
    ),
    TestCase(
        query="Comment protéger la biodiversité marine ?",
        relevant_docs=[
            "Les récifs coralliens abritent 25% de la vie marine mondiale.",
            "Les océans jouent un rôle crucial dans la régulation du climat et l'absorption du CO2.",
            "La protection des abeilles est cruciale pour la préservation de la biodiversité.",
        ],
        category="biodiversity"
    ),
    TestCase(
        query="Quelles sont les avancées en énergies renouvelables ?",
        relevant_docs=[
            "Les énergies renouvelables sont essentielles pour lutter contre le réchauffement climatique.",
            "L'énergie solaire devient de plus en plus accessible et efficace pour les particuliers.",
            "Les éoliennes offshore représentent une source importante d'énergie propre en Europe.",
        ],
        category="energy"
    ),
    TestCase(
        query="Comment l'éducation évolue avec la technologie ?",
        relevant_docs=[
            "L'éducation numérique transforme les méthodes d'apprentissage traditionnelles.",
            "Les nouvelles technologies facilitent l'accès à l'éducation pour tous.",
            "La formation continue devient essentielle dans un monde en constante évolution.",
        ],
        category="education"
    )
]

class RetrievalEvaluator:
    def __init__(self, documents: List[str], device: str = "mps"):
        self.documents = documents
        self.device = device
        self.setup_models()
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text by removing stopwords, punctuation, and normalizing."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove stopwords
        words = word_tokenize(text)
        words = [w for w in words if w not in french_stopwords]
        
        return ' '.join(words)
    
    def setup_models(self):
        # Setup CamemBERT
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base")
        self.model = CamembertModel.from_pretrained("camembert/camembert-base")
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Preprocess documents
        self.processed_docs = [self.preprocess_text(doc) for doc in self.documents]
        
        # Setup BM25
        self.tokenized_corpus = [word_tokenize(doc) for doc in self.processed_docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Precompute document embeddings
        self.doc_embeddings = self._compute_document_embeddings()
        
    def _compute_document_embeddings(self) -> torch.Tensor:
        embeddings = []
        with torch.no_grad():
            for doc in self.processed_docs:
                # Use sliding window for long documents
                tokens = self.tokenizer(
                    doc,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                outputs = self.model(**tokens)
                
                # Use attention-weighted mean pooling
                attention_mask = tokens['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings.append(embedding.squeeze(0))
        return torch.stack(embeddings)
    
    def semantic_search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        # Preprocess query
        processed_query = self.preprocess_text(query)
        
        with torch.no_grad():
            # Encode query
            tokens = self.tokenizer(
                processed_query,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            outputs = self.model(**tokens)
            
            # Use attention-weighted mean pooling for query
            attention_mask = tokens['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            query_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            query_embedding = query_embedding.squeeze(0)
            
            # Normalize embeddings
            query_embedding = query_embedding / query_embedding.norm()
            doc_embeddings = self.doc_embeddings / self.doc_embeddings.norm(dim=1, keepdim=True)
            
            # Compute cosine similarity
            similarities = torch.matmul(query_embedding.unsqueeze(0), doc_embeddings.T).squeeze(0)
            
            # Get top k results
            top_k_values, top_k_indices = torch.topk(similarities, k)
            return [(self.documents[idx], score.item()) for idx, score in zip(top_k_indices, top_k_values)]
    
    def keyword_search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        # Preprocess query
        processed_query = self.preprocess_text(query)
        tokenized_query = word_tokenize(processed_query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[-k:][::-1]
        return [(self.documents[i], scores[i]) for i in top_k_indices]
    
    def expand_query(self, query: str) -> str:
        """Expand query using semantic similarity to add relevant terms."""
        # Preprocess query
        processed_query = self.preprocess_text(query)
        
        # Get query embedding
        with torch.no_grad():
            tokens = self.tokenizer(
                processed_query,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            outputs = self.model(**tokens)
            query_embedding = outputs.last_hidden_state.mean(dim=1)
            
            # Find most similar documents
            similarities = torch.nn.functional.cosine_similarity(
                query_embedding, self.doc_embeddings
            )
            top_k = 2  # Use top 2 documents for expansion
            _, top_indices = torch.topk(similarities, top_k)
            
            # Extract key terms from similar documents
            expansion_docs = [self.processed_docs[idx] for idx in top_indices]
            expansion_terms = []
            for doc in expansion_docs:
                terms = word_tokenize(doc)
                expansion_terms.extend(terms)
            
            # Add unique terms to query
            query_terms = word_tokenize(processed_query)
            expanded_terms = list(set(query_terms + expansion_terms))
            expanded_query = " ".join(expanded_terms)
            
            return expanded_query
    
    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores using softmax."""
        if len(scores) == 0:
            return scores
        scores = np.array(scores)
        scores = scores - scores.min()
        scores = np.exp(scores) / np.sum(np.exp(scores))
        return scores
    
    def hybrid_search(self, query: str, alpha: float = 0.5, k: int = 3) -> List[Tuple[str, float]]:
        # Expand query for both semantic and keyword search
        expanded_query = self.expand_query(query)
        
        # Get results from both methods
        semantic_results = dict(self.semantic_search(query, k=k*2))
        keyword_results = dict(self.keyword_search(expanded_query, k=k*2))
        
        # Normalize scores using softmax
        semantic_scores = self.normalize_scores(np.array(list(semantic_results.values())))
        keyword_scores = self.normalize_scores(np.array(list(keyword_results.values())))
        
        # Update scores in results
        semantic_results = dict(zip(semantic_results.keys(), semantic_scores))
        keyword_results = dict(zip(keyword_results.keys(), keyword_scores))
        
        # Combine scores with dynamic weighting
        combined_scores = {}
        for doc in set(semantic_results) | set(keyword_results):
            semantic_score = semantic_results.get(doc, 0)
            keyword_score = keyword_results.get(doc, 0)
            
            # Use confidence-based weighting
            semantic_conf = max(semantic_results.values()) if semantic_results else 0
            keyword_conf = max(keyword_results.values()) if keyword_results else 0
            
            # Adjust alpha based on confidence
            dynamic_alpha = alpha * (semantic_conf / (semantic_conf + keyword_conf + 1e-10))
            
            combined_scores[doc] = dynamic_alpha * semantic_score + (1 - dynamic_alpha) * keyword_score
            
        # Sort and return top k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]
    
    def evaluate_retrieval(self, test_cases: List[TestCase], alpha: float = 0.5) -> Dict[str, float]:
        metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'ndcg': []
        }
        
        for test_case in test_cases:
            # Get retrieval results
            results = self.hybrid_search(test_case.query, alpha=alpha)
            retrieved_docs = [doc for doc, _ in results]
            
            # Create binary relevance scores for all documents
            all_docs = list(set(retrieved_docs + test_case.relevant_docs))
            y_true = [1 if doc in test_case.relevant_docs else 0 for doc in all_docs]
            y_pred = [1 if doc in retrieved_docs else 0 for doc in all_docs]
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            
            # Calculate NDCG
            ideal_dcg = np.sum([1 / np.log2(i + 2) for i in range(len(test_case.relevant_docs))])
            dcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(y_pred)])
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
            
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1'].append(f1)
            metrics['ndcg'].append(ndcg)
        
        # Average metrics across all test cases
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def optimize_alpha(self, test_cases: List[TestCase], alpha_range: List[float] = None) -> Tuple[float, Dict[str, float]]:
        if alpha_range is None:
            alpha_range = np.linspace(0, 1, 11)
        
        best_f1 = 0
        best_alpha = 0.5
        best_metrics = {}
        
        for alpha in alpha_range:
            metrics = self.evaluate_retrieval(test_cases, alpha=alpha)
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_alpha = alpha
                best_metrics = metrics
                
        return best_alpha, best_metrics 