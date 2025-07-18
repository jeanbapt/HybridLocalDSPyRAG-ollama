"""
ModernBERT French Implementation Guide
=====================================

This module provides practical examples for experimenting with ModernBERT
for French NLP tasks, including fine-tuning strategies and hybrid approaches.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    CamembertTokenizer, 
    CamembertModel,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import numpy as np
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModernBERTFrenchAdapter:
    """
    Adapter class for using ModernBERT with French text.
    Provides various strategies for leveraging ModernBERT's advantages.
    """
    
    def __init__(self, strategy: str = "fine-tune"):
        """
        Initialize the adapter with a specific strategy.
        
        Args:
            strategy: One of ["fine-tune", "zero-shot", "hybrid"]
        """
        self.strategy = strategy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_modernbert_for_finetuning(self):
        """Load ModernBERT model for fine-tuning on French data."""
        logger.info("Loading ModernBERT for fine-tuning...")
        
        # Load English ModernBERT
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.model = AutoModelForMaskedLM.from_pretrained(
            "answerdotai/ModernBERT-base",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        return self.model, self.tokenizer
    
    def prepare_french_data(self, texts: List[str], max_length: int = 8192):
        """
        Prepare French text data for ModernBERT processing.
        
        Args:
            texts: List of French text strings
            max_length: Maximum sequence length (up to 8192 for ModernBERT)
        """
        # Tokenize with ModernBERT tokenizer
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        logger.info(f"Tokenized {len(texts)} texts with max length {max_length}")
        return encoded
    
    def create_mlm_dataset(self, texts: List[str], mlm_probability: float = 0.30):
        """
        Create masked language modeling dataset for French fine-tuning.
        ModernBERT uses 30% masking rate instead of BERT's 15%.
        """
        def mask_tokens(inputs):
            labels = inputs["input_ids"].clone()
            
            # Create probability matrix for masking
            probability_matrix = torch.full(labels.shape, mlm_probability)
            
            # Don't mask special tokens
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
                for val in labels.tolist()
            ]
            probability_matrix.masked_fill_(
                torch.tensor(special_tokens_mask, dtype=torch.bool), 
                value=0.0
            )
            
            # Mask tokens
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # Only compute loss on masked tokens
            
            # Replace masked tokens
            inputs["input_ids"][masked_indices] = self.tokenizer.mask_token_id
            inputs["labels"] = labels
            
            return inputs
        
        # Tokenize texts
        encoded = self.prepare_french_data(texts)
        
        # Apply masking
        dataset = []
        for i in range(len(texts)):
            item = {
                "input_ids": encoded["input_ids"][i],
                "attention_mask": encoded["attention_mask"][i]
            }
            masked_item = mask_tokens(item)
            dataset.append(masked_item)
        
        return Dataset.from_list(dataset)


class HybridFrenchEncoder:
    """
    Hybrid approach combining CamemBERT's French expertise 
    with ModernBERT's architectural advantages.
    """
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load both models
        self.camembert_tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base")
        self.camembert_model = CamembertModel.from_pretrained("camembert/camembert-base").to(self.device)
        
        self.modernbert_tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.modernbert_model = AutoModel.from_pretrained("answerdotai/ModernBERT-base").to(self.device)
        
    def encode_short_french(self, text: str):
        """Use CamemBERT for short French texts (< 512 tokens)."""
        inputs = self.camembert_tokenizer(
            text, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.camembert_model(**inputs)
        
        return outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    
    def encode_long_mixed(self, text: str):
        """
        Use ModernBERT for long or code-mixed French texts.
        Useful for technical documentation or code + French.
        """
        inputs = self.modernbert_tokenizer(
            text, 
            return_tensors="pt", 
            max_length=8192, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.modernbert_model(**inputs)
        
        return outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    
    def smart_encode(self, text: str):
        """
        Intelligently choose encoder based on text characteristics.
        """
        # Simple heuristic: use token count and code detection
        token_count = len(self.camembert_tokenizer.tokenize(text))
        has_code = any(marker in text for marker in ["```", "def ", "class ", "import "])
        
        if token_count > 400 or has_code:
            logger.info("Using ModernBERT for long/technical text")
            return self.encode_long_mixed(text)
        else:
            logger.info("Using CamemBERT for short French text")
            return self.encode_short_french(text)


class ModernBERTFrenchFineTuner:
    """
    Fine-tuning ModernBERT for French NLP tasks.
    """
    
    def __init__(self, task: str = "classification"):
        self.task = task
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def prepare_for_classification(self, num_labels: int = 2):
        """Prepare ModernBERT for French text classification."""
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "answerdotai/ModernBERT-base",
            num_labels=num_labels,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        
        return self.model, self.tokenizer
    
    def create_training_args(self, output_dir: str = "./modernbert-french"):
        """Create optimized training arguments for ModernBERT."""
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,  # Adjust based on GPU memory
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=1000,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=torch.cuda.is_available(),  # Use mixed precision if available
            gradient_checkpointing=True,  # Save memory for long sequences
            dataloader_num_workers=4,
        )


def compare_models_example():
    """
    Example comparing CamemBERT and ModernBERT on French text.
    """
    # Sample French texts of different lengths
    short_text = "Paris est la capitale de la France."
    long_text = """
    L'intelligence artificielle transforme notre société. Les modèles de langage 
    comme BERT ont révolutionné le traitement du langage naturel. Ces modèles 
    peuvent comprendre le contexte et les nuances du langage humain...
    """ * 50  # Repeat to make it long
    
    code_mixed_text = """
    Voici un exemple de code Python:
    ```python
    def calculer_moyenne(nombres: List[float]) -> float:
        '''Calcule la moyenne d'une liste de nombres.'''
        return sum(nombres) / len(nombres)
    ```
    Cette fonction est très utile pour l'analyse statistique.
    """
    
    # Initialize hybrid encoder
    hybrid = HybridFrenchEncoder()
    
    # Test encodings
    print("Testing short French text...")
    short_encoding = hybrid.smart_encode(short_text)
    print(f"Encoding shape: {short_encoding.shape}")
    
    print("\nTesting long French text...")
    long_encoding = hybrid.smart_encode(long_text)
    print(f"Encoding shape: {long_encoding.shape}")
    
    print("\nTesting code-mixed French text...")
    code_encoding = hybrid.smart_encode(code_mixed_text)
    print(f"Encoding shape: {code_encoding.shape}")


def fine_tuning_example():
    """
    Example of fine-tuning ModernBERT on French data.
    """
    # Initialize fine-tuner
    fine_tuner = ModernBERTFrenchFineTuner(task="classification")
    model, tokenizer = fine_tuner.prepare_for_classification(num_labels=3)
    
    # Example French classification data
    french_texts = [
        "Ce film est absolument magnifique!",
        "Je n'ai pas aimé ce livre.",
        "Le service était correct, sans plus.",
    ]
    labels = [2, 0, 1]  # Positive, Negative, Neutral
    
    # Prepare dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True,
            max_length=512
        )
    
    # Create dataset
    dataset = Dataset.from_dict({
        "text": french_texts,
        "labels": labels
    })
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Create trainer
    training_args = fine_tuner.create_training_args()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,  # In practice, use separate validation set
        tokenizer=tokenizer,
    )
    
    # Fine-tune (commented out to avoid actual training in example)
    # trainer.train()
    
    print("Fine-tuning setup complete!")


def migration_path_example():
    """
    Example showing migration path from CamemBERT to ModernBERT.
    """
    class FrenchTextProcessor:
        def __init__(self, use_modernbert: bool = False):
            self.use_modernbert = use_modernbert
            
            if use_modernbert:
                # Future: Use French ModernBERT when available
                self.model_name = "answerdotai/ModernBERT-base-fr"  # Hypothetical
                self.max_length = 8192
            else:
                # Current: Use CamemBERT
                self.model_name = "camembert/camembert-base"
                self.max_length = 512
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
        
        def process(self, text: str):
            """Process French text with appropriate model."""
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            return outputs.last_hidden_state
    
    # Example usage
    processor_current = FrenchTextProcessor(use_modernbert=False)
    # processor_future = FrenchTextProcessor(use_modernbert=True)
    
    text = "Voici un exemple de texte en français."
    result = processor_current.process(text)
    print(f"Processed with {processor_current.model_name}: {result.shape}")


if __name__ == "__main__":
    print("ModernBERT French Implementation Examples")
    print("=" * 50)
    
    print("\n1. Model Comparison Example:")
    compare_models_example()
    
    print("\n2. Fine-tuning Setup Example:")
    fine_tuning_example()
    
    print("\n3. Migration Path Example:")
    migration_path_example()
    
    print("\nNote: ModernBERT-fr is hypothetical. Monitor HuggingFace for updates!")