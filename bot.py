import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob
import warnings
import sys
import datetime
import random
import time
import traceback
import requests
from io import BytesIO
from zipfile import ZipFile
from collections import deque
from typing import List, Dict, Any, Tuple, Optional, Union
import pickle

# Suppress warnings
warnings.filterwarnings("ignore")

# Check and install dependencies
def check_and_install_dependencies():
    dependencies = {
        "transformers": "transformers",
        "tiktoken": "tiktoken",
        "google.protobuf": "protobuf",
        "sentencepiece": "sentencepiece",
        "torch": "torch",
        "numpy": "numpy",
        "pandas": "pandas",
        "tqdm": "tqdm"
    }
    
    missing_deps = []
    for module, package in dependencies.items():
        try:
            __import__(module)
        except ImportError:
            missing_deps.append(package)
    
    if missing_deps:
        print(f"Installing missing dependencies: {', '.join(missing_deps)}")
        import subprocess
        for dep in missing_deps:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        print("Dependencies installed successfully.")

# Install dependencies
check_and_install_dependencies()

# Import necessary modules
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel

class CustomDataset(Dataset):
    """Custom dataset for training the neural network with your data"""
    
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int = 1024):
        """
        Initialize the dataset
        
        Args:
            data_path: Path to data directory or file
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum token length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.load_data(data_path)
        
    def load_data(self, data_path: str) -> None:
        """
        Load data from file or directory
        
        Args:
            data_path: Path to data directory or file
        """
        if os.path.isdir(data_path):
            # Load all json files from directory
            files = glob.glob(os.path.join(data_path, "*.json"))
            for file in files:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.data.extend(self.process_data(data))
        elif os.path.isfile(data_path):
            # Load single file
            _, ext = os.path.splitext(data_path)
            if ext == '.json':
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.data = self.process_data(data)
            elif ext in ['.csv', '.tsv']:
                df = pd.read_csv(data_path, sep=',' if ext == '.csv' else '\t')
                self.data = self.process_dataframe(df)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")
    
    def process_data(self, data: Union[List, Dict]) -> List[Dict]:
        """
        Process raw data into training examples
        
        Args:
            data: Raw data from files
            
        Returns:
            List of processed training examples
        """
        processed = []
        
        # Handle different data formats
        if isinstance(data, list):
            for item in data:
                if "input" in item and "output" in item:
                    processed.append({
                        "input": item["input"],
                        "output": item["output"]
                    })
                elif "question" in item and "answer" in item:
                    processed.append({
                        "input": item["question"],
                        "output": item["answer"]
                    })
                elif "context" in item and "response" in item:
                    processed.append({
                        "input": item["context"],
                        "output": item["response"]
                    })
        elif isinstance(data, dict):
            # Extract conversation pairs
            if "conversations" in data:
                for conv in data["conversations"]:
                    if isinstance(conv, list) and len(conv) > 1:
                        for i in range(0, len(conv) - 1, 2):
                            if i + 1 < len(conv):
                                processed.append({
                                    "input": conv[i]["content"] if isinstance(conv[i], dict) else conv[i],
                                    "output": conv[i+1]["content"] if isinstance(conv[i+1], dict) else conv[i+1]
                                })
        
        return processed
    
    def process_dataframe(self, df: pd.DataFrame) -> List[Dict]:
        """
        Process dataframe into training examples
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            List of processed training examples
        """
        processed = []
        
        # Try to infer input/output column names
        input_col = None
        output_col = None
        
        # Common column name patterns
        input_patterns = ["input", "question", "prompt", "context", "source", "request"]
        output_patterns = ["output", "answer", "response", "target", "result", "reply"]
        
        for col in df.columns:
            col_lower = col.lower()
            for pattern in input_patterns:
                if pattern in col_lower:
                    input_col = col
                    break
            for pattern in output_patterns:
                if pattern in col_lower:
                    output_col = col
                    break
        
        # If we couldn't find the columns, use the first two
        if input_col is None or output_col is None:
            if len(df.columns) >= 2:
                input_col = df.columns[0]
                output_col = df.columns[1]
                print(f"Using columns '{input_col}' and '{output_col}' as input and output")
            else:
                raise ValueError("DataFrame should have at least 2 columns")
        
        # Create processed examples
        for _, row in df.iterrows():
            processed.append({
                "input": str(row[input_col]),
                "output": str(row[output_col])
            })
        
        return processed
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get the training example at index idx"""
        example = self.data[idx]
        
        # Tokenize input
        input_encoding = self.tokenizer(
            example["input"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize output
        output_encoding = self.tokenizer(
            example["output"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "input_attention_mask": input_encoding["attention_mask"].squeeze(),
            "output_ids": output_encoding["input_ids"].squeeze(),
            "output_attention_mask": output_encoding["attention_mask"].squeeze()
        }

class EncoderLayer(nn.Module):
    """Transformer encoder layer with multi-head attention and feed-forward network"""
    
    def __init__(self, d_model: int = 768, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        """
        Initialize the encoder layer
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of feed-forward network
            dropout: Dropout probability
        """
        super(EncoderLayer, self).__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the encoder layer
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection and layer normalization
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=attention_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class EnhancedLatentManipulator(nn.Module):
    """Enhanced latent manipulator with multi-layer transformer architecture"""
    
    def __init__(
        self, 
        embedding_dim: int = 768,
        n_layers: int = 4,
        n_heads: int = 8,
        ff_dim: int = 2048,
        dropout: float = 0.1
    ):
        """
        Initialize the enhanced latent manipulator
        
        Args:
            embedding_dim: Dimension of input embeddings
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            ff_dim: Feed-forward network dimension
            dropout: Dropout probability
        """
        super(EnhancedLatentManipulator, self).__init__()
        
        # Input normalization
        self.input_norm = nn.LayerNorm(embedding_dim)
        
        # Encoder
        self.encoder = nn.Linear(embedding_dim, embedding_dim)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            EncoderLayer(embedding_dim, n_heads, ff_dim, dropout) 
            for _ in range(n_layers)
        ])
        
        # Decoder
        self.decoder = nn.Linear(embedding_dim, embedding_dim)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the manipulator
        
        Args:
            x: Input tensor [batch_size, embedding_dim]
            
        Returns:
            Output tensor [batch_size, embedding_dim]
        """
        # Normalize input
        x = self.input_norm(x)
        
        # Encode
        x = self.encoder(x)
        
        # Add batch and sequence dimensions if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Squeeze sequence dimension if it was added
        if x.size(1) == 1:
            x = x.squeeze(1)
        
        # Decode
        x = self.decoder(x)
        
        # Normalize output
        x = self.output_norm(x)
        
        return x

class CrossAttention(nn.Module):
    """Cross-attention mechanism for attending to memory"""
    
    def __init__(self, embedding_dim: int = 768, nhead: int = 8, dropout: float = 0.1):
        """
        Initialize cross-attention module
        
        Args:
            embedding_dim: Dimension of embeddings
            nhead: Number of attention heads
            dropout: Dropout probability
        """
        super(CrossAttention, self).__init__()
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through cross-attention
        
        Args:
            query: Query tensor [batch_size, query_len, embedding_dim]
            key_value: Key-value tensor [batch_size, kv_len, embedding_dim]
            key_padding_mask: Key padding mask [batch_size, kv_len]
            
        Returns:
            Output tensor [batch_size, query_len, embedding_dim]
        """
        attn_output, _ = self.multihead_attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask
        )
        
        return self.norm(query + self.dropout(attn_output))

class MemoryBank(nn.Module):
    """Memory bank for storing and retrieving context"""
    
    def __init__(
        self, 
        embedding_dim: int = 768,
        memory_size: int = 100,
        temperature: float = 0.1
    ):
        """
        Initialize memory bank
        
        Args:
            embedding_dim: Dimension of embeddings
            memory_size: Maximum number of items in memory
            temperature: Temperature for softmax
        """
        super(MemoryBank, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.temperature = temperature
        
        # Initialize memory
        self.memory = nn.Parameter(torch.zeros(memory_size, embedding_dim))
        self.memory_mask = torch.zeros(memory_size, dtype=torch.bool)
        self.memory_counter = 0
        
        # Cross-attention for memory retrieval
        self.cross_attention = CrossAttention(embedding_dim)
    
    def add_to_memory(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> None:
        """
        Add embeddings to memory
        
        Args:
            embeddings: Embeddings to add [batch_size, embedding_dim]
            mask: Mask for embeddings [batch_size]
        """
        batch_size = embeddings.size(0)
        
        # Get memory indices to update
        indices = torch.arange(
            self.memory_counter,
            self.memory_counter + batch_size
        ) % self.memory_size
        
        # Update memory
        self.memory.data[indices] = embeddings.detach()
        
        # Update mask
        if mask is not None:
            self.memory_mask[indices] = mask
        else:
            self.memory_mask[indices] = True
        
        # Update counter
        self.memory_counter = (self.memory_counter + batch_size) % self.memory_size
    
    def query_memory(self, query: torch.Tensor) -> torch.Tensor:
        """
        Query memory with attention
        
        Args:
            query: Query tensor [batch_size, embedding_dim]
            
        Returns:
            Retrieved memory context [batch_size, embedding_dim]
        """
        # Add sequence dimension if needed
        if query.dim() == 2:
            query = query.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        
        # Get active memory
        active_mask = self.memory_mask
        active_memory = self.memory[active_mask]
        
        if len(active_memory) == 0:
            # No active memory, return query
            return query.squeeze(1) if query.size(1) == 1 else query
        
        # Add batch dimension to memory
        memory_batch = active_memory.unsqueeze(0).expand(query.size(0), -1, -1)
        
        # Query memory with cross-attention
        context = self.cross_attention(query, memory_batch)
        
        # Remove sequence dimension if it was added
        if query.size(1) == 1:
            context = context.squeeze(1)
        
        return context

class EnhancedAutoencoder:
    """Enhanced autoencoder using pretrained language model for embedding"""
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased",
        device: str = "cpu",
        max_length: int = 1400,
        pooling_strategy: str = "cls"
    ):
        """
        Initialize enhanced autoencoder
        
        Args:
            model_name: Name of pretrained model
            device: Device to use (cpu, cuda, mps)
            max_length: Maximum token length
            pooling_strategy: Pooling strategy (cls, mean)
        """
        self.device = device
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy
        
        print(f"Using device: {self.device}")
        print(f"Loading model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.hidden_size
        print(f"Model loaded with embedding dimension: {self.embedding_dim}")
    
    @torch.no_grad()
    def embed(self, text: str) -> torch.FloatTensor:
        """
        Embed text using pretrained model
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding tensor [1, embedding_dim]
        """
        # Check if text needs to be chunked
        if len(self.tokenizer.encode(text)) > self.max_length:
            chunks = self._chunk_text(text)
            embeddings = []
            
            for chunk in chunks:
                inputs = self.tokenizer(
                    chunk, 
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length'
                ).to(self.device)
                
                outputs = self.model(**inputs)
                
                # Apply pooling strategy
                if self.pooling_strategy == "cls":
                    embedding = outputs.last_hidden_state[:, 0, :]
                elif self.pooling_strategy == "mean":
                    # Mean pooling with attention mask
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    embedding = torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.sum(mask, dim=1)
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
                
                embeddings.append(embedding)
            
            # Average embeddings from all chunks
            return torch.mean(torch.stack(embeddings, dim=0), dim=0)
        else:
            # Process single chunk
            inputs = self.tokenizer(
                text, 
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length,
                padding='max_length'
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            # Apply pooling strategy
            if self.pooling_strategy == "cls":
                return outputs.last_hidden_state[:, 0, :]
            elif self.pooling_strategy == "mean":
                # Mean pooling with attention mask
                mask = inputs["attention_mask"].unsqueeze(-1)
                return torch.sum(outputs.last_hidden_state * mask, dim=1) / torch.sum(mask, dim=1)
            else:
                raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk text into smaller pieces
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.max_length):
            chunk_tokens = tokens[i:i + self.max_length]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        
        return chunks
    
    @torch.no_grad()
    def embed_batch(self, texts: List[str]) -> torch.FloatTensor:
        """
        Embed a batch of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Embedding tensor [batch_size, embedding_dim]
        """
        embeddings = []
        
        for text in texts:
            embedding = self.embed(text)
            embeddings.append(embedding)
        
        return torch.cat(embeddings, dim=0)

class EnhancedConversationBuffer:
    """Enhanced conversation buffer with better context management"""
    
    def __init__(
        self, 
        max_tokens: int = 1400,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_name: str = "bert-base-uncased"
    ):
        """
        Initialize conversation buffer
        
        Args:
            max_tokens: Maximum number of tokens to keep
            tokenizer: Tokenizer to use
            model_name: Model name for tokenizer if not provided
        """
        self.max_tokens = max_tokens
        self.buffer = deque()
        self.current_tokens = 0
        
        # Load tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
        
        # Entity memory
        self.entity_memory = {}
        
        # Topic memory
        self.topic_memory = []
    
    def add_message(self, message: Dict[str, str]) -> None:
        """
        Add message to buffer
        
        Args:
            message: Message dictionary with 'role' and 'content'
        """
        message_text = f"{message['role']}: {message['content']}"
        tokens = len(self.tokenizer.encode(message_text))
        
        # Add message to buffer
        self.buffer.append({
            'message': message,
            'tokens': tokens,
            'timestamp': datetime.datetime.now()
        })
        
        self.current_tokens += tokens
        
        # Extract entities and topics
        self._extract_entities(message)
        self._extract_topics(message)
        
        # Remove old messages if buffer is full
        while self.current_tokens > self.max_tokens and len(self.buffer) > 1:
            removed = self.buffer.popleft()
            self.current_tokens -= removed['tokens']
    
    def _extract_entities(self, message: Dict[str, str]) -> None:
        """
        Extract entities from message
        
        Args:
            message: Message dictionary
        """
        content = message['content'].lower()
        
        # Extract user information
        name_patterns = ["my name is", "i am", "call me"]
        for pattern in name_patterns:
            if pattern in content:
                try:
                    name = content.split(pattern)[1].strip().split()[0].capitalize()
                    self.entity_memory['user_name'] = {
                        'value': name,
                        'timestamp': datetime.datetime.now(),
                        'confidence': 0.9
                    }
                except:
                    pass
        
        # Extract preferences
        like_patterns = ["i like", "i love", "i enjoy", "i prefer"]
        for pattern in like_patterns:
            if pattern in content:
                try:
                    preference = content.split(pattern)[1].strip().split('.')[0]
                    if 'preferences' not in self.entity_memory:
                        self.entity_memory['preferences'] = []
                    
                    self.entity_memory['preferences'].append({
                        'value': preference,
                        'timestamp': datetime.datetime.now(),
                        'confidence': 0.8
                    })
                except:
                    pass
    
    def _extract_topics(self, message: Dict[str, str]) -> None:
        """
        Extract topics from message
        
        Args:
            message: Message dictionary
        """
        content = message['content'].lower()
        
        # Simple keyword-based topic extraction
        topics = {
            'python': ['python', 'code', 'programming', 'function', 'class'],
            'ai': ['ai', 'neural', 'machine learning', 'deep learning', 'model'],
            'data': ['data', 'csv', 'dataset', 'database', 'analytics'],
            'help': ['help', 'assist', 'support', 'guide', 'explain']
        }
        
        for topic, keywords in topics.items():
            if any(keyword in content for keyword in keywords):
                # Add topic if not already present
                if topic not in [t['topic'] for t in self.topic_memory]:
                    self.topic_memory.append({
                        'topic': topic,
                        'timestamp': datetime.datetime.now(),
                        'count': 1
                    })
                else:
                    # Update existing topic
                    for t in self.topic_memory:
                        if t['topic'] == topic:
                            t['count'] += 1
                            t['timestamp'] = datetime.datetime.now()
    
    def get_context_string(self) -> str:
        """
        Get context string from buffer
        
        Returns:
            Context string
        """
        return "\n".join([f"{item['message']['role']}: {item['message']['content']}" 
                         for item in self.buffer])
    
    def get_recent_messages(self, n: int = 5) -> List[Dict]:
        """
        Get recent messages
        
        Args:
            n: Number of messages to return
            
        Returns:
            List of recent messages
        """
        return [item['message'] for item in list(self.buffer)[-n:]]
    
    def get_active_topics(self, n: int = 3) -> List[str]:
        """
        Get active topics
        
        Args:
            n: Number of topics to return
            
        Returns:
            List of active topics
        """
        # Sort topics by recency and count
        sorted_topics = sorted(
            self.topic_memory,
            key=lambda x: (x['timestamp'], x['count']),
            reverse=True
        )
        
        return [topic['topic'] for topic in sorted_topics[:n]]

class EnhancedAI:
    """Enhanced AI assistant with multiple neural layers"""
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased",
        device: str = None,
        max_length: int = 1400,
        n_layers: int = 4,
        n_heads: int = 8,
        memory_size: int = 100
    ):
        """
        Initialize enhanced AI
        
        Args:
            model_name: Name of pretrained model
            device: Device to use (cpu, cuda, mps)
            max_length: Maximum token length
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            memory_size: Size of memory bank
        """
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else \
                         'mps' if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize autoencoder
        print(f"Loading autoencoder with model: {model_name}")
        self.autoencoder = EnhancedAutoencoder(
            model_name=model_name,
            device=self.device,
            max_length=max_length,
            pooling_strategy="mean"
        )
        print("Autoencoder loaded successfully")
        
        # Initialize latent manipulator
        print("Initializing latent manipulator")
        self.manipulator = EnhancedLatentManipulator(
            embedding_dim=self.autoencoder.embedding_dim,
            n_layers=n_layers,
            n_heads=n_heads
        ).to(self.device)
        print("Latent manipulator initialized")
        
        # Initialize memory bank
        print("Initializing memory bank")
        self.memory_bank = MemoryBank(
            embedding_dim=self.autoencoder.embedding_dim,
            memory_size=memory_size
        ).to(self.device)
        print("Memory bank initialized")
        
        # Initialize conversation buffer
        self.conversation_buffer = EnhancedConversationBuffer(
            max_tokens=max_length,
            tokenizer=self.autoencoder.tokenizer
        )
        
        # Load knowledge base
        self.load_knowledge_base()
        
        print("Enhanced AI initialized successfully")
    
    def load_knowledge_base(self) -> None:
        """Load knowledge base for the AI"""
        self.knowledge_base = {
            "greetings": [
                "Hello! I'm ready to assist you with your queries.",
                "Hi there! How can I help you today?",
                "Welcome! What can I help you with?",
                "Greetings! How may I assist you today?"
            ],
            "python": {
                "general": "Python is a high-level, interpreted programming language known for its simplicity and versatility. It's widely used in web development, data science, AI, and automation.",
                "functions": "Python functions are defined using the 'def' keyword, followed by the function name and parameters in parentheses. They can include default parameter values, variable arguments, and return statements.",
                "classes": "Python classes are defined using the 'class' keyword and can contain attributes and methods. The 'self' parameter refers to the instance of the class and must be the first parameter of instance methods."
            },
            "ai": {
                "general": "Artificial intelligence (AI) is a field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.",
                "neural_networks": "Neural networks are a type of machine learning model inspired by the human brain. They consist of layers of interconnected nodes (neurons) that process input data and learn patterns through training.",
                "transformers": "Transformer models are a type of neural network architecture that uses self-attention mechanisms to process sequential data. They excel at natural language processing tasks and have revolutionized AI capabilities."
            },
            "identity": "I am an enhanced AI assistant with advanced memory and reasoning capabilities. I'm designed to understand context, remember important information, and provide helpful responses."
        }
    
    def get_response(self, user_input: str) -> str:
        """
        Generate response to user input
        
        Args:
            user_input: User input text
            
        Returns:
            Response text
        """
        # Add user message to conversation buffer
        self.conversation_buffer.add_message({
            "role": "user",
            "content": user_input
        })
        
        # Generate response
        response = self._generate_response(user_input)
        
        # Add assistant message to conversation buffer
        self.conversation_buffer.add_message({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def _generate_response(self, user_input: str) -> str:
        """
        Internal method to generate response
        
        Args:
            user_input: User input text
            
        Returns:
            Generated response
        """
        lower_input = user_input.lower()
        
        # Get user's name if known
        user_name = self.conversation_buffer.entity_memory.get('user_name', {}).get('value', '')
        
        # Handle greetings
        if any(greeting in lower_input for greeting in ["hi", "hello", "hey"]):
            greeting = random.choice(self.knowledge_base["greetings"])
            return f"{greeting}{f' {user_name}' if user_name else ''}"
        
        # Handle name queries
        if "what is my name" in lower_input and user_name:
            return f"Your name is {user_name}, as you mentioned earlier!"
        
        # Handle topic-specific queries
        for topic, content in self.knowledge_base.items():
            if topic in lower_input:
                if isinstance(content, dict):
                    for subtopic, info in content.items():
                        if subtopic in lower_input:
                            return info
                    # Return general info if no specific subtopic matched
                    if "general" in content:
                        return content["general"]
                else:
                    return content
        
        # Handle identity questions
        if any(q in lower_input for q in ["who are you", "what are you", "your name"]):
            return self.knowledge_base["identity"]

        # Generate neural response
        try:
            # Get context
            context = self.conversation_buffer.get_context_string()
            
            # Embed user input
            input_embedding = self.autoencoder.embed(user_input)
            
            # Process embedding through manipulator
            with torch.no_grad():
                processed_embedding = self.manipulator(input_embedding)
            
            # Query memory bank for relevant information
            memory_context = self.memory_bank.query_memory(processed_embedding)
            
            # Generate response template based on similarity
            response_template = self._get_response_template(
                torch.nn.functional.cosine_similarity(
                    processed_embedding, memory_context, dim=1
                ).item()
            )
            
            # Add to memory bank
            self.memory_bank.add_to_memory(processed_embedding)
            
            # Format response
            response = response_template.format(context=context[-100:] if len(context) > 100 else context)
            
            # Add personal touch if we know the user's name
            if user_name and random.random() < 0.3:
                response = f"{user_name}, {response}"
            
            return response

        except Exception as e:
            print(f"Error in neural processing: {str(e)}")
            return f"I'm still learning to process that{f', {user_name}' if user_name else ''}. Could you please elaborate or try a different question?"
    
    def _get_response_template(self, similarity: float) -> str:
        """
        Get response template based on similarity score
        
        Args:
            similarity: Cosine similarity score
            
        Returns:
            Response template string
        """
        if similarity > 0.8:
            templates = [
                "Based on our discussion, I think I understand what you're asking about {context}",
                "I understand your question clearly. {context} is what we've been discussing.",
                "From what we've covered, I can tell you about {context}"
            ]
        elif similarity > 0.5:
            templates = [
                "I think you're asking about {context}, but let me know if I've misunderstood.",
                "From what I understand, you're interested in {context}.",
                "It seems that you want to know about {context}, is that correct?"
            ]
        else:
            templates = [
                "I'm learning more about this topic. Could you tell me more about {context}?",
                "I'd like to understand better. Can you elaborate on {context}?",
                "Let's explore this further. What specific aspects of {context} interest you?"
            ]
        
        return random.choice(templates)
    
    def train(
        self, 
        data_path: str,
        batch_size: int = 16,
        num_epochs: int = 5,
        learning_rate: float = 1e-4,
        save_path: str = "model_weights"
    ) -> None:
        """
        Train the model on custom data
        
        Args:
            data_path: Path to training data
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            save_path: Path to save model weights
        """
        print(f"Loading training data from {data_path}")
        
        # Create dataset
        dataset = CustomDataset(
            data_path=data_path,
            tokenizer=self.autoencoder.tokenizer,
            max_length=self.autoencoder.max_length
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        print(f"Loaded {len(dataset)} training examples")
        
        # Set models to training mode
        self.manipulator.train()
        
        # Create optimizer
        optimizer = optim.AdamW(
            params=self.manipulator.parameters(),
            lr=learning_rate
        )
        
        # Create loss function
        criterion = nn.MSELoss()
        
        # Training loop
        print("Starting training")
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in progress_bar:
                # Get input and output
                input_ids = batch["input_ids"].to(self.device)
                input_attention_mask = batch["input_attention_mask"].to(self.device)
                output_ids = batch["output_ids"].to(self.device)
                output_attention_mask = batch["output_attention_mask"].to(self.device)
                
                # Encode input
                with torch.no_grad():
                    input_embeds = []
                    for i in range(input_ids.size(0)):
                        # Get non-padding tokens
                        mask = input_attention_mask[i].bool()
                        tokens = input_ids[i][mask]
                        
                        # Skip empty sequences
                        if len(tokens) == 0:
                            input_embeds.append(torch.zeros(1, self.autoencoder.embedding_dim, device=self.device))
                            continue
                        
                        # Embed input
                        text = self.autoencoder.tokenizer.decode(tokens)
                        embed = self.autoencoder.embed(text)
                        input_embeds.append(embed)
                    
                    input_embeds = torch.cat(input_embeds, dim=0)
                
                # Encode output (target)
                with torch.no_grad():
                    target_embeds = []
                    for i in range(output_ids.size(0)):
                        # Get non-padding tokens
                        mask = output_attention_mask[i].bool()
                        tokens = output_ids[i][mask]
                        
                        # Skip empty sequences
                        if len(tokens) == 0:
                            target_embeds.append(torch.zeros(1, self.autoencoder.embedding_dim, device=self.device))
                            continue
                        
                        # Embed output
                        text = self.autoencoder.tokenizer.decode(tokens)
                        embed = self.autoencoder.embed(text)
                        target_embeds.append(embed)
                    
                    target_embeds = torch.cat(target_embeds, dim=0)
                
                # Process through manipulator
                processed_embeds = self.manipulator(input_embeds)
                
                # Calculate loss
                loss = criterion(processed_embeds, target_embeds)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update progress
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (progress_bar.n + 1)})
            
            # Print epoch statistics
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss / len(dataloader):.4f}")
        
        # Save model
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        torch.save(self.manipulator.state_dict(), os.path.join(save_path, "manipulator.pt"))
        torch.save(self.memory_bank.state_dict(), os.path.join(save_path, "memory_bank.pt"))
        
        # Save tokenizer and config
        config = {
            "model_name": self.autoencoder.model.config._name_or_path,
            "embedding_dim": self.autoencoder.embedding_dim,
            "max_length": self.autoencoder.max_length,
            "pooling_strategy": self.autoencoder.pooling_strategy
        }
        
        with open(os.path.join(save_path, "config.json"), 'w') as f:
            json.dump(config, f)
        
        print(f"Model saved to {save_path}")
        
        # Set models back to evaluation mode
        self.manipulator.eval()
    
    def load_weights(self, model_path: str) -> None:
        """
        Load model weights from disk
        
        Args:
            model_path: Path to model weights
        """
        # Load config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"Loaded config: {config}")
        
        # Load manipulator weights
        manipulator_path = os.path.join(model_path, "manipulator.pt")
        if os.path.exists(manipulator_path):
            self.manipulator.load_state_dict(torch.load(manipulator_path, map_location=self.device))
            print(f"Loaded manipulator weights from {manipulator_path}")
        
        # Load memory bank weights
        memory_path = os.path.join(model_path, "memory_bank.pt")
        if os.path.exists(memory_path):
            self.memory_bank.load_state_dict(torch.load(memory_path, map_location=self.device))
            print(f"Loaded memory bank weights from {memory_path}")
    
    def save_conversation(self, file_path: str) -> None:
        """
        Save conversation history to file
        
        Args:
            file_path: Path to save conversation
        """
        # Get messages from buffer
        messages = [item['message'] for item in self.conversation_buffer.buffer]
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(messages, f, indent=2)
        
        print(f"Conversation saved to {file_path}")
    
    def load_conversation(self, file_path: str) -> None:
        """
        Load conversation history from file
        
        Args:
            file_path: Path to load conversation from
        """
        # Clear current buffer
        self.conversation_buffer = EnhancedConversationBuffer(
            max_tokens=self.autoencoder.max_length,
            tokenizer=self.autoencoder.tokenizer
        )
        
        # Load messages
        with open(file_path, 'r', encoding='utf-8') as f:
            messages = json.load(f)
        
        # Add messages to buffer
        for message in messages:
            self.conversation_buffer.add_message(message)
        
        print(f"Loaded {len(messages)} messages from {file_path}")

def create_training_dataset(
    output_path: str,
    data_format: str = "json",
    sources: Optional[List[str]] = None,
    sample_data: bool = False
) -> None:
    """
    Create a training dataset
    
    Args:
        output_path: Path to save dataset
        data_format: Format of the dataset (json, csv)
        sources: List of source files/directories
        sample_data: Whether to include sample data
    """
    dataset = []
    
    # Process source files if provided
    if sources:
        for source in sources:
            if os.path.isfile(source):
                _, ext = os.path.splitext(source)
                
                if ext == '.json':
                    with open(source, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                        if isinstance(data, list):
                            dataset.extend(data)
                        elif isinstance(data, dict) and "conversations" in data:
                            # Extract conversation pairs
                            for conv in data["conversations"]:
                                if isinstance(conv, list):
                                    for i in range(0, len(conv) - 1, 2):
                                        if i + 1 < len(conv):
                                            dataset.append({
                                                "input": conv[i]["content"] if isinstance(conv[i], dict) else conv[i],
                                                "output": conv[i+1]["content"] if isinstance(conv[i+1], dict) else conv[i+1]
                                            })
                elif ext in ['.csv', '.tsv']:
                    df = pd.read_csv(source, sep=',' if ext == '.csv' else '\t')
                    
                    for _, row in df.iterrows():
                        if len(df.columns) >= 2:
                            dataset.append({
                                "input": row[df.columns[0]],
                                "output": row[df.columns[1]]
                            })
            elif os.path.isdir(source):
                # Process directory
                for root, _, files in os.walk(source):
                    for file in files:
                        _, ext = os.path.splitext(file)
                        
                        if ext in ['.json', '.csv', '.tsv']:
                            file_path = os.path.join(root, file)
                            create_training_dataset(
                                output_path=None,
                                sources=[file_path],
                                sample_data=False
                            )
    
    # Add sample data if requested
    if sample_data:
        sample_conversations = [
            {
                "input": "Hello, how are you?",
                "output": "I'm doing well! Thank you for asking. How can I help you today?"
            },
            {
                "input": "What is Python?",
                "output": "Python is a high-level, interpreted programming language known for its simplicity and versatility. It's widely used in web development, data science, AI, and automation."
            },
            {
                "input": "Explain neural networks",
                "output": "Neural networks are computational models inspired by the human brain. They consist of layers of interconnected nodes (neurons) that process data, learn patterns, and make predictions. Deep learning uses neural networks with many layers to solve complex problems like image recognition and natural language processing."
            },
            {
                "input": "How do I train a model?",
                "output": "Training a model involves providing it with labeled data, allowing it to learn patterns through an optimization process. You need to prepare your data, choose a model architecture, define a loss function, and use an optimization algorithm like gradient descent to minimize errors. During training, the model's parameters are adjusted to improve its performance on the task."
            }
        ]
        
        dataset.extend(sample_conversations)
    
    # Save dataset if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if data_format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2)
        elif data_format == 'csv':
            df = pd.DataFrame(dataset)
            df.to_csv(output_path, index=False)
        
        print(f"Created dataset with {len(dataset)} examples at {output_path}")
    
    return dataset

def main():
    """Main function to run the AI"""
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced AI with Multiple Neural Layers")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="Pretrained model name")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--max_length", type=int, default=1400, help="Maximum token length")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--data_path", type=str, default=None, help="Path to training data")
    parser.add_argument("--load_model", type=str, default=None, help="Path to load model weights")
    parser.add_argument("--save_model", type=str, default="model_weights", help="Path to save model weights")
    parser.add_argument("--create_dataset", action="store_true", help="Create a training dataset")
    parser.add_argument("--dataset_output", type=str, default="dataset.json", help="Path to save dataset")
    parser.add_argument("--dataset_format", type=str, default="json", choices=["json", "csv"], help="Dataset format")
    parser.add_argument("--sample_data", action="store_true", help="Include sample data in dataset")
    
    args = parser.parse_args()
    
    # Create dataset if requested
    if args.create_dataset:
        create_training_dataset(
            output_path=args.dataset_output,
            data_format=args.dataset_format,
            sources=[args.data_path] if args.data_path else None,
            sample_data=args.sample_data
        )
        
        if not args.train:
            return
    
    # Set device
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else \
                'mps' if torch.backends.mps.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    
    # Create AI
    print("\nInitializing Enhanced AI...")
    ai = EnhancedAI(
        model_name=args.model,
        device=device,
        max_length=args.max_length,
        n_layers=args.n_layers
    )
    
    # Load model weights if provided
    if args.load_model:
        ai.load_weights(args.load_model)
    
    # Train model if requested
    if args.train and args.data_path:
        ai.train(
            data_path=args.data_path,
            save_path=args.save_model
        )
    
    # Interactive mode
    print("\nEnhanced AI Assistant is ready!")
    print(f"Using model: {args.model}")
    print(f"Neural architecture: {args.n_layers} transformer layers")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("Type 'save' to save the conversation.")
    print("Type 'help' for more commands.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit']:
                print("\nAI: Goodbye! Thank you for the conversation.")
                break
            elif user_input.lower() == 'save':
                file_path = input("Enter file path to save conversation: ").strip()
                ai.save_conversation(file_path)
                continue
            elif user_input.lower() == 'load':
                file_path = input("Enter file path to load conversation: ").strip()
                ai.load_conversation(file_path)
                print("AI: Conversation loaded successfully. How can I help you?")
                continue
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  exit, quit - End the conversation")
                print("  save - Save the conversation to a file")
                print("  load - Load a conversation from a file")
                print("  help - Show this help message")
                continue
            elif not user_input:
                print("AI: I didn't catch that. Could you please say something?")
                continue
            
            # Generate response
            response = ai.get_response(user_input)
            print(f"\nAI: {response}")
            
        except KeyboardInterrupt:
            print("\n\nAI: Conversation terminated. Goodbye!")
            break
        except Exception as e:
            print(f"\nAI: I apologize, but I encountered an error: {str(e)}")
            print("Please try again or restart the program if the issue persists.")
            print(traceback.format_exc())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error occurred: {str(e)}")
        print("\nFull error traceback:")
        print(traceback.format_exc())
