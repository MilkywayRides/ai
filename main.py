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
from typing import List, Dict, Any
import pickle

# Suppress warnings
warnings.filterwarnings("ignore", message="Instantiating a decoder T5Attention without passing `layer_idx`")

# Check and install dependencies
missing_deps = []
try:
    import transformers
except ImportError:
    missing_deps.append("transformers")
try:
    import tiktoken
except ImportError:
    missing_deps.append("tiktoken")
try:
    import google.protobuf
except ImportError:
    missing_deps.append("protobuf")
try:
    import sentencepiece
except ImportError:
    missing_deps.append("sentencepiece")

if missing_deps:
    print("Installing missing dependencies...")
    import subprocess
    for dep in missing_deps:
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
    print("Dependencies installed successfully.")

    if "transformers" in missing_deps:
        import transformers
        from transformers import AutoTokenizer, AutoModel

from transformers import AutoTokenizer, AutoModel

class SimpleAutoencoder:
    def __init__(self, device='cpu', max_length=1400):
        self.device = device
        self.max_length = max_length
        print(f"Using device: {self.device}")
        
        model_name = "bert-base-uncased"
        print(f"Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.hidden_size
        self.memory_bank = {}
        print(f"Model loaded with embedding dimension: {self.embedding_dim}")

    @torch.no_grad()
    def embed(self, text: str) -> torch.FloatTensor:
        if len(self.tokenizer.encode(text)) > self.max_length:
            chunks = self._chunk_text(text)
            embeddings = []
            for chunk in chunks:
                inputs = self.tokenizer(chunk, return_tensors='pt', 
                                      truncation=True, 
                                      max_length=self.max_length).to(self.device)
                outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :])
            return torch.mean(torch.stack(embeddings, dim=0), dim=0)
        else:
            inputs = self.tokenizer(text, return_tensors='pt', 
                                  truncation=True, 
                                  max_length=self.max_length).to(self.device)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0, :]

    def _chunk_text(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), self.max_length):
            chunk_tokens = tokens[i:i + self.max_length]
            chunks.append(self.tokenizer.decode(chunk_tokens))
        return chunks

    @torch.no_grad()
    def embed_batch(self, texts: list) -> torch.FloatTensor:
        embeddings = []
        for text in texts:
            embedding = self.embed(text)
            embeddings.append(embedding)
        return torch.stack(embeddings, dim=0)

    def generate_response(self, context: str, query_embedding: torch.Tensor) -> str:
        with torch.no_grad():
            context_embedding = self.embed(context)
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding, context_embedding)
            
            response_template = self._get_response_template(similarity.item())
            return response_template.format(context=context)

    def _get_response_template(self, similarity: float) -> str:
        if similarity > 0.8:
            templates = [
                "Based on our discussion, {context}",
                "I understand clearly that {context}",
                "From what we've discussed, {context}"
            ]
        elif similarity > 0.5:
            templates = [
                "I think {context}, but let me know if I should clarify anything",
                "From what I understand, {context}",
                "It seems that {context}, is that correct?"
            ]
        else:
            templates = [
                "I'm processing this information. Could you tell me more about {context}?",
                "I'd like to understand better. Can you elaborate on {context}?",
                "Let's explore this further. What specific aspects of {context} interest you?"
            ]
        return random.choice(templates)

class LatentManipulator(nn.Module):
    def __init__(self, embedding_dim=768):
        super(LatentManipulator, self).__init__()
        
        self.input_norm = nn.LayerNorm(embedding_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        self.processor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1024, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(3)
        ])
        
        self.decoder = nn.Sequential(
            nn.Linear(1024, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.encoder(x)
        
        for layer in self.processor:
            residual = x
            x = layer(x)
            x = x + residual  # Residual connection
            
        x = self.decoder(x)
        return x

class ConversationBuffer:
    def __init__(self, max_tokens=1400):
        self.max_tokens = max_tokens
        self.buffer = deque()
        self.current_tokens = 0
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.entity_memory = {}

    def add_message(self, message: Dict[str, str]) -> None:
        message_text = f"{message['role']}: {message['content']}"
        tokens = len(self.tokenizer.encode(message_text))
        
        self.buffer.append({
            'message': message,
            'tokens': tokens,
            'timestamp': datetime.datetime.now()
        })
        self.current_tokens += tokens
        
        self._extract_entities(message)
        
        while self.current_tokens > self.max_tokens and len(self.buffer) > 1:
            removed = self.buffer.popleft()
            self.current_tokens -= removed['tokens']

    def _extract_entities(self, message: Dict[str, str]) -> None:
        content = message['content'].lower()
        
        # Extract names
        name_patterns = ["my name is", "i am", "call me"]
        for pattern in name_patterns:
            if pattern in content:
                try:
                    name = content.split(pattern)[1].strip().split()[0].capitalize()
                    self.entity_memory['user_name'] = {
                        'value': name,
                        'timestamp': datetime.datetime.now()
                    }
                except:
                    pass

    def get_context_string(self) -> str:
        return "\n".join([f"{item['message']['role']}: {item['message']['content']}" 
                         for item in self.buffer])

class DavidBot:
    def __init__(self, autoencoder, manipulator):
        self.autoencoder = autoencoder
        self.manipulator = manipulator
        self.conversation_buffer = ConversationBuffer(max_tokens=1400)
        self.load_knowledge_base()

    def load_knowledge_base(self):
        self.knowledge_base = {
            "greetings": [
                "Hello! I remember our previous conversations. How can I assist you today?",
                "Hi there! I'm ready to help you.",
                "Welcome! How may I be of service?",
            ],
            "python": {
                "general": "Python is a high-level, interpreted programming language known for its simplicity and versatility. It's widely used in web development, data science, AI, and automation.",
                "functions": """Let me show you how to create a Python function:

def greet(name):
    return f"Hello, {name}! Welcome to Python programming!"

# Here's a more complex example:
def calculate_area(length, width=None):
    if width is None:
        # If width is not provided, assume it's a square
        return length * length
    return length * width""",
            },
            "identity": "I am David, an AI assistant with enhanced memory and learning capabilities. I can maintain detailed conversations while remembering our previous interactions.",
        }

    def get_response(self, user_input: str) -> str:
        self.conversation_buffer.add_message({
            "role": "user",
            "content": user_input
        })
        
        response = self._generate_response(user_input)
        
        self.conversation_buffer.add_message({
            "role": "assistant",
            "content": response
        })
        
        return response

    def _generate_response(self, user_input: str) -> str:
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
        
        # Handle Python queries
        if "python" in lower_input:
            if "function" in lower_input:
                return self.knowledge_base["python"]["functions"]
            return self.knowledge_base["python"]["general"]
        
        # Handle identity questions
        if "who are you" in lower_input:
            return self.knowledge_base["identity"]

        # Generate response using neural components
        try:
            context = self.conversation_buffer.get_context_string()
            input_embedding = self.autoencoder.embed(user_input)
            
            with torch.no_grad():
                processed_embedding = self.manipulator(input_embedding)
            
            response = self.autoencoder.generate_response(context, processed_embedding)
            
            # Add personal touch if we know the user's name
            if user_name and random.random() < 0.3:
                response = f"{user_name}, {response}"
            
            return response

        except Exception as e:
            print(f"Error in neural processing: {str(e)}")
            return f"I'm still processing that{f', {user_name}' if user_name else ''}. Could you please elaborate?"

def main():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading autoencoder...")
    autoencoder = SimpleAutoencoder(device=device, max_length=1400)
    print("Autoencoder loaded successfully.")

    manipulator = LatentManipulator(embedding_dim=768).to(device)
    manipulator.eval()

    David = DavidBot(autoencoder, manipulator)

    print("\nDavid AI Assistant is ready!")
    print("Enhanced with deep learning and 1400-token context memory")
    print("Type 'exit' or 'quit' to end the conversation.")
    print("-" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\nDavid: Goodbye! I'll remember our conversation for next time!")
                break
            
            if not user_input:
                print("David: I didn't catch that. Could you please say something?")
                continue
            
            response = David.get_response(user_input)
            print(f"\nDavid: {response}")
            
        except KeyboardInterrupt:
            print("\n\nDavid: Conversation terminated by user. I'll remember our chat!")
            break
        except Exception as e:
            print(f"\nDavid: I apologize, but I encountered an error: {str(e)}")
            print("Please try again or restart the program if the issue persists.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error occurred: {str(e)}")
        print("\nFull error traceback:")
        print(traceback.format_exc())
