import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        
        # 1. Embedding Layer
        # Takes an index (word) and outputs a vector of size 16
        self.embedding = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=16)
        
        # 2. Linear Layer
        # Takes the averaged embedding (size 16) and outputs a single value
        self.linear = nn.Linear(in_features=16, out_features=1)
        
        # 3. Sigmoid Activation
        # Squashes the output between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Step 1: Convert input to embeddings
        # Input x shape: (Batch_Size, Sequence_Length)
        # We assume x might be float from the example, so we cast to .long() for embedding lookup
        x = self.embedding(x.long()) 
        # New shape: (Batch_Size, Sequence_Length, 16)

        # Step 2: "Bag of Words" averaging
        # We calculate the mean across the time dimension (dim=1)
        x = x.mean(dim=1)
        # New shape: (Batch_Size, 16)

        # Step 3: Linear transformation
        x = self.linear(x)
        # New shape: (Batch_Size, 1)

        # Step 4: Sigmoid activation
        x = self.sigmoid(x)
        
        # Step 5: Round to 4 decimal places
        return torch.round(x, decimals=4)
