import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):
    
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # 0. Instantiate the linear layers in the following order: Key, Query, Value.
        self.key_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value_layer = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.attention_dim = attention_dim
    
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # embedded shape: (batch_size, context_length, embedding_dim)
        batch_size, context_length, embedding_dim = embedded.shape

        # Step 1: Project inputs to Key, Query, and Value
        # Shape: (batch_size, context_length, attention_dim)
        k = self.key_layer(embedded)
        q = self.query_layer(embedded)
        v = self.value_layer(embedded)

        # Step 2: Calculate Attention Scores
        # Transpose K to (batch_size, attention_dim, context_length) for matmul
        k_t = torch.transpose(k, 1, 2)
        
        # scores shape: (batch_size, context_length, context_length)
        scores = torch.matmul(q, k_t)
        
        # *** FIX: Scale the scores by sqrt(attention_dim) ***
        scores = scores / (self.attention_dim ** 0.5)

        # Step 3: Apply Masking
        # Create a lower triangular mask
        mask = torch.tril(torch.ones(context_length, context_length))
        mask = mask.to(embedded.device) # Ensure device compatibility
        
        # Apply mask: Set future positions to -infinity
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # Step 4: Softmax
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)

        # Step 5: Aggregate Values
        output = torch.matmul(attention_weights, v)
        
        # Return answer to 4 decimal places
        return torch.round(output * 10000) / 10000
