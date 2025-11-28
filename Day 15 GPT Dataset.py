import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]]]:
        torch.manual_seed(0)
        
        # Split raw_dataset into words (do NOT use .lower())
        data = raw_dataset.split()
        
        # We need to pick a random starting index 'i'.
        # The input X will use data[i : i + context_length]
        # The target Y will use data[i+1 : i + context_length + 1]
        # The largest index accessed in Y is (i + context_length).
        # This index must be < len(data).
        # So, i + context_length < len(data)  =>  i < len(data) - context_length.
        # Since torch.randint high is exclusive, we set high to len(data) - context_length.
        high = len(data) - context_length
        
        # Generate batch_size random starting indices
        random_indices = torch.randint(low=0, high=high, size=(batch_size,))
        
        X = []
        Y = []
        
        for idx in random_indices:
            idx = idx.item() # Convert tensor value to int
            
            # Input: sequence of length context_length starting at idx
            X.append(data[idx : idx + context_length])
            
            # Target: sequence of length context_length starting at idx + 1
            Y.append(data[idx + 1 : idx + context_length + 1])
            
        return X, Y
