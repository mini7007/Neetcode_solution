import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import List
from torchtyping import TensorType

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. specific order: positive first, then negative
        all_sentences = positive + negative
        
        # 2. Build the vocabulary set to find unique words
        vocab_set = set()
        for sentence in all_sentences:
            words = sentence.split()
            for word in words:
                vocab_set.add(word)
        
        # 3. Sort vocabulary lexicographically (A-Z)
        sorted_vocab = sorted(list(vocab_set))
        
        # 4. Create mapping: Word -> Index (starting at 1)
        word_to_idx = {word: i + 1 for i, word in enumerate(sorted_vocab)}
        
        # 5. Encode sentences into tensors of indices
        encoded_tensors = []
        for sentence in all_sentences:
            words = sentence.split()
            # Map words to indices using the dictionary
            indices = [word_to_idx[word] for word in words]
            encoded_tensors.append(torch.tensor(indices))
            
        # 6. Pad the sequence to form a rectangular tensor (2N x T)
        # batch_first=True ensures the shape is [Batch_Size, Max_Length]
        padded_dataset = pad_sequence(encoded_tensors, batch_first=True, padding_value=0)
        
        # Return as float as indicated by the example output [1.0, 7.0, etc.]
        return padded_dataset.float()
