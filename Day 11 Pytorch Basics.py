import torch
import torch.nn
from torchtyping import TensorType

class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # Reshape into (M * N // 2) x 2
        # The '-1' tells PyTorch to calculate the row dimension automatically
        reshaped = torch.reshape(to_reshape, (-1, 2))
        return torch.round(reshaped, decimals=4)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # Average column-wise (dim=0 collapses the rows)
        averaged = torch.mean(to_avg, dim=0)
        return torch.round(averaged, decimals=4)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # Concatenate along dimension 1 (columns)
        concatenated = torch.cat((cat_one, cat_two), dim=1)
        return torch.round(concatenated, decimals=4)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # Calculate Mean Squared Error loss
        loss = torch.nn.functional.mse_loss(prediction, target)
        return torch.round(loss, decimals=4)
