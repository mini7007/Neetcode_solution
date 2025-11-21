import numpy as np
from numpy.typing import NDArray

class Solution:
    
    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is an Nx3 NumPy array
        # weights is a 3x1 NumPy array
        
        # Perform Matrix Multiplication: (N x 3) dot (3 x 1) = (N x 1)
        # This calculates w1*x1 + w2*x2 + w3*x3 for every row simultaneously
        prediction = np.matmul(X, weights)
        
        return np.round(prediction, 5)


    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # model_prediction is an Nx1 NumPy array
        # ground_truth is an Nx1 NumPy array
        
        # Calculate Mean Squared Error (MSE)
        # 1. Calculate the difference (residuals)
        difference = model_prediction - ground_truth
        
        # 2. Square the differences
        squared_errors = np.square(difference)
        
        # 3. Calculate the mean of the squared errors
        mse = np.mean(squared_errors)
        
        return round(mse, 5)
