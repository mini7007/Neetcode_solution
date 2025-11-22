import numpy as np
from numpy.typing import NDArray

class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        # Initialize weights
        weights = initial_weights
        
        # N is the number of samples in the dataset
        N = len(X)
        
        # Loop for the specified number of iterations
        for _ in range(num_iterations):
            
            # 1. Get current predictions based on current weights
            model_prediction = self.get_model_prediction(X, weights)
            
            # 2. Calculate derivatives for all weights (w1, w2, w3)
            # We must calculate ALL gradients based on the CURRENT weights 
            # before updating any of them to ensure a simultaneous update.
            gradients = []
            for i in range(len(weights)):
                gradient = self.get_derivative(model_prediction, Y, N, X, i)
                gradients.append(gradient)
            
            # 3. Update weights
            # New Weight = Old Weight - (Learning Rate * Gradient)
            weights = weights - (self.learning_rate * np.array(gradients))
            
        # Return weights rounded to 5 decimal places as requested
        return np.round(weights, 5)
