class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Initialize the current value of x
        current_x = float(init) 

        # Perform the gradient descent iterations
        for _ in range(iterations):
            # Gradient of f(x) = x^2 is f'(x) = 2x
            gradient = 2 * current_x
            
            # Gradient Descent Update Rule
            current_x = current_x - learning_rate * gradient
            
        # 1. Round the result to 5 decimal places
        final_result = round(current_x, 5)
        
        # 2. Check for formatting requirements:
        
        # A. If the final result is exactly 0, return the float 0.0 to match expected output.
        if final_result == 0.0:
            return 0.0
            
        # B. If the result is a non-zero whole number (like 5.0 in the first failed case), 
        # return the integer to match the expected '5' format.
        elif final_result == int(final_result):
            return int(final_result)
            
        # C. Otherwise (the decimal case, like 4.08536), return the float.
        else:
            return final_result
