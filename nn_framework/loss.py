import numpy as np

"""
This script provides loss functions to measure the model's performance. 
It includes a combined Softmax and Cross-Entropy loss for classification tasks 
and a Mean Squared Error (MSE) loss for regression-style predictions.

Concept: Weighted Cross-Entropy (WCE)

"Instead of using standard Cross-Entropy, we use Weighted Cross-Entropy to switch the model into a 'Focus Mode'.

- The Problem: In our dataset, 'Idle' is the majority class. Standard CE allows the model to 'take the easy way out' by guessing 'Idle' most of the time, as it will still achieve high accuracy without truly learning gestures.

- The Solution: We assign a low weight to 'Idle' and a high weight to gestures (like 'Swipe').

- The Incentive: By doing this, we teach the model that missing a gesture is much worse than misclassifying 'Idle'. A high weight creates 'high mathematical pain' (high loss) when a gesture is ignored.

- The Result: This higher penalty forces the model to work harder to recognize actual movements, making these connections and information related to gestures far more important during training."
"""

class ActivationSoftmaxAndCCE: 
    """
    Combined Softmax activation and Categorical Cross-Entropy (CCE) loss.
    This combination allows for a much faster and more stable gradient calculation.
    """
    def __init__(self, weights=None): # Optional class weights (used for Weighted Cross Entropy Loss)
        self.output = None
        self.loss = None
        self.weights = weights  

    def forward(self, inputs, y_true):
        # 1. Softmax stability trick: subtract the max to prevent overflow (e^x becoming too large)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # 2. Calculate probabilities (Softmax)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        # 3. Calculate Loss (CCE)
        samples = len(self.output)

        # Prevent mathematical errors (log(0)) by clipping values between a tiny number and 1
        y_pred_clipped = np.clip(self.output, 1e-9, 1 - 1e-9)

        # Extract the confidence for the correct class for each sample
        correct_confidences = y_pred_clipped[range(samples), y_true]
        # Negative Log Likelihood: 
        # We take the negative log of the correct class probability.
        # 1. The 'log' punishes wrong predictions exponentially.
        # 2. The 'minus' turns the negative result into a positive 'Loss' value.
        # High confidence = Low Loss | Low confidence = High Loss.  
        self.loss = -np.log(correct_confidences)

        # Apply weighting if weights are provided
        if self.weights is not None:
            # Select the appropriate weight for each sample based on its true class
            batch_weights = self.weights[y_true]

            # Multiply the loss by the weight. This increases the "cost" for failing 
            # to recognize important classes like gestures.

            # the loss of each image is multiplied by its class weight 
            # (so that the model learns to focus more on important classes like gestures and less on the majority class 'Idle')
            self.loss *= batch_weights

        return np.mean(self.loss)

    def backward(self, y_pred, y_true):
        # 1. Determine number of samples
        samples = len(self.output)

        # 2. Create a copy of the predictions for the gradient (y_pred), outgoing = LA
        self.delta_outgoing = self.output.copy()

        # 3. Calculate the gradient using the One-Hot Vector logic:
        # Instead of creating a full One-Hot matrix, we simply subtract 1 
        # from the probability of the correct class (where the One-Hot value would be 1.0)
        self.delta_outgoing[range(samples), y_true] -= 1

        # Apply weighting in the backward pass
        if self.weights is not None:
            batch_weights = self.weights[y_true]
            # Reshape is used for broadcasting (multiplying the column vector)
            self.delta_outgoing *= batch_weights.reshape(-1, 1) # verhilft zu broadcasting also dass wir spalte haben statt flache zeile

        # 4. Normalize the gradient
        # Dividing by sample size ensures updates aren't influenced by batch size 
        # (sprich: Stärke der Korrektur (update) ist NICHT abh. davon wie groß das Batch ist)
        # otherwise the accumulation of gradients will be different for different batch sizes
        # it's the 'm' in the formula: 1/m * sum(loss)
        # J/z
        self.delta_outgoing = self.delta_outgoing / samples

        return self.delta_outgoing


class MSE:
    """
    Mean Squared Error (MSE) loss function. 
    Typically used for predicting continuous values rather than categories.
    """
    def __init__(self):
        self.output = None 
        self.loss = None

    def forward(self, y_pred, y_true):
        # Formula: 1/n * sum((y_pred - y_true)^2)
        self.loss = np.mean((y_pred - y_true) **2)
        return self.loss
    
    def backward(self, y_pred, y_true):
        # Derivative: 2/n * (y_pred - y_true)
        samples = len(y_pred)
        self.data_outgoing = 2 * (y_pred - y_true) / samples
        return self.data_outgoing
    