import numpy as np

"""
This script defines the Linear Layer of our model. It takes all incoming data
and calculates a "weighted sum" for each neuron. This is the place where the actual
learning happens by adjusting weights and biases over time.
"""

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        """
        Initialize the layer with a specific number of inputs and neurons.
        """
        # 1. Initialize weights with small random values (scaled by 0.1)# 1. Initialize weights with small random values (scaled by 0.1)
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.1
        # 2. Initialize biases as a row vector of zeros
        self.biases = np.zeros((1, n_neurons))

        # Storage for data needed during the backward pass
        self.inputs = None
        self.weights_grad = None
        self.biases_grad = None

    def forward(self, inputs, training=False):
        """
        Forward pass: Calculate the weighted sum of inputs plus biases.
        Formula: Z = X @ W + b
        """
        # Save inputs to use them as "leverage" during backpropagation
        self.inputs = inputs 
        self.output = inputs @ self.weights + self.biases
        return self.output
        
    def backward(self, delta_LB):
        """
        Backward pass: Calculate gradients for optimization and pass the error back.
        delta_incoming: The error gradient from the next layer (moving right to left).
        """
        # 1. Calculate gradients for weights and biases (to update them later)
        # 1.1 Weights: Uses the input as a "lever" to see how much they affected the error

        # Berechnung der Gradienten für Gewichte (thetas) und Biases
        # man gibt fehler für den nächsten (vorherigen!!) Layer (LA) zurück
        # 1. Gradienten für diesen Layer (LB) berechnen
        # 1.1 weights -> nutz hier den input als Hebel
        self.weights_grad = self.inputs.T @ delta_LB
        # 1.2 Biases: Sum the error across all samples for each neuron
        # dbiases = sum of dvalues (lokales Minimum für besten fehler aller 60.000 samples für dieses Neuron)
        self.biases_grad = np.sum(delta_LB, axis=0, keepdims=True)
        # 2. Calculate the error (delta) for the previous layer using the chain rule
        # This passes the error signal further back toward the start of the network
        # das neue Delta für Layer A nutzt die Gewichte als hebel  (teil eins der delta berechnung)
        delta_LA = delta_LB @ self.weights.T
        return delta_LA
    

class DropoutLayer:
    """
    Dropout Layer is a regularization technique used to prevent overfitting.
    It randomly sets a fraction of input units to 0 for every batch (forward pass),
    which prevents the network from relying on specific neurons.
    """
    def __init__(self, rate):
        # The probability of dropping a neuron (e.g., 0.1 means 10% are dropped)
        self.rate = rate
        self.mask = None

    def forward(self, inputs, training=False):
        """
        During training, randomly zeroes out neurons. During inference, 
        it passes the input through unchanged.
        """
        if training:
            # Create a binary mask: 1 with probability (1 - rate), else 0.
            # We divide by (1 - rate) to scale the remaining values (Inverted Dropout).
            # This ensures the sum of inputs remains consistent during training and testing.
            self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape) / (1 - self.rate)
            return inputs * self.mask
        else:
            # During testing/inference, we use all neurons (full power).
            return inputs

    def backward(self, d_inputs):
        """
        Only the neurons that were 'active' during the forward pass 
        receive a gradient update.
        """
        return d_inputs * self.mask
            