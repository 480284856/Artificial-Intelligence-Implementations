import numpy as np

"""
OPTIMIZER MODULE
----------------
This script contains the update logic for the neural network's weights. 
Optimizers take the gradients calculated during backpropagation and 
adjust the parameters to minimize the loss function.

Algorithms:
1. SGD (Stochastic Gradient Descent): The classic, straightforward approach.
2. Adam (Adaptive Moment Estimation): A sophisticated, state-of-the-art 
   optimizer that adjusts the learning rate for each parameter individually.
"""

class OptimizerSGD:
    """
    Stochastic Gradient Descent (SGD) Optimizer.
    A straightforward optimizer that updates weights by moving in the 
    opposite direction of the gradient.
    """
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate

    def update(self, layers):
        """
        Performs a parameter update for each layer in the model.
        """
        for layer in layers:
            # Check if the layer has weights and gradients to update
            if hasattr(layer, 'weights') and hasattr(layer, 'weights_grad'):
                # Formula: theta_new = theta_old - learning_rate * gradient
                # The optimizer accesses layer attributes directly and performs in-place updates.
                # Since layers are passed by reference, no return value is needed.
                layer.weights -= self.learning_rate * layer.weights_grad
                layer.biases -= self.learning_rate * layer.biases_grad


class OptimizerAdam:
    """
    Adaptive Moment Estimation (Adam) Optimizer.
    A more advanced algorithm that maintains a 'memory' (moments) of 
    past gradients to adjust the learning rate for each parameter individually.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.t = 0
        
        # Dictionaries to store state (m and v) for each layer separately
        self.m_w = {}
        self.v_w = {}
        self.m_b = {}
        self.v_b = {}

    def update(self, layers, grad_noise_std=0.0):
        """
        Updates parameters using the Adam algorithm, including optional 
        gradient noise and mandatory clipping for stability.
        """
        self.t += 1
    
        # 1. Bias correction for the learning rate
        # This compensates for the fact that moments are initialized at zero.
        lr_t = self.lr * (np.sqrt(1.0 - (self.beta2 ** self.t)) / (1.0 - (self.beta1 ** self.t)))
        
        for layer in layers:
            # Skip layers that do not contain trainable weights
            if not hasattr(layer, 'weights'):
                continue

            # 2. Add Gradient Noise (Regularization)
            # Helps the model escape local minima and generalize better.
            if grad_noise_std > 0:
                layer.weights_grad += np.random.normal(0, grad_noise_std, layer.weights_grad.shape)
                layer.biases_grad += np.random.normal(0, grad_noise_std, layer.biases_grad.shape)

            # 3. Gradient Clipping (The "Emergency Brake")
            # Prevents 'Exploding Gradients' by capping the max value of an update.
            np.clip(layer.weights_grad, -1.0, 1.0, out=layer.weights_grad)
            np.clip(layer.biases_grad, -1.0, 1.0, out=layer.biases_grad)
            
            layer_id = id(layer)

            # Initialize moments if this is the first time seeing this layer
            if layer_id not in self.m_w:
                self.m_w[layer_id] = np.zeros_like(layer.weights)
                self.v_w[layer_id] = np.zeros_like(layer.weights)
                self.m_b[layer_id] = np.zeros_like(layer.biases)
                self.v_b[layer_id] = np.zeros_like(layer.biases)
            
            # 4. Weight (W) Update
            # m = momentum (average of gradients), v = velocity (average of squared gradients)
            self.m_w[layer_id] = self.beta1 * self.m_w[layer_id] + (1.0 - self.beta1) * layer.weights_grad
            self.v_w[layer_id] = self.beta2 * self.v_w[layer_id] + (1.0 - self.beta2) * (layer.weights_grad**2)
            layer.weights -= lr_t * self.m_w[layer_id] / (np.sqrt(self.v_w[layer_id]) + self.eps)

            # 5. Bias (b) Update
            self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1.0 - self.beta1) * layer.biases_grad
            self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1.0 - self.beta2) * (layer.biases_grad**2)
            layer.biases -= lr_t * self.m_b[layer_id] / (np.sqrt(self.v_b[layer_id]) + self.eps)


class OptimizerMomentum:
    def __init__(self, learning_rate=0.01, beta=0.9):
        self.lr = learning_rate
        self.beta = beta
        self.vW = None # Velocity for weights
        self.vb = None # Velocity for biases

    def update(self, layers):
        if self.vW is None:
            # Initialize velocities with zeros on the first pass
            self.vW = [np.zeros_like(layer.weights) for layer in layers if hasattr(layer, 'weights')]
            self.vb = [np.zeros_like(layer.biases) for layer in layers if hasattr(layer, 'biases')]

        v_idx = 0
        for layer in layers:
            if hasattr(layer, 'weights'):
                # The core Momentum logic from Xiaojie's script
                # v = beta * v + dW
                self.vW[v_idx] = self.beta * self.vW[v_idx] + layer.weights_grad
                self.vb[v_idx] = self.beta * self.vb[v_idx] + layer.biases_grad

                # W = W - lr * v
                layer.weights -= self.lr * self.vW[v_idx]
                layer.biases -= self.lr * self.vb[v_idx]
                v_idx += 1