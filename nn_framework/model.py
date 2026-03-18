import numpy as np

"""
MODEL MODULE
------------
This script defines the 'Sequential' class, which acts as the core engine 
of the neural network. It manages the linear stack of layers, coordinating 
the flow of data during training and inference.

Core Responsibilities:
1. Orchestration: Linking layers together in a specific sequence.
2. Forward Propagation: Passing input data through each layer to get a prediction.
3. Backward Propagation: Implementing the chain rule by passing error gradients 
   in reverse order to update weights.
"""

class Sequential:
    """
    A linear stack of layers. Data flows from the first added layer 
    to the last during the forward pass and in reverse during the backward pass.
    """
    def __init__(self):
        # List to store layer objects (Dense, ReLu, etc.) - our "train wagons"
        self.layers = []
        # The referee that calculates the final error score
        self.loss = None

        self.history = {'loss': [], 'val_acc': [], 'f1': []}

    def add(self, layer):
        """Adds a new layer to the end of the model stack."""
        self.layers.append(layer)

    def set_loss(self, loss_layer):
        """Assigns the loss function (the 'referee') to the model."""
        self.loss_layer = loss_layer

    def forward(self, inputs, training=False): 
        """
        Runs the forward pass through all layers.
        The 'training' flag is broadcasted to every layer.
        """
        self.output = inputs
        for layer in self.layers:
            # BROADCASTING THE FLAG:
            # This ensures that layers like Dropout know whether to 
            # activate their logic or remain passive
            self.output = layer.forward(self.output, training=training)
        return self.output
    
    def predict_proba(self, X):
        """
        Returns the probability distribution for each class.
        This is essentially a forward pass with training=False.
        """
        # We ensure training=False so Dropout layers don't stay active
        return self.forward(X, training=False)

    def backward(self, d_inputs):
        """
        Runs the backward pass (Backpropagation) through all layers in REVERSE.
        This implements the chain rule to calculate gradients for each layer.
        """
        # 1. THE REVERSE PASS (Chain Rule Execution)
        # We propagate the error signal (d_inputs) from the output layer 
        # back toward the input layer.
        for layer in reversed(self.layers):
            # Each layer calculates its own local gradient and passes 
            # the signal to the previous layer.
            d_inputs = layer.backward(d_inputs)

        # 2. GRADIENT COLLECTION (Preparation for Optimizer)
        # We collect the specific weight and bias gradients (dW, db) from 
        # layers that possess trainable parameters (like DenseLayer).
        grads = []
        for layer in self.layers:
            # We verify if the layer has stored its internal gradients
            if hasattr(layer, 'weights_grad') and hasattr(layer, 'biases_grad'):
                grads.append((layer.weights_grad, layer.biases_grad))
        
        # 3. THE HANDOVER
        # Return the collected gradients to the training loop/optimizer.
        # This ensures 'grads' is no longer 'None', fixing the TypeError.
        return grads
        

    @property
    def W(self):
        """Shortcut to get all layer weights."""
        return [l.weights for l in self.layers if hasattr(l, 'weights')]

    @property
    def b(self):
        """Shortcut to get all layer biases."""
        return [l.biases for l in self.layers if hasattr(l, 'biases')]

    def predict(self, X):
        """Predicts the final class index (0, 1, 2...)."""
        probs = self.forward(X, training=False)
        return np.argmax(probs, axis=1)
    
    @property
    def layer_sizes(self):
        """Helper to get the architecture for saving/loading."""
        sizes = []
        # Get the input size from the first layer
        if self.layers:
            sizes.append(self.layers[0].weights.shape[0])
            # Get the output sizes of all DenseLayers
            for layer in self.layers:
                if hasattr(layer, 'weights'):
                    sizes.append(layer.weights.shape[1])
        return np.array(sizes)
    
    def save_history(self, filename):
        """
        Exports the training history to a JSON file (Requirement O1.1).
        This ensures metrics are stored persistently and can be loaded for comparison.
        """
        import json
        import os
        
        # Ensure the output directory exists
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # Convert numpy values to native Python floats for JSON serialization
        export_data = {k: [float(v) for v in vals] for k, vals in self.history.items()}
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=4)
        print(f"Training history successfully saved to: {filename}")


# --- MATH UTILITIES ---

def compute_class_weights(y, num_classes):
    """Calculates weights to balance imbalanced data (O1 requirement)."""
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    # Avoid division by zero
    weights = total / (num_classes * np.maximum(counts, 1))
    return weights.astype("float32")

def weighted_cross_entropy(probs, y_true, sample_weights):
    """Calculates the CCE loss with weight support."""
    samples = len(probs)
    # Clip to avoid log(0)
    probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    
    # Calculate loss for each sample
    core_loss = -np.sum(y_true * np.log(probs_clipped), axis=1)
    # Apply the weight (e.g., make 'rotate' more important than 'idle')
    return np.mean(core_loss * sample_weights)

def clip_gradients_in_place(grads, max_norm):
    """Prevents gradients from 'exploding' (exploding gradient problem)."""
    total_norm = 0
    # dW and db are the gradients of the weights and biases
    # imagine we combine all the gradients into a single vector
    # so the total norm is the norm of this vector (using the Pythagorean theorem)
    # it means the length of the gradient vector
    for dW, db in grads:
        total_norm += np.sum(dW**2) + np.sum(db**2)
    total_norm = np.sqrt(total_norm)

    # if the total norm is greater than the max norm, we need to scale down the gradients
    # to prevent the gradients from exploding
    if total_norm > max_norm:
        factor = max_norm / (total_norm + 1e-6)
        for dW, db in grads:
            dW *= factor
            db *= factor