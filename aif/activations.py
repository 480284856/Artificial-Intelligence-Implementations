import numpy as np
from .utils.module import Module

class ReLU(Module):
    def __init__(self):
        """
        ReLU(X) = 0, if X<0
                = X, else
        :param self: Description
        """
    
    def forward(self, X:np.ndarray) -> np.ndarray:
        return np.maximum(
            x1=X,
            x2=0
        )
    
    def backward(self, delta:np.ndarray) -> np.ndarray:
        pass

class Sigmoid(Module):
    def __init__(self):
        """
        y = 1/(1+e^(-x))
        """
        super().__init__()
    def forward(self, X:np.ndarray) -> np.ndarray:
        self.X = X
        self.Y = 1/(1+np.exp(-X))
        return self.Y
    
    def backward(self, delta):
        """
        \partial sigmoid(x) / \partial x = sigmoid(x) * (1-sigmoid(x))
        """
        pass