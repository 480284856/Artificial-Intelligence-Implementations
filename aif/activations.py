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
        self.mask = X<=0
        X=X.copy()
        X[self.mask] = 0
        return X
    
    def backward(self, delta:np.ndarray) -> np.ndarray:

        # delta: [bs, feature dim]
        delta=delta.copy()
        delta[self.mask] = 0
        return delta

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
        delta=delta.copy()
        result=delta*(1-delta)
        return result