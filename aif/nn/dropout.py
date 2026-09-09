from .utils.module import Module
import numpy as np

class Dropout(Module):
    def __init__(self,p=0.5, training=True):
        super().__init__()
        self.p = p
        self.training = training
    
    def forward(self, X: np.ndarray):
        if self.training:
            self.X = X
            self.mask = np.random.rand(*X.shape) > self.p
            return X * self.mask / (1-self.p)
        else:
            return X
    
    def backward(self, delta: np.ndarray):
        return delta * self.mask / (1-self.p)