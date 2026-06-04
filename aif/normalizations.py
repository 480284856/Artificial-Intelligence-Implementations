import numpy as np
from .utils.module import Module

class BatchNorm(Module):
    def __init__(self, eps=1e-5, train=True):
        super().__init__()
        self.eps = eps
        self.train = train

    def forward(self, X: np.ndarray):
        if self.train:
            self.X = X
            self.mean = X.mean(axis=0, keepdims=True)
            self.var = X.var(axis=0, keepdims=True)
            self.X_hat = (X - self.mean) / np.sqrt(self.var + self.eps)
        
        return self.X_hat
    
    def backward(self, delta: np.ndarray):
        if self.train:
            
        