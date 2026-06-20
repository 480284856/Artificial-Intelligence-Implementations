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
            self.std = X.var(axis=0, keepdims=True)
            self.std = np.sqrt(self.std + self.eps)
            self.X_hat = (X - self.mean) / self.std
        
        return self.X_hat
    
    def backward(self, delta: np.ndarray):
        if self.train:
            # written by gemini(checked)
            bs = delta.shape[0]
            # Standard batch normalization backward formula
            dX = (1.0 / (bs * self.std)) * (
                bs * delta 
                - np.sum(delta, axis=0, keepdims=True) 
                - self.X_hat * np.sum(delta * self.X_hat, axis=0, keepdims=True)
            )
            return dX
        else:
            return delta / self.std

if __name__ == "__main__":
    X = np.array([
        [1,2,3],
        [1,2,3],
        [1,2,3]
    ])
    delta = np.array([
        [0.1,0.2,0.3],
        [0.1,0.2,0.3],
        [0.1,0.2,0.3]
    ])
    normalizer = BatchNorm()
    normalizer.forward(X)
    normalizer.backward(delta)
            
        