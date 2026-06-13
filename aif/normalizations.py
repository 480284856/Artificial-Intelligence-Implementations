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
            # written by gemini
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
    
        if self.train:
            bs = delta.shape[0]
            # path I: L -> X
            gradient1 = np.tile(-1/self.var, (bs, 1))
            # path II: L -> mean(X) -> X
            numerator = delta.sum(axis=0)
            denominator = bs * self.var
            gradient2 = numerator/denominator
            gradient2 = np.tile(gradient2, (bs,1))
            # path III: L -> Std(X) -> X
            gradient3 = -1*delta.sum(axis=0)/self.var**2 * (self.X - self.mean)/(bs*self.var)
            gradient = gradient1+gradient2+gradient3
            return gradient

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
            
        