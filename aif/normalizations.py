import numpy as np
from .utils.module import TrainableModule, Parameter

class BatchNorm(TrainableModule):
    def __init__(self, num_features, eps=1e-5, train=True, momentum: float = 0.9,):
        super().__init__()
        self.eps = eps
        self.train = train

        self.gamma = Parameter(np.ones((1, num_features)))
        self.beta = Parameter(np.zeros((1, num_features)))
        self._parameters = [self.gamma, self.beta]

        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones_like(self.running_mean)
        self.momentum = momentum

    def forward(self, X: np.ndarray):
        if self.train:
            self.X = X
            mean = X.mean(axis=0, keepdims=True)
            self.std = X.var(axis=0, keepdims=True)
            self.std = np.sqrt(self.std + self.eps)
            self.X_hat = (X - mean) / self.std

            self.running_mean = self.momentum*self.running_mean + (1-self.momentum)*mean
            self.running_var = self.momentum*self.running_var + (1-self.momentum)*self.std
        else:
            self.std = np.sqrt(self.running_var + self.eps)
            self.X_hat = (X - self.running_mean) / self.std
        return self.gamma.value * self.X_hat + self.beta.value
    
    def backward(self, delta: np.ndarray):
        if self.train:
            self.gamma.grad = np.sum(delta * self.X_hat, axis=0, keepdims=True)
            self.beta.grad = np.sum(delta, axis=0, keepdims=True)
            
            bs = delta.shape[0]
            # Standard batch normalization backward formula
            dX = (self.gamma.value / (bs * self.std)) * (
                bs * delta 
                - np.sum(delta, axis=0, keepdims=True) 
                - self.X_hat * np.sum(delta * self.X_hat, axis=0, keepdims=True)
            )
            return dX
        else:
            return (delta * self.gamma.value) / self.std

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
            
        