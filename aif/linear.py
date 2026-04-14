import numpy as np
from .utils.module import Module

class Linear(Module):
    def __init__(self, in_features: int, out_features:int, bias:bool=True, *args, **kwargs):
        self.in_features = in_features
        self.out_features = out_features
        self.set_bias = bias

        self.W = np.random.randn(self.in_features, self.out_features)
        self.B = np.random.randn(self.out_features) if self.set_bias else 0

    def forward(self, X: np.ndarray):
        return X@self.W+self.B
    
    def backward(self, delta: np.ndarray):
        """
        gradient calculation
        
        :param delta: $delta=\\frac{ \partial L }{ \partial Z_{L+1} }$
        :type delta: np.ndarray
        """
        # TODO: implement it after finishing forward propagation
        pass
