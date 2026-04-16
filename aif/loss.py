from turtle import forward
import numpy as np
from .utils.module import Module

class BinaryEntropyLoss(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, p:np.ndarray, q:np.ndarray):
        """
        H(p,q) = -p(x)*log q(x)

        p: shape: [bs, category] | real distribution
        q: shape: [bs,] | predicted distribution
        """
        assert len(q.shape) == 1
        tmp = np.zeros_like(p)
        tmp[:, q] = 1
        q = tmp

        loss = -p*np.log(q)
        loss:np.ndarray = loss.sum(axis=-1)
        loss = loss.mean()

        return loss
    
    def backward(self, delta: np.ndarray):
        # do it tomorrow
        pass

