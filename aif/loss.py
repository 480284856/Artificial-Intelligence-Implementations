import numpy as np
from .utils.module import Module
from .utils.functional import softmax

class BinaryEntropyLoss(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, p:np.ndarray, q:np.ndarray):
        """
        H(p,q) = -p(x)*log q(x)

        p: shape: [bs,] | real distribution
        q: shape: [bs, category] | predicted distribution, should be logits instead of probabilities.
        """
        assert len(p.shape) == 2 and p.shape[-1] == 1
        assert len(q.shape) == 2
        
        tmp = np.zeros_like(q)
        row_index = np.arange(tmp.shape[0]).reshape(-1,1)
        tmp[row_index, p] = 1
        p = tmp

        q = softmax(q)

        tmp = -p*np.log(q)
        loss:np.ndarray = tmp.sum(axis=-1)
        loss_mean = loss.mean()

        return loss_mean
    
    def backward(self, delta: np.ndarray):
        # do it tomorrow
        pass