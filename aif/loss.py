import numpy as np
from .utils.module import Module
from .utils.functional import softmax

class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()
    
    def __str__(self) -> str:
        try:
            return str(self.loss_mean)
        except:
            return ""

    def __call__(self, p:np.ndarray, q:np.ndarray):
        return self.forward(p,q)

    def forward(self, p:np.ndarray, q:np.ndarray):  # type: ignore
        """
        H(p,q) = -p(x)*log q(x)

        p: shape: [bs,1] | real distribution
        q: shape: [bs, category] | predicted distribution, should be logits instead of probabilities.
        """
        assert len(p.shape) == 2 and p.shape[-1] == 1
        assert len(q.shape) == 2
        
        # backup for backward
        self._label = p
        self._row_index = np.arange(p.shape[0]).reshape(-1,1)
        tmp = np.zeros_like(q)
        tmp[self._row_index, p] = 1
        self._onehot_label = tmp
        self._prob = softmax(q)

        tmp = q-q.max(axis=-1,keepdims=True)
        log_prob = tmp - np.log(np.sum(np.exp(tmp), axis=-1,keepdims=True))

        tmp = -1 * self._onehot_label*log_prob
        loss:np.ndarray = tmp.sum(axis=-1)
        self.loss_mean = loss.mean()

        return self
    
    def backward(self,): # type: ignore
        """
        \\frac{ \\partial L}{ \\partial z }  = 
            p_i - y_i, if p_i is the probability of real class
            p_i * y_i, others
        In engineering, both cases can be processed as: p-y, where y is the one hot vector.
        [
            [0.7 -1, 0.1 -0, 0.1 -0, 0.1 -0],
            [0.1 -0, 0.7 -1, 0.1 -0, 0.1 -0],
        ]
        """

        # self._prob[self._row_index, self._label] -= 1
        # delta = self._prob
        # self._prob: [bs, category]
        delta = self._prob - self._onehot_label
        return delta