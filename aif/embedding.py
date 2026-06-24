import numpy as np
from .utils.module import TrainableModule, Parameter, Module

class Embedding(TrainableModule):
    def __init__(self,
                 vocab,
                 embedding_dim:int = 512,
                 ):
        self.embedding = Parameter(
            value=np.random.randn(len(vocab), embedding_dim)
        )
    
    def forward(self, X:np.ndarray):
        assert len(X.shape) == 1, "X = [1,2,3,...]"
        self.inputs = X.copy()
        return self.embedding.value[X]
    
    def backward(self, delta):
        # delta = αL/α_output
        # α_output/α_weight = 1
        # αL/α_weight = delta
        grad = np.zeros_like(self.embedding.value)
        np.add.at(grad, self.inputs, delta)
        self.embedding.grad = grad

class PositionalEmbedding(Module):
    def __init__(self, d_model: int = None, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = None
        if d_model is not None:
            self._precompute_pe()
            
    def _precompute_pe(self):
        pe = np.zeros((self.max_len, self.d_model))
        pos = np.arange(self.max_len).reshape(-1,1)
        idx_e = np.arange(0, self.d_model,step=2).reshape(1,-1)
        idx_o = np.arange(1, self.d_model,step=2).reshape(1,-1)
        tmp_e = pos/10000**(idx_e/self.d_model)
        tmp_o = pos/10000**((idx_o - 1)/self.d_model)
        pe[:, 0::2] = np.sin(tmp_e)
        pe[:, 1::2] = np.cos(tmp_o)
        
        self.pe = pe

    def forward(self, X: np.ndarray):
        """
        X: token embeddings
           shape: (seq_len, d_model) or (batch_size, seq_len, d_model)
        """
        ...
    def backward(self, delta: np.ndarray):
        """
        For X_out = X_in + PE, the gradient with respect to X_in is delta.
        """
        ...