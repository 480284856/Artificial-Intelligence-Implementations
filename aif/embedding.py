import numpy as np
from .utils.module import TrainableModule, Parameter

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
        return self.embedding[X]
    
    def backward(self, delta):
        # delta = αL/α_output
        # α_output/α_weight = 1
        # αL/α_weight = delta
        grad = np.zeros_like(self.embedding.value)
        np.add.at(grad, self.inputs, delta)
        self.embedding.grad = grad
