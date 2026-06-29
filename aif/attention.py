import numpy as np
from .utils.module import TrainableModule, Parameter, Module
from linear import Linear

class Attention(TrainableModule):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features
        self.qkv = Linear(in_features=in_features, out_features=in_features*3,bias=False)

    def forward(self, X):
        '''
        X: [bs, sequence length, embedding dimension]
           [2, 512, 768]
        '''
        qkv = self.qkv(X)
        q,k,v = np.split(qkv, 3, axis=-1)

        attention_score_raw = q @ k.transpose(0,2,1) / np.sqrt(self.in_features)