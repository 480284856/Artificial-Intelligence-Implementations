import numpy as np
from .utils.module import TrainableModule, Parameter

class Linear(TrainableModule):
    def __init__(self, in_features: int, out_features:int, bias:bool=True, *args, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.set_bias = bias

        self.W = Parameter(np.random.randn(self.in_features, self.out_features)*1e-1)
        self.B = Parameter(np.random.randn(self.out_features)*1e-1 if self.set_bias else np.zeros(1))

        self._parameters = [self.W, self.B]

    def forward(self, X: np.ndarray):
        self.X=X
        return X@self.W.value+self.B.value
    
    def backward(self, delta: np.ndarray):
        """
        gradient calculation
        
        :param delta: $delta=\\frac{ \partial L }{ \partial Z_{L+1} }$
        :type delta: np.ndarray
        """
        # delta: [bs, out_features]
        # W: [in_features, out_features]
        # X: [bs, in_features]

        # gradient
        self.W.grad = self.X.T @ delta
        # 对于偏置的求导来说，它就等于那一层的 delta 乘以输入（也就是 1），其实也就是 delta。因为又是批量的（batch size 大于 1），所以会有多个 delta，最后取平均。
        assert len(delta.shape)==2, "对偏置求导的假设是基于：第一个维度是批大小维度，第二个维度则是特征维度，也就是神经元的个数的维度"
        self.B.grad = delta.sum(axis=0)

        delta = delta @ self.W.value.T
        return delta