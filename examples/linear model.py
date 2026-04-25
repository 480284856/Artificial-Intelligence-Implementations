import numpy as np

from aif.linear import Linear
from aif.activations import ReLU,Sigmoid
from aif.utils.sequential import Sequential
from aif.loss import BinaryEntropyLoss

class LinearModel:
    def __init__(self, in_features:int, out_features:int, bias:bool) -> None:
        self.modules = Sequential(
            Linear(in_features, out_features, bias),
            Sigmoid()
        )
    
    def forward(self, X:np.ndarray) -> np.ndarray:
        return self.modules(X)

if __name__ == "__main__":
    model = LinearModel(2,1,True)
    loss_module = BinaryEntropyLoss()

    X = np.random.randn(2,2) 
    label = np.array([
        [1],
        [0]
    ])   
    Y = model.forward(X)
    Y = np.concat([
        Y,
        1-Y
    ], axis=-1)
    print("Y=", Y)
    loss = loss_module(p=label, q=Y)
    print(loss)
