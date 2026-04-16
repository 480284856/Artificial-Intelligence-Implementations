import numpy as np

from aif.linear import Linear
from aif.activations import ReLU,Sigmoid
from aif.utils.sequential import Sequential

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
    X = np.random.randn(2,2)    
    Y = model.forward(X)
    print(Y)
