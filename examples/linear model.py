import numpy as np

from aif.linear import Linear
from aif.activations import ReLU,Sigmoid
from aif.utils.sequential import Sequential
from aif.loss import CrossEntropyLoss

class LinearModel:
    def __init__(self, in_features:int, out_features:int, hidden_features:int=10, bias:bool=True) -> None:
        self.modules = Sequential(
            Linear(in_features, hidden_features, bias),
            ReLU(),
            Linear(hidden_features, out_features, bias),
        )
        self.activation = Sigmoid()
    def forward(self, X:np.ndarray) -> np.ndarray:
        return self.modules(X)
    
    def backward(self, delta:np.ndarray) -> None:
        self.modules.backward(delta)
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        return self.activation(self.forward(X))

if __name__ == "__main__":
    model = LinearModel(2,2, hidden_features=10)
    loss_module = CrossEntropyLoss()

    X = np.random.randn(2,2) 
    label = np.array([
        [1],
        [0]
    ])   
    logits = model.forward(X)

    loss = loss_module(p=label, q=logits)
    delta = loss.backward()
    model.backward(delta)
