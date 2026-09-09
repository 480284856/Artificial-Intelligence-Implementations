import numpy as np
from typing import Any

class Module:
    def __init__(self):
        pass
    def forward(self, X:np.ndarray) -> np.ndarray: # type: ignore
        pass
    def backward(self, delta:np.ndarray):
        pass
    def __call__(self, X:np.ndarray, *args: Any, **kwds: Any) -> Any:
        return self.forward(X)

class TrainableModule(Module):
    def __init__(self):
        super().__init__()
        self._parameters = list()

    @property
    def parameters(self):
        return self._parameters

class Model(TrainableModule):
    def __init__(self):
        self.modules = []
        super().__init__()
    

class Parameter:
    def __init__(self, value: np.ndarray):
        self.value = value
        self.grad = None

class Optimizer:
    def __init__(self, 
        learning_rate: float,
        params: list[Module]
    ):
        self.learning_rate = learning_rate
        self.params = params

    def step(self):
        pass