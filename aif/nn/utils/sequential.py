import numpy as np
from typing import Any
from .module import TrainableModule

class Sequential(TrainableModule):
    def __init__(self, *modules) -> None:
        super().__init__()
        self.modules = [*modules]

        for m in modules:
            if isinstance(m, TrainableModule):
                self._parameters.extend(m.parameters)

    def __call__(self, X:np.ndarray, *args: Any, **kwds: Any) -> Any:
        return self.forward(X)

    def forward(self, X:np.ndarray) -> np.ndarray:
        for m in self.modules:
            X = m(X)
        o=X
        return o
    
    def backward(self, delta:np.ndarray) -> np.ndarray:
        for m in reversed(self.modules):
            delta = m.backward(delta)
        return delta