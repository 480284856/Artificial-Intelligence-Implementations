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