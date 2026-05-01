import numpy as np
from .utils.module import Optimizer, Parameter


class SGD(Optimizer):
    def __init__(self, learning_rate: float, params: list[Parameter]):
        super().__init__(learning_rate, params)

    def step(self):
        for p in self.params:
            p:Parameter
            p.value -= self.learning_rate * p.grad