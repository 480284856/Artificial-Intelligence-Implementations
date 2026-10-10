import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = np.array(data)
        