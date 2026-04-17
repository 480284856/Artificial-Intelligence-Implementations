import numpy as np

def softmax(X:np.ndarray):
    """
    X: [bs, feature_dim|categories]
    """
    assert len(X.shape) == 2

    tmp = np.exp(X)
    return tmp/np.sum(tmp, axis=-1,keepdims=True)