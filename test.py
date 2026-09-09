import numpy as np
from aif.nn.embedding import PositionalEmbedding

if __name__ == "__main__":
    # loss_module = BinaryEntropyLoss()
    # p = np.array([
    #     [1],
    #     [2],
    #     [3]
    # ])
    # q = np.array([
    #     [1,20,1,1],
    #     [1,1,20,1],
    #     [1,1,1,20],
    # ])
    # loss_mean = loss_module.forward(p=p,q=q)
    # print(loss_mean)
    # print(loss_mean.backward())

    posembedding = PositionalEmbedding(512)