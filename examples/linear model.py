import numpy as np

from aif.linear import Linear
from aif.activations import ReLU,Sigmoid
from aif.utils.sequential import Sequential
from aif.loss import CrossEntropyLoss
from aif.utils.module import Model
from aif.optimizers import SGD

class LinearModel(Model):
    def __init__(self, in_features:int, out_features:int, hidden_features:int=10, bias:bool=True) -> None:
        super().__init__()
        self.model = Sequential(
            Linear(in_features, hidden_features, bias),
            ReLU(),
            Linear(hidden_features, out_features, bias),
        )
        self.activation = Sigmoid()

        self.parameters.extend(self.model.parameters)

    def forward(self, X:np.ndarray) -> np.ndarray:
        return self.model(X)
    
    def backward(self, delta:np.ndarray) -> None:
        self.model.backward(delta)
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        return self.activation(self.forward(X))

if __name__ == "__main__":
    model = LinearModel(2,2, hidden_features=100)
    loss_module = CrossEntropyLoss()
    optimizer = SGD(
        learning_rate=0.0001,
        params=model.parameters
    )

    # 生成二分类数据：左边一团（类别 0），右边一团（类别 1）
    n = 1000
    X0 = np.random.randn(n, 2) + np.array([-2, 0])
    X1 = np.random.randn(n, 2) + np.array([2, 0])
    X = np.vstack([X0, X1])                                    # [200, 2]
    label = np.vstack([np.zeros((n, 1), dtype=int),            # [200, 1]
                       np.ones((n, 1), dtype=int)])

    # 训练循环
    loss_value = np.inf
    epoch = 0
    while loss_value > 1e-3:
        logits = model.forward(X)
        loss = loss_module(p=label, q=logits)
        loss_value = loss.loss_mean
        delta = loss.backward()
        model.backward(delta)
        optimizer.step()
        epoch += 1


        print(f"Epoch {epoch:>3}  loss: {loss_value:.4f}")
    print("训练结束")
    # print(model.predict(X))