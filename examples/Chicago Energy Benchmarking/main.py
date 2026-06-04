"""
Chicago Energy Benchmarking 数据加载

输入:
    data_dir (str): 数据集目录路径，默认 "dataset"

输出:
    X (pd.DataFrame): 特征矩阵，包含以下列：
        - Gross Floor Area - Buildings (sq ft)
        - Year Built
        - # of Buildings
        - Electricity Use (kBtu)
        - Natural Gas Use (kBtu)
        - Site EUI (kBtu/sq ft)
        - Source EUI (kBtu/sq ft)
        - Weather Normalized Site EUI (kBtu/sq ft)
        - Weather Normalized Source EUI (kBtu/sq ft)
        - Total GHG Emissions (Metric Tons CO2e)
        - GHG Intensity (kg CO2e/sq ft)
    Y (pd.Series): 目标变量 — ENERGY STAR Score
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from aif.linear import Linear
from aif.activations import ReLU,Sigmoid
from aif.utils.sequential import Sequential
from aif.loss import MSELoss
from aif.utils.module import Model
from aif.optimizers import SGD
from aif.dropout import Dropout

FEATURE_COLS = [
    "Gross Floor Area - Buildings (sq ft)",
    "Year Built",
    "# of Buildings",
    "Electricity Use (kBtu)",
    "Natural Gas Use (kBtu)",
    "Site EUI (kBtu/sq ft)",
    "Source EUI (kBtu/sq ft)",
    "Weather Normalized Site EUI (kBtu/sq ft)",
    "Weather Normalized Source EUI (kBtu/sq ft)",
    "Total GHG Emissions (Metric Tons CO2e)",
    "GHG Intensity (kg CO2e/sq ft)",
]
TARGET_COL = "Total GHG Emissions (Metric Tons CO2e)"


def load_chicago_energy(data_dir: str = "dataset"):
    """加载 Chicago Energy Benchmarking，返回 (X, Y)。"""
    df = pd.read_csv(os.path.join(data_dir, "chicago-energy-benchmarking.csv"))
    
    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS)
    X = df[FEATURE_COLS]
    X = (X-X.mean())/X.std()
    # drop rows where any cell's absolute value > 1.5
    mask = np.abs(X).max(axis=1) < 1.5
    X = X[mask]
    Y = df[TARGET_COL]
    Y = Y[mask]
    Y = (Y - Y.mean())/Y.std()
    return X, Y

class LinearModel(Model):
    def __init__(self, in_features:int, out_features:int, hidden_features:int=10, bias:bool=True) -> None:
        super().__init__()
        self.model = Sequential(
            Linear(in_features, hidden_features, bias),
            ReLU(),
            Linear(hidden_features, hidden_features, bias),
            # Dropout(0.1),
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

def train(
    model:Model, 
    X,
    Y,
    loss_func,
    optimizer,
    epochs:int=1000,
    bs:int=4,
):
    loss_list = []
    for epoch in range(epochs):
        loss_epoch = 0
        for i in range(0, X.shape[0], bs):
            x = X.values[i:i+bs]
            y = Y.values.reshape(-1,1)[i:i+bs]

            logits = model.forward(x)
            loss = loss_func(p=y, q=logits)
            loss_value = loss.loss_mean
            loss_epoch += loss_value
            delta = loss.backward()
            model.backward(delta)
            optimizer.step()
        loss_list.append(loss_epoch/X.shape[0])
        print(f"Epoch {epoch:>3}  loss: {loss_epoch/X.shape[0]:.4f}")
    return loss_list

if __name__ == "__main__":
    X, Y = load_chicago_energy(data_dir="examples/Chicago Energy Benchmarking/dataset")
    bs = 10
    lr = np.sqrt(bs)*0.01
    # lr = 0.01 * bs
    print(f"Batch size: {bs}, Learning rate: {lr}")
    model = LinearModel(X.shape[1],1, hidden_features=200)
    loss_module = MSELoss()
    optimizer = SGD(
        learning_rate=lr,
        params=model.parameters
    )

    loss_list = train(
        model,
        X,
        Y,
        loss_module,
        optimizer,
        epochs=100,
        bs=bs,
    )
    sns.lineplot(x=range(len(loss_list)), y=loss_list)
    plt.show()

    print("训练结束")