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
from aif.normalizations import BatchNorm
from aif.activations import ReLU,Sigmoid
from aif.utils.sequential import Sequential
from aif.loss import MSELoss
from aif.utils.module import Model
from aif.optimizers import SGD
from aif.dropout import Dropout

FEATURE_COLS = [
    "gross_floor_area_buildings_sq_ft",
    "year_built",
    # "of_buildings",
    # # "electricity_use_kbtu",
    # "natural_gas_use_kbtu",
    # "site_eui_kbtu_sq_ft",
    # "source_eui_kbtu_sq_ft",
    # "weather_normalized_site_eui_kbtu_sq_ft",
    # "weather_normalized_source_eui_kbtu_sq_ft",
    # "total_ghg_emissions_metric_tons_co2e",
    # "ghg_intensity_kg_co2e_sq_ft",
]
TARGET_COL = "total_ghg_emissions_metric_tons_co2e"


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
            # Dropout(0.3),
            BatchNorm(hidden_features),
            # Linear(hidden_features, hidden_features, bias),
            # ReLU(),
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
    X_train,
    Y_train,
    loss_func,
    optimizer,
    epochs:int=1000,
    bs:int=4,
):
    loss_list = []
    for epoch in range(epochs):
        loss_epoch = 0
        for i in range(0, X_train.shape[0], bs):
            x = X_train.values[i:i+bs]
            y = Y_train.values.reshape(-1,1)[i:i+bs]

            logits = model.forward(x)
            loss = loss_func(p=y, q=logits)
            loss_value = loss.loss_mean
            loss_epoch += loss_value
            delta = loss.backward()
            model.backward(delta)
            optimizer.step()
        loss_list.append(loss_epoch/X_train.shape[0])
        print(f"Epoch {epoch:>3}  loss: {loss_epoch/X_train.shape[0]:.4f}")
    return loss_list

if __name__ == "__main__":
    X, Y = load_chicago_energy(data_dir="examples/Chicago Energy Benchmarking/dataset")
    
    # Split the dataset into train and test sets (80% train, 20% test)
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(X))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train = X.iloc[train_indices]
    Y_train = Y.iloc[train_indices]
    X_test = X.iloc[test_indices]
    Y_test = Y.iloc[test_indices]
    
    bs = 10
    # lr = 0.001
    lr = 0.01 * np.sqrt(bs)
    print(f"Batch size: {bs}, Learning rate: {lr}")
    model = LinearModel(X_train.shape[1],1, hidden_features=100)
    loss_module = MSELoss()
    optimizer = SGD(
        learning_rate=lr,
        params=model.parameters
    )

    loss_list = train(
        model,
        X_train,
        Y_train,
        loss_module,
        optimizer,
        epochs=100,
        bs=bs,
    )
    
    # Evaluate R2 score on the test set
    y_pred = model.forward(X_test.values).flatten()
    y_true = Y_test.values
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"Test set R2 score: {r2:.4f}")
    
    sns.lineplot(x=range(len(loss_list)), y=loss_list)
    plt.show()

    print("训练结束")