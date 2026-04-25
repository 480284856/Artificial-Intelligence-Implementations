# Neural Network Framework Package - Team 04

A lightweight, modular, and pure-NumPy neural-network framework designed for general multi-class classification tasks. This package was developed to fulfill Requirements O1 and O1.1.

---

## Installation

To ensure all dependencies are resolved and the package is correctly integrated into your environment, install it directly via pip:

pip install git+[https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/courses/machine-learning/student-submissions/ws25/Team-04/neural-network-framework.git](https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/courses/machine-learning/student-submissions/ws25/Team-04/neural-network-framework.git)

---
## Package layout

```
nn_framework_package/
└── nn_framework/
    ├── __init__.py        # Public API exports
    ├── model.py           # Sequential model + math utilities
    ├── layers.py          # DenseLayer, DropoutLayer
    ├── activations.py     # ActivationReLu
    ├── loss.py            # ActivationSoftmaxAndCCE, MSE
    ├── optimizers.py      # OptimizerSGD, OptimizerAdam, OptimizerMomentum
    ├── metrics.py         # accuracy, confusion_matrix, f1, iou, balanced_accuracy
    └── serialization.py   # save_model, load_model
```

---
### 1 — Build a model

Use `Sequential` to stack layers one by one. Always finish with a `DenseLayer` that
outputs one neuron per class, then attach a loss function via `set_loss`.

```python

import numpy as np
from nn_framework import Sequential, DenseLayer, DropoutLayer, ActivationReLu
from nn_framework import ActivationSoftmaxAndCCE

model = Sequential()
model.add(DenseLayer(784, 128))   # input_size=784, hidden_neurons=128
model.add(ActivationReLu())
model.add(DropoutLayer(rate=0.1)) # drop 10 % of neurons during training
model.add(DenseLayer(128, 64))
model.add(ActivationReLu())
model.add(DenseLayer(64, 10))     # 10 output neurons for 10 classes

model.set_loss(ActivationSoftmaxAndCCE())
```
**Class-weighted loss** (useful for imbalanced datasets):

```python
# weights: low value for the majority class, high value for rare classes
class_weights = np.array([0.5, 2.0, 2.0])  # e.g. [idle, swipe, rotate]
model.set_loss(ActivationSoftmaxAndCCE(weights=class_weights))
```

---

### 2 — Train the model

The training loop is fully manual, giving you complete control:

```python
from nn_framework import OptimizerAdam

optimizer = OptimizerAdam(learning_rate=0.001)

EPOCHS     = 10
BATCH_SIZE = 128
n_samples  = X_train.shape[0]

for epoch in range(EPOCHS):
    # Shuffle each epoch
    idx = np.random.permutation(n_samples)
    X_shuff, y_shuff = X_train[idx], y_train[idx]

    epoch_loss = 0.0
    for i in range(0, n_samples, BATCH_SIZE):
        Xb = X_shuff[i : i + BATCH_SIZE]
        yb = y_shuff[i : i + BATCH_SIZE]

        # Forward pass
        preds = model.forward(Xb, training=True)

        # Compute loss
        loss = model.loss_layer.forward(preds, yb)
        epoch_loss += loss

        # Backward pass
        model.backward(model.loss_layer.backward(preds, yb))

        # Parameter update
        optimizer.update(model.layers)

    avg_loss = epoch_loss / (n_samples / BATCH_SIZE)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")
```

> **Note:** pass `training=True` during the forward pass so that `DropoutLayer`
> activates its masking logic. Use `training=False` (the default) for evaluation.

---

### 3 — Evaluate & Compare the model

To ensure results are stored outside of memory, the framework can export raw training history to JSON files:

#### 3.1 — Persistence (JSON Export)

```python
# Saves loss, accuracy, and F1-score to disk
model.save_history("results/experiment_v1.json")
```

---

#### 3.2 — Distinguishable Model Comparison

You can load multiple JSON files to generate combined plots, allowing for a clear comparison of different hyperparameter configurations or architectures:

```python
from nn_framework.visualization import plot_comparison

# Compare multiple training runs in a single chart
plot_comparison(["results/adam_run.json", "results/sgd_run.json"], metric="f1")
```

---

### Demonstration

The included `framework_test.ipynb` provides an executable walkthrough using the MNIST dataset of handwritten digits.