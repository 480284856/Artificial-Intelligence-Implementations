import numpy as np
import json

"""
SERIALIZATION MODULE
--------------------
Provides save_model() and load_model() utilities for persisting a trained
Sequential model to disk and restoring it later.

Format: a single .npz file containing
  - All DenseLayer weight and bias arrays
  - A JSON metadata string that encodes the layer order and types
    (so the architecture can be fully reconstructed without the original
    Python code that built the model).

Supported layer types: DenseLayer, ActivationReLu, DropoutLayer.
Supported loss types : ActivationSoftmaxAndCCE, MSE.
"""


def save_model(model, filepath, extra_arrays=None):
    """
    Save a trained Sequential model to a .npz file.

    Parameters
    ----------
    model        : Sequential
        A trained model instance.
    filepath     : str or Path
        Destination path.  A '.npz' extension is added automatically if
        the path does not already end with it.
    extra_arrays : dict, optional
        Additional numpy arrays to store alongside the model weights
        (e.g. scaler_mean, scaler_std, class_names, feature_names).

    Example
    -------
    >>> from nn_framework.serialization import save_model
    >>> save_model(model, "my_model")          # writes my_model.npz
    >>> save_model(model, "checkpoints/best")  # writes checkpoints/best.npz
    """
    arrays = {}
    layer_configs = []

    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__

        if layer_type == "DenseLayer":
            arrays[f"layer_{i}_weights"] = layer.weights
            arrays[f"layer_{i}_biases"] = layer.biases
            layer_configs.append({"type": layer_type, "idx": i})

        elif layer_type == "DropoutLayer":
            layer_configs.append({"type": layer_type, "idx": i, "rate": layer.rate})

        elif layer_type == "ActivationReLu":
            layer_configs.append({"type": layer_type, "idx": i})

        else:
            raise ValueError(
                f"Layer type '{layer_type}' at index {i} is not supported by "
                "save_model. Supported types: DenseLayer, ActivationReLu, DropoutLayer."
            )

    # Persist optional class weights stored in the loss layer
    loss_config = None
    if hasattr(model, "loss_layer") and model.loss_layer is not None:
        loss_type = type(model.loss_layer).__name__
        loss_config = {"type": loss_type}
        if (
            hasattr(model.loss_layer, "weights")
            and model.loss_layer.weights is not None
        ):
            arrays["loss_class_weights"] = model.loss_layer.weights

    meta = json.dumps({"layers": layer_configs, "loss": loss_config})
    arrays["__meta__"] = np.array([meta])

    if extra_arrays:
        arrays.update(extra_arrays)

    np.savez(filepath, **arrays)


def load_model(filepath):
    """
    Load a Sequential model that was previously saved with save_model().

    Parameters
    ----------
    filepath : str or Path
        Path to the .npz file (with or without the '.npz' extension).

    Returns
    -------
    Sequential
        A fully reconstructed model with all weights restored.

    Example
    -------
    >>> from nn_framework.serialization import load_model
    >>> model = load_model("my_model.npz")
    >>> predictions = model.predict(X_test)
    """
    from nn_framework.model import Sequential
    from nn_framework.layers import DenseLayer, DropoutLayer
    from nn_framework.activations import ActivationReLu
    from nn_framework.loss import ActivationSoftmaxAndCCE, MSE

    data = np.load(filepath, allow_pickle=False)
    meta = json.loads(str(data["__meta__"][0]))

    model = Sequential()

    for config in meta["layers"]:
        layer_type = config["type"]
        idx = config["idx"]

        if layer_type == "DenseLayer":
            w = data[f"layer_{idx}_weights"]
            b = data[f"layer_{idx}_biases"]
            layer = DenseLayer(w.shape[0], w.shape[1])
            layer.weights = w.copy()
            layer.biases = b.copy()
            model.add(layer)

        elif layer_type == "DropoutLayer":
            model.add(DropoutLayer(config["rate"]))

        elif layer_type == "ActivationReLu":
            model.add(ActivationReLu())

        else:
            raise ValueError(
                f"Unknown layer type '{layer_type}' found in saved model. "
                "The file may have been created with a different version of the framework."
            )

    # Restore the loss layer
    loss_info = meta.get("loss")
    if loss_info:
        loss_type = loss_info["type"]
        if loss_type == "ActivationSoftmaxAndCCE":
            class_weights = (
                data["loss_class_weights"].copy()
                if "loss_class_weights" in data
                else None
            )
            model.set_loss(ActivationSoftmaxAndCCE(weights=class_weights))
        elif loss_type == "MSE":
            model.set_loss(MSE())
        else:
            raise ValueError(
                f"Unknown loss type '{loss_type}' found in saved model."
            )

    return model
