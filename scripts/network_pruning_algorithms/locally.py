import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch.nn.utils.prune as prune
from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import \
    TabularNeuralNetTorchModel
from torch import Tensor, nn, no_grad
from torch.utils.hooks import RemovableHandle
from utils.common import get_logger

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")


def _locally(model: TabularNeuralNetTorchModel, layers: List[nn.Module], amount, grade=1, dim=0, **_):
    # dim=0 prune neuron, dim=1 prune connection.
    # when called with dim=0 it produces the same output of unstructured
    logger.info(
        f"Locally Pruning {'Neurons' if dim==0 else 'Connections'} using {amount=} {grade=} {dim=} and n.{len(layers)=}")
    if grade < 0:
        [prune.random_structured(k, 'weight', amount=v, dim=dim) for k, v in zip(layers, amount)]
    elif grade in (1, 2):
        [prune.ln_structured(k, 'weight', amount=v, n=grade, dim=dim) for k, v in zip(layers, amount)]
    else:
        raise NotImplementedError(f"Norm Grade {grade} not implemented")

    logger.info("Applying mask removing buffers")
    [prune.remove(k, 'weight') for k in layers]

    logger.info("Removing biases corresponding to zero neurons")

    with no_grad():
        for lay in layers:
            for i, w in enumerate(lay.weight):
                if not w.any().item():
                    lay.bias[i] = 0.
    return model, layers


def _get_neurons_activations(model: TabularNeuralNetTorchModel, layers: List[nn.Module], X: pd.DataFrame, grade):
    activations: Dict[nn.Module, np.ndarray] = {}

    logger.info(f"Registering forward hooks for {len(layers)=}")

    def get_activation(module: nn.Module, _, output: Tensor):
        activations[module] = np.linalg.norm(output.detach().numpy(), ord=grade, axis=0)

    hooks: List[RemovableHandle] = []

    for lay in layers:
        hooks.append(lay.register_forward_hook(get_activation))

    logger.info("Calling forward")
    model.predict(X)

    logger.info("Removing forward hooks")
    for h in hooks:
        h.remove()
    return activations


def _locally_neurons_activation(
        model: TabularNeuralNetTorchModel, layers: List[nn.Module],
        amount, X: pd.DataFrame, grade=1, subset_features: List[str] = [], **_):
    logger.info(
        f"Locally Pruning Neurons looking for activation, length of subset provided {len(subset_features)=}"
        f" {amount=} and n.{len(layers)=} layers")

    if subset_features:
        logger.info("Zeroing inactive features for subset")
        X = X.copy(deep=True)
        X[X.columns.difference(subset_features)] = 0.

    logger.info("Getting neurons' activations")
    activations = _get_neurons_activations(model, layers, X, grade)

    logger.info("Pruning in each layer")
    for am, (lay, acts) in zip(amount, activations.items()):
        indexes_to_zero = acts.argsort()[:round(am * len(acts))]
        with no_grad():
            lay.weight[indexes_to_zero, :] = 0.
            lay.bias[indexes_to_zero] = 0

    return model, layers


def locally_neurons_l1(model: TabularNeuralNetTorchModel, layers: List[nn.Module], amount, **kwargs):
    return _locally(model, layers, amount, grade=1, dim=0, **kwargs)


def locally_neurons_l2(model: TabularNeuralNetTorchModel, layers: List[nn.Module], amount, **kwargs):
    return _locally(model, layers, amount, grade=2, dim=0, **kwargs)


def locally_neurons_random(model: TabularNeuralNetTorchModel, layers: List[nn.Module], amount, **kwargs):
    return _locally(model, layers, amount, grade=-1, dim=0, **kwargs)


def locally_connections_l1(model: TabularNeuralNetTorchModel, layers: List[nn.Module], amount, **kwargs):
    return _locally(model, layers, amount, grade=1, dim=1, **kwargs)


def locally_connections_l2(model: TabularNeuralNetTorchModel, layers: List[nn.Module], amount, **kwargs):
    return _locally(model, layers, amount, grade=2, dim=1, **kwargs)


def locally_connections_random(model: TabularNeuralNetTorchModel, layers: List[nn.Module], amount, **kwargs):
    return _locally(model, layers, amount, grade=-1, dim=1, **kwargs)


def locally_neurons_activation_l1(
        model: TabularNeuralNetTorchModel, layers: List[nn.Module],
        amount, X: pd.DataFrame, subset_features: List[str] = [], **_):
    return _locally_neurons_activation(model, layers, amount, X, grade=1, subset_features=subset_features)


def locally_neurons_activation_l2(
        model: TabularNeuralNetTorchModel, layers: List[nn.Module],
        amount, X: pd.DataFrame, subset_features: List[str] = [], **_):
    return _locally_neurons_activation(model, layers, amount, X, grade=2, subset_features=subset_features)


def locally_neurons_activation_l1_for_subset(
        model: TabularNeuralNetTorchModel, layers: List[nn.Module],
        amount, X: pd.DataFrame, subset_features: List[str], **kwargs):
    return _locally_neurons_activation(model, layers, amount, X, grade=1, subset_features=subset_features, **kwargs)


def locally_neurons_activation_l2_for_subset(
        model: TabularNeuralNetTorchModel, layers: List[nn.Module],
        amount, X: pd.DataFrame, subset_features: List[str], **kwargs):
    return _locally_neurons_activation(model, layers, amount, X, grade=2, subset_features=subset_features, **kwargs)
