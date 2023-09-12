import os
from typing import List, Dict

import numpy as np
import pandas as pd
import torch.nn.utils.prune as prune
from autogluon.tabular.models.tabular_nn.torch.tabular_nn_torch import \
    TabularNeuralNetTorchModel
from .locally import _get_neurons_activations
from torch import nn, no_grad
from utils.common import get_logger

logger = get_logger(f"{os.path.basename(__file__).replace('.py', '')}")


def _globally(
        model: TabularNeuralNetTorchModel, layers: List[nn.Module],
        amount, method=prune.L1Unstructured, **_):
    logger.info(f"Globally Pruning using {amount=} method={method.__class__.__name__} and n.{len(layers)=} layers")

    prune.global_unstructured({(x, 'weight') for x in layers},
                              pruning_method=method,
                              importance_scores=None,
                              amount=amount)

    logger.info("Applying mask removing buffers")

    [prune.remove(k, 'weight') for k in layers]
    logger.info("Check for neurons with empty weights and remove corresponding biases")
    with no_grad():
        for lay in layers:
            for i, w in enumerate(lay.weight):
                if not w.any().item():
                    lay.bias[i] = 0.
    return model, layers


def _globally_neurons_activation(
        model: TabularNeuralNetTorchModel, layers: List[nn.Module],
        amount, X: pd.DataFrame, grade, subset_features: List[str] = [], **_):
    logger.info(
        f"Globally Pruning Neurons looking for activation, length of subset provided {len(subset_features)=}"
        f" {amount=} and n.{len(layers)=} layers {grade=}")
    if subset_features:
        logger.info("Zeroing inactive features for subset")
        X = X.copy(deep=True)
        X[X.columns.difference(subset_features)] = 0.

    logger.info("Getting neurons' activations")
    activations = _get_neurons_activations(model, layers, X, grade)

    logger.info("Computing indexes of neurons to prune")
    keys = []
    values = np.array([])
    for k, v in activations.items():
        keys += [(k, i) for i in range(len(v))]
        values = np.concatenate((values, v))

    indexes = values.argsort()[:round(amount * len(values))]

    logger.info("Pruning by setting weights and biases to zero")
    for i in indexes:
        lay, ii = keys[i]
        with no_grad():
            lay.weight[ii, :] = 0.
            lay.bias[ii] = 0.
    return model, layers


def _globally_neurons(model: TabularNeuralNetTorchModel, layers: List[nn.Module], amount, grade, **kwargs):
    logger.info(
        f"Globally Pruning Neurons {amount=} and n.{len(layers)=} layers {grade=}")

    if grade != -1:
        logger.info("Computing neurons' score")
        activations: Dict[nn.Module, np.ndarray] = {}

        for lay in layers:
            activations[lay] = np.linalg.norm(lay.weight.detach().numpy(), ord=grade, axis=1)

        logger.info("Computing indexes of neurons to prune")
        keys = []
        values = np.array([])
        for k, v in activations.items():
            keys += [(k, i) for i in range(len(v))]
            values = np.concatenate((values, v))

        indexes = values.argsort()[:round(amount * len(values))]
    else:
        keys = [(lay, i) for lay in layers for i in range(lay.weight.shape[0])]
        indexes = np.random.choice(list(range(len(keys))), round(amount * len(keys)))

    logger.info("Pruning by setting weights and biases to zero")
    for i in indexes:
        lay, ii = keys[i]
        with no_grad():
            lay.weight[ii, :] = 0.
            lay.bias[ii] = 0.
    return model, layers


def globally_connections_l1(model: TabularNeuralNetTorchModel, layers: List[nn.Module], amount, **kwargs):
    return _globally(model, layers, amount, method=prune.L1Unstructured, **kwargs)


def globally_connections_random(model: TabularNeuralNetTorchModel, layers: List[nn.Module], amount, **kwargs):
    return _globally(model, layers, amount, method=prune.RandomUnstructured, **kwargs)


def globally_connections_l2(model: TabularNeuralNetTorchModel, layers: List[nn.Module], amount, **kwargs):
    logger.info(
        f"Globally Pruning Connection {amount=} and n.{len(layers)=} layers")

    logger.info("Computing neurons' score")
    activations: Dict[nn.Module, np.ndarray] = {}

    for lay in layers:
        activations[lay] = np.linalg.norm(lay.weight.detach().numpy(), ord=2, axis=0)

    logger.info("Computing indexes of neurons to prune")
    keys = []
    values = np.array([])
    for k, v in activations.items():
        keys += [(k, i) for i in range(len(v))]
        values = np.concatenate((values, v))

    indexes = values.argsort()[:round(amount * len(values))]
    logger.info("Pruning by setting connections and check biases to zero")
    for i in indexes:
        lay, ii = keys[i]
        with no_grad():
            lay.weight[:, ii] = 0.

    with no_grad():
        for lay in layers:
            for i, w in enumerate(lay.weight):
                if not w.any().item():
                    lay.bias[i] = 0.
    return model, layers


def globally_neurons_l1(model: TabularNeuralNetTorchModel, layers: List[nn.Module], amount, **kwargs):
    return _globally_neurons(model, layers, amount, grade=1, **kwargs)


def globally_neurons_l2(model: TabularNeuralNetTorchModel, layers: List[nn.Module], amount, **kwargs):
    return _globally_neurons(model, layers, amount, grade=2, **kwargs)


def globally_neurons_random(model: TabularNeuralNetTorchModel, layers: List[nn.Module], amount, **kwargs):
    return _globally_neurons(model, layers, amount, grade=-1, **kwargs)


def globally_neurons_activation_l1(
        model: TabularNeuralNetTorchModel, layers: List[nn.Module],
        amount, X: pd.DataFrame, subset_features: List[str] = [], **_):
    return _globally_neurons_activation(model, layers, amount, X, grade=1, subset_features=subset_features)


def globally_neurons_activation_l2(
        model: TabularNeuralNetTorchModel, layers: List[nn.Module],
        amount, X: pd.DataFrame, subset_features: List[str] = [], **_):
    return _globally_neurons_activation(model, layers, amount, X, grade=2, subset_features=subset_features)


def globally_neurons_activation_l1_for_subset(
        model: TabularNeuralNetTorchModel, layers: List[nn.Module],
        amount, X: pd.DataFrame, subset_features: List[str], **kwargs):
    return _globally_neurons_activation(model, layers, amount, X, grade=1, subset_features=subset_features, **kwargs)


def globally_neurons_activation_l2_for_subset(
        model: TabularNeuralNetTorchModel, layers: List[nn.Module],
        amount, X: pd.DataFrame, subset_features: List[str], **kwargs):
    return _globally_neurons_activation(model, layers, amount, X, grade=2, subset_features=subset_features, **kwargs)
