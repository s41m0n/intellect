from collections.abc import Iterable

import pandas as pd
import torch
import torch.nn.utils.prune as prune
from torch import nn

from . import TorchModel

PRUNE_CONNECTIONS_DIM = 1
PRUNE_NEURONS_DIM = 0
RANDOM = 0
L1_NORM = 1
L2_NORM = 2

###################################
# LOCAL PRUNING
###################################


def _helper_locally(
        model: TorchModel, amount, n=L1_NORM, dim=PRUNE_NEURONS_DIM, structured=False, name="weight", **_):
    layers = model.prunable
    # dim=0 prune neuron, dim=1 prune connection.
    # when called with dim=0 it produces the same output of unstructured
    if structured:
        if n == RANDOM:
            [prune.random_structured(k, name, amount=v, dim=dim) for k, v in zip(layers, amount)]
        elif n == L1_NORM or n == L2_NORM:
            [prune.ln_structured(k, name, amount=v, n=n, dim=dim) for k, v in zip(layers, amount)]
        else:
            raise NotImplementedError(f"Norm n {n} not implemented")
    else:
        if n == RANDOM:
            [prune.random_unstructured(k, name, amount=v) for k, v in zip(layers, amount)]
        elif n == L1_NORM:
            [prune.l1_unstructured(k, name, amount=v) for k, v in zip(layers, amount)]
        else:
            raise NotImplementedError(f"Norm n {n} not implemented")


def _helper_locally_structured_neurons_activation(
        model: TorchModel, amount, X: pd.DataFrame, n=L1_NORM, name="weight", subset_features: list[str] = [], **_):
    layers = model.prunable
    if subset_features:
        X = X.copy(deep=True)
        X[X.columns.difference(subset_features)] = 0.

    activations = model.get_neurons_activation(layers, X, n, shape_fill=name)

    for lay, a in activations.items():
        prune.ln_structured(lay, name=name, amount=amount, n=n, dim=PRUNE_NEURONS_DIM, importance_scores=a)


def locally_structured_neurons_l1(model: nn.Module, amount, **kwargs):
    return _helper_locally(model, amount, n=L1_NORM, dim=PRUNE_NEURONS_DIM, structured=True, **kwargs)


def locally_structured_neurons_l2(model: nn.Module, amount, **kwargs):
    return _helper_locally(model, amount, n=L2_NORM, dim=PRUNE_NEURONS_DIM, structured=True ** kwargs)


def locally_structured_neurons_random(model: nn.Module, amount, **kwargs):
    return _helper_locally(model, amount, n=RANDOM, dim=PRUNE_NEURONS_DIM, structured=True, **kwargs)


def locally_structured_connections_l1(model: nn.Module, amount, **kwargs):
    return _helper_locally(model, amount, n=L1_NORM, dim=PRUNE_CONNECTIONS_DIM, structured=True, **kwargs)


def locally_structured_connections_l2(model: nn.Module, amount, **kwargs):
    return _helper_locally(model, amount, n=L2_NORM, dim=PRUNE_CONNECTIONS_DIM, structured=True ** kwargs)


def locally_structured_connections_random(model: nn.Module, amount, **kwargs):
    return _helper_locally(model, amount, n=RANDOM, dim=PRUNE_CONNECTIONS_DIM, structured=True, **kwargs)


def locally_unstructured_connections_l1(model: nn.Module, amount, **kwargs):
    return _helper_locally(model, amount, n=L1_NORM, dim=PRUNE_CONNECTIONS_DIM, structured=False, **kwargs)


def locally_unstructured_connections_random(model: nn.Module, amount, **kwargs):
    return _helper_locally(model, amount, n=RANDOM, dim=PRUNE_CONNECTIONS_DIM, structured=False, **kwargs)


def locally_structured_neurons_activation_l1(
        model: nn.Module, amount, X: pd.DataFrame, subset_features: list[str] = [], **_):
    return _helper_locally_structured_neurons_activation(
        model, amount, X, n=L1_NORM, subset_features=subset_features)


def locally_structured_neurons_activation_l2(
        model: nn.Module, amount, X: pd.DataFrame, subset_features: list[str] = [], **_):
    return _helper_locally_structured_neurons_activation(
        model, amount, X, n=L2_NORM, subset_features=subset_features)


def locally_structured_neurons_activation_l1_for_subset(
        model: nn.Module, amount, X: pd.DataFrame, subset_features: list[str], **kwargs):
    return _helper_locally_structured_neurons_activation(
        model, amount, X, n=L1_NORM, subset_features=subset_features, **kwargs)


def locally_structured_neurons_activation_l2_for_subset(
        model: nn.Module, amount, X: pd.DataFrame, subset_features: list[str], **kwargs):
    return _helper_locally_structured_neurons_activation(
        model, amount, X, n=L2_NORM, subset_features=subset_features, **kwargs)


###################################
# GLOBAL PRUNING
###################################

def _torch_missing_globally_structured(
        parameters, pruning_method, amount, dim=PRUNE_NEURONS_DIM, importance_scores=None, **kwargs):
    if not isinstance(parameters, Iterable):
        raise TypeError("global_unstructured(): parameters is not an Iterable")

    importance_scores = importance_scores if importance_scores is not None else {}
    if not isinstance(importance_scores, dict):
        raise TypeError("global_unstructured(): importance_scores must be of type dict")

    to_look = int(not dim)

    def find_max_shape():
        max_dim = 0
        for (module, name) in parameters:
            v = importance_scores.get((module, name), getattr(module, name))
            max_dim = max(max_dim, v.shape[to_look])
        return max_dim

    max_dim = find_max_shape()

    def tensor_to_shape(t: torch.Tensor):
        new_shape = list(t.shape)
        new_shape[to_look] = max_dim
        new_shape = tuple(new_shape)
        tmp = torch.zeros(new_shape)
        tmp[:t.shape[0], :t.shape[1]] = t
        return tmp

    relevant_importance_scores = torch.cat(
        [
            tensor_to_shape(importance_scores.get((module, name), getattr(module, name)))
            for (module, name) in parameters
        ], dim
    )

    default_mask = torch.cat(
        [
            tensor_to_shape(getattr(module, name + "_mask", torch.ones_like(getattr(module, name))))
            for (module, name) in parameters
        ], dim
    )

    container = prune.PruningContainer()
    container._tensor_name = "temp"
    method = pruning_method(amount, dim=dim, **kwargs)
    method._tensor_name = "temp"
    if method.PRUNING_TYPE != "structured":
        raise TypeError(
            'Only "structured" PRUNING_TYPE supported for '
            "the `pruning_method`. Found method {} of type {}".format(
                pruning_method, method.PRUNING_TYPE
            )
        )

    container.add_pruning_method(method)

    final_mask = container.compute_mask(relevant_importance_scores, default_mask)
    pointer = 0
    for module, name in parameters:
        param = getattr(module, name)
        if dim == 0:
            param_mask = final_mask[pointer: pointer + param.shape[0], :param.shape[1]]
        if dim == 1:
            param_mask = final_mask[:param.shape[0], pointer: pointer + param.shape[1]]
        pointer += param.shape[dim]
        prune.custom_from_mask(module, name, mask=param_mask)


def _helper_globally(
        model: TorchModel, amount, method=prune.L1Unstructured, dim=PRUNE_CONNECTIONS_DIM, name="weight", **kwargs):
    layers = model.prunable
    parameters = {(x, name) for x in layers}

    if method.PRUNING_TYPE == "structured":
        return _torch_missing_globally_structured(parameters, method, amount, dim=dim, **kwargs)
    return prune.global_unstructured(parameters,
                                     pruning_method=method,
                                     amount=amount)


def _helper_globally_structured_neurons_activation(
        model: TorchModel, amount, X: pd.DataFrame, n, subset_features: list[str] = [], name="weight", **_):
    layers = model.prunable
    if subset_features:
        X = X.copy(deep=True)
        X[X.columns.difference(subset_features)] = 0.

    activations = model.get_neurons_activation(layers, X, n, shape_fill=name)

    return _helper_globally(model, layers, amount, method=prune.LnStructured, dim=PRUNE_NEURONS_DIM,
                            n=n, importance_scores=activations)


def globally_structured_neurons_l1(model: nn.Module, amount, **kwargs):
    return _helper_globally(
        model, amount, method=prune.LnStructured, n=L1_NORM, dim=PRUNE_NEURONS_DIM, **kwargs)


def globally_structured_neurons_l2(model: nn.Module, amount, **kwargs):
    return _helper_globally(
        model, amount, method=prune.LnStructured, n=L2_NORM, dim=PRUNE_NEURONS_DIM, **kwargs)


def globally_structured_neurons_random(
        model: nn.Module, amount, **kwargs):
    return _helper_globally(
        model, amount, method=prune.RandomStructured, dim=PRUNE_NEURONS_DIM, **kwargs)


def globally_structured_connections_l1(model: nn.Module, amount, **_):
    return _helper_globally(
        model, amount, method=prune.LnStructured, n=L1_NORM, dim=PRUNE_CONNECTIONS_DIM)


def globally_structured_connections_l2(model: nn.Module, amount, **_):
    return _helper_globally(
        model, amount, method=prune.LnStructured, n=L2_NORM, dim=PRUNE_CONNECTIONS_DIM)


def globally_structured_connections_random(model: nn.Module, amount, **_):
    return _helper_globally(
        model, amount, method=prune.LnStructured, n=RANDOM, dim=PRUNE_CONNECTIONS_DIM)


def globally_unstructured_connections_l1(model: nn.Module, amount, **kwargs):
    return _helper_globally(
        model, amount, method=prune.L1Unstructured, n=L1_NORM, dim=PRUNE_CONNECTIONS_DIM, **kwargs)


def globally_unstructured_connections_random(
        model: nn.Module, amount, **kwargs):
    return _helper_globally(
        model, amount, method=prune.RandomUnstructured, n=RANDOM, dim=PRUNE_CONNECTIONS_DIM, **kwargs)


def globally_structured_neurons_activation_l1(
        model: nn.Module, amount, X: pd.DataFrame, **_):
    return _helper_globally_structured_neurons_activation(
        model, amount, X, n=L1_NORM)


def globally_structured_neurons_activation_l2(
        model: nn.Module, amount, X: pd.DataFrame, **_):
    return _helper_globally_structured_neurons_activation(
        model, amount, X, n=L2_NORM)


def globally_structured_neurons_activation_l1_for_subset(
        model: nn.Module, amount, X: pd.DataFrame, subset_features: list[str], **kwargs):
    return _helper_globally_structured_neurons_activation(
        model, amount, X, n=L1_NORM, subset_features=subset_features, **kwargs)


def globally_structured_neurons_activation_l2_for_subset(
        model: nn.Module, amount, X: pd.DataFrame, subset_features: list[str], **kwargs):
    return _helper_globally_structured_neurons_activation(
        model, amount, X, n=L2_NORM, subset_features=subset_features, **kwargs)
