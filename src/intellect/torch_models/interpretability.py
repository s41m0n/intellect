import numpy as np
import torch
from captum.attr import (IntegratedGradients, LayerActivation,
                         LayerConductance, NeuronConductance, Occlusion)

from ..model import MyDataset
from . import TorchModel


def get_neurons_activation(model: TorchModel, data: MyDataset, only_prunable=True, **kwargs):
    """https://captum.ai/tutorials/Titanic_Basic_Interpret"""
    model.cuda()
    layers = model.prunable if only_prunable else tuple(model.layers.children())
    ig = LayerActivation(model, layers)
    X = model.safe_cast_input(data.X)
    X.requires_grad_()
    attr = ig.attribute(X)
    model.cpu()
    return dict(zip(layers, attr))


def get_neurons_importance(model: TorchModel, data: MyDataset, only_prunable=True, n_steps=1, **kwargs):
    model.cuda()
    layers = model.prunable if only_prunable else tuple(model.layers.children())
    X = model.safe_cast_input(data.X)
    ret = {}
    for l in layers:
        cond = LayerConductance(model, l)
        ret[l] = np.mean(cond.attribute(X, target=1, n_steps=n_steps).detach().cpu().numpy(), axis=0)
    model.cpu()
    return ret


def get_learned_features_per_neuron(model: TorchModel, data: MyDataset, only_prunable=True, n_steps=2, **kwargs):
    model.cuda()
    layers = model.prunable if only_prunable else tuple(model.layers.children())
    X = model.safe_cast_input(data.X)
    ret = {}
    for l in layers:
        neuron_cond = NeuronConductance(model, l)
        ret[l] = np.mean(neuron_cond.attribute(X, neuron_selector=1,
                         target=1, n_steps=n_steps).detach().cpu().numpy(), axis=0)
    model.cpu()
    return ret


def rank_gradient_captum(model: TorchModel, data: MyDataset, current_features, n_steps=1, **_):
    """Compute feature importance with integrated gradient method using captum"""
    model.cuda()
    positions = [i for i, x in enumerate(data.X.columns.values) if x in current_features]
    ig = IntegratedGradients(model)
    X = model.safe_cast_input(data.X)
    X.requires_grad_()
    attr = ig.attribute(X, target=1, n_steps=n_steps)
    attr = attr.detach().numpy()
    importances = np.mean(attr, axis=0)
    model.cpu()
    return dict(zip(current_features, [importances[i] for i in positions]))


def rank_perturbation_captum(model: TorchModel, data: MyDataset, current_features, **_):
    model.cuda()
    positions = [i for i, x in enumerate(data.X.columns.values) if x in current_features]
    occlusion = Occlusion(model)

    attributions_occ = occlusion.attribute(torch.tensor(data.X.values, device=model.device),
                                           target=1,
                                           sliding_window_shapes=(1,),
                                           baselines=0)
    attr = attributions_occ.detach().cpu().numpy()
    importances = np.mean(attr, axis=0)
    model.cpu()
    return dict(zip(current_features, [importances[i] for i in positions]))
