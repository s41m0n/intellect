import numpy as np
from sklearn.utils.validation import check_is_fitted

from .model import EnhancedMlp


def distance_l2(model: EnhancedMlp, other: EnhancedMlp, only_prunable: bool = True) -> float:
    """Return the norm 2 difference between the two model parameters

    Args:
        model (EnhancedMlp): one model
        other (EnhancedMlp): the other model
        only_prunable (bool, optional): whether to consider only prunable layers.
            Defaults to True.

    Returns:
        float: the l2 norm
    """
    t_own = model.prunable if only_prunable else model.parameters()
    t_other = other.prunable if only_prunable else other.parameters()
    ret = 0
    for i_lay_lvm, ii_lay_brm in zip(t_own, t_other):
        ret += np.sum((other.coefs_[ii_lay_brm] - model.coefs_[i_lay_lvm])**2).item()
        ret += np.sum((other.intercepts_[ii_lay_brm] - model.intercepts_[i_lay_lvm])**2).item()
    return ret


def sparsity(model: EnhancedMlp) -> tuple[float, list[int, float]]:
    """Function to compute the sparsity of a network

    Args:
        model (EnhancedMlp): target network

    Returns:
        tuple[float, list[int, float]]: tuple with global and per-layer sparsity
    """
    if model.prune_masks is None:
        return 0, []
    single = [np.sum(model.intercepts_[i] == 0) + np.sum(k == 0) / k.size for i, k in enumerate(model.prune_masks)]
    return np.mean(single), single


def prune_unstructured_connections_l1(model: EnhancedMlp, prune_ratio: float) -> EnhancedMlp:
    """Function to prune CONNECTION-UNSTRUCTURED with L1 norm

    Args:
        model (EnhancedMlp): model to be pruned
        prune_ratio (float): prune ratio between 0 and 1

    Returns:
        EnhancedMlp: the pruned model
    """
    check_is_fitted(model, msg="Model not fitted, fit before pruning")
    new_model = model.clone(init=False)
    new_model.prune_masks = [np.ones_like(k) for k in new_model.coefs_]
    all_weights = np.concatenate([np.asarray(c).reshape(-1) for c in new_model.coefs_], axis=0)
    k = round(len(all_weights) * prune_ratio)
    all_weights = np.absolute(all_weights)
    idx = all_weights.argsort()[:k]
    mask = np.ones_like(all_weights)
    mask[idx] = 0
    pointer = 0
    for i, v in enumerate(new_model.coefs_):
        num_param = v.size
        new_model.prune_masks[i] = mask[pointer: pointer + num_param].reshape(v.shape)
        pointer += num_param
    return new_model
