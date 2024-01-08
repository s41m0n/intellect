import numpy as np
from sklearn.utils.validation import check_is_fitted

from .model import EnhancedMlp


def distance(model: EnhancedMlp, other: EnhancedMlp, only_prunable=True, method="L2_NORM"):
    """Return the norm 2 difference between the two model parameters
    """
    if method == "L2_NORM":
        t_own = model.prunable if only_prunable else model.parameters()
        t_other = other.prunable if only_prunable else other.parameters()
        ret = 0
        for i_lay_lvm, ii_lay_brm in zip(t_own, t_other):
            ret += np.sum((other.coefs_[ii_lay_brm] - model.coefs_[i_lay_lvm])**2).item()
            ret += np.sum((other.intercepts_[ii_lay_brm] - model.intercepts_[i_lay_lvm])**2).item()
        return ret
    raise ValueError()


def sparsity(model: EnhancedMlp):
    return [np.sum(k == 0) / k.size for k in model.prune_masks]


def prune_unstructured_connections_l1(model: EnhancedMlp, amount) -> EnhancedMlp:
    check_is_fitted(model, msg="Model not fitted, fit before pruning")
    new_model = model.clone(init=False)
    new_model.prune_masks = [np.ones_like(k) for k in new_model.coefs_]
    all_weights = np.concatenate([np.asarray(c).reshape(-1) for c in new_model.coefs_], axis=0)
    k = round(len(all_weights) * amount)
    all_weights = np.absolute(all_weights)
    idx = all_weights.argsort()[:k]
    mask = np.ones_like(all_weights)
    mask[idx] = 0
    pointer = 0
    for i in range(len(new_model.coefs_)):
        num_param = new_model.coefs_[i].size
        new_model.prune_masks[i] = mask[pointer: pointer + num_param].reshape(new_model.coefs_[i].shape)
        pointer += num_param
    return new_model
