from abc import abstractmethod
from copy import deepcopy
from numbers import Real
from typing import Literal

import numpy as np
import sklearn
from numpy import ndarray
from numpy.random import RandomState
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neural_network._base import (ACTIVATIONS, DERIVATIVES,
                                          LOSS_FUNCTIONS)
from sklearn.neural_network._multilayer_perceptron import \
    BaseMultilayerPerceptron
from sklearn.utils._param_validation import Interval
from sklearn.utils.extmath import safe_sparse_dot

from ..base import BaseModel


class EnhancedMlp(BaseMultilayerPerceptron, BaseModel):
    _parameter_constraints: dict = {
        **BaseMultilayerPerceptron._parameter_constraints,
        "dropout": [Interval(Real, 0, 1, closed="left")],
        "prune_masks": [np.ndarray, list, None]
    }

    @abstractmethod
    def _parse_ypred(self, y):
        ...

    def __init__(
            self, dropout: float = 0., prune_masks: list[np.ndarray] = None, hidden_layer_sizes=...,
            activation: Literal['relu', 'identity', 'logistic', 'tanh'] = "relu", *,
            solver: Literal['lbfgs', 'sgd', 'adam'] = "adam", alpha: float = 0.0001,
            batch_size: int | str = "auto", learning_rate_init: float = 0.001,
            learning_rate: Literal['constant', 'invscaling', 'adaptive'] = "constant",
            power_t: float = 0.5, max_iter: int = 200, loss=None, shuffle: bool = True,
            random_state: int | RandomState | None = None, tol: float = 0.0001, verbose: bool = False,
            warm_start: bool = False, momentum: float = 0.9, nesterovs_momentum: bool = True,
            early_stopping: bool = False, validation_fraction: float = 0.1, beta_1:
            float = 0.9, beta_2: float = 0.999, epsilon: float = 1e-8, n_iter_no_change: int = 10,
            max_fun: int = 15000):
        super().__init__(
            hidden_layer_sizes, activation, solver=solver, alpha=alpha, batch_size=batch_size,
            learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter,
            loss=loss, shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose, warm_start=warm_start,
            momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping,
            validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
            n_iter_no_change=n_iter_no_change, max_fun=max_fun)
        self.dropout = dropout
        self.prune_masks = prune_masks

    def _compute_dropout_masks(self, layer_units):
        dropout_masks = None

        # Create the Dropout Mask (DROPOUT ADDITION)
        if self.dropout not in (0, None):
            if not 0 < self.dropout < 1:
                raise ValueError('Dropout must be between zero and one. If Dropout=X then, 0 < X < 1.')
            keep_probability = 1 - self.dropout
            dropout_masks = [np.ones(layer_units[0])]

            # Create hidden Layer Dropout Masks
            for units in layer_units[1:-1]:
                # Create inverted Dropout Mask, check for random_state
                if self.random_state is not None:
                    layer_mask = self._random_state.random(units) < keep_probability
                    layer_mask = layer_mask.astype(int) / keep_probability
                else:
                    layer_mask = (np.random.rand(units) < keep_probability).astype(int) / keep_probability
                dropout_masks.append(layer_mask)
        return dropout_masks

    def _backprop(self, X, y, activations, deltas, coef_grads, intercept_grads,):
        n_samples = X.shape[0]
        layer_units = [X.shape[1]] + list(self.hidden_layer_sizes) + [self.n_outputs_]

        dropout_masks = self._compute_dropout_masks(layer_units)

        # Forward propagate
        activations = self._forward_pass(activations, dropout_masks=dropout_masks)

        # Get loss
        loss_func_name = self.loss
        if loss_func_name == "log_loss" and self.out_activation_ == "logistic":
            loss_func_name = "binary_log_loss"
        loss = LOSS_FUNCTIONS[loss_func_name](y, activations[-1])
        # Add L2 regularization term to loss
        values = 0
        for s in self.coefs_:
            s = s.ravel()
            values += np.dot(s, s)
        loss += (0.5 * self.alpha) * values / n_samples

        # Backward propagate
        last = self.n_layers_ - 2

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss
        deltas[last] = activations[-1] - y

        # Compute gradient for the last layer
        self._compute_loss_grad(
            last, n_samples, activations, deltas, coef_grads, intercept_grads
        )

        inplace_derivative = DERIVATIVES[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative(activations[i], deltas[i - 1])

            self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads, intercept_grads
            )

        # Apply Dropout Masks to the Parameter Gradients (DROPOUT ADDITION)
        if dropout_masks is not None:
            for layer in range(len(coef_grads) - 1):
                mask = (~(dropout_masks[layer + 1] == 0)).astype(int)
                coef_grads[layer] = coef_grads[layer] * mask[None, :]
                coef_grads[layer + 1] = coef_grads[layer + 1] * mask.reshape(-1, 1)
                intercept_grads[layer] = intercept_grads[layer] * mask
        return loss, coef_grads, intercept_grads

    def _forward_pass(self, activations, dropout_masks=None):
        hidden_activation = ACTIVATIONS[self.activation]
        for i in range(self.n_layers_ - 1):
            tmp_coef = self.coefs_[i]
            if self.prune_masks:
                tmp_coef *= self.prune_masks[i]
            activations[i + 1] = safe_sparse_dot(activations[i], tmp_coef)
            activations[i + 1] += self.intercepts_[i]
            if (i + 1) != (self.n_layers_ - 1):
                if dropout_masks is not None:
                    activations[i + 1] = activations[i + 1] * dropout_masks[i + 1][None, :]
                hidden_activation(activations[i + 1])
        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activations[i + 1])
        return activations

    def _forward_pass_fast(self, X, check_input=True):
        if check_input:
            X = self._validate_data(X, accept_sparse=["csr", "csc"], reset=False)
        activation = X
        hidden_activation = ACTIVATIONS[self.activation]
        for i in range(self.n_layers_ - 1):
            tmp_coef = self.coefs_[i]
            if self.prune_masks:
                tmp_coef *= self.prune_masks[i]
            activation = safe_sparse_dot(activation, tmp_coef)
            activation += self.intercepts_[i]
            if i != self.n_layers_ - 2:
                hidden_activation(activation)
        output_activation = ACTIVATIONS[self.out_activation_]
        output_activation(activation)
        return activation

    def clone(self, init=True, copy_prune=True):
        if not init:
            new = deepcopy(self)
        else:
            new = sklearn.base.clone(self, safe=True)
        if copy_prune:
            new.copy_prune(self)
        return new

    def copy_prune(self, other: BaseModel):
        self.prune_masks = other.prune_masks

    def predict(self, X, *args, **kwargs) -> ndarray:
        return self._parse_ypred(super().predict(X))

    def concept_react(self, *args, **kwargs):
        raise NotImplementedError()

    def learn(self, *args, **kwargs):
        self.partial_fit(*args, **kwargs)

    def continuous_learning(self, *args, **kwargs):
        raise NotImplementedError()

    @property
    def is_concept_drift(self, *args, **kwargs):
        raise NotImplementedError()


class EnhancedMlpClassifier(EnhancedMlp, MLPClassifier):
    @property
    def prunable(self):
        return tuple(i for i in range(len(self.coefs_)))

    def _parse_ypred(self, y):
        return np.argmax(y, axis=1)


class EnhancedMlpRegressor(EnhancedMlp, MLPRegressor):
    @property
    def prunable(self):
        return tuple(i for i in range(len(self.coefs_)))

    def predict_proba(self, X, *args, **kwargs):
        return MLPRegressor.predict(self, X)

    def _parse_ypred(self, y):
        return np.round(y)
