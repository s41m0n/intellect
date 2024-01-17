
import inspect
import io
import os
import sys
from abc import abstractmethod
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.nn import KLDivLoss
from torch.nn.functional import log_softmax, softmax
from torch.nn.utils import prune
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...dataset import (ContinuouLearningAlgorithm, Dataset,
                        FeatureAvailability, InputForLearn)
from ..base import BaseModel


class TorchModel(BaseModel):

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        super().__init__(kwargs.get('drift_detector', None))
        self.init_params: dict[str, object] = kwargs
        self.model: torch.nn.Sequential = torch.nn.Sequential()
        self.optimizer: torch.optim.AdamW = None
        self.loss_fn: torch.nn.CrossEntropyLoss = None

    @abstractmethod
    def _parse_ypred(self, y):
        ...

    @property
    def is_autoencoder(self):
        return len(self.init_params['in_features']) == self.init_params['n_outputs']

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def xtype(self):
        return next(self.model.parameters()).dtype

    def copy_prune(self, other: 'TorchModel'):
        for l1, l2 in zip(self.model.children(), other.model.children()):
            for param in ('weight', 'bias'):
                if not hasattr(l2, f'{param}_mask'):
                    continue
                prune.custom_from_mask(l1, param, getattr(l2, f'{param}_mask').detach())

    def clone(self, init=True):
        if init is False:
            buffer = io.BytesIO()
            torch.save({'init_params': self.init_params, 'model': self.model,
                       'drift_detector': self.drift_detector}, buffer)
            buffer.seek(0)
            tmp = torch.load(buffer)
            ret = self.__class__()
            for k, v in tmp.items():
                setattr(ret, k, v)
            return ret
        new = self.__class__(**self.init_params)
        new.copy_prune(self)
        return new

    def save(self, path: str):
        if not path.endswith('.pt'):
            path += '.pt'
        with open(path, 'wb') as fp:
            torch.save({'init_params': self.init_params, 'model': self.model,
                       'drift_detector': self.drift_detector}, fp)

    @classmethod
    def load(cls, path: str) -> 'TorchModel':
        if not path.endswith('.pt'):
            path += '.pt'
        try:
            tmp = torch.load(path)
        except RuntimeError:
            tmp = torch.load(path, map_location='cpu')
        new = cls()
        for k, v in tmp.items():
            setattr(new, k, v)
        new.model.eval()
        return new

    def safe_cast_input(self, x: torch.Tensor | dict | pd.DataFrame | np.ndarray, is_y=False):
        ret: torch.Tensor = None
        dtype = torch.long if is_y and not self.is_autoencoder else self.xtype

        if isinstance(x, torch.Tensor):
            ret = x
        elif isinstance(x, dict):
            ret = torch.tensor([0 if k not in x else x[k]
                               for k in self.init_params['in_features']])
        elif isinstance(x, (pd.DataFrame, pd.Series)):
            ret = torch.from_numpy(x.values)
        elif isinstance(x, np.ndarray):
            ret = torch.from_numpy(x)
        else:
            return torch.tensor([x], dtype=dtype, device=self.device)
        if ret.ndim == 1 and not is_y:
            ret = ret[None, :]
        return ret.to(self.device, dtype=dtype)

    def predict(self, X: torch.Tensor, *args, **kwargs):
        return self._parse_ypred(self.predict_proba(X, *args, **kwargs))

    def predict_proba(self, X: torch.Tensor, *args, **kwargs):
        as_dict = kwargs.get('as_dict', False)

        with torch.no_grad():
            y = self.model(self.safe_cast_input(X)).detach()
        if self.model[-1].__class__.__name__ == 'LogSoftmax':
            y = torch.exp(y)
        y = y.numpy()
        if as_dict:
            return [{i: v.item() for i, v in enumerate(j)} for j in y]
        return y

    def fit(self, train_dataset: Dataset, validation_dataset: Dataset | float = 0.2, batch_size: int = 256,
            max_epochs: int = 100, epochs_wo_improve: int = None, metric=accuracy_score, shuffle=True,
            optimizer: torch.optim.Optimizer = None, loss_fn: torch.nn.modules.loss._Loss = None,
            higher_better=True, **kwargs):

        if torch.cuda.is_available():
            self.model.cuda()

        if metric is None:
            higher_better = False

        best_metric = 0 if higher_better else sys.maxsize
        current_epochs_wo_improve = 0
        history = {'loss_train': []}
        if metric is not None:
            history[f'{metric.__name__}_train'] = []

        best_state_dict = deepcopy(self.model.state_dict())
        self.check_optim_loss(optimizer, loss_fn)

        if validation_dataset is not None and validation_dataset != 0:
            if isinstance(validation_dataset, float):
                validation_dataset = train_dataset.sample(validation_dataset, by_category=True)
                train_dataset = train_dataset.filter_indexes(validation_dataset.X.index.values)
            X_val = self.safe_cast_input(validation_dataset.X)
            if self.is_autoencoder:
                y_val = X_val
            else:
                y_val = self.safe_cast_input(validation_dataset.y, is_y=True)
            if metric is not None:
                history[f'{metric.__name__}_validation'] = []
            history['loss_validation'] = []

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=os.cpu_count(), pin_memory=True, persistent_workers=False)

        with tqdm(range(max_epochs)) as pbar:
            for i in range(max_epochs):
                training_loss = 0.0
                training_pred = None
                training_true = None

                for _, (inputs, labels) in enumerate(train_loader):
                    if self.is_autoencoder:
                        labels = inputs
                    predictions, loss = self.learn(inputs, labels)
                    training_loss += loss
                    training_pred = predictions if training_pred is None else np.concatenate(
                        (training_pred, predictions), axis=0)
                    training_true = labels if training_true is None else np.concatenate((training_true, labels), axis=0)

                training_loss /= len(train_loader)

                metric_train = training_loss
                if metric is not None:
                    metric_train = metric(training_true, self._parse_ypred(training_pred))
                    history[f'{metric.__name__}_train'].append(metric_train)

                history['loss_train'].append(training_loss)

                if validation_dataset is not None and validation_dataset != 0:
                    with torch.no_grad():
                        y_raw_val = self.model(X_val).detach()
                        validation_loss = loss_fn(y_raw_val, y_val).item()
                    history['loss_validation'].append(validation_loss)
                    metric_validation = validation_loss
                    if metric is not None:
                        metric_validation = metric(y_val, self._parse_ypred(y_raw_val))
                        history[f'{metric.__name__}_validation'].append(metric_validation)

                pbar.set_description(f'Epoch {i+1} {dict((k, v[-1]) for k, v in history.items())}', refresh=False)
                pbar.update()

                if not epochs_wo_improve:
                    continue

                eval_metric = metric_validation if (
                    validation_dataset is not None and validation_dataset != 0) else metric_train
                cond = eval_metric > best_metric if higher_better else eval_metric < best_metric
                if cond:
                    current_epochs_wo_improve = 0
                    best_metric = eval_metric
                    best_state_dict = deepcopy(self.model.state_dict())
                else:
                    current_epochs_wo_improve += 1

                if current_epochs_wo_improve == epochs_wo_improve:
                    break
            self.model.load_state_dict(best_state_dict)

        self.model.cpu()
        return history

    def continuous_learning(
            self, data: Dataset, algorithm: ContinuouLearningAlgorithm, oracle: BaseModel = None, epochs=1,
            batch_size='auto', features_available: list[str] = None,
            availability: FeatureAvailability = FeatureAvailability.bilateral,
            learn_input=InputForLearn.client, oracle_input=InputForLearn.oracle, drop_ratio=0.,
            optimizer: torch.optim.Optimizer = None, weight_decay: float = 0.,
            loss_fn: torch.nn.modules.loss._Loss = None, n_verbosity=1, **kwargs):
        if torch.cuda.is_available():
            self.model.cuda()
            if oracle:
                oracle.cuda()

        idx, idx_oracle = [], []
        if features_available is not None and len(features_available):
            if availability.value == FeatureAvailability.oracle.value:
                idx = data.filter_features(features_available, get_idx=True)
            elif availability.value == FeatureAvailability.client.value:  # strange use case
                idx_oracle = data.filter_features(features_available, get_idx=True)
            elif availability.value == FeatureAvailability.none.value:
                idx = idx_oracle = data.filter_features(features_available, get_idx=True)
            elif availability.value not in (None, FeatureAvailability.bilateral.value):
                raise NotImplementedError('Daje')

        self.check_optim_loss(optimizer, loss_fn, weight_decay=weight_decay)

        loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(),
                            pin_memory=True, persistent_workers=False)

        y_preds = np.empty(0)
        drifts = np.empty(0)

        with tqdm(range(len(loader))) as pbar:
            for ii, (inputs, true_labels) in enumerate(loader):
                inputs_oracle = inputs.detach().clone()

                if oracle_input.value == InputForLearn.oracle.value:
                    inputs_oracle[:, idx_oracle] = 0.
                elif oracle_input.value == InputForLearn.client.value:
                    inputs_oracle[:, idx] = 0.
                else:
                    raise ValueError(f'Unknown oracle input {oracle_input} {oracle_input.value}')

                if learn_input.value == InputForLearn.oracle.value:
                    inputs[:, idx_oracle] = 0.
                elif learn_input.value == InputForLearn.client.value:
                    inputs[:, idx] = 0.
                else:
                    raise ValueError(f'Unknown learn input {learn_input} {learn_input.value}')

                predicted = self.predict(inputs)
                if drop_ratio != 0:
                    indexes = np.random.choice(inputs.shape[0], size=int(
                        (1-drop_ratio) * inputs.shape[0]), replace=False)
                    inputs = inputs[indexes]
                    inputs_oracle = inputs_oracle[indexes]

                for i in range(epochs):

                    if algorithm.value == ContinuouLearningAlgorithm.ground_inferred.value:
                        inferred_labels = oracle.predict(inputs_oracle)
                        _, loss = self.learn(inputs, inferred_labels)
                    elif algorithm.value == ContinuouLearningAlgorithm.ground_truth.value:
                        _, loss = self.learn(inputs, true_labels)
                    elif algorithm.value == ContinuouLearningAlgorithm.knowledge_distillation.value:
                        _, loss = self.learn_knowledge_distillation(
                            oracle, inputs, inputs_oracle, true_labels, **kwargs)
                    elif algorithm.value == ContinuouLearningAlgorithm.none.value:
                        loss = torch.tensor(np.nan)
                    else:
                        raise ValueError(f'Unknown algorithm {algorithm} {algorithm.value}')

                    if ii % n_verbosity == 0:
                        pbar.set_description(f'Batch {ii+1} Epoch {i+1} {loss}', refresh=False)
                        pbar.update(n_verbosity)

                if self.drift_detector is not None:
                    predicted_after = self.predict(inputs)
                    labels = true_labels if (
                        algorithm.value != ContinuouLearningAlgorithm.ground_inferred) else inferred_labels
                    for j, (vt, vp) in enumerate(zip(labels, predicted_after)):
                        self.drift_detector.update(int(not vt == vp))
                        if self.is_concept_drift:
                            drifts = np.append(drifts, len(y_preds) + j)
                            self.concept_react()

                y_preds = np.concatenate((y_preds, self._parse_ypred(predicted)), axis=0)

        if torch.cuda.is_available():
            self.model.cpu()
            if oracle:
                oracle.model.cpu()
        return y_preds, data.y.to_numpy(), data._y, drifts

    @property
    def prunable(self) -> list[torch.nn.Module]:
        """Property to return the list of prunable modules within the model

        Returns:
            list[torch.nn.Module]: list of prunable modules
        """
        return tuple(m for m in self.model.children() if isinstance(m, torch.nn.Linear))

    def learn_knowledge_distillation(
            self, oracle: BaseModel, inputs, inputs_oracle, true_labels, temperature: int = 2,
            alpha: float = 0.25):
        if self.optimizer is None or self.loss_fn is None:
            self.check_optim_loss(self.optimizer, self.loss_fn)

        inputs = self.safe_cast_input(inputs)
        teacher_logits = torch.tensor(oracle.predict_proba(inputs_oracle))
        self.model.train()
        self.optimizer.zero_grad()
        predicted = self.model(inputs)
        target_loss = self.loss_fn(predicted, true_labels)

        if self.model[-1].__class__.__name__ == 'LogSoftmax':
            predicted = torch.exp(predicted)
        a = log_softmax(predicted / temperature, dim=-1)
        b = softmax(teacher_logits / temperature, dim=-1)
        distill_loss = KLDivLoss(reduction='batchmean')(a, b) * (temperature**2)

        loss = alpha * distill_loss + (1 - alpha) * target_loss
        loss.backward()
        self.optimizer.step()
        self.model.eval()
        return predicted.detach(), loss.item()

    def learn(self, X: torch.Tensor, y: torch.Tensor):
        if self.optimizer is None or self.loss_fn is None:
            self.check_optim_loss(self.optimizer, self.loss_fn)

        self.model.train()
        X, y = self.safe_cast_input(X), self.safe_cast_input(y, is_y=True)
        self.optimizer.zero_grad()
        outputs = self.model(X)
        loss: torch.Tensor = self.loss_fn(outputs, y)
        loss.backward()
        self.optimizer.step()
        self.model.eval()
        return outputs.detach(), loss.item()

    def check_optim_loss(self, optimizer, loss_fn, **kwargs):
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), **kwargs)
        if loss_fn is None:
            if self.model[-1].__class__.__name__ == 'LogSoftmax':
                loss_fn = torch.nn.NLLLoss()
            elif self.model[-1].__class__.__name__ == 'Softmax':
                loss_fn = torch.nn.CrossEntropyLoss()
            elif self.model[-1].__class__.__name__ == 'Identity':
                loss_fn = torch.nn.MSELoss()
            elif self.model[-1].__class__.__name__ == 'Sigmoid':
                loss_fn = torch.nn.MSELoss()
            else:
                raise ValueError(f'Unknown {self.model[-1].__class__.__name__}')
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    @property
    def is_concept_drift(self):
        return self.drift_detector is not None and self.drift_detector.drift_detected

    def concept_react(self):
        ...


class Mlp(TorchModel):
    def __init__(
            self, in_features: list[str] = None, n_outputs=2,
            hidden_units=None, hidden_layers=None, batch_norm=False, dropout_prob=0.2, activation='ReLU',
            final_activation='LogSoftmax', dtype=torch.float, drift_detector=None):
        super().__init__(in_features=in_features, n_outputs=n_outputs, hidden_units=hidden_units,
                         batch_norm=batch_norm, dropout_prob=dropout_prob, hidden_layers=hidden_layers,
                         activation=activation, final_activation=final_activation, dtype=dtype,
                         drift_detector=drift_detector)

        if hidden_layers is None:
            return

        activation: torch.nn.ReLU = getattr(torch.nn, activation)
        final_activation: torch.nn.LogSoftmax = getattr(torch.nn, final_activation)
        final_kwargs = {'dim': -1} if 'dim' in inspect.signature(final_activation.__init__).parameters else {}

        if batch_norm:
            self.model.add_module('input_batch', torch.nn.BatchNorm1d(len(in_features), dtype=dtype))

        self.model.add_module('input_layer', torch.nn.Linear(
            len(in_features), hidden_units, dtype=dtype))
        self.model.add_module('input_act', activation())

        for i in range(hidden_layers):
            if batch_norm:
                self.model.add_module(f'hidden_{i}_batch', torch.nn.BatchNorm1d(hidden_units, dtype=dtype))
            if dropout_prob > 0.:
                self.model.add_module(f'hidden_{i}_drop', torch.nn.Dropout(p=dropout_prob))
            self.model.add_module(f'hidden_{i}_layer', torch.nn.Linear(
                hidden_units, hidden_units, dtype=dtype))
            self.model.add_module(f'hidden_{i}_act', activation())

        self.model.add_module('final_layer', torch.nn.Linear(hidden_units, n_outputs, dtype=dtype))
        self.model.add_module('final_act', final_activation(**final_kwargs))

    def _parse_ypred(self, y: torch.Tensor):
        if y.ndim > 1:
            return np.argmax(y, axis=-1)
        return y


class MlpEncoder(TorchModel):
    def __init__(
            self, in_features: list[str] = None, hidden_units=None, hidden_layers=None,
            dropout_prob=0.2, activation='ReLU', final_activation='Sigmoid', dtype=torch.float, drift_detector=None):
        super().__init__(in_features=in_features, n_outputs=len(in_features),
                         hidden_units=hidden_units, dropout_prob=dropout_prob, hidden_layers=hidden_layers,
                         activation=activation, final_activation=final_activation, dtype=dtype,
                         drift_detector=drift_detector)

        if hidden_layers is None:
            return

        activation: torch.nn.ReLU = getattr(torch.nn, activation)
        final_activation: torch.nn.LogSoftmax = getattr(torch.nn, final_activation)

        self.model.add_module('input_layer', torch.nn.Linear(
            len(in_features), hidden_units, dtype=dtype))
        self.model.add_module('input_act', activation())

        current_size, next_size = hidden_units, 0
        for i in range(hidden_layers):
            next_size = round(current_size / 2)
            if dropout_prob > 0.:
                self.model.add_module(f'encoder_{i}_drop', torch.nn.Dropout(p=dropout_prob))
            self.model.add_module(f'encoder_{i}_layer', torch.nn.Linear(
                current_size, next_size, dtype=dtype))
            self.model.add_module(f'encoder_{i}_act', activation())
            current_size = next_size

        for i in range(hidden_layers):
            next_size = current_size * 2
            if dropout_prob > 0.:
                self.model.add_module(f'decoder_{i}_drop', torch.nn.Dropout(p=dropout_prob))
            self.model.add_module(f'decoder_{i}_layer', torch.nn.Linear(
                current_size, next_size, dtype=dtype))
            self.model.add_module(f'decoder_{i}_act', activation())
            current_size = next_size

        self.model.add_module('final_layer', torch.nn.Linear(hidden_units, len(in_features), dtype=dtype))
        self.model.add_module('final_act', final_activation())

    def _parse_ypred(self, y):
        return y
