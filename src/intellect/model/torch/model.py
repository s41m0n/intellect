
import inspect
import io
import os
import sys
from abc import abstractmethod
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.nn import KLDivLoss
from torch.nn.functional import log_softmax, softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...dataset import (ContinuouLearningAlgorithm, Dataset,
                        FeatureAvailability, InputForLearn)
from ..base import BaseModel


class TorchModel(BaseModel):

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.init_params: dict[str, object] = kwargs
        self.model: torch.nn.Sequential = torch.nn.Sequential()

    @abstractmethod
    def _parse_ypred(self, y):
        pass

    @property
    def is_autoencoder(self):
        return len(self.init_params["in_features"]) == self.init_params["n_outputs"]

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def xtype(self):
        return next(self.model.parameters()).dtype

    def clone(self, init=True):
        if init is False:
            buffer = io.BytesIO()
            torch.save({"params": self.init_params, "model": self.model}, buffer)
            buffer.seek(0)
            tmp = torch.load(buffer)
            ret = self.__class__()
            ret.init_params = tmp["params"]
            ret.model = tmp["model"]
            return ret
        return self.__class__(**self.init_params)

    def save(self, path: str):
        if not path.endswith(".pt"):
            path += ".pt"
        with open(path, "wb") as fp:
            torch.save({"params": self.init_params, "model": self.model}, fp)

    @classmethod
    def load(cls, path: str) -> "TorchModel":
        if not path.endswith(".pt"):
            path += ".pt"
        try:
            ret = torch.load(path)
        except Exception as e:
            ret = torch.load(path, map_location="cpu")
        new = cls()
        new.init_params = ret["params"]
        new.model = ret["model"]
        new.model.eval()
        return new

    def safe_cast_input(self, x: Union[torch.Tensor, dict, pd.DataFrame, np.ndarray], is_y=False):
        ret: torch.Tensor = None
        dtype = torch.long if is_y and not self.is_autoencoder else self.xtype
        if isinstance(x, torch.Tensor):
            ret = x
        elif isinstance(x, dict):
            ret = torch.tensor([0 if k not in x else x[k]
                               for k in self.init_params["in_features"]])
        elif isinstance(x, (pd.DataFrame, pd.Series)):
            ret = torch.from_numpy(x.values)
        elif isinstance(x, np.ndarray):
            ret = torch.from_numpy(x)
        else:
            return torch.tensor([x], dtype=dtype, device=self.device)

        return ret.to(self.device, dtype=dtype)

    def predict(self, x: torch.Tensor):
        return self._parse_ypred(self.predict_proba(x))

    def predict_proba(self, x: torch.Tensor):
        with torch.no_grad():
            y = self.model(self.safe_cast_input(x)).detach()
        if self.model[-1].__class__.__name__ == "LogSoftmax":
            y = torch.exp(y)
        return y

    def fit(self, train_dataset: Dataset, validation_dataset: Dataset | float = 0.2, batch_size: int = 256,
            max_epochs: int = 100, epochs_wo_improve: int = None, metric=accuracy_score, shuffle=True,
            optimizer: torch.optim.Optimizer = None, loss_fn: torch.nn.modules.loss._Loss = None,
            higher_better=True, **kwargs):

        if torch.cuda.is_available():
            self.model.cuda()

        best_metric = 0 if higher_better else sys.maxsize
        current_epochs_wo_improve = 0
        history = {"loss_train": []}
        best_state_dict: deepcopy(self.model.state_dict())
        optimizer, loss_fn = self.check_optim_loss(optimizer, loss_fn)

        if validation_dataset is not None:
            if isinstance(validation_dataset, float):
                validation_dataset = train_dataset.sample(validation_dataset, by_category=True)
                train_dataset = train_dataset.filter_indexes(validation_dataset.X.index.values)
            X_val = self.safe_cast_input(validation_dataset.X)
            if self.is_autoencoder:
                y_val = X_val
            else:
                y_val = self.safe_cast_input(validation_dataset.y, is_y=True)
            history[f"{metric.__name__}_validation"] = []
            history["loss_validation"] = []

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=os.cpu_count(), pin_memory=True, persistent_workers=False)

        with tqdm(range(max_epochs)) as pbar:
            for i in range(max_epochs):
                training_loss = 0.0
                for ii, (inputs, labels) in enumerate(train_loader):
                    if self.is_autoencoder:
                        labels = inputs
                    _, loss = self.learn(inputs, labels, optimizer, loss_fn)
                    training_loss += loss

                training_loss /= (ii + 1)

                # metric_train = metric(y_true.cpu(), y_preds.cpu())
                # history[f"{metric.__name__}_train"].append(metric_train)
                history["loss_train"].append(training_loss)

                if validation_dataset is not None:
                    with torch.no_grad():
                        y_raw_val = self.model(X_val).detach()
                        validation_loss = loss_fn(y_raw_val, y_val).item()
                    metric_validation = metric(self._parse_ypred(y_val), self._parse_ypred(y_raw_val))
                    history[f"{metric.__name__}_validation"].append(metric_validation)
                    history["loss_validation"].append(validation_loss)

                pbar.set_description("Epoch {} {}".format(i + 1, {k: v[-1] for k, v in history.items()}), refresh=False)
                pbar.update()

                if epochs_wo_improve:
                    if validation_dataset:
                        eval_metric = metric_validation
                        cond = metric_validation > best_metric if higher_better else metric_validation < best_metric
                    else:
                        eval_metric = training_loss
                        cond = training_loss < best_metric
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
            batch_size="auto", features_available: list[str] = None, received_categories=[],
            availability: FeatureAvailability = FeatureAvailability.bilateral,
            learn_input=InputForLearn.client, oracle_input=InputForLearn.oracle, drop_ratio=0.,
            optimizer: torch.optim.Optimizer = None, weight_decay: float = 0., loss_fn: torch.nn.modules.loss._Loss = None, **kwargs):
        if received_categories:
            if not all(v in data.categories for v in received_categories):
                raise Exception("Invalid features, not present in dataset")
            data = data.filter_categories(received_categories)

        if torch.cuda.is_available():
            self.model.cuda()
            if oracle:
                oracle.cuda()

        idx, idx_oracle = [], []
        if features_available is not None or len(features_available):
            if availability.value == FeatureAvailability.oracle.value:
                idx = data.filter_features(features_available, get_idx=True)
            elif availability.value == FeatureAvailability.client.value:  # strange use case
                idx_oracle = data.filter_features(features_available, get_idx=True)
            elif availability.value == FeatureAvailability.none.value:
                idx = idx_oracle = data.filter_features(features_available, get_idx=True)
            elif availability.value not in (None, FeatureAvailability.bilateral.value):
                raise NotImplementedError("Daje")

        optimizer, loss_fn = self.check_optim_loss(optimizer, loss_fn, weight_decay=weight_decay)

        loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(),
                            pin_memory=True, persistent_workers=False)

        y_preds = torch.empty(0)
        y_true = torch.empty(0)
        with tqdm(range(len(loader))) as pbar:
            for ii, (inputs, true_labels) in enumerate(loader):
                y_true = torch.cat((y_true, true_labels), dim=0)
                inputs_oracle = inputs.detach().clone()

                if oracle_input.value == InputForLearn.oracle.value:
                    inputs_oracle[:, idx_oracle] = 0.
                elif oracle_input.value == InputForLearn.client.value:
                    inputs_oracle[:, idx] = 0.
                else:
                    raise Exception("Unknown")

                if learn_input.value == InputForLearn.oracle.value:
                    inputs[:, idx_oracle] = 0.
                elif learn_input.value == InputForLearn.client.value:
                    inputs[:, idx] = 0.
                else:
                    raise Exception("Unknown")

                for i in range(epochs):

                    if algorithm.value == ContinuouLearningAlgorithm.ground_inferred.value:
                        inferred_labels = oracle.predict(inputs_oracle)
                        predicted, loss = self.learn(inputs, inferred_labels, optimizer, loss_fn)
                    elif algorithm.value == ContinuouLearningAlgorithm.ground_truth.value:
                        predicted, loss = self.learn(inputs, true_labels, optimizer, loss_fn)
                    elif algorithm.value == ContinuouLearningAlgorithm.knowledge_distillation.value:
                        predicted, loss = self.learn_knowledge_distillation(
                            oracle, inputs, inputs_oracle, true_labels, optimizer, loss_fn, **kwargs)
                    elif algorithm.value == ContinuouLearningAlgorithm.none.value:
                        loss = torch.tensor(np.nan)
                        predicted = self.predict_proba(inputs)
                    else:
                        raise Exception("Unknown value")

                    pbar.set_description("Batch {} Epoch {} {}".format(ii + 1, i + 1, loss), refresh=False)
                    pbar.update()
                y_preds = torch.cat((y_preds, self._parse_ypred(predicted)), dim=0)

        if torch.cuda.is_available():
            self.model.cpu()
            if oracle:
                oracle.model.cpu()
        return y_preds, y_true

    @property
    def prunable(self) -> list[torch.nn.Module]:
        """Property to return the list of prunable modules within the model

        Returns:
            list[torch.nn.Module]: list of prunable modules
        """
        return tuple(m for m in self.model.children() if isinstance(m, torch.nn.Linear))

    def learn_knowledge_distillation(
            self, oracle: BaseModel, inputs, inputs_oracle, true_labels, optimizer, loss_fn, temperature: int = 2,
            alpha: float = 0.25):
        inputs = self.safe_cast_input(inputs)
        teacher_logits = oracle.predict_proba(inputs_oracle)
        self.model.train()
        optimizer.zero_grad()
        predicted = self.model(inputs)
        target_loss = loss_fn(predicted, true_labels)

        if self.model[-1].__class__.__name__ == "LogSoftmax":
            predicted = torch.exp(predicted)
        a = log_softmax(predicted / temperature, dim=-1)
        b = softmax(teacher_logits / temperature, dim=-1)
        distill_loss = KLDivLoss()(a, b) * (temperature**2)

        loss = alpha * distill_loss + (1 - alpha) * target_loss
        loss.backward()
        optimizer.step()
        self.model.eval()
        return predicted.detach(), loss.item()

    def learn(self, x: torch.Tensor, y: torch.Tensor, optimizer: torch.optim.AdamW,
              loss_fn: torch.nn.CrossEntropyLoss):
        self.model.train()
        x, y = self.safe_cast_input(x), self.safe_cast_input(y, is_y=True)
        optimizer.zero_grad()
        outputs = self.model(x)
        loss: torch.Tensor = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        self.model.eval()
        return outputs.detach(), loss.item()

    def check_optim_loss(self, optimizer, loss_fn, **kwargs):
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), **kwargs)
        if loss_fn is None:
            if self.model[-1].__class__.__name__ == "LogSoftmax":
                loss_fn = torch.nn.NLLLoss()
            elif self.model[-1].__class__.__name__ == "Softmax":
                loss_fn = torch.nn.CrossEntropyLoss()
            elif self.model[-1].__class__.__name__ == "Identity":
                loss_fn = torch.nn.MSELoss()
            elif self.model[-1].__class__.__name__ == "Sigmoid":
                loss_fn = torch.nn.MSELoss()
            else:
                raise Exception("Unknown")
        return optimizer, loss_fn


class Mlp(TorchModel):
    def __init__(
            self, in_features: list[str] = [], n_outputs=2,
            hidden_units=None, hidden_layers=None, batch_norm=False, dropout_prob=0.2, activation="ReLU",
            final_activation="LogSoftmax", dtype=torch.float):
        super().__init__(in_features=in_features, n_outputs=n_outputs, hidden_units=hidden_units,
                         batch_norm=batch_norm, dropout_prob=dropout_prob, hidden_layers=hidden_layers,
                         activation=activation, final_activation=final_activation, dtype=dtype)

        if hidden_layers is None:
            return

        activation: torch.nn.ReLU = getattr(torch.nn, activation)
        final_activation: torch.nn.LogSoftmax = getattr(torch.nn, final_activation)
        final_kwargs = {"dim": -1} if "dim" in inspect.signature(final_activation.__init__).parameters else {}

        if batch_norm:
            self.model.add_module("input_batch", torch.nn.BatchNorm1d(len(in_features), dtype=dtype))

        self.model.add_module("input_layer", torch.nn.Linear(
            len(in_features), hidden_units, dtype=dtype))
        self.model.add_module("input_act", activation())

        for i in range(hidden_layers):
            if batch_norm:
                self.model.add_module(f"hidden_{i}_batch", torch.nn.BatchNorm1d(hidden_units, dtype=dtype))
            if dropout_prob > 0.:
                self.model.add_module(f"hidden_{i}_drop", torch.nn.Dropout(p=dropout_prob))
            self.model.add_module(f"hidden_{i}_layer", torch.nn.Linear(
                hidden_units, hidden_units, dtype=dtype))
            self.model.add_module(f"hidden_{i}_act", activation())

        self.model.add_module("final_layer", torch.nn.Linear(hidden_units, n_outputs, dtype=dtype))
        self.model.add_module("final_act", final_activation(**final_kwargs))

    def _parse_ypred(self, y):
        if y.ndim > 1:
            return torch.argmax(y, dim=-1)
        return y

    concept_react = is_concept_drift = None  # TODO: add


class MlpEncoder(TorchModel):
    def __init__(
            self, in_features: list[str] = [], hidden_units=None, hidden_layers=None,
            dropout_prob=0.2, activation="ReLU", final_activation="Sigmoid", dtype=torch.float):
        super().__init__(in_features=in_features, n_outputs=len(in_features), hidden_units=hidden_units,
                         dropout_prob=dropout_prob, hidden_layers=hidden_layers,
                         activation=activation, final_activation=final_activation, dtype=dtype)

        if hidden_layers is None:
            return

        activation: torch.nn.ReLU = getattr(torch.nn, activation)
        final_activation: torch.nn.LogSoftmax = getattr(torch.nn, final_activation)

        self.model.add_module("input_layer", torch.nn.Linear(
            len(in_features), hidden_units, dtype=dtype))
        self.model.add_module("input_act", activation())

        current_size, next_size = hidden_units, 0
        for i in range(hidden_layers):
            next_size = round(current_size / 2)
            if dropout_prob > 0.:
                self.model.add_module(f"encoder_{i}_drop", torch.nn.Dropout(p=dropout_prob))
            self.model.add_module(f"encoder_{i}_layer", torch.nn.Linear(
                current_size, next_size, dtype=dtype))
            self.model.add_module(f"encoder_{i}_act", activation())
            current_size = next_size

        for i in range(hidden_layers):
            next_size = current_size * 2
            if dropout_prob > 0.:
                self.model.add_module(f"decoder_{i}_drop", torch.nn.Dropout(p=dropout_prob))
            self.model.add_module(f"decoder_{i}_layer", torch.nn.Linear(
                current_size, next_size, dtype=dtype))
            self.model.add_module(f"decoder_{i}_act", activation())
            current_size = next_size

        self.model.add_module("final_layer", torch.nn.Linear(hidden_units, len(in_features), dtype=dtype))
        self.model.add_module("final_act", final_activation())

    def _parse_ypred(self, y):
        return y

    concept_react = is_concept_drift = None  # TODO: add
